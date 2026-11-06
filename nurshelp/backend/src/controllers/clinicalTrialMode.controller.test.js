import test from 'node:test';
import assert from 'node:assert/strict';
import { createAccountHospital } from './accountHospital.controller.js';
import { getHospitalTrialMode } from './hospital.controller.js';
import { AuditLog } from '../models/auditLog.model.js';
import { ClinicalTrial } from '../models/clinicalTrial.model.js';
import { Hospital } from '../models/hospital.model.js';
import { Manager } from '../models/manager.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { SandboxAccount } from '../models/sandboxAccount.model.js';
import { requireClinicalTrialMode } from '../middleware/clinicalTrialMode.js';

const createResponse = () => ({
  statusCode: 200,
  body: null,
  status(code) {
    this.statusCode = code;
    return this;
  },
  json(body) {
    this.body = body;
    return this;
  },
});

test('direct production hospital creation is rejected with a specific code', async () => {
  const response = createResponse();
  await createAccountHospital({
    body: {
      hospitalName: 'Production Test Hospital',
      environment: 'production',
      adminName: 'Test',
      adminSurname: 'Admin',
      adminEmail: 'admin@example.test',
      adminPassword: 'a-secure-password',
    },
  }, response);

  assert.equal(response.statusCode, 403);
  assert.equal(response.body.code, 'PRODUCTION_ONBOARDING_UNAVAILABLE');
});

test('hospital environment is immutable and active-trial count has a zero minimum/default', () => {
  assert.equal(Hospital.schema.path('environment').options.immutable, true);
  assert.equal(Hospital.schema.path('activeTrialCount').options.default, 0);
  assert.equal(Hospital.schema.path('activeTrialCount').options.min, 0);
});

test('hospital creation reports and records failed admin provisioning as a partial result', async () => {
  const originals = {
    hospitalCreate: Hospital.create,
    hospitalFindByIdAndUpdate: Hospital.findByIdAndUpdate,
    managerExists: Manager.exists,
    managerCreate: Manager.create,
    staffExists: ClinicalStaff.exists,
    accountExists: SandboxAccount.exists,
    auditCreate: AuditLog.create,
  };
  let createdHospital;
  let pushedHistory;
  Hospital.create = async (data) => {
    createdHospital = {
      ...data,
      _id: 'hospital-1',
      departments: data.departments,
      save: async () => {},
    };
    return createdHospital;
  };
  Hospital.findByIdAndUpdate = async (_id, update) => {
    pushedHistory = update.$push.clinicalTrialModeHistory;
    return createdHospital;
  };
  Manager.exists = async () => null;
  Manager.create = async () => {
    throw new Error('test admin provisioning failure');
  };
  ClinicalStaff.exists = async () => null;
  SandboxAccount.exists = async () => null;
  AuditLog.create = async () => {};

  try {
    const response = createResponse();
    await createAccountHospital({
      body: {
        hospitalName: 'Partial Result Hospital',
        environment: 'SANDBOX',
        adminName: 'Test',
        adminSurname: 'Admin',
        adminEmail: 'partial-admin@example.test',
        adminPassword: 'a-secure-password',
      },
      sandboxAccount: { _id: 'owner-1', name: 'Owner' },
      ip: '127.0.0.1',
      get: () => 'test-agent',
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(response.body.partial, true);
    assert.equal(response.body.adminCreated, false);
    assert.equal(response.body.code, 'ADMIN_CREATION_FAILED');
    assert.equal(createdHospital.clinicalTrialModeHistory[0].eventType, 'MODE_CHANGED');
    assert.equal(pushedHistory.eventType, 'ADMIN_CREATION_FAILED');
  } finally {
    Hospital.create = originals.hospitalCreate;
    Hospital.findByIdAndUpdate = originals.hospitalFindByIdAndUpdate;
    Manager.exists = originals.managerExists;
    Manager.create = originals.managerCreate;
    ClinicalStaff.exists = originals.staffExists;
    SandboxAccount.exists = originals.accountExists;
    AuditLog.create = originals.auditCreate;
  }
});

test('hospital trial-mode read uses only the authenticated hospital id', async () => {
  const originalFindById = Hospital.findById;
  let queriedHospitalId;
  Hospital.findById = (hospitalId) => {
    queriedHospitalId = hospitalId;
    return {
      select: () => ({
        lean: async () => ({
          name: 'Authenticated Hospital',
          environment: 'SANDBOX',
          clinicalTrialMode: false,
        }),
      }),
    };
  };

  try {
    const response = createResponse();
    await getHospitalTrialMode({
      hospitalId: 'authenticated-hospital',
      params: { hospitalId: 'other-hospital' },
      body: { hospitalId: 'other-hospital' },
    }, response);
    assert.equal(queriedHospitalId, 'authenticated-hospital');
    assert.equal(response.statusCode, 200);
  } finally {
    Hospital.findById = originalFindById;
  }
});

test('trial-mode middleware distinguishes missing hospital from disabled mode in every environment', async () => {
  const originalFindById = Hospital.findById;
  try {
    for (const environment of ['SANDBOX', 'PRODUCTION']) {
      Hospital.findById = () => ({
        select: () => ({ lean: async () => ({ environment, clinicalTrialMode: false }) }),
      });
      const response = createResponse();
      await requireClinicalTrialMode({ hospitalId: 'hospital-1' }, response, () => {
        throw new Error('next should not be called when trial mode is off');
      });
      assert.equal(response.statusCode, 403);
      assert.equal(response.body.code, 'TRIAL_MODE_DISABLED');
    }

    const missingResponse = createResponse();
    await requireClinicalTrialMode({ hospitalId: null }, missingResponse, () => {
      throw new Error('next should not be called when hospital is missing');
    });
    assert.equal(missingResponse.statusCode, 404);
    assert.equal(missingResponse.body.code, 'HOSPITAL_NOT_FOUND');

    Hospital.findById = () => ({
      select: () => ({ lean: async () => null }),
    });
    const unknownResponse = createResponse();
    await requireClinicalTrialMode({ hospitalId: 'unknown-hospital' }, unknownResponse, () => {
      throw new Error('next should not be called when hospital cannot be found');
    });
    assert.equal(unknownResponse.statusCode, 404);
    assert.equal(unknownResponse.body.code, 'HOSPITAL_NOT_FOUND');
  } finally {
    Hospital.findById = originalFindById;
  }
});

test('reconciliation embeds and audits mismatches without repairing the counter', async () => {
  const originals = {
    findById: Hospital.findById,
    updateOne: Hospital.updateOne,
    countDocuments: ClinicalTrial.countDocuments,
    auditCreate: AuditLog.create,
  };
  let embeddedEvent;
  let auditEvent;
  let storedCounter = 4;
  Hospital.findById = () => ({
    select: () => ({ lean: async () => ({ activeTrialCount: storedCounter }) }),
  });
  Hospital.updateOne = async (_filter, update) => {
    embeddedEvent = update.$push.clinicalTrialModeHistory;
    return { modifiedCount: 1 };
  };
  ClinicalTrial.countDocuments = async () => 3;
  AuditLog.create = async (event) => {
    auditEvent = event;
  };

  try {
    const result = await ClinicalTrial.reconcileHospitalActiveTrialCount({
      hospitalId: 'hospital-1',
      actorId: 'account-1',
      actorName: 'Account Owner',
    });
    assert.deepEqual(result, {
      matches: false,
      activeTrialCount: 4,
      activeTrialActualCount: 3,
    });
    assert.equal(embeddedEvent.eventType, 'TRIAL_COUNT_MISMATCH');
    assert.equal(auditEvent.action, 'CLINICAL_TRIAL_COUNT_MISMATCH');
    assert.equal(storedCounter, 4);
  } finally {
    Hospital.findById = originals.findById;
    Hospital.updateOne = originals.updateOne;
    ClinicalTrial.countDocuments = originals.countDocuments;
    AuditLog.create = originals.auditCreate;
  }
});
