import test from 'node:test';
import assert from 'node:assert/strict';
import { enableSandboxEhrAccess, resolveAuthenticatedEhrHospitalId } from './operations.controller.js';
import { updateAccountHospitalEhrAccess } from './accountHospital.controller.js';
import { AuditLog } from '../models/auditLog.model.js';
import { Hospital } from '../models/hospital.model.js';

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

test('EHR hospital scope comes only from authenticated identity', () => {
  assert.equal(resolveAuthenticatedEhrHospitalId({
    hospitalId: 'token-hospital',
    params: { hospitalId: 'other-hospital' },
    body: { hospitalId: 'other-hospital' },
    query: { hospitalId: 'other-hospital' },
  }), 'token-hospital');
  assert.throws(
    () => resolveAuthenticatedEhrHospitalId({ params: { hospitalId: 'untrusted-hospital' }, body: { hospitalId: 'untrusted-hospital' } }),
    (error) => error.statusCode === 401
  );
});

test('sandbox account owner saves a hospital-scoped EHR protocol policy', async () => {
  const originals = {
    hospitalFindOne: Hospital.findOne,
    hospitalFindOneAndUpdate: Hospital.findOneAndUpdate,
    auditCreate: AuditLog.create,
  };
  let updateQuery;
  let updateDocument;
  Hospital.findOne = async () => ({
    _id: 'hospital-1',
    environment: 'SANDBOX',
    ehrConfig: { integrationAccess: null },
  });
  Hospital.findOneAndUpdate = async (query, update) => {
    updateQuery = query;
    updateDocument = update;
    return { _id: 'hospital-1' };
  };
  AuditLog.create = async () => {};

  try {
    const response = createResponse();
    await updateAccountHospitalEhrAccess({
      sandboxAccount: { _id: 'owner-1', name: 'Sandbox owner' },
      params: { hospitalId: 'hospital-1' },
      body: { enabledProtocols: ['hl7v2', 'fhir-r4'], reason: 'QA access attestation' },
      ip: '127.0.0.1',
      get: () => 'test-agent',
    }, response);

    assert.equal(response.statusCode, 200);
    assert.deepEqual(response.body.ehrIntegrationAccess.enabledProtocols, ['hl7v2', 'fhir-r4']);
    assert.equal(response.body.ehrIntegrationAccess.source, 'ACCOUNT_OWNER');
    assert.equal(updateQuery.ownerAccountId, 'owner-1');
    assert.equal(updateQuery._id, 'hospital-1');
    assert.equal(updateDocument.$set['ehrConfig.integrationAccess'].reason, 'QA access attestation');
  } finally {
    Hospital.findOne = originals.hospitalFindOne;
    Hospital.findOneAndUpdate = originals.hospitalFindOneAndUpdate;
    AuditLog.create = originals.auditCreate;
  }
});

test('sandbox account owner cannot enable EHR access for production or another account hospital', async () => {
  const originalFindOne = Hospital.findOne;
  const originalFindOneAndUpdate = Hospital.findOneAndUpdate;
  let updateCalled = false;
  Hospital.findOneAndUpdate = async () => {
    updateCalled = true;
    return { _id: 'hospital-1' };
  };

  try {
    Hospital.findOne = async () => ({ _id: 'hospital-1', environment: 'PRODUCTION' });
    const productionResponse = createResponse();
    await updateAccountHospitalEhrAccess({
      sandboxAccount: { _id: 'owner-1' },
      params: { hospitalId: 'hospital-1' },
      body: { enabledProtocols: ['fhir-r4'], reason: 'test' },
    }, productionResponse);
    assert.equal(productionResponse.statusCode, 403);
    assert.equal(productionResponse.body.code, 'PRODUCTION_ONBOARDING_UNAVAILABLE');

    Hospital.findOne = async () => null;
    const ownershipResponse = createResponse();
    await updateAccountHospitalEhrAccess({
      sandboxAccount: { _id: 'owner-1' },
      params: { hospitalId: 'other-hospital' },
      body: { enabledProtocols: ['fhir-r4'], reason: 'test' },
    }, ownershipResponse);
    assert.equal(ownershipResponse.statusCode, 404);
    assert.equal(ownershipResponse.body.code, 'HOSPITAL_NOT_FOUND');
    assert.equal(updateCalled, false);
  } finally {
    Hospital.findOne = originalFindOne;
    Hospital.findOneAndUpdate = originalFindOneAndUpdate;
  }
});

test('clearing account policy returns the hospital to the sandbox admin fallback', async () => {
  const originals = {
    hospitalFindOne: Hospital.findOne,
    hospitalFindOneAndUpdate: Hospital.findOneAndUpdate,
    auditCreate: AuditLog.create,
  };
  let updateDocument;
  Hospital.findOne = async () => ({
    _id: 'hospital-1',
    environment: 'SANDBOX',
    ehrConfig: { integrationAccess: { source: 'ACCOUNT_OWNER', enabledProtocols: ['fhir-r4'] } },
  });
  Hospital.findOneAndUpdate = async (_query, update) => {
    updateDocument = update;
    return { _id: 'hospital-1' };
  };
  AuditLog.create = async () => {};

  try {
    const response = createResponse();
    await updateAccountHospitalEhrAccess({
      sandboxAccount: { _id: 'owner-1', name: 'Sandbox owner' },
      params: { hospitalId: 'hospital-1' },
      body: { clearAccountConfiguration: true, reason: 'Return configuration access to hospital admin' },
      ip: '127.0.0.1',
      get: () => 'test-agent',
    }, response);

    assert.equal(response.statusCode, 200);
    assert.equal(response.body.ehrIntegrationAccess.source, null);
    assert.equal(response.body.ehrIntegrationAccess.canEnableSandboxAccess, true);
    assert.equal(updateDocument.$set['ehrConfig.integrationAccess'], null);
  } finally {
    Hospital.findOne = originals.hospitalFindOne;
    Hospital.findOneAndUpdate = originals.hospitalFindOneAndUpdate;
    AuditLog.create = originals.auditCreate;
  }
});

test('hospital admin fallback saves only a sandbox attestation and audit event', async () => {
  const originals = {
    hospitalFindOne: Hospital.findOne,
    hospitalFindOneAndUpdate: Hospital.findOneAndUpdate,
    auditCreate: AuditLog.create,
  };
  let updateDocument;
  let auditEvent;
  Hospital.findOne = async () => ({
    _id: 'hospital-1',
    environment: undefined,
    ehrConfig: { integrationAccess: null },
  });
  Hospital.findOneAndUpdate = async (_query, update) => {
    updateDocument = update;
    return { _id: 'hospital-1' };
  };
  AuditLog.create = async (event) => {
    auditEvent = event;
  };

  try {
    const response = createResponse();
    await enableSandboxEhrAccess({
      user: { _id: 'manager-1', role: 'ADMIN', hospitalId: 'hospital-1', name: 'QA admin' },
      body: { reason: 'QA only; no licensed connector configured' },
      ip: '127.0.0.1',
      get: () => 'test-agent',
    }, response);

    assert.equal(response.statusCode, 200);
    assert.equal(response.body.integrationAccess.source, 'HOSPITAL_ADMIN_SANDBOX');
    assert.equal(response.body.integrationAccess.isSandboxConfiguration, true);
    assert.deepEqual(updateDocument.$set['ehrConfig.integrationAccess'].enabledProtocols, ['hl7v2', 'fhir-r4']);
    assert.equal(auditEvent.action, 'EHR_INTEGRATION_ACCESS_CHANGED');
  } finally {
    Hospital.findOne = originals.hospitalFindOne;
    Hospital.findOneAndUpdate = originals.hospitalFindOneAndUpdate;
    AuditLog.create = originals.auditCreate;
  }
});
