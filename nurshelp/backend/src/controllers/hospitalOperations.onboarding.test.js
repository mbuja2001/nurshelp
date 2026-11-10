import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { AuditLog } from '../models/auditLog.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { Hospital } from '../models/hospital.model.js';
import { bulkOnboardHospitalStaff, updateHospitalStaff } from './hospitalOperations.controller.js';

const originalMethods = {
  auditCreate: AuditLog.create,
  staffFind: ClinicalStaff.find,
  staffFindOne: ClinicalStaff.findOne,
  staffInsertMany: ClinicalStaff.insertMany,
  hospitalFindById: Hospital.findById,
};

afterEach(() => {
  AuditLog.create = originalMethods.auditCreate;
  ClinicalStaff.find = originalMethods.staffFind;
  ClinicalStaff.findOne = originalMethods.staffFindOne;
  ClinicalStaff.insertMany = originalMethods.staffInsertMany;
  Hospital.findById = originalMethods.hospitalFindById;
});

describe('bulk hospital staff onboarding', () => {
  it('stores the selected position separately from its permission role', async () => {
    Hospital.findById = () => ({
      select: () => ({ lean: async () => ({ departments: [{ name: 'ED', isActive: true, specialties: [{ name: 'Emergency Medicine' }] }] }) }),
    });
    ClinicalStaff.find = () => ({
      select: () => ({ lean: async () => [] }),
    });
    ClinicalStaff.insertMany = async (documents) => documents.map((document, index) => ({
      ...document,
      _id: `staff-${index + 1}`,
    }));
    AuditLog.create = async () => ({});

    const response = {
      statusCode: null,
      body: null,
      status(code) {
        this.statusCode = code;
        return this;
      },
      json(body) {
        this.body = body;
        return this;
      },
    };

    await bulkOnboardHospitalStaff({
      hospitalId: 'hospital-1',
      user: { _id: 'admin-1' },
      body: {
        staff: ['TRIAGE_NURSE', 'CLINICAL_NURSE'].map((positionCode) => ({
          firstName: 'Nurse',
          lastName: positionCode,
          email: `${positionCode.toLowerCase()}@hospital.test`,
          password: 'secure-password',
          positionCode,
          role: 'NURSE',
          department: 'ED',
          specialty: 'Emergency Medicine',
          title: 'RN',
          licenseNumber: `LICENSE-${positionCode}`,
        })),
      },
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(response.body.staff.length, 2);
    assert.deepEqual(response.body.staff.map(({ role, positionCode }) => ({ role, positionCode })), [
      { role: 'NURSE', positionCode: 'TRIAGE_NURSE' },
      { role: 'NURSE', positionCode: 'CLINICAL_NURSE' },
    ]);
  });

  it('rejects a specialty that is not configured for the selected department', async () => {
    Hospital.findById = () => ({
      select: () => ({ lean: async () => ({ departments: [{ name: 'ED', isActive: true, specialties: [{ name: 'Emergency Medicine' }] }] }) }),
    });
    ClinicalStaff.find = () => ({
      select: () => ({ lean: async () => [] }),
    });
    let insertCalled = false;
    ClinicalStaff.insertMany = async () => {
      insertCalled = true;
      return [];
    };

    const response = {
      statusCode: null,
      body: null,
      status(code) {
        this.statusCode = code;
        return this;
      },
      json(body) {
        this.body = body;
        return this;
      },
    };

    await bulkOnboardHospitalStaff({
      hospitalId: 'hospital-1',
      user: { _id: 'admin-1' },
      body: {
        staff: [{
          firstName: 'Nurse',
          lastName: 'Test',
          email: 'nurse@hospital.test',
          password: 'secure-password',
          positionCode: 'TRIAGE_NURSE',
          department: 'ED',
          specialty: 'General Practice',
          title: 'RN',
          licenseNumber: 'LICENSE-1',
        }],
      },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.code, 'SPECIALTY_NOT_CONFIGURED');
    assert.equal(insertCalled, false);
  });

  it('rejects inactive and unconfigured departments', async () => {
    Hospital.findById = () => ({
      select: () => ({ lean: async () => ({ departments: [{ name: 'ED', isActive: false, specialties: [{ name: 'Emergency Medicine' }] }] }) }),
    });
    ClinicalStaff.find = () => ({
      select: () => ({ lean: async () => [] }),
    });

    const response = {
      statusCode: null,
      body: null,
      status(code) {
        this.statusCode = code;
        return this;
      },
      json(body) {
        this.body = body;
        return this;
      },
    };

    await bulkOnboardHospitalStaff({
      hospitalId: 'hospital-1',
      user: { _id: 'admin-1' },
      body: {
        staff: [{
          firstName: 'Nurse',
          lastName: 'Test',
          email: 'nurse@hospital.test',
          password: 'secure-password',
          positionCode: 'TRIAGE_NURSE',
          department: 'ED',
          specialty: 'Emergency Medicine',
          title: 'RN',
          licenseNumber: 'LICENSE-1',
        }],
      },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.match(response.body.message, /active department configured/);
  });
});

describe('hospital staff credential updates', () => {
  const createResponse = () => ({
    statusCode: null,
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

  const createStaff = () => ({
    _id: 'staff-1',
    hospitalId: 'hospital-1',
    firstName: 'Triage',
    lastName: 'Nurse',
    email: 'triage@hospital.test',
    systemRole: 'NURSE',
    positionCode: 'TRIAGE_NURSE',
    positionTitle: 'Triage nurse',
    clinicalSpecialty: 'Emergency Medicine',
    assignment: { department: 'ED', isOnCall: true },
    credentials: { title: 'RN', licenseNumber: 'RN-OLD' },
    status: 'ACTIVE',
    save: async () => {},
  });

  const createRequest = (body) => ({
    hospitalId: 'hospital-1',
    params: { staffID: 'staff-1' },
    user: { _id: 'admin-1' },
    body,
  });

  it('updates position and permission role consistently and stays scoped to the hospital', async () => {
    const staff = createStaff();
    Hospital.findById = () => ({
      select: () => ({ lean: async () => ({ departments: [{ name: 'ED', isActive: true, specialties: [{ name: 'Emergency Medicine' }] }] }) }),
    });
    ClinicalStaff.findOne = async (query) => typeof query._id === 'string' ? staff : null;
    const response = createResponse();

    await updateHospitalStaff(createRequest({
      firstName: 'Triage',
      lastName: 'Nurse',
      email: 'triage@hospital.test',
      positionCode: 'CLINICAL_NURSE',
      assignment: { department: 'ED' },
      clinicalSpecialty: 'Emergency Medicine',
      credentials: { title: 'RN', licenseNumber: 'RN-NEW' },
      status: 'ACTIVE',
    }), response);

    assert.equal(response.statusCode, 200);
    assert.equal(staff.positionCode, 'CLINICAL_NURSE');
    assert.equal(staff.systemRole, 'NURSE');
    assert.equal(staff.positionCodeNeedsBackfill, false);
    assert.equal(staff.assignment.department, 'ED');
    assert.equal(staff.assignment.isOnCall, true);
    assert.equal(staff.credentials.licenseNumber, 'RN-NEW');
  });

  it('rejects specialties not configured for the selected department', async () => {
    const staff = createStaff();
    Hospital.findById = () => ({
      select: () => ({ lean: async () => ({ departments: [{ name: 'ED', isActive: true, specialties: [{ name: 'Emergency Medicine' }] }] }) }),
    });
    ClinicalStaff.findOne = async (query) => typeof query._id === 'string' ? staff : null;
    const response = createResponse();

    await updateHospitalStaff(createRequest({
      positionCode: 'TRIAGE_NURSE',
      assignment: { department: 'ED' },
      clinicalSpecialty: 'General Practice',
      credentials: { title: 'RN', licenseNumber: 'RN-OLD' },
    }), response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.code, 'SPECIALTY_NOT_CONFIGURED');
  });
});
