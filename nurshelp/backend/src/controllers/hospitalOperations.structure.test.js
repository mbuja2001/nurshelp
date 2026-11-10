import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { AuditLog } from '../models/auditLog.model.js';
import { Hospital } from '../models/hospital.model.js';
import { Ward } from '../models/ward.model.js';
import { HOSPITAL_SPECIALTIES } from '../config/hospitalStructureCatalog.js';
import { saveHospitalWards, updateHospitalStructure } from './hospitalOperations.controller.js';

const originalAuditCreate = AuditLog.create;
const originalHospitalFindById = Hospital.findById;
const originalWardFind = Ward.find;
const originalWardFindOne = Ward.findOne;

afterEach(() => {
  AuditLog.create = originalAuditCreate;
  Hospital.findById = originalHospitalFindById;
  Ward.find = originalWardFind;
  Ward.findOne = originalWardFindOne;
});

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

const configureHospital = (initialDepartments = []) => {
  const hospital = {
    departments: initialDepartments,
    save: async () => {},
  };
  Hospital.findById = () => ({ select: async () => hospital });
  AuditLog.create = async () => ({ _id: 'audit-1' });
  return hospital;
};

const createRequest = (departments) => ({
  hospitalId: 'hospital-1',
  user: { _id: 'admin-1' },
  body: { departments },
});

describe('hospital structure department configuration', () => {
  it('saves the 11-department configuration with 319 specialty assignments', async () => {
    const hospital = configureHospital();
    const allSpecialties = HOSPITAL_SPECIALTIES.map(({ name }) => name);
    const assignments = [
      ['ED', allSpecialties.filter((name) => name !== 'Neurosurgery')],
      ['ICU', allSpecialties],
      ['OR', allSpecialties],
      ['OB', allSpecialties],
      ['Medical-Surgical', allSpecialties],
      ['Pediatrics', allSpecialties],
      ['Psychiatry', ['Psychiatry', 'Psychiatric Nursing']],
      ['General', allSpecialties],
      ['Pharmacy', ['Hospital Pharmacist', 'Clinical Pharmacist']],
      ['Radiology', ['Radiology', 'CT Technician', 'X-Ray Technician', 'MRI Technician', 'Ultrasound Technician']],
      ['Pathology', ['Lab Technician', 'Hematology Technician', 'Clinical Pathology']],
    ];
    const response = createResponse();

    await updateHospitalStructure(createRequest(assignments.map(([name, specialties]) => ({
      name,
      serviceTypes: ['CLINICAL_SUPPORT'],
      specialties,
    }))), response);

    assert.equal(response.statusCode, null);
    assert.equal(response.body.success, true);
    assert.equal(hospital.departments.length, 11);
    assert.equal(hospital.departments.reduce((total, department) => total + department.specialties.length, 0), 319);
  });

  it('saves catalog and custom departments with stable department IDs', async () => {
    const hospital = configureHospital([{ name: 'Pharmacy', departmentId: 'PHARMACY-OPS-001', isActive: true }]);
    const response = createResponse();

    await updateHospitalStructure(createRequest([
      { name: 'Pharmacy', serviceTypes: ['CLINICAL_SUPPORT'], specialties: ['Clinical Pharmacist'] },
      { name: 'Ambulatory Infusion Center', serviceTypes: ['OUTPATIENT'], specialties: ['Oncology'] },
    ]), response);

    assert.equal(response.statusCode, null);
    assert.equal(response.body.success, true);
    assert.deepEqual(hospital.departments.map(({ name, departmentId }) => ({ name, departmentId })), [
      { name: 'Pharmacy', departmentId: 'PHARMACY-OPS-001' },
      { name: 'Ambulatory Infusion Center', departmentId: 'DEPT-AMBULATORY-INFUSION-CENTER' },
    ]);
  });

  it('rejects duplicate names regardless of letter case without saving', async () => {
    const hospital = configureHospital();
    let saveCalled = false;
    hospital.save = async () => { saveCalled = true; };
    const response = createResponse();

    await updateHospitalStructure(createRequest([
      { name: 'Pharmacy', serviceTypes: ['CLINICAL_SUPPORT'], specialties: ['Clinical Pharmacist'] },
      { name: 'pharmacy', serviceTypes: ['CLINICAL_SUPPORT'], specialties: ['Hospital Pharmacist'] },
    ]), response);

    assert.equal(response.statusCode, 400);
    assert.match(response.body.message, /duplicate department/i);
    assert.equal(saveCalled, false);
  });

  it('rejects names that collide after department ID normalization', async () => {
    const hospital = configureHospital();
    let saveCalled = false;
    hospital.save = async () => { saveCalled = true; };
    const response = createResponse();

    await updateHospitalStructure(createRequest([
      { name: 'A-B', serviceTypes: ['CLINICAL_SUPPORT'], specialties: ['Clinical Pharmacist'] },
      { name: 'A B', serviceTypes: ['CLINICAL_SUPPORT'], specialties: ['Hospital Pharmacist'] },
    ]), response);

    assert.equal(response.statusCode, 400);
    assert.match(response.body.message, /unique IDs/i);
    assert.equal(saveCalled, false);
  });

  it('saves a custom-department ward with the configured department ID snapshot', async () => {
    const hospital = configureHospital([
      { name: 'Pharmacy', departmentId: 'PHARMACY-OPS-001', isActive: true },
    ]);
    const ward = {
      occupiedBeds: 0,
      cleaningBeds: 0,
      maintenanceBeds: 0,
      save: async function () { savedWard = { ...this }; },
    };
    let savedWard;
    Ward.find = (query) => query?.hospitalId
      ? { lean: async () => [], sort: () => ({ lean: async () => savedWard ? [savedWard] : [] }) }
      : { lean: async () => [] };
    Ward.findOne = async () => ward;
    const response = createResponse();

    await saveHospitalWards({
      ...createRequest([]),
      body: {
        wards: [{
          wardId: 'PHARMACY-WARD-01',
          wardName: 'Medication Care Unit',
          departmentName: 'Pharmacy',
          careTier: 'GENERAL_INPATIENT',
          acuityLevelAllowed: ['Green'],
          totalBeds: 10,
        }],
      },
    }, response);

    assert.equal(response.body.success, true);
    assert.equal(response.body.wards[0].department, 'GENERAL');
    assert.equal(response.body.wards[0].departmentId, 'PHARMACY-OPS-001');
    assert.equal(response.body.wards[0].departmentName, 'Pharmacy');
  });
});
