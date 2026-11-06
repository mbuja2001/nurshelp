import test from 'node:test';
import assert from 'node:assert/strict';
import mongoose from 'mongoose';
import {
  buildAssignedStaff,
  getClinicalTeamRoleField,
  normalizeShiftAssignmentRole,
  validateClinicalTeam,
} from './shiftTeamAssignment.js';

const hospitalId = new mongoose.Types.ObjectId();
const staff = (id, systemRole, status = 'ACTIVE') => ({
  _id: new mongoose.Types.ObjectId(id),
  hospitalId,
  systemRole,
  status,
  firstName: systemRole === 'PHYSICIAN' ? 'Dr. Test' : 'Test',
  lastName: systemRole === 'PHYSICIAN' ? 'Physician' : 'Nurse',
  clinicalSpecialty: 'General Practice',
});

const staffById = new Map([
  [staff('000000000000000000000001', 'NURSE')._id.toString(), staff('000000000000000000000001', 'NURSE')],
  [staff('000000000000000000000002', 'NURSE')._id.toString(), staff('000000000000000000000002', 'NURSE')],
  [staff('000000000000000000000003', 'PHYSICIAN')._id.toString(), staff('000000000000000000000003', 'PHYSICIAN')],
]);

test('keeps triage and clinical nurse shift roles explicit and distinct', () => {
  assert.equal(normalizeShiftAssignmentRole('TRIAGE_NURSE', 'NURSE'), 'TRIAGE_NURSE');
  assert.equal(normalizeShiftAssignmentRole('CLINICAL_NURSE', 'NURSE'), 'CLINICAL_NURSE');
});

test('rejects missing or system-role-incompatible shift assignments', () => {
  assert.throws(() => normalizeShiftAssignmentRole('', 'NURSE'), /explicit valid shift assignment role/);
  assert.throws(() => normalizeShiftAssignmentRole('TRIAGE_NURSE', 'PHYSICIAN'), /incompatible/);
});

test('maps explicit clinical duties to their stable compatibility team fields', () => {
  assert.equal(getClinicalTeamRoleField('TRIAGE_NURSE'), 'triageNurseId');
  assert.equal(getClinicalTeamRoleField('CLINICAL_NURSE'), 'clinicalNurseId');
  assert.equal(getClinicalTeamRoleField('PHYSICIAN'), 'physicianId');
  assert.equal(getClinicalTeamRoleField('SUPPORT'), null);
});

test('validates and normalizes a hospital-scoped clinical team', () => {
  const normalized = validateClinicalTeam({
    shiftId: '000000000000000000000004',
    triageNurseId: '000000000000000000000001',
    clinicalNurseId: '000000000000000000000002',
    physicianId: '000000000000000000000003',
    staffById,
    hospitalId,
  });

  assert.equal(normalized.triageNurseId.toString(), '000000000000000000000001');
  assert.equal(normalized.clinicalNurseId.toString(), '000000000000000000000002');
  assert.equal(normalized.physicianId.toString(), '000000000000000000000003');
});

test('rejects malformed shift IDs before querying MongoDB', () => {
  assert.throws(
    () => validateClinicalTeam({
      shiftId: 'not-an-object-id',
      staffById,
      hospitalId,
    }),
    /Invalid shiftId/,
  );
});

test('rejects staff from another hospital', () => {
  const otherHospitalStaff = staff('000000000000000000000005', 'PHYSICIAN');
  otherHospitalStaff.hospitalId = new mongoose.Types.ObjectId();
  const otherStaffById = new Map([[otherHospitalStaff._id.toString(), otherHospitalStaff]]);

  assert.throws(
    () => validateClinicalTeam({
      shiftId: '000000000000000000000004',
      physicianId: otherHospitalStaff._id.toString(),
      staffById: otherStaffById,
      hospitalId,
    }),
    /outside this hospital/,
  );
});

test('rejects inactive staff', () => {
  const inactiveStaff = staff('000000000000000000000006', 'PHYSICIAN', 'TERMINATED');
  const inactiveStaffById = new Map([[inactiveStaff._id.toString(), inactiveStaff]]);

  assert.throws(
    () => validateClinicalTeam({
      shiftId: '000000000000000000000004',
      physicianId: inactiveStaff._id.toString(),
      staffById: inactiveStaffById,
      hospitalId,
    }),
    /inactive staff member/,
  );
});

test('builds normalized assigned-staff entries without string ObjectIds', () => {
  const assignedStaff = buildAssignedStaff([], staffById, {
    triageNurseId: new mongoose.Types.ObjectId('000000000000000000000001'),
  });

  assert.equal(assignedStaff.length, 1);
  assert.equal(assignedStaff[0].staffID.toString(), '000000000000000000000001');
  assert.equal(assignedStaff[0].type, 'NURSE');
  assert.equal(assignedStaff[0].status, 'ASSIGNED');
});

test('deduplicates existing roster entries and enriches them from canonical staff', () => {
  const nurseId = '000000000000000000000001';
  const duplicateRoster = [
    { staffID: nurseId, type: 'NURSE', name: 'Emma Thompson', specialty: '', status: 'ASSIGNED' },
    { staffID: new mongoose.Types.ObjectId(nurseId), type: 'NURSE', name: 'Emma Thompson Thompson', specialty: '', status: 'ASSIGNED' },
  ];

  const assignedStaff = buildAssignedStaff(duplicateRoster, staffById, {
    clinicalNurseId: new mongoose.Types.ObjectId(nurseId),
  });

  assert.equal(assignedStaff.length, 1);
  assert.equal(assignedStaff[0].staffID.toString(), nurseId);
  assert.equal(assignedStaff[0].name, 'Test Nurse');
  assert.equal(assignedStaff[0].specialty, 'General Practice');
});
