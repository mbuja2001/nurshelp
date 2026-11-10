import test from 'node:test';
import assert from 'node:assert/strict';
import {
  findHospitalStaffRole,
  HOSPITAL_STAFF_ROLES,
  legacyHospitalStaffRole
} from './hospitalStaffRoleCatalog.js';

test('staff role catalog has unique stable codes and display names', () => {
  assert.equal(new Set(HOSPITAL_STAFF_ROLES.map((role) => role.code)).size, HOSPITAL_STAFF_ROLES.length);
  assert.equal(new Set(HOSPITAL_STAFF_ROLES.map((role) => role.label)).size, HOSPITAL_STAFF_ROLES.length);
});

test('all non-manager support positions use the restricted SUPPORT system role', () => {
  const supportRoles = HOSPITAL_STAFF_ROLES.filter((role) => role.category !== 'Licensed clinical' && role.code !== 'TECHNICIAN');
  assert.ok(supportRoles.length >= 25);
  assert.ok(supportRoles.every((role) => role.systemRole === 'SUPPORT'));
  assert.equal(findHospitalStaffRole('ADMINISTRATIVE_SUPPORT').systemRole, 'SUPPORT');
});

test('catalog covers the requested patient-care, allied-health, procedural, and logistics positions', () => {
  const expectedCodes = [
    'CERTIFIED_NURSING_ASSISTANT', 'NURSE_AIDE', 'PATIENT_CARE_TECHNICIAN',
    'GERIATRIC_CARE_ASSISTANT', 'PEDIATRIC_CARE_ASSISTANT', 'BEHAVIORAL_HEALTH_AIDE',
    'UNIT_SECRETARY', 'WARD_CLERK', 'HOSPITAL_ORDERLY', 'PHYSICAL_THERAPY_ASSISTANT',
    'OCCUPATIONAL_THERAPY_ASSISTANT', 'SPEECH_LANGUAGE_PATHOLOGY_ASSISTANT',
    'DIETETIC_ASSISTANT', 'DIETARY_AIDE', 'PHARMACY_ASSISTANT', 'MEDICAL_ASSISTANT',
    'DENTAL_ASSISTANT', 'ANESTHESIA_ASSISTANT', 'ANESTHESIA_TECHNOLOGIST_ASSISTANT',
    'ENDOSCOPY_GI_ASSISTANT', 'CENTRAL_STERILE_SUPPLY_ASSISTANT', 'PATIENT_TRANSPORTER',
    'HOSPITAL_PORTER', 'MEDICAL_SOCIAL_WORK_ASSISTANT', 'DISCHARGE_PLANNER',
    'PATIENT_NAVIGATOR', 'LANGUAGE_INTERPRETER'
  ];
  assert.ok(expectedCodes.every((code) => findHospitalStaffRole(code)));
});

test('legacy clinical roles remain mappable without granting administrative access', () => {
  assert.equal(legacyHospitalStaffRole('NURSE').code, 'REGISTERED_NURSE');
  assert.equal(legacyHospitalStaffRole('PHYSICIAN').code, 'PHYSICIAN');
  assert.equal(legacyHospitalStaffRole('ADMINISTRATIVE'), null);
});

test('role identity stays separate from the permission-bearing system role', () => {
  const patientTransporter = findHospitalStaffRole('PATIENT_TRANSPORTER');
  assert.equal(patientTransporter.label, 'Patient transporter');
  assert.equal(patientTransporter.systemRole, 'SUPPORT');
});
