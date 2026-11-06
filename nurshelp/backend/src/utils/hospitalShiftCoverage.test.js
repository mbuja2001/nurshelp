import test from 'node:test';
import assert from 'node:assert/strict';
import {
  canonicalizeServiceCoverage,
  inferAssignmentServiceCoverage,
} from './hospitalShiftCoverage.js';

const configuredSpecialties = [
  'Emergency Medicine',
  'Emergency Nursing',
  'Radiology',
  'Clinical Pathology',
  'Lab Technician',
  'Clinical Pharmacist',
];

test('canonicalizes roster service labels to hospital-configured specialties', () => {
  assert.equal(canonicalizeServiceCoverage('Radiology Technician', configuredSpecialties), 'Radiology');
  assert.equal(canonicalizeServiceCoverage('Laboratory Technician', configuredSpecialties), 'Lab Technician');
  assert.equal(canonicalizeServiceCoverage('Pharmacist', configuredSpecialties), 'Clinical Pharmacist');
});

test('infers emergency and specialist services from explicit assignment roles', () => {
  assert.equal(inferAssignmentServiceCoverage({
    staff: { clinicalSpecialty: 'Radiology Technician' },
    assignmentRole: 'SUPPORT',
    configuredSpecialties,
  }), 'Radiology');
  assert.equal(inferAssignmentServiceCoverage({
    staff: {},
    assignmentRole: 'PHYSICIAN',
    configuredSpecialties,
  }), 'Emergency Medicine');
});

test('does not accept services absent from the hospital structure', () => {
  assert.equal(canonicalizeServiceCoverage('Radiology', ['Emergency Medicine']), null);
});
