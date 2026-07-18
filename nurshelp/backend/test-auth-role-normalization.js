import assert from 'node:assert/strict';
import { deriveTokenRole } from './src/controllers/auth.controller.js';

const cases = [
  { role: 'PHARMACIST', specialty: 'Clinical Pharmacist', expected: 'PHARMACIST' },
  { role: 'RADIOLOGIST', specialty: 'Radiology', expected: 'RADIOLOGIST' },
  { role: 'PATHOLOGIST', specialty: 'Pathology', expected: 'PATHOLOGIST' },
  { role: 'TECHNICIAN', specialty: 'Pharmacy', expected: 'PHARMACIST' },
  { role: 'TECHNICIAN', specialty: 'Radiology', expected: 'RADIOLOGIST' },
  { role: 'TECHNICIAN', specialty: 'Pathology', expected: 'PATHOLOGIST' },
  { role: 'ADMINISTRATIVE', specialty: 'General', expected: 'ADMIN' }
];

for (const testCase of cases) {
  const actual = deriveTokenRole(testCase.role, testCase.specialty);
  assert.equal(actual, testCase.expected, `${testCase.role}/${testCase.specialty} -> ${actual}`);
}

console.log(`Verified ${cases.length} auth role normalization cases.`);
