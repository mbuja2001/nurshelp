import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveConfiguredDepartment } from './hospitalDepartment.js';

const departments = [
  { name: 'Emergency Department', departmentId: 'DEPT-ED' },
  { name: 'Radiology', departmentId: 'DEPT-RAD', isActive: false },
];

test('resolves active departments by configured name and stable id', () => {
  assert.deepEqual(resolveConfiguredDepartment(departments, 'Emergency Department'), {
    departmentId: 'DEPT-ED',
    departmentName: 'Emergency Department',
  });
  assert.deepEqual(resolveConfiguredDepartment(departments, 'DEPT-ED'), {
    departmentId: 'DEPT-ED',
    departmentName: 'Emergency Department',
  });
});

test('rejects unknown and inactive departments', () => {
  assert.equal(resolveConfiguredDepartment(departments, 'Radiology'), null);
  assert.equal(resolveConfiguredDepartment(departments, 'Unconfigured'), null);
});

test('uses a stable code for legacy configured departments without one', () => {
  assert.deepEqual(resolveConfiguredDepartment([{ name: 'Medical-Surgical' }], 'Medical-Surgical'), {
    departmentId: 'DEPT-MEDICAL-SURGICAL',
    departmentName: 'Medical-Surgical',
  });
});
