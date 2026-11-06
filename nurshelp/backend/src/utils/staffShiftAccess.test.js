import test from 'node:test';
import assert from 'node:assert/strict';
import { isShiftApprovedForClockIn, isStaffAssignedToShift } from './staffShiftAccess.js';

const shift = {
  hospitalId: 'hospital-a',
  assignedStaff: [
    { staffID: 'rostered-staff', status: 'ASSIGNED' },
    { staffID: 'absent-staff', status: 'ABSENT' }
  ],
  teamAssignment: { triageNurseId: 'triage-staff', physicianId: 'physician-staff' }
};

test('allows a same-hospital roster or clinical team assignment', () => {
  assert.equal(isStaffAssignedToShift(shift, 'rostered-staff', 'hospital-a'), true);
  assert.equal(isStaffAssignedToShift(shift, 'triage-staff', 'hospital-a'), true);
  assert.equal(isStaffAssignedToShift(shift, 'physician-staff', 'hospital-a'), true);
});

test('denies absent, unrelated, and cross-hospital staff', () => {
  assert.equal(isStaffAssignedToShift(shift, 'absent-staff', 'hospital-a'), false);
  assert.equal(isStaffAssignedToShift(shift, 'unassigned-staff', 'hospital-a'), false);
  assert.equal(isStaffAssignedToShift(shift, 'rostered-staff', 'hospital-b'), false);
});

test('only an administrator-started shift is eligible for staff clock-in', () => {
  assert.equal(isShiftApprovedForClockIn({ status: 'PUBLISHED' }), false);
  assert.equal(isShiftApprovedForClockIn({ status: 'CANCELLED' }), false);
  assert.equal(isShiftApprovedForClockIn({ status: 'IN_PROGRESS' }), true);
});
