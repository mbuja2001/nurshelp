import test from 'node:test';
import assert from 'node:assert/strict';
import { isSelfAssignmentAllowed, resolveShiftAssignmentRole } from './assignmentDutyRole.js';

test('shift roster assignment role takes precedence over broad team role', () => {
  const shift = {
    assignedStaff: [{ staffID: 'staff-1', assignmentRole: 'CLINICAL_NURSE' }],
    teamAssignment: { triageNurseId: 'staff-1' }
  };

  assert.equal(resolveShiftAssignmentRole(shift, 'staff-1'), 'CLINICAL_NURSE');
});

test('shift team assignment role is used when roster role is absent', () => {
  const shift = { teamAssignment: { triageNurseId: { _id: 'staff-2' } } };
  assert.equal(resolveShiftAssignmentRole(shift, 'staff-2'), 'TRIAGE_NURSE');
});

test('triage duty cannot self-assign downstream; clinical nurse duty can', () => {
  assert.equal(isSelfAssignmentAllowed('staff-1', 'staff-1', 'TRIAGE_NURSE'), false);
  assert.equal(isSelfAssignmentAllowed('staff-1', 'staff-1', null), false);
  assert.equal(isSelfAssignmentAllowed('staff-1', 'staff-1', 'CLINICAL_NURSE'), true);
  assert.equal(isSelfAssignmentAllowed('staff-1', 'staff-2', 'TRIAGE_NURSE'), true);
});
