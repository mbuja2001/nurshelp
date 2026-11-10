import test from 'node:test';
import assert from 'node:assert/strict';
import mongoose from 'mongoose';
import { Shift } from './shift.model.js';

test('hydrates a legacy shift and preserves its original department as the snapshot', () => {
  const shift = Shift.hydrate({ department: 'Emergency Department' });

  assert.equal(shift.department, 'Emergency Department');
  assert.equal(shift.departmentName, 'Emergency Department');
  assert.equal(shift.departmentId, null);
});

test('allows a shift without a department', () => {
  const shift = new Shift({
    hospitalId: new mongoose.Types.ObjectId(),
    shiftID: 'SHIFT-HOSPITAL-WIDE',
    shiftName: 'Hospital Day Shift',
    shiftStart: '08:00',
    shiftEnd: '16:00',
    duration: 8,
    maxStaffRequired: 1,
    createdBy: new mongoose.Types.ObjectId(),
  });

  assert.equal(shift.department, null);
  assert.equal(shift.departmentName, null);
  assert.equal(shift.validateSync(), undefined);
});
