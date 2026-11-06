import test from 'node:test';
import assert from 'node:assert/strict';
import { getScheduledShiftAutoEndAt, getScheduledShiftEndAt, SCHEDULED_SHIFT_END_GRACE_MS } from './scheduledShiftEnd.js';

test('scheduled shift end grace is calculated from the scheduled end time', () => {
  const shift = { date: '2026-10-14', shiftStart: '08:00', shiftEnd: '16:00' };
  const scheduledEnd = getScheduledShiftEndAt(shift);
  const autoEnd = getScheduledShiftAutoEndAt(shift);

  assert.ok(scheduledEnd instanceof Date);
  assert.equal(autoEnd.getTime() - scheduledEnd.getTime(), SCHEDULED_SHIFT_END_GRACE_MS);
  assert.equal(scheduledEnd.getHours(), 16);
});

test('overnight shifts end on the following day', () => {
  const scheduledEnd = getScheduledShiftEndAt({
    date: '2026-10-14',
    shiftStart: '19:00',
    shiftEnd: '07:00'
  });

  assert.equal(scheduledEnd.getDate(), 15);
  assert.equal(scheduledEnd.getHours(), 7);
});

test('incomplete or invalid shift times do not get an automatic end deadline', () => {
  assert.equal(getScheduledShiftAutoEndAt({ date: new Date(), shiftStart: '8am', shiftEnd: '16:00' }), null);
  assert.equal(getScheduledShiftAutoEndAt({ date: null, shiftStart: '08:00', shiftEnd: '16:00' }), null);
});
