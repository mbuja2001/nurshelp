import test from 'node:test';
import assert from 'node:assert/strict';
import { getScheduledShiftAutoEndAt, getScheduledShiftEndAt, getScheduledShiftStartAt, getScheduledShiftWindow, isPrematureAutoClockOut, SCHEDULED_SHIFT_END_GRACE_MS } from './scheduledShiftEnd.js';

test('scheduled shift end grace is calculated from the scheduled end time', () => {
  const shift = { date: '2026-10-14', shiftStart: '08:00', shiftEnd: '16:00' };
  const scheduledEnd = getScheduledShiftEndAt(shift);
  const autoEnd = getScheduledShiftAutoEndAt(shift);

  assert.ok(scheduledEnd instanceof Date);
  assert.equal(autoEnd.getTime() - scheduledEnd.getTime(), SCHEDULED_SHIFT_END_GRACE_MS);
  assert.equal(scheduledEnd.getHours(), 16);
});

test('overnight shifts end on the following day', () => {
  const window = getScheduledShiftWindow({
    date: '2026-10-14',
    shiftStart: '19:00',
    shiftEnd: '07:00'
  });

  assert.equal(window.startAt.getDate(), 14);
  assert.equal(window.startAt.getHours(), 19);
  assert.equal(window.endAt.getDate(), 15);
  assert.equal(window.endAt.getHours(), 7);
  assert.equal(getScheduledShiftStartAt({ date: '2026-10-14', shiftStart: '19:00', shiftEnd: '07:00' }).getTime(), window.startAt.getTime());
  assert.equal(getScheduledShiftEndAt({ date: '2026-10-14', shiftStart: '19:00', shiftEnd: '07:00' }).getTime(), window.endAt.getTime());
});

test('incomplete or invalid shift times do not get an automatic end deadline', () => {
  assert.equal(getScheduledShiftAutoEndAt({ date: new Date(), shiftStart: '8am', shiftEnd: '16:00' }), null);
  assert.equal(getScheduledShiftAutoEndAt({ date: null, shiftStart: '08:00', shiftEnd: '16:00' }), null);
  assert.equal(getScheduledShiftWindow({ date: '2026-02-30', shiftStart: '08:00', shiftEnd: '16:00' }), null);
});

test('only an early automatic clock-out is retryable for the same shift', () => {
  const shift = { date: '2026-10-14', shiftStart: '19:30', shiftEnd: '03:30' };
  const earlyAutoClockOut = {
    status: 'CLOCKED_OUT',
    flags: ['AUTO_CLOCKOUT'],
    clockOutTime: new Date(2026, 9, 14, 20, 26)
  };
  const completedAutoClockOut = {
    ...earlyAutoClockOut,
    clockOutTime: new Date(2026, 9, 15, 3, 35)
  };

  assert.equal(isPrematureAutoClockOut(earlyAutoClockOut, shift), true);
  assert.equal(isPrematureAutoClockOut(completedAutoClockOut, shift), false);
  assert.equal(isPrematureAutoClockOut({ ...earlyAutoClockOut, flags: [] }, shift), false);
});
