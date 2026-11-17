import { describe, expect, it } from 'vitest';
import { getFuturePublishedShifts, getScheduledShiftWindow, getStartablePublishedShifts, isPrematureAutoClockOut } from './shiftLifecycle.js';

describe('shift lifecycle scheduling', () => {
  it('uses the calendar date and carries overnight shift ends into the next day', () => {
    const window = getScheduledShiftWindow({
      date: '2030-10-14T00:00:00.000Z',
      shiftStart: '19:00',
      shiftEnd: '07:00',
    });

    expect(window.startAt.getDate()).toBe(14);
    expect(window.startAt.getHours()).toBe(19);
    expect(window.endAt.getDate()).toBe(15);
    expect(window.endAt.getHours()).toBe(7);
  });

  it('only offers published shifts with a future start time', () => {
    const now = new Date(2030, 9, 14, 12, 0);
    const shifts = [
      { status: 'PUBLISHED', date: '2030-10-14', shiftStart: '13:00', shiftEnd: '21:00' },
      { status: 'PUBLISHED', date: '2030-10-14', shiftStart: '08:00', shiftEnd: '16:00' },
      { status: 'IN_PROGRESS', date: '2030-10-14', shiftStart: '13:00', shiftEnd: '21:00' },
    ];

    expect(getFuturePublishedShifts(shifts, now)).toEqual([shifts[0]]);
  });

  it('allows a scheduled shift to start after its start time but before its end', () => {
    const now = new Date(2030, 9, 14, 12, 0);
    const shifts = [
      { status: 'PUBLISHED', date: '2030-10-14', shiftStart: '08:00', shiftEnd: '16:00' },
      { status: 'PUBLISHED', date: '2030-10-14', shiftStart: '07:00', shiftEnd: '11:00' },
      { status: 'IN_PROGRESS', date: '2030-10-14', shiftStart: '08:00', shiftEnd: '16:00' },
    ];

    expect(getStartablePublishedShifts(shifts, now)).toEqual([shifts[0]]);
  });

  it('rejects invalid calendar dates and clock times', () => {
    expect(getScheduledShiftWindow({ date: '2030-02-30', shiftStart: '08:00', shiftEnd: '16:00' })).toBeNull();
    expect(getScheduledShiftWindow({ date: '2030-10-14', shiftStart: '25:00', shiftEnd: '16:00' })).toBeNull();
  });

  it('allows retry after an automatic clock-out that happened before the scheduled shift end', () => {
    const shift = {
      date: '2030-10-14',
      shiftStart: '19:30',
      shiftEnd: '03:30',
    };

    expect(isPrematureAutoClockOut({
      status: 'CLOCKED_OUT',
      flags: ['AUTO_CLOCKOUT'],
      clockOutTime: '08:26:00 PM',
      clockOutTimestamp: new Date(2030, 9, 14, 20, 26).toISOString(),
    }, shift)).toBe(true);

    expect(isPrematureAutoClockOut({
      status: 'CLOCKED_OUT',
      flags: ['AUTO_CLOCKOUT'],
      clockOutTime: new Date(2030, 9, 14, 20, 26),
    }, shift)).toBe(true);
  });

  it('keeps normally completed and end-of-shift auto-clocked records completed', () => {
    const shift = {
      date: '2030-10-14',
      shiftStart: '19:30',
      shiftEnd: '03:30',
    };
    const autoClockOutAtEnd = {
      status: 'CLOCKED_OUT',
      flags: ['AUTO_CLOCKOUT'],
      clockOutTime: new Date(2030, 9, 15, 3, 35),
    };
    const manualClockOut = {
      status: 'CLOCKED_OUT',
      flags: [],
      clockOutTime: new Date(2030, 9, 14, 22, 0),
    };

    expect(isPrematureAutoClockOut(autoClockOutAtEnd, shift)).toBe(false);
    expect(isPrematureAutoClockOut(manualClockOut, shift)).toBe(false);
  });
});
