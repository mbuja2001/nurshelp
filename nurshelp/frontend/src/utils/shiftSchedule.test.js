import { describe, expect, it } from 'vitest';
import { getHospitalDayShiftSchedule } from './shiftSchedule';

describe('getHospitalDayShiftSchedule', () => {
  it('moves an elapsed same-day start to the next minute and preserves shift duration', () => {
    const now = new Date(2026, 10, 14, 8, 21, 28);

    expect(getHospitalDayShiftSchedule({
      date: '2026-11-14',
      shiftStart: '08:00',
      shiftEnd: '16:00'
    }, now)).toEqual({
      date: '2026-11-14',
      shiftStart: '08:22',
      shiftEnd: '16:22',
      adjusted: true
    });
  });

  it('leaves a future scheduled start unchanged', () => {
    const now = new Date(2026, 10, 14, 8, 21, 28);

    expect(getHospitalDayShiftSchedule({
      date: '2026-11-14',
      shiftStart: '09:00',
      shiftEnd: '17:00'
    }, now)).toEqual({
      date: '2026-11-14',
      shiftStart: '09:00',
      shiftEnd: '17:00',
      adjusted: false
    });
  });

  it('rejects invalid and past schedule dates', () => {
    const now = new Date(2026, 10, 14, 8, 21, 28);

    expect(getHospitalDayShiftSchedule({
      date: '2026-11-13',
      shiftStart: '08:00',
      shiftEnd: '16:00'
    }, now)).toBeNull();
    expect(getHospitalDayShiftSchedule({
      date: '2026-11-14',
      shiftStart: 'invalid',
      shiftEnd: '16:00'
    }, now)).toBeNull();
  });
});