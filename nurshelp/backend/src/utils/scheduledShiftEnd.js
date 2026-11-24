export const SCHEDULED_SHIFT_END_GRACE_MS = 5 * 60 * 1000;
export const SCHEDULED_SHIFT_START_GRACE_MS = 5 * 60 * 1000;

export const getScheduledShiftWindow = (shift) => {
  if (!shift?.date || !shift?.shiftStart || !shift?.shiftEnd) return null;

  let year;
  let month;
  let day;
  if (typeof shift.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(shift.date)) {
    [year, month, day] = shift.date.split('-').map(Number);
  } else {
    const shiftDate = new Date(shift.date);
    if (Number.isNaN(shiftDate.getTime())) return null;
    year = shiftDate.getUTCFullYear();
    month = shiftDate.getUTCMonth() + 1;
    day = shiftDate.getUTCDate();
  }
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  const calendarDate = new Date(Date.UTC(year, month - 1, day));
  if (calendarDate.getUTCFullYear() !== year || calendarDate.getUTCMonth() !== month - 1 || calendarDate.getUTCDate() !== day) return null;

  const [startHour, startMinute] = String(shift.shiftStart).split(':').map(Number);
  const [endHour, endMinute] = String(shift.shiftEnd).split(':').map(Number);
  if ([startHour, startMinute, endHour, endMinute].some((value) => !Number.isInteger(value))) return null;
  if (
    startHour < 0 || startHour > 23 || endHour < 0 || endHour > 23 ||
    startMinute < 0 || startMinute > 59 || endMinute < 0 || endMinute > 59
  ) return null;

  const startAt = new Date(year, month - 1, day, startHour, startMinute, 0, 0);
  const endAt = new Date(year, month - 1, day, endHour, endMinute, 0, 0);
  if (endAt <= startAt) endAt.setDate(endAt.getDate() + 1);
  return { startAt, endAt };
};

export const getScheduledShiftStartAt = (shift) => getScheduledShiftWindow(shift)?.startAt || null;

export const getScheduledShiftEndAt = (shift) => getScheduledShiftWindow(shift)?.endAt || null;

export const isPrematureAutoClockOut = (record, shift) => {
  if (record?.status !== 'CLOCKED_OUT' || !record.flags?.includes('AUTO_CLOCKOUT')) return false;

  const scheduledEndAt = getScheduledShiftEndAt(shift);
  const clockOutAt = new Date(record.clockOutTime);
  return Boolean(scheduledEndAt && !Number.isNaN(clockOutAt.getTime()) && clockOutAt < scheduledEndAt);
};

export const getScheduledShiftAutoEndAt = (shift) => {
  const scheduledEndAt = getScheduledShiftEndAt(shift);
  return scheduledEndAt ? new Date(scheduledEndAt.getTime() + SCHEDULED_SHIFT_END_GRACE_MS) : null;
};
