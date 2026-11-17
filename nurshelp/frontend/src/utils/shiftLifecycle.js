export const getScheduledShiftWindow = (shift) => {
  if (!shift?.date || !shift?.shiftStart || !shift?.shiftEnd) return null;

  const dateValue = shift.date instanceof Date
    ? shift.date.toISOString().slice(0, 10)
    : String(shift.date).slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(dateValue)) return null;
  const [year, month, day] = dateValue.split('-').map(Number);
  const calendarDate = new Date(year, month - 1, day);
  if (calendarDate.getFullYear() !== year || calendarDate.getMonth() !== month - 1 || calendarDate.getDate() !== day) return null;

  const parseTime = (value) => {
    if (!/^\d{2}:\d{2}$/.test(value)) return null;
    const [hour, minute] = value.split(':').map(Number);
    if (hour > 23 || minute > 59) return null;
    return { hour, minute };
  };
  const start = parseTime(shift.shiftStart);
  const end = parseTime(shift.shiftEnd);
  if (!start || !end) return null;

  const startAt = new Date(year, month - 1, day, start.hour, start.minute);
  const endAt = new Date(year, month - 1, day, end.hour, end.minute);
  if (endAt <= startAt) endAt.setDate(endAt.getDate() + 1);
  return { startAt, endAt };
};

export const isPrematureAutoClockOut = (record, shift) => {
  if (
    record?.status !== 'CLOCKED_OUT' ||
    !record.flags?.includes('AUTO_CLOCKOUT')
  ) return false;

  const shiftWindow = getScheduledShiftWindow(shift);
  const clockOutAt = new Date(record.clockOutTimestamp ?? record.clockOutTime);
  return Boolean(
    shiftWindow &&
    !Number.isNaN(clockOutAt.getTime()) &&
    clockOutAt < shiftWindow.endAt
  );
};

export const getFuturePublishedShifts = (shifts, now = new Date()) => (
  (shifts || []).filter((shift) => {
    if (shift.status !== 'PUBLISHED') return false;
    const window = getScheduledShiftWindow(shift);
    return Boolean(window && window.startAt > now);
  })
);

export const getStartablePublishedShifts = (shifts, now = new Date()) => (
  (shifts || []).filter((shift) => {
    if (shift.status !== 'PUBLISHED') return false;
    const window = getScheduledShiftWindow(shift);
    return Boolean(window && window.endAt > now);
  })
);
