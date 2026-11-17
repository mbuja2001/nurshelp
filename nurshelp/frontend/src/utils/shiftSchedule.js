import { getScheduledShiftWindow } from './shiftLifecycle';

const formatDate = (date) => [
  date.getFullYear(),
  String(date.getMonth() + 1).padStart(2, '0'),
  String(date.getDate()).padStart(2, '0')
].join('-');

const formatTime = (date) => [
  String(date.getHours()).padStart(2, '0'),
  String(date.getMinutes()).padStart(2, '0')
].join(':');

export const getHospitalDayShiftSchedule = ({ date, shiftStart, shiftEnd }, now = new Date()) => {
  const window = getScheduledShiftWindow({ date, shiftStart, shiftEnd });
  if (!window) return null;

  const scheduleDate = String(date).slice(0, 10);
  const today = formatDate(now);
  if (scheduleDate < today) return null;
  if (scheduleDate > today || window.startAt > now) {
    return { date: scheduleDate, shiftStart, shiftEnd, adjusted: false };
  }

  const durationMs = window.endAt.getTime() - window.startAt.getTime();
  const adjustedStart = new Date(now);
  adjustedStart.setSeconds(0, 0);
  adjustedStart.setMinutes(adjustedStart.getMinutes() + 1);
  const adjustedEnd = new Date(adjustedStart.getTime() + durationMs);

  return {
    date: formatDate(adjustedStart),
    shiftStart: formatTime(adjustedStart),
    shiftEnd: formatTime(adjustedEnd),
    adjusted: true
  };
};