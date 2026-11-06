import { getScheduledShiftAutoEndAt } from "./scheduledShiftEnd.js";

export const getClinicalStaffAssignmentAvailability = (clockRecord, now = new Date()) => {
  if (!clockRecord || clockRecord.clockOutTime) {
    return { available: false, availabilityStatus: "OFF_SHIFT" };
  }

  if (clockRecord.status === "ON_BREAK") {
    return { available: false, availabilityStatus: "ON_BREAK" };
  }

  const autoEndAt = getScheduledShiftAutoEndAt(clockRecord.shiftId);
  const available = clockRecord.status === "CLOCKED_IN"
    && clockRecord.shiftId?.status === "IN_PROGRESS"
    && autoEndAt !== null
    && now <= autoEndAt;

  return available
    ? { available: true, availabilityStatus: "ON_SHIFT" }
    : { available: false, availabilityStatus: "OFF_SHIFT" };
};
