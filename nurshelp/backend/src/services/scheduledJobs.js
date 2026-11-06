/**
 * Scheduled Background Jobs Service
 * 
 * Manages recurring automated tasks:
 * - Auto clock-out for expired shifts
 * - Other scheduled maintenance tasks
 */

import { NurseClockRecord } from "../models/staffClockRecord.model.js";
import { Shift } from "../models/shift.model.js";
import { AuditLog } from "../models/auditLog.model.js";
import { getScheduledShiftAutoEndAt } from "../utils/scheduledShiftEnd.js";

/**
 * Auto Clock Out Expired Shifts
 * Runs every 5 minutes
 * 
 * Finds all staff (clinical staff) still clocked in after their shift ends
 * Automatically clocks them out with AUTO_CLOCKOUT and DIDNT_CLOCKOUT flags
 * Uses NurseClockRecord model (collection: 'nurseClockRecords')
 */
const autoClockOutJob = async () => {
  try {
    const now = new Date();
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    // Find all active clock records (CLOCKED_IN or ON_BREAK status)
    const activeRecords = await NurseClockRecord.find({
      date: { $gte: yesterday, $lt: tomorrow },
      status: { $in: ["CLOCKED_IN", "ON_BREAK"] },
      clockOutTime: null
    }).populate("shiftId");

    let clockedOutCount = 0;

    for (const record of activeRecords) {
      try {
        if (!record.shiftId || !record.shiftId.shiftEnd) {
          continue;
        }

        const autoClockOutAt = getScheduledShiftAutoEndAt({
          date: record.date,
          shiftStart: record.shiftId.shiftStart,
          shiftEnd: record.shiftId.shiftEnd
        });

        if (autoClockOutAt && now >= autoClockOutAt) {
          console.log(`[SCHEDULED AUTO CLOCK OUT] Shift expired for staff ${record.staffId}. Auto-clocking out...`);

          // Update record
          record.clockOutTime = now;
          record.status = "CLOCKED_OUT";
          
          // Calculate duration
          if (record.clockInTime) {
            record.totalShiftDuration = Math.round((now - record.clockInTime) / 1000);
          }

          // Add flags
          record.flags = record.flags || [];
          if (!record.flags.includes("AUTO_CLOCKOUT")) {
            record.flags.push("AUTO_CLOCKOUT");
          }
          if (!record.flags.includes("DIDNT_CLOCKOUT")) {
            record.flags.push("DIDNT_CLOCKOUT");
          }

          await record.save();
          clockedOutCount++;
        }
      } catch (err) {
        console.error(`[SCHEDULED AUTO CLOCK OUT] Error processing record:`, err);
      }
    }

    if (clockedOutCount > 0) {
      console.log(`[SCHEDULED AUTO CLOCK OUT] Auto-clocked out ${clockedOutCount} staff members with expired shifts`);
    }
  } catch (err) {
    console.error("[SCHEDULED AUTO CLOCK OUT] Error:", err);
  }
};

const autoCompleteScheduledHospitalShifts = async () => {
  const now = new Date();
  const activeShifts = await Shift.find({ status: 'IN_PROGRESS' });

  for (const shift of activeShifts) {
    try {
      const autoEndAt = getScheduledShiftAutoEndAt(shift);
      if (!autoEndAt || now < autoEndAt) continue;
      const completedShift = await Shift.findOneAndUpdate(
        { _id: shift._id, status: 'IN_PROGRESS' },
        { $set: { status: 'COMPLETED', completedAt: now, completedSource: 'AUTO_SCHEDULED' } },
        { new: true }
      );
      if (!completedShift) continue;

      await AuditLog.create({
        action: 'HOSPITAL_SHIFT_AUTO_ENDED',
        actorId: completedShift.startedBy || completedShift.createdBy,
        hospitalId: completedShift.hospitalId,
        resourceId: completedShift._id,
        resourceType: 'SHIFT',
        authorized: true,
        changeDetails: {
          shiftId: String(completedShift._id),
          shiftID: completedShift.shiftID,
          shiftName: completedShift.shiftName,
          status: completedShift.status,
          scheduledEndAt: new Date(autoEndAt.getTime() - 5 * 60 * 1000),
          autoEndAt,
          completedAt: completedShift.completedAt,
          completionSource: 'AUTO_SCHEDULED'
        },
        description: `Hospital shift ${completedShift.shiftName} automatically completed five minutes after its scheduled end`
      });
      console.info(`[SCHEDULED SHIFT END] Automatically completed shift ${completedShift.shiftID} after scheduled-end grace`);
    } catch (error) {
      console.error(`[SCHEDULED SHIFT END] Failed to automatically end shift ${shift.shiftID}:`, error);
    }
  }
};

/**
 * Initialize all scheduled jobs
 */
export const initializeScheduledJobs = () => {
  // Run auto clock out job every 5 minutes
  setInterval(autoClockOutJob, 5 * 60 * 1000);
  console.log("[SCHEDULED JOBS] Auto clock-out job scheduled to run every 5 minutes");
  autoCompleteScheduledHospitalShifts().catch((error) => console.error('[SCHEDULED SHIFT END] Initial sweep failed:', error));
  setInterval(() => {
    autoCompleteScheduledHospitalShifts().catch((error) => console.error('[SCHEDULED SHIFT END] Sweep failed:', error));
  }, 30 * 1000);
  console.log("[SCHEDULED SHIFT END] Scheduled shift end deadlines checked every 30 seconds");
};

export default { initializeScheduledJobs };
