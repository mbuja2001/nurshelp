/**
 * Scheduled Background Jobs Service
 * 
 * Manages recurring automated tasks:
 * - Auto clock-out for expired shifts
 * - Other scheduled maintenance tasks
 */

import { NurseClockRecord } from "../models/staffClockRecord.model.js";

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
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    // Find all active clock records (CLOCKED_IN or ON_BREAK status)
    const activeRecords = await NurseClockRecord.find({
      date: { $gte: today, $lt: tomorrow },
      status: { $in: ["CLOCKED_IN", "ON_BREAK"] },
      clockOutTime: null
    }).populate("shiftId");

    let clockedOutCount = 0;

    for (const record of activeRecords) {
      try {
        if (!record.shiftId || !record.shiftId.shiftEnd) {
          continue;
        }

        // Parse shift end time
        const [hours, minutes] = record.shiftId.shiftEnd.split(":").map(Number);
        const shiftEndDate = new Date(record.date);
        shiftEndDate.setHours(hours, minutes, 0, 0);

        // Check if shift has ended
        if (now > shiftEndDate) {
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

/**
 * Initialize all scheduled jobs
 */
export const initializeScheduledJobs = () => {
  // Run auto clock out job every 5 minutes
  setInterval(autoClockOutJob, 5 * 60 * 1000);
  console.log("[SCHEDULED JOBS] Auto clock-out job scheduled to run every 5 minutes");
};

export default { initializeScheduledJobs };
