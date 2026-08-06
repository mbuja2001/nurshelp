import { ClinicalStaff } from "../models/clinicalstaff.model.js";
import { NurseClockRecord } from "../models/staffClockRecord.model.js";
import { NurseSchedule } from "../models/nurseSchedule.model.js";
import { Shift } from "../models/shift.model.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

// ----------------------
// REGISTER
// ----------------------
export const registerStaff = async (req, res) => {
  // Legacy/self-registration endpoint disabled.
  // Staff accounts must be created by a Manager via /api/manager/staff/register
  console.warn('[REGISTER STAFF] Self-registration endpoint called - disabled');
  return res.status(403).json({
    success: false,
    message: 'Self-registration disabled. Create staff via POST /api/manager/staff/register (Manager only)'
  });
};

// ----------------------
// LOGIN
// ----------------------
export const loginStaff = async (req, res) => {
  try {
    console.log("[STAFF LOGIN] Attempting login with:", req.body.email);

    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ message: "Email & password required" });
    }

    const staff = await ClinicalStaff.findOne({ email }).select('+password');
    const staffDisplayName = staff ? (staff.name || `${staff.firstName || ''} ${staff.lastName || ''}`.trim()) : null;
    const staffDisplayRole = staff ? (staff.role || staff.systemRole) : null;
    console.log("[STAFF LOGIN] Staff found:", staff ? `${staffDisplayName} (Role: ${staffDisplayRole})` : "NOT FOUND");

    if (!staff) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    const valid = await bcrypt.compare(password, staff.password);
    console.log("[STAFF LOGIN] Password valid:", valid);

    if (!valid) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    // Populate hospital info if available
    const populatedStaff = await ClinicalStaff.findById(staff._id).populate('hospitalId');
    const hospitalName = populatedStaff?.hospitalId?.name || 'Hospital';
    const hospitalId = populatedStaff?.hospitalId?._id || staff.hospitalId;

    // Update metadata.lastLogin
    if (!staff.metadata) staff.metadata = {};
    staff.metadata.lastLogin = new Date();
    await staff.save();

    // Normalize values
    const tokenName = staff.name || `${staff.firstName || ''} ${staff.lastName || ''}`.trim();
    const tokenRole = staff.role || staff.systemRole;
    const tokenDept = staff.department || staff.assignment?.department;

    // Create token with role and hospital info (like manager login)
    const token = jwt.sign(
      {
        id: staff._id,
        email: staff.email,
        name: tokenName,
        role: tokenRole,
        department: tokenDept,
        hospitalId: hospitalId,
        hospitalName: hospitalName
      },
      process.env.JWT_SECRET || "supersecretkey",
      { expiresIn: "1d" }
    );

    console.log("[STAFF LOGIN] Token created with role:", tokenRole);

    res.status(200).json({
      success: true,
      message: "Login successful",
      token,
      user: {
        id: staff._id,
        firstName: staff.firstName,
        lastName: staff.lastName,
        name: tokenName,
        email: staff.email,
        role: tokenRole,
        department: tokenDept,
        hospitalId: hospitalId,
        hospitalName: hospitalName
      }
    });

  } catch (err) {
    console.error("LOGIN ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// CLOCK IN
// ----------------------
export const clockIn = async (req, res) => {
  try {
    const { shiftId, location } = req.body;
      const staffId = req.user?._id; // From JWT middleware

    console.log('\n[CLOCK IN] REQUEST RECEIVED');
    console.log('  Body:', JSON.stringify(req.body));
      console.log('  Staff ID:', staffId);

      if (!staffId) {
        console.log('  ❌ VALIDATION ERROR: No staff ID');
        return res.status(401).json({ success: false, message: "Unauthorized - no staff ID" });
    }

    if (!shiftId) {
      console.log('  ❌ VALIDATION ERROR: No shift ID provided');
      return res.status(400).json({ success: false, message: "shiftId is required" });
    }

    console.log('  🔍 Looking up Shift document by ID...');
    // Fetch shift details from Shift model
    const shift = await Shift.findById(shiftId);
    if (!shift) {
      console.log('  ❌ LOOKUP ERROR: Shift not found with ID:', shiftId);
      return res.status(404).json({ success: false, message: "Shift not found" });
    }

    console.log('  ✓ Shift found:', shift.shiftName);

    // Validate shift timing - cannot clock in before 5 mins before start or after end
    const now = new Date();
    if (shift.shiftStart) {
      const [hours, minutes] = shift.shiftStart.split(":").map(Number);
      const scheduledStartDate = new Date();
      scheduledStartDate.setHours(hours, minutes, 0, 0);
      const fiveMinsBeforeStart = new Date(scheduledStartDate.getTime() - 5 * 60000);
      
      if (now < fiveMinsBeforeStart) {
        const minsUntil = Math.ceil((fiveMinsBeforeStart - now) / 60000);
        console.log(`  ❌ TIMING ERROR: Shift hasn't started yet. Clock-in available in ${minsUntil} minutes`);
        return res.status(409).json({ 
          success: false, 
          message: `Shift hasn't started yet. Clock-in opens ${minsUntil} minutes before shift start` 
        });
      }
    }

    if (shift.shiftEnd) {
      const [hours, minutes] = shift.shiftEnd.split(":").map(Number);
      const scheduledEndDate = new Date();
      scheduledEndDate.setHours(hours, minutes, 0, 0);
      
      if (now > scheduledEndDate) {
        console.log(`  ❌ TIMING ERROR: Shift has already ended at ${shift.shiftEnd}`);
        return res.status(409).json({ 
          success: false, 
          message: `Shift has ended at ${shift.shiftEnd}. You cannot clock in to this shift.` 
        });
      }
    }

    console.log('  ✓ Shift timing validated');

    // Check if already clocked in today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    console.log('  🔍 Checking for existing clock record today...');
      let existingClock = await NurseClockRecord.findOne({
          staffId: staffId,
        date: { $gte: today, $lt: tomorrow },
        status: { $in: ["CLOCKED_IN", "ON_BREAK"] }
      }).populate('shiftId');

    // ============================================
    // AUTO CLOCK OUT - Check if shift has expired
    // ============================================
    if (existingClock && existingClock.shiftId && existingClock.shiftId.shiftEnd) {
      const [hours, minutes] = existingClock.shiftId.shiftEnd.split(':').map(Number);
      const shiftEndDate = new Date(existingClock.date);
      shiftEndDate.setHours(hours, minutes, 0, 0);

      if (now > shiftEndDate) {
        console.log(`  ⏰ SHIFT EXPIRED! Auto-clocking out staff ${staffId}`);
        console.log(`    Shift ended at: ${shiftEndDate}`);
        console.log(`    Current time: ${now}`);

        // Auto clock-out
        existingClock.clockOutTime = now;
        existingClock.status = 'CLOCKED_OUT';

        // Add flags
        if (!existingClock.flags) existingClock.flags = [];
        if (!existingClock.flags.includes('AUTO_CLOCKOUT')) {
          existingClock.flags.push('AUTO_CLOCKOUT');
        }
        if (!existingClock.flags.includes('DIDNT_CLOCKOUT')) {
          existingClock.flags.push('DIDNT_CLOCKOUT');
        }

        // Calculate duration
        if (existingClock.clockInTime) {
          existingClock.totalShiftDuration = Math.round((now - existingClock.clockInTime) / 1000);
        }

        await existingClock.save();
        console.log(`  ✓ Auto clock-out completed. Flags: ${existingClock.flags.join(', ')}`);
        existingClock = null; // Clear so we can proceed with new clock-in
      }
    }

    if (existingClock) {
      console.log('  ❌ DUPLICATE ERROR: Already clocked in today');
      return res.status(409).json({ 
        success: false,
        message: "Already clocked in today",
        clockRecord: existingClock
      });
    }

    // Check if staff has already completed a clock-in/clock-out for this shift today
    console.log('  🔍 Checking if staff has already covered this shift...');
      const alreadyCovered = await NurseClockRecord.findOne({
        staffId: staffId,
      shiftId,
      date: { $gte: today, $lt: tomorrow },
      status: "CLOCKED_OUT",
      clockOutTime: { $ne: null }
    });

    if (alreadyCovered) {
      console.log('  ❌ DUPLICATE SHIFT ERROR: Already clocked in and out for this shift');
      return res.status(409).json({ 
        success: false,
        message: `You have already completed a shift for "${shift.shiftName}". You cannot clock in to the same shift twice.`
      });
    }

    console.log('  ✓ No existing clock record. Creating new one...');

    // Create clock record with hospitalId for multi-tenant isolation
      const clockRecord = new NurseClockRecord({
          staffId: staffId,
        hospitalId: req.hospitalId,  // From auth middleware
        shiftId,
        date: new Date(),
        clockInTime: new Date(),
        location: location || "ED",
        status: "CLOCKED_IN",
        flags: []
      });

    // Check for flags based on shift timing
    if (shift.shiftStart) {
      const [hours, minutes] = shift.shiftStart.split(":").map(Number);
      const scheduledStartDate = new Date();
      scheduledStartDate.setHours(hours, minutes, 0, 0);
      const fiveMinsBeforeStart = new Date(scheduledStartDate.getTime() - 5 * 60000);
      const tenMinsAfterStart = new Date(scheduledStartDate.getTime() + 10 * 60000);

      // 1. EARLY_CLOCKIN: if clocking in more than 5 mins before shift start
      if (clockRecord.clockInTime < fiveMinsBeforeStart) {
        clockRecord.flags.push("EARLY_CLOCKIN");
        console.log('  ⚠ Flag added: EARLY_CLOCKIN');
      }
      
      // 2. LATE_CLOCKIN: if clocking in more than 10 mins after shift start time
      if (clockRecord.clockInTime > tenMinsAfterStart) {
        clockRecord.flags.push("LATE_CLOCKIN");
        console.log('  ⚠ Flag added: LATE_CLOCKIN');
      }
    }

    console.log('  💾 Saving clock record...');
    await clockRecord.save();

    console.log(`  ✓ CLOCK IN SUCCESSFUL!`);
    console.log(`    Record ID: ${clockRecord._id}`);
    console.log(`    Time: ${clockRecord.clockInTime}`);
    console.log(`    Status: ${clockRecord.status}\n`);

    res.status(201).json({
      success: true,
      message: "Clocked in successfully",
      record: {
        _id: clockRecord._id,
        clockInTime: clockRecord.clockInTime,
        shiftStart: shift.shiftStart,
        shiftEnd: shift.shiftEnd,
        location: clockRecord.location,
        status: clockRecord.status,
        flags: clockRecord.flags
      }
    });

  } catch (err) {
    console.error("\n[CLOCK IN ERROR] Exception caught:");
    console.error("  Error Type:", err.constructor.name);
    console.error("  Error Message:", err.message);
    console.error("  Stack:", err.stack);
    
    res.status(500).json({ 
      success: false,
      message: err.message 
    });
  }
};

// ----------------------
// CLOCK OUT
// ----------------------
export const clockOut = async (req, res) => {
  try {
    const staffId = req.user?._id;
    const { notes } = req.body;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized - no staff ID" });
    }

    console.log(`[CLOCK OUT] Staff: ${staffId} - checking for active clock record to close out`);

    // Find open clock record for today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    const clockRecord = await NurseClockRecord.findOne({
      staffId: staffId,
      date: { $gte: today, $lt: tomorrow },
      status: { $in: ["CLOCKED_IN", "ON_BREAK"] }
    }).sort({ updatedAt: -1 });

    if (!clockRecord) {
      return res.status(404).json({ message: "No active clock record found" });
    }

    // Record clock out
    clockRecord.clockOutTime = new Date();
    clockRecord.status = "CLOCKED_OUT";
    if (notes) clockRecord.notes = notes;

    // Check for flags: clock-out before shift end time
    if (clockRecord.shiftId) {
      const shift = await Shift.findById(clockRecord.shiftId);
      if (shift && shift.shiftEnd) {
        // Parse shiftEnd (e.g., "16:00") into a Date on the same day as shift date
        const [hours, minutes] = shift.shiftEnd.split(":").map(Number);
        const scheduledEndDate = new Date(clockRecord.date);
        scheduledEndDate.setHours(hours, minutes, 0, 0);

        // If clocking out before scheduled end time, add EARLY_CLOCKOUT flag
        if (clockRecord.clockOutTime < scheduledEndDate) {
          clockRecord.flags = clockRecord.flags || [];
          if (!clockRecord.flags.includes("EARLY_CLOCKOUT")) {
            clockRecord.flags.push("EARLY_CLOCKOUT");
            console.log(`  ⚠ Flag added: EARLY_CLOCKOUT - clocked out before shift end time`);
          }
        }
      }
    }

    // Save and respond
    await clockRecord.save();

    console.log(`[CLOCK] Staff ${staffId} clocked out at ${clockRecord.clockOutTime}, Flags: ${clockRecord.flags.join(', ')}`);

    res.status(200).json({
      success: true,
      message: "Clocked out successfully",
      summary: {
        clockInTime: clockRecord.clockInTime,
        clockOutTime: clockRecord.clockOutTime,
        totalShiftDuration: clockRecord.totalShiftDuration,
        totalBreakTime: clockRecord.totalBreakTime,
        totalWorkTime: clockRecord.totalWorkTime,
        flags: clockRecord.flags
      }
    });

  } catch (err) {
    console.error("CLOCK OUT ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// BREAK START
// ----------------------
export const breakStart = async (req, res) => {
  try {
    const staffId = req.user?._id;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }
    console.log(`[BREAK START] Staff: ${staffId} - attempting to start break`);

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    const clockRecord = await NurseClockRecord.findOne({
      staffId: staffId,
      date: { $gte: today, $lt: tomorrow },
      status: "CLOCKED_IN"
    }).sort({ updatedAt: -1 });

    console.log(`[BREAK START] Found record: ${clockRecord ? clockRecord.status + ' (updated ' + clockRecord.updatedAt + ') @ ' + clockRecord._id : 'none'}`);

    if (!clockRecord) {
      return res.status(404).json({ message: "No active clock record found" });
    }

    clockRecord.breakStartTime = new Date();
    clockRecord.status = "ON_BREAK";
    await clockRecord.save();

    console.log(`[BREAK START] ✓ Status changed to ON_BREAK, breakStartTime: ${clockRecord.breakStartTime}`);

    res.status(200).json({
      success: true,
      message: "Break started",
      record: {
        _id: clockRecord._id,
        status: clockRecord.status,
        breakStartTime: clockRecord.breakStartTime,
        clockInTime: clockRecord.clockInTime,
        clockOutTime: clockRecord.clockOutTime
      }
    });

  } catch (err) {
    console.error("BREAK START ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// BREAK END
// ----------------------
export const breakEnd = async (req, res) => {
  try {
    const staffId = req.user?._id;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }
    console.log(`[BREAK END] Staff: ${staffId} - attempting to end break`);

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    const clockRecord = await NurseClockRecord.findOne({
      staffId: staffId,
      date: { $gte: today, $lt: tomorrow },
      status: "ON_BREAK"
    }).sort({ updatedAt: -1 });

    console.log(`[BREAK END] Found record: ${clockRecord ? clockRecord.status + ' (updated ' + clockRecord.updatedAt + ') @ ' + clockRecord._id : 'none'}`);

    if (!clockRecord) {
      return res.status(404).json({ message: "No active break found" });
    }

    clockRecord.breakEndTime = new Date();
    clockRecord.status = "CLOCKED_IN";
    await clockRecord.save();

    const breakDuration = clockRecord.breakEndTime - clockRecord.breakStartTime;

    console.log(`[BREAK END] ✓ Status changed back to CLOCKED_IN, breakDuration: ${breakDuration / 1000 / 60} minutes`);

    res.status(200).json({
      success: true,
      message: "Break ended",
      record: {
        _id: clockRecord._id,
        status: clockRecord.status,
        breakStartTime: clockRecord.breakStartTime,
        breakEndTime: clockRecord.breakEndTime,
        clockInTime: clockRecord.clockInTime,
        clockOutTime: clockRecord.clockOutTime
      },
      breakDuration: breakDuration / 1000 / 60 // minutes
    });

  } catch (err) {
    console.error("BREAK END ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// GET CLOCK STATUS
// ----------------------
export const getClockStatus = async (req, res) => {
  try {
    const staffId = req.user?._id;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }
    console.log(`[GET CLOCK STATUS] Staff: ${staffId} - checking today's clock status`);

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    const clockRecord = await NurseClockRecord.findOne({
      staffId: staffId,
      date: { $gte: today, $lt: tomorrow }
    }).sort({ clockInTime: -1 });

    console.log(`[GET CLOCK STATUS] Result: ${clockRecord ? clockRecord.status + ' (id: ' + clockRecord._id + ')' : 'NOT_CLOCKED_IN'}`);
    try {
      console.log('[GET CLOCK STATUS] Record dump:', JSON.stringify(clockRecord ? (clockRecord.toObject ? clockRecord.toObject() : clockRecord) : null, null, 2));
    } catch (e) {
      console.log('[GET CLOCK STATUS] Record dump failed:', e && e.message);
    }

    if (!clockRecord) {
      return res.status(200).json({
        success: true,
        status: "NOT_CLOCKED_IN",
        message: "Not clocked in today"
      });
    }

    const now = new Date();
    const elapsedTime = now - clockRecord.clockInTime;

    res.status(200).json({
      success: true,
      clockRecord: {
        _id: clockRecord._id,
        status: clockRecord.status,
        clockInTime: clockRecord.clockInTime,
        clockOutTime: clockRecord.clockOutTime,
        breakStartTime: clockRecord.breakStartTime,
        breakEndTime: clockRecord.breakEndTime,
        elapsedTime: elapsedTime / 1000 / 60, // minutes
        location: clockRecord.location
      }
    });

  } catch (err) {
    console.error("GET CLOCK STATUS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// GET TODAY'S RECORD
// ----------------------
export const getTodayRecord = async (req, res) => {
  try {
    const staffId = req.user?._id;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }
    console.log(`\n[GET TODAY RECORD] Staff: ${staffId} - fetching today's record`);

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    console.log(`[GET TODAY RECORD] Query: staffId=${staffId}, date range ${today.toISOString()} to ${tomorrow.toISOString()}`);

    const record = await NurseClockRecord.findOne({
      staffId: staffId,
      date: { $gte: today, $lt: tomorrow }
    }).sort({ clockInTime: -1 }).populate('shiftId');

    console.log(`[GET TODAY RECORD] Found: ${record ? record.status + ' (id: ' + record._id + ') at ' + record.clockInTime : 'none'}`);
    
    // ============================================
    // AUTO CLOCK OUT - Check if shift has expired
    // ============================================
    if (record && (record.status === 'CLOCKED_IN' || record.status === 'ON_BREAK') && record.clockOutTime === null) {
      // Check if shift has ended
      if (record.shiftId && record.shiftId.shiftEnd) {
        const [hours, minutes] = record.shiftId.shiftEnd.split(':').map(Number);
        const shiftEndDate = new Date(record.date);
        shiftEndDate.setHours(hours, minutes, 0, 0);
        const now = new Date();

        if (now > shiftEndDate) {
          console.log(`[GET TODAY RECORD] ⏰ SHIFT EXPIRED! Auto-clocking out staff ${staffId}`);
          console.log(`  Shift ended at: ${shiftEndDate}`);
          console.log(`  Current time: ${now}`);

          // Auto clock-out
          record.clockOutTime = now;
          record.status = 'CLOCKED_OUT';

          // Add flags
          if (!record.flags) record.flags = [];
          if (!record.flags.includes('AUTO_CLOCKOUT')) {
            record.flags.push('AUTO_CLOCKOUT');
          }
          if (!record.flags.includes('DIDNT_CLOCKOUT')) {
            record.flags.push('DIDNT_CLOCKOUT');
          }

          // Calculate duration
          if (record.clockInTime) {
            record.totalShiftDuration = Math.round((now - record.clockInTime) / 1000);
          }

          await record.save();
          console.log(`[GET TODAY RECORD] ✓ Auto clock-out completed. Flags: ${record.flags.join(', ')}`);
        }
      }
    }

    try {
      if (record) {
        const recordObj = record.toObject ? record.toObject() : record;
        console.log('[GET TODAY RECORD] ✓ RETURNING RECORD:', JSON.stringify({
          _id: recordObj._id,
          status: recordObj.status,
          clockInTime: recordObj.clockInTime,
          clockOutTime: recordObj.clockOutTime,
          breakStartTime: recordObj.breakStartTime,
          breakEndTime: recordObj.breakEndTime,
          shiftId: recordObj.shiftId?._id || recordObj.shiftId,
          shiftName: recordObj.shiftId?.shiftName,
          flags: recordObj.flags,
          createdAt: recordObj.createdAt,
          updatedAt: recordObj.updatedAt
        }, null, 2));
      } else {
        console.log('[GET TODAY RECORD] No record found for today');
      }
    } catch (e) {
      console.log('[GET TODAY RECORD] Record dump failed:', e && e.message);
    }

    res.status(200).json({
      success: true,
      record: record || null
    });

  } catch (err) {
    console.error("GET TODAY RECORD ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// DEBUG: LIST ALL TODAY'S RECORDS
// ----------------------
export const debugTodayRecords = async (req, res) => {
  try {
    const staffId = req.user?._id;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    console.log(`\n[DEBUG RECORDS] Staff: ${staffId} - listing ALL records for ${today.toDateString()}`);

    const allRecords = await NurseClockRecord.find({
      staffId: staffId,
      date: { $gte: today, $lt: tomorrow }
    }).sort({ clockInTime: -1 });

    console.log(`[DEBUG RECORDS] Found ${allRecords.length} total records`);
    allRecords.forEach((r, i) => {
      console.log(`  [${i}] Status: ${r.status} | ClockIn: ${r.clockInTime} | ClockOut: ${r.clockOutTime} | Created: ${r.createdAt} | Updated: ${r.updatedAt}`);
    });

    const latestRecord = allRecords[0];
    console.log(`[DEBUG RECORDS] Latest (by clockInTime): ${latestRecord ? latestRecord.status + ' at ' + latestRecord.clockInTime : 'none'}`);

    res.status(200).json({
      success: true,
      debug: {
        staffId,
        totalRecords: allRecords.length,
        dateRange: {
          start: today.toISOString(),
          end: tomorrow.toISOString()
        },
        allRecords: allRecords.map(r => ({
          _id: r._id,
          status: r.status,
          clockInTime: r.clockInTime,
          clockOutTime: r.clockOutTime,
          breakStartTime: r.breakStartTime,
          breakEndTime: r.breakEndTime,
          createdAt: r.createdAt,
          updatedAt: r.updatedAt
        })),
        latestByClockInTime: latestRecord ? {
          _id: latestRecord._id,
          status: latestRecord.status,
          clockInTime: latestRecord.clockInTime,
          clockOutTime: latestRecord.clockOutTime
        } : null
      }
    });

  } catch (err) {
    console.error("DEBUG RECORDS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// GET CLOCK HISTORY
// ----------------------
export const getClockHistory = async (req, res) => {
  try {
    const staffId = req.user?._id;
    const { startDate, endDate } = req.query;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const query = { staffId: staffId };

    if (startDate || endDate) {
      query.date = {};
      if (startDate) query.date.$gte = new Date(startDate);
      if (endDate) query.date.$lte = new Date(endDate);
    }

    const records = await NurseClockRecord.find(query)
      .sort({ date: -1 })
      .limit(30);

    res.status(200).json({
      success: true,
      records
    });

  } catch (err) {
    console.error("GET CLOCK HISTORY ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

const computeExpectedHours = (shiftStart, shiftEnd) => {
  if (!shiftStart || !shiftEnd) return 8;

  const [startHour, startMin] = shiftStart.split(":").map(Number);
  const [endHour, endMin] = shiftEnd.split(":").map(Number);
  if (Number.isNaN(startHour) || Number.isNaN(endHour)) return 8;

  const start = new Date();
  start.setHours(startHour, startMin || 0, 0, 0);
  const end = new Date();
  end.setHours(endHour, endMin || 0, 0, 0);
  if (end <= start) {
    end.setDate(end.getDate() + 1);
  }

  return Math.max(1, Math.round((end - start) / (1000 * 60 * 60)));
};

const buildStaffProfile = (staff, schedule) => ({
  firstName: staff.firstName || staff.name || '',
  lastName: staff.lastName || staff.surname || '',
  idNumber: staff.credentials?.licenseNumber || staff.licenseNumber || staff._id?.toString(),
  employeeId: staff.credentials?.npiNumber || staff.npiNumber || staff._id?.toString(),
  role: staff.systemRole || staff.role || 'NURSE',
  department: staff.assignment?.department || staff.department || 'Nursing',
  shift: {
    expectedHours: schedule?.expectedHours || computeExpectedHours(schedule?.shiftStart, schedule?.shiftEnd),
    signIn: schedule?.shiftStart || '',
    signOut: schedule?.shiftEnd || ''
  },
  createdAt: staff.createdAt,
  updatedAt: staff.updatedAt
});

// ----------------------
// GET AVAILABLE SHIFTS
// ----------------------
export const getAvailableShifts = async (req, res) => {
  try {
    const staffId = req.user?._id;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    // Get today and next 7 days
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const sevenDaysLater = new Date(today.getTime() + 7 * 24 * 60 * 60 * 1000);

    const shifts = await NurseSchedule.find({
      nurseId: staffId.toString(),
      date: { $gte: today, $lt: sevenDaysLater },
      status: { $in: ["ON_DUTY", "ON_CALL"] }
    }).sort({ date: 1 });

    const shiftsWithDetails = shifts.map(shift => ({
      _id: shift._id,
      date: shift.date,
      status: shift.status,
      shiftStart: shift.shiftStart,
      shiftEnd: shift.shiftEnd,
      expectedHours: shift.expectedHours || computeExpectedHours(shift.shiftStart, shift.shiftEnd),
      breakStart: shift.breakStart,
      breakEnd: shift.breakEnd,
      location: "ED",
      maxBeds: shift.maxBeds,
      assignedBeds: shift.assignedBeds || [],
      currentBedCount: shift.currentBedCount || 0,
      licenseType: shift.licenseType,
      specializations: shift.specializations || []
    }));

    res.status(200).json({
      success: true,
      shifts: shiftsWithDetails,
      message: shiftsWithDetails.length === 0 ? "No scheduled shifts for next 7 days" : `Found ${shiftsWithDetails.length} scheduled shift(s)`
    });

  } catch (err) {
    console.error("GET AVAILABLE SHIFTS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

export const getStaffProfile = async (req, res) => {
  try {
    const staffId = req.user?._id;
    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const staff = await ClinicalStaff.findById(staffId).select('-password').lean();
    if (!staff) {
      return res.status(404).json({ message: "Staff profile not found" });
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    const schedule = await NurseSchedule.findOne({
      nurseId: staffId.toString(),
      date: { $gte: today, $lt: tomorrow }
    }).lean();

    res.status(200).json(buildStaffProfile(staff, schedule));
  } catch (err) {
    console.error("GET STAFF PROFILE ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

export const updateStaffProfile = async (req, res) => {
  try {
    const staffId = req.user?._id;
    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const { firstName, lastName, idNumber, employeeId } = req.body;
    const update = {};

    if (firstName !== undefined) update.firstName = firstName;
    if (lastName !== undefined) update.lastName = lastName;
    if (idNumber !== undefined) {
      update.credentials = {
        ...(req.user.credentials || {}),
        licenseNumber: idNumber
      };
    }
    if (employeeId !== undefined) {
      update.credentials = {
        ...(update.credentials || req.user.credentials || {}),
        npiNumber: employeeId
      };
    }

    const updatedStaff = await ClinicalStaff.findByIdAndUpdate(
      staffId,
      { $set: update },
      { new: true, runValidators: true, context: 'query' }
    ).select('-password').lean();

    if (!updatedStaff) {
      return res.status(404).json({ message: "Staff profile not found" });
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    const schedule = await NurseSchedule.findOne({
      nurseId: staffId.toString(),
      date: { $gte: today, $lt: tomorrow }
    }).lean();

    res.status(200).json(buildStaffProfile(updatedStaff, schedule));
  } catch (err) {
    console.error("UPDATE STAFF PROFILE ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

export const updateStaffShift = async (req, res) => {
  try {
    const staffId = req.user?._id;
    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const { expectedHours, signIn, signOut } = req.body;
    if (!signIn) {
      return res.status(400).json({ message: "signIn is required" });
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    let schedule = await NurseSchedule.findOne({
      nurseId: staffId.toString(),
      date: { $gte: today, $lt: tomorrow }
    });

    const scheduleData = {
      nurseId: staffId.toString(),
      name: req.user.firstName || req.user.name || '',
      surname: req.user.lastName || req.user.surname || '',
      email: req.user.email || '',
      licenseNumber: req.user.credentials?.licenseNumber || req.user.npiNumber || '',
      date: today,
      status: "ON_DUTY",
      shiftStart: signIn,
      shiftEnd: signOut || '',
      expectedHours: expectedHours ? parseInt(expectedHours, 10) : computeExpectedHours(signIn, signOut)
    };

    if (!schedule) {
      schedule = new NurseSchedule(scheduleData);
    } else {
      Object.assign(schedule, scheduleData);
    }

    await schedule.save();

    res.status(200).json({
      expectedHours: schedule.expectedHours,
      signIn: schedule.shiftStart,
      signOut: schedule.shiftEnd
    });
  } catch (err) {
    console.error("UPDATE STAFF SHIFT ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// REQUEST BREAK
// ----------------------
export const requestBreak = async (req, res) => {
  try {
    const staffId = req.user?._id;
    const { reason } = req.body;

    if (!staffId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    // Find active clock record
    const clockRecord = await NurseClockRecord.findOne({
      staffId: staffId,
      date: { $gte: today, $lt: tomorrow },
      status: "CLOCKED_IN"
    });

    if (!clockRecord) {
      return res.status(404).json({ message: "No active shift found. Clock in first." });
    }

    // Check if break already pending/approved
    if (clockRecord.breakApprovalStatus === "PENDING" || clockRecord.breakApprovalStatus === "APPROVED") {
      return res.status(409).json({ 
        message: "Break request already exists",
        status: clockRecord.breakApprovalStatus
      });
    }

    // Submit break request
    clockRecord.breakApprovalStatus = "PENDING";
    clockRecord.breakReason = reason || "Break request";
    await clockRecord.save();

    res.status(200).json({
      success: true,
      message: "Break request submitted. Awaiting manager approval.",
      clockRecord: {
        _id: clockRecord._id,
        breakApprovalStatus: clockRecord.breakApprovalStatus,
        breakReason: clockRecord.breakReason
      }
    });

  } catch (err) {
    console.error("REQUEST BREAK ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// APPROVE/REJECT LATE ARRIVAL (Manager)
// ----------------------
export const approveLateArrival = async (req, res) => {
  try {
    const managerId = req.user?._id;
    const { recordId } = req.params;
    const { approved, reason } = req.body;

    if (!managerId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    if (approved === undefined) {
      return res.status(400).json({ message: "Approval status required (approved: true/false)" });
    }

    const clockRecord = await NurseClockRecord.findById(recordId);
    if (!clockRecord) {
      return res.status(404).json({ message: "Clock record not found" });
    }

    // Check if late arrival approval is needed
    if (clockRecord.lateArrivalApprovalStatus !== "PENDING") {
      return res.status(400).json({ 
        message: "Late arrival approval not required or already processed",
        currentStatus: clockRecord.lateArrivalApprovalStatus
      });
    }

    // Update approval status
    if (approved) {
      clockRecord.lateArrivalApprovalStatus = "APPROVED";
    } else {
      clockRecord.lateArrivalApprovalStatus = "REJECTED";
      if (reason) {
        clockRecord.lateArrivalRejectionReason = reason;
      }
    }

    clockRecord.lateArrivalApprovedBy = managerId;
    clockRecord.lateArrivalApprovalTime = new Date();
    await clockRecord.save();

    console.log(`[APPROVAL] Late arrival ${approved ? "APPROVED" : "REJECTED"} for staff ${clockRecord.staffId}`);

    res.status(200).json({
      success: true,
      message: `Late arrival ${approved ? "approved" : "rejected"}`,
      clockRecord: {
        _id: clockRecord._id,
        staffId: clockRecord.staffId,
        lateArrivalMinutes: clockRecord.lateArrivalMinutes,
        lateArrivalApprovalStatus: clockRecord.lateArrivalApprovalStatus,
        lateArrivalApprovedBy: clockRecord.lateArrivalApprovedBy,
        lateArrivalApprovalTime: clockRecord.lateArrivalApprovalTime
      }
    });

  } catch (err) {
    console.error("APPROVE LATE ARRIVAL ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// APPROVE/REJECT BREAK REQUEST (Manager)
// ----------------------
export const approveBreakRequest = async (req, res) => {
  try {
    const managerId = req.user?._id;
    const { recordId } = req.params;
    const { approved, reason } = req.body;

    if (!managerId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    if (approved === undefined) {
      return res.status(400).json({ message: "Approval status required (approved: true/false)" });
    }

    const clockRecord = await NurseClockRecord.findById(recordId);
    if (!clockRecord) {
      return res.status(404).json({ message: "Clock record not found" });
    }

    // Check if break approval is pending
    if (clockRecord.breakApprovalStatus !== "PENDING") {
      return res.status(400).json({ 
        message: "Break request not pending or already processed",
        currentStatus: clockRecord.breakApprovalStatus
      });
    }

    // Update approval status
    if (approved) {
      clockRecord.breakApprovalStatus = "APPROVED";
      // Automatically start the break
      clockRecord.breakStartTime = new Date();
      clockRecord.status = "ON_BREAK";
    } else {
      clockRecord.breakApprovalStatus = "REJECTED";
      if (reason) {
        clockRecord.breakRejectionReason = reason;
      }
      // Status remains CLOCKED_IN if rejected
      clockRecord.status = "CLOCKED_IN";
    }

    clockRecord.breakApprovedBy = managerId;
    clockRecord.breakApprovalTime = new Date();
    await clockRecord.save();

    console.log(`[APPROVAL] Break request ${approved ? "APPROVED" : "REJECTED"} for staff ${clockRecord.staffId}`);

    res.status(200).json({
      success: true,
      message: `Break request ${approved ? "approved" : "rejected"}`,
      clockRecord: {
        _id: clockRecord._id,
        staffId: clockRecord.staffId,
        breakApprovalStatus: clockRecord.breakApprovalStatus,
        breakApprovedBy: clockRecord.breakApprovedBy,
        breakApprovalTime: clockRecord.breakApprovalTime,
        breakStartTime: clockRecord.breakStartTime,
        status: clockRecord.status
      }
    });

  } catch (err) {
    console.error("APPROVE BREAK REQUEST ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Get attendance history for the authenticated nurse
 * Returns all clock records with flags and shift context
 */
export const  getAttendanceHistory = async (req, res) => {
  try {
    const userId = req.user._id;
    
    // Get all clock records for this nurse sorted by date descending
    const clockRecords = await NurseClockRecord.find({ staffId: userId })
      .sort({ clockInTime: -1 })
      .populate('shiftId')
      .lean()
      .exec();

    // Get all assigned shifts for this nurse
    const assignedShifts = await Shift.find({
      'assignedStaff.staffID': userId
    }).lean().exec();

    const now = new Date();
    const enrichedRecords = [];

    // Process actual clock records
    if (clockRecords && clockRecords.length > 0) {
      clockRecords.forEach((record) => {
        const shiftDetails = record.shiftId || {};
        
        // Check if shift has ended and user didn't clock out
        let flags = record.flags || [];
        if (record.clockInTime && !record.clockOutTime && shiftDetails.shiftEnd) {
          const [hours, minutes] = shiftDetails.shiftEnd.split(':').map(Number);
          const shiftEndDate = new Date(new Date(record.clockInTime).toDateString());
          shiftEndDate.setHours(hours, minutes, 0, 0);
          
          // If shift time has passed and no clock out, flag as DIDNT_CLOCKOUT
          if (now > shiftEndDate && !flags.includes('DIDNT_CLOCKOUT')) {
            flags = [...flags, 'DIDNT_CLOCKOUT'];
          }
        }

        // Calculate work duration
        let workDuration = null;
        if (record.clockInTime && record.clockOutTime) {
          workDuration = Math.round((new Date(record.clockOutTime) - new Date(record.clockInTime)) / 60000);
        }

        enrichedRecords.push({
          _id: record._id,
          date: record.clockInTime ? new Date(record.clockInTime).toLocaleDateString() : 'N/A',
          clockInTime: record.clockInTime ? new Date(record.clockInTime).toLocaleTimeString() : 'N/A',
          clockOutTime: record.clockOutTime ? new Date(record.clockOutTime).toLocaleTimeString() : (record.status === 'CLOCKED_IN' ? 'Pending' : 'Not clocked out'),
          status: record.status,
          flags: flags,
          workDuration: workDuration ? `${workDuration} min` : (record.clockOutTime ? 'Complete' : 'In progress'),
          shiftId: record.shiftId,
          shiftName: shiftDetails.shiftName || 'Unknown Shift',
          department: shiftDetails.department || 'N/A',
          shiftTime: shiftDetails.shiftStart && shiftDetails.shiftEnd 
            ? `${shiftDetails.shiftStart}–${shiftDetails.shiftEnd}` 
            : 'N/A',
          breakApprovalStatus: record.breakApprovalStatus,
          notes: record.notes || null
        });
      });
    }

    // Process assigned shifts to find absences
    if (assignedShifts && assignedShifts.length > 0) {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

      assignedShifts.forEach((shift) => {
        // Check if shift is for today or in the past
        const shiftDate = new Date(shift.date);
        if (shiftDate < tomorrow) {
          // Check if there's a clock record for this shift
          const hasClockRecord = clockRecords.some(r => r.shiftId && r.shiftId._id && r.shiftId._id.toString() === shift._id.toString());
          
          if (!hasClockRecord) {
            // Check if shift time has ended
            const [hours, minutes] = shift.shiftStart.split(':').map(Number);
            const shiftStartDate = new Date(shift.date);
            shiftStartDate.setHours(hours, minutes, 0, 0);
            
            const [endHours, endMinutes] = shift.shiftEnd.split(':').map(Number);
            const shiftEndDate = new Date(shift.date);
            shiftEndDate.setHours(endHours, endMinutes, 0, 0);

            // Only flag as absent if shift time has passed
            if (now > shiftEndDate) {
              enrichedRecords.push({
                date: new Date(shift.date).toLocaleDateString(),
                clockInTime: 'N/A',
                clockOutTime: 'N/A',
                status: 'ABSENT',
                flags: ['ABSENT'],
                workDuration: '0 min',
                shiftId: shift._id,
                shiftName: shift.shiftName || 'Unknown Shift',
                department: shift.department || 'N/A',
                shiftTime: `${shift.shiftStart}–${shift.shiftEnd}`,
                notes: 'No clock record found for assigned shift'
              });
            }
          }
        }
      });
    }

    // Sort by date descending
    enrichedRecords.sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return dateB - dateA;
    });

    res.status(200).json({
      success: true,
      count: enrichedRecords.length,
      records: enrichedRecords
    });

  } catch (err) {
    console.error("GET ATTENDANCE HISTORY ERROR:", err);
    res.status(500).json({ 
      success: false,
      message: err.message 
    });
  }
};

// ----------------------
// AUTO CLOCK OUT EXPIRED SHIFTS
// ----------------------
export const autoClockOutExpiredShifts = async (req, res) => {
  try {
    console.log("[AUTO CLOCK OUT] Running automatic clock-out for expired shifts...");

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

    console.log(`[AUTO CLOCK OUT] Found ${activeRecords.length} active records to check`);

    let clockedOutCount = 0;
    const results = [];

    for (const record of activeRecords) {
      try {
        if (!record.shiftId || !record.shiftId.shiftEnd) {
          console.log(`[AUTO CLOCK OUT] Skipping record ${record._id} - no shift info`);
          continue;
        }

        // Parse shift end time
        const [hours, minutes] = record.shiftId.shiftEnd.split(":").map(Number);
        const shiftEndDate = new Date(record.date);
        shiftEndDate.setHours(hours, minutes, 0, 0);

        // Check if shift has ended
        if (now > shiftEndDate) {
          console.log(`[AUTO CLOCK OUT] Shift expired for staff ${record.staffId} at ${shiftEndDate}. Auto-clocking out...`);

          // Update record
          record.clockOutTime = now;
          record.status = "CLOCKED_OUT";
          
          // Calculate durations
          if (record.clockInTime) {
            record.totalShiftDuration = Math.round((now - record.clockInTime) / 1000);
          }

          // Add AUTO_CLOCKOUT flag
          record.flags = record.flags || [];
          if (!record.flags.includes("AUTO_CLOCKOUT")) {
            record.flags.push("AUTO_CLOCKOUT");
          }

          // Add DIDNT_CLOCKOUT if not already present (and no ON_BREAK)
          if (record.status !== "ON_BREAK" && !record.flags.includes("DIDNT_CLOCKOUT")) {
            record.flags.push("DIDNT_CLOCKOUT");
          }

          await record.save();
          clockedOutCount++;

          results.push({
            staffId: record.staffId,
            shiftId: record.shiftId._id,
            shiftName: record.shiftId.shiftName,
            clockInTime: record.clockInTime,
            autoClockOutTime: record.clockOutTime,
            flags: record.flags
          });
        }
      } catch (err) {
        console.error(`[AUTO CLOCK OUT] Error processing record ${record._id}:`, err);
      }
    }

    const responseMsg = `Auto-clocked out ${clockedOutCount} staff members with expired shifts`;
    console.log(`[AUTO CLOCK OUT] ${responseMsg}`);

    res.status(200).json({
      success: true,
      message: responseMsg,
      clockedOutCount,
      results
    });

  } catch (err) {
    console.error("AUTO CLOCK OUT ERROR:", err);
    res.status(500).json({ 
      success: false,
      message: err.message 
    });
  }
};
