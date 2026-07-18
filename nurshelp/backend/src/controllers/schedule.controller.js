import { PhysicianSchedule } from "../models/physicianSchedule.model.js";
import { NurseSchedule } from "../models/nurseSchedule.model.js";
import { BedSchedule } from "../models/bedSchedule.model.js";

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/**
 * Parse date string (YYYY-MM-DD) consistently to avoid timezone issues
 * Returns a Date object set to 00:00:00 in UTC
 */
function parseDateString(dateStr) {
  if (!dateStr) return null;
  const [year, month, day] = dateStr.split('-').map(Number);
  const date = new Date(Date.UTC(year, month - 1, day, 0, 0, 0, 0));
  return date;
}

// ============================================================
// PHYSICIAN SCHEDULE MANAGEMENT
// ============================================================

/**
 * Get physicians available for a specific date
 */
export const getAvailablePhysicians = async (req, res) => {
  try {
    const { date, specialty, status } = req.query;
    
    let query = {};
    
    if (date) {
      const dateObj = parseDateString(date);
      const nextDay = new Date(dateObj.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: dateObj, $lt: nextDay };
    } else {
      // Default to today (UTC)
      const today = new Date();
      today.setUTCHours(0, 0, 0, 0);
      const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: today, $lt: tomorrow };
    }
    
    if (specialty) {
      query.specialty = specialty;
    }
    
    if (status) {
      query.status = status;
    } else {
      // Default: only ON_DUTY and ON_CALL
      query.status = { $in: ["ON_DUTY", "ON_CALL"] };
    }
    
    const physicians = await PhysicianSchedule.find(query)
      .sort({ currentPatientCount: 1 }) // Sort by least busy
      .lean();
    
    res.json({ 
      count: physicians.length,
      physicians 
    });
  } catch (err) {
    console.error("GET AVAILABLE PHYSICIANS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Get physician's schedule for specific date
 */
export const getPhysicianSchedule = async (req, res) => {
  try {
    const { physicianId, date } = req.query;
    
    if (!physicianId) {
      return res.status(400).json({ message: "physicianId required" });
    }
    
    let query = { physicianId };
    
    if (date) {
      const dateObj = parseDateString(date);
      const nextDay = new Date(dateObj.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: dateObj, $lt: nextDay };
    } else {
      const today = new Date();
      today.setUTCHours(0, 0, 0, 0);
      const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: today, $lt: tomorrow };
    }
    
    const schedule = await PhysicianSchedule.findOne(query);
    
    if (!schedule) {
      return res.status(404).json({ message: "Schedule not found" });
    }
    
    res.json(schedule);
  } catch (err) {
    console.error("GET PHYSICIAN SCHEDULE ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Create/Update physician schedule (single)
 */
export const upsertPhysicianSchedule = async (req, res) => {
  try {
    const { physicianId, name, surname, email, specialty, date, status, shiftStart, shiftEnd, breakStart, breakEnd, maxPatients, notes } = req.body;
    
    if (!physicianId || !name || !specialty || !date) {
      return res.status(400).json({ message: "physicianId, name, specialty, date required" });
    }
    
    const dateObj = parseDateString(date);
    
    // Generate availability mask (24 hours) with break times
    const availabilityMask = generateAvailabilityMask(shiftStart, shiftEnd, breakStart, breakEnd);
    
    const schedule = await PhysicianSchedule.findOneAndUpdate(
      { physicianId, date: { $gte: dateObj, $lt: new Date(dateObj.getTime() + 24 * 60 * 60 * 1000) } },
      {
        physicianId,
        name,
        surname,
        email,
        specialty,
        date: dateObj,
        status: status || "OFF",
        shiftStart,
        shiftEnd,
        breakStart,
        breakEnd,
        maxPatients: maxPatients || 20,
        availabilityMask,
        notes: notes || ""
      },
      { upsert: true, new: true }
    );
    
    res.json({ message: "Schedule saved", schedule });
  } catch (err) {
    console.error("UPSERT PHYSICIAN SCHEDULE ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Import physician schedules from CSV
 * CSV Format: physicianId,name,surname,email,specialty,date,status,shiftStart,shiftEnd,breakStart,breakEnd,maxPatients,notes
 */
export const importPhysicianSchedulesCSV = async (req, res) => {
  try {
    if (!req.body.csvData) {
      return res.status(400).json({ message: "csvData required in body" });
    }
    
    const csvData = req.body.csvData;
    const lines = csvData.split(/\r?\n/).filter(l => l.trim());
    
    if (lines.length < 2) {
      return res.status(400).json({ message: "CSV must have header and at least 1 data row" });
    }

    const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
    const results = [];
    const errors = [];
    
    for (let i = 1; i < lines.length; i++) {
      try {
        const values = lines[i].split(",").map(v => v.trim());
        
        if (values.length !== headers.length) {
          errors.push(`Row ${i + 1}: Column count mismatch`);
          continue;
        }
        
        const row = {};
        headers.forEach((h, idx) => {
          row[h] = values[idx];
        });
        
        // Validate required fields
        if (!row.physicianid || !row.name || !row.specialty || !row.date) {
          errors.push(`Row ${i + 1}: Missing required fields (physicianId, name, specialty, date)`);
          continue;
        }
        
        const dateObj = new Date(row.date);
        if (isNaN(dateObj)) {
          errors.push(`Row ${i + 1}: Invalid date format`);
          continue;
        }
        
        dateObj.setHours(0, 0, 0, 0);
        
        // Generate availability mask (with optional break times)
        const availabilityMask = generateAvailabilityMask(row.shiftstart, row.shiftend, row.breakstart, row.breakend);
        const schedule = await PhysicianSchedule.findOneAndUpdate(
          { physicianId: row.physicianid, date: { $gte: dateObj, $lt: new Date(dateObj.getTime() + 24 * 60 * 60 * 1000) } },
          {
            physicianId: row.physicianid,
            name: row.name,
            surname: row.surname || "",
            email: row.email || "",
            specialty: row.specialty,
            date: dateObj,
            status: row.status || "OFF",
            shiftStart: row.shiftstart || "",
            shiftEnd: row.shiftend || "",
            breakStart: row.breakstart || "",
            breakEnd: row.breakend || "",
            maxPatients: parseInt(row.maxpatients) || 20,
            availabilityMask,
            notes: row.notes || ""
          },
          { upsert: true, new: true }
        );
        
        results.push({ row: i + 1, physicianId: row.physicianid, status: "imported" });
      } catch (e) {
        errors.push(`Row ${i + 1}: ${e.message}`);
      }
    }
    
    res.json({ 
      message: `Imported ${results.length} physicians`,
      imported: results.length,
      errors: errors.length > 0 ? errors : undefined,
      results 
    });
  } catch (err) {
    console.error("IMPORT PHYSICIAN CSV ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ============================================================
// NURSE SCHEDULE MANAGEMENT
// ============================================================

/**
 * Get nurses available for a specific date
 */
export const getAvailableNurses = async (req, res) => {
  try {
    const { date, status, esiLevel } = req.query;
    
    let query = {};
    
    if (date) {
      const dateObj = parseDateString(date);
      const nextDay = new Date(dateObj.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: dateObj, $lt: nextDay };
    } else {
      const today = new Date();
      today.setUTCHours(0, 0, 0, 0);
      const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: today, $lt: tomorrow };
    }
    
    if (status) {
      query.status = status;
    } else {
      query.status = { $in: ["ON_DUTY", "ON_CALL"] };
    }
    
    if (esiLevel) {
      query.esiLevel = parseInt(esiLevel);
    }
    
    const nurses = await NurseSchedule.find(query)
      .sort({ currentBedCount: 1 }) // Sort by least busy
      .lean();
    
    res.json({ 
      count: nurses.length,
      nurses 
    });
  } catch (err) {
    console.error("GET AVAILABLE NURSES ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Get nurse's schedule for specific date
 */
export const getNurseSchedule = async (req, res) => {
  try {
    const { nurseId, date } = req.query;
    
    if (!nurseId) {
      return res.status(400).json({ message: "nurseId required" });
    }
    
    let query = { nurseId };
    
    if (date) {
      const dateObj = parseDateString(date);
      const nextDay = new Date(dateObj.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: dateObj, $lt: nextDay };
    } else {
      const today = new Date();
      today.setUTCHours(0, 0, 0, 0);
      const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);
      query.date = { $gte: today, $lt: tomorrow };
    }
    
    const schedule = await NurseSchedule.findOne(query);
    
    if (!schedule) {
      return res.status(404).json({ message: "Schedule not found" });
    }
    
    res.json(schedule);
  } catch (err) {
    console.error("GET NURSE SCHEDULE ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Create/Update nurse schedule (single)
 */
export const upsertNurseSchedule = async (req, res) => {
  try {
    const { nurseId, name, surname, email, licenseNumber, date, status, shiftStart, shiftEnd, breakStart, breakEnd, maxBeds, esiLevel, certifications, notes } = req.body;
    
    if (!nurseId || !name || !date) {
      return res.status(400).json({ message: "nurseId, name, date required" });
    }
    
    const dateObj = parseDateString(date);
    
    const availabilityMask = generateAvailabilityMask(shiftStart, shiftEnd, breakStart, breakEnd);
    
    const schedule = await NurseSchedule.findOneAndUpdate(
      { nurseId, date: { $gte: dateObj, $lt: new Date(dateObj.getTime() + 24 * 60 * 60 * 1000) } },
      {
        nurseId,
        name,
        surname,
        email,
        licenseNumber,
        date: dateObj,
        status: status || "OFF",
        shiftStart,
        shiftEnd,
        breakStart,
        breakEnd,
        maxBeds: maxBeds || 5,
        esiLevel: esiLevel || 3,
        certifications: certifications || [],
        availabilityMask,
        notes: notes || ""
      },
      { upsert: true, new: true }
    );
    
    res.json({ message: "Schedule saved", schedule });
  } catch (err) {
    console.error("UPSERT NURSE SCHEDULE ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Import nurse schedules from CSV
 * CSV Format: nurseId,name,surname,email,licenseNumber,date,status,shiftStart,shiftEnd,breakStart,breakEnd,maxBeds,esiLevel,certifications,notes
 */
export const importNurseSchedulesCSV = async (req, res) => {
  try {
    if (!req.body.csvData) {
      return res.status(400).json({ message: "csvData required in body" });
    }
    
    const csvData = req.body.csvData;
    const lines = csvData.split(/\r?\n/).filter(l => l.trim());
    
    if (lines.length < 2) {
      return res.status(400).json({ message: "CSV must have header and at least 1 data row" });
    }
    
    const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
    const results = [];
    const errors = [];
    
    for (let i = 1; i < lines.length; i++) {
      try {
        const values = lines[i].split(",").map(v => v.trim());
        
        if (values.length !== headers.length) {
          errors.push(`Row ${i + 1}: Column count mismatch`);
          continue;
        }
        
        const row = {};
        headers.forEach((h, idx) => {
          row[h] = values[idx];
        });
        
        // Validate required fields
        if (!row.nurseid || !row.name || !row.date) {
          errors.push(`Row ${i + 1}: Missing required fields (nurseId, name, date)`);
          continue;
        }
        
        const dateObj = new Date(row.date);
        if (isNaN(dateObj)) {
          errors.push(`Row ${i + 1}: Invalid date format`);
          continue;
        }
        
        dateObj.setHours(0, 0, 0, 0);
        
        const availabilityMask = generateAvailabilityMask(row.shiftstart, row.shiftend, row.breakstart, row.breakend);
        
        const schedule = await NurseSchedule.findOneAndUpdate(
          { nurseId: row.nurseid, date: { $gte: dateObj, $lt: new Date(dateObj.getTime() + 24 * 60 * 60 * 1000) } },
          {
            nurseId: row.nurseid,
            name: row.name,
            surname: row.surname || "",
            email: row.email || "",
            licenseNumber: row.licensenumber || "",
            date: dateObj,
            status: row.status || "OFF",
            shiftStart: row.shiftstart || "",
            shiftEnd: row.shiftend || "",
            breakStart: row.breakstart || "",
            breakEnd: row.breakend || "",
            maxBeds: parseInt(row.maxbeds) || 5,
            esiLevel: parseInt(row.esilevel) || 3,
            certifications: row.certifications ? row.certifications.split("|") : [],
            availabilityMask,
            notes: row.notes || ""
          },
          { upsert: true, new: true }
        );
        
        results.push({ row: i + 1, nurseId: row.nurseid, status: "imported" });
      } catch (e) {
        errors.push(`Row ${i + 1}: ${e.message}`);
      }
    }
    
    res.json({ 
      message: `Imported ${results.length} nurses`,
      imported: results.length,
      errors: errors.length > 0 ? errors : undefined,
      results 
    });
  } catch (err) {
    console.error("IMPORT NURSE CSV ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ============================================================
// BED SCHEDULE MANAGEMENT
// ============================================================

/**
 * Get available beds
 */
export const getAvailableBeds = async (req, res) => {
  try {
    const { department, bedType, acuityLevel } = req.query;
    
    let query = { status: "AVAILABLE" };
    
    if (department) {
      query.department = department;
    }
    
    if (bedType) {
      query.bedType = bedType;
    }
    
    if (acuityLevel) {
      query.acuityLevel = acuityLevel;
    }
    
    const beds = await BedSchedule.find(query).lean();
    
    res.json({ 
      count: beds.length,
      beds 
    });
  } catch (err) {
    console.error("GET AVAILABLE BEDS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Get all beds (including occupied)
 */
export const getAllBeds = async (req, res) => {
  try {
    const { department } = req.query;
    
    let query = {};
    if (department) {
      query.department = department;
    }
    
    const beds = await BedSchedule.find(query).lean();
    
    res.json({ 
      count: beds.length,
      beds 
    });
  } catch (err) {
    console.error("GET ALL BEDS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Assign bed to nurse and encounter
 */
export const assignBed = async (req, res) => {
  try {
    const { bedId, encounterId, nurseId, acuityLevel } = req.body;
    
    if (!bedId) {
      return res.status(400).json({ message: "bedId required" });
    }
    
    const bed = await BedSchedule.findOneAndUpdate(
      { bedId },
      {
        status: "OCCUPIED",
        currentEncounterId: encounterId || null,
        assignedNurseId: nurseId || null,
        acuityLevel: acuityLevel || "MODERATE",
        occupancyStartTime: new Date()
      },
      { new: true }
    );
    
    if (!bed) {
      return res.status(404).json({ message: "Bed not found" });
    }
    
    res.json({ message: "Bed assigned", bed });
  } catch (err) {
    console.error("ASSIGN BED ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Release bed (mark as available for cleaning)
 */
export const releaseBed = async (req, res) => {
  try {
    const { bedId } = req.body;
    
    if (!bedId) {
      return res.status(400).json({ message: "bedId required" });
    }
    
    const bed = await BedSchedule.findOneAndUpdate(
      { bedId },
      {
        status: "CLEANING",
        currentEncounterId: null,
        assignedNurseId: null,
        occupancyStartTime: null
      },
      { new: true }
    );
    
    if (!bed) {
      return res.status(404).json({ message: "Bed not found" });
    }
    
    res.json({ message: "Bed released", bed });
  } catch (err) {
    console.error("RELEASE BED ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

/**
 * Initialize beds (create default bed layout)
 */
export const initializeBeds = async (req, res) => {
  try {
    const beds = [
      // ER beds
      { bedId: "ER-01", roomNumber: "101", department: "ER", bedType: "STANDARD", hasMonitor: true, canIsolate: false },
      { bedId: "ER-02", roomNumber: "102", department: "ER", bedType: "MONITORED", hasMonitor: true, canIsolate: true },
      { bedId: "ER-03", roomNumber: "103", department: "ER", bedType: "STANDARD", hasMonitor: true, canIsolate: false },
      // ICU beds
      { bedId: "ICU-01", roomNumber: "201", department: "ICU", bedType: "INTENSIVE", hasMonitor: true, hasVentilator: true, canIsolate: true },
      { bedId: "ICU-02", roomNumber: "202", department: "ICU", bedType: "INTENSIVE", hasMonitor: true, hasVentilator: true, canIsolate: true },
      { bedId: "ICU-03", roomNumber: "203", department: "ICU", bedType: "INTENSIVE", hasMonitor: true, hasVentilator: false, canIsolate: true },
      // General beds
      { bedId: "Gen-01", roomNumber: "301", department: "General", bedType: "STANDARD", hasMonitor: false, canIsolate: false },
      { bedId: "Gen-02", roomNumber: "302", department: "General", bedType: "STANDARD", hasMonitor: false, canIsolate: false },
      { bedId: "Gen-03", roomNumber: "303", department: "General", bedType: "STANDARD", hasMonitor: false, canIsolate: false },
    ];
    
    const results = [];
    for (const bedData of beds) {
      const bed = await BedSchedule.findOneAndUpdate(
        { bedId: bedData.bedId },
        { ...bedData, status: "AVAILABLE" },
        { upsert: true, new: true }
      );
      results.push(bed.bedId);
    }
    
    res.json({ message: `Initialized ${results.length} beds`, bedIds: results });
  } catch (err) {
    console.error("INITIALIZE BEDS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/**
 * Generate 24-hour availability mask from shift times with optional break time
 * Example: shiftStart="08:00", shiftEnd="16:00", breakStart="12:00", breakEnd="13:00"
 * Output: [0,0,0,0,0,0,0,0, 1,1,1,1, 0, 1,1,1, 0,0,0,0,0,0,0,0]
 * Available: 8am-12pm, 1pm-4pm (lunch break 12pm-1pm)
 * @param {string} shiftStart - HH:MM format
 * @param {string} shiftEnd - HH:MM format
 * @param {string} breakStart - Optional break start HH:MM format  
 * @param {string} breakEnd - Optional break end HH:MM format
 * @returns {number[]} Array of 24 values (1=available, 0=not available)
 */
function generateAvailabilityMask(shiftStart, shiftEnd, breakStart, breakEnd) {
  const mask = new Array(24).fill(0);
  
  if (!shiftStart || !shiftEnd) {
    return mask;
  }
  
  try {
    const [startHour] = shiftStart.split(":").map(Number);
    const [endHour] = shiftEnd.split(":").map(Number);
    
    // Mark all shift hours as available
    for (let h = startHour; h < endHour && h < 24; h++) {
      mask[h] = 1;
    }
    
    // Mark break hours as unavailable (if break times provided)
    if (breakStart && breakEnd) {
      try {
        const [breakStartHour] = breakStart.split(":").map(Number);
        const [breakEndHour] = breakEnd.split(":").map(Number);
        
        for (let h = breakStartHour; h < breakEndHour && h < 24; h++) {
          mask[h] = 0;
        }
      } catch (e) {
        // Invalid break format, ignore and continue with shift-only mask
      }
    }
  } catch (e) {
    // Invalid format, return empty mask
  }
  
  return mask;
}

export default {
  // Physician
  getAvailablePhysicians,
  getPhysicianSchedule,
  upsertPhysicianSchedule,
  importPhysicianSchedulesCSV,
  // Nurse
  getAvailableNurses,
  getNurseSchedule,
  upsertNurseSchedule,
  importNurseSchedulesCSV,
  // Bed
  getAvailableBeds,
  getAllBeds,
  assignBed,
  releaseBed,
  initializeBeds
};
