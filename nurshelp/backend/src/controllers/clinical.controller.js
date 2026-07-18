/**
 * Clinical operations controller
 * Handles: staff assignments, bed assignments, clinical handover, resource logging
 */

import { Encounter } from "../models/encounter.model.js";
import { ClinicalStaff } from "../models/clinicalstaff.model.js";
import { StaffClockRecord } from "../models/staffClockRecord.model.js";
import { BedSchedule } from "../models/bedSchedule.model.js";

export const getEncounterPatientIdentifier = (encounter = {}) => {
  const patient = encounter?.patient || {};
  const candidates = [patient.id_number, patient.id, patient._id, patient.mrn, patient.patientId];
  const value = candidates.find(candidate => candidate !== undefined && candidate !== null && String(candidate).trim() !== "");

  return value ? String(value) : null;
};

export const findActiveHospitalizationForPatient = async (encounter, EncounterModel = Encounter) => {
  const patientIdentifier = getEncounterPatientIdentifier(encounter);
  if (!patientIdentifier) {
    return null;
  }

  const currentEncounterId = encounter?._id?.toString?.() || encounter?.id?.toString?.();
  const query = {
    isHospitalized: true,
    $or: [
      { "patient.id_number": patientIdentifier },
      { "patient.id": patientIdentifier },
      { "patient._id": patientIdentifier },
      { "patient.mrn": patientIdentifier }
    ]
  };

  if (currentEncounterId) {
    query._id = { $ne: currentEncounterId };
  }

  const result = await EncounterModel.findOne(query);
  if (result && typeof result.lean === "function") {
    return result.lean();
  }

  return result;
};

/**
 * GET /api/clinical/staff/clocked-in/:hospitalId
 * Get all clocked-in staff (physicians and nurses) for a hospital today
 */
export const getClockedInStaff = async (req, res) => {
  try {
    const { role, encounterId } = req.query; // Optional filter: PHYSICIAN, NURSE, TRIAGE_NURSE
    const hospitalId = req.hospitalId;

    if (!hospitalId) {
      return res.status(400).json({ message: "Hospital ID not found in authentication" });
    }

    // If encounterId provided, get specialty from encounter and filter staff
    let specialtyFilter = null;
    if (encounterId) {
      const encounter = await Encounter.findById(encounterId).lean();
      specialtyFilter = encounter?.triage?.aiTriageReport?.specialty || encounter?.triage?.assigned_specialty || encounter?.triage?.specialty || null;
      if (specialtyFilter) {
        specialtyFilter = String(specialtyFilter).trim();
        console.log(`[CLINICAL] Filtering staff by specialty: ${specialtyFilter}`);
      }
    }

    // Get today's date (UTC)
    const today = new Date();
    today.setUTCHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    // Find all staff clocked in today (status = CLOCKED_IN or ON_BREAK)
    const clockQuery = {
      date: { $gte: today, $lt: tomorrow },
      status: { $in: ["CLOCKED_IN", "ON_BREAK"] }
    };

    const clockRecords = await StaffClockRecord.find(clockQuery)
      .populate("staffId", "firstName lastName email clinicalSpecialty systemRole assignment hospitalId")
      .lean();

    // Extract unique staff IDs and their roles
    const staffMap = new Map();
    for (const record of clockRecords) {
      if (!record.staffId) continue;
      const recordHospitalId = record.hospitalId?.toString() || record.staffId.hospitalId?.toString();
      if (recordHospitalId !== hospitalId.toString()) continue;
      if (!staffMap.has(record.staffId._id.toString())) {
        staffMap.set(record.staffId._id.toString(), {
          id: record.staffId._id,
          name: `${record.staffId.firstName} ${record.staffId.lastName}`,
          email: record.staffId.email,
          specialty: record.staffId.clinicalSpecialty,
          role: record.staffId.systemRole,
          department: record.staffId.assignment?.department,
          clock_status: record.status,
          clock_in_time: record.clockInTime,
          clock_out_time: record.clockOutTime,
          shift_start: record.clockInTime,
          shift_end: record.clockOutTime,
          last_break_end: record.breakEndTime
        });
      }
    }

    let staff = Array.from(staffMap.values());

    // Filter by role if specified (PHYSICIAN, NURSE, or both)
    if (role) {
      const roleUpper = role.toUpperCase();
      staff = staff.filter(s => s.role === roleUpper);
    }

    // Filter by specialty if encounterId provided and specialty found
    // IMPORTANT: Try to match specialty, but fallback to all physicians if no match found
    let specialtyMatched = false;
    let physiciansMatched = [];
    let physiciansOther = [];
    let nursesAll = [];
    
    if (specialtyFilter) {
      const specialtyLower = specialtyFilter.toLowerCase();
      
      // Filter physicians by specialty (primary match)
      physiciansMatched = staff.filter(s => {
        if (s.role !== 'PHYSICIAN') return false;
        const staffSpec = String(s.specialty || "").toLowerCase();
        return staffSpec.includes(specialtyLower) || specialtyLower.includes(staffSpec);
      });
      
      // Get all other physicians (for fallback if no specialty match)
      physiciansOther = staff.filter(s => {
        if (s.role !== 'PHYSICIAN') return false;
        const staffSpec = String(s.specialty || "").toLowerCase();
        return !(staffSpec.includes(specialtyLower) || specialtyLower.includes(staffSpec));
      });
      
      // Get all nurses (no specialty filter for nurses - always include as support)
      nursesAll = staff.filter(s => s.role === 'NURSE');
      
      // Check if physicians matched specialty
      if (physiciansMatched.length > 0) {
        specialtyMatched = true;
        // Found matching physicians: use matched + nurses
        staff = [...physiciansMatched, ...nursesAll];
      } else {
        // No matching physicians: fallback to ALL physicians + nurses
        console.log(`[CLINICAL] No ${specialtyFilter} physicians found, showing all available physicians`);
        staff = [...physiciansOther, ...nursesAll];
      }
    }

    // Separate physicians and nurses with scoring
    const physicians = staff
      .filter(s => s.role === 'PHYSICIAN')
      .map(s => {
        let reason;
        if (specialtyMatched && specialtyFilter) {
          reason = `Specialty match: ${specialtyFilter}`;
        } else if (specialtyFilter) {
          reason = `No ${specialtyFilter} specialists available; showing all physicians`;
        } else {
          reason = `Currently on duty`;
        }

        return {
          id: s.id,
          name: s.name,
          role: 'PHYSICIAN',
          specialty: s.specialty || "General",
          score: 85 + Math.floor(Math.random() * 15), // Mock score 85-100
          reason,
          clock_in_time: s.clock_in_time,
          clock_out_time: s.clock_out_time,
          shift_start: s.shift_start,
          shift_end: s.shift_end
        };
      });

    const nurses = staff
      .filter(s => s.role === 'NURSE')
      .map(s => ({
        id: s.id,
        name: s.name,
        role: 'NURSE',
        specialty: s.specialty || "General Nursing",
        score: 75 + Math.floor(Math.random() * 20), // Mock score 75-95
        // Always include nurses as support staff, even if specialty doesn't match exactly
        reason: `Support nursing staff - available for ${specialtyFilter || 'patient care'}`,
        clock_in_time: s.clock_in_time,
        clock_out_time: s.clock_out_time,
        shift_start: s.shift_start,
        shift_end: s.shift_end
      }));

    console.log(`[CLINICAL] Returning ${physicians.length} physicians + ${nurses.length} nurses`);

    res.json({
      success: true,
      count: physicians.length + nurses.length,
      physicians,
      nurses,
      clinicians: [...physicians, ...nurses] // combined array for convenience
    });
  } catch (err) {
    console.error("[CLINICAL] GET Clocked-in Staff Error:", err);
    res.status(500).json({ success: false, message: err.message });
  }
};

/**
 * GET /api/clinical/beds/available/:hospitalId
 * Get available beds for hospitalization
 */
export const getAvailableBeds = async (req, res) => {
  try {
    const { hospitalId } = req.params;
    const { ward, status } = req.query; // Optional: filter by ward, status

    if (!hospitalId) {
      return res.status(400).json({ message: "hospitalId required" });
    }

    let bedQuery = {
      status: { $in: ["AVAILABLE"] } // Unoccupied beds only
    };

    if (ward) {
      bedQuery.department = ward;
    }

    if (status) {
      bedQuery.status = status;
    }

    const beds = await BedSchedule.find(bedQuery)
      .select("bedId roomNumber department bedType status currentEncounterId")
      .lean();

    console.log(`[CLINICAL] Found ${beds.length} available beds for hospital ${hospitalId}`);

    // Normalize bed data for API response
    const normalizedBeds = beds.map(bed => ({
      _id: bed._id,
      bed_number: bed.bedId,
      room_number: bed.roomNumber,
      ward: bed.department,
      bed_type: bed.bedType,
      status: bed.status,
      is_occupied: bed.status !== "AVAILABLE"
    }));

    res.json({
      success: true,
      count: normalizedBeds.length,
      hospital_id: hospitalId,
      beds: normalizedBeds
    });
  } catch (err) {
    console.error("[CLINICAL] GET Available Beds Error:", err);
    res.status(500).json({ success: false, message: err.message });
  }
};

/**
 * POST /api/clinical/encounters/:encounterId/assign-staff
 * Assign physician or nurse to encounter (with edit tracking)
 */
export const assignStaffToEncounter = async (req, res) => {
  try {
    const { encounterId } = req.params;
    const { staffId, role } = req.body; // role: PHYSICIAN or NURSE

    if (!encounterId || !staffId || !role) {
      return res.status(400).json({ message: "encounterId, staffId, and role required" });
    }

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: "Encounter not found" });
    }

    // Get staff details
    const staff = await ClinicalStaff.findById(staffId).select("name email specialty role");
    if (!staff) {
      return res.status(404).json({ message: "Staff not found" });
    }

    // Assign staff based on role
    const roleUpper = role.toUpperCase();
    if (roleUpper === "PHYSICIAN") {
      encounter.assignedPhysicianId = staffId;
      encounter.assignedPhysicianName = staff.name;
      if (!encounter.triage) encounter.triage = {};
      encounter.triage.assigned_physician = {
        id: staffId,
        name: staff.name,
        specialty: staff.specialty
      };
      // Track edit
      if (!encounter.assignment_edits) encounter.assignment_edits = [];
      encounter.assignment_edits.push({
        type: "PHYSICIAN_ASSIGNED",
        staff_name: staff.name,
        staff_id: staffId,
        assigned_by: req.user?.id || "system",
        timestamp: new Date(),
        notes: `Physician assigned: ${staff.name}`
      });
    } else if (roleUpper === "NURSE" || roleUpper === "CLINICAL_NURSE") {
      encounter.assignedNurseId = staffId;
      encounter.assignedNurseName = staff.name;
      if (!encounter.triage) encounter.triage = {};
      encounter.triage.assigned_nurse = {
        id: staffId,
        name: staff.name,
        specialty: staff.specialty
      };
      // Track edit
      if (!encounter.assignment_edits) encounter.assignment_edits = [];
      encounter.assignment_edits.push({
        type: "NURSE_ASSIGNED",
        staff_name: staff.name,
        staff_id: staffId,
        assigned_by: req.user?.id || "system",
        timestamp: new Date(),
        notes: `Nurse assigned: ${staff.name}`
      });
    } else {
      return res.status(400).json({ message: "Invalid role. Use PHYSICIAN or NURSE" });
    }

    await encounter.save();

    console.log(`[CLINICAL] ${roleUpper} assigned to encounter ${encounterId}: ${staff.name}`);

    res.json({
      success: true,
      message: `${roleUpper} assigned successfully`,
      encounter_id: encounterId,
      assigned_staff: {
        id: staffId,
        name: staff.name,
        role: roleUpper
      }
    });
  } catch (err) {
    console.error("[CLINICAL] Assign Staff Error:", err);
    res.status(500).json({ success: false, message: err.message });
  }
};

/**
 * POST /api/clinical/encounters/:encounterId/assign-bed
 * Assign bed to encounter for hospitalization
 */
export const assignBedToEncounter = async (req, res) => {
  try {
    const { encounterId } = req.params;
    const { bedId, hospitalize } = req.body; // hospitalize: true/false

    if (!encounterId || !bedId) {
      return res.status(400).json({ message: "encounterId and bedId required" });
    }

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: "Encounter not found" });
    }

    if (hospitalize === true) {
      // Mark patient as hospitalized and assign bed
      const bed = await BedSchedule.findById(bedId);
      if (!bed) {
        return res.status(404).json({ message: "Bed not found" });
      }

      if (bed.status !== "AVAILABLE" && bed.currentEncounterId?.toString() !== encounter._id.toString()) {
        return res.status(409).json({ message: "Bed is already occupied" });
      }

      const activeHospitalization = await findActiveHospitalizationForPatient(encounter);
      if (activeHospitalization && activeHospitalization._id?.toString() !== encounter._id.toString()) {
        const activeBedLabel = activeHospitalization.triage?.assigned_bed?.bed_number || activeHospitalization.currentBedId || "an active bed";
        return res.status(409).json({
          success: false,
          message: `Patient already hospitalized in another bed (${activeBedLabel})`,
          active_encounter_id: activeHospitalization._id
        });
      }

      // Assign bed
      encounter.currentBedId = bedId;
      encounter.isHospitalized = true;
      encounter.hospitalization_date = new Date();

      if (!encounter.triage) encounter.triage = {};
      encounter.triage.assigned_bed = {
        id: bedId,
        bed_number: bed.bedId,
        ward: bed.department
      };

      // Mark bed as occupied
      bed.status = "OCCUPIED";
      bed.currentEncounterId = encounter._id;
      bed.occupancyStartTime = new Date();
      await bed.save();

      // Track edit
      if (!encounter.assignment_edits) encounter.assignment_edits = [];
      encounter.assignment_edits.push({
        type: "HOSPITALIZATION",
        bed_id: bedId,
        bed_number: bed.bedId,
        authorized_by: req.user?.id || "system",
        timestamp: new Date(),
        notes: `Patient hospitalized - Bed ${bed.bedId} assigned`
      });

      console.log(`[CLINICAL] Patient ${encounter.patient.id} hospitalized in bed ${bed.bedId}`);
    } else {
      // Discharge from bed if hospitalized
      if (encounter.currentBedId) {
        const bed = await BedSchedule.findById(encounter.currentBedId);
        if (bed) {
          bed.status = "CLEANING";
          bed.currentEncounterId = null;
          bed.lastCleanedAt = new Date();
          bed.estimatedReadyTime = new Date(Date.now() + (bed.cleaningDurationMinutes || 30) * 60000);
          await bed.save();
        }
      }

      encounter.currentBedId = null;
      encounter.isHospitalized = false;
      encounter.discharge_date = new Date();

      if (!encounter.triage) encounter.triage = {};
      encounter.triage.assigned_bed = null;

      // Track edit
      if (!encounter.assignment_edits) encounter.assignment_edits = [];
      encounter.assignment_edits.push({
        type: "DISCHARGE",
        authorized_by: req.user?.id || "system",
        timestamp: new Date(),
        notes: `Patient discharged from bed`
      });

      console.log(`[CLINICAL] Patient ${encounter.patient.id} discharged`);
    }

    await encounter.save();

    res.json({
      success: true,
      message: hospitalize ? "Patient hospitalized" : "Patient discharged",
      encounter_id: encounterId,
      hospitalized: encounter.isHospitalized,
      bed_id: hospitalize ? bedId : null
    });
  } catch (err) {
    console.error("[CLINICAL] Assign Bed Error:", err);
    res.status(500).json({ success: false, message: err.message });
  }
};

export default {
  getClockedInStaff,
  getAvailableBeds,
  assignStaffToEncounter,
  assignBedToEncounter
};
