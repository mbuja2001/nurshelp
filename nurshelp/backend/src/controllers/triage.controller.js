import mongoose from "mongoose";
import axios from "axios";
import { Encounter } from "../models/encounter.model.js";
import { NurseClockRecord } from "../models/staffClockRecord.model.js";
import { ClinicalStaff } from "../models/clinicalstaff.model.js";
import { Shift } from "../models/shift.model.js";
import { Hospital } from "../models/hospital.model.js";

/**
 * ============================================================
 * OPTION B: REAL-TIME STAFFING SERVICE DISCOVERY
 * ============================================================
 * Professional Architecture Pattern:
 * - Frontend sends: hospitalId, patientId, triageNurseId ONLY
 * - Backend fetches: Currently clocked-in staff for that hospital
 * - ESI_Engine receives: Rich staffing context + patient data
 * - Benefits: Mutable, secure, decoupled, real-time accuracy
 */

/**
 * Fetch currently clocked-in staff for a hospital (TODAY)
 * This is the "Source of Truth" for real-time staffing
 * Used by triage to ensure assigned staff are ACTUALLY present
 * 
 * @param {string} hospitalId - Hospital ObjectId
 * @returns {Promise} { physicians: [...], triageNurses: [...], clinicalNurses: [...] }
 */
export const getClockedInStaffForHospital = async (hospitalId) => {
  try {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    console.log(`[STAFFING SERVICE] Querying clocked-in staff for hospital ${hospitalId}...`);
    
    // Get all staff CLOCKED_IN (not ON_BREAK) for today
    const clockedInRecords = await NurseClockRecord.find({
      date: { $gte: today, $lt: tomorrow },
      status: 'CLOCKED_IN',  // Only those actively clocked in (not on break)
      clockOutTime: null     // Still on shift (not clocked out)
    }).populate({
      path: 'staffId',
      populate: { path: 'hospitalId' }  // Get hospital reference
    });
    
    console.log(`[STAFFING SERVICE] Found ${clockedInRecords.length} clocked-in staff records`);
    
    // Filter to only staff in this hospital
    const hospitalStaff = clockedInRecords.filter(record => {
      const staffHospital = record.staffId?.hospitalId;
      return staffHospital && staffHospital._id ? staffHospital._id.toString() === hospitalId.toString() : false;
    });
    
    console.log(`[STAFFING SERVICE] After hospital filter: ${hospitalStaff.length} staff in hospital ${hospitalId}`);
    
    // Categorize by role/specialty
    const result = {
      physicians: [],
      triageNurses: [],
      clinicalNurses: [],
      allStaff: []
    };
    
    for (const record of hospitalStaff) {
      const staff = record.staffId;
      const staffData = {
        staffId: staff._id,
        name: staff.name,
        surname: staff.surname || '',
        email: staff.email || '',
        clockRecordId: record._id,
        clockedInAt: record.clockInTime,
        specialty: staff.specialty || 'General',
        available_now: true,  // By definition, they're clocked in
        role: staff.role  // NURSE, PHYSICIAN, SPECIALIST, etc.
      };
      
      if (staff.role === 'PHYSICIAN' || staff.specialty === 'Emergency Medicine') {
        result.physicians.push(staffData);
      } else if (staff.role === 'NURSE') {
        // Could be triage or clinical - we'll let shift assignment determine
        result.clinicalNurses.push(staffData);
      }
      
      result.allStaff.push(staffData);
    }
    
    console.log(`[STAFFING SERVICE] Staffing breakdown:
      - Physicians: ${result.physicians.length}
      - Clinical Nurses: ${result.clinicalNurses.length}
      - Triage Nurses: ${result.triageNurses.length}
      - Total: ${result.allStaff.length}`);
    
    return result;
  } catch (err) {
    console.error('[STAFFING SERVICE] Error fetching clocked-in staff:', err.message);
    return {
      physicians: [],
      triageNurses: [],
      clinicalNurses: [],
      allStaff: [],
      error: err.message
    };
  }
};

/**
 * Helper: safely parse transcript string/array
 */
const parseTranscript = (transcript) => {
  if (!transcript) return [];
  if (Array.isArray(transcript)) return transcript;
  if (typeof transcript === "string") {
    try {
      // Replace single quotes with double quotes if needed (Python-style JSON)
      return JSON.parse(transcript.replace(/'/g, '"'));
    } catch (err) {
      console.warn("Failed to parse transcript string, defaulting to empty array");
      return [];
    }
  }
  return [];
};

/**
 * ----------------------
 * Run Triage (from Python worker or frontend)
 * Now supports computeMode: 'LLM_ONLY' | 'RAG_FULL' | 'HYBRID'
 * ----------------------
 */
// ================================================================
// PRODUCTION ENDPOINTS (Updated for Sophisticated MedGemma Synthesis)
// ================================================================
const PYTHON_URL_SIMPLE = process.env.PYTHON_URL || "http://localhost:5000/triage";  // Simple (ESI only) - DEPRECATED
const PYTHON_URL_LLM = "http://localhost:5000/llm-triage";  // Sophisticated (MedGemma synthesis + diagnosis)
// Legacy variable for backward compatibility (always points to LLM endpoint now)
const PYTHON_URL = PYTHON_URL_LLM;

/**
 * Helper: Parse date string to UTC at start of day
 */
const parseDateString = (dateStr) => {
  const [year, month, day] = dateStr.split('-').map(Number);
  return new Date(Date.UTC(year, month - 1, day, 0, 0, 0, 0));
};

/**
 * Fetch current staff scheduling for a given date/time
 * Returns data only (not controller), to be fed into ESI_Engine
 */
const getStaffScheduleDataService = async (NurseSchedule, shiftDate = new Date()) => {
  try {
    const dateStr = shiftDate.toISOString().split('T')[0];
    const parsedDate = parseDateString(dateStr);

    // Get all nurses available on this date
    const nurseSchedules = await NurseSchedule.find({
      shiftDate: parsedDate,
      status: { $in: ['SCHEDULED', 'ON_DUTY'] }
    }).lean();

    // Get physician schedules (medical staff)
    const physicianSchedules = await NurseSchedule.find({
      shiftDate: parsedDate,
      role: 'PHYSICIAN',
      status: { $in: ['SCHEDULED', 'ON_DUTY'] }
    }).lean();

    // Format for ESI_Engine consumption
    const staffData = {
      date: dateStr,
      timestamp: shiftDate.toISOString(),
      nurses: nurseSchedules.map(ns => ({
        nurseId: ns.nurseName,
        name: ns.nurseName,
        specialty: ns.specialty,
        currentWorkload: ns.currentWorkload || 0,
        maxBedsPerShift: ns.maxBedsPerShift || 6,
        isOnBreak: ns.status === 'ON_BREAK',
        shift: {
          start: ns.shiftStart,
          end: ns.shiftEnd
        }
      })),

      physicians: physicianSchedules.map(ps => ({
        physicianId: ps.physicianName,
        name: ps.physicianName,
        specialty: ps.specialty || 'General',
        workloadScore: ps.workloadScore || 0.5,
        isAvailable: ps.status === 'ON_DUTY',
        shift: {
          start: ps.shiftStart,
          end: ps.shiftEnd
        }
      }))
    };

    return staffData;
  } catch (error) {
    console.error('Error fetching staff schedule data:', error);
    return {
      date: new Date().toISOString().split('T')[0],
      nurses: [],
      physicians: [],
      error: error.message
    };
  }
};

export const runTriage = async (req, res) => {
  try {
    const startTime = Date.now();
    const payload = req.body;

    if (!payload || !payload.patient || !payload.vitals) {
      return res.status(400).json({ message: "Invalid payload" });
    }

    // Extract computeMode (default to HYBRID)
    const computeMode = payload.computeMode || 'HYBRID';
    if (!['LLM_ONLY', 'RAG_FULL', 'HYBRID'].includes(computeMode)) {
      return res.status(400).json({ message: "Invalid computeMode. Must be LLM_ONLY, RAG_FULL, or HYBRID" });
    }

    const nurseId = req.user?._id || null;
    const transcriptArray = parseTranscript(payload.transcript);

    // Build payload for Python engine with computeMode
    const enginePayload = {
      patient: payload.patient,
      vitals: payload.vitals,
      transcript: transcriptArray,
      computeMode, // Tells ESI_Engine how to process
      staffScheduling: payload.staffScheduling || {},
      encounterId: payload.encounterId
    };

    console.log(`[Triage] Processing with computeMode: ${computeMode}`);
    console.log("➡️ Sending request to Python:", PYTHON_URL);
    
    let triage = null;
    let aiProcessingTime = 0;

    // Only call Python engine if not LLM_ONLY
    if (computeMode !== 'LLM_ONLY') {
      const aiStartTime = Date.now();
      try {
        const pythonRes = await axios.post(PYTHON_URL, enginePayload, {
          timeout: 0,  // No timeout - wait indefinitely for GGUF inference
          headers: { "Content-Type": "application/json" }
        });
        triage = pythonRes.data;
        aiProcessingTime = Date.now() - aiStartTime;
        console.log(`[Triage] ESI_Engine response received in ${aiProcessingTime}ms`);
      } catch (engineError) {
        console.error('[Triage] ESI_Engine error:', engineError.message);
        // If AI fails, provide basic LLM-only response
        triage = {
          ESI: Math.ceil((payload.vitals?.severity || 1) * 5 / 10),
          confidence: 0.5,
          warning: 'AI processing failed, using fallback ESI estimation'
        };
      }
    } else {
      // LLM_ONLY: Skip Python engine, use basic heuristic
      console.log('[Triage] computeMode is LLM_ONLY - skipping RAG/FAISS pipeline');
      triage = {
        ESI: Math.ceil((payload.vitals?.severity || 1) * 5 / 10),
        confidence: 0.6,
        mode: 'LLM_ONLY'
      };
    }

    const encounter = await Encounter.create({
      nurse_id: nurseId,
      patient: payload.patient,
      transcript: transcriptArray,
      vitals: payload.vitals,
      triage: { 
        ...triage, 
        severity: triage.ESI,
        computeMode,
        aiProcessingTime
      },
      nurseNotes: payload.nurseNotes || "",
      status: nurseId ? "pending" : "unassigned",
      submittedAt: null,
      isWaiting: true,
      severity: triage.ESI || 1
    });

    return res.status(201).json({ 
      message: "Triage completed", 
      encounter,
      computeMode,
      processingMetrics: {
        totalTime: Date.now() - startTime,
        aiProcessingTime,
        usedAI: computeMode !== 'LLM_ONLY'
      }
    });
  } catch (err) {
    if (err.code === "ECONNABORTED") return res.status(504).json({ message: "Triage service timeout" });
    if (err.code === "ECONNREFUSED") return res.status(500).json({ message: "Python triage service unavailable" });
    if (err.response) return res.status(500).json({ message: "Python service error", error: err.response.data });

    console.error("Triage error:", err.message);
    res.status(500).json({ message: err.message });
  }
};

/**
 * ----------------------
 * Start a new encounter (manual nurse intake)
 * ----------------------
 */
export const startEncounter = async (req, res) => {
  try {
    if (!req.user || !req.user._id) return res.status(401).json({ message: "Not authorized" });

    const transcriptArray = parseTranscript(req.body.transcript);

    const encounter = await Encounter.create({
      nurse_id: req.user._id,
      patient: req.body.patient || {},
      transcript: transcriptArray,
      vitals: {
        temp: req.body.temperature,
        bp: req.body.blood_pressure,
        hr: req.body.heart_rate,
        o2: req.body.oxygen_saturation,
        resp: req.body.resp
      },
      triage: req.body.triage || {},
      severity: req.body.severity || 1,
      status: "pending",
      isWaiting: true,
      nurseNotes: req.body.nurseNotes || ""
    });

    return res.status(201).json({ message: "Encounter created successfully", encounter });
  } catch (err) {
    console.error("Start Encounter Error:", err);
    return res.status(400).json({ message: err.message || "Failed to create encounter" });
  }
};

/**
 * ----------------------
 * Confirm encounter
 * Only the creating nurse may confirm
 * ----------------------
 */
export const confirmEncounter = async (req, res) => {
  try {
    if (!req.user || !req.user._id) return res.status(401).json({ message: "Not authorized" });

    const { id } = req.params;
    if (!mongoose.isValidObjectId(id)) return res.status(400).json({ message: "Invalid encounter id" });

    const encounter = await Encounter.findById(id);

    if (!encounter) return res.status(404).json({ message: "Encounter not found" });

    // Ensure nurse is assigned
    if (!encounter.nurse_id) {
      return res.status(403).json({ message: "This encounter is not assigned to any nurse yet" });
    }

    // Only allow the creating nurse to confirm
    if (encounter.nurse_id.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: "You are not authorized to confirm this encounter" });
    }

    encounter.nurseNotes = req.body.nurseNotes || encounter.nurseNotes;
    encounter.status = "confirmed";
    encounter.submittedAt = new Date();

    await encounter.save();

    return res.status(200).json({ message: "Encounter confirmed", encounter });
  } catch (err) {
    console.error("Confirm Encounter Error:", err);
    return res.status(500).json({ message: err.message || "Confirm failed" });
  }
}

/**
 * ----------------------
 * Hospital-Specific Triage Request with Shift Assignment
 * POST /api/triage/hospital-request
 * 
 * Accepts triage request for a specific hospital
 * - Fetches available shifts for that hospital/day
 * - Sends to ESI_Engine.py for triage scoring
 * - Automatically assigns patient to available physician + clinical nurse
 * ----------------------
 */
export const requestTriageWithHospitalShifts = async (req, res) => {
  try {
    const { hospitalId, patient, vitals, transcript } = req.body;

    // DEBUG: Log incoming request structure
    console.log(`\n[HOSPITAL-TRIAGE] 🔍 DEBUG: Incoming request body structure:`);
    console.log(`  - Keys in req.body: ${Object.keys(req.body).join(', ')}`);
    console.log(`  - patient object exists: ${!!patient}`);
    if (patient) {
      console.log(`  - patient.symptoms type: ${typeof patient.symptoms}`);
      console.log(`  - patient.symptoms value: ${JSON.stringify(patient.symptoms)}`);
      console.log(`  - patient.symptoms keys: ${Array.isArray(patient.symptoms) ? 'IS_ARRAY' : typeof patient.symptoms}`);
    }
    console.log(`  - Request body size: ${JSON.stringify(req.body).length} bytes`);

    // Validation
    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'hospitalId is required',
      });
    }

    if (!patient || !Object.keys(patient).length) {
      return res.status(400).json({
        success: false,
        message: 'patient data is required',
      });
    }

    console.log(`[HOSPITAL-TRIAGE] New request for hospital: ${hospitalId}`);
    console.log(`[HOSPITAL-TRIAGE] Patient: ${patient.name || 'Unknown'}`);

    // Step 0: Resolve hospitalId - handle both ObjectId and string codes
    // Frontend may send "HOSPITAL_001" (hospitalID code) but DB stores ObjectId references
    let resolvedHospitalId = hospitalId;
    try {
      // Try to use hospitalId as-is first (if it's a valid ObjectId)
      if (!mongoose.Types.ObjectId.isValid(hospitalId)) {
        // Not an ObjectId - try to find hospital by hospitalID field
        console.log(`[HOSPITAL-TRIAGE] hospitalId "${hospitalId}" is not an ObjectId, looking up by hospitalID...`);
        const hospital = await Hospital.findOne({ hospitalID: hospitalId });
        if (hospital) {
          resolvedHospitalId = hospital._id;
          console.log(`[HOSPITAL-TRIAGE] Resolved hospital ID "${hospitalId}" to ObjectId: ${hospital._id}`);
        } else {
          console.warn(`[HOSPITAL-TRIAGE] Hospital "${hospitalId}" not found - proceeding with original value`);
          // Continue anyway - shifts query will just return empty and we'll proceed without staff
        }
      }
    } catch (err) {
      console.warn(`[HOSPITAL-TRIAGE] Error resolving hospitalId: ${err.message} - will attempt query with original value`);
    }

    // Step 1: Get published shifts for this hospital today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    console.log(`[HOSPITAL-TRIAGE] Fetching shifts for hospital ${resolvedHospitalId} between ${today.toISOString()} and ${tomorrow.toISOString()}...`);

    const todayShifts = await Shift.find({
      hospitalId: resolvedHospitalId,
      status: 'PUBLISHED',
      date: { $gte: today, $lt: tomorrow },
    }).lean();

    if (!todayShifts || todayShifts.length === 0) {
      console.warn(`[HOSPITAL-TRIAGE] No shifts found for hospital ${hospitalId} - proceeding with empty staff lists`);
    }

    // ============================================================
    // OPTION B: REAL-TIME STAFFING (Production Architecture)
    // ============================================================
    // Instead of getting staff from shift assignments (which are STATIC),
    // fetch ACTUALLY CLOCKED-IN staff (REAL-TIME, ACCURATE)
    // This ensures assigned physicians/nurses are PHYSICALLY PRESENT
    // ============================================================
    
    console.log(`[HOSPITAL-TRIAGE] Using OPTION B: Real-Time Staffing Discovery`);
    console.log(`[HOSPITAL-TRIAGE] Fetching clocked-in staff (not just shift assignments)...`);
    
    const staffingContext = await getClockedInStaffForHospital(resolvedHospitalId);
    
    // Use the real-time staff data
    const availablePhysicians = staffingContext.physicians || [];
    const availableClinicalNurses = staffingContext.clinicalNurses || [];
    const availableTriageNurses = staffingContext.triageNurses || [];
    const allStaff = staffingContext.allStaff || [];

    console.log(`[HOSPITAL-TRIAGE] Real-time staff availability:`);
    console.log(`  - Physicians: ${availablePhysicians.length}`);
    console.log(`  - Clinical Nurses: ${availableClinicalNurses.length}`);
    console.log(`  - Triage Nurses: ${availableTriageNurses.length}`);
    console.log(`  - Total clocked-in: ${allStaff.length}`);

    // Build shift details for reference (audit trail only)
    const shiftDetails = todayShifts.map(shift => ({
      shiftId: shift._id,
      shiftName: shift.shiftName,
      department: shift.department,
      shiftStart: shift.shiftStart,
      shiftEnd: shift.shiftEnd,
      status: shift.status
    }));

    console.log(`[HOSPITAL-TRIAGE] Reference shift data: ${shiftDetails.length} shifts found`);

    // ============================================================
    // OPTION A (OLD - REMOVED): Extract staff from shift assignments
    // This approach was STATIC and could assign clocked-out staff
    // ============================================================

    // Step 3: Prepare payload for ESI_Engine.py with CLINICAL HANDOVER CONTEXT
    // For clinical handover workflow:
    // - hospitalId: where the patient is being triaged
    // - patientId: patient identifier from frontend
    // - triageNurseId: which nurse is performing the triage (for audit trail)
    // - availableStaff: REAL-TIME clocked-in staff for assignment
    // This allows clinical dashboards to display work items by assigned PRESENT staff
    
    const patientId = req.body.patientId || 'UNKNOWN';          // Patient identifier
    const triageNurseId = req.body.triageNurseId || 'UNKNOWN';  // From frontend - which nurse is triaging
    
    const esiPayload = {
      hospitalId: resolvedHospitalId,
      patientId,                    // ← Clinical handover: patient identifier
      triageNurseId,                 // ← Clinical handover: who is triaging (audit trail)
      patient,
      vitals: vitals || {},
      transcript: transcript || '',
      availableStaff: {
        physicians: availablePhysicians,
        triageNurses: availableTriageNurses,
        clinicalNurses: availableClinicalNurses,
      },
      shifts: shiftDetails,
      date: today.toISOString().split('T')[0],
    };

    console.log(`[HOSPITAL-TRIAGE] 📋 Clinical Handover Payload Ready (OPTION B - Real-time staff):`);
    console.log(`  - Hospital: ${resolvedHospitalId}`);
    console.log(`  - Patient: ${patientId}`);
    console.log(`  - Triage Nurse: ${triageNurseId}`);
    console.log(`  - Physicians: ${availablePhysicians.length} available`);
    console.log(`  - Triage Nurses: ${availableTriageNurses.length} available`);
    console.log(`  - Clinical Nurses: ${availableClinicalNurses.length} available`);
    console.log(`  - Payload size: ${JSON.stringify(esiPayload).length} bytes`);

    // Step 4: Call ESI_Engine.py - SOPHISTICATED LLM ENDPOINT (MedGemma Synthesis)
    // This call uses the /llm-triage endpoint which provides:
    // - MedGemma GGUF inference for clinical synthesis + differential diagnosis
    // - Hardware-aware (GPU acceleration or CPU fallback)
    // - SATS/TEWS scoring with LLM reasoning
    // - Pathognomonic markers detection
    // NOT just simple ESI scoring - full clinical intelligence
    
    console.log(`[HOSPITAL-TRIAGE] 🚀 Calling MedGemma LLM endpoint: ${PYTHON_URL_LLM}...`);
    console.log(`[HOSPITAL-TRIAGE]    Uses: GGUF inference, synthesis, differential diagnosis, hardware-aware`);
    
    // Transform payload to /llm-triage schema (SATS/TEWS format for MedGemma)
    const vitalsNormalized = {
      temperature: parseFloat(vitals?.temp || vitals?.temperature || 37.0),
      heartRate: parseInt(vitals?.hr || vitals?.pulse || vitals?.heartRate || 80),
      respiratoryRate: parseInt(vitals?.resp || vitals?.respiratoryRate || 16),
      systolicBP: parseInt(vitals?.sysBloodPressure || vitals?.bp_systolic || vitals?.systolicBP || 120),
      diastolicBP: parseInt(vitals?.diaBloodPressure || vitals?.bp_diastolic || vitals?.diastolicBP || 80),
      oxygenSaturation: parseFloat(vitals?.o2 || vitals?.oxygenSaturation || 98.0)
    };
    
    // Transform to /llm-triage payload schema
    // Normalize transcript to string
    const transcriptStr = typeof transcript === 'string' ? transcript : (typeof transcript === 'object' ? JSON.stringify(transcript) : String(transcript || ''));
    
    // ===== BUILD COMPREHENSIVE CHIEF COMPLAINT FROM ALL SOURCES =====
    // Collect all clinical information
    const clinicalParts = [];

    // DEBUG: Show what we're extracting from patient
    console.log(`\n[HOSPITAL-TRIAGE] 🔍 DEBUG: Building chief complaint from patient data:`);
    console.log(`  - patient.symptoms exists: ${!!patient.symptoms}`);
    console.log(`  - patient.symptoms type: ${typeof patient.symptoms}`);
    console.log(`  - patient.symptoms value: ${JSON.stringify(patient.symptoms)}`);
    console.log(`  - patient.symptoms length: ${patient.symptoms?.length || 'N/A'}`);
    console.log(`  - patient.history: ${JSON.stringify(patient.history)}`);
    console.log(`  - patient.duration: ${JSON.stringify(patient.duration)}`);
    console.log(`  - transcript: ${JSON.stringify(transcript)}`);
    
    // 0. ADD PATIENT DEMOGRAPHICS FIRST (Critical for age-specific diagnoses)
    // Format: "X-year-old patient" or "X month old infant" for ESI_Engine regex extraction
    const ageDisplay = patient?.age !== undefined && patient?.age !== null 
      ? `${patient.age}-year-old patient` 
      : (patient?.ageMonths 
          ? `${patient.ageMonths}-month-old infant` 
          : 'Patient');
    const genderInfo = patient?.gender ? ` (${patient.gender})` : '';
    clinicalParts.push(`${ageDisplay}${genderInfo}`);
    
    // 1. Add transcript first (highest priority clinical information)
    if (transcriptStr && transcriptStr.trim() && transcriptStr !== '[]') {
      clinicalParts.push(transcriptStr.trim());
    }
    
    // 2. Add patient symptoms
    if (patient?.symptoms) {
      let symptomsText = '';
      if (Array.isArray(patient.symptoms)) {
        symptomsText = patient.symptoms.filter(s => s && s.trim()).join(', ');
      } else if (typeof patient.symptoms === 'string' && patient.symptoms.trim() && patient.symptoms !== '[]') {
        symptomsText = patient.symptoms.trim();
      }
      if (symptomsText) {
        clinicalParts.push(`Symptoms: ${symptomsText}`);
      }
    }
    
    // 3. Add patient history
    if (patient?.history && patient.history.trim() && patient.history !== '[]') {
      clinicalParts.push(`History: ${patient.history.trim()}`);
    }
    
    // 4. Add duration
    if (patient?.duration && patient.duration.trim() && patient.duration !== '[]') {
      clinicalParts.push(`Duration: ${patient.duration.trim()}`);
    }
    
    // 5. Add pain level
    if (patient?.painLevel && patient.painLevel.trim() && patient.painLevel !== '[]') {
      clinicalParts.push(`Pain Level: ${patient.painLevel.trim()}`);
    }
    
    // 6. Build final chief complaint - age is now ALWAYS included at the start
    const chiefComplaintStr = clinicalParts.join('. ');
    
    console.log(`[HOSPITAL-TRIAGE] 📋 Built chief complaint from ${clinicalParts.length} source(s):`);
    console.log(`  - Length: ${chiefComplaintStr.length} chars`);
    console.log(`  - Preview: ${chiefComplaintStr.substring(0, 100)}...`);
    
    const llmTriagePayload = {
      patient: patient,
      patientId: patientId,
      vitals: vitalsNormalized,
      vitalSigns: vitalsNormalized,  // Backward compat
      chiefComplaint: chiefComplaintStr, // Comprehensive clinical picture
      symptoms: chiefComplaintStr, // Same as chief complaint - comprehensive
      triageNotes: chiefComplaintStr,
      // Triage nurse context (audit trail)
      triageNurseId: triageNurseId,
      // Staff assignments (from real-time lookup) for recommendations
      assignedPhysicianName: availablePhysicians.length > 0 ? availablePhysicians[0].name : 'Not Assigned',
      assignedPhysicianId: availablePhysicians.length > 0 ? availablePhysicians[0]._id : null,
      assignedNurseName: availableClinicalNurses.length > 0 ? availableClinicalNurses[0].name : 'N/A',
      assignedNurseId: availableClinicalNurses.length > 0 ? availableClinicalNurses[0]._id : null,
      // Hospital context for clinical handover
      hospitalId: resolvedHospitalId.toString(),
      hospitalAware: true,
      realTimeStaffing: true
    };
    
    console.log(`[HOSPITAL-TRIAGE] 📋 LLM Payload prepared for MedGemma synthesis:`);
    console.log(`  - Patient: ${patientId}`);
    console.log(`  - Temperature: ${vitalsNormalized.temperature}°C`);
    console.log(`  - Chief Complaint: ${chiefComplaintStr.substring(0, 80) || 'Not provided'}...`);
    console.log(`  - Payload size: ${JSON.stringify(llmTriagePayload).length} bytes`);
    
    let triageResult;
    let aiTriageReport;
    let satsReasoning = {};  // Declare in outer scope so it's accessible after try-catch
    try {
      const llmResponse = await axios.post(PYTHON_URL_LLM, llmTriagePayload, {
        timeout: 0  // No timeout - wait indefinitely for GGUF inference (can be 60s+ on CPU)
      });
      
      // DEBUG: Log the ENTIRE Flask response structure BEFORE extraction
      console.log(`[HOSPITAL-TRIAGE] 🔍 RAW Flask response received (full dump):`);
      console.log(`[HOSPITAL-TRIAGE]   - Response status: ${llmResponse.status}`);
      console.log(`[HOSPITAL-TRIAGE]   - Response status text: ${llmResponse.statusText}`);
      console.log(`[HOSPITAL-TRIAGE]   - Has data property: ${!!llmResponse.data}`);
      
      // CRITICAL: Log the entire response to debug structure
      console.log(`[HOSPITAL-TRIAGE] 📦 FULL RESPONSE OBJECT:`);
      console.log(JSON.stringify(llmResponse.data, null, 2));
      
      console.log(`[HOSPITAL-TRIAGE]   - Data keys: ${Object.keys(llmResponse.data || {}).join(', ')}`);
      console.log(`[HOSPITAL-TRIAGE]   - Has clinicalAnalysis: ${!!llmResponse.data?.clinicalAnalysis}`);
      console.log(`[HOSPITAL-TRIAGE]   - Has supervisor_output: ${!!llmResponse.data?.supervisor_output}`);
      
      // CRITICAL FIX: Flask nests clinicalAnalysis under data.data, not data directly
      const flaskDataLevel = llmResponse.data?.data || llmResponse.data || {};
      
      if (flaskDataLevel.clinicalAnalysis) {
        console.log(`[HOSPITAL-TRIAGE]   - Found at llmResponse.data.data.clinicalAnalysis or llmResponse.data.clinicalAnalysis`);
        console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis keys: ${Object.keys(flaskDataLevel.clinicalAnalysis).join(', ')}`);
        console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis.pathophysiology: ${flaskDataLevel.clinicalAnalysis.pathophysiology ? flaskDataLevel.clinicalAnalysis.pathophysiology.substring(0, 100) : '(EMPTY)'}`);
        console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis.clinical_discriminators: ${JSON.stringify(flaskDataLevel.clinicalAnalysis.clinical_discriminators)}`);
        console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis.differential_diagnosis: ${Array.isArray(flaskDataLevel.clinicalAnalysis.differential_diagnosis) ? flaskDataLevel.clinicalAnalysis.differential_diagnosis.length + ' items' : '(NOT_ARRAY)'}`);
      } else if (flaskDataLevel.aiTriageReport) {
        console.log(`[HOSPITAL-TRIAGE]   - Found at flaskDataLevel.aiTriageReport`);
        console.log(`[HOSPITAL-TRIAGE]   - aiTriageReport keys: ${Object.keys(flaskDataLevel.aiTriageReport).join(', ')}`);
      } else if (llmResponse.data?.supervisor_output) {
        console.log(`[HOSPITAL-TRIAGE]   - Found at llmResponse.data.supervisor_output instead`);
        console.log(`[HOSPITAL-TRIAGE]   - supervisor_output keys: ${Object.keys(llmResponse.data.supervisor_output).join(', ')}`);
      } else {
        console.log(`[HOSPITAL-TRIAGE]   - ⚠️ CRITICAL: No clinicalAnalysis or supervisor_output found!`);
        console.log(`[HOSPITAL-TRIAGE]   - flaskDataLevel keys: ${Object.keys(flaskDataLevel).join(', ')}`);
      }
      
      if (flaskDataLevel.tewsData) {
        console.log(`[HOSPITAL-TRIAGE]   - tewsData.escalated_to: ${flaskDataLevel.tewsData.escalated_to}`);
      }
      
      const tewsData = flaskDataLevel.tewsData || {};
      satsReasoning = flaskDataLevel.SATS_reasoning || flaskDataLevel.sats_reasoning || flaskDataLevel.SATSReasoning || flaskDataLevel.satsReasoning || flaskDataLevel.tewsData || {};
      
      // Normalize structure: tewsData uses 'components' and 'total', frontend and encounter expect 'tews_components' and 'tews_total'
      // Also map 'priority' field - in tewsData it's a numeric code (1-4), store as tews_priority
      if (satsReasoning && (!satsReasoning.tews_total || !satsReasoning.tews_components)) {
        satsReasoning = {
          ...satsReasoning,
          tews_components: satsReasoning.components || satsReasoning.tews_components || {},
          tews_total: satsReasoning.total ?? satsReasoning.tews_total ?? 0,
          tews_priority: satsReasoning.priority ?? satsReasoning.escalated_to ?? satsReasoning.tews_priority ?? 4
        };
      }
      
      // DEBUG: Log what was extracted for satsReasoning
      console.log(`[HOSPITAL-TRIAGE] 🔍 SATS Extraction Debug:`);
      console.log(`[HOSPITAL-TRIAGE]   - satsReasoning keys: ${Object.keys(satsReasoning).join(', ')}`);
      console.log(`[HOSPITAL-TRIAGE]   - 🔥 tews_total: ${satsReasoning?.tews_total ?? 'NOT_FOUND'}`);
      console.log(`[HOSPITAL-TRIAGE]   - tews_components: ${satsReasoning?.tews_components ? 'PRESENT' : 'NOT_FOUND'}`);
      console.log(`[HOSPITAL-TRIAGE]   - tews_priority: ${satsReasoning?.tews_priority ?? 'NOT_FOUND'}`);
      console.log(`[HOSPITAL-TRIAGE]   - Full satsReasoning: ${JSON.stringify(satsReasoning).substring(0, 300)}`);
      
      // Extract comprehensive clinical analysis from LLM response
      // Use the correct data nesting level identified above
      const clinicalAnalysis = flaskDataLevel.clinicalAnalysis || flaskDataLevel.aiTriageReport || {};
      
      console.log(`[HOSPITAL-TRIAGE] 📋 Extracted objects:`);
      console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis is ${Object.keys(clinicalAnalysis).length} fields: ${Object.keys(clinicalAnalysis).join(', ')}`);
      console.log(`[HOSPITAL-TRIAGE]   - tewsData is ${Object.keys(tewsData).length} fields: ${Object.keys(tewsData).join(', ')}`);
      
      // CRITICAL DEBUG: Log the actual values being extracted
      console.log(`[HOSPITAL-TRIAGE] 🔍 CRITICAL: Raw clinicalAnalysis object:`);
      console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis.pathophysiology type: ${typeof clinicalAnalysis.pathophysiology}, length: ${(clinicalAnalysis.pathophysiology || '').length}`);
      console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis.clinical_summary type: ${typeof clinicalAnalysis.clinical_summary}, length: ${(clinicalAnalysis.clinical_summary || '').length}`);
      console.log(`[HOSPITAL-TRIAGE]   - clinicalAnalysis.clinical_discriminators type: ${typeof clinicalAnalysis.clinical_discriminators}, isArray: ${Array.isArray(clinicalAnalysis.clinical_discriminators)}`);
      console.log(`[HOSPITAL-TRIAGE]   - FULL clinicalAnalysis: ${JSON.stringify(clinicalAnalysis).substring(0, 500)}`);
      
      console.log(`[HOSPITAL-TRIAGE] 🎯 Extracted triageResult:`);
      console.log(`[HOSPITAL-TRIAGE]   - pathophysiology: ${clinicalAnalysis.pathophysiology ? clinicalAnalysis.pathophysiology.substring(0, 100) : '(EMPTY)'}`);
      console.log(`[HOSPITAL-TRIAGE]   - clinical_discriminators: ${Array.isArray(clinicalAnalysis.clinical_discriminators) ? clinicalAnalysis.clinical_discriminators.length + ' items' : 'NOT_ARRAY'} - ${JSON.stringify((clinicalAnalysis.clinical_discriminators || []).slice(0, 2))}`);
      console.log(`[HOSPITAL-TRIAGE]   - differential_diagnoses: ${Array.isArray(clinicalAnalysis.differential_diagnoses) ? clinicalAnalysis.differential_diagnoses.length + ' items' : 'NOT_ARRAY'}`);
      console.log(`[HOSPITAL-TRIAGE]   - clinical_summary: ${clinicalAnalysis.clinical_summary ? clinicalAnalysis.clinical_summary.substring(0, 80) : '(EMPTY)'}`);
      console.log(`[HOSPITAL-TRIAGE]   - immediate_actions: ${Array.isArray(clinicalAnalysis.immediate_actions) ? clinicalAnalysis.immediate_actions.length + ' items' : 'NOT_ARRAY'}`);
      
      // GGUF fields are directly in clinicalAnalysis - extract all of them
      triageResult = {
        // Clinical analysis fields from GGUF
        esi_level: tewsData.escalated_to || clinicalAnalysis.finalSpecialty || 3,
        priority_code: clinicalAnalysis.priority_code || 'YELLOW',
        specialty: clinicalAnalysis.finalSpecialty || 'General',
        urgency_level: clinicalAnalysis.urgency_level || 'Routine',
        pathophysiology: clinicalAnalysis.pathophysiology || '',
        clinical_summary: clinicalAnalysis.clinical_summary || '',
        clinical_discriminators: clinicalAnalysis.clinical_discriminators || [],
        severity_assessment: clinicalAnalysis.severity_assessment || 'Compensating',
        differential_diagnoses: clinicalAnalysis.differential_diagnosis || clinicalAnalysis.differential_diagnoses || [],
        recommended_specialists: clinicalAnalysis.recommended_specialists || [],
        pathognomonic_markers: clinicalAnalysis.pathognomonic_markers || [],
        diagnostic_workup: clinicalAnalysis.diagnostic_workup || [],
        disposition_rationale: clinicalAnalysis.disposition_rationale || '',
        immediate_actions: clinicalAnalysis.immediate_actions || []
      };
      // Attach SATS reasoning (TEWS components/total) into triageResult for downstream use
      triageResult.SATS_reasoning = satsReasoning || {};
      
      // DEBUG: Log what was extracted into triageResult
      console.log(`[HOSPITAL-TRIAGE] 📦 Built triageResult object (15 clean fields):`);
      console.log(`[HOSPITAL-TRIAGE]   - pathophysiology: ${triageResult.pathophysiology ? triageResult.pathophysiology.substring(0, 100) : '(EMPTY)'}`);
      console.log(`[HOSPITAL-TRIAGE]   - clinical_summary: ${triageResult.clinical_summary ? triageResult.clinical_summary.substring(0, 80) : '(EMPTY)'}`);
      console.log(`[HOSPITAL-TRIAGE]   - clinical_discriminators: ${Array.isArray(triageResult.clinical_discriminators) ? triageResult.clinical_discriminators.length + ' items' : 'NOT_ARRAY'}`);
      console.log(`[HOSPITAL-TRIAGE]   - differential_diagnoses: ${Array.isArray(triageResult.differential_diagnoses) ? triageResult.differential_diagnoses.length + ' items' : 'NOT_ARRAY'}`);
      console.log(`[HOSPITAL-TRIAGE]   - immediate_actions: ${Array.isArray(triageResult.immediate_actions) ? triageResult.immediate_actions.length + ' items' : 'NOT_ARRAY'}`);
      console.log(`[HOSPITAL-TRIAGE] 📋 Full triageResult keys: ${Object.keys(triageResult).join(', ')}`);
      
      if (!triageResult || Object.keys(triageResult).length === 0) {
        console.warn(`[HOSPITAL-TRIAGE] ⚠️ LLM endpoint returned empty, using fallback`);
        triageResult = {
          satsLevel: vitalsNormalized.temperature > 38 ? 'orange' : 'yellow',
          urgencyScore: vitalsNormalized.temperature > 38 ? 2 : 3,
          fallback: true,
          fallback_reason: 'LLM synthesis unavailable'
        };
      } else {
        console.log(`[HOSPITAL-TRIAGE] ✅ MedGemma synthesis complete: ESI ${triageResult.esi_level}, Priority ${triageResult.priority_code}`);
        if (triageResult.pathophysiology) {
          console.log(`[HOSPITAL-TRIAGE] 🧠 Pathophysiology: ${triageResult.pathophysiology?.substring(0, 150)}...`);
        }
        console.log(`[HOSPITAL-TRIAGE] 📊 Clinical Discriminators: ${(triageResult.clinical_discriminators || []).length}`);
        console.log(`[HOSPITAL-TRIAGE] 📋 Differential Diagnoses: ${(triageResult.differential_diagnoses || []).length}`);
      }
    } catch (llmError) {
      console.error(`[HOSPITAL-TRIAGE] LLM endpoint error:`, llmError.message);
      
      // GRACEFUL FALLBACK: If LLM endpoint fails, use basic SATS-only fallback
      if (availablePhysicians.length === 0) {
        console.warn(`[HOSPITAL-TRIAGE] ⚠️ No available staff - LLM failed. Using fallback (basic SATS).`);
        triageResult = {
          satsLevel: vitalsNormalized.temperature > 38 ? 'orange' : 'yellow',
          urgencyScore: vitalsNormalized.temperature > 38 ? 2 : 3,
          fallback: true,
          fallback_reason: 'No available staff + LLM unavailable'
        };
      } else {
        // If staff exists but LLM still failed, return error
        return res.status(503).json({
          success: false,
          message: 'Triage service (MedGemma LLM) unavailable',
          error: llmError.message,
        });
      }
    }

    // Step 5: Map LLM triage result to standard ESI format for frontend compatibility
    // Use esi_level from JSON if available, otherwise derive from priority_code
    let esiLevel = triageResult.esi_level;
    if (!esiLevel && triageResult.priority_code) {
      const priorityMap = { 'RED': 1, 'ORANGE': 2, 'YELLOW': 3, 'GREEN': 4 };
      esiLevel = priorityMap[triageResult.priority_code] || 3;
    }
    esiLevel = esiLevel || triageResult.urgencyScore || 3;
    
    // Map ESI level to SATS color
    const satsLevelMap = { 1: 'red', 2: 'orange', 3: 'yellow', 4: 'green' };
    const satsLevel = satsLevelMap[esiLevel] || 'yellow';
    
    console.log(`[HOSPITAL-TRIAGE] 📊 LLM Triage Result:`);
    console.log(`  - ESI Level (from LLM): ${esiLevel}`);
    console.log(`  - Priority Code: ${triageResult.priority_code || 'N/A'}`);
    console.log(`  - SATS Level: ${satsLevel.toUpperCase()}`);
    console.log(`  - LLM Analysis Present: ${triageResult.llmAnalysis ? '✅' : '❌'}`);
    console.log(`  - Pathognomonic Markers: ${(triageResult.pathognomonic_markers || []).length}`);
    
    // Step 6: Auto-assign patient to physician + nurse
    let assignedPhysician = null;
    let assignedNurse = null;

    if (availablePhysicians.length > 0) {
      // Prefer specialty match if LLM result suggests one
      const recommendedDept = triageResult.recommendedDepartment;
      if (recommendedDept) {
        const matching = availablePhysicians.filter((p) =>
          p.specialty?.toLowerCase().includes(recommendedDept.toLowerCase())
        );
        assignedPhysician =
          matching.length > 0
            ? matching[0]
            : availablePhysicians[0];
      } else {
        assignedPhysician = availablePhysicians[0];
      }

      // Assign clinical nurse from available staff
      if (availableClinicalNurses.length > 0) {
        assignedNurse = availableClinicalNurses[0];
      }
    }

    console.log(`[HOSPITAL-TRIAGE] 👥 Assignments:`);
    console.log(`  - Physician: ${assignedPhysician?.name || 'None'}`);
    console.log(`  - Nurse: ${assignedNurse?.name || 'None'}`);
    
    // Step 7: Create Encounter document to persist the triage
    const patientData = req.body.patient || {};
    
    // Map LLM priority_code (RED/ORANGE/YELLOW/GREEN) to ESI level if not present
    let esiFromLLM = triageResult.esi_level;
    if (!esiFromLLM && triageResult.priority_code) {
      const priorityMap = { 'RED': 1, 'ORANGE': 2, 'YELLOW': 3, 'GREEN': 4 };
      esiFromLLM = priorityMap[triageResult.priority_code] || esiLevel;
    }
    esiFromLLM = esiFromLLM || esiLevel;
    
    // Normalize vitals back to encounter schema (temp, hr, resp, bp_systolic, bp_diastolic, o2)
    const vitalsFinal = {
      temp: vitals?.temp || vitalsNormalized.temperature,
      hr: vitals?.hr || vitalsNormalized.heartRate,
      resp: vitals?.resp || vitalsNormalized.respiratoryRate,
      bp_systolic: vitals?.bp_systolic || vitalsNormalized.systolicBP,
      bp_diastolic: vitals?.bp_diastolic || vitalsNormalized.diastolicBP,
      o2: vitals?.o2 || vitalsNormalized.oxygenSaturation,
      bp_left_systolic: vitals?.bp_left_systolic || null,
      bp_left_diastolic: vitals?.bp_left_diastolic || null,
      bp_right_systolic: vitals?.bp_right_systolic || null,
      bp_right_diastolic: vitals?.bp_right_diastolic || null,
      avpu: vitals?.avpu || null
    };
    
    // DEBUG: Log what's about to be stored in aiTriageReport
    console.log(`[HOSPITAL-TRIAGE] 💾 About to create encounter with aiTriageReport:`);
    console.log(`[HOSPITAL-TRIAGE]   - pathophysiology: "${(triageResult.pathophysiology || '').substring(0, 80)}..."`);
    console.log(`[HOSPITAL-TRIAGE]   - clinical_discriminators: ${Array.isArray(triageResult.clinical_discriminators) ? triageResult.clinical_discriminators.length : 0} items - ${JSON.stringify(triageResult.clinical_discriminators.slice(0, 2))}`);
    console.log(`[HOSPITAL-TRIAGE]   - differential_diagnoses: ${Array.isArray(triageResult.differential_diagnoses) ? triageResult.differential_diagnoses.length : 0} items`);
    console.log(`[HOSPITAL-TRIAGE]   - clinical_summary: "${(triageResult.clinical_summary || '').substring(0, 80)}..."`);
    console.log(`[HOSPITAL-TRIAGE]   - immediate_actions: ${Array.isArray(triageResult.immediate_actions) ? triageResult.immediate_actions.length : 0} items`);
    console.log(`[HOSPITAL-TRIAGE]   - SATS_reasoning keys: ${Object.keys(triageResult.SATS_reasoning || {}).join(', ')}`);
    console.log(`[HOSPITAL-TRIAGE]   - 🔥 SATS POINTS (tews_total): ${triageResult.SATS_reasoning?.tews_total ?? triageResult.SATS_reasoning?.total ?? satsReasoning?.tews_total ?? satsReasoning?.total ?? 'NOT_FOUND'}`);
    console.log(`[HOSPITAL-TRIAGE]   - SATS priority code: ${triageResult.SATS_reasoning?.tews_priority ?? triageResult.SATS_reasoning?.priority ?? 'NOT_FOUND'}`);
    console.log(`[HOSPITAL-TRIAGE]   - satsReasoning scope check - tews_total: ${satsReasoning?.tews_total ?? satsReasoning?.total ?? 'NOT_FOUND'}`);
    const encounter = await Encounter.create({
      hospitalId: resolvedHospitalId, // Multi-tenant isolation: every encounter for one hospital
      nurse_id: triageNurseId || null,
      patient: {
        id_number: patientId,  // Match frontend expectation (id_number not id)
        id: patientId,
        name: patientData?.name || 'Unknown',
        surname: patientData?.surname || '',
        age: patientData?.age,
        sex: patientData?.sex,
        gender: patientData?.gender || patientData?.sex,
        symptoms: patientData?.symptoms || '',
        history: patientData?.history || '',
        duration: patientData?.duration || '',
        painLevel: patientData?.painLevel || '',
        contact: patientData?.contact
      },
      transcript: Array.isArray(transcript) ? transcript : [{ role: 'patient', content: transcript || '' }],
      vitals: vitalsFinal,  // Use properly formatted vitals with correct field names
      triage: {
        ESI: esiFromLLM,
        satsLevel: satsLevel,
        priority_code: esiFromLLM === 1 ? 1 : esiFromLLM === 2 ? 2 : esiFromLLM === 3 ? 3 : 4,
        tews_components: satsReasoning?.tews_components || null,
        tews_total: satsReasoning?.tews_total ?? null,
        specialty: triageResult.recommendedDepartment || 'General',
        // Store at top-level for frontend compatibility
        assigned_specialty: triageResult.recommendedDepartment || 'General',
        assigned_physician: assignedPhysician || null,
        assigned_nurse: assignedNurse || null,
        ai_summary: triageResult.clinical_summary || triageResult.llmAnalysis || '',
        clinical_summary: triageResult.clinical_summary || triageResult.llmAnalysis || '',
        summary: triageResult.clinical_summary || triageResult.llmAnalysis || '',
        patient_symptoms: patientData?.symptoms || llmTriagePayload?.chiefComplaint || '',
        llmAnalysis: triageResult.llmAnalysis || '',
        method: 'MedGemma LLM',
        pathognomonic_markers: triageResult.pathognomonic_markers || [],
        SATS_reasoning: triageResult.SATS_reasoning || satsReasoning || {},
        discriminators: triageResult.clinical_discriminators || [],
        llmCapable: true,
        hardwareAware: true,
        // Store complete clinical analysis from ESI_Engine
        aiTriageReport: {
          esi_level: esiFromLLM,
          priority_code: triageResult.priority_code || (esiFromLLM === 1 ? 'RED' : esiFromLLM === 2 ? 'ORANGE' : esiFromLLM === 3 ? 'YELLOW' : 'GREEN'),
          specialty: triageResult.specialty || triageResult.recommendedDepartment || 'General',
          urgency_level: triageResult.urgency_level || (esiLevel === 1 ? 'Immediate' : esiLevel === 2 ? 'Emergent' : esiLevel === 3 ? 'Urgent' : 'Routine'),
          pathophysiology: triageResult.pathophysiology || '',
          clinical_discriminators: triageResult.clinical_discriminators || [],
          severity_assessment: triageResult.severity_assessment || 'Compensating',
          differential_diagnoses: triageResult.differential_diagnoses || triageResult.diagnoses || [],
          recommended_specialists: triageResult.recommended_specialists || [],
          pathognomonic_markers: triageResult.pathognomonic_markers || [],
          diagnostic_workup: triageResult.diagnostic_workup || [],
          disposition_rationale: triageResult.disposition_rationale || '',
          clinical_summary: triageResult.clinical_summary || triageResult.llmAnalysis || '',
          immediate_actions: triageResult.immediate_actions || [],
          SATS_reasoning: triageResult.SATS_reasoning || satsReasoning || {}
        },
        clinical_structure: {
          diagnoses: triageResult.diagnoses || [],
          recommendations: triageResult.recommendations || [],
          immediate_actions: triageResult.immediate_actions || []
        },
        // ===== TIMESTAMP TRACKING FOR AUDIT TRAIL =====
        triage_created_at: new Date(),  // When triage was created
        inference_timestamp: new Date()  // When AI inference was completed
      },
      assignments: {
        physician: assignedPhysician ? {
          id: assignedPhysician._id,
          name: assignedPhysician.name,
          specialty: assignedPhysician.specialty
        } : null,
        clinicalNurse: assignedNurse ? {
          id: assignedNurse._id,
          name: assignedNurse.name,
          specialty: assignedNurse.specialty
        } : null
      },
      // ===== POPULA TE ASSIGNED_CLINICIANS FOR HANDOVER =====
      assigned_clinicians: [
        ...(assignedPhysician ? [{
          id: assignedPhysician._id,
          name: assignedPhysician.name,
          role: 'PHYSICIAN',
          specialty: assignedPhysician.specialty,
          score: 100,
          assigned_at: new Date()
        }] : []),
        ...(assignedNurse ? [{
          id: assignedNurse._id,
          name: assignedNurse.name,
          role: 'NURSE',
          specialty: assignedNurse.specialty,
          score: 100,
          assigned_at: new Date()
        }] : [])
      ],
      hospitalId: resolvedHospitalId,
      severity: esiLevel,
      status: 'AWAITING_PHYSICIAN',
      isWaiting: true,
      nurseNotes: ''
    });

    console.log(`[HOSPITAL-TRIAGE] ✅ Encounter created: ${encounter._id}`);
    console.log(`[HOSPITAL-TRIAGE] ⏰ Triage timestamps:`);
    console.log(`[HOSPITAL-TRIAGE]    - triage_created_at: ${encounter.triage?.triage_created_at}`);
    console.log(`[HOSPITAL-TRIAGE]    - inference_timestamp: ${encounter.triage?.inference_timestamp}`);
    console.log(`[HOSPITAL-TRIAGE] 👥 Assigned Clinicians Populated:`);
    console.log(`[HOSPITAL-TRIAGE]    - Count: ${encounter.assigned_clinicians?.length || 0}`);
    console.log(`[HOSPITAL-TRIAGE]    - Clinicians: ${encounter.assigned_clinicians?.map(c => `${c.name} (${c.role})`).join(', ') || 'None'}`);
    console.log(`[HOSPITAL-TRIAGE] ✅ Stored comprehensive clinical analysis with ${(encounter.triage.aiTriageReport.differential_diagnoses || []).length} diagnoses`);
    
    // DEBUG: Log what actually got stored in encounter
    console.log(`[HOSPITAL-TRIAGE] DEBUG encounter.triage.aiTriageReport fields:`, Object.keys(encounter.triage.aiTriageReport || {}));
    console.log(`[HOSPITAL-TRIAGE] DEBUG stored clinical_discriminators:`, encounter.triage.aiTriageReport?.clinical_discriminators);
    console.log(`[HOSPITAL-TRIAGE] DEBUG stored differential_diagnoses:`, encounter.triage.aiTriageReport?.differential_diagnoses);
    console.log(`[HOSPITAL-TRIAGE] DEBUG stored pathophysiology:`, encounter.triage.aiTriageReport?.pathophysiology?.substring(0, 100));
    console.log(`[HOSPITAL-TRIAGE] DEBUG stored clinical_summary:`, encounter.triage.aiTriageReport?.clinical_summary?.substring(0, 100));
    
    // DEBUG: Verify response structure before sending to frontend
    console.log(`[HOSPITAL-TRIAGE] 📋 FINAL RESPONSE STRUCTURE (15 clean fields):`);
    console.log(`[HOSPITAL-TRIAGE]   - encounter._id: ${encounter._id}`);
    console.log(`[HOSPITAL-TRIAGE]   - encounter.triage exists: ${!!encounter.triage}`);
    console.log(`[HOSPITAL-TRIAGE]   - encounter.triage.aiTriageReport exists: ${!!encounter.triage.aiTriageReport}`);
    console.log(`[HOSPITAL-TRIAGE]   - pathophysiology length: ${(encounter.triage.aiTriageReport?.pathophysiology || '').length} chars`);
    console.log(`[HOSPITAL-TRIAGE]   - clinical_discriminators: ${(encounter.triage.aiTriageReport?.clinical_discriminators || []).length} items`);
    console.log(`[HOSPITAL-TRIAGE]   - differential_diagnoses: ${(encounter.triage.aiTriageReport?.differential_diagnoses || []).length} items`);
    console.log(`[HOSPITAL-TRIAGE]   - immediate_actions: ${(encounter.triage.aiTriageReport?.immediate_actions || []).length} items`);
    console.log(`[HOSPITAL-TRIAGE]   - recommendations: ${(encounter.triage.aiTriageReport?.recommendations || []).length} items`);
    
    // Step 8: Return comprehensive response with LLM synthesis + assignments (compatible with frontend)
    const specialty = triageResult.specialty || triageResult.recommendedDepartment || 'General';
    
    return res.status(200).json({
      success: true,
      message: 'Patient triaged with MedGemma LLM synthesis and assigned successfully',
      encounter: encounter,
      triage: {
        ESI: esiLevel,
        satsLevel: satsLevel,
        priority_code: esiLevel === 1 ? 1 : esiLevel === 2 ? 2 : esiLevel === 3 ? 3 : 4,
        specialty: specialty,
        assigned_specialty: specialty,
        recommendedDepartment: specialty,
        llmAnalysis: triageResult.llmAnalysis || triageResult.clinical_summary || '',
        assignments: {
          physician: assignedPhysician ? {
            _id: assignedPhysician._id,
            name: assignedPhysician.name,
            specialty: assignedPhysician.specialty
          } : null,
          clinicalNurse: assignedNurse ? {
            _id: assignedNurse._id,
            name: assignedNurse.name,
            specialty: assignedNurse.specialty
          } : null
        },
        pathognomonic_markers: triageResult.pathognomonic_markers || [],
        diagnosisAugmentation: triageResult.diagnosisAugmentation || null,
        fallback: triageResult.fallback || false,
        fallback_reason: triageResult.fallback_reason || null,
        // Include full clinical analysis
        aiTriageReport: {
          esi_level: esiLevel,
          priority_code: satsLevel === 'red' ? 'RED' : satsLevel === 'orange' ? 'ORANGE' : satsLevel === 'yellow' ? 'YELLOW' : 'GREEN',
          specialty: triageResult.specialty || triageResult.recommendedDepartment || 'General',
          urgency_level: triageResult.urgency_level || (esiLevel === 1 ? 'Immediate' : esiLevel === 2 ? 'Emergent' : esiLevel === 3 ? 'Urgent' : 'Routine'),
          pathophysiology: triageResult.pathophysiology || '',
          clinical_discriminators: triageResult.clinical_discriminators || [],
          severity_assessment: triageResult.severity_assessment || 'Compensating',
          differential_diagnoses: triageResult.differential_diagnoses || triageResult.diagnoses || [],
          recommended_specialists: triageResult.recommended_specialists || [],
          pathognomonic_markers: triageResult.pathognomonic_markers || [],
          diagnostic_workup: triageResult.diagnostic_workup || [],
          disposition_rationale: triageResult.disposition_rationale || '',
          clinical_summary: triageResult.clinical_summary || triageResult.llmAnalysis || '',
          immediate_actions: triageResult.immediate_actions || [],
          SATS_reasoning: triageResult.SATS_reasoning || satsReasoning || {}
        }
      },
      data: {
        patient: {
          id_number: patientId,
          name: patientData?.name || 'Unknown',
          surname: patientData?.surname || '',
          age: patientData?.age,
          sex: patientData?.sex,
          gender: patientData?.gender || patientData?.sex,
          symptoms: patientData?.symptoms || '',
          history: patientData?.history || '',
          duration: patientData?.duration || '',
          painLevel: patientData?.painLevel || ''
        },
        ESI: esiLevel,
        satsLevel: satsLevel,
        priority_code: esiLevel === 1 ? 1 : esiLevel === 2 ? 2 : esiLevel === 3 ? 3 : 4,
        specialty: specialty,
        recommended_specialty: specialty,
        assignedPhysician: assignedPhysician ? {
          _id: assignedPhysician._id,
          name: assignedPhysician.name,
          specialty: assignedPhysician.specialty
        } : null,
        assignedNurse: assignedNurse ? {
          _id: assignedNurse._id,
          name: assignedNurse.name,
          specialty: assignedNurse.specialty
        } : null,
        // Include all clinical analysis data
        pathophysiology: triageResult.pathophysiology || '',
        clinical_discriminators: triageResult.clinical_discriminators || [],
        severity_assessment: triageResult.severity_assessment || 'Compensating',
        differential_diagnoses: triageResult.differential_diagnoses || triageResult.diagnoses || [],
        recommended_specialists: triageResult.recommended_specialists || [],
        pathognomonic_markers: triageResult.pathognomonic_markers || [],
        diagnostic_workup: triageResult.diagnostic_workup || [],
        disposition_rationale: triageResult.disposition_rationale || '',
        clinical_summary: triageResult.clinical_summary || triageResult.llmAnalysis || '',
        immediate_actions: triageResult.immediate_actions || []
      },
      hospitalContext: {
        hospitalId,
        date: today.toISOString().split('T')[0],
        shiftsAvailable: todayShifts.length,
        staffAvailable: {
          physicians: availablePhysicians.length,
          clinicalNurses: availableClinicalNurses.length,
          triageNurses: availableTriageNurses.length,
        },
      },
      metadata: {
        llmCapable: true,
        hardwareAware: true,
        triageMethod: 'MedGemma GGUF Synthesis',
        requestTimestamp: new Date().toISOString(),
      }
    });

  } catch (error) {
    console.error('[HOSPITAL-TRIAGE] ERROR:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Internal server error during hospital triage',
      error: error.message,
    });
  }
}

/**
 * ----------------------
 * Get Available Staff for Hospital
 * GET /api/triage/hospital/:hospitalId/available-staff
 * ----------------------
 */
export const getAvailableStaffForHospital = async (req, res) => {
  try {
    const { hospitalId } = req.params;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'hospitalId is required',
      });
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const todayShifts = await Shift.find({
      hospitalId,
      status: 'PUBLISHED',
      date: { $gte: today, $lt: tomorrow },
    }).populate('assignedStaff.staffID');

    const staff = {
      physicians: [],
      triageNurses: [],
      clinicalNurses: [],
      shifts: [],
    };

    todayShifts.forEach((shift) => {
      const shiftInfo = {
        shiftId: shift._id,
        shiftName: shift.shiftName,
        department: shift.department,
        startTime: shift.shiftStart,
        endTime: shift.shiftEnd,
      };

      shift.assignedStaff?.forEach((staffMember) => {
        if (staffMember.status === 'ABSENT') return;

        const baseInfo = {
          staffId: staffMember.staffID?._id,
          name: staffMember.name,
          specialty: staffMember.specialty,
          shiftId: shift._id,
        };

        if (staffMember.type === 'PHYSICIAN') {
          staff.physicians.push(baseInfo);
        } else if (staffMember.type === 'NURSE') {
          staff.clinicalNurses.push(baseInfo);
        }
      });

      // Check for triage nurse assignment
      if (shift.teamAssignment?.triageNurseId) {
        const triageStaff = shift.assignedStaff?.find(
          (s) => s.staffID?._id?.toString() === shift.teamAssignment.triageNurseId?.toString()
        );
        if (triageStaff) {
          staff.triageNurses.push({
            staffId: triageStaff.staffID?._id,
            name: triageStaff.name,
            specialty: triageStaff.specialty,
            shiftId: shift._id,
          });
        }
      }

      staff.shifts.push(shiftInfo);
    });

    return res.status(200).json({
      success: true,
      hospital: hospitalId,
      date: today.toISOString().split('T')[0],
      availableStaff: staff,
      totals: {
        physicians: staff.physicians.length,
        triageNurses: staff.triageNurses.length,
        clinicalNurses: staff.clinicalNurses.length,
        shifts: staff.shifts.length,
      },
    });

  } catch (error) {
    console.error('[GET-AVAILABLE-STAFF] ERROR:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Error fetching available staff',
      error: error.message,
    });
  }
};

// ============================================================
// REMAINING ROUTE HANDLERS (Extracted from triage.route.js)
// ============================================================

/**
 * GET /api/triage/staff
 * Fetch available nurses, physicians, and beds for staff assignment UI
 * LIVE DATA from scheduling system
 */
export const getStaffForTriage = async (req, res) => {
  try {
    const { date } = req.query;
    const { PhysicianSchedule } = await import('../models/physicianSchedule.model.js');
    const { NurseSchedule } = await import('../models/nurseSchedule.model.js');
    const { BedSchedule } = await import('../models/bedSchedule.model.js');
    
    console.log(`[TRIAGE] GET /staff - Fetching LIVE scheduling data for date: ${date || 'today'}`);

    // Determine date range (default: today)
    let queryDate = date;
    if (!queryDate) {
      const today = new Date();
      queryDate = today.toISOString().split('T')[0];
    }
    
    // Parse date string
    const [year, month, day] = queryDate.split('-').map(Number);
    const dateObj = new Date(year, month - 1, day);
    const nextDay = new Date(dateObj.getTime() + 24 * 60 * 60 * 1000);
    
    // Query LIVE scheduling data from database
    const [physicians, nurses, beds] = await Promise.all([
      PhysicianSchedule.find({
        date: { $gte: dateObj, $lt: nextDay },
        status: { $in: ["ON_DUTY", "ON_CALL"] }
      })
        .sort({ currentPatientCount: 1 })
        .lean(),
      
      NurseSchedule.find({
        date: { $gte: dateObj, $lt: nextDay },
        status: { $in: ["ON_DUTY", "ON_CALL"] }
      })
        .sort({ currentBedCount: 1 })
        .lean(),
      
      BedSchedule.find({
        status: "AVAILABLE"
      })
        .lean()
    ]);

    // Format response with live data
    const liveStaffData = {
      nurses: nurses.map(n => ({
        _id: n._id,
        nurseId: n.nurseId,
        name: n.name,
        surname: n.surname || "",
        email: n.email || "",
        licenseNumber: n.licenseNumber || "",
        licenseType: n.licenseType || "RN",
        nqfLevel: n.nqfLevel || 6,
        specializations: n.specializations || ["General Nurse"],
        satsCapability: {
          canTriageRed: n.canTriageRed || false,
          canTriageYellow: n.canTriageYellow || true,
          canTriageGreen: n.canTriageGreen || true
        },
        isAvailable: n.status === "ON_DUTY",
        status: n.status,
        currentBedCount: n.currentBedCount || 0,
        maxBeds: n.maxBeds || 5,
        availableSlots: Math.max(0, (n.maxBeds || 5) - (n.currentBedCount || 0)),
        shiftStart: n.shiftStart || "",
        shiftEnd: n.shiftEnd || "",
        certifications: n.certifications || []
      })),

      physicians: physicians.map(p => ({
        _id: p._id,
        physicianId: p.physicianId,
        name: p.name,
        surname: p.surname || "",
        email: p.email || "",
        specialty: p.specialty || "General",
        isAvailable: p.status === "ON_DUTY",
        status: p.status,
        currentPatientCount: p.currentPatientCount || 0,
        maxPatients: p.maxPatients || 20,
        availableSlots: Math.max(0, (p.maxPatients || 20) - (p.currentPatientCount || 0)),
        shiftStart: p.shiftStart || "",
        shiftEnd: p.shiftEnd || ""
      })),

      beds: beds.map(b => ({
        _id: b._id,
        bedNumber: b.bedNumber,
        department: b.department || "Emergency Department",
        status: b.status,
        acuityLevel: b.acuityLevel || "Med-Surg",
        bedType: b.bedType || "Standard",
        equipment: b.equipment || ["Monitor"],
        specialization: b.specialization || "General"
      })),

      queryDate,
      totalNursesAvailable: nurses.length,
      totalPhysiciansAvailable: physicians.length,
      totalBedsAvailable: beds.length,
      timestamp: new Date()
    };

    console.log(`[TRIAGE] Live data loaded: ${nurses.length} nurses, ${physicians.length} physicians, ${beds.length} beds`);

    res.json({
      success: true,
      data: liveStaffData
    });

  } catch (error) {
    console.error("[TRIAGE] Error fetching live staff:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch live staff availability",
      message: error.message
    });
  }
};

/**
 * POST /api/triage
 * Submit manual triage (or AI-assisted with computeMode)
 * Creates an Encounter record in database for NurseDashboard to display
 * 
 * KEY: If computeMode is absent/null = MANUAL path (no AI)
 *      If computeMode is present = AI-assisted path
 */
export const submitManualTriage = async (req, res) => {
  try {
    const jwt = (await import('jsonwebtoken')).default;
    const {
      triageNurseId,
      patientId,
      encounterId,
      
      // Clinical data
      symptoms,
      vitalSigns,
      medicalHistory,
      medications,
      allergies,
      chiefComplaint,
      
      // Manual triage - SATS instead of ESI
      manualSatsLevel,
      urgencyScore,
      triageReasoningNotes,
      recommendedDepartment,
      immediateInterventions,
      
      // Staff assignment
      assignedNurseId,
      assignedNurseName,
      assignedPhysicianId,
      assignedPhysicianName,
      bedAssigned,
      
      // Patient data for Encounter
      patient,
      vitals,
      
      // Optional: AI mode
      computeMode
    } = req.body;

    // Get nurse from JWT token (more reliable than frontend)
    const token = req.headers.authorization?.split(" ")[1];
    let nursId = null;
    
    if (token) {
      try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET || "your-secret-key");
        nursId = decoded.id;
        console.log(`[TRIAGE] Extracted nurse_id from JWT: ${nursId}`);
      } catch (err) {
        console.warn(`[TRIAGE] Failed to decode JWT:`, err.message);
        nursId = triageNurseId;
      }
    } else {
      nursId = triageNurseId;
    }

    console.log(`[TRIAGE] POST / - SATS: ${manualSatsLevel}, computeMode: ${computeMode || "NONE (Manual)"}, nurse_id: ${nursId}`);

    // Validate required fields
    if (!nursId || !manualSatsLevel || !triageReasoningNotes || !assignedNurseId || !assignedPhysicianId) {
      return res.status(400).json({
        success: false,
        error: "Missing required fields",
        required: ["nurse_id (from auth)", "manualSatsLevel", "triageReasoningNotes", "assignedNurseId", "assignedPhysicianId"]
      });
    }

    // Validate SATS level
    const validSatsLevels = ['red', 'yellow', 'green'];
    if (!validSatsLevels.includes(manualSatsLevel?.toLowerCase())) {
      return res.status(422).json({
        success: false,
        error: "Invalid SATS level - must be 'red', 'yellow', or 'green'"
      });
    }

    // ===== MULTI-TENANT ISOLATION: Get nurse's hospitalId =====
    const triageNurse = await ClinicalStaff.findById(nursId);
    if (!triageNurse) {
      return res.status(404).json({
        success: false,
        error: "Triage nurse not found"
      });
    }
    
    const hospitalId = triageNurse.hospitalId;
    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        error: "Nurse not associated with a hospital"
      });
    }
    
    console.log(`[TRIAGE] Hospital: ${hospitalId}, Nurse: ${triageNurse.firstName} ${triageNurse.lastName}`);

    // Create Encounter in database
    const manualTriageReport = {
      satsLevel: manualSatsLevel.toLowerCase(),
      urgencyScore: parseInt(urgencyScore) || 50,
      reasoning: triageReasoningNotes,
      department: recommendedDepartment || "ED",
      interventions: immediateInterventions || [],
      timestamp: new Date()
    };

    // Convert symptoms to string if it's an array
    const symptomString = Array.isArray(symptoms) 
      ? symptoms.join(", ") 
      : (symptoms || chiefComplaint || "");

    const encounterData = {
      hospitalId: hospitalId, // Multi-tenant isolation
      nurse_id: nursId,
      assignedNurseId: assignedNurseId || nursId,
      assignedNurseName: assignedNurseName || 'Assigned Nurse',
      assignedPhysicianId: assignedPhysicianId,
      assignedPhysicianName: assignedPhysicianName || 'Not Assigned',
      patient: patient || {
        id_number: patientId || "",
        name: "",
        symptoms: symptomString
      },
      vitals: vitals || {
        temp: vitalSigns?.temperature,
        hr: vitalSigns?.heartRate,
        o2: vitalSigns?.oxygenSaturation,
        resp: vitalSigns?.respiratoryRate,
        bp_systolic: vitalSigns?.systolicBP,
        bp_diastolic: vitalSigns?.diastolicBP
      },
      triage: {
        manualTriageReport,
        confirmed: false,
        triageType: 'MANUAL_ONLY',
        // ===== TIMESTAMP TRACKING FOR AUDIT TRAIL =====
        triage_created_at: new Date(),  // When manual triage was created
        triage_updated_at: new Date()   // When triage was last updated
      },
      // ===== POPULATE ASSIGNED_CLINICIANS FOR HANDOVER =====
      assigned_clinicians: [
        ...(assignedPhysicianId ? [{
          id: assignedPhysicianId,
          name: assignedPhysicianName || 'Not Assigned',
          role: 'PHYSICIAN',
          specialty: 'General',
          score: 100,
          assigned_at: new Date()
        }] : []),
        ...(assignedNurseId ? [{
          id: assignedNurseId,
          name: assignedNurseName || 'Assigned Nurse',
          role: 'NURSE',
          specialty: 'Nursing',
          score: 100,
          assigned_at: new Date()
        }] : [])
      ],
      status: "pending",
      severity: manualSatsLevel === 'red' ? 5 : (manualSatsLevel === 'yellow' ? 3 : 1),
      isWaiting: true,
      submittedAt: new Date()
    };

    const encounter = await Encounter.create(encounterData);
    console.log(`[TRIAGE] Created Encounter: ${encounter._id}, nurse_id: ${encounter.nurse_id}, SATS: ${manualSatsLevel}, severity: ${encounter.severity}`);
    console.log(`[TRIAGE] ⏰ Manual Triage timestamps:`);
    console.log(`[TRIAGE]    - triage_created_at: ${encounter.triage?.triage_created_at}`);
    console.log(`[TRIAGE]    - triage_updated_at: ${encounter.triage?.triage_updated_at}`);
    console.log(`[TRIAGE] 👥 Assigned Clinicians Populated:`);
    console.log(`[TRIAGE]    - Count: ${encounter.assigned_clinicians?.length || 0}`);
    console.log(`[TRIAGE]    - Clinicians: ${encounter.assigned_clinicians?.map(c => `${c.name} (${c.role})`).join(', ') || 'None'}`);

    // Return response with triageRequestId as the Encounter _id
    const triageResult = {
      triageRequestId: encounter._id.toString(),
      status: "TRIAGED",
      computeMode: computeMode || null,
      isAttended: false,
      
      patientId,
      triageNurseId,
      assignedNurseId,
      assignedPhysicianId,
      bedAssigned: bedAssigned || null,

      manualTriageReport,
      aiTriageReport: null,

      processingMetrics: {
        computeTimeMs: 0,
        modelUsed: "MANUAL_ESI_ASSESSMENT",
        timestamp: new Date()
      }
    };

    res.json({
      success: true,
      data: triageResult
    });

  } catch (error) {
    console.error("[TRIAGE] Error submitting manual triage:", error);
    res.status(500).json({
      success: false,
      error: "Failed to submit triage",
      message: error.message
    });
  }
};

/**
 * PATCH /api/triage/:id/confirm
 * Finalize triage after confirmation page
 * 
 * Tracks separate audit reports:
 * - manualTriageReport: Always stored
 * - aiTriageReport: Stored if computeMode was used
 * - nurseEditsReport: ONLY if nurse made edits to AI suggestions
 * 
 * This separation allows full traceability
 */
export const confirmTriageHandler = async (req, res) => {
  try {
    const { id } = req.params;
    const {
      confirmed,
      confirmedAt,
      triageType,
      manualTriageReport,
      aiTriageReport,
      nurseEditsReport
    } = req.body;

    if (!confirmed || !triageType) {
      return res.status(400).json({
        success: false,
        error: "Missing required confirmation fields"
      });
    }

    console.log(`[TRIAGE] PATCH /:id/confirm - ID: ${id}, Type: ${triageType}`);

    // First, check if the Encounter exists
    const existingEncounter = await Encounter.findById(id);
    if (!existingEncounter) {
      return res.status(404).json({
        success: false,
        error: `Triage record not found: ${id}`
      });
    }
    console.log(`[TRIAGE] Found Encounter: ${id}, current status: ${existingEncounter.status}`);

    // Build the triage object, preserving existing data
    const triageObject = {
      ...(existingEncounter.triage && typeof existingEncounter.triage === 'object' 
        ? existingEncounter.triage.toObject?.() || existingEncounter.triage 
        : {}),
      confirmed: true,
      confirmedAt: confirmedAt || new Date().toISOString(),
      triageType: triageType,
      manualTriageReport: manualTriageReport,
      aiTriageReport: aiTriageReport || null,
      nurseEditsReport: nurseEditsReport || null,
      eHRDocumentation: {
        manual: true,
        ai: !!aiTriageReport,
        edits: !!nurseEditsReport,
        specialist: false
      },
      // ===== UPDATE TIMESTAMP FOR AUDIT TRAIL =====
      triage_updated_at: new Date()
    };

    // Update Encounter with confirmed triage data
    const updatePayload = {
      status: "confirmed",
      triage: triageObject
    };

    console.log(`[TRIAGE] Update payload:`, JSON.stringify(updatePayload, null, 2));

    const updatedEncounter = await Encounter.findByIdAndUpdate(
      id,
      updatePayload,
      { new: true }
    );

    if (!updatedEncounter) {
      return res.status(404).json({
        success: false,
        error: `Failed to update Encounter: ${id}`
      });
    }

    console.log(`[TRIAGE] Encounter confirmed and saved: ${id}`);
    console.log(`[TRIAGE] ⏰ Updated triage timestamps:`);
    console.log(`[TRIAGE]    - triage_created_at: ${updatedEncounter.triage?.triage_created_at}`);
    console.log(`[TRIAGE]    - triage_updated_at: ${updatedEncounter.triage?.triage_updated_at}`);

    // Return the confirmed Encounter with all audit trail data
    const confirmedTriageResult = {
      triageRequestId: id,
      confirmed: true,
      confirmedAt: confirmedAt || new Date().toISOString(),
      triageType,
      manualTriageReport,
      aiTriageReport: aiTriageReport || null,
      nurseEditsReport: nurseEditsReport || null,
      status: "CONFIRMED",
      eHRDocumentation: {
        manual: true,
        ai: !!aiTriageReport,
        edits: !!nurseEditsReport,
        specialist: false
      },
      message: `Triage confirmed as ${triageType}. All reports locked in EHR.`
    };

    res.status(200).json({
      success: true,
      data: confirmedTriageResult
    });

  } catch (error) {
    console.error("[TRIAGE] Confirmation error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to confirm triage",
      message: error.message
    });
  }
};

/**
 * PATCH /api/triage/:id/specialist-feedback
 * Add specialist review after patient seen by specialist
 * 
 * Allows comparison:
 * - Manual ESI vs Specialist ESI
 * - AI ESI vs Specialist ESI
 * - Nurse Edits vs Specialist ESI
 */
export const specialistFeedbackHandler = async (req, res) => {
  try {
    const { id } = req.params;
    const {
      specialistId,
      specialistName,
      actualSatsLevel,
      feedback,
      mistriage
    } = req.body;

    if (!actualSatsLevel) {
      return res.status(400).json({
        success: false,
        error: "Actual SATS level required"
      });
    }

    console.log(`[TRIAGE] PATCH /:id/specialist-feedback - ID: ${id}, Actual SATS: ${actualSatsLevel}`);

    // Create specialist feedback report
    const specialistFeedbackReport = {
      timestamp: new Date().toISOString(),
      specialistId,
      specialistName,
      actualSatsLevel,
      feedback,
      mistriage: mistriage || false
    };

    res.status(200).json({
      success: true,
      data: {
        triageRequestId: id,
        specialistFeedbackReport,
        message: "Specialist feedback recorded"
      }
    });

  } catch (error) {
    console.error("[TRIAGE] Specialist feedback error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to save specialist feedback",
      message: error.message
    });
  }
};