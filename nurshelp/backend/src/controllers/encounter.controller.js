import mongoose from "mongoose";
import axios from "axios";
import { Encounter } from "../models/encounter.model.js";
import { BedSchedule } from "../models/bedSchedule.model.js";
import { Vital } from "../models/vitals.model.js";
import { ClinicalStaff } from "../models/clinicalstaff.model.js";
import { canAccessEncounter } from "./encounterAccess.js";
import { normalizeMedicationRequest, isMedicationRequestValid } from "../utils/medicationRequestUtils.js";
import { buildDischargeReportData } from "../utils/dischargeReportBuilder.js";
import { buildLongitudinalHistoryContext } from "../utils/longitudinalPatientHistory.js";

/** Safe transcript parser used across functions */
const parseTranscript = (transcript) => {
  if (!transcript) return [];
  if (Array.isArray(transcript)) return transcript;
  if (typeof transcript === "string") {
    try {
      return JSON.parse(transcript.replace(/'/g, '"'));
    } catch (e) {
      // fallback: treat as single text entry
      return [{ id: 1, type: "note", text: transcript }];
    }
  }
  return [];
};

const PYTHON_URL = process.env.PYTHON_URL || "http://localhost:5000/triage";

/** Waiting severity mapping (SA context) */
const calculateWaitingSeverity = (createdAt, currentTime = Date.now()) => {
  if (!createdAt) return 1;
  const minutes = Math.floor((new Date(currentTime) - new Date(createdAt)) / 60000);

  if (minutes >= 180) return 5; // Critical delay
  if (minutes >= 120) return 4; // Long wait
  if (minutes >= 60) return 3;  // Moderate delay
  return 1; // acceptable
};

/** Map AI priority_code (SATS 1..4) to a numeric severity for UI sorting.
 * Keep it simple: higher number = higher clinical urgency for UI.
 * You can change mapping easily here without touching controller logic later.
 */
const mapPriorityToSeverity = (priority_code) => {
  // SATS: 1=RED (most urgent) -> severity 5
  if (!priority_code) return 1;
  switch (Number(priority_code)) {
    case 1: return 5;
    case 2: return 4;
    case 3: return 3;
    case 4: return 1;
    default: return 1;
  }
};

// ----------------------
// Run Triage (calls Python AI)
// ----------------------
export const runTriage = async (req, res) => {
  try {
    const payload = req.body;
    if (!payload?.patient || !payload?.vitals) {
      return res.status(400).json({ message: "Patient and vitals required" });
    }

    const nurseId = req.user?._id || null;
    
    // ===== MULTI-TENANT ISOLATION: Get nurse's hospitalId =====
    let hospitalId = null;
    if (nurseId) {
      const nurse = await ClinicalStaff.findById(nurseId);
      if (nurse && nurse.hospitalId) {
        hospitalId = nurse.hospitalId;
        console.log(`[TRIAGE] Nurse ${nurseId} belongs to hospital ${hospitalId}`);
      } else {
        console.warn(`[TRIAGE] Nurse ${nurseId} has no associated hospital`);
      }
    }
    
    const transcriptArray = parseTranscript(payload.transcript);

    const patientIdentifier = payload.patient?.id_number || payload.patient?._id || payload.patientId;
    const historicalEncounters = patientIdentifier && hospitalId
      ? await Encounter.find({ hospitalId, 'patient.id_number': patientIdentifier }).sort({ submittedAt: -1, createdAt: -1 }).lean()
      : [];
    const longitudinalHistory = buildLongitudinalHistoryContext({
      encounters: historicalEncounters,
      currentEncounter: { ...payload.patient, vitals: payload.vitals, date: new Date().toISOString() },
      maxCharacters: 12000
    });

    // ===== BUILD COMPREHENSIVE PATIENT CLINICAL TEXT =====
    // Collect all available clinical information
    const clinicalParts = [];
    
    // 1. Add transcript first (highest priority)
    const transcriptStr = Array.isArray(transcriptArray) && transcriptArray.length > 0
      ? transcriptArray.map(t => t.text || t).join(". ")
      : "";
    
    if (transcriptStr && transcriptStr.trim() && transcriptStr !== '[]') {
      clinicalParts.push(transcriptStr.trim());
    }
    
    // 2. Add symptoms
    if (payload.patient?.symptoms) {
      let symptomsStr = '';
      if (Array.isArray(payload.patient.symptoms)) {
        symptomsStr = payload.patient.symptoms.filter(s => s && String(s).trim()).join(', ');
      } else if (typeof payload.patient.symptoms === 'string' && payload.patient.symptoms.trim() && payload.patient.symptoms !== '[]') {
        symptomsStr = payload.patient.symptoms.trim();
      }
      if (symptomsStr) {
        clinicalParts.push(`Symptoms: ${symptomsStr}`);
      }
    }
    
    // 3. Add history
    if (payload.patient?.history && payload.patient.history.trim() && payload.patient.history !== '[]') {
      clinicalParts.push(`History: ${payload.patient.history.trim()}`);
    }
    
    // 4. Add duration
    if (payload.patient?.duration && payload.patient.duration.trim() && payload.patient.duration !== '[]') {
      clinicalParts.push(`Duration: ${payload.patient.duration.trim()}`);
    }
    
    // 5. Add pain level
    if (payload.patient?.painLevel && payload.patient.painLevel.trim() && payload.patient.painLevel !== '[]') {
      clinicalParts.push(`Pain Level: ${payload.patient.painLevel.trim()}`);
    }
    
    // Build final comprehensive text
    const comprehensiveClinicialText = clinicalParts.length > 0 
      ? clinicalParts.join(". ")
      : `Patient: ${payload.patient?.name || 'Unknown'}, Age: ${payload.patient?.age || 'Unknown'}`;

    // Build enriched payload for the AI (strict schema per SATS integration)
    const aiPayload = {
      patient: {
        id_number: payload.patient?.id_number || "",
        name: payload.patient?.name || "",
        surname: payload.patient?.surname || "",
        gender: payload.patient?.gender || "",
        age: payload.patient?.age ?? null,
        height_cm: payload.patient?.height_cm ?? null,
        symptoms: comprehensiveClinicialText, // Use comprehensive text for symptoms field
        history: [payload.patient?.history, longitudinalHistory.serialized].filter(Boolean).join("\n\n"),
        duration: payload.patient?.duration || "",
        painLevel: payload.patient?.painLevel || "",
        criticalHistoryStatus: payload.patient?.criticalHistoryStatus || "",
        criticalHistory: payload.patient?.criticalHistory || payload.patient?.critical_history || {}
      },
      vitals: {
        temp: payload.vitals?.temp ?? payload.vitals?.temperature ?? null,
        bp: payload.vitals?.bp ?? payload.vitals?.blood_pressure ?? payload.vitals?.blood_pressure_systolic ? `${payload.vitals?.blood_pressure_systolic}/${payload.vitals?.blood_pressure_diastolic}` : payload.vitals?.bp,
        bp_left_systolic: payload.vitals?.bp_left_systolic ?? null,
        bp_left_diastolic: payload.vitals?.bp_left_diastolic ?? null,
        bp_right_systolic: payload.vitals?.bp_right_systolic ?? null,
        bp_right_diastolic: payload.vitals?.bp_right_diastolic ?? null,
        bp_systolic: payload.vitals?.bp_systolic ?? null,
        bp_diastolic: payload.vitals?.bp_diastolic ?? null,
        hr: payload.vitals?.hr ?? payload.vitals?.heart_rate ?? null,
        o2: payload.vitals?.o2 ?? payload.vitals?.oxygen_saturation ?? null,
        resp: payload.vitals?.resp ?? null,
        avpu: payload.vitals?.avpu ?? null,
        mobility: payload.vitals?.mobility ?? 'Walks',
        traumaPresent: Boolean(payload.vitals?.traumaPresent)
      },
      history: longitudinalHistory.serialized,
      longitudinalHistory: longitudinalHistory.payload,
      transcript: transcriptArray.map(t => t.text).join(" "),
      transcript_array: transcriptArray,
      arrival_time: payload.arrival_time || new Date().toISOString(),
      current_time: new Date().toISOString()
    };

    // Call Python triage service (AI)
    let triage = null;
    try {
      const pythonRes = await axios.post(PYTHON_URL, aiPayload, {
        timeout: 300000,
        headers: { "Content-Type": "application/json" },
      });
      triage = pythonRes?.data;
      if (!triage || typeof triage !== "object") {
        throw new Error("Invalid triage response");
      }

      // If the Python service returned "vitals_parsed" (extracted from transcript),
      // merge those values back into our aiPayload so that the stored encounter
      // reflects the more complete set of vitals (e.g. BP parsed from speech).
      if (triage.vitals_parsed && typeof triage.vitals_parsed === "object") {
        aiPayload.vitals = {
          ...aiPayload.vitals,
          ...triage.vitals_parsed
        };
      }
    } catch (err) {
      console.warn("Python triage call failed:", err.message);
      // fallback minimal triage object (so we still save encounter)
      triage = {
        priority_code: null,
        priority_colour: null,
        target_time_mins: null,
        SATS_reasoning: {},
        ai_summary: typeof aiPayload.transcript === "string" ? aiPayload.transcript.slice(0, 200) : "No summary",
        clinical_structure: {},
        assigned_specialty: null,
        assigned_physician: null,
        ward: null,
        waitingSeverity: calculateWaitingSeverity(aiPayload.arrival_time)
      };
    }

    // SAFETY SANITY CHECK (minimal change):
    // If AI flagged hypoxia but the incoming payload has no respiratory symptoms
    // and the raw O2 is low (<90), treat as suspected invalid reading and avoid forcing RED.
    let finalPriorityCode = triage.priority_code ?? null;
    try {
      const discriminators = (triage?.SATS_reasoning?.discriminators_found) || (triage?.clinical_structure?.discriminators) || [];
      const rawO2 = aiPayload.vitals?.o2;
      const hasRespSymptoms = (aiPayload.patient?.symptoms || "").toLowerCase().includes("breath")
        || (aiPayload.patient?.symptoms || "").toLowerCase().includes("cough")
        || (aiPayload.patient?.symptoms || "").toLowerCase().includes("wheez")
        || (aiPayload.patient?.symptoms || "").toLowerCase().includes("dyspn");

      if (discriminators.includes("hypoxia") && rawO2 != null) {
        const parsedO2 = parseFloat(rawO2);
        if (!isNaN(parsedO2) && parsedO2 < 90 && !hasRespSymptoms) {
          // Downgrade to YELLOW (3) — leave a trace so audit can see we intervened.
          // This preserves your original triage object for auditing while preventing a spurious RED.
          finalPriorityCode = finalPriorityCode ? Number(finalPriorityCode) : 3;
          if (finalPriorityCode === 1) {
            finalPriorityCode = 3;
          }
          // annotate triage object so caller can see we adjusted it
          triage._controller_sanity_override = {
            reason: "Suspicious low O2 without respiratory symptoms; downgraded priority_code to avoid false RED",
            original_priority_code: triage.priority_code,
            adjusted_priority_code: finalPriorityCode
          };
        }
      }
    } catch (e) {
      // don't let sanity check crash the flow
      console.warn("Sanity check error:", e.message || e);
    }

    // Use AI's priority_code as authoritative severity mapping (with our small override)
    const priority_code = finalPriorityCode ?? triage.priority_code ?? null;
    const severity = mapPriorityToSeverity(priority_code);

    // Reconcile hard-rule safety flags with final recommendations and specialty assignment
    try {
      const safetyFlags = triage?.safety_flags || triage?.SATS_reasoning?.discriminators_found || [];
      const aiSummary = (triage?.ai_summary || "").toString().toLowerCase();
      const ieFlag = Array.isArray(safetyFlags) && safetyFlags.includes('infective_endocarditis_suspected');
      const ieInSummary = aiSummary.includes('infective_endocarditis') || aiSummary.includes('infective endocarditis');
      const isEndocarditis = ieFlag || ieInSummary;

      if (isEndocarditis) {
        // Ensure IE-specific recommendations are visible to the frontend
        const ieRecs = [
          'IMMEDIATE: Blood cultures (2-3 sets) BEFORE antibiotics - STAT',
          'STAT: 12-lead ECG (assess for conduction abnormalities)',
          'STAT: Echocardiography (TTE ± TEE) to evaluate for vegetations',
          'URGENT: Infectious Diseases consultation for antimicrobial management',
          'Start empiric IV antibiotics after cultures per local ID protocol',
          'Prepare for Cardiology/Cardiothoracic review for possible surgical intervention'
        ];

        // Inject into both clinical_structure and top-level recommendation fields
        if (!triage.clinical_structure) triage.clinical_structure = {};
        triage.clinical_structure.recommendations = ieRecs;
        triage.clinical_recommendations = ieRecs;

        // Primary specialty should be Infectious Diseases with Cardiology as co-service
        triage.assigned_specialty = 'Infectious Diseases';
        triage.assigned_specialties = ['Infectious Diseases', 'Cardiology'];

        // Annotate controller-level override for auditability
        triage._controller_override = triage._controller_override || {};
        triage._controller_override.infective_endocarditis_forced = {
          timestamp: new Date().toISOString(),
          reason: 'Hard-rule detected murmur + embolic signs — injected IE protocol and dual-specialty assignment'
        };
      }
    } catch (e) {
      console.warn('Error reconciling safety flags for IE override:', e && e.message ? e.message : e);
    }

    // Create encounter doc (store the full triage object for audit)
    // Ensure vitals are properly typed (numeric values, not strings)  
    const cleanVitals = {
      temp: aiPayload.vitals.temp ? Number(aiPayload.vitals.temp) : null,
      bp: aiPayload.vitals.bp || null,
      bp_left_systolic: aiPayload.vitals.bp_left_systolic ? Number(aiPayload.vitals.bp_left_systolic) : null,
      bp_left_diastolic: aiPayload.vitals.bp_left_diastolic ? Number(aiPayload.vitals.bp_left_diastolic) : null,
      bp_right_systolic: aiPayload.vitals.bp_right_systolic ? Number(aiPayload.vitals.bp_right_systolic) : null,
      bp_right_diastolic: aiPayload.vitals.bp_right_diastolic ? Number(aiPayload.vitals.bp_right_diastolic) : null,
      bp_systolic: aiPayload.vitals.bp_systolic ? Number(aiPayload.vitals.bp_systolic) : null,
      bp_diastolic: aiPayload.vitals.bp_diastolic ? Number(aiPayload.vitals.bp_diastolic) : null,
      hr: aiPayload.vitals.hr ? Number(aiPayload.vitals.hr) : null,
      o2: aiPayload.vitals.o2 ? Number(aiPayload.vitals.o2) : null,
      resp: aiPayload.vitals.resp ? Number(aiPayload.vitals.resp) : null,
      avpu: aiPayload.vitals.avpu || null
    };

    const encounter = await Encounter.create({
      hospitalId: hospitalId, // Multi-tenant isolation: every encounter for one hospital
      nurse_id: nurseId,
      patient: aiPayload.patient,
      transcript: transcriptArray,
      vitals: cleanVitals,
      ai_triage_report: triage, // Store as AI report (audit trail)
      triage: triage, // Keep for backward compatibility
      recommendations: (triage?.clinical_structure?.recommendations || triage?.recommendations || []),
      nurseNotes: payload.nurseNotes || "",
      status: nurseId ? "pending" : "unassigned",
      submittedAt: null,
      isWaiting: true,
      severity
    });

    // Optionally store vitals snapshot (ensure numeric types)
    if (aiPayload.vitals && (aiPayload.vitals.hr || aiPayload.vitals.temp || aiPayload.vitals.o2 || aiPayload.vitals.bp || aiPayload.vitals.bp_systolic)) {
      await Vital.create({
        encounter_id: encounter._id,
        temperature: aiPayload.vitals.temp ? Number(aiPayload.vitals.temp) : null,
        blood_pressure: aiPayload.vitals.bp,
        blood_pressure_left_systolic: aiPayload.vitals.bp_left_systolic ? Number(aiPayload.vitals.bp_left_systolic) : null,
        blood_pressure_left_diastolic: aiPayload.vitals.bp_left_diastolic ? Number(aiPayload.vitals.bp_left_diastolic) : null,
        blood_pressure_right_systolic: aiPayload.vitals.bp_right_systolic ? Number(aiPayload.vitals.bp_right_systolic) : null,
        blood_pressure_right_diastolic: aiPayload.vitals.bp_right_diastolic ? Number(aiPayload.vitals.bp_right_diastolic) : null,
        blood_pressure_systolic: aiPayload.vitals.bp_systolic ? Number(aiPayload.vitals.bp_systolic) : null,
        blood_pressure_diastolic: aiPayload.vitals.bp_diastolic ? Number(aiPayload.vitals.bp_diastolic) : null,
        heart_rate: aiPayload.vitals.hr ? Number(aiPayload.vitals.hr) : null,
        oxygen_saturation: aiPayload.vitals.o2 ? Number(aiPayload.vitals.o2) : null,
        resp: aiPayload.vitals.resp ? Number(aiPayload.vitals.resp) : null
      });
    }

    // Ensure response includes merged vitals and full triage for frontend fallback
    const responseEncounter = encounter.toObject();
    // sanitize any malformed vitals before returning (e.g., 'undefined/undefined')
    const sanitizeEncounter = (obj) => {
      if (!obj) return obj;
      obj.vitals = obj.vitals || {};
      const bp = obj.vitals.bp;
      if (typeof bp === 'string' && bp.toLowerCase().includes('undefined')) {
        obj.vitals.bp = null;
      }
      ['bp_systolic','bp_diastolic','hr','o2','resp','temp'].forEach(k => {
        const v = obj.vitals[k];
        if (v === undefined || v === null) {
          obj.vitals[k] = null;
        } else if (typeof v === 'number' && Number.isNaN(v)) {
          obj.vitals[k] = null;
        }
      });
      return obj;
    };

    sanitizeEncounter(responseEncounter);
    return res.status(201).json({ 
      message: "Triage completed", 
      encounter: responseEncounter,
      vitals_parsed: triage.vitals_parsed  // explicitly pass vitals_parsed for frontend fallback
    });
  } catch (err) {
    console.error("Triage error:", err);
    return res.status(500).json({ message: err.message });
  }
};

// ----------------------
// Start Encounter (manual nurse intake)
// ----------------------
export const startEncounter = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const nurseId = req.user._id;
    
    // ===== MULTI-TENANT ISOLATION: Get nurse's hospitalId =====
    let hospitalId = null;
    const nurse = await ClinicalStaff.findById(nurseId);
    if (nurse && nurse.hospitalId) {
      hospitalId = nurse.hospitalId;
      console.log(`[ENCOUNTER] Nurse ${nurseId} belongs to hospital ${hospitalId}`);
    } else {
      console.warn(`[ENCOUNTER] Nurse ${nurseId} has no associated hospital`);
    }
    
    const transcriptArray = parseTranscript(req.body.transcript);

    const encounter = await Encounter.create({
      hospitalId: hospitalId, // Multi-tenant isolation: every encounter for one hospital
      nurse_id: nurseId,
      patient: req.body.patient || {},
      transcript: transcriptArray,
      vitals: {
        temp: req.body.temperature,
        bp: req.body.blood_pressure,
        hr: req.body.heart_rate,
        o2: req.body.oxygen_saturation,
        resp: req.body.resp,
      },
      triage: req.body.triage || {},
      severity: req.body.severity || 1,
      status: "pending",
      isWaiting: true,
      nurseNotes: req.body.nurseNotes || "",
    });

    if (
      req.body.temperature ||
      req.body.blood_pressure ||
      req.body.blood_pressure_systolic ||
      req.body.blood_pressure_diastolic ||
      req.body.heart_rate ||
      req.body.oxygen_saturation
    ) {
      await Vital.create({
        encounter_id: encounter._id,
        temperature: req.body.temperature,
        blood_pressure: req.body.blood_pressure,
        blood_pressure_systolic: req.body.blood_pressure_systolic ?? null,
        blood_pressure_diastolic: req.body.blood_pressure_diastolic ?? null,
        heart_rate: req.body.heart_rate,
        oxygen_saturation: req.body.oxygen_saturation,
        resp: req.body.resp,
      });
    }

    // sanitize before returning
    const obj = encounter.toObject();
    obj.vitals = obj.vitals || {};
    if (typeof obj.vitals.bp === 'string' && obj.vitals.bp.toLowerCase().includes('undefined')) obj.vitals.bp = null;
    return res.status(201).json({ message: "Encounter created successfully", encounter: obj });
  } catch (err) {
    console.error("Start Encounter Error:", err);
    return res.status(400).json({ message: err.message || "Failed to create encounter" });
  }
};

// ----------------------
// Confirm Encounter
// ----------------------
export const confirmEncounter = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const { id } = req.params;
    const nurseId = req.user._id;
    const nurseNotes = req.body.nurseNotes || "";

    if (!mongoose.isValidObjectId(id)) return res.status(400).json({ message: "Invalid encounter id" });

    const encounter = await Encounter.findOneAndUpdate(
      { _id: id, $or: [{ nurse_id: nurseId }, { nurse_id: null }] },
      { $set: { status: "confirmed", submittedAt: new Date(), nurseNotes, nurse_id: nurseId } },
      { new: true, select: "_id patient vitals severity status nurseNotes isWaiting triage createdAt" }
    );

    if (!encounter) return res.status(404).json({ message: "Encounter not found or assigned to another nurse" });

    const obj = encounter.toObject();
    obj.vitals = obj.vitals || {};
    if (typeof obj.vitals.bp === 'string' && obj.vitals.bp.toLowerCase().includes('undefined')) obj.vitals.bp = null;
    return res.status(200).json({ message: "Encounter confirmed", encounter: obj });
  } catch (err) {
    console.error("Confirm Encounter Error:", err);
    return res.status(500).json({ message: err.message || "Confirm failed" });
  }
};

// ----------------------
// Attend Encounter
// ----------------------
export const attendEncounter = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const { id } = req.params;
    const nurseId = req.user._id;

    if (!mongoose.isValidObjectId(id)) return res.status(400).json({ message: "Invalid encounter id" });

    const encounter = await Encounter.findOne({ _id: id, nurse_id: nurseId });
    if (!encounter) return res.status(404).json({ message: "Encounter not found" });

    encounter.status = "SEEN";
    encounter.attendedAt = new Date();
    encounter.isWaiting = false;
    await encounter.save();

    return res.status(200).json({ message: "Encounter marked as attended", encounter });
  } catch (err) {
    console.error("Attend Encounter Error:", err);
    return res.status(500).json({ message: err.message || "Attend failed" });
  }
};

// ----------------------
// Assign Physician / Ward / Bed
// ----------------------
export const assignPhysician = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const { id } = req.params;
    if (!mongoose.isValidObjectId(id)) return res.status(400).json({ message: "Invalid encounter id" });

    const nurseId = req.user._id;
    const { physicianId, physicianName, specialty, ward, bed } = req.body || {};

    const encounter = await Encounter.findOne({ _id: id, $or: [{ nurse_id: nurseId }, { nurse_id: null }] });
    if (!encounter) return res.status(404).json({ message: "Encounter not found or assigned to another nurse" });

    // Update triage.assigned_physician and other triage metadata
    encounter.triage = encounter.triage || {};
    if (physicianId) {
      encounter.triage.assigned_physician = { id: physicianId, name: physicianName ?? (`ID:${physicianId}`) };
    }
    if (specialty) encounter.triage.assigned_specialty = specialty;
    if (ward) encounter.triage.ward = ward;
    if (bed) encounter.triage.bed = bed;

    // persist and return sanitized object
    await encounter.save();
    const obj = encounter.toObject();
    obj.vitals = obj.vitals || {};
    if (typeof obj.vitals.bp === 'string' && obj.vitals.bp.toLowerCase().includes('undefined')) obj.vitals.bp = null;

    return res.status(200).json({ message: "Assigned", encounter: obj });
  } catch (err) {
    console.error("Assign Physician Error:", err);
    return res.status(500).json({ message: err.message || "Assignment failed" });
  }
};

// ----------------------
// Get waiting encounters (enriched with waitingSeverity)
// ----------------------
export const getWaitingEncounters = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const nurseId = req.user._id;
    const currentNurse = req.user;

    const encounters = await Encounter.find({ nurse_id: nurseId, isWaiting: true }).sort({ createdAt: 1 });
    
    console.log(`[WAITING] Query: nurse_id=${nurseId}, isWaiting=true, found: ${encounters.length} encounters`);
    encounters.forEach(e => {
      console.log(`  - Encounter: ${e._id}, patient: ${e.patient?.name}, status: ${e.status}, severity: ${e.severity}, SATS: ${e.triage?.manualTriageReport?.satsLevel || 'N/A'}`);
    });

    const enriched = encounters.map(e => {
      const obj = e.toObject();
      obj.waitingSeverity = calculateWaitingSeverity(obj.createdAt);
      // If AI triage includes waitingSeverity prefer that (AI authoritative)
      if (obj.triage && obj.triage.waitingSeverity) {
        obj.waitingSeverity = obj.triage.waitingSeverity;
      }
      // sanitize vitals
      obj.vitals = obj.vitals || {};
      if (typeof obj.vitals.bp === 'string' && obj.vitals.bp.toLowerCase().includes('undefined')) obj.vitals.bp = null;
      ['bp_systolic','bp_diastolic','hr','o2','resp','temp'].forEach(k => {
        const v = obj.vitals[k];
        if (v === undefined || v === null) obj.vitals[k] = null;
        else if (typeof v === 'number' && Number.isNaN(v)) obj.vitals[k] = null;
      });
      
      // Format patient name for frontend display (multiple field name options)
      if (obj.patient) {
        obj.patientName = obj.patient.name || obj.patient.firstName || '';
        obj.firstName = obj.patient.name || obj.patient.firstName || '';
        obj.lastName = obj.patient.surname || obj.patient.lastName || '';
      }
      
      // Add assigned nurse name for frontend display
      const assignedNurseName = obj.assignedNurseName || 
        ((obj.assignedNurseId && typeof obj.assignedNurseId === 'object') 
          ? `${obj.assignedNurseId.name || ''} ${obj.assignedNurseId.surname || ''}`.trim()
          : (currentNurse?.name || 'Assigned Nurse'));
      obj.nurseName = assignedNurseName;
      obj.assignedNurse = assignedNurseName;
      obj.nurse = assignedNurseName;
      obj.assignedNurseName = assignedNurseName;
      
      // Add assigned physician name for frontend display
      const assignedPhysicianName = obj.assignedPhysicianName || 
        ((obj.assignedPhysicianId && typeof obj.assignedPhysicianId === 'object')
          ? `${obj.assignedPhysicianId.name || ''} ${obj.assignedPhysicianId.surname || ''}`.trim()
          : 'Not Assigned');
      obj.specialistDoctor = assignedPhysicianName;
      obj.assignedPhysician = assignedPhysicianName;
      obj.attendedByName = assignedPhysicianName;
      
      return obj;
    });

    return res.status(200).json(enriched);
  } catch (err) {
    console.error("Waiting Queue Error:", err);
    return res.status(500).json({ message: "Failed to fetch waiting encounters" });
  }
};

// ----------------------
// Get encounter by ID
// ----------------------
export const getEncounterById = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const { id } = req.params;
    if (!mongoose.isValidObjectId(id)) return res.status(400).json({ message: "Invalid encounter id" });

    const currentUser = req.user;
    const encounter = await Encounter.findById(id);
    if (!encounter) return res.status(404).json({ message: "Encounter not found" });

    const hasAccess = canAccessEncounter(encounter, currentUser);
    if (!hasAccess) {
      return res.status(403).json({ message: "Not authorized to access this encounter" });
    }

    const obj = encounter.toObject();
    obj.waitingSeverity = calculateWaitingSeverity(obj.createdAt);
    if (obj.triage && obj.triage.waitingSeverity) obj.waitingSeverity = obj.triage.waitingSeverity;
    
    // Format patient name for frontend display
    const patientName = obj.patient?.name || obj.patientName || obj.firstName || obj.patient?.firstName || 'Unknown';
    obj.patientName = patientName;
    obj.firstName = patientName;
    obj.lastName = obj.patient?.surname || obj.patient?.lastName || obj.lastName || '';
    obj.age = obj.patient?.age ?? obj.age ?? null;
    
    // Add assigned nurse name for frontend display
    const assignedNurseName = obj.assignedNurseName || 
      ((obj.assignedNurseId && typeof obj.assignedNurseId === 'object') 
        ? `${obj.assignedNurseId.name || ''} ${obj.assignedNurseId.surname || ''}`.trim()
        : (currentUser?.name || 'Assigned Nurse'));
    obj.nurseName = assignedNurseName;
    obj.assignedNurse = assignedNurseName;
    obj.nurse = assignedNurseName;
    obj.assignedNurseName = assignedNurseName;
    
    // Add assigned physician name for frontend display
    const assignedPhysicianName = obj.assignedPhysicianName || 
      ((obj.assignedPhysicianId && typeof obj.assignedPhysicianId === 'object')
        ? `${obj.assignedPhysicianId.name || ''} ${obj.assignedPhysicianId.surname || ''}`.trim()
        : 'Not Assigned');
    obj.specialistDoctor = assignedPhysicianName;
    obj.assignedPhysician = assignedPhysicianName;
    obj.attendedByName = assignedPhysicianName;
    obj.manual_triage_report = obj.manual_triage_report || obj.triage?.manualTriageReport || {};

    if (!obj.ward_display_info && (obj.manual_triage_report?.assignedWardName || obj.assigned_ward)) {
      obj.ward_display_info = {
        name: obj.manual_triage_report?.assignedWardName || obj.assigned_ward?.name || 'General Ward',
        wardName: obj.manual_triage_report?.assignedWardName || obj.assigned_ward?.name || 'General Ward',
        department: obj.manual_triage_report?.assignedWardName || 'General Ward',
        wardId: obj.assigned_ward?.toString?.() || null
      };
    }

    if (!obj.bed_display_info && (obj.manual_triage_report?.assignedBedNumber || obj.assigned_bed)) {
      obj.bed_display_info = {
        bedId: obj.assigned_bed?.toString?.() || obj.manual_triage_report?.assignedBedNumber || null,
        bedNumber: obj.manual_triage_report?.assignedBedNumber || obj.assigned_bed?.toString?.() || 'Not Assigned',
        bed_number: obj.manual_triage_report?.assignedBedNumber || obj.assigned_bed?.toString?.() || 'Not Assigned',
        department: obj.manual_triage_report?.assignedWardName || 'General',
        wardName: obj.manual_triage_report?.assignedWardName || 'General Ward'
      };
    }

    // LOGGING: What's being returned when fetching for confirmation page
    console.log(`[GetEncounterById] Fetching encounter ${id} for confirmation page`);
    console.log(`[GetEncounterById] - assigned_specialty:`, obj.triage?.assigned_specialty);
    console.log(`[GetEncounterById] - assigned_ward (ID):`, obj.assigned_ward);
    console.log(`[GetEncounterById] - assigned_bed (ID):`, obj.assigned_bed);
    console.log(`[GetEncounterById] - manually_selected_clinicians:`, obj.manually_selected_clinicians);
    console.log(`[GetEncounterById] - assigned_clinicians (before fix):`, obj.assigned_clinicians);

    // FIX: If assigned_clinicians is empty but clinician IDs exist, populate on-the-fly
    if ((!obj.assigned_clinicians || obj.assigned_clinicians.length === 0) && 
        (obj.assignedPhysicianId || obj.assignedNurseId)) {
      console.log(`[GetEncounterById] ⚠️  assigned_clinicians empty, populating from IDs...`);
      obj.assigned_clinicians = [];
      
      if (obj.assignedPhysicianId) {
        // Fetch actual physician specialty from ClinicalStaff
        const physician = await ClinicalStaff.findById(obj.assignedPhysicianId).select('clinicalSpecialty specialty name firstName lastName').lean().catch(() => null);
        const physicianSpecialty = physician?.clinicalSpecialty || physician?.specialty || 'General';
        const physicianName = physician?.name || `${physician?.firstName || ''} ${physician?.lastName || ''}`.trim() || obj.assignedPhysicianName || 'Not Assigned';
        
        obj.assigned_clinicians.push({
          id: obj.assignedPhysicianId,
          name: physicianName,
          role: 'PHYSICIAN',
          specialty: physicianSpecialty,
          score: 100,
          assigned_at: new Date()
        });
        console.log(`[GetEncounterById]    ✅ Added physician: ${physicianName} (${physicianSpecialty})`);
      }
      
      if (obj.assignedNurseId) {
        // Fetch actual nurse specialty from ClinicalStaff
        const nurse = await ClinicalStaff.findById(obj.assignedNurseId).select('clinicalSpecialty specialty name firstName lastName').lean().catch(() => null);
        const nurseSpecialty = nurse?.clinicalSpecialty || nurse?.specialty || 'Nursing';
        const nurseName = nurse?.name || `${nurse?.firstName || ''} ${nurse?.lastName || ''}`.trim() || obj.assignedNurseName || 'Assigned Nurse';
        
        obj.assigned_clinicians.push({
          id: obj.assignedNurseId,
          name: nurseName,
          role: 'NURSE',
          specialty: nurseSpecialty,
          score: 100,
          assigned_at: new Date()
        });
        console.log(`[GetEncounterById]    ✅ Added nurse: ${nurseName} (${nurseSpecialty})`);
      }
      
      console.log(`[GetEncounterById] ✅ assigned_clinicians populated: ${obj.assigned_clinicians.length} clinicians`);
    } else if (obj.assigned_clinicians && obj.assigned_clinicians.length > 0) {
      // ENRICH: If assigned_clinicians exist but have missing/empty specialty, fetch from ClinicalStaff
      console.log(`[GetEncounterById] 🔍 Enriching assigned_clinicians specialty fields...`);
      
      for (let i = 0; i < obj.assigned_clinicians.length; i++) {
        const clinician = obj.assigned_clinicians[i];
        if (!clinician.specialty || clinician.specialty === 'General' || clinician.specialty === 'Nursing') {
          // Try to fetch real specialty from ClinicalStaff
          const staff = await ClinicalStaff.findById(clinician.id).select('clinicalSpecialty specialty').lean().catch(() => null);
          if (staff && (staff.clinicalSpecialty || staff.specialty)) {
            const realSpecialty = staff.clinicalSpecialty || staff.specialty;
            console.log(`[GetEncounterById]    ✅ Enriched ${clinician.name}: '${clinician.specialty}' → '${realSpecialty}'`);
            obj.assigned_clinicians[i].specialty = realSpecialty;
          }
        }
      }
      console.log(`[GetEncounterById] ✅ assigned_clinicians enriched with real specialties`);
    }

    console.log(`[GetEncounterById] - assigned_clinicians (after fix):`, obj.assigned_clinicians?.map(c => `${c.name} (${c.role})`));
    console.log(`[GetEncounterById] - ward_display_info:`, obj.ward_display_info?.wardName);
    console.log(`[GetEncounterById] - bed_display_info:`, obj.bed_display_info?.bedId);

    return res.status(200).json({ encounter: obj });
  } catch (err) {
    console.error("Get Encounter Error:", err);
    return res.status(500).json({ message: "Failed to fetch encounter" });
  }
};

// ----------------------
// Update AI Fields (Manual overrides by nurse)
// ----------------------
export const updateAIFields = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    
    const { id } = req.params;
    const { changedFields, nurseNotes, ward_display_info, bed_display_info } = req.body;
    
    // Log the entire payload received
    console.log('\n=== [UpdateAIFields] RECEIVED PAYLOAD ===');
    console.log('[UpdateAIFields] Encounter ID:', id);
    console.log('[UpdateAIFields] Changed Fields:', JSON.stringify(changedFields, null, 2));
    console.log('[UpdateAIFields] Nurse Notes:', nurseNotes);
    console.log('[UpdateAIFields] Ward Display Info:', JSON.stringify(ward_display_info, null, 2));
    console.log('[UpdateAIFields] Bed Display Info:', JSON.stringify(bed_display_info, null, 2));
    console.log('=====================================\n');
    
    if (!mongoose.isValidObjectId(id)) {
      return res.status(400).json({ message: "Invalid encounter id" });
    }

    const nurseId = req.user._id;
    const encounter = await Encounter.findOne({ _id: id, nurse_id: nurseId });
    
    if (!encounter) {
      return res.status(404).json({ message: "Encounter not found or not authorized" });
    }

    // Build field changes tracking object
    const fieldChanges = {};
    const updatesToApply = {};

    if (changedFields && typeof changedFields === 'object') {
      // Track what changed from original AI report
      for (const [field, newValue] of Object.entries(changedFields)) {
        const originalValue = encounter.ai_triage_report?.[field] || null;
        fieldChanges[field] = {
          original: originalValue,
          override: newValue
        };
        
        // Apply specific field updates
        switch (field) {
          case 'ai_summary':
            updatesToApply.ai_summary = newValue;
            break;
          case 'assigned_specialty':
            updatesToApply.assigned_specialty = newValue;
            // Also update in triage so confirmation page displays it
            if (encounter.triage) {
              encounter.triage.assigned_specialty = newValue;
            }
            break;
          case 'assigned_physician':
            updatesToApply.assigned_physician = newValue;
            break;
          case 'assigned_clinicians':
            updatesToApply.assigned_clinicians = newValue;
            encounter.assigned_clinicians = newValue; // Also set on encounter level
            if (encounter.triage) {
              encounter.triage.assigned_clinicians = newValue; // Also set in triage so confirmation page displays them
            }
            break;
          case 'manually_selected_clinicians':
            console.log('[UpdateAIFields] Saving manually_selected_clinicians:', newValue);
            encounter.manually_selected_clinicians = newValue;
            break;
          case 'discriminators':
            updatesToApply.discriminators = Array.isArray(newValue) ? newValue : [newValue];
            break;
          case 'sats_points':
            updatesToApply.sats_points = Number(newValue);
            break;
          case 'sats_color':
            updatesToApply.sats_color = newValue;
            break;
          case 'severity':
            const numericSeverity = Number(newValue) || 1;
            updatesToApply.severity = numericSeverity;
            encounter.severity = numericSeverity;
            if (encounter.triage) {
              encounter.triage.severity = numericSeverity;
            }
            if (encounter.ai_triage_report) {
              encounter.ai_triage_report.severity = numericSeverity;
            }
            break;
          case 'recommendations':
            updatesToApply.recommendations = Array.isArray(newValue) ? newValue : [newValue];
            break;
          case 'ward':
          case 'assigned_ward':
            // Extract just the ID from object or use the value directly
            const wardId = newValue?.id || newValue?._id || newValue;
            updatesToApply.assigned_ward = wardId;
            encounter.assigned_ward = wardId;
            break;
          case 'bed':
          case 'assigned_bed':
            // Extract just the ID from object or use the value directly
            const bedId = newValue?.id || newValue?._id || newValue;
            updatesToApply.assigned_bed = bedId;
            encounter.assigned_bed = bedId;
            break;
        }
      }
    }

    // Update nurse notes if provided
    if (nurseNotes !== undefined) {
      encounter.nurseNotes = nurseNotes;
      fieldChanges.nurseNotes = {
        original: encounter.nurseNotes || "",
        override: nurseNotes
      };
    }

    // Update manual triage report with overrides
    encounter.manual_triage_report = {
      ...encounter.manual_triage_report,
      ...updatesToApply,
      field_changes: fieldChanges,
      overridden_at: new Date(),
      overridden_by: nurseId,
      notes: nurseNotes || encounter.manual_triage_report?.notes || ""
    };

    // Store display info for confirmation pages (not in schema, just in memory before save)
    if (ward_display_info) {
      encounter.ward_display_info = ward_display_info;
      console.log('[UpdateAIFields] Stored ward_display_info:', ward_display_info.wardName);
    }
    if (bed_display_info) {
      encounter.bed_display_info = bed_display_info;
      console.log('[UpdateAIFields] Stored bed_display_info:', bed_display_info.bedId);
    }

    console.log('[UpdateAIFields] Before save - encounter.triage.assigned_specialty:', encounter.triage?.assigned_specialty);
    console.log('[UpdateAIFields] Before save - encounter.assigned_ward:', encounter.assigned_ward);
    console.log('[UpdateAIFields] Before save - encounter.assigned_bed:', encounter.assigned_bed);
    console.log('[UpdateAIFields] Before save - encounter.manually_selected_clinicians:', encounter.manually_selected_clinicians);
    console.log('[UpdateAIFields] Before save - encounter.assigned_clinicians:', encounter.assigned_clinicians);

    // Save encounter
    await encounter.save();

    console.log(`[Encounter] Fields updated for ${id} by nurse ${nurseId}`);
    console.log(`[Encounter] Changed fields:`, Object.keys(fieldChanges));
    console.log(`[Encounter] **AFTER SAVE** - assigned_specialty:`, encounter.triage?.assigned_specialty);
    console.log(`[Encounter] **AFTER SAVE** - assigned_ward:`, encounter.assigned_ward);
    console.log(`[Encounter] **AFTER SAVE** - assigned_bed:`, encounter.assigned_bed);
    console.log(`[Encounter] **AFTER SAVE** - manually_selected_clinicians:`, encounter.manually_selected_clinicians);
    console.log(`[Encounter] **AFTER SAVE** - assigned_clinicians:`, encounter.assigned_clinicians);
    console.log(`[Encounter] **AFTER SAVE** - ward_display_info:`, encounter.ward_display_info?.wardName);
    console.log(`[Encounter] **AFTER SAVE** - bed_display_info:`, encounter.bed_display_info?.bedId);

    return res.status(200).json({
      message: "Fields updated successfully",
      encounter: encounter.toObject(),
      changedCount: Object.keys(fieldChanges).length,
      auditTrail: {
        updatedBy: nurseId,
        updatedAt: new Date(),
        fieldsChanged: fieldChanges
      }
    });

  } catch (err) {
    console.error("Update AI Fields Error:", err);
    return res.status(500).json({ 
      message: "Failed to update fields",
      error: err.message 
    });
  }
};

// ----------------------
// Get Encounters for Nurse Dashboard
// ----------------------
export const getEncountersForNurse = async (req, res) => {
  try {
    console.log('[GET_ENCOUNTERS] ========== FETCHING ENCOUNTERS FOR NURSE ==========');
    
    const nurseId = req.user?._id;
    if (!nurseId) {
      console.log('[GET_ENCOUNTERS] ❌ FAILED: User not authenticated');
      return res.status(401).json({ message: "Not authorized" });
    }
    
    console.log('[GET_ENCOUNTERS] Nurse ID:', nurseId.toString());
    console.log('[GET_ENCOUNTERS] User role:', req.user?.role);
    const currentNurse = req.user;
    
    // Support filtering by status or handover_status via query parameter
    const { status, handover_status, bypassNurseFilter } = req.query;
    
    // Allow admins/managers to bypass nurse_id filter to view ALL historical encounters
    const isAdmin = req.user?.role === 'ADMIN' || req.user?.role === 'MANAGER';
    const shouldBypassFilter = bypassNurseFilter === 'true' && isAdmin;
    
    console.log('[GET_ENCOUNTERS] isAdmin:', isAdmin, 'bypassNurseFilter:', bypassNurseFilter, 'shouldBypass:', shouldBypassFilter);
    
    let query = shouldBypassFilter ? {} : { nurse_id: nurseId };
    
    if (handover_status) {
      // Query for encounters where any clinician handover is in the requested state(s)
      const statuses = String(handover_status)
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);

      query = {
        ...query,
        "clinical_handovers.handover_status": statuses.length > 1 ? { $in: statuses } : statuses[0]
      };
      console.log('[GET_ENCOUNTERS] Filtering by handover_status:', statuses);
    } else if (status) {
      // Support comma-separated statuses
      const statusList = String(status)
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      
      query.status = statusList.length > 1 ? { $in: statusList } : statusList[0];
      console.log('[GET_ENCOUNTERS] Filtering by status:', statusList);
    } else {
      console.log('[GET_ENCOUNTERS] No status filter');
    }

    console.log('[GET_ENCOUNTERS] Query:', JSON.stringify(query));
    const encounters = await Encounter.find(query).sort({ createdAt: -1 });
    
    console.log('[GET_ENCOUNTERS] Found', encounters.length, 'encounters');
    if (encounters.length > 0) {
      console.log('[GET_ENCOUNTERS] First encounter status:', encounters[0].status);
      console.log('[GET_ENCOUNTERS] Sample nurse_ids:', encounters.slice(0, 3).map(e => e.nurse_id?.toString()));
    }
    
    // Format encounters for frontend display
    const formatted = encounters.map(e => {
      const obj = e.toObject ? e.toObject() : e;
      
      // Format patient name for frontend display (multiple field name options)
      if (obj.patient) {
        obj.patientName = obj.patient.name || obj.patient.firstName || '';
        obj.firstName = obj.patient.name || obj.patient.firstName || '';
        obj.lastName = obj.patient.surname || obj.patient.lastName || '';
      }
      
      // Add assigned nurse name for frontend display
      const assignedNurseName = obj.assignedNurseName || 
        ((obj.assignedNurseId && typeof obj.assignedNurseId === 'object') 
          ? `${obj.assignedNurseId.name || ''} ${obj.assignedNurseId.surname || ''}`.trim()
          : (currentNurse?.name || 'Assigned Nurse'));
      obj.nurseName = assignedNurseName;
      obj.assignedNurse = assignedNurseName;
      obj.nurse = assignedNurseName;
      obj.assignedNurseName = assignedNurseName;
      
      // Add assigned physician name for frontend display
      const assignedPhysicianName = obj.assignedPhysicianName || 
        ((obj.assignedPhysicianId && typeof obj.assignedPhysicianId === 'object')
          ? `${obj.assignedPhysicianId.name || ''} ${obj.assignedPhysicianId.surname || ''}`.trim()
          : 'Not Assigned');
      obj.specialistDoctor = assignedPhysicianName;
      obj.assignedPhysician = assignedPhysicianName;
      obj.attendedByName = assignedPhysicianName;
      
      return obj;
    });
    
    console.log('[GET_ENCOUNTERS] Returning', formatted.length, 'formatted encounters');
    console.log('[GET_ENCOUNTERS] ========== SUCCESS ==========');
    
    return res.status(200).json({ success: true, encounters: formatted });
  } catch (err) {
    console.error('[GET_ENCOUNTERS] ❌ Error:', err.message);
    console.error('[GET_ENCOUNTERS] Stack:', err.stack);
    return res.status(500).json({ message: "Failed to list encounters" });
  }
};

// ----------------------
// Get Clinician Dashboard
// ----------------------
export const getClinicianDashboard = async (req, res) => {
  try {
    const { clinicianId } = req.params;
    
    // Find all encounters where this clinician has a handover record
    const encounters = await Encounter.find({
      "clinical_handovers.clinician_id": clinicianId,
      "clinical_handovers.handover_status": { $in: ["NOTIFIED", "VIEWED", "ACKNOWLEDGED", "IN_PROGRESS"] }
    })
      .select("patient triage assigned_ward assigned_bed assigned_clinicians manually_selected_clinicians nurseNotes clinical_handovers status createdAt")
      .sort({ createdAt: -1 });

    // Format encounters for clinician dashboard
    const formatted = encounters.map(e => {
      const obj = e.toObject ? e.toObject() : e;
      
      if (obj.patient) {
        obj.patientName = obj.patient.name || obj.patient.firstName || '';
        obj.firstName = obj.patient.name || obj.patient.firstName || '';
        obj.lastName = obj.patient.surname || obj.patient.lastName || '';
      }
      
      // Get the handover status for this specific clinician
      const thisClinicianHandover = obj.clinical_handovers?.find(h => h.clinician_id?.toString() === clinicianId);
      obj.handover = thisClinicianHandover;
      
      return obj;
    });

    console.log(`[ClinicianDashboard] Fetching cases for clinician ${clinicianId}: found ${formatted.length} cases`);
    
    return res.status(200).json({
      clinician_id: clinicianId,
      cases: formatted,
      total: formatted.length
    });
  } catch (err) {
    console.error("Clinician Dashboard Error:", err);
    return res.status(500).json({
      message: "Failed to fetch clinician cases",
      error: err.message
    });
  }
};

// ----------------------
// Initiate Clinical Handover
// ----------------------
export const initiateHandover = async (req, res) => {
  try {
    const { id: encounterId } = req.params;
    const { 
      reportType = 'AI_TRIAGE_REPORT',
      clinicians = [],
      manuallySelected = [],
      triageNurse
    } = req.body;

    const { hospitalId: requestingHospitalId, _id: userId } = req.user;

    console.log('[HANDOVER] ========== INITIATING HANDOVER ==========');
    console.log('[HANDOVER] Encounter ID:', encounterId);
    console.log('[HANDOVER] Report Type:', reportType);
    console.log('[HANDOVER] Hospital:', requestingHospitalId);
    console.log('[HANDOVER] User:', userId);

    // Fetch encounter
    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      console.log('[HANDOVER] ❌ Encounter not found:', encounterId);
      return res.status(404).json({ error: 'Encounter not found' });
    }

    // Verify hospital isolation
    if (encounter.hospitalId.toString() !== requestingHospitalId) {
      console.error('[HANDOVER] ❌ Hospital mismatch - Access Denied');
      return res.status(403).json({ error: 'Access denied - different hospital' });
    }

    console.log('[HANDOVER] ✅ Authorization verified');

    // Calculate validated priority
    let validated_priority = 'YELLOW';
    let reportTypeUsed = reportType;

    if (encounter.manual_triage_report?.selected_priority) {
      validated_priority = encounter.manual_triage_report.selected_priority;
      reportTypeUsed = 'MANUAL_TRIAGE_REPORT';
      console.log('[HANDOVER] Priority from manual override:', validated_priority);
    } else if (encounter.ai_triage_report?.calculated_priority) {
      validated_priority = encounter.ai_triage_report.calculated_priority;
      reportTypeUsed = 'AI_TRIAGE_REPORT';
      console.log('[HANDOVER] Priority from AI report:', validated_priority);
    } else if (encounter.triage?.priority_code) {
      validated_priority = encounter.triage.priority_code;
      console.log('[HANDOVER] Priority from triage:', validated_priority);
    }

    // Mark as official
    encounter.status = 'OFFICIAL_REPORT';
    encounter.validated_effective_priority = validated_priority;
    encounter.handover_completed = false;

    // Record triage nurse from JWT extraction
    if (triageNurse) {
      encounter.original_triage_nurse = {
        nurse_id: triageNurse.id,
        nurse_name: triageNurse.name,
        nurse_email: triageNurse.email,
        nurse_role: triageNurse.role,
        triaged_at: new Date()
      };
      console.log('[HANDOVER] Triage nurse recorded:', triageNurse.name);
    }

    // Deduplicate clinicians
    const allClinicians = [...clinicians, ...manuallySelected];
    const uniqueClinicians = [];
    const seenIds = new Set();

    for (const clinician of allClinicians) {
      const cId = clinician.id?.toString() || clinician._id?.toString();
      if (!seenIds.has(cId)) {
        uniqueClinicians.push(clinician);
        seenIds.add(cId);
      }
    }

    console.log('[HANDOVER] Unique clinicians:', uniqueClinicians.length);

    // Import Notification here to avoid circular dependencies
    const { Notification } = await import('../models/notification.model.js');

    // Create notifications for each clinician
    const notificationsSent = [];

    for (const clinician of uniqueClinicians) {
      try {
        const notification = await Notification.create({
          encounter_id: encounterId,
          hospital_id: encounter.hospitalId,
          clinician_id: clinician.id || clinician._id,
          clinician_name: clinician.name,
          clinician_role: clinician.role || 'PHYSICIAN',
          clinician_specialty: clinician.specialty,
          triage_nurse_id: triageNurse?.id,
          triage_nurse_name: triageNurse?.name,
          triage_nurse_email: triageNurse?.email,
          report_type: reportTypeUsed,
          priority_code: validated_priority,
          patient_mrn: encounter.patient?.id_number,
          patient_name: `${encounter.patient?.name || ''} ${encounter.patient?.surname || ''}`.trim(),
          chief_complaint: encounter.patient?.symptoms,
          status: 'UNREAD',
          created_at: new Date()
        });

        console.log('[HANDOVER] Notification created for', clinician.name);

        notificationsSent.push({
          clinician_id: clinician.id || clinician._id,
          clinician_name: clinician.name,
          notification_id: notification._id,
          sent_at: new Date(),
          status: 'NOTIFIED'
        });

        // Add to clinical_handovers
        encounter.clinical_handovers.push({
          clinician_id: clinician.id || clinician._id,
          clinician_name: clinician.name,
          clinician_role: clinician.role || 'PHYSICIAN',
          clinician_specialty: clinician.specialty,
          handover_type: 'INITIAL',
          report_type: reportTypeUsed,
          handover_status: 'NOTIFIED',
          handover_at: new Date(),
          notification_id: notification._id,
          is_active: true
        });

      } catch (err) {
        console.error('[HANDOVER] Error creating notification for', clinician.name, err);
      }
    }

    // Add to status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }

    encounter.status_history.push({
      status: 'OFFICIAL_REPORT',
      changed_by: userId,
      changed_at: new Date(),
      notes: `${reportTypeUsed} finalized. ${uniqueClinicians.length} clinician(s) notified.`
    });

    // Save encounter
    await encounter.save();

    console.log('[HANDOVER] ✅ Handover completed successfully');
    console.log('[HANDOVER] Status:', encounter.status);
    console.log('[HANDOVER] Priority:', validated_priority);
    console.log('[HANDOVER] Notifications sent:', notificationsSent.length);

    res.json({
      success: true,
      encounterId: encounterId,
      reportType: reportTypeUsed,
      priority: validated_priority,
      notificationsCreated: notificationsSent.length,
      triageNurse: triageNurse?.name,
      timestamp: new Date()
    });

  } catch (err) {
    console.error('[HANDOVER] ❌ Error:', err);
    res.status(500).json({ error: err.message });
  }
};

// ----------------------
// Acknowledge Handover
// ----------------------
export const acknowledgeHandover = async (req, res) => {
  try {
    console.log('\n════════════════════════════════════════════════════════════');
    console.log('[HANDOVER ACKNOWLEDGE] ✓ Request received');
    
    if (!req.user?._id) {
      console.error('[HANDOVER ACKNOWLEDGE] ✗ User not authenticated');
      return res.status(401).json({ message: "Not authorized" });
    }
    
    const { id } = req.params;
    const { acknowledgedNotes } = req.body;
    
    console.log('[HANDOVER ACKNOWLEDGE] Encounter ID:', id);
    console.log('[HANDOVER ACKNOWLEDGE] Clinician ID:', req.user._id);
    console.log('[HANDOVER ACKNOWLEDGE] Clinician Name:', req.user.name);
    console.log('[HANDOVER ACKNOWLEDGE] Clinician Role:', req.user.role);
    console.log('[HANDOVER ACKNOWLEDGE] Acknowledged Notes:', acknowledgedNotes);
    
    if (!mongoose.Types.ObjectId.isValid(id)) {
      console.error('[HANDOVER ACKNOWLEDGE] ✗ Invalid encounter ID format');
      return res.status(400).json({ message: "Invalid encounter ID" });
    }

    const encounter = await Encounter.findById(id);
    console.log('[HANDOVER ACKNOWLEDGE] Encounter found:', !!encounter);
    
    if (!encounter) {
      console.error('[HANDOVER ACKNOWLEDGE] ✗ Encounter not found');
      return res.status(404).json({ message: "Encounter not found" });
    }

    console.log('[HANDOVER ACKNOWLEDGE] Encounter status before:', encounter.status);
    console.log('[HANDOVER ACKNOWLEDGE] Patient:', encounter.patient?.name || encounter.patient?.firstName);
    console.log('[HANDOVER ACKNOWLEDGE] Clinical handovers count:', encounter.clinical_handovers?.length || 0);

    // Find and update this clinician's handover record
    const clinicianHandover = encounter.clinical_handovers?.find(h => 
      h.clinician_id?.toString() === req.user._id.toString()
    );
    
    if (!clinicianHandover) {
      console.error('[HANDOVER ACKNOWLEDGE] ✗ No handover record found for this clinician');
      console.log('[HANDOVER ACKNOWLEDGE] Available handover records:', 
        encounter.clinical_handovers?.map(h => ({
          clinician_id: h.clinician_id?.toString(),
          clinician_name: h.clinician_name,
          clinician_role: h.clinician_role
        }))
      );
      return res.status(404).json({ message: "Handover record not found for this clinician" });
    }

    console.log('[HANDOVER ACKNOWLEDGE] Handover record found');
    console.log('[HANDOVER ACKNOWLEDGE] Previous status:', clinicianHandover.handover_status);
    console.log('[HANDOVER ACKNOWLEDGE] Clinician role:', clinicianHandover.clinician_role);

    // Update handover status
    clinicianHandover.handover_status = 'ACKNOWLEDGED';
    clinicianHandover.viewed_at = clinicianHandover.viewed_at || new Date();
    clinicianHandover.acknowledged_at = new Date();
    clinicianHandover.acknowledged_notes = acknowledgedNotes || '';
    
    console.log('[HANDOVER ACKNOWLEDGE] Updated handover status:', clinicianHandover.handover_status);
    console.log('[HANDOVER ACKNOWLEDGE] Acknowledged at:', clinicianHandover.acknowledged_at);
    
    // If this is the first physician to acknowledge, update encounter status to SEEN
    const isPhysician = clinicianHandover.clinician_role === 'PHYSICIAN';
    console.log('[HANDOVER ACKNOWLEDGE] Is physician?', isPhysician);
    
    if (isPhysician && encounter.status === 'AWAITING_PHYSICIAN') {
      console.log('[HANDOVER ACKNOWLEDGE] Updating encounter status to SEEN...');
      encounter.status = 'SEEN';
      
      // Add to status history
      if (!encounter.status_history) {
        encounter.status_history = [];
      }
      encounter.status_history.push({
        status: 'SEEN',
        changed_by: req.user._id,
        changed_at: new Date(),
        notes: `Physician acknowledged handover: ${clinicianHandover.clinician_name}`
      });
      console.log('[HANDOVER ACKNOWLEDGE] ✓ Encounter status updated to SEEN');
    } else {
      console.log('[HANDOVER ACKNOWLEDGE] Encounter status unchanged (current: ' + encounter.status + ')');
    }
    
    await encounter.save();
    console.log('[HANDOVER ACKNOWLEDGE] ✓ Encounter saved to database');

    const result = {
      message: "Handover acknowledged successfully",
      encounter: encounter.toObject(),
      handover: clinicianHandover
    };
    
    console.log('[HANDOVER ACKNOWLEDGE] ✓ ACKNOWLEDGMENT COMPLETE');
    console.log('[HANDOVER ACKNOWLEDGE] Final encounter status:', encounter.status);
    console.log('[HANDOVER ACKNOWLEDGE] Final handover status:', clinicianHandover.handover_status);
    console.log('════════════════════════════════════════════════════════════\n');

    return res.status(200).json(result);
  } catch (err) {
    console.error('════════════════════════════════════════════════════════════');
    console.error('[HANDOVER ACKNOWLEDGE] ✗ ERROR:', err.message);
    console.error('[HANDOVER ACKNOWLEDGE] Stack:', err.stack);
    console.error('════════════════════════════════════════════════════════════\n');
    return res.status(500).json({
      message: "Failed to acknowledge handover",
      error: err.message
    });
  }
};

// ----------------------
// Record Diagnosis
// ----------------------
export const recordDiagnosis = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    
    const { id } = req.params;
    const { diagnosis, treatment_plan, notes, treatments, recommendations, progression_notes } = req.body;
    
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: "Invalid encounter ID" });
    }

    const encounter = await Encounter.findById(id);
    
    if (!encounter) {
      return res.status(404).json({ message: "Encounter not found" });
    }

    // Ensure clinician has authority over this encounter
    const clinicianHandover = encounter.clinical_handovers?.find(h => 
      h.clinician_id?.toString() === req.user._id.toString()
    );
    
    if (!clinicianHandover) {
      return res.status(403).json({ message: "Not authorized to update this encounter" });
    }

    // Update encounter with diagnosis
    encounter.diagnosis = diagnosis;
    encounter.treatment_plan = treatment_plan || '';
    encounter.physician_notes = notes || '';
    
    // Add treatments array with physician context
    if (Array.isArray(treatments) && treatments.length > 0) {
      encounter.treatments = treatments.map(t => ({
        ...t,
        prescribed_by: clinicianHandover.clinician_name,
        prescribed_at: new Date()
      }));
    }
    
    // Add recommendations array with physician context
    if (Array.isArray(recommendations) && recommendations.length > 0) {
      encounter.recommendations = recommendations.map(r => ({
        ...r,
        recommended_by: clinicianHandover.clinician_name,
        recommended_at: new Date()
      }));
    }
    
    // Add progression notes with physician context
    if (Array.isArray(progression_notes) && progression_notes.length > 0) {
      encounter.progression_notes = progression_notes.map(p => ({
        ...p,
        recorded_by: clinicianHandover.clinician_name,
        recorded_at: new Date()
      }));
    }

    // NOTE: Medication requests are now created via separate pharmacy endpoint in bed modal workflow
    // This keeps diagnosis endpoint focused on treatment plan only (diagnosis, treatments, recommendations)
    
    // Update physician reference
    encounter.assigned_physician = {
      id: req.user._id,
      name: clinicianHandover.clinician_name,
      specialty: clinicianHandover.clinician_specialty,
      diagnosed_at: new Date()
    };

    encounter.time_to_diagnosis_minutes = encounter.createdAt
      ? Math.max(0, Math.floor((Date.now() - new Date(encounter.createdAt).getTime()) / 60000))
      : 0;
    
    // Update status to DIAGNOSED
    encounter.status = 'DIAGNOSED';
    
    // Add to status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: 'DIAGNOSED',
      changed_by: req.user._id,
      changed_at: new Date(),
      notes: `Diagnosis completed with ${treatments?.length || 0} treatments and ${recommendations?.length || 0} recommendations`
    });
    
    // Mark handover as IN_PROGRESS (physician working on case)
    clinicianHandover.handover_status = 'IN_PROGRESS';
    
    await encounter.save();

    console.log(`[Encounter] Diagnosis recorded for ${id} by physician ${req.user._id}`);
    console.log(`[Encounter] Treatments: ${treatments?.length || 0}, Recommendations: ${recommendations?.length || 0}`);
    console.log(`[Encounter] Encounter status: DIAGNOSED`);

    return res.status(200).json({
      message: "Diagnosis recorded successfully",
      encounter: encounter.toObject()
    });
  } catch (err) {
    console.error("Diagnosis Error:", err);
    return res.status(500).json({
      message: "Failed to record diagnosis",
      error: err.message
    });
  }
};

// ----------------------
// Create Inpatient Progress Report
// ----------------------
export const createInpatientProgressReport = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });

    const { id } = req.params;
    const { diagnosis, treatment_plan, notes, treatments, recommendations, progression_notes } = req.body;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: "Invalid encounter ID" });
    }

    const encounter = await Encounter.findById(id);
    if (!encounter) {
      return res.status(404).json({ message: "Encounter not found" });
    }

    const isHospitalized = Boolean(
      encounter.isHospitalized ||
      encounter.currentBedId ||
      encounter.assigned_bed ||
      encounter.triage?.assigned_bed?.id ||
      encounter.status === "HOSPITALIZED"
    );

    if (!isHospitalized) {
      return res.status(409).json({
        message: "Patient is not currently hospitalized. Use Admit to Hospital first."
      });
    }

    const clinicianName = req.user?.name || req.user?.email || "Clinical team";

    if (diagnosis?.trim()) {
      encounter.diagnosis = diagnosis;
    }

    if (treatment_plan?.trim()) {
      encounter.treatment_plan = treatment_plan;
    }

    if (notes?.trim()) {
      encounter.physician_notes = notes;
    }

    if (Array.isArray(treatments) && treatments.length > 0) {
      encounter.treatments = [
        ...(encounter.treatments || []),
        ...treatments.map((t) => ({
          ...t,
          prescribed_by: clinicianName,
          prescribed_at: new Date()
        }))
      ];
    }

    if (Array.isArray(recommendations) && recommendations.length > 0) {
      encounter.recommendations = [
        ...(encounter.recommendations || []),
        ...recommendations.map((r) => ({
          ...r,
          recommended_by: clinicianName,
          recommended_at: new Date()
        }))
      ];
    }

    if (Array.isArray(progression_notes) && progression_notes.length > 0) {
      encounter.progression_notes = [
        ...(encounter.progression_notes || []),
        ...progression_notes.map((p) => ({
          ...p,
          recorded_by: clinicianName,
          recorded_at: new Date()
        }))
      ];
    }

    // NOTE: Medication requests are now created via separate pharmacy endpoint in bed modal workflow

    encounter.isHospitalized = true;
    encounter.status = encounter.status === "DISCHARGED" ? "HOSPITALIZED" : (encounter.status || "HOSPITALIZED");

    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: "HOSPITALIZED",
      changed_by: req.user._id,
      changed_at: new Date(),
      notes: "New inpatient progression report created for the active stay"
    });

    await encounter.save();

    return res.status(200).json({
      message: "Inpatient progression report created successfully",
      encounter: encounter.toObject()
    });
  } catch (err) {
    console.error("Create Inpatient Progress Report Error:", err);
    return res.status(500).json({
      message: "Failed to create inpatient progress report",
      error: err.message
    });
  }
};

// ----------------------
// Update Patient Disposition
// ----------------------
const isPendingLabRequest = (request) => request && typeof request === 'object' && !['COMPLETED', 'DECLINED'].includes((request.status || '').toUpperCase());
const isPendingMedicationRequest = (request) => request && typeof request === 'object' && !['DECLINED'].includes((request.status || '').toUpperCase());
const isPendingMedicationAdministration = (admin) => admin && typeof admin === 'object' && !['ADMINISTERED', 'COMPLETED', 'DECLINED'].includes((admin.status || '').toUpperCase());
const isBlockingSurgery = (surgery) => surgery && typeof surgery === 'object' && ['URGENT', 'EMERGENCY'].includes((surgery.surgery_type || '').toUpperCase()) && !['COMPLETED', 'CANCELLED'].includes((surgery.status || '').toUpperCase());

export const updateDisposition = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    
    const { id } = req.params;
    const { disposition, disposition_notes, scheduled_at } = req.body; // disposition = "HOSPITALIZED" or "DISCHARGED"
    
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: "Invalid encounter ID" });
    }

    if (!['HOSPITALIZED', 'DISCHARGED'].includes(disposition)) {
      return res.status(400).json({ message: "Invalid disposition. Must be HOSPITALIZED or DISCHARGED" });
    }

    const encounter = await Encounter.findById(id);
    
    if (!encounter) {
      return res.status(404).json({ message: "Encounter not found" });
    }

    const pendingLabs = (encounter.lab_requests || []).filter(isPendingLabRequest);
    const pendingMedicationRequests = (encounter.medication_requests || []).filter(isPendingMedicationRequest);
    const unadministeredMeds = (encounter.medication_administrations || []).filter(isPendingMedicationAdministration);
    const blockingSurgeries = (encounter.surgery_records || []).filter(isBlockingSurgery);

    if (disposition === 'DISCHARGED' && blockingSurgeries.length > 0) {
      return res.status(400).json({
        message: 'Discharge blocked because urgent or emergency surgery is still pending for this patient.',
        details: {
          blockingSurgeries: blockingSurgeries.map((s) => ({
            procedure_name: s.procedure_name,
            surgery_type: s.surgery_type,
            status: s.status,
            scheduled_at: s.scheduled_at
          }))
        }
      });
    }

    const now = new Date();
    const scheduleDate = scheduled_at ? new Date(scheduled_at) : null;
    const hasFutureDischarge = scheduleDate instanceof Date && !Number.isNaN(scheduleDate.getTime()) && scheduleDate > now;

    if (disposition === 'DISCHARGED' && hasFutureDischarge) {
      encounter.discharge_scheduled_at = scheduleDate;
      encounter.discharge_scheduled_by = req.user._id;
      encounter.discharge_schedule_notes = disposition_notes || '';
      encounter.disposition_notes = disposition_notes || '';
      encounter.disposition_decided_at = now;
      encounter.disposition_decided_by = req.user._id;

      if (!encounter.status_history) {
        encounter.status_history = [];
      }
      encounter.status_history.push({
        status: encounter.status || 'HOSPITALIZED',
        changed_by: req.user._id,
        changed_at: now,
        notes: `Discharge scheduled for ${scheduleDate.toISOString()}. ${disposition_notes || ''}`
      });

      if (encounter.assigned_bed && mongoose.Types.ObjectId.isValid(encounter.assigned_bed)) {
        const bed = await BedSchedule.findById(encounter.assigned_bed);
        if (bed) {
          bed.expectedDischargeTime = scheduleDate;
          await bed.save();
        }
      }

      await encounter.save();
      return res.status(200).json({
        message: `Discharge scheduled for ${scheduleDate.toISOString()}`,
        encounter: encounter.toObject(),
        warnings: {
          pendingLabs: pendingLabs.length,
          pendingMedicationRequests: pendingMedicationRequests.length,
          unadministeredMeds: unadministeredMeds.length
        }
      });
    }

    // Immediate discharge path
    encounter.discharge_scheduled_at = null;
    encounter.discharge_schedule_notes = '';
    encounter.disposition_notes = disposition_notes || '';
    encounter.disposition_decided_at = now;
    encounter.disposition_decided_by = req.user._id;
    encounter.status = disposition;

    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: disposition,
      changed_by: req.user._id,
      changed_at: now,
      notes: `Patient ${disposition === 'HOSPITALIZED' ? 'admitted to hospital' : 'discharged'}. ${disposition_notes || ''}`
    });

    encounter.clinical_handovers?.forEach(h => {
      if (h.is_active) {
        h.handover_status = 'COMPLETED';
      }
    });

    if (disposition === 'DISCHARGED') {
      // Fetch full clinician record with credentials for signature
      let signingClinician = null;
      if (req.user?._id) {
        try {
          const { Practitioner } = await import('../models/clinicalstaff.model.js');
          signingClinician = await Practitioner.findById(req.user._id).lean();
        } catch (err) {
          console.warn('[Encounter] Failed to fetch clinician for discharge signature:', err.message);
        }
      }

      // Build discharge report with clinician signature
      encounter.discharge_report = buildDischargeReportData({ 
        encounter, 
        clinician: signingClinician,
        overrides: { authorId: req.user?.name || req.user?.email || '' } 
      });
      encounter.discharge_report.status = 'draft';
      encounter.discharge_report.updatedAt = now;
    }

    if (disposition === 'DISCHARGED' && encounter.assigned_bed && mongoose.Types.ObjectId.isValid(encounter.assigned_bed)) {
      const bed = await BedSchedule.findById(encounter.assigned_bed);
      if (bed) {
        bed.status = 'CLEANING';
        bed.statusChangedAt = now;
        bed.currentEncounterId = null;
        bed.assignedNurseId = null;
        bed.occupancyStartTime = null;
        bed.expectedDischargeTime = null;
        bed.totalOccupancyCount = (bed.totalOccupancyCount || 0) + 1;
        await bed.save();
      }
    }

    await encounter.save();

    console.log(`[Encounter] Patient disposition: ${disposition} for encounter ${id}`);
    console.log(`[Encounter] Encounter status: ${disposition}`);

    return res.status(200).json({
      message: `Patient ${disposition === 'HOSPITALIZED' ? 'admitted' : 'discharged'} successfully`,
      encounter: encounter.toObject(),
      warnings: {
        pendingLabs: pendingLabs.length,
        pendingMedicationRequests: pendingMedicationRequests.length,
        unadministeredMeds: unadministeredMeds.length
      }
    });
  } catch (err) {
    console.error("Disposition Error:", err);
    return res.status(500).json({
      message: "Failed to update patient disposition",
      error: err.message
    });
  }
};

// ========================
// MEDICATION ADMINISTRATION
// ========================
export const recordMedicationAdministration = async (req, res) => {
  try {
    const { id } = req.params;
    const {
      medication_name,
      medication_id,
      dosage,
      frequency,
      route,
      indication,
      prescription_notes,
      administered_at,
      status = 'ADMINISTERED',
      administered_by,
      prescribed_by_role
    } = req.body;

    if (!id || !medication_name || !dosage) {
      return res.status(400).json({
        message: "Encounter ID, medication name, and dosage are required"
      });
    }

    const encounter = await Encounter.findById(id);
    if (!encounter) {
      return res.status(404).json({ message: "Encounter not found" });
    }

    // Check access
    const access = await canAccessEncounter(req.user, encounter);
    if (!access) {
      return res.status(403).json({ message: "Access denied to this encounter" });
    }

    // Initialize medication_administrations array if it doesn't exist
    if (!encounter.medication_administrations) {
      encounter.medication_administrations = [];
    }

    // Create medication administration record
    const medicationRecord = {
      _id: new mongoose.Types.ObjectId(),
      medication_name,
      medication_id: medication_id || null,
      dosage,
      frequency: frequency || 'As prescribed',
      route: route || 'Oral',
      indication: indication || 'No indication provided',
      prescription_notes: prescription_notes || '',
      administered_at: administered_at ? new Date(administered_at) : new Date(),
      recorded_at: new Date(),
      status,
      administered_by: administered_by || req.user?.name || 'System',
      prescribed_by_role: prescribed_by_role || 'PHYSICIAN'
    };

    encounter.medication_administrations.push(medicationRecord);

    // Update status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: 'MEDICATION_ADMINISTERED',
      changed_by: req.user._id,
      changed_at: new Date(),
      notes: `Medication administered: ${medication_name} (${dosage})`
    });

    await encounter.save();

    console.log(`[Medication Administration] Recorded for encounter ${id}: ${medication_name}`);

    return res.status(200).json({
      message: 'Medication administration recorded successfully',
      encounter: encounter.toObject(),
      medicationRecord
    });
  } catch (err) {
    console.error('Medication Administration Error:', err);
    return res.status(500).json({
      message: 'Failed to record medication administration',
      error: err.message
    });
  }
};

export const createSurgeryRecord = async (req, res) => {
  try {
    const { id } = req.params;
    const {
      procedure_name,
      surgery_type = 'ELECTIVE',
      rationale,
      priority = 'MEDIUM',
      scheduled_at,
      surgeon,
      surgical_team = [],
      notes
    } = req.body;

    if (!id || !procedure_name) {
      return res.status(400).json({ message: 'Encounter ID and procedure name are required' });
    }

    const encounter = await Encounter.findById(id);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    let access = await canAccessEncounter(req.user, encounter);
    // Allow physicians/surgeons to create surgery records for encounters within the same hospital
    // even when not explicitly assigned (common workflow where surgeons schedule cases).
    if (!access) {
      const role = (req.user?.role || '').toString().toUpperCase();
      const isPhysician = ['PHYSICIAN', 'DOCTOR', 'SURGEON', 'SPECIALIST'].includes(role);
      const sameHospital = req.user?.hospitalId && encounter?.hospitalId && req.user.hospitalId.toString() === encounter.hospitalId.toString();
      if (isPhysician && sameHospital) {
        access = true;
      }
    }

    if (!access) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    if (!encounter.surgery_records) {
      encounter.surgery_records = [];
    }

    const surgeryRecord = {
      _id: new mongoose.Types.ObjectId(),
      procedure_name,
      surgery_type,
      rationale: rationale || '',
      priority,
      scheduled_at: scheduled_at ? new Date(scheduled_at) : null,
      surgeon: surgeon || req.user?.name || 'Unknown',
      surgical_team: Array.isArray(surgical_team) ? surgical_team : [String(surgical_team).trim()].filter(Boolean),
      requested_by: req.user?.name || 'Physician',
      requested_at: new Date(),
      status: scheduled_at ? 'SCHEDULED' : 'PENDING',
      recorded_by: req.user?.name || 'Physician',
      recorded_at: new Date(),
      updated_at: new Date(),
      notes: notes || ''
    };

    encounter.surgery_records.push(surgeryRecord);

    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: 'SURGERY_REQUESTED',
      changed_by: req.user._id,
      changed_at: new Date(),
      notes: `Surgery requested: ${procedure_name}`
    });

    await encounter.save();

    return res.status(201).json({
      message: 'Surgery record created successfully',
      encounter: encounter.toObject(),
      surgeryRecord
    });
  } catch (err) {
    console.error('Create Surgery Record Error:', err);
    return res.status(500).json({
      message: 'Failed to create surgery record',
      error: err.message
    });
  }
};

export const updateSurgeryRecord = async (req, res) => {
  try {
    const { id, surgeryId } = req.params;
    const {
      status,
      outcome,
      findings,
      complications,
      performed_at,
      scheduled_at,
      surgeon,
      surgical_team,
      notes
    } = req.body;

    if (!id || !surgeryId) {
      return res.status(400).json({ message: 'Encounter ID and surgery ID are required' });
    }

    const encounter = await Encounter.findById(id);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    const access = await canAccessEncounter(req.user, encounter);
    if (!access) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    const surgeryRecord = encounter.surgery_records?.find(sr => sr._id?.toString() === surgeryId);
    if (!surgeryRecord) {
      return res.status(404).json({ message: 'Surgery record not found' });
    }

    if (status) surgeryRecord.status = status;
    if (outcome) surgeryRecord.outcome = outcome;
    if (findings) surgeryRecord.findings = findings;
    if (complications) surgeryRecord.complications = complications;
    if (performed_at) surgeryRecord.performed_at = new Date(performed_at);
    if (scheduled_at) surgeryRecord.scheduled_at = new Date(scheduled_at);
    if (surgeon) surgeryRecord.surgeon = surgeon;
    if (surgical_team) surgeryRecord.surgical_team = Array.isArray(surgical_team) ? surgical_team : [String(surgical_team).trim()].filter(Boolean);
    if (notes) surgeryRecord.notes = notes;
    surgeryRecord.updated_at = new Date();

    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: `SURGERY_${(status || surgeryRecord.status).replace(/\s+/g, '_')}`,
      changed_by: req.user._id,
      changed_at: new Date(),
      notes: `Surgery ${surgeryRecord.procedure_name} updated to ${surgeryRecord.status}`
    });

    await encounter.save();

    return res.status(200).json({
      message: 'Surgery record updated successfully',
      encounter: encounter.toObject(),
      surgeryRecord
    });
  } catch (err) {
    console.error('Update Surgery Record Error:', err);
    return res.status(500).json({
      message: 'Failed to update surgery record',
      error: err.message
    });
  }
};

export const getSurgeryRecords = async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) {
      return res.status(400).json({ message: 'Encounter ID is required' });
    }

    const encounter = await Encounter.findById(id);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    const access = await canAccessEncounter(req.user, encounter);
    if (!access) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    return res.status(200).json({
      message: 'Surgery records retrieved successfully',
      surgeryRecords: encounter.surgery_records || []
    });
  } catch (err) {
    console.error('Get Surgery Records Error:', err);
    return res.status(500).json({
      message: 'Failed to retrieve surgery records',
      error: err.message
    });
  }
};

/**
 * Get all surgeries for a physician (aggregated across encounters)
 * Supports filtering by status
 */
export const getSurgeriesForPhysician = async (req, res) => {
  try {
    const { status } = req.query;
    const hospitalId = req.user?.hospitalId || req.hospitalId;

    if (!hospitalId) {
      return res.status(400).json({ message: 'Hospital ID not found' });
    }

    // Build query to find encounters with surgeries
    const query = {
      hospitalId: hospitalId,
      'surgery_records.0': { $exists: true } // has at least one surgery record
    };

    // Add status filter if provided
    if (status) {
      query['surgery_records.status'] = status;
    }

    // Find encounters with surgery records
    const encounters = await Encounter.find(query).lean();

    // Flatten surgeries with encounter context
    const surgeries = [];
    encounters.forEach(encounter => {
      let surgeryRecords = encounter.surgery_records || [];

      // Apply status filter
      if (status) {
        surgeryRecords = surgeryRecords.filter(s => s.status === status);
      }

      surgeryRecords.forEach(surgery => {
        surgeries.push({
          _id: surgery._id,
          encounterId: encounter._id,
          procedure_name: surgery.procedure_name || 'Surgery',
          procedure_type: surgery.procedure_type,
          priority: surgery.priority,
          status: surgery.status,
          scheduled_at: surgery.scheduled_at,
          surgery_start: surgery.surgery_start,
          surgery_end: surgery.surgery_end,
          surgeon: surgery.surgeon,
          theatre: surgery.theatre,
          anesthesia: surgery.anesthesia,
          findings: surgery.findings,
          pre_op_notes: surgery.pre_op_notes,
          post_op_notes: surgery.post_op_notes,
          complications: surgery.complications,
          estimated_duration: surgery.estimated_duration,
          actual_duration: surgery.actual_duration,

          // Encounter context
          patientName: encounter.patient?.name || encounter.patientName || 'Unknown',
          patientAge: encounter.patient?.age || 'N/A',
          patientId: encounter.patient?.id_number || 'N/A',
          diagnosis: encounter.diagnosis || '',
          ward: encounter.assigned_ward?.name || 'N/A'
        });
      });
    });

    // Sort by scheduled_at (newest first)
    surgeries.sort((a, b) => new Date(b.scheduled_at) - new Date(a.scheduled_at));

    return res.status(200).json({
      count: surgeries.length,
      surgeries: surgeries
    });
  } catch (err) {
    console.error('[ENCOUNTER CONTROLLER] getSurgeriesForPhysician error:', err);
    return res.status(500).json({ message: 'Error fetching surgeries', error: err.message });
  }
};

/**
 * Get the authenticated user's personal activity history
 * based on their JWT identity and the status_history / handover records they created.
 */
export const getUserHistoryActions = async (req, res) => {
  try {
    const userId = req.user?._id || req.user?.id;
    const hospitalId = req.user?.hospitalId || req.hospitalId;

    console.log('[BACKEND] getUserHistoryActions start', {
      userId: userId?.toString?.(),
      hospitalId: hospitalId?.toString?.(),
      authUser: req.user ? { _id: req.user._id?.toString?.(), email: req.user.email, role: req.user.role } : null
    });

    if (!userId || !hospitalId) {
      console.log('[BACKEND] getUserHistoryActions unauthorized', { userId, hospitalId });
      return res.status(401).json({ message: 'Not authorized' });
    }

    const encounters = await Encounter.find({
      hospitalId: new mongoose.Types.ObjectId(hospitalId),
      $or: [
        { 'status_history.changed_by': new mongoose.Types.ObjectId(userId) },
        { 'clinical_handovers.clinician_id': new mongoose.Types.ObjectId(userId) }
      ]
    })
      .select('patient status status_history clinical_handovers createdAt updatedAt')
      .sort({ updatedAt: -1 })
      .lean();

    console.log('[BACKEND] getUserHistoryActions raw encounters count', encounters.length);
    console.log('[BACKEND] getUserHistoryActions raw encounters sample', JSON.stringify(encounters.slice(0, 2), null, 2));

    const events = [];

    for (const encounter of encounters) {
      const patientName = encounter.patient?.name || 'Unknown Patient';
      const patientId = encounter.patient?._id || encounter.patient?.id_number || null;

      for (const event of encounter.status_history || []) {
        const eventActorId = event.changed_by?.toString?.();
        if (!eventActorId || eventActorId !== userId.toString()) continue;

        events.push({
          id: `${encounter._id}-status-${event._id || event.changed_at || Date.now()}`,
          encounterId: encounter._id,
          patientName,
          patientId,
          status: event.status || encounter.status || 'ACTION',
          timestamp: event.changed_at || event.updatedAt || encounter.updatedAt,
          notes: event.notes || '',
          action: 'status_change',
          updatedBy: req.user?.name || req.user?.email || 'You'
        });
      }

      for (const handover of encounter.clinical_handovers || []) {
        const handoverActorId = handover.clinician_id?.toString?.();
        if (!handoverActorId || handoverActorId !== userId.toString()) continue;

        events.push({
          id: `${encounter._id}-handover-${handover._id || handover.handover_at || Date.now()}`,
          encounterId: encounter._id,
          patientName,
          patientId,
          status: 'HANDOVER',
          timestamp: handover.handover_at || handover.created_at || encounter.updatedAt,
          notes: handover.acknowledged_notes || handover.summary || `Handover ${handover.handover_status || 'updated'}`,
          action: 'handover',
          updatedBy: req.user?.name || req.user?.email || 'You',
          handoverTo: handover.clinician_name || 'Assigned clinician'
        });
      }
    }

    events.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    console.log('[BACKEND] getUserHistoryActions final events count', events.length);
    console.log('[BACKEND] getUserHistoryActions final events', JSON.stringify(events.slice(0, 10), null, 2));

    return res.status(200).json({
      count: events.length,
      events
    });
  } catch (error) {
    console.error('[ENCOUNTER CONTROLLER] getUserHistoryActions error:', error);
    return res.status(500).json({ message: 'Error fetching user history', error: error.message });
  }
};

/**
 * Get all encounters for a physician (aggregated)
 * Encounters where physician is assigned or has active handovers
 */
export const getEncountersForPhysician = async (req, res) => {
  try {
    const { status } = req.query;
    const physicianId = req.user?.id || req.user?._id;
    const hospitalId = req.user?.hospitalId || req.hospitalId;

    if (!physicianId || !hospitalId) {
      return res.status(401).json({ message: 'Not authorized' });
    }

    // Build query to find encounters for this physician
    const query = {
      hospitalId: hospitalId,
      $or: [
        { 'assigned_clinicians.id': new mongoose.Types.ObjectId(physicianId) },
        { 'clinical_handovers.clinician_id': new mongoose.Types.ObjectId(physicianId) },
        { 'assigned_physician.physician_id': new mongoose.Types.ObjectId(physicianId) }
      ]
    };

    // Add status filter if provided
    if (status) {
      query.status = status;
    }

    const encounters = await Encounter.find(query)
      .select('patient assigned_physician assigned_ward assigned_bed status diagnosis createdAt updatedAt patient triage')
      .sort({ createdAt: -1 })
      .lean();

    // Format encounters
    const formatted = encounters.map(enc => ({
      _id: enc._id,
      patientName: enc.patient?.name || 'Unknown',
      patientId: enc.patient?.id_number || 'N/A',
      diagnosis: enc.diagnosis || '',
      status: enc.status || 'AWAITING_PHYSICIAN',
      ward: enc.assigned_ward?.name || 'N/A',
      createdAt: enc.createdAt,
      updatedAt: enc.updatedAt,
      triageId: enc.triage?._id || null
    }));

    return res.status(200).json({
      count: formatted.length,
      encounters: formatted
    });
  } catch (err) {
    console.error('[ENCOUNTER CONTROLLER] getEncountersForPhysician error:', err);
    return res.status(500).json({ message: 'Error fetching encounters', error: err.message });
  }
};