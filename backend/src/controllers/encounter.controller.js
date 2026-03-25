import mongoose from "mongoose";
import axios from "axios";
import { Encounter } from "../models/encounter.model.js";
import { Vital } from "../models/vitals.model.js";

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
    const transcriptArray = parseTranscript(payload.transcript);

    // Build enriched payload for the AI (strict schema per SATS integration)
    const aiPayload = {
      patient: {
        id_number: payload.patient?.id_number || "",
        name: payload.patient?.name || "",
        surname: payload.patient?.surname || "",
        gender: payload.patient?.gender || "",
        age: payload.patient?.age ?? null,
        height_cm: payload.patient?.height_cm ?? null,
        symptoms: payload.patient?.symptoms || "",
        history: payload.patient?.history || "",
        duration: payload.patient?.duration || "",
        painLevel: payload.patient?.painLevel || ""
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
        avpu: payload.vitals?.avpu ?? null
      },
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
      nurse_id: nurseId,
      patient: aiPayload.patient,
      transcript: transcriptArray,
      vitals: cleanVitals,
      triage: triage,
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
    const transcriptArray = parseTranscript(req.body.transcript);

    const encounter = await Encounter.create({
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

    encounter.status = "completed";
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

    const encounters = await Encounter.find({ nurse_id: nurseId, isWaiting: true }).sort({ createdAt: 1 });

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

    const nurseId = req.user._id;
    const encounter = await Encounter.findOne({ _id: id, nurse_id: nurseId });
    if (!encounter) return res.status(404).json({ message: "Encounter not found" });

    const obj = encounter.toObject();
    obj.waitingSeverity = calculateWaitingSeverity(obj.createdAt);
    if (obj.triage && obj.triage.waitingSeverity) obj.waitingSeverity = obj.triage.waitingSeverity;

    return res.status(200).json({ encounter: obj });
  } catch (err) {
    console.error("Get Encounter Error:", err);
    return res.status(500).json({ message: "Failed to fetch encounter" });
  }
};