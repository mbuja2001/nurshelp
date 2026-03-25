import mongoose from "mongoose";
import axios from "axios";
import { Encounter } from "../models/encounter.model.js";

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
 * ----------------------
 */
const PYTHON_URL = process.env.PYTHON_URL || "http://localhost:5000/triage";

export const runTriage = async (req, res) => {
  try {
    const payload = req.body;

    if (!payload || !payload.patient || !payload.vitals) {
      return res.status(400).json({ message: "Invalid payload" });
    }

    const nurseId = req.user?._id || null; // nullable

    const transcriptArray = parseTranscript(payload.transcript);

    console.log("➡️ Sending request to Python:", PYTHON_URL);
    const pythonRes = await axios.post(PYTHON_URL, payload, {
      timeout: 300000,
      headers: { "Content-Type": "application/json" }
    });

    const triage = pythonRes.data;

    const encounter = await Encounter.create({
      nurse_id: nurseId,
      patient: payload.patient,
      transcript: transcriptArray,
      vitals: payload.vitals,
      triage: { ...triage, severity: triage.ESI },
      nurseNotes: payload.nurseNotes || "",
      status: nurseId ? "pending" : "unassigned",
      submittedAt: null,
      isWaiting: true,
      severity: triage.ESI || 1
    });

    return res.status(201).json({ message: "Triage completed", encounter });
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
};