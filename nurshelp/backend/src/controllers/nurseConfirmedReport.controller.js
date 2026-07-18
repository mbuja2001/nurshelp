import mongoose from "mongoose";
import { NurseConfirmedReport } from "../models/nurseConfirmedReport.model.js";
import { Encounter } from "../models/encounter.model.js";

// Create a nurse-confirmed report (linked to an Encounter)
export const createNurseConfirmedReport = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const nurseId = req.user._id;
    const payload = req.body;
    const encounterId = payload.encounter;
    if (!encounterId || !mongoose.isValidObjectId(encounterId)) return res.status(400).json({ message: "Valid encounter id required" });

    // Optionally fetch encounter for validation
    const encounter = await Encounter.findById(encounterId);
    if (!encounter) return res.status(404).json({ message: "Encounter not found" });

    const report = await NurseConfirmedReport.create({
      encounter: encounterId,
      confirmedByNurse: nurseId,
      confirmedAt: new Date(),
      patient: payload.patient || encounter.patient || {},
      transcript: payload.transcript || encounter.transcript || [],
      vitals: payload.vitals || encounter.vitals || {},
      triage: payload.triage || encounter.triage || {},
      recommendations: payload.recommendations || encounter.recommendations || [],
      nurseNotes: payload.nurseNotes || encounter.nurseNotes || "",
      status: payload.status || "confirmed",
      severity: payload.severity ?? encounter.severity ?? 1,
      isWaiting: payload.isWaiting ?? encounter.isWaiting ?? true,
      attendedAt: payload.attendedAt || encounter.attendedAt || null
    });

    return res.status(201).json({ message: "NurseConfirmedReport created", report });
  } catch (err) {
    console.error("Create NurseConfirmedReport Error:", err);
    return res.status(500).json({ message: err.message || "Failed to create report" });
  }
};

// Update existing nurse-confirmed report (by its id)
export const updateNurseConfirmedReport = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const { id } = req.params;
    if (!mongoose.isValidObjectId(id)) return res.status(400).json({ message: "Invalid report id" });

    const payload = req.body;
    const report = await NurseConfirmedReport.findById(id);
    if (!report) return res.status(404).json({ message: "Report not found" });

    // Only allow nurse who created report (confirmedByNurse) or any nurse to update? For now allow the same nurse only
    if (report.confirmedByNurse && report.confirmedByNurse.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: "Forbidden: only creating nurse may modify this report" });
    }

    // merge allowed fields
    const allowed = ["patient","transcript","vitals","triage","recommendations","nurseNotes","status","severity","isWaiting","attendedAt"];
    allowed.forEach(k => { if (typeof payload[k] !== 'undefined') report[k] = payload[k]; });
    report.confirmedAt = new Date();
    await report.save();

    return res.status(200).json({ message: "NurseConfirmedReport updated", report });
  } catch (err) {
    console.error("Update NurseConfirmedReport Error:", err);
    return res.status(500).json({ message: err.message || "Failed to update report" });
  }
};

// Get nurse-confirmed report by encounter id (latest)
export const getReportByEncounter = async (req, res) => {
  try {
    if (!req.user?._id) return res.status(401).json({ message: "Not authorized" });
    const { encounterId } = req.params;
    if (!mongoose.isValidObjectId(encounterId)) return res.status(400).json({ message: "Invalid encounter id" });

    const report = await NurseConfirmedReport.findOne({ encounter: encounterId }).sort({ confirmedAt: -1 });
    if (!report) return res.status(404).json({ message: "No confirmed report for this encounter" });
    return res.status(200).json({ report });
  } catch (err) {
    console.error("Get Report By Encounter Error:", err);
    return res.status(500).json({ message: err.message || "Failed to get report" });
  }
};

export default {
  createNurseConfirmedReport,
  updateNurseConfirmedReport,
  getReportByEncounter
};
