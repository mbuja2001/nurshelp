import express from "express";
import {
  startEncounter,
  getEncounterById,
  confirmEncounter,
  attendEncounter,
  assignPhysician,
  getWaitingEncounters,
  runTriage,
  updateAIFields,
  getEncountersForNurse,
  getClinicianDashboard,
  acknowledgeHandover,
  recordDiagnosis,
  createInpatientProgressReport,
  updateDisposition,
  recordMedicationAdministration,
  createSurgeryRecord,
  updateSurgeryRecord,
  getSurgeryRecords,
  getSurgeriesForPhysician,
  getEncountersForPhysician,
  getUserHistoryActions
} from "../controllers/encounter.controller.js";
import { createHandover } from "../controllers/handover.controller.js";
import { protect } from "../middleware/authmiddleware.js";
import { buildDischargeReportData } from "../utils/dischargeReportBuilder.js";

const router = express.Router();

// Public create triage endpoint is via /api/triage/run (handled in triage router)
// But we allow manual creation via protected route:
router.post("/", protect, startEncounter);

// Waiting queue (must come BEFORE /:id so '/waiting' isn't treated as an id)
router.get("/waiting", protect, getWaitingEncounters);

// Get all encounters for a physician (aggregated)
router.get("/physician/all", protect, getEncountersForPhysician);

// Get the current user's own action history from status changes and handovers
router.get("/my-history", protect, getUserHistoryActions);

// Get all surgeries for a physician (aggregated)
router.get("/physician/surgeries", protect, getSurgeriesForPhysician);

// Clinician dashboard - get encounters assigned to a specific clinician
// (must come BEFORE /:id so '/clinician' isn't treated as an id)
router.get("/clinician/:clinicianId", protect, getClinicianDashboard);

// Get all encounters for authenticated nurse
router.get("/", protect, getEncountersForNurse);

// Confirm patient report (only the assigned nurse may confirm)
router.patch("/:id/confirm", protect, confirmEncounter);

// Mark encounter as attended (physician)
router.put("/:id/attend", protect, attendEncounter);

// Assign physician / ward / bed
router.patch("/:id/assign", protect, assignPhysician);

// Get single encounter by id (must belong to nurse)
router.get("/:id", protect, getEncounterById);

// Update AI fields with manual overrides (nurse can override AI recommendations)
router.put("/:id/ai-fields", protect, updateAIFields);

// Initiate clinical handover to assigned clinicians
router.post("/:id/handover", protect, createHandover);

// Clinician acknowledges handover and transitions status to SEEN
router.put("/:id/handover/acknowledge", protect, acknowledgeHandover);

// Physician records diagnosis
router.put("/:id/diagnosis", protect, recordDiagnosis);

// Create a new inpatient progression report for an active stay
router.put("/:id/inpatient-report", protect, createInpatientProgressReport);

// Update patient discharge/hospitalization status
router.put("/:id/disposition", protect, updateDisposition);

router.get("/:id/discharge-report", protect, async (req, res) => {
  try {
    const encounter = await (await import('../models/encounter.model.js')).Encounter.findById(req.params.id);
    if (!encounter) return res.status(404).json({ message: 'Encounter not found' });
    const { canAccessReport, REPORT_TYPES } = await import('../utils/reportAccess.js');
    if (!canAccessReport({ user: req.user, encounter, reportType: REPORT_TYPES.DISCHARGE })) {
      return res.status(403).json({ message: 'Discharge report access denied' });
    }
    return res.json({ dischargeReport: encounter.discharge_report || null });
  } catch (error) {
    return res.status(500).json({ message: 'Unable to fetch discharge report', error: error.message });
  }
});

router.post("/:id/discharge-report", protect, async (req, res) => {
  try {
    const encounter = await (await import('../models/encounter.model.js')).Encounter.findById(req.params.id);
    if (!encounter) return res.status(404).json({ message: 'Encounter not found' });
    const { canAccessReport, REPORT_TYPES } = await import('../utils/reportAccess.js');
    if (!canAccessReport({ user: req.user, encounter, reportType: REPORT_TYPES.DISCHARGE, action: 'edit' })) {
      return res.status(403).json({ message: 'Discharge report authoring denied' });
    }
    
    // Fetch full clinician record with credentials for signature
    let signingClinician = null;
    if (req.user?._id) {
      try {
        const { Practitioner } = await import('../models/clinicalstaff.model.js');
        signingClinician = await Practitioner.findById(req.user._id).lean();
      } catch (err) {
        console.warn('[Discharge Route] Failed to fetch clinician for signature:', err.message);
      }
    }

    const { buildDischargeReportData } = await import('../utils/dischargeReportBuilder.js');
    const dischargeReport = buildDischargeReportData({ 
      encounter, 
      clinician: signingClinician,
      overrides: { ...req.body, authorId: req.user?.name || req.user?.email || '' } 
    });
    encounter.discharge_report = dischargeReport;
    await encounter.save();
    return res.status(201).json({ dischargeReport });
  } catch (error) {
    return res.status(500).json({ message: 'Unable to create discharge report', error: error.message });
  }
});

// Record medication administration from pharmacy inventory
router.post("/:id/medication-administration", protect, recordMedicationAdministration);

// Surgery workflow endpoints
router.post("/:id/surgery", protect, createSurgeryRecord);
router.get("/:id/surgery", protect, getSurgeryRecords);
router.patch("/:id/surgery/:surgeryId", protect, updateSurgeryRecord);

export default router;