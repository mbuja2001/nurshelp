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
  getSurgeryRecords
} from "../controllers/encounter.controller.js";
import { createHandover } from "../controllers/handover.controller.js";
import { protect } from "../middleware/authmiddleware.js";

const router = express.Router();

// Public create triage endpoint is via /api/triage/run (handled in triage router)
// But we allow manual creation via protected route:
router.post("/", protect, startEncounter);

// Waiting queue (must come BEFORE /:id so '/waiting' isn't treated as an id)
router.get("/waiting", protect, getWaitingEncounters);

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

// Record medication administration from pharmacy inventory
router.post("/:id/medication-administration", protect, recordMedicationAdministration);

// Surgery workflow endpoints
router.post("/:id/surgery", protect, createSurgeryRecord);
router.get("/:id/surgery", protect, getSurgeryRecords);
router.patch("/:id/surgery/:surgeryId", protect, updateSurgeryRecord);

export default router;