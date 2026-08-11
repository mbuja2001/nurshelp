import express from "express";
import { protect } from "../middleware/authmiddleware.js";
import { runTriage } from "../controllers/encounter.controller.js";
import { 
  requestTriageWithHospitalShifts,
  getAvailableStaffForHospital,
  getStaffForTriage,
  submitManualTriage,
  confirmTriageHandler,
  specialistFeedbackHandler
} from "../controllers/triage.controller.js";

const router = express.Router();

// ========== OLD ENDPOINT (Backward Compatible) ==========
// Frontend calls: POST /api/triage/run
router.post("/run", runTriage);

// ========== NEW ENDPOINTS (Manual Triage Workflow) ==========

/**
 * GET /api/triage/staff?date=YYYY-MM-DD
 * Fetch available nurses, physicians, and beds for staff assignment UI
 * LIVE DATA from scheduling system
 * Handler: getStaffForTriage (triage.controller.js)
 */
router.get("/staff", getStaffForTriage);

/**
 * POST /api/triage
 * Submit manual triage (or AI-assisted with computeMode)
 * Creates an Encounter record in database for NurseDashboard to display
 * 
 * KEY: If computeMode is absent/null = MANUAL path (no AI)
 *      If computeMode is present = AI-assisted path
 * Handler: submitManualTriage (triage.controller.js)
 */
router.post("/", protect, submitManualTriage);

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
 * Handler: confirmTriageHandler (triage.controller.js)
 */
router.patch("/:id/confirm", confirmTriageHandler);

/**
 * PATCH /api/triage/:id/specialist-feedback
 * Add specialist review after patient seen by specialist
 * Handler: specialistFeedbackHandler (triage.controller.js)
 */
router.patch("/:id/specialist-feedback", specialistFeedbackHandler);

// ========== HOSPITAL-SPECIFIC TRIAGE WITH SHIFT ASSIGNMENT ==========

/**
 * POST /api/triage/hospital-request
 * New endpoint for hospital-specific triage with automatic staff assignment
 * Request body:
 * {
 *   "hospitalId": "...",
 *   "patient": { "name": "...", "age": 45, "symptoms": "..." },
 *   "vitals": { "bp_systolic": 140, "bp_diastolic": 90, "hr": 100, "temp": 98.6, "rr": 20 },
 *   "transcript": "Optional clinical notes"
 * }
 * 
 * Response includes:
 * - ESI triage result
 * - Hospital-specific context (available shifts, staff counts)
 * - Automatic assignment: physician + clinical nurse
 */
router.post("/hospital-request", requestTriageWithHospitalShifts);

/**
 * GET /api/triage/hospital/:hospitalId/available-staff
 * Get available staff for a hospital today
 * Useful for UI display or ESI_Engine.py to preview available team
 */
router.get("/hospital/:hospitalId/available-staff", getAvailableStaffForHospital);

export default router;