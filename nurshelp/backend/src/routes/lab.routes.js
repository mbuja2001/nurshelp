import express from "express";
import {
  createLabRequest,
  getEncounterLabRequests,
  getLabRequestsForSpecialty,
  acknowledgeLabRequest,
  submitLabReport,
  declineLabRequest,
  reviewLabReport,
  getLabReportsForPhysician,
  getLabRequestsForPhysician
} from "../controllers/lab.controller.js";
import { protect } from "../middleware/authmiddleware.js"; // Authentication middleware

const router = express.Router();

/**
 * PHYSICIAN ENDPOINTS
 * These endpoints allow physicians to request labs and review reports
 */

// GET /api/lab-requests/physician/all?status=PENDING&specialty=RADIOLOGY
// Physician fetches all lab requests across all encounters (aggregated)
router.get("/physician/all", protect, getLabRequestsForPhysician);

// POST /api/lab-requests/:encounterId/create
// Physician creates a new lab request
router.post("/:encounterId/create", protect, createLabRequest);

// GET /api/lab-requests/:encounterId
// Get all lab requests for a specific encounter
router.get("/:encounterId", protect, getEncounterLabRequests);

// GET /api/lab-requests/queue?specialty=RADIOLOGY&status=PENDING
// Radiologist/Pathologist fetches their pending queue
router.get("/queue/requests", protect, getLabRequestsForSpecialty);

// GET /api/lab-reports?status=COMPLETED
// Physician fetches completed lab reports for review
router.get("/physician/reports", protect, getLabReportsForPhysician);

/**
 * SPECIALIST ENDPOINTS
 * These endpoints allow radiologists/pathologists to manage lab requests
 */

// PATCH /api/lab-requests/:encounterId/:requestId/acknowledge
// Specialist acknowledges and accepts a lab request
router.patch("/:encounterId/:requestId/acknowledge", protect, acknowledgeLabRequest);

// PATCH /api/lab-requests/:encounterId/:requestId/submit-report
// Specialist submits completed lab report with findings
router.patch("/:encounterId/:requestId/submit-report", protect, submitLabReport);

// PATCH /api/lab-requests/:encounterId/:requestId/decline
// Specialist declines a lab request with reason
router.patch("/:encounterId/:requestId/decline", protect, declineLabRequest);

/**
 * PHYSICIAN REVIEW ENDPOINTS
 */

// PATCH /api/lab-requests/:encounterId/:requestId/review
// Physician reviews and acknowledges completed lab report
router.patch("/:encounterId/:requestId/review", protect, reviewLabReport);

export default router;
