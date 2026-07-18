/**
 * Clinical operations routes
 * - Staff management
 * - Bed assignments
 * - Clinical handover
 * - Resource logging
 */

import express from "express";
import {
  getClockedInStaff,
  getAvailableBeds,
  assignStaffToEncounter,
  assignBedToEncounter
} from "../controllers/clinical.controller.js";
import { authenticateToken } from "../middleware/auth.middleware.js";

const router = express.Router();

// All clinical routes require authentication
router.use(authenticateToken);

/**
 * STAFF MANAGEMENT
 */

// GET /api/clinical/staff/clocked-in
// Get all clocked-in staff for the authenticated user's hospital (physicians, nurses, triage nurses)
// Optional query params: role=PHYSICIAN|NURSE|TRIAGE_NURSE
router.get("/staff/clocked-in", getClockedInStaff);

/**
 * BED MANAGEMENT
 */

// GET /api/clinical/beds/available/:hospitalId
// Get available beds for hospitalization
// Optional query params: ward=ICU|GENERAL|PEDIATRIC, status=AVAILABLE|CLEAN|MAINTENANCE
router.get("/beds/available/:hospitalId", getAvailableBeds);

/**
 * ENCOUNTER MANAGEMENT - STAFF ASSIGNMENT
 */

// POST /api/clinical/encounters/:encounterId/assign-staff
// Assign physician or nurse to encounter
// Body: { staffId, role: "PHYSICIAN" | "NURSE" }
router.post("/encounters/:encounterId/assign-staff", assignStaffToEncounter);

/**
 * ENCOUNTER MANAGEMENT - BED ASSIGNMENT
 */

// POST /api/clinical/encounters/:encounterId/assign-bed
// Assign bed to encounter (hospitalize) or remove bed (discharge)
// Body: { bedId, hospitalize: true|false }
router.post("/encounters/:encounterId/assign-bed", assignBedToEncounter);

export default router;
