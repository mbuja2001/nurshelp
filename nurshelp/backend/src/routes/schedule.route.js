import express from "express";
import {
  // Physician
  getAvailablePhysicians,
  getPhysicianSchedule,
  upsertPhysicianSchedule,
  // Nurse
  getAvailableNurses,
  getNurseSchedule,
  upsertNurseSchedule,
  // Bed
  getAvailableBeds,
  getAllBeds,
  assignBed,
  releaseBed,
  initializeBeds
} from "../controllers/schedule.controller.js";

const router = express.Router();

// ============================================================
// PHYSICIAN SCHEDULE ENDPOINTS
// ============================================================

/**
 * GET /api/schedules/physicians
 * Query params: date, specialty, status
 */
router.get("/physicians", getAvailablePhysicians);

/**
 * GET /api/schedules/physicians/me
 * Query params: date
 */
router.get("/physicians/me", getPhysicianSchedule);

/**
 * POST /api/schedules/physicians
 * Create/Update single physician schedule
 */
router.post("/physicians", upsertPhysicianSchedule);

// ============================================================
// NURSE SCHEDULE ENDPOINTS
// ============================================================

/**
 * GET /api/schedules/nurses
 * Query params: date, status, esiLevel
 */
router.get("/nurses", getAvailableNurses);

/**
 * GET /api/schedules/nurses/me
 * Query params: date
 */
router.get("/nurses/me", getNurseSchedule);

/**
 * POST /api/schedules/nurses
 * Create/Update single nurse schedule
 */
router.post("/nurses", upsertNurseSchedule);

// ============================================================
// BED SCHEDULE ENDPOINTS
// ============================================================

/**
 * GET /api/schedules/beds
 * Query params: department, bedType, acuityLevel
 */
router.get("/beds", getAvailableBeds);

/**
 * GET /api/schedules/beds/all
 * Query params: department
 */
router.get("/beds/all", getAllBeds);

/**
 * POST /api/schedules/beds/assign
 * Assign bed to patient and nurse
 */
router.post("/beds/assign", assignBed);

/**
 * POST /api/schedules/beds/release
 * Release bed (mark for cleaning)
 */
router.post("/beds/release", releaseBed);

/**
 * POST /api/schedules/beds/initialize
 * Create default bed layout
 */
router.post("/beds/initialize", initializeBeds);

export default router;
