import express from "express";
import { protect } from "../middleware/authmiddleware.js";
import { authorize } from "../middleware/rbac.js";
import {
  createBedsBatch,
  getWardsWithBedStatus,
  getWardDetail,
  getHospitalBedStats,
  getBedEquipment,
  getLiveBedStatus,
  updateBedStatus,
  assignPatientToBed,
  dischargePatientFromBed,
  getBedHistory,
  getOccupancyAnalytics
  ,getBedConfigurations
  ,createBedConfiguration
  ,updateBedConfiguration
  ,deleteBedConfiguration
} from "../controllers/bed.controller.js";

const router = express.Router();

// =====================================================
// BATCH CREATION (Hospital Registration)
// =====================================================

/**
 * POST /api/beds/batch
 * Create multiple beds during hospital registration
 * Protected: Manager/Admin only
 */
router.post("/batch", protect, authorize('MANAGER', 'ADMIN'), createBedsBatch);
router.get("/configuration", protect, authorize('MANAGER', 'ADMIN'), getBedConfigurations);
router.post("/configuration", protect, authorize('MANAGER', 'ADMIN'), createBedConfiguration);
router.patch("/configuration/:bedId", protect, authorize('MANAGER', 'ADMIN'), updateBedConfiguration);
router.delete("/configuration/:bedId", protect, authorize('MANAGER', 'ADMIN'), deleteBedConfiguration);

// =====================================================
// Authenticated hospital-scoped live status
// =====================================================

/**
 * GET /api/beds/wards
 * Get all wards with live bed status
 */
router.get("/wards", protect, getWardsWithBedStatus);

/**
 * GET /api/beds/ward/:wardId
 * Get detailed ward view
 */
router.get("/ward/:wardId", protect, getWardDetail);

/**
 * GET /api/beds/stats
 * Get hospital-wide bed statistics
 */
router.get("/stats", getHospitalBedStats);

/**
 * GET /api/beds/live-status
 * Get real-time bed status across all wards
 */
router.get("/live-status", getLiveBedStatus);

/**
 * GET /api/beds/:bedId/equipment
 * Get equipment list for a bed (enriched)
 */
router.get("/:bedId/equipment", getBedEquipment);

// =====================================================
// PROTECTED ENDPOINTS (staff only)
// =====================================================

/**
 * PATCH /api/beds/:bedId/status
 * Update bed status (must be authenticated)
 */
router.patch("/:bedId/status", protect, updateBedStatus);

/**
 * POST /api/beds/:bedId/assign
 * Assign patient to bed
 */
router.post("/:bedId/assign", protect, assignPatientToBed);

/**
 * POST /api/beds/:bedId/discharge
 * Discharge patient from bed
 */
router.post("/:bedId/discharge", protect, dischargePatientFromBed);

// =====================================================
// ANALYTICS ENDPOINTS (protected)
// =====================================================

/**
 * GET /api/beds/:bedId/history
 * Get bed status change history
 */
router.get("/:bedId/history", protect, getBedHistory);

/**
 * GET /api/beds/analytics/occupancy
 * Get occupancy analytics
 */
router.get("/analytics/occupancy", protect, getOccupancyAnalytics);

export default router;
