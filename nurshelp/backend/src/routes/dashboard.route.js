/**
 * Dashboard Routes
 * All routes require authenticated user
 * Returns role-based configuration based on TODAY's shift assignment
 */

import express from 'express';
import { protect } from '../middleware/authmiddleware.js';
import {
  getDashboardConfig,
  getRolePermissions,
  getPatientQueueForContext
} from '../controllers/dashboard.controller.js';

const router = express.Router();

// ============ GET DASHBOARD CONFIGURATION ============
// Checks JWT token → finds today's shift → determines role → returns dashboard config
// Returns 403 if user not assigned to any role in today's shift
router.get(
  '/config',
  protect,
  getDashboardConfig
);

// ============ GET ROLE-BASED PERMISSIONS ============
// Returns permission matrix for all roles (reference endpoint)
router.get(
  '/permissions',
  protect,
  getRolePermissions
);

// ============ GET PATIENT QUEUE FOR CONTEXT ============
// Returns patients appropriate for staff member's assigned role
router.get(
  '/queue',
  protect,
  getPatientQueueForContext
);

export default router;
