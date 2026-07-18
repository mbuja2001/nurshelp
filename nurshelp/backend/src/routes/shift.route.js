/**
 * Shift Management Routes
 * All routes require Manager/Admin role
 */

import express from 'express';
import { protect } from '../middleware/authmiddleware.js';
import { authorize, restrictByDepartment, auditLog } from '../middleware/rbac.js';
import {
  createShift,
  getShifts,
  getShiftsForDay,
  getShiftsForWeek,
  getShiftsForMonth,
  publishShift,
  assignStaffToShift,
  removeStaffFromShift,
  updateShift,
  deleteShift,
  getAvailableStaff,
  getAssignedShiftsForUser,
  getTodayAssignedShift,
  assignClinicalTeam
} from '../controllers/shift.controller.js';

const router = express.Router();

// ============ ALL SHIFT ROUTES REQUIRE MANAGER/ADMIN ============
// protect: Authenticate user
// authorize: Ensure user is MANAGER or ADMIN
// restrictByDepartment: Managers only see their department
// auditLog: Log sensitive actions

// ============ SHIFT CREATION & MANAGEMENT ============
router.post(
  '/create',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('SHIFT_CREATED'),
  createShift
);

router.get(
  '/list',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  getShifts
);

router.get(
  '/day',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  getShiftsForDay
);

router.get(
  '/week',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  getShiftsForWeek
);

router.get(
  '/month',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  getShiftsForMonth
);

router.put(
  '/:shiftID',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('SHIFT_UPDATED'),
  updateShift
);

router.delete(
  '/:shiftID',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('SHIFT_DELETED'),
  deleteShift
);

router.post(
  '/:shiftID/publish',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('SHIFT_PUBLISHED'),
  publishShift
);

// ============ STAFF ASSIGNMENT ============
router.post(
  '/:shiftID/assign-staff',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('STAFF_ASSIGNED'),
  assignStaffToShift
);

router.post(
  '/:shiftID/remove-staff',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('STAFF_REMOVED'),
  removeStaffFromShift
);

// ============ ASSIGN CLINICAL TEAM (TRIAGE/CLINICAL/PHYSICIAN) ============
// New endpoint for role-based team assignment in the full day simulation
// This determines which dashboard each staff member will see
router.post(
  '/assign-clinical-team',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('CLINICAL_TEAM_ASSIGNED'),
  assignClinicalTeam
);

// ============ STAFF AVAILABILITY (MANAGERS/ADMIN ONLY) ============
router.get(
  '/staff/available',
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  getAvailableStaff
);

// ============ ASSIGNED SHIFTS FOR AUTHENTICATED USER ============
// Allows staff to view their assigned upcoming shifts
router.get(
  '/assigned',
  protect,
  getAssignedShiftsForUser
);

// ============ TODAY'S ASSIGNED SHIFT FOR CLOCK-IN VALIDATION ============
// Allows staff to fetch today's shift for clock-in eligibility check
router.get(
  '/assigned/today',
  protect,
  getTodayAssignedShift
);

export default router;
