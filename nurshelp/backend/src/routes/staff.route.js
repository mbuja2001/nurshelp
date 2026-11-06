/**
 * Staff Routes (Clinical Staff - Nurses, Physicians, Specialists)
 * PHASE 4: Staff login and clock in/out workflows
 * 
 * NOTE: Staff registration is ONLY done by managers through /api/manager/staff/register
 * Staff cannot self-register. Managers create staff accounts with passwords.
 */

import express from "express";
import { 
  clockIn,
  clockOut,
  breakStart,
  breakEnd,
  getClockStatus,
  getTodayRecord,
  debugTodayRecords,
  getClockHistory,
  getAvailableShifts,
  getStaffProfile,
  updateStaffProfile,
  updateStaffShift,
  requestBreak,
  approveLateArrival,
  approveBreakRequest,
  getAttendanceHistory,
  autoClockOutExpiredShifts
} from "../controllers/clinicalstaff.controller.js";
import { protect } from "../middleware/authmiddleware.js";
import { authorize, auditLog } from "../middleware/rbac.js";

const router = express.Router();
const STAFF_ATTENDANCE_ROLES = [
  'NURSE', 'PHYSICIAN', 'PHARMACIST', 'PHARMACY_TECHNICIAN',
  'RADIOLOGIST', 'RADIOLOGY_TECHNICIAN', 'PATHOLOGIST', 'PATHOLOGY_TECHNICIAN', 'TECHNICIAN'
];

// ============================================
// NOTE: Staff login moved to unified /api/auth/login
// This endpoint is now deprecated - use /api/auth/login instead
// ============================================

// ============================================
// PHASE 4.2: CLOCK IN / OUT (Protected)
// Staff members clock in and out based on assigned shifts
// ============================================

/**
 * Clock In
 * POST /api/staff/clock-in
 * Protected: authenticated clinical staff with a supported workforce role
 * 
 * Body: {
 *   "location": "ED",
 *   "shiftID": "...",
 *   "notes": "Arrived on time"
 * }
 */
router.post(
  "/clock-in",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  auditLog('CLOCK_IN'),
  clockIn
);

/**
 * Clock Out
 * POST /api/staff/clock-out
 * Protected: authenticated clinical staff with a supported workforce role
 * 
 * Body: {
 *   "notes": "Shift completed"
 * }
 */
router.post(
  "/clock-out",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  auditLog('CLOCK_OUT'),
  clockOut
);

/**
 * Break Start
 * POST /api/staff/break-start
 * Protected: authenticated clinical staff with a supported workforce role
 * 
 * Body: {
 *   "reason": "Lunch break"
 * }
 */
router.post(
  "/break-start",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  auditLog('BREAK_START'),
  breakStart
);

/**
 * Break End
 * POST /api/staff/break-end
 * Protected: authenticated clinical staff with a supported workforce role
 */
router.post(
  "/break-end",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  auditLog('BREAK_END'),
  breakEnd
);

// ============================================
// PHASE 4.3: STATUS & HISTORY (Protected)
// ============================================

/**
 * Get current clock status
 * GET /api/staff/clock-status
 * Protected: authenticated clinical staff with a supported workforce role
 */
router.get(
  "/clock-status",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  getClockStatus
);

/**
 * Get today's record
 * GET /api/staff/today-record
 * Protected: authenticated clinical staff with a supported workforce role
 * 
 * Returns today's clock record if it exists, otherwise returns null
 */
router.get(
  "/today-record",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  getTodayRecord
);

/**
 * Debug: List all today's records
 * GET /api/staff/debug/today-records
 * Protected: authenticated clinical staff with a supported workforce role
 * Returns all clock records for today (useful for debugging)
 */
router.get(
  "/debug/today-records",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  debugTodayRecords
);

/**
 * Get clock history
 * GET /api/staff/clock-history
 * Protected: authenticated clinical staff with a supported workforce role
 * Query: ?startDate=...&endDate=...&limit=50
 */
router.get(
  "/clock-history",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  getClockHistory
);

// ============================================
// PHASE 4.4: SHIFT & BREAK MANAGEMENT (Protected)
// ============================================

/**
 * Get available shifts
 * GET /api/staff/available-shifts
 * Protected: authenticated clinical staff with a supported workforce role
 * 
 * Returns shifts that are assigned to the staff member
 */
router.get(
  "/available-shifts",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  getAvailableShifts
);

/**
 * Get staff profile
 * GET /api/staff/profile
 * Protected: clinical staff roles
 */
router.get(
  "/profile",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  getStaffProfile
);

/**
 * Update staff profile
 * PATCH /api/staff/profile
 * Protected: clinical staff roles
 */
router.patch(
  "/profile",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  updateStaffProfile
);

/**
 * Update today's shift for authenticated staff
 * PATCH /api/staff/shift
 */
router.patch(
  "/shift",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  updateStaffShift
);

/**
 * Request break
 * POST /api/staff/break-request
 * Protected: NURSE, PHYSICIAN only
 * 
 * Body: {
 *   "reason": "Personal emergency",
 *   "duration": 30
 * }
 */
router.post(
  "/break-request",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  auditLog('BREAK_REQUESTED'),
  requestBreak
);

// ============================================
// PHASE 4.5: MANAGER APPROVALS (Protected)
// Managers approve late arrivals and break requests
// ============================================

/**
 * Approve late arrival
 * POST /api/staff/approve-late-arrival/:recordId
 * Protected: MANAGER, ADMIN only
 */
router.post(
  "/approve-late-arrival/:recordId",
  protect,
  authorize('MANAGER', 'ADMIN'),
  auditLog('LATE_ARRIVAL_APPROVED'),
  approveLateArrival
);

/**
 * Approve break request
 * POST /api/staff/approve-break/:recordId
 * Protected: MANAGER, ADMIN only
 */
router.post(
  "/approve-break/:recordId",
  protect,
  authorize('MANAGER', 'ADMIN'),
  auditLog('BREAK_APPROVED'),
  approveBreakRequest
);

// ============================================
// PHASE 4.6: ATTENDANCE HISTORY (Protected)
// ============================================

/**
 * Get attendance history
 * GET /api/staff/attendance-history
 * Protected: NURSE, PHYSICIAN only
 * 
 * Returns all clock records with flags and shift context
 * sorted by most recent first
 */
router.get(
  "/attendance-history",
  protect,
  authorize(...STAFF_ATTENDANCE_ROLES),
  getAttendanceHistory
);

// ============================================
// PHASE 4.7: AUTO CLOCK OUT (Protected - Manager/Admin)
// ============================================

/**
 * Auto Clock Out Expired Shifts
 * POST /api/staff/auto-clockout
 * Protected: MANAGER, ADMIN only
 * 
 * Automatically clocks out any staff members whose shift time has passed
 * Adds AUTO_CLOCKOUT and DIDNT_CLOCKOUT flags
 * This ensures no staff are left clocked in after their shift ends
 */
router.post(
  "/auto-clockout",  
  protect,
  authorize('MANAGER', 'ADMIN'),
  auditLog('AUTO_CLOCKOUT'),
  autoClockOutExpiredShifts
);

export default router;
