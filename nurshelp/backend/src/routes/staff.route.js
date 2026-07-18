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
  requestBreak,
  approveLateArrival,
  approveBreakRequest,
  getAttendanceHistory,
  autoClockOutExpiredShifts
} from "../controllers/clinicalstaff.controller.js";
import { protect } from "../middleware/authmiddleware.js";
import { authorize, auditLog } from "../middleware/rbac.js";

const router = express.Router();

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
 * Protected: NURSE, PHYSICIAN only
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
  authorize('NURSE', 'PHYSICIAN'),
  auditLog('CLOCK_IN'),
  clockIn
);

/**
 * Clock Out
 * POST /api/staff/clock-out
 * Protected: NURSE, PHYSICIAN only
 * 
 * Body: {
 *   "notes": "Shift completed"
 * }
 */
router.post(
  "/clock-out",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  auditLog('CLOCK_OUT'),
  clockOut
);

/**
 * Break Start
 * POST /api/staff/break-start
 * Protected: NURSE, PHYSICIAN only
 * 
 * Body: {
 *   "reason": "Lunch break"
 * }
 */
router.post(
  "/break-start",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  auditLog('BREAK_START'),
  breakStart
);

/**
 * Break End
 * POST /api/staff/break-end
 * Protected: NURSE, PHYSICIAN only
 */
router.post(
  "/break-end",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  auditLog('BREAK_END'),
  breakEnd
);

// ============================================
// PHASE 4.3: STATUS & HISTORY (Protected)
// ============================================

/**
 * Get current clock status
 * GET /api/staff/clock-status
 * Protected: NURSE, PHYSICIAN only
 */
router.get(
  "/clock-status",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  getClockStatus
);

/**
 * Get today's record
 * GET /api/staff/today-record
 * Protected: NURSE, PHYSICIAN only
 * 
 * Returns today's clock record if it exists, otherwise returns null
 */
router.get(
  "/today-record",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  getTodayRecord
);

/**
 * Debug: List all today's records
 * GET /api/staff/debug/today-records
 * Protected: NURSE, PHYSICIAN only
 * Returns all clock records for today (useful for debugging)
 */
router.get(
  "/debug/today-records",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  debugTodayRecords
);

/**
 * Get clock history
 * GET /api/staff/clock-history
 * Protected: NURSE, PHYSICIAN only
 * Query: ?startDate=...&endDate=...&limit=50
 */
router.get(
  "/clock-history",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  getClockHistory
);

// ============================================
// PHASE 4.4: SHIFT & BREAK MANAGEMENT (Protected)
// ============================================

/**
 * Get available shifts
 * GET /api/staff/available-shifts
 * Protected: NURSE, PHYSICIAN only
 * 
 * Returns shifts that are assigned to the staff member
 */
router.get(
  "/available-shifts",
  protect,
  authorize('NURSE', 'PHYSICIAN'),
  getAvailableShifts
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
  authorize('NURSE', 'PHYSICIAN'),
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
