/**
 * Authentication Routes
 * Unified login endpoint for all user types (managers and staff)
 */

import express from 'express';
import { login, debugToken, findUserByEmail } from '../controllers/auth.controller.js';

const router = express.Router();

/**
 * POST /api/auth/login
 * Unified login endpoint for both managers and staff
 * Public endpoint - no authentication required
 * 
 * Checks user in both Manager and Nurse collections
 * Returns JWT token with role information
 * Role-based access control is handled by middleware on protected routes
 * 
 * Body:
 * {
 *   "email": "john@stmarys.com" OR "sarah@stmarys.com",
 *   "password": "admin123" OR "nurse123"
 * }
 * 
 * Response: 
 * {
 *   "success": true,
 *   "message": "Login successful",
 *   "token": "eyJhbGciOi...",
 *   "user": {
 *     "id": "...",
 *     "name": "John",
 *     "email": "john@stmarys.com",
 *     "role": "ADMIN" | "MANAGER" | "NURSE" | "PHYSICIAN",
 *     "department": "Emergency",
 *     "hospitalId": "...",
 *     "hospitalName": "St. Mary's Hospital",
 *     "permissions": {...}  // Only for managers
 *   }
 * }
 * 
 * Errors:
 * - 400: Email and password are required
 * - 401: Invalid credentials (user not found or wrong password)
 * - 500: Server error
 */
router.post('/login', login);

// ============================================
// DEBUG ENDPOINTS (Development Only)
// ============================================

if (process.env.NODE_ENV !== 'production') {
  /**
   * DEBUG: Inspect token payload
   * POST /api/auth/debug-token (development only)
   */
  router.post('/debug-token', debugToken);

  /**
   * DEBUG: Find user by email
   * POST /api/auth/find-user (development only)
   */
  router.post('/find-user', findUserByEmail);
}

export default router;
