/**
 * Hospital Routes
 * Registration and authentication for hospital setup
 */

import express from 'express';
import { createManagedHospital, getHospitalTrialMode, listManagedHospitals, loginManager, registerHospital, selectManagedHospital, updateManagedHospitalCredentials } from '../controllers/hospital.controller.js';
import { Manager } from '../models/manager.model.js';
import { protect } from '../middleware/authmiddleware.js';
import { authorize } from '../middleware/rbac.js';

const router = express.Router();

/**
 * POST /api/hospital/register
 * PHASE 1: Register new hospital and create admin/manager account
 * Public endpoint - no authentication required
 * 
 * Body:
 * {
 *   "hospitalName": "St. Mary's Hospital",
 *   "managerName": "John",
 *   "managerSurname": "Smith",
 *   "managerEmail": "john@stmarys.com",
 *   "managerPassword": "secure123",
 *   "department": "All"
 * }
 * Response: { token, user, hospital }
 */
router.post('/register', registerHospital);
router.post('/login', loginManager);
router.get('/clinical-trial-mode', protect, authorize('ADMIN', 'MANAGER', 'SUPERUSER', 'RADIOLOGIST', 'PATHOLOGIST'), getHospitalTrialMode);
router.get('/managed', protect, listManagedHospitals);
router.post('/managed', protect, createManagedHospital);
router.post('/managed/:hospitalId/select', protect, selectManagedHospital);
router.patch('/managed/:hospitalId/credentials', protect, updateManagedHospitalCredentials);

// ============================================
// DEBUG ENDPOINTS (Development Only)
// ============================================

if (process.env.NODE_ENV !== 'production') {
  /**
   * DEBUG: Show all managers in database
   * GET /api/hospital/debug/managers (development only)
   * Used to verify seeding and database state
   */
  router.get('/debug/managers', async (req, res) => {
    try {
      const managers = await Manager.find().select('-password');
      console.log('[DEBUG] Found managers:', managers.length);
      managers.forEach(m => {
        console.log(`  - ${m.name} (${m.email}) [ID: ${m._id}] [Role: ${m.role}]`);
      });
      res.json({
        count: managers.length,
        managers: managers.map(m => ({
          _id: m._id,
          name: m.name,
          email: m.email,
          role: m.role,
          hospitalId: m.hospitalId
        }))
      });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });
}

export default router;
