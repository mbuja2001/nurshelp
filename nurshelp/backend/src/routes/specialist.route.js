import express from 'express';
import { protect } from '../middleware/authmiddleware.js';
// If you have an authorize function, import from rbac.js or define here
import { authorize } from '../middleware/rbac.js';
import * as specialistController from '../controllers/specialist.controller.js';

const router = express.Router();

// ============================================================
// RADIOLOGIST ROUTES
// ============================================================

// Get radiology requests queue for radiologist
router.get(
  '/radiology/queue',
  protect,
  authorize('RADIOLOGIST', 'RADIOLOGY_TECHNICIAN'),
  specialistController.getRadiologyQueue
);

// Update radiology order status
router.post(
  '/radiology/:requestId/status',
  protect,
  authorize('RADIOLOGIST', 'RADIOLOGY_TECHNICIAN'),
  specialistController.updateRadiologyStatus
);

// Submit radiology report results
router.post(
  '/radiology/:requestId/results',
  protect,
  authorize('RADIOLOGIST', 'RADIOLOGY_TECHNICIAN'),
  specialistController.submitRadiologyResults
);

// ============================================================
// PATHOLOGIST ROUTES
// ============================================================

// Get pathology requests queue for pathologist
router.get(
  '/pathology/queue',
  protect,
  authorize('PATHOLOGIST', 'PATHOLOGY_TECHNICIAN'),
  specialistController.getPathologyQueue
);

// Update lab order status
router.post(
  '/pathology/:requestId/status',
  protect,
  authorize('PATHOLOGIST', 'PATHOLOGY_TECHNICIAN'),
  specialistController.updatePathologyStatus
);

// Submit lab test results
router.post(
  '/pathology/:requestId/results',
  protect,
  authorize('PATHOLOGIST', 'PATHOLOGY_TECHNICIAN'),
  specialistController.submitPathologyResults
);

// ============================================================
// PHARMACIST ROUTES
// ============================================================

// Get pharmacy prescription queue
router.get(
  '/pharmacy/queue',
  protect,
  authorize('PHARMACIST', 'PHARMACY_TECHNICIAN'),
  specialistController.getPharmacyQueue
);

// Dispense medication
router.post(
  '/pharmacy/:requestId/dispense',
  protect,
  authorize('PHARMACIST', 'PHARMACY_TECHNICIAN'),
  specialistController.dispenseMedication
);

// ============================================================
// PHYSICIAN ENDPOINTS (View Specialist Results)
// ============================================================

// Get results for physician (radiology + pathology combined)
router.get(
  '/physician/results',
  protect,
  authorize('PHYSICIAN'),
  specialistController.getPhysicianResults
);

export default router;
