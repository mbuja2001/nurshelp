import express from 'express';
import { appendFileSync } from 'node:fs';
import { authenticateToken } from '../middleware/auth.middleware.js';
import {
  createPrescription,
  getPrescription,
  getPrescriptionsByEncounter,
  getOrderedMedsForPhysician,
  submitPrescriptionToPharmacy,
  updatePrescriptionItemStatus,
  cancelPrescription
} from '../controllers/prescriptionController.js';

const router = express.Router();
const debugLogPath = new URL('../temp-prescription-route.log', import.meta.url).pathname;

// Middleware to apply to all prescription routes
router.use((req, res, next) => {
  try {
    appendFileSync(debugLogPath, `${new Date().toISOString()} ${req.method} ${req.originalUrl} -> ${req.path}\n`);
  } catch (err) {
    console.error('Failed to write prescription route debug log', err);
  }
  next();
});
router.use(authenticateToken);

// ============ PRESCRIPTION ROUTES ============

// POST /api/prescriptions - Create new prescription
router.post('/', createPrescription);

// GET /api/prescriptions/ordered-meds - Get ordered medication reports for a physician
router.get('/ordered-meds', getOrderedMedsForPhysician);

// GET /api/prescriptions/:prescriptionId - Get single prescription
router.get('/:prescriptionId', getPrescription);

// GET /api/prescriptions/encounter/:encounterId - Get all prescriptions for encounter
router.get('/encounter/:encounterId', getPrescriptionsByEncounter);

// POST /api/prescriptions/:prescriptionId/submit - Submit to pharmacy
router.post('/:prescriptionId/submit', submitPrescriptionToPharmacy);

// PATCH /api/prescriptions/:prescriptionId/items/:itemId - Update item status
router.patch('/:prescriptionId/items/:itemId', updatePrescriptionItemStatus);

// POST /api/prescriptions/:prescriptionId/cancel - Cancel prescription
router.post('/:prescriptionId/cancel', cancelPrescription);

export default router;
