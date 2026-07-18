/**
 * Pharmacy Routes
 * Handles pharmaceutical inventory management endpoints
 */

import express from 'express';
import multer from 'multer';
import { protect } from '../middleware/authmiddleware.js';
import { authorize } from '../middleware/rbac.js';
import {
  createPharmaceuticalsBatch,
  getPharmaceuticals,
  getPharmaceutical,
  updatePharmaceutical,
  updateStock,
  getLowStockItems,
  deletePharmaceutical,
  getPharmacyInventory,
  dispenseMedication,
  recordAdministration,
  refillMedication,
  addMedicationToStock,
  getMedicationDispensingHistory,
  getInventoryWithStats,
  getDispensingRecordsByDateRange,
  uploadCSVRefill,
  exportInventoryCSV,
  getPendingMedicationRequests,
  updateMedicationRequestStatus
} from '../controllers/pharmacy.controller.js';

const router = express.Router();

// Configure multer for CSV file uploads (memory storage to parse directly)
const upload = multer({
  storage: multer.memoryStorage(),
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'), false);
    }
  },
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
  }
});

// ============================================================
// BATCH OPERATIONS (Hospital Registration)
// ============================================================

/**
 * POST /api/pharmacy/batch
 * Create multiple pharmaceutical records during hospital registration
 * Protected: Manager/Admin only
 */
router.post('/batch', protect, authorize('MANAGER', 'ADMIN'), createPharmaceuticalsBatch);

// ============================================================
// CLINIC STAFF ENDPOINTS (All authenticated staff)
// ============================================================

/**
 * GET /api/pharmacy/inventory
 * Get pharmacy inventory for hospital (accessible to all clinical staff)
 */
router.get('/inventory', protect, getPharmacyInventory);

/**
 * POST /api/pharmacy/dispense
 * Dispense and handover medication to patient
 * Tracks: Pharmacist authorization, Prescriber (doctor/nurse), Schedule, Patient, Clinician giving med
 * Protected: All authenticated clinical staff
 */
router.post('/dispense', protect, dispenseMedication);

/**
 * POST /api/pharmacy/administration
 * Record medication administration with full audit trail
 * Protected: Clinical staff (Nurses, Physicians)
 */
router.post('/administration', protect, recordAdministration);

/**
 * POST /api/pharmacy/refill
 * Refill existing medication stock (restock)
 * Protected: Pharmacy staff, Managers
 */
router.post('/refill', protect, refillMedication);

/**
 * POST /api/pharmacy/add-medication
 * Add new medication to hospital pharmacy stock
 * Protected: Pharmacy staff, Managers
 */
router.post('/add-medication', protect, addMedicationToStock);

/**
 * GET /api/pharmacy/requests
 * Get pending medication requests from physicians for pharmacy review
 */
router.get('/requests', protect, getPendingMedicationRequests);

/**
 * PATCH /api/pharmacy/requests/:encounterId/:requestId
 * Approve or decline a medication request
 */
router.patch('/requests/:encounterId/:requestId', protect, updateMedicationRequestStatus);

// ============================================================
// PROTECTED ENDPOINTS (Manager/Admin only)
// ============================================================

/**
 * GET /api/pharmacy
 * Get all pharmaceuticals for the hospital
 */
router.get('/', protect, authorize('MANAGER', 'ADMIN'), getPharmaceuticals);

/**
 * GET /api/pharmacy/low-stock
 * Get low stock items for the hospital
 */
router.get('/low-stock', protect, authorize('MANAGER', 'ADMIN'), getLowStockItems);

/**
 * GET /api/pharmacy/:pharmaceuticalId
 * Get single pharmaceutical details
 */
router.get('/:pharmaceuticalId', protect, authorize('MANAGER', 'ADMIN'), getPharmaceutical);

/**
 * PATCH /api/pharmacy/:pharmaceuticalId
 * Update pharmaceutical details
 */
router.patch('/:pharmaceuticalId', protect, authorize('MANAGER', 'ADMIN'), updatePharmaceutical);

/**
 * PATCH /api/pharmacy/:pharmaceuticalId/stock
 * Update pharmaceutical stock
 */
router.patch('/:pharmaceuticalId/stock', protect, authorize('MANAGER', 'ADMIN'), updateStock);

// ============================================================
// AUDIT & ANALYTICS ENDPOINTS (Manager/Admin only)
// ============================================================

/**
 * GET /api/pharmacy/dispensing-history/:medicationId
 * Get dispensing audit trail and statistics for a specific medication
 * Shows: which staff dispensed, when, to which patient, quantities
 * Protected: Manager/Admin - for audit and compliance
 */
router.get('/dispensing-history/:medicationId', protect, authorize('MANAGER', 'ADMIN'), getMedicationDispensingHistory);

/**
 * GET /api/pharmacy/inventory-stats
 * Get all medications with comprehensive dispensing statistics
 * Shows: dispatch patterns, total dispensed, recently dispensed items, low stock alerts
 * Protected: Manager/Admin - for inventory planning
 */
router.get('/inventory-stats', protect, authorize('MANAGER', 'ADMIN'), getInventoryWithStats);

/**
 * GET /api/pharmacy/dispensing-records
 * Get dispensing records filtered by date range
 * Query params: startDate, endDate (ISO format)
 * Protected: Manager/Admin - for compliance reporting
 */
router.get('/dispensing-records', protect, authorize('MANAGER', 'ADMIN'), getDispensingRecordsByDateRange);

// ============================================================
// CSV OPERATIONS (Manager/Admin only)
// ============================================================

/**
 * POST /api/pharmacy/upload-csv
 * Upload CSV file to bulk refill or add medications to stock
 * CSV columns: medicationName, strength, quantity, batchNumber, expiryDate (optional), manufacturer (optional), form, route, unitPrice, minStockLevel
 * Protected: Manager/Admin - for inventory management
 */
router.post('/upload-csv', protect, authorize('MANAGER', 'ADMIN'), upload.single('file'), uploadCSVRefill);

/**
 * GET /api/pharmacy/export-csv
 * Export full inventory as CSV file for download
 * Protected: Manager/Admin - for inventory reporting
 */
router.get('/export-csv', protect, authorize('MANAGER', 'ADMIN'), exportInventoryCSV);

/**
 * DELETE /api/pharmacy/:pharmaceuticalId
 * Delete pharmaceutical
 */
router.delete('/:pharmaceuticalId', protect, authorize('MANAGER', 'ADMIN'), deletePharmaceutical);

export default router;
