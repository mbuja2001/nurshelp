import express from 'express';
import { bulkCreateHospitalEncounters, bulkCreateHospitalShifts, bulkOnboardHospitalStaff, getHospitalActivity, getHospitalOperationsStatus, getHospitalStructure, getHospitalWards, recordHospitalShiftEvent, saveHospitalWards, updateHospitalStructure } from '../controllers/hospitalOperations.controller.js';
import { protect } from '../middleware/authmiddleware.js';
import { authorize } from '../middleware/rbac.js';

const router = express.Router();
const administratorOnly = authorize('ADMIN', 'MANAGER', 'SUPERUSER');

router.use(protect, administratorOnly);
router.get('/status', getHospitalOperationsStatus);
router.get('/activity', getHospitalActivity);
router.get('/structure', getHospitalStructure);
router.put('/structure', updateHospitalStructure);
router.get('/wards', getHospitalWards);
router.put('/wards', saveHospitalWards);
router.post('/shift-events', recordHospitalShiftEvent);
router.post('/staff/bulk', bulkOnboardHospitalStaff);
router.post('/shifts/bulk', bulkCreateHospitalShifts);
router.post('/encounters/bulk', bulkCreateHospitalEncounters);

console.info('[HOSPITAL OPERATIONS ROUTES] Registered administrator hospital population endpoints');

export default router;
