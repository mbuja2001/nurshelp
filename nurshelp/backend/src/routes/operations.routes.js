import express from 'express';
import {
  getCapacityConfig,
  getClinicianWorkload,
  getEhrConfig,
  getOperationsFeed,
  enableSandboxEhrAccess,
  testEhrAdapter,
  updateAcuityWeights,
  updateCapacity,
  updateEhrConfig
} from '../controllers/operations.controller.js';
import { protect } from '../middleware/authmiddleware.js';
import { authorize } from '../middleware/rbac.js';
import { getHospitalStructure, getHospitalWards, saveHospitalWards, updateHospitalStructure } from '../controllers/hospitalOperations.controller.js';

const router = express.Router();

router.get('/feed', protect, getOperationsFeed);
router.get('/clinician-workload', protect, getClinicianWorkload);
router.get('/capacity-config', protect, getCapacityConfig);
router.get('/ehr-config', protect, getEhrConfig);
router.patch('/capacity', protect, authorize('ADMIN', 'SUPERUSER', 'MANAGER'), updateCapacity);
router.patch('/capacity/acuity-weights', protect, authorize('ADMIN', 'SUPERUSER'), updateAcuityWeights);
router.patch('/ehr-config', protect, authorize('ADMIN', 'SUPERUSER', 'MANAGER'), updateEhrConfig);
router.post('/ehr-config/sandbox-access', protect, authorize('ADMIN', 'SUPERUSER', 'MANAGER'), enableSandboxEhrAccess);
router.post('/ehr-config/test-adapter', protect, authorize('ADMIN', 'SUPERUSER', 'MANAGER'), testEhrAdapter);
router.get('/hospital-structure', protect, authorize('ADMIN', 'MANAGER', 'SUPERUSER'), getHospitalStructure);
router.put('/hospital-structure', protect, authorize('ADMIN', 'MANAGER', 'SUPERUSER'), updateHospitalStructure);
router.get('/hospital-wards', protect, authorize('ADMIN', 'MANAGER', 'SUPERUSER'), getHospitalWards);
router.put('/hospital-wards', protect, authorize('ADMIN', 'MANAGER', 'SUPERUSER'), saveHospitalWards);

console.info('[OPERATIONS ROUTES] Registered /api/operations endpoints:', [
  'GET /api/operations/feed',
  'GET /api/operations/clinician-workload',
  'GET /api/operations/capacity-config',
  'GET /api/operations/ehr-config',
  'PATCH /api/operations/capacity',
  'PATCH /api/operations/capacity/acuity-weights',
  'PATCH /api/operations/ehr-config',
  'POST /api/operations/ehr-config/sandbox-access',
  'POST /api/operations/ehr-config/test-adapter',
  'GET/PUT /api/operations/hospital-structure',
  'GET/PUT /api/operations/hospital-wards'
].join(', '));

export default router;
