import express from 'express';
import { openClinicalChartSession, updateClinicalChartSession, listActiveShiftChartSessions, listActiveStaffChartSessions } from '../controllers/clinicalChartSession.controller.js';
import { protect } from '../middleware/authmiddleware.js';
import { authorize } from '../middleware/rbac.js';

const router = express.Router();
router.use(protect);
router.post('/open', openClinicalChartSession);
router.get('/shifts/:shiftId/active', authorize('ADMIN', 'MANAGER', 'SUPERUSER'), listActiveShiftChartSessions);
router.get('/staff/:staffId/active', authorize('ADMIN', 'MANAGER', 'SUPERUSER'), listActiveStaffChartSessions);
router.patch('/:sessionId', updateClinicalChartSession);

export default router;
