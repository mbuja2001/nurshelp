import express from 'express';
import { protect } from '../middleware/authmiddleware.js';
import { getTriageAnalytics } from '../controllers/analytics.controller.js';

const router = express.Router();

router.get('/triage', protect, getTriageAnalytics);

export default router;
