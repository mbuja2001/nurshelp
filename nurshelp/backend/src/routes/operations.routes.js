import express from 'express';
import { getOperationsFeed } from '../controllers/operations.controller.js';
import { protect } from '../middleware/authmiddleware.js';

const router = express.Router();

router.get('/feed', protect, getOperationsFeed);

export default router;
