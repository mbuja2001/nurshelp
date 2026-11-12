import express from 'express';
import { createSupportTicket, listSupportTickets, updateSupportTicket } from '../controllers/supportTicket.controller.js';
import { protect } from '../middleware/authmiddleware.js';
import { authorize } from '../middleware/rbac.js';

const router = express.Router();

router.use(protect);
router.post('/', createSupportTicket);
router.get('/', authorize('ADMIN', 'MANAGER'), listSupportTickets);
router.patch('/:ticketId', authorize('ADMIN', 'MANAGER'), updateSupportTicket);

export default router;
