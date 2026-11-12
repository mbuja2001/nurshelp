import mongoose from 'mongoose';
import { SupportTicket } from '../models/supportTicket.model.js';
import { normalizeSupportTicketInput, normalizeSupportTicketUpdate } from '../utils/supportTicketValidation.js';

const getHospitalId = (req) => req.user?.hospitalId || req.hospitalId;
const getActorId = (req) => req.user?._id || req.user?.id;
const getActorName = (user = {}) => (
  [user.name || user.firstName, user.surname || user.lastName].filter(Boolean).join(' ').trim() ||
  user.email ||
  'Hospital staff'
).slice(0, 120);

export const createSupportTicket = async (req, res) => {
  try {
    const hospitalId = getHospitalId(req);
    const reporterId = getActorId(req);
    if (!mongoose.isValidObjectId(hospitalId) || !mongoose.isValidObjectId(reporterId)) {
      return res.status(401).json({ success: false, message: 'Authenticated staff and hospital context are required' });
    }

    const normalized = normalizeSupportTicketInput(req.body);
    if (normalized.error) return res.status(400).json({ success: false, message: normalized.error });

    const ticket = await SupportTicket.create({
      ...normalized.value,
      hospitalId,
      reporterId,
      reporterName: getActorName(req.user),
      reporterRole: String(req.user?.role || req.userRole || 'STAFF').toUpperCase().slice(0, 60)
    });

    return res.status(201).json({
      success: true,
      message: 'IT support ticket submitted to your hospital administrators',
      ticket: {
        _id: ticket._id,
        status: ticket.status,
        createdAt: ticket.createdAt
      }
    });
  } catch (error) {
    console.error('[IT SUPPORT] Ticket creation failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to submit the IT support ticket' });
  }
};

export const listSupportTickets = async (req, res) => {
  try {
    const hospitalId = getHospitalId(req);
    if (!mongoose.isValidObjectId(hospitalId)) {
      return res.status(401).json({ success: false, message: 'Hospital context is required' });
    }

    const status = req.query.status ? String(req.query.status).toUpperCase() : null;
    if (status && !['OPEN', 'NEW', 'IN_PROGRESS', 'RESOLVED'].includes(status)) {
      return res.status(400).json({ success: false, message: 'Select a valid ticket status filter' });
    }
    const filter = { hospitalId };
    if (status === 'OPEN') {
      filter.status = { $in: ['NEW', 'IN_PROGRESS'] };
    } else if (status) {
      filter.status = status;
    }

    const tickets = await SupportTicket.find(filter)
      .sort({ createdAt: -1 })
      .limit(100)
      .lean();

    return res.json({ success: true, tickets });
  } catch (error) {
    console.error('[IT SUPPORT] Ticket list failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to load IT support tickets' });
  }
};

export const updateSupportTicket = async (req, res) => {
  try {
    const hospitalId = getHospitalId(req);
    const actorId = getActorId(req);
    if (!mongoose.isValidObjectId(hospitalId) || !mongoose.isValidObjectId(req.params.ticketId)) {
      return res.status(400).json({ success: false, message: 'A valid hospital and ticket are required' });
    }

    const normalized = normalizeSupportTicketUpdate(req.body);
    if (normalized.error) return res.status(400).json({ success: false, message: normalized.error });

    const updates = {};
    const { assignToMe, ...fields } = normalized.value;
    Object.assign(updates, fields);
    if (assignToMe !== undefined) {
      if (assignToMe) {
        if (!mongoose.isValidObjectId(actorId)) {
          return res.status(401).json({ success: false, message: 'Authenticated administrator is required' });
        }
        updates.assignedToId = actorId;
        updates.assignedToName = getActorName(req.user);
      } else {
        updates.assignedToId = null;
        updates.assignedToName = '';
      }
    }

    const ticket = await SupportTicket.findOneAndUpdate(
      { _id: req.params.ticketId, hospitalId },
      { $set: updates },
      { new: true, runValidators: true }
    ).lean();

    if (!ticket) return res.status(404).json({ success: false, message: 'IT support ticket not found for this hospital' });
    return res.json({ success: true, ticket });
  } catch (error) {
    console.error('[IT SUPPORT] Ticket update failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to update the IT support ticket' });
  }
};
