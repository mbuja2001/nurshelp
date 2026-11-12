import mongoose from 'mongoose';
import { SUPPORT_TICKET_CATEGORIES, SUPPORT_TICKET_SEVERITIES, SUPPORT_TICKET_STATUSES } from '../utils/supportTicketValidation.js';

const supportTicketSchema = new mongoose.Schema({
  hospitalId: { type: mongoose.Schema.Types.ObjectId, ref: 'Hospital', required: true, index: true },
  reporterId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  reporterName: { type: String, trim: true, required: true, maxlength: 120 },
  reporterRole: { type: String, trim: true, required: true, maxlength: 60 },
  title: { type: String, trim: true, required: true, minlength: 5, maxlength: 120 },
  description: { type: String, trim: true, required: true, minlength: 10, maxlength: 2000 },
  category: { type: String, enum: SUPPORT_TICKET_CATEGORIES, required: true, index: true },
  severity: { type: String, enum: SUPPORT_TICKET_SEVERITIES, required: true, index: true },
  status: { type: String, enum: SUPPORT_TICKET_STATUSES, default: 'NEW', required: true, index: true },
  pagePath: { type: String, trim: true, maxlength: 240, default: '' },
  failedRequest: { type: String, trim: true, maxlength: 120, default: '' },
  errorCode: { type: String, trim: true, maxlength: 80, default: '' },
  errorMessage: { type: String, trim: true, maxlength: 500, default: '' },
  primaryStatus: { type: Number, min: 100, max: 599, default: null },
  fallbackStatus: { type: Number, min: 100, max: 599, default: null },
  occurredAt: { type: Date, default: Date.now, required: true },
  assignedToId: { type: mongoose.Schema.Types.ObjectId, default: null },
  assignedToName: { type: String, trim: true, maxlength: 120, default: '' },
  resolutionNote: { type: String, trim: true, maxlength: 1000, default: '' }
}, { timestamps: true });

supportTicketSchema.index({ hospitalId: 1, status: 1, createdAt: -1 });

export const SupportTicket = mongoose.model('SupportTicket', supportTicketSchema);
