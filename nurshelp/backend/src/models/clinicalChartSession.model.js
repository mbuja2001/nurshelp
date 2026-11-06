import mongoose from 'mongoose';

const clinicalChartSessionSchema = new mongoose.Schema({
  sessionKey: { type: String, required: true },
  hospitalId: { type: mongoose.Schema.Types.ObjectId, ref: 'Hospital', required: true, index: true },
  staffId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },
  encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter', required: true, index: true },
  shiftId: { type: mongoose.Schema.Types.ObjectId, ref: 'Shift', default: null, index: true },
  status: { type: String, enum: ['OPEN', 'TRANSFERRED', 'CLOSED'], required: true, default: 'OPEN', index: true },
  hasUnsavedChanges: { type: Boolean, default: false },
  lastHeartbeatAt: { type: Date, required: true, default: Date.now },
  openedAt: { type: Date, required: true, default: Date.now },
  closedAt: { type: Date, default: null },
  transferNote: { type: String, trim: true, default: null }
}, { timestamps: true });

clinicalChartSessionSchema.index({ hospitalId: 1, staffId: 1, sessionKey: 1 }, { unique: true });
clinicalChartSessionSchema.index({ hospitalId: 1, shiftId: 1, status: 1, staffId: 1 });

export const ClinicalChartSession = mongoose.model('ClinicalChartSession', clinicalChartSessionSchema);
