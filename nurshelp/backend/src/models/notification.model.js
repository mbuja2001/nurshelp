import mongoose from 'mongoose';

const notificationSchema = new mongoose.Schema({
  // ============ ENCOUNTER & PATIENT ============
  encounter_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Encounter',
    required: true,
    index: true
  },

  patient_mrn: String,
  patient_name: String,
  chief_complaint: String,

  // ============ CLINICIAN (RECEIVER) ============
  clinician_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Staff',
    required: true,
    index: true
  },
  clinician_name: String,
  clinician_role: {
    type: String,
    enum: ['PHYSICIAN', 'NURSE', 'SPECIALIST', 'CONSULTANT'],
    default: 'PHYSICIAN'
  },
  clinician_specialty: String,

  // ============ TRIAGE NURSE (SENDER) ============
  triage_nurse_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Staff'
  },
  triage_nurse_name: String,
  triage_nurse_email: String,

  // ============ REPORT DETAILS ============
  report_type: {
    type: String,
    enum: ['AI_TRIAGE_REPORT', 'MANUAL_TRIAGE_REPORT'],
    default: 'AI_TRIAGE_REPORT',
    index: true
  },

  priority_code: {
    type: String,
    enum: ['RED', 'ORANGE', 'YELLOW', 'GREEN'],
    index: true
  },

  // ============ STATUS TRACKING ============
  status: {
    type: String,
    enum: ['UNREAD', 'READ', 'ACKNOWLEDGED', 'DISMISSED', 'EXPIRED'],
    default: 'UNREAD',
    index: true
  },

  read_at: Date,
  acknowledged_at: Date,
  acknowledged_notes: String,
  dismissed_at: Date,
  dismiss_reason: String,

  // ============ ENGAGEMENT METRICS ============
  time_to_first_view_minutes: Number,  // How long until clinician viewed
  time_to_acknowledgment_minutes: Number,  // How long until acknowledged
  engagement_level: {
    type: String,
    enum: ['NOT_ENGAGED', 'DELAYED', 'TIMELY', 'IMMEDIATE'],
    default: 'NOT_ENGAGED'
  },

  // ============ MULTI-TENANCY & AUDIT ============
  hospital_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Hospital',
    required: true,
    index: true
  },

  created_at: {
    type: Date,
    default: Date.now,
    index: true
  },

  expires_at: Date,  // Auto-expire old notifications

  // ============ AUDIT TRAIL ============
  _version: { type: Number, default: 1 },
  audit_log: [
    {
      action: String,  // 'READ', 'ACKNOWLEDGED', 'DISMISSED'
      timestamp: { type: Date, default: Date.now },
      notes: String
    }
  ]

}, { timestamps: true });

// Index for efficient queries
notificationSchema.index({
  clinician_id: 1,
  hospital_id: 1,
  status: 1
});

notificationSchema.index({
  hospital_id: 1,
  created_at: -1
});

notificationSchema.index({
  encounter_id: 1
});

export const Notification = mongoose.model('Notification', notificationSchema);
