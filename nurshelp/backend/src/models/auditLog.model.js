import mongoose from 'mongoose';

const auditLogSchema = new mongoose.Schema(
  {
    // What action happened
    action: {
      type: String,
      enum: [
        'PATIENT_VIEW', // Staff viewed a patient record
        'PATIENT_EDIT', // Staff edited patient data
        'BREAK_GLASS_ACCESS', // Staff accessed patient outside their department
        'DISPOSITION_MADE', // Physician made a disposition decision
        'ESCALATION', // Patient escalated to higher acuity
      ],
      required: true,
      index: true,
    },

    // Who did it
    actorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Practitioner',
      required: true,
      index: true,
    },

    // What resource was accessed
    resourceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'PatientEHRRecord',
      required: true,
      index: true,
    },

    // Hospital context
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },

    // Reason (for sensitive actions like break-glass access)
    reason: String,

    // Authorization details
    authorized: {
      type: Boolean,
      default: true,
    },
    approvedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Practitioner',
    },

    // Audit trail details
    ipAddress: String,
    userAgent: String,
    changeDetails: mongoose.Schema.Types.Mixed, // Before/after for edits

    // Compliance tracking
    circleOfCare: {
      type: String,
      enum: ['SAME_DEPARTMENT', 'BREAK_GLASS', 'CONSULTATION', 'EMERGENCY'],
      default: 'SAME_DEPARTMENT',
    },

    // Metadata
    timestamp: { type: Date, default: Date.now, index: true },
    description: String,
  },
  { timestamps: true }
);

// Indexes for common audit queries
auditLogSchema.index({ hospitalId: 1, timestamp: -1 });
auditLogSchema.index({ actorId: 1, timestamp: -1 });
auditLogSchema.index({ resourceId: 1, timestamp: -1 });
auditLogSchema.index({ action: 1, circleOfCare: 1 });

export const AuditLog = mongoose.model('AuditLog', auditLogSchema);
