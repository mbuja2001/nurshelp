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
        'CAPACITY_CHANGED', // Clinician/department/hospital workload capacity changed
        'HOSPITAL_CREATED',
        'HOSPITAL_TRIAL_MODE_CHANGED',
        'CLINICAL_TRIAL_COUNT_MISMATCH',
        'EHR_CONFIG_CHANGED',
        'EHR_INTEGRATION_ACCESS_CHANGED',
        'HOSPITAL_STRUCTURE_UPDATED',
        'HOSPITAL_CREDENTIALS_UPDATED',
        'HOSPITAL_STAFF_ONBOARDED',
        'HOSPITAL_SHIFTS_PROVISIONED',
        'HOSPITAL_ENCOUNTERS_PROVISIONED',
        'STAFF_REGISTERED',
        'STAFF_UPDATED',
        'STAFF_DEACTIVATED',
        'APPROVALS_PROCESSED',
        'SHIFT_CREATED',
        'SHIFT_UPDATED',
        'SHIFT_DELETED',
        'SHIFT_PUBLISHED',
        'SHIFT_CANCELLED',
        'STAFF_ASSIGNED',
        'STAFF_REMOVED',
        'CLINICAL_TEAM_ASSIGNED',
        'CLOCK_IN',
        'CLOCK_OUT',
        'BREAK_START',
        'BREAK_END',
        'BREAK_REQUESTED',
        'LATE_ARRIVAL_APPROVED',
        'BREAK_APPROVED',
        'AUTO_CLOCKOUT',
        'HOSPITAL_SHIFT_STARTED',
        'HOSPITAL_SHIFT_ENDED',
        'HOSPITAL_SHIFT_AUTO_ENDED',
        'CLINICAL_CHART_SESSION_OPENED',
        'CLINICAL_CHART_SESSION_UPDATED',
        'CLINICAL_CHART_SESSION_TRANSFERRED',
        'CLINICAL_CHART_SESSION_CLOSED',
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
    actorIdentityType: {
      type: String,
      enum: ['PRACTITIONER', 'MANAGER', 'ACCOUNT_OWNER'],
      default: 'PRACTITIONER',
    },

    // What resource was accessed
    resourceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'PatientEHRRecord',
      required: false,
      index: true,
    },

    resourceType: { type: String, default: null },

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
