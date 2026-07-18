import mongoose from 'mongoose';

const acuityAlertSchema = new mongoose.Schema(
  {
    alertId: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    encounterId: {
      type: String,
      required: true,
      index: true,
    },
    nurseId: {
      type: String,
      index: true,
    },
    departmentManager: String,

    alertType: {
      type: String,
      enum: ['VITAL_DEVIATION', 'NEW_ORDER', 'SURVEILLANCE_FLAG', 'ACUITY_ESCALATION'],
      required: true,
    },

    severity: {
      type: String,
      enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
      required: true,
      index: true,
    },

    // Alert Details
    details: {
      parameter: String, // 'Heart Rate', 'BP Systolic', 'Respiratory Rate', etc
      currentValue: Number,
      normalRange: {
        min: Number,
        max: Number,
      },
      threshold: String, // 'exceeded max' or 'below min'

      // For ORDER alerts
      orderDescription: String,

      // For SURVEILLANCE alerts
      riskType: String, // 'sepsis_risk', 'fall_risk', etc
    },

    // Response
    createdAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
    acknowledgedAt: Date,
    acknowledgedBy: String,
    actionTaken: String,
    resolvedAt: Date,

    // Escalation
    escalatedToPhysician: {
      type: Boolean,
      default: false,
    },
    escalationTime: Date,
    physicianResponse: String,

    status: {
      type: String,
      enum: ['NEW', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'ESCALATED'],
      default: 'NEW',
      index: true,
    },

    // For tracking which vital in sequence of vitals
    sequenceNumber: Number,
  },
  {
    timestamps: false, // We use explicit createdAt/updatedAt handling
  }
);

// Indexes for common queries
acuityAlertSchema.index({ encounterId: 1, status: 1 });
acuityAlertSchema.index({ nurseId: 1, status: 1 });
acuityAlertSchema.index({ severity: 1, createdAt: -1 });
acuityAlertSchema.index({ status: 1, severity: 1 });

const AcuityAlert = mongoose.model('AcuityAlert', acuityAlertSchema);

export default AcuityAlert;
