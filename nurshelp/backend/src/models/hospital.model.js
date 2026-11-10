import mongoose from 'mongoose';

const hospitalSchema = new mongoose.Schema(
  {
    hospitalID: {
      type: String,
      required: true,
      unique: true,
      index: true,
      // Format: HOSP-{timestamp}
    },
    name: {
      type: String,
      required: true,
      trim: true,
      index: true,
    },
    ownerAccountId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'SandboxAccount',
      default: null,
      index: true,
    },
    environment: {
      type: String,
      enum: ['SANDBOX', 'PRODUCTION'],
      default: 'SANDBOX',
      immutable: true,
      index: true,
    },
    clinicalTrialMode: {
      type: Boolean,
      default: false,
      index: true,
    },
    clinicalTrialModeUpdatedAt: {
      type: Date,
      default: null,
    },
    clinicalTrialModeUpdatedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'SandboxAccount',
      default: null,
    },
    clinicalTrialModeUpdatedByName: {
      type: String,
      default: null,
    },
    clinicalTrialModeReason: {
      type: String,
      trim: true,
      default: null,
    },
    clinicalTrialModeHistory: [{
      eventType: {
        type: String,
        enum: ['MODE_CHANGED', 'ADMIN_CREATION_FAILED', 'TRIAL_COUNT_MISMATCH'],
        default: 'MODE_CHANGED',
      },
      from: { type: Boolean, default: null },
      to: { type: Boolean, default: null },
      reason: { type: String, required: true, trim: true },
      by: { type: mongoose.Schema.Types.ObjectId, ref: 'SandboxAccount', default: null },
      byName: { type: String, default: 'System', trim: true },
      at: { type: Date, required: true, default: Date.now },
      details: { type: mongoose.Schema.Types.Mixed, default: null },
    }],
    activeTrialCount: {
      type: Number,
      min: 0,
      default: 0,
    },
    country: {
      type: String,
      default: 'USA',
    },
    city: {
      type: String,
      trim: true,
    },
    state: {
      type: String,
      trim: true,
    },
    zipCode: {
      type: String,
      trim: true,
    },
    phone: {
      type: String,
      trim: true,
    },
    email: {
      type: String,
      trim: true,
      lowercase: true,
    },
    // Primary admin who registered the hospital
    adminManagerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Manager',
      required: false,
    },
    adminManagerName: {
      type: String,
      required: true,
    },
    adminManagerEmail: {
      type: String,
      required: true,
    },
    // Hospital status
    isActive: {
      type: Boolean,
      default: true,
      index: true,
    },
    // Department configuration for this hospital
    departments: [
      {
        name: {
          type: String,
          trim: true,
          minlength: 2,
          maxlength: 60,
          required: true,
        },
        departmentId: { type: String, default: null },
        isActive: {
          type: Boolean,
          default: true,
        },
        serviceTypes: {
          type: [{ type: String, enum: ['OUTPATIENT', 'INPATIENT', 'CLINICAL_SUPPORT'] }],
          default: [],
        },
        specialties: {
          type: [{
            name: { type: String, required: true },
            code: { type: String, default: null },
            codeSystem: { type: String, default: null },
            display: { type: String, default: null },
          }],
          default: [],
        },
      },
    ],
    workloadPolicy: {
      acuityWeightByPriority: {
        RED: { type: Number, min: 0.1, default: 1 },
        ORANGE: { type: Number, min: 0.1, default: 1 },
        YELLOW: { type: Number, min: 0.1, default: 1 },
        GREEN: { type: Number, min: 0.1, default: 1 },
      },
      defaultCapacityByRole: {
        TRIAGE_NURSE: { type: Number, min: 1, default: null },
        CLINICAL_NURSE: { type: Number, min: 1, default: null },
        PHYSICIAN: { type: Number, min: 1, default: null },
      },
      departmentShiftConfig: {
        type: mongoose.Schema.Types.Mixed,
        default: {},
      },
    },
    ehrConfig: {
      canonicalFormat: {
        type: String,
        default: 'FHIR R4',
      },
      integrationAccess: {
        source: {
          type: String,
          enum: ['ACCOUNT_OWNER', 'HOSPITAL_ADMIN_SANDBOX'],
          default: null,
        },
        enabledProtocols: {
          type: [String],
          enum: ['hl7v2', 'fhir-r4'],
          default: [],
        },
        attestedAt: { type: Date, default: null },
        attestedBy: { type: mongoose.Schema.Types.ObjectId, default: null },
        attestedByName: { type: String, default: null },
        reason: { type: String, trim: true, default: null },
      },
      adapters: {
        type: [mongoose.Schema.Types.Mixed],
        default: [
          {
            id: 'hl7v2',
            name: 'HL7v2',
            status: 'Not configured',
            connection: 'Not configured',
            host: null,
            port: '2575',
            lastTest: null,
            protocol: 'MLLP'
          },
          {
            id: 'fhir-r4',
            name: 'FHIR R4',
            status: 'Not configured',
            connection: 'Not configured',
            endpoint: null,
            lastTest: null,
            protocol: 'FHIR REST'
          }
        ]
      },
      deadLetter: {
        type: mongoose.Schema.Types.Mixed,
        default: {
          inboundFailures: 0,
          retryQueue: 0,
          failedMessages: [],
          retryItems: []
        }
      },
      retention: {
        type: [mongoose.Schema.Types.Mixed],
        default: [
          { id: 'primary-records', category: 'Primary clinical record', scope: 'Primary record', window: '4 years', minimum: '6 years', compliant: false },
          { id: 'audit-logs', category: 'Audit logs', scope: 'Primary record', window: '7 years', minimum: '6 years', compliant: true }
        ]
      },
      terminologyRows: {
        type: [mongoose.Schema.Types.Mixed],
        default: [
          { code: 'RED', status: 'Mapped', value: 'Priority = RED', color: '#ef4444' },
          { code: 'ORANGE', status: 'Mapped', value: 'Priority = ORANGE', color: '#f97316' },
          { code: 'YELLOW', status: 'Mapped', value: 'Priority = YELLOW', color: '#eab308' },
          { code: 'GREEN', status: 'Mapped', value: 'Priority = GREEN', color: '#22c55e' }
        ]
      },
      accessMatrix: {
        type: [mongoose.Schema.Types.Mixed],
        default: [
          { role: 'Physician', canView: true, canEdit: true, scope: 'Full encounter access' },
          { role: 'Nurse', canView: true, canEdit: false, scope: 'Ward-scoped records' }
        ]
      },
      protocolPolicy: {
        type: mongoose.Schema.Types.Mixed,
        default: {
          satsVersion: 'SATS v3.1',
          tewsVersion: 'TEWS v2.4',
          reportRequirements: 'System default',
          retentionGovernance: 'POPIA-aligned'
        }
      },
      alertRouting: {
        type: [mongoose.Schema.Types.Mixed],
        default: [
          { alertType: 'QUEUE_CONGESTION', targets: ['Charge nurse', 'ED duty lead'], escalation: 'High' }
        ]
      }
    },
    // Subscription/License info
    subscription: {
      plan: {
        type: String,
        enum: ['FREE', 'BASIC', 'PROFESSIONAL', 'ENTERPRISE'],
        default: 'FREE',
      },
      maxManagers: {
        type: Number,
        default: 3,
      },
      maxStaff: {
        type: Number,
        default: 50,
      },
      maxPatients: {
        type: Number,
        default: 500,
      },
      startDate: {
        type: Date,
        default: Date.now,
      },
      expiryDate: {
        type: Date,
      },
    },
    // Audit
    createdAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
    collection: 'hospitals',
  }
);

// Index for querying by admin
hospitalSchema.index({ adminManagerId: 1 });
hospitalSchema.index({ isActive: 1, createdAt: -1 });

export const Hospital = mongoose.model('Hospital', hospitalSchema);
