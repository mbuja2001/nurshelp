import mongoose from 'mongoose';

const patientEHRRecordSchema = new mongoose.Schema(
  {
    // Hospital context (multi-tenant isolation)
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },

    encounterId: {
      type: String,
      required: true,
      index: true,
    },
    patientId: {
      type: String,
      required: true,
      index: true,
    },
    name: {
      type: String,
      required: true,
    },
    surname: {
      type: String,
      required: true,
    },
    age: Number,
    gender: String,
    mrn: String, // Medical Record Number

    // Current Location
    currentBedId: {
      type: String,
      index: true,
    },
    currentDepartment: String,
    admissionTime: Date,

    // Clinical Status
    esiLevel: {
      type: Number,
      enum: [1, 2, 3, 4, 5],
      required: true,
      index: true,
    },
    acuityScore: {
      type: Number,
      min: 0,
      max: 100,
      default: 50,
    },

    // Current Vitals (last recorded)
    latestVitals: {
      temperature: Number,
      heartRate: Number,
      bpLeft: {
        systolic: Number,
        diastolic: Number,
      },
      bpRight: {
        systolic: Number,
        diastolic: Number,
      },
      respiratoryRate: Number,
      o2Saturation: Number,
      recordedAt: Date,
    },

    // Vital Thresholds (alert if outside range)
    vitalThresholds: {
      hr_min: {
        type: Number,
        default: 60,
      },
      hr_max: {
        type: Number,
        default: 100,
      },
      bp_systolic_min: {
        type: Number,
        default: 90,
      },
      bp_systolic_max: {
        type: Number,
        default: 160,
      },
      bp_diastolic_min: {
        type: Number,
        default: 60,
      },
      bp_diastolic_max: {
        type: Number,
        default: 100,
      },
      resp_min: {
        type: Number,
        default: 12,
      },
      resp_max: {
        type: Number,
        default: 20,
      },
    },

    // Clinical Orders
    activeOrders: [
      {
        orderId: String,
        orderType: String, // 'medication', 'procedure', 'imaging', 'lab'
        description: String,
        isHighRisk: Boolean,
        orderedAt: Date,
      },
    ],

    // Surveillance (Watchlist flags)
    surveillanceFlags: [
      {
        flag: String, // 'sepsis_risk', 'fall_risk', 'cardiac_arrest_risk', 'delirium_risk'
        severity: String, // 'low', 'medium', 'high'
        reason: String,
        flaggedAt: Date,
      },
    ],

    // Chief Complaint
    chiefComplaint: String,
    symptoms: String,
    painLevel: {
      type: Number,
      min: 0,
      max: 10,
    },

    // Nursing Notes (pulled from EHR)
    nursingNotes: [
      {
        noteId: String,
        createdBy: String,
        content: String, // SOAP format
        createdAt: Date,
      },
    ],

    // Allergies & Contraindications
    allergies: [String],
    contraindications: [String],

    // Assignment Status
    assignedNurseId: {
      type: String,
      index: true,
    },
    assignedPhysicianId: String,
    assignmentStartTime: Date,

    // Trend Analysis
    lastVitalUpdateAt: Date,
    vitalTrendDirection: String, // 'stable', 'improving', 'deteriorating'

    // Consultation
    consultingSpecialties: [String], // e.g., ['Cardiology', 'Infectious Disease']

    status: {
      type: String,
      enum: ['ADMITTED', 'IN_PROGRESS', 'DISCHARGE_PENDING', 'DISCHARGED'],
      default: 'IN_PROGRESS',
    },
  },
  {
    timestamps: true,
  }
);

// Indexes for common queries
patientEHRRecordSchema.index({ hospitalId: 1, encounterId: 1 });
patientEHRRecordSchema.index({ hospitalId: 1, currentBedId: 1 });
patientEHRRecordSchema.index({ hospitalId: 1, esiLevel: 1, currentDepartment: 1 });
patientEHRRecordSchema.index({ hospitalId: 1, assignedNurseId: 1 });
patientEHRRecordSchema.index({ hospitalId: 1, 'surveillanceFlags.severity': 1 });
patientEHRRecordSchema.index({ hospitalId: 1, status: 1 });
patientEHRRecordSchema.index({ encounterId: 1 });
patientEHRRecordSchema.index({ currentBedId: 1 });
patientEHRRecordSchema.index({ esiLevel: 1, currentDepartment: 1 });
patientEHRRecordSchema.index({ assignedNurseId: 1 });
patientEHRRecordSchema.index({ 'surveillanceFlags.severity': 1 });
patientEHRRecordSchema.index({ status: 1 });

const PatientEHRRecord = mongoose.model('PatientEHRRecord', patientEHRRecordSchema);

export default PatientEHRRecord;
