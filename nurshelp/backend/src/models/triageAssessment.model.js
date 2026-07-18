import mongoose from 'mongoose';

/**
 * TriageAssessment Model
 * 
 * Records the triage nurse's assessment when a patient arrives.
 * Critical field: needsBed - determines if patient will likely need admission
 * 
 * Used by:
 * - PhysicianDashboard (shows triage recommendation)
 * - Disposition model (tracks accuracy/discrepancy)
 * - Analytics (calculates triage accuracy metrics)
 */

const triageAssessmentSchema = new mongoose.Schema(
  {
    // References
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },
    patientId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Patient',
      required: true,
      index: true,
    },
    triageNurseId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'ClinicalStaff',
      required: true,
    },
    shiftId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Shift',
      index: true,
    },

    // ============ PATIENT PRESENTATION ============
    chiefComplaint: {
      type: String,
      required: true,
    },
    symptoms: [String],
    medicalHistory: String,
    medications: [String],
    allergies: [String],

    // ============ VITAL SIGNS ============
    vitalSigns: {
      temperature: Number,
      heartRate: Number,
      respiratoryRate: Number,
      systolic: Number,
      diastolic: Number,
      oxygenSaturation: Number,
      painScale: Number,
    },

    // ============ TRIAGE PRIORITY ============
    triagePriority: {
      type: String,
      enum: ['RED', 'YELLOW', 'GREEN', 'BLUE'],
      required: true,
      index: true,
    },
    triageReasoning: String,

    // ============ CRITICAL: BED NEED ASSESSMENT ============
    /**
     * KEY FIELD: Does this patient NEED A BED (admission)?
     * - true: Triage nurse thinks patient will likely be admitted
     * - false: Triage nurse thinks patient will likely be discharged
     * 
     * Used to:
     * 1. Show physician a recommendation
     * 2. Track triage accuracy
     * 3. Detect FALSE_POSITIVE / FALSE_NEGATIVE discrepancies
     */
    needsBed: {
      type: Boolean,
      required: true,
      default: false,
    },
    bedNeedReasoning: String,

    // ============ TIMESTAMPS ============
    arrivedAt: {
      type: Date,
      required: true,
      default: () => new Date(),
    },
    triageStartedAt: Date,
    triageCompletedAt: {
      type: Date,
      default: () => new Date(),
    },

    // ============ STATUS ============
    status: {
      type: String,
      enum: ['AWAITING_CLINICAL_NURSE', 'IN_OBSERVATION', 'AWAITING_PHYSICIAN_DECISION', 'DISPOSED'],
      default: 'AWAITING_CLINICAL_NURSE',
      index: true,
    },

    // ============ PHYSICIAN DISPOSITION REFERENCE ============
    dispositionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Disposition',
      default: null,
    },
  },
  {
    timestamps: true,
  }
);

// Indexes for fast querying
triageAssessmentSchema.index({ hospitalId: 1, triageCompletedAt: -1 });
triageAssessmentSchema.index({ triageNurseId: 1, triageCompletedAt: -1 });
triageAssessmentSchema.index({ triagePriority: 1, status: 1 });

export const TriageAssessment = mongoose.model('TriageAssessment', triageAssessmentSchema);
