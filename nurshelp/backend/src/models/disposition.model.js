import mongoose from 'mongoose';

/**
 * Disposition Model
 * 
 * Records the physician's decision on a patient.
 * Tracks accuracy by comparing to triage recommendation.
 * 
 * Fields:
 * - physicianDecision: ADMITTED, DISCHARGED, TRANSFERRED
 * - triageRecommendation: what triage nurse predicted (true=needs bed, false=no bed)
 * - discrepancy: detected FALSE_POSITIVE or FALSE_NEGATIVE
 */

const dispositionSchema = new mongoose.Schema(
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
    triageAssessmentId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'TriageAssessment',
      required: true,
    },
    physicianId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'ClinicalStaff',
      required: true,
    },
    shiftId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Shift',
      index: true,
    },

    // ============ PHYSICIAN DECISION ============
    decision: {
      type: String,
      enum: ['ADMITTED', 'DISCHARGED', 'TRANSFERRED'],
      required: true,
      index: true,
    },

    // If ADMITTED - which ward/bed
    admissionDetails: {
      wardName: String,
      bedNumber: String,
      bedType: String,
      icuRequired: Boolean,
    },

    // If TRANSFERRED
    transferDestination: String,

    // Clinical notes from physician
    notes: String,

    // ============ TRIAGE ACCURACY TRACKING ============
    /**
     * What did triage say?
     * true = "patient needs bed" (predicted admission)
     * false = "patient doesn't need bed" (predicted discharge)
     */
    triageRecommendation: {
      type: Boolean,
      required: true,
    },

    /**
     * DISCREPANCY DETECTION
     * Compares:
     * - Triage prediction (needsBed)
     * - Physician decision (ADMITTED vs DISCHARGED)
     * 
     * Types:
     * - TRUE_POSITIVE: Triage said "needs bed", Physician said "ADMITTED" ✓
     * - TRUE_NEGATIVE: Triage said "no bed", Physician said "DISCHARGED" ✓
     * - FALSE_POSITIVE: Triage said "needs bed", but Physician said "DISCHARGED" ✗
     * - FALSE_NEGATIVE: Triage said "no bed", but Physician said "ADMITTED" ✗
     */
    discrepancy: {
      hasDiscrepancy: {
        type: Boolean,
        default: false,
      },
      type: {
        type: String,
        enum: ['TRUE_POSITIVE', 'TRUE_NEGATIVE', 'FALSE_POSITIVE', 'FALSE_NEGATIVE'],
      },
      reason: String,
      severity: {
        type: String,
        enum: ['LOW', 'MEDIUM', 'HIGH'],
        default: 'LOW',
      },
    },

    // ============ RESPONSE TIME METRICS ============
    triageCompletedAt: Date,
    physicianReviewStartedAt: Date,
    decisionMadeAt: {
      type: Date,
      default: () => new Date(),
    },
    responseTimeMinutes: Number, // Calculated: decisionMadeAt - triageCompletedAt

    // ============ STATUS ============
    status: {
      type: String,
      enum: ['PENDING', 'COMPLETED', 'CANCELLED'],
      default: 'PENDING',
      index: true,
    },
  },
  {
    timestamps: true,
  }
);

// Indexes
dispositionSchema.index({ hospitalId: 1, decisionMadeAt: -1 });
dispositionSchema.index({ physicianId: 1, decisionMadeAt: -1 });
dispositionSchema.index({ decision: 1, createdAt: -1 });
dispositionSchema.index({ 'discrepancy.hasDiscrepancy': 1 });

export const Disposition = mongoose.model('Disposition', dispositionSchema);
