import mongoose from 'mongoose';

/**
 * PatientJourney Model
 * 
 * Tracks the patient's state throughout their hospital visit.
 * State machine: ARRIVAL → TRIAGE → OBSERVATION → DISPOSITION → (ADMITTED/DISCHARGED/TRANSFERRED)
 * 
 * Used by:
 * - PhysicianDashboard (determine which lane patient is in)
 * - Analytics (track patient flow)
 * - Ward dashboards (show current patients)
 */

const patientJourneySchema = new mongoose.Schema(
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
      unique: true,
      index: true,
    },
    shiftId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Shift',
      index: true,
    },

    // ============ STATE MACHINE ============
    currentState: {
      type: String,
      enum: [
        'ARRIVED',           // Patient arrived, waiting for triage
        'TRIAGED',           // Triage completed, waiting for clinical nurse
        'OBSERVATION',       // In observation room, waiting for physician decision
        'ESCALATED',         // Clinical nurse escalated to physician (urgent)
        'ADMITTED',          // Physician admitted - now in ward
        'DISCHARGED',        // Physician discharged
        'TRANSFERRED',       // Transferred to another facility
      ],
      required: true,
      default: 'ARRIVED',
      index: true,
    },

    // ============ LOCATION TRACKING ============
    currentLocation: {
      locationType: {
        type: String,
        enum: ['TRIAGE_DESK', 'OBSERVATION_ROOM', 'WARD_BED', 'TRANSFERRED', 'DISCHARGED'],
      },
      departmentContext: String, // 'ER', 'ICU', 'SURGICAL', etc
      observationRoomNumber: String,
      bedNumber: String, // If admitted to ward
      wardName: String,
    },

    // ============ TIMELINE ============
    arrivedAt: {
      type: Date,
      required: true,
      default: () => new Date(),
    },
    triageStartedAt: Date,
    triageCompletedAt: Date,
    observationStartedAt: Date,
    escalatedAt: Date, // When clinical nurse escalated to physician
    physicianReviewStartedAt: Date,
    dispositionDecidedAt: Date,
    admittedAt: Date,
    dischargedAt: Date,
    transferredAt: Date,

    // ============ ESCALATION TRACKING ============
    escalations: [
      {
        escalatedAt: Date,
        escalatedBy: {
          type: mongoose.Schema.Types.ObjectId,
          ref: 'ClinicalStaff',
        },
        reason: String,
        priority: {
          type: String,
          enum: ['RED', 'YELLOW', 'GREEN', 'BLUE'],
        },
      },
    ],

    // ============ REFERENCES TO KEY RECORDS ============
    triageAssessmentId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'TriageAssessment',
    },
    dispositionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Disposition',
    },

    // ============ METRICS ============
    timeInTriageMinutes: Number,
    timeInObservationMinutes: Number,
    responseTimeMinutes: Number, // Time from triage complete to physician decision
    totalTimeMinutes: Number,

    // ============ STATUS ============
    isActive: {
      type: Boolean,
      default: true,
      index: true,
    },
  },
  {
    timestamps: true,
  }
);

// Indexes
patientJourneySchema.index({ hospitalId: 1, currentState: 1, isActive: 1 });
patientJourneySchema.index({ hospitalId: 1, arrivedAt: -1 });
patientJourneySchema.index({ currentState: 1, escalatedAt: -1 });

export const PatientJourney = mongoose.model('PatientJourney', patientJourneySchema);
