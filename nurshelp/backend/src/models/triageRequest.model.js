const mongoose = require('mongoose');

const triageRequestSchema = new mongoose.Schema(
  {
    // Patient information
    patientId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'PatientEHRRecord',
      required: false, // May not exist yet on first triage
    },
    encounterId: {
      type: String,
      required: true,
    },

    // Triage nurse information
    triageNurseId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'NurseProfile',
      required: true,
    },

    // Clinical data - manual entry
    manualTriageData: {
      symptoms: [String], // ['chest pain', 'shortness of breath']
      vitalSigns: {
        temperature: Number, // Celsius
        heartRate: Number, // bpm
        respiratoryRate: Number, // breaths/min
        systolicBP: Number,
        diastolicBP: Number,
        oxygenSaturation: Number, // %
      },
      medicalHistory: [String],
      medications: [String],
      allergies: [String],
      chiefComplaint: String,
      triageNotes: String,
    },

    // AI Processing Mode
    computeMode: {
      type: String,
      enum: ['LLM_ONLY', 'RAG_FULL', 'HYBRID'],
      default: 'HYBRID',
      required: true,
    },

    // Manual Triage Report (always created by nurse)
    manualTriageReport: {
      esiLevel: {
        type: Number,
        min: 1,
        max: 5,
      },
      estimatedAcuity: Number, // 0-100 score
      reasoning: String, // Why nurse chose this ESI level
      recommendedDepartment: String,
      immediateInterventions: [String],
      createdBy: mongoose.Schema.Types.ObjectId, // triageNurseId
      createdAt: Date,
    },

    // AI Triage Report (if computeMode !== 'LLM_ONLY')
    aiTriageReport: {
      esiLevel: Number,
      estimatedAcuity: Number,
      confidence: Number, // 0-1
      reasoning: String,
      recommendedDepartment: String,
      recommendedPhysician: {
        physicianId: mongoose.Schema.Types.ObjectId,
        name: String,
        specialization: String,
        availabilityScore: Number,
        workloadScore: Number,
      },
      recommendedNurse: {
        nurseId: mongoose.Schema.Types.ObjectId,
        name: String,
        licenseType: String,
        currentWorkload: Number,
        matchScore: Number,
      },
      relevantCasesRetrieved: [
        {
          caseId: String,
          similarity: Number,
          esiLevel: Number,
          outcome: String,
        },
      ],
      createdAt: Date,
    },

    // Specialist Feedback Report (filled in later, tracks accuracy)
    specialistFeedbackReport: {
      specialistId: mongoose.Schema.Types.ObjectId,
      actualESILevel: Number,
      actualAcuity: Number,
      finalDiagnosis: String,
      wasMisTriaged: Boolean,
      misTriageType: String, // 'UNDER_TRIAGED' | 'OVER_TRIAGED' | 'WRONG_DEPT'
      feedbackNotes: String,
      createdAt: Date,
    },

    // Tracking
    status: {
      type: String,
      enum: ['AWAITING_TRIAGE', 'TRIAGED', 'ASSIGNED_TO_NURSE', 'ASSIGNED_TO_PHYSICIAN', 'IN_CARE'],
      default: 'TRIAGED',
    },
    isAttended: {
      type: Boolean,
      default: false, // Tracks unattended triaged patients
    },
    assignedNurseId: mongoose.Schema.Types.ObjectId,
    assignedPhysicianId: mongoose.Schema.Types.ObjectId,
    departmentAssigned: String,
    bedAssigned: String,

    // Audit trail
    triageDuration: Number, // milliseconds
    aiProcessingTime: Number, // milliseconds (if used)
    rawESIEngineResponse: mongoose.Schema.Types.Mixed, // Store full response for audit
  },
  {
    timestamps: true,
  }
);

// Indexes for common queries
triageRequestSchema.index({ encounterId: 1 });
triageRequestSchema.index({ triageNurseId: 1, createdAt: -1 });
triageRequestSchema.index({ status: 1, isAttended: 1 }); // Find unattended triages
triageRequestSchema.index({ 'manualTriageReport.esiLevel': 1 });
triageRequestSchema.index({ 'aiTriageReport.esiLevel': 1 });
triageRequestSchema.index({ computeMode: 1, createdAt: -1 }); // Analyze mode usage
triageRequestSchema.index({ 'specialistFeedbackReport.wasMisTriaged': 1 }); // Track accuracy

module.exports = mongoose.model('TriageRequest', triageRequestSchema);
