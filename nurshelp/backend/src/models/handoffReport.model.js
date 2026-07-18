import mongoose from 'mongoose';

const handoffReportSchema = new mongoose.Schema(
  {
    handoffId: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    date: {
      type: Date,
      required: true,
      index: true,
    },

    // Shift Info
    fromNurse: {
      nurseId: String,
      name: String,
      shiftEnd: Date,
    },
    toNurse: {
      nurseId: String,
      name: String,
      shiftStart: Date,
    },

    // Handoff Data (one entry per patient assigned to this nurse's shift)
    patientHandoffs: [
      {
        encounterId: String,
        patientName: String,
        bedId: String,

        // SBAR = Situation, Background, Assessment, Recommendation
        situation: String,
        background: String,
        assessment: String,
        recommendation: String,

        // Trend Analysis
        trendDirection: String, // 'improving', 'stable', 'deteriorating'
        esiLevelAtHandoff: Number,
        acuityScoreAtHandoff: Number,
        keyAlerts: [String], // Summary of active alerts

        // Current Orders
        activeOrders: [String],
        highlightedOrders: [String], // Critical new orders

        // Metrics from outgoing nurse
        vitalChecksPerformed: Number,
        interventionsCompleted: [String],
        unfinishedTasks: [String],
      },
    ],

    // Manager Notes
    managerNotes: String,
    bedAvailability: {
      totalBeds: Number,
      occupiedBeds: Number,
      cleaningBeds: Number,
      availableBeds: Number,
    },

    // Acknowledgment
    acknowledgedAt: Date,
    acknowledgedBy: String, // Incoming nurse
    acknowledgedByNurseId: String,

    handoffType: {
      type: String,
      enum: ['SHIFT_CHANGE', 'MID_SHIFT_COVERAGE', 'PATIENT_TRANSFER'],
      default: 'SHIFT_CHANGE',
    },

    status: {
      type: String,
      enum: ['DRAFT', 'PENDING', 'ACKNOWLEDGED', 'COMPLETED'],
      default: 'DRAFT',
      index: true,
    },
  },
  {
    timestamps: true,
  }
);

// Indexes for common queries
handoffReportSchema.index({ date: 1, 'toNurse.nurseId': 1 });
handoffReportSchema.index({ 'fromNurse.nurseId': 1, date: -1 });
handoffReportSchema.index({ status: 1 });

const HandoffReport = mongoose.model('HandoffReport', handoffReportSchema);

export default HandoffReport;
