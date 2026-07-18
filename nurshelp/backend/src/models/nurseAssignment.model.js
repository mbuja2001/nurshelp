import mongoose from 'mongoose';

const nurseAssignmentSchema = new mongoose.Schema(
  {
    assignmentId: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    nurseId: {
      type: String,
      required: true,
      index: true,
    },
    nurseProfile: {
      name: String,
      licenseType: String,
      specialty: String,
      maxESILevel: Number,
    },

    shiftDate: {
      type: Date,
      required: true,
      index: true,
    },
    shiftStartTime: Date,
    shiftEndTime: Date,

    assignedPatients: [
      {
        encounterId: String,
        bedId: String,
        patientName: String,
        esiLevel: Number,
        acuityScore: Number,
        assignmentTime: Date,

        // Tracking
        lastVitalCheckTime: Date,
        tasksCompleted: {
          type: Number,
          default: 0,
        },
        tasksPending: {
          type: Number,
          default: 0,
        },
      },
    ],

    maxPatients: Number,
    currentPatientCount: {
      type: Number,
      default: 0,
    },
    workloadScore: {
      type: Number,
      min: 0,
      max: 100,
      default: 0,
    },

    // Performance during this shift
    shiftMetrics: {
      averageResponseTime: Number, // Minutes to respond to alert
      alertsAcknowledged: {
        type: Number,
        default: 0,
      },
      mistakesLogged: {
        type: Number,
        default: 0,
      },
      patientSatisfactionScore: Number,
    },

    clockInTime: Date,
    clockOutTime: Date,
    status: {
      type: String,
      enum: ['WAITING_TO_START', 'ACTIVE', 'ON_BREAK', 'WRAPPING_UP', 'CLOCKED_OUT'],
      default: 'WAITING_TO_START',
      index: true,
    },

    breakStartTime: Date,
    breakEndTime: Date,
  },
  {
    timestamps: true,
  }
);

// Indexes for common queries
nurseAssignmentSchema.index({ nurseId: 1, shiftDate: 1 });
nurseAssignmentSchema.index({ shiftDate: 1, status: 1 });
nurseAssignmentSchema.index({ 'assignedPatients.encounterId': 1 });

const NurseAssignment = mongoose.model('NurseAssignment', nurseAssignmentSchema);

export default NurseAssignment;
