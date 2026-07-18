import mongoose from 'mongoose';

const shiftSchema = new mongoose.Schema(
  {
    // Hospital reference - each shift belongs to a hospital
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },
    shiftID: {
      type: String,
      required: true,
      index: true,
    },
    shiftName: {
      type: String,
      required: true,
    },
    department: {
      type: String,
      enum: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'Nursing'],
      required: true,
      index: true,
    },
    shiftStart: {
      type: String, // HH:MM format
      required: true,
    },
    shiftEnd: {
      type: String, // HH:MM format
      required: true,
    },
    duration: {
      type: Number, // in hours
      required: true,
    },
    date: {
      type: Date,
      index: true,
    },
    dayOfWeek: {
      type: String,
      enum: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    },
    recurrence: {
      type: String,
      enum: ['ONCE', 'DAILY', 'WEEKLY', 'BIWEEKLY', 'MONTHLY'],
      default: 'ONCE',
    },
    recurrenceEndDate: {
      type: Date,
      // Only required if recurrence is not ONCE
    },
    maxStaffRequired: {
      type: Number,
      required: true,
      default: 1,
    },
    assignedStaff: [
      {
        staffID: {
          type: mongoose.Schema.Types.ObjectId,
          ref: 'Nurse',
        },
        type: {
          type: String,
          enum: ['MANAGER', 'NURSE', 'PHYSICIAN', 'SPECIALIST', 'SUPPORT'],
        },
        name: String,
        surname: String,
        specialty: String,
        status: {
          type: String,
          enum: ['ASSIGNED', 'CONFIRMED', 'ABSENT', 'PENDING'],
          default: 'ASSIGNED',
        },
      },
    ],
    status: {
      type: String,
      enum: ['DRAFT', 'PUBLISHED', 'CANCELLED', 'COMPLETED'],
      default: 'DRAFT',
      index: true,
    },
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Manager',
      required: true,
    },
    notes: String,
    // Full Day Simulation: Team Assignment (determines which dashboard each staff member sees)
    teamAssignment: {
      triageNurseId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'ClinicalStaff',
        default: null,
      },
      clinicalNurseId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'ClinicalStaff',
        default: null,
      },
      clinicalNurseContext: {
        type: String,
        enum: ['CLINICAL_NURSE_ER', 'CLINICAL_NURSE_ICU', 'CLINICAL_NURSE_SURGICAL'],
        default: 'CLINICAL_NURSE_ER',
      },
      physicianId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'ClinicalStaff',
        default: null,
      },
      physicianContext: {
        type: String,
        enum: ['PHYSICIAN_ER', 'PHYSICIAN_ICU', 'PHYSICIAN_SURGICAL'],
        default: 'PHYSICIAN_ER',
      },
      physicianSpecialty: {
        type: String,
        default: 'GENERAL',
      },
      teamAssignedAt: {
        type: Date,
        default: null,
      },
      assignedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'ClinicalStaff',
        default: null,
      },
    },
    shiftContext: {
      type: String,
      enum: ['TRIAGE', 'ER', 'ICU', 'SURGICAL', 'WARD'],
      default: 'WARD',
    },
  },
  { 
    timestamps: true,
    collection: 'shifts',
  }
);

// Composite index for efficient hospital + department queries
shiftSchema.index({ hospitalId: 1, department: 1, date: 1 });
shiftSchema.index({ hospitalId: 1, date: 1 });
shiftSchema.index({ hospitalId: 1, status: 1 });

export const Shift = mongoose.model('Shift', shiftSchema);
