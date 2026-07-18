import mongoose from "mongoose";

const staffClockRecordSchema = new mongoose.Schema({
  // Staff reference (ClinicalStaff)
  staffId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "ClinicalStaff",
    required: true,
    index: true,
  },

  // Hospital context for the clock record
  hospitalId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Hospital",
    required: true,
    index: true,
  },

  // Shift reference
  shiftId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Shift",
    required: true,
  },

  // The date of the shift
  date: {
    type: Date,
    required: true,
    index: true,
  },

  // Clock-in/out times
  clockInTime: {
    type: Date,
    required: true,
  },

  clockOutTime: {
    type: Date,
    default: null, // Null while shift is active
  },

  // Break tracking
  breakStartTime: {
    type: Date,
    default: null,
  },

  breakEndTime: {
    type: Date,
    default: null,
  },

  // Status of the clock record
  status: {
    type: String,
    enum: ["CLOCKED_IN", "ON_BREAK", "CLOCKED_OUT", "ABSENT"],
    default: "CLOCKED_IN",
  },

  // Location/Department
  location: {
    type: String,
    enum: ["ED", "ICU", "OR", "MATERNITY", "NICU", "PEDIATRICS", "PSYCH", "GENERAL"],
    default: "ED",
  },

  // Calculated metrics
  totalShiftDuration: {
    type: Number, // milliseconds
    default: null,
  },

  totalBreakTime: {
    type: Number, // milliseconds
    default: null,
  },

  totalWorkTime: {
    type: Number, // total shift time minus breaks
    default: null,
  },

  // Device & security metadata
  metadata: {
    deviceType: String, // 'mobile', 'tablet', 'desktop'
    userAgent: String,
    ipAddress: String,
    location: {
      latitude: Number,
      longitude: Number,
    },
  },

  // Notes
  notes: String, // e.g., "Extended due to emergency", "Early clock-out approved"

  // ============================================
  // BREAK REQUEST & APPROVAL TRACKING
  // ============================================
  breakApprovalStatus: {
    type: String,
    enum: ["NOT_REQUESTED", "PENDING", "APPROVED", "REJECTED"],
    default: "NOT_REQUESTED",
  },

  breakApprovedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Manager", // Manager who approved
    default: null,
  },

  breakApprovalTime: {
    type: Date,
    default: null,
  },

  breakReason: {
    type: String, // Why the nurse needs a break
    default: null,
  },

  breakRejectionReason: {
    type: String, // Why the break was rejected
    default: null,
  },

  // ============================================
  // LATE ARRIVAL & APPROVAL TRACKING
  // ============================================
  lateArrivalMinutes: {
    type: Number, // calculated: clock-in time vs scheduled shift start
    default: 0,
  },

  lateArrivalApprovalStatus: {
    type: String,
    enum: ["NOT_REQUIRED", "PENDING", "APPROVED", "REJECTED"],
    default: "NOT_REQUIRED", // Set to PENDING if >15 min late
  },

  lateArrivalApprovedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Manager", // Manager who approved
    default: null,
  },

  lateArrivalApprovalTime: {
    type: Date,
    default: null,
  },

  lateArrivalReason: {
    type: String, // Why the nurse was late
    default: null,
  },

  lateArrivalRejectionReason: {
    type: String, // Why late arrival was rejected
    default: null,
  },

  // ============================================
  // COMPLIANCE FLAGS
  // ============================================
  flags: {
    type: [String],
    enum: ["EARLY_CLOCKIN", "LATE_CLOCKIN", "EARLY_CLOCKOUT", "NO_SHIFT", "UNAPPROVED_BREAK", "AUTO_CLOCKOUT", "DIDNT_CLOCKOUT", "ABSENT"],
    default: [],
  },

  // Timestamps
  createdAt: {
    type: Date,
    default: Date.now,
  },

  updatedAt: {
    type: Date,
    default: Date.now,
  },
}, { timestamps: true });

// Indexes for efficient queries
// Indexes for efficient queries
staffClockRecordSchema.index({ staffId: 1, date: 1 });
staffClockRecordSchema.index({ staffId: 1, status: 1 });
staffClockRecordSchema.index({ date: 1, status: 1 });
staffClockRecordSchema.index({ shiftId: 1 });
staffClockRecordSchema.index({ breakApprovalStatus: 1 });
staffClockRecordSchema.index({ lateArrivalApprovalStatus: 1 });
staffClockRecordSchema.index({ staffId: 1, breakApprovalStatus: 1 });
staffClockRecordSchema.index({ staffId: 1, lateArrivalApprovalStatus: 1 });

// Calculate metrics before saving
staffClockRecordSchema.pre("save", async function () {
  try {
    // Calculate late arrival minutes if shift is referenced
    if (this.shiftId && this.clockInTime && !this.lateArrivalMinutes) {
      // Use existing NurseSchedule model for schedules until schedule models are unified
      const shift = await mongoose.model("NurseSchedule").findById(this.shiftId);
      if (shift && shift.shiftStart) {
        // Parse shiftStart (e.g., "07:00") into a Date on the same day as shift date
        const [hours, minutes] = shift.shiftStart.split(":").map(Number);
        const scheduledStartDate = new Date(this.date);
        scheduledStartDate.setHours(hours, minutes, 0, 0);

        // Calculate difference in minutes
        const diffMs = this.clockInTime - scheduledStartDate;
        const diffMinutes = Math.floor(diffMs / 1000 / 60);

        if (diffMinutes > 0) {
          this.lateArrivalMinutes = diffMinutes;
          // If >15 min late, require approval
          if (diffMinutes > 15) {
            this.lateArrivalApprovalStatus = "PENDING";
          }
        }
      }
    }

    if (this.clockOutTime && !this.totalShiftDuration) {
      this.totalShiftDuration = this.clockOutTime - this.clockInTime;
    }

    if (this.breakStartTime && this.breakEndTime && !this.totalBreakTime) {
      this.totalBreakTime = this.breakEndTime - this.breakStartTime;
    }

    if (this.totalShiftDuration && this.totalBreakTime) {
      this.totalWorkTime = this.totalShiftDuration - this.totalBreakTime;
    }
  } catch (err) {
    throw err;
  }
});

// Primary model name is now StaffClockRecord; keep NurseClockRecord as alias for compatibility
const model = mongoose.model("StaffClockRecord", staffClockRecordSchema);
export const StaffClockRecord = model;
export const NurseClockRecord = model;
