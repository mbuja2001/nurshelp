import mongoose from "mongoose";

const nurseScheduleSchema = new mongoose.Schema({
  // Nurse info
  nurseId: { type: String, required: true, unique: true },
  nurseProfileId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'NurseProfile',
  }, // Link to NurseProfile for qualifications
  name: { type: String, required: true },
  surname: { type: String, required: true },
  email: { type: String, lowercase: true, trim: true },
  licenseNumber: { type: String }, // RN license number
  
  // Daily schedule
  date: { type: Date, required: true }, // The day this schedule is for
  status: { 
    type: String, 
    enum: ["ON_DUTY", "ON_CALL", "OFF", "VACATION", "CALL_OUT"], 
    default: "OFF" 
  },
  
  // Shift timing
  shiftStart: { type: String }, // e.g., "07:00" (24-hour format)
  shiftEnd: { type: String },   // e.g., "19:00"
  breakStart: { type: String }, // e.g., "12:00" (break/lunch start)
  breakEnd: { type: String },   // e.g., "13:00" (break/lunch end)
  
  // Bed/Patient assignment
  maxBeds: { type: Number, default: 5 }, // Max beds this nurse can manage
  assignedBeds: { type: [String], default: [] }, // Bed IDs assigned to this nurse (e.g., ["ER-01", "ER-02"])
  currentBedCount: { type: Number, default: 0 }, // Number of beds currently assigned
  
  // Professional qualifications
  licenseType: {
    type: String,
    enum: ['RN', 'PHC', 'SPECIALIST'],
    default: 'RN'
  },
  nqfLevel: {
    type: Number,
    enum: [6, 7, 8],
    default: 6
  },
  specializations: {
    type: [String],
    default: []
  },
  
  // SATS capability (new system)
  canTriageRed: {
    type: Boolean,
    default: false // SATS Red = Critical, requires certification
  },
  canTriageYellow: {
    type: Boolean,
    default: true // SATS Yellow = Medium Risk
  },
  canTriageGreen: {
    type: Boolean,
    default: true // SATS Green = Low Risk
  },
  
  // Certifications (ACLS, PALS, BLS, etc.)
  certifications: { 
    type: [String], 
    default: [] 
  },
  
  // Performance metrics
  averageAssessmentTime: { type: Number }, // minutes to assess patient
  triageAccuracy: { type: Number }, // % accuracy in SATS triage
  averageChargeTime: { type: Number }, // minutes from charge to report
  
  // Availability
  availabilityMask: { type: [Number], default: [] }, // 24-hour availability
  
  // Notes
  notes: { type: String, default: "" }, // e.g., "Prefers pediatrics", "Specialty: Trauma"
  
  // Metadata
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
}, { timestamps: true });

// Index for efficient queries
nurseScheduleSchema.index({ date: 1, status: 1 });
nurseScheduleSchema.index({ nurseId: 1, date: 1 });
nurseScheduleSchema.index({ date: 1 });

export const NurseSchedule = mongoose.model("NurseSchedule", nurseScheduleSchema);
