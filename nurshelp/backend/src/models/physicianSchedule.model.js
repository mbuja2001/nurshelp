import mongoose from "mongoose";

const physicianScheduleSchema = new mongoose.Schema({
  // Physician info
  physicianId: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  surname: { type: String, required: true },
  email: { type: String, lowercase: true, trim: true },
  specialty: { type: String, required: true }, // e.g., "Cardiology", "ER", "Orthopedics"
  
  // Daily schedule
  date: { type: Date, required: true }, // The day this schedule is for
  status: { 
    type: String, 
    enum: ["ON_DUTY", "ON_CALL", "OFF", "VACATION"], 
    default: "OFF" 
  },
  
  // Shift timing
  shiftStart: { type: String }, // e.g., "08:00" (24-hour format)
  shiftEnd: { type: String },   // e.g., "16:00"
  breakStart: { type: String }, // e.g., "12:00" (break/lunch start)
  breakEnd: { type: String },   // e.g., "13:00" (break/lunch end)
  
  // Workload/availability
  maxPatients: { type: Number, default: 20 }, // Max patients this physician can handle
  currentPatientCount: { type: Number, default: 0 }, // Patients currently assigned
  availabilityMask: { type: [Number], default: [] }, // 24-hour availability (1=available, 0=not available)
  workloadScore: { type: Number, default: 0 }, // ESI workload calculation
  
  // Performance metrics
  averageTriageTime: { type: Number }, // minutes to triage a patient
  consultationRate: { type: Number }, // % of cases needing consultation
  averageConfirmationTime: { type: Number }, // minutes from assignment to confirmation
  
  // Notes
  notes: { type: String, default: "" }, // e.g., "Limited to trauma cases", "Teaching day"
  
  // Metadata
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
}, { timestamps: true });

// Index for efficient queries
physicianScheduleSchema.index({ date: 1, status: 1, specialty: 1 });
physicianScheduleSchema.index({ physicianId: 1, date: 1 });
physicianScheduleSchema.index({ date: 1 });

export const PhysicianSchedule = mongoose.model("PhysicianSchedule", physicianScheduleSchema);
