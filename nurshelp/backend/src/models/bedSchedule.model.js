import mongoose from "mongoose";

const bedScheduleSchema = new mongoose.Schema({
  // Ward reference
  wardId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Ward",
    required: true,
    index: true
  },
  
  // Bed info
  bedId: { 
    type: String, 
    required: true, 
    unique: true,
    index: true
    // e.g., "ICU-W1-B01", "ED-W2-B03" (Ward-id-Bed-number)
  },
  
  roomNumber: { type: String, required: true },
  department: { type: String, required: true }, // e.g., "ER", "ICU", "General"
  
  // Bed type/capability
  bedType: { 
    type: String, 
    enum: ["STANDARD", "MONITORED", "ISOLATION", "INTENSIVE", "HIGH_DEPENDENCY"], 
    default: "STANDARD" 
  },
  
  // Real-time status (MAIN tracking field)
  status: { 
    type: String, 
    enum: ["AVAILABLE", "OCCUPIED", "CLEANING", "MAINTENANCE"], 
    default: "AVAILABLE",
    index: true
  },
  
  // Status timing
  statusChangedAt: {
    type: Date,
    default: Date.now
  },
  
  // Current occupation
  currentEncounterId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: "Encounter", 
    default: null 
  },
  
  assignedNurseId: { 
    type: String, 
    default: null 
  },
  
  occupancyStartTime: { 
    type: Date, 
    default: null 
  },
  
  expectedDischargeTime: {
    type: Date,
    default: null
  },
  
  // Acuity level
  acuityLevel: { 
    type: String, 
    enum: ["CRITICAL", "HIGH", "MODERATE", "LOW"], 
    default: "MODERATE" 
  },
  
  // Bed capacity/features
  hasMonitor: { type: Boolean, default: false },
  hasVentilator: { type: Boolean, default: false },
  hasDialysis: { type: Boolean, default: false },
  canIsolate: { type: Boolean, default: false },
  
  // Cleaning/maintenance tracking
  lastCleanedAt: {
    type: Date,
    default: null
  },
  
  cleaningDurationMinutes: {
    type: Number,
    default: 30 // Standard cleaning time
  },
  
  estimatedReadyTime: {
    type: Date,
    default: null // When bed will be ready for next patient
  },
  
  cleanedByStaffId: {
    type: String,
    default: null
  },
  
  maintenanceNotes: {
    type: String,
    default: ""
  },
  
  // Performance tracking
  averageStayHours: { type: Number }, // Historical average stay duration
  turnaroundTimeMinutes: { type: Number, default: 30 }, // Time to clean between patients
  totalOccupancyCount: { type: Number, default: 0 }, // Total times bed has been used
  
  // Metadata
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
}, { timestamps: true, toJSON: { virtuals: true }, toObject: { virtuals: true } });

bedScheduleSchema.virtual('fhirStatus').get(function() {
  if (this.status === 'OCCUPIED') return 'occupied';
  if (this.status === 'AVAILABLE') return 'available';
  if (this.status === 'RESERVED') return 'reserved';
  return 'unusable';
});

// Index for efficient queries
bedScheduleSchema.index({ status: 1, department: 1 });
bedScheduleSchema.index({ status: 1, wardId: 1 });
bedScheduleSchema.index({ department: 1 });
bedScheduleSchema.index({ currentEncounterId: 1 });
bedScheduleSchema.index({ status: 1 });
bedScheduleSchema.index({ wardId: 1, status: 1 });
bedScheduleSchema.index({ statusChangedAt: -1 });

// Pre-save hook for calculations
bedScheduleSchema.pre("save", async function() {
  // Calculate estimated ready time if bed is in CLEANING status
  if (this.status === "CLEANING" && this.statusChangedAt) {
    const estimatedTime = new Date(this.statusChangedAt.getTime() + this.cleaningDurationMinutes * 60000);
    this.estimatedReadyTime = estimatedTime;
  }
});

export const BedSchedule = mongoose.model("BedSchedule", bedScheduleSchema);
