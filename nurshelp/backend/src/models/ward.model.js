import mongoose from "mongoose";

const wardSchema = new mongoose.Schema({
  // Ward identification
  wardId: { 
    type: String, 
    required: true, 
    unique: true,
    index: true
    // e.g., "ICU-Ward-A", "ED-Ward-1", "Gen-Ward-B"
  },
  
  wardName: {
    type: String,
    required: true
    // e.g., "Intensive Care Unit - Ward A"
  },
  
  // Department assignment
  department: {
    type: String,
    enum: ["ED", "ICU", "OR", "MATERNITY", "NICU", "PEDIATRICS", "PSYCH", "GENERAL", "CARDIO", "NEURO", "ONCOLOGY"],
    required: true,
    index: true
  },
  
  // Hospital reference
  hospitalId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Hospital",
    required: true,
    index: true
  },
  
  // Beds in this ward
  beds: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: "BedSchedule"
  }],
  
  // Ward location
  location: {
    floor: { type: Number, required: true }, // 1, 2, 3, etc.
    wing: { type: String, default: "Main" }, // East, West, Main, etc.
    area: { type: String, default: "" }
  },
  
  // Bed capacity
  totalBeds: {
    type: Number,
    required: true,
    min: 1
  },
  
  // Current status
  occupiedBeds: {
    type: Number,
    default: 0
  },
  
  availableBeds: {
    type: Number,
    default: 0
  },
  
  cleaningBeds: {
    type: Number,
    default: 0
  },
  
  maintenanceBeds: {
    type: Number,
    default: 0
  },
  
  // Ward capabilities
  specialization: {
    type: [String],
    default: [],
    // e.g., ["Trauma", "Burns", "Cardiac"]
  },
  
  averageOccupancyRate: {
    type: Number,
    default: 0, // percentage
    min: 0,
    max: 100
  },
  
  // Ward metadata
  isActive: {
    type: Boolean,
    default: true
  },
  
  managingNurseId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Nurse",
    default: null
  },
  
  notes: {
    type: String,
    default: ""
  },
  
  // Timestamps
  createdAt: {
    type: Date,
    default: Date.now
  },
  
  updatedAt: {
    type: Date,
    default: Date.now
  }
}, { timestamps: true });

// Indexes
wardSchema.index({ department: 1, isActive: 1 });
wardSchema.index({ wardId: 1 });

// Pre-save hook to calculate available beds
wardSchema.pre("save", async function() {
  this.availableBeds = Math.max(0, this.totalBeds - this.occupiedBeds - this.cleaningBeds - this.maintenanceBeds);
});

export const Ward = mongoose.model("Ward", wardSchema);
