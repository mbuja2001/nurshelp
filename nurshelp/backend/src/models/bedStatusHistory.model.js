import mongoose from "mongoose";

const bedStatusHistorySchema = new mongoose.Schema({
  // Bed reference
  bedId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "BedSchedule",
    required: true,
    index: true
  },
  
  bedIdString: {
    type: String,
    required: true // e.g., "ICU-01"
  },
  
  // Ward reference
  wardId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Ward",
    required: true,
    index: true
  },
  
  // Status tracking
  previousStatus: {
    type: String,
    enum: ["AVAILABLE", "OCCUPIED", "CLEANING", "MAINTENANCE"],
    required: true
  },
  
  currentStatus: {
    type: String,
    enum: ["AVAILABLE", "OCCUPIED", "CLEANING", "MAINTENANCE"],
    required: true,
    index: true
  },
  
  // Status change details
  changedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Nurse",
    required: true // Who changed the status (nurse, housekeeper, etc.)
  },
  
  changedByName: String,
  
  // Time tracking
  timestamp: {
    type: Date,
    default: Date.now,
    index: true
  },
  
  // WHY the status changed
  reason: {
    type: String,
    enum: ["PATIENT_ADMITTED", "PATIENT_DISCHARGED", "CLEANING_STARTED", "CLEANING_COMPLETED", "MAINTENANCE_STARTED", "MAINTENANCE_COMPLETED", "UPGRADE", "DOWNGRADE"],
    default: "CLEANING_STARTED"
  },
  
  // Additional context
  encounterId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Encounter",
    default: null
  },
  
  notes: String,
  
  // Duration of status (calculated when status changes)
  durationMinutes: {
    type: Number,
    default: 0
  }
}, { timestamps: true });

// Index for efficient queries
bedStatusHistorySchema.index({ bedId: 1, timestamp: -1 });
bedStatusHistorySchema.index({ wardId: 1, currentStatus: 1 });
bedStatusHistorySchema.index({ timestamp: -1 });

export const BedStatusHistory = mongoose.model("BedStatusHistory", bedStatusHistorySchema);
