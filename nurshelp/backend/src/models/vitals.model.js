import mongoose from "mongoose";

const vitalsSchema = new mongoose.Schema({
  encounter_id: { type: mongoose.Schema.Types.ObjectId, ref: "Encounter" },
  temperature: Number,
  blood_pressure: String,
  blood_pressure_systolic: Number,
  blood_pressure_diastolic: Number,
  blood_pressure_left_systolic: Number,
  blood_pressure_left_diastolic: Number,
  blood_pressure_right_systolic: Number,
  blood_pressure_right_diastolic: Number,
  heart_rate: Number,
  oxygen_saturation: Number,
  resp: Number
}, { timestamps: true });

export const Vital = mongoose.model("Vital", vitalsSchema);