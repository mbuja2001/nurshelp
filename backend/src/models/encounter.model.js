import mongoose from "mongoose";

const transcriptSchema = new mongoose.Schema({
  id: { type: Number },
  type: { type: String },
  text: { type: String }
}, { _id: false }); // avoid nested _id

const encounterSchema = new mongoose.Schema({
  nurse_id: { type: mongoose.Schema.Types.ObjectId, ref: "Nurse", default: null },
  patient: {
    id_number: String,
    name: String,
    surname: String,
    gender: String,
    age: Number,
    height_cm: Number,
    symptoms: String,
    duration: String,
    painLevel: String,
    history: String
  },
  transcript: { type: [transcriptSchema], default: [] }, // important: normalized array
  vitals: {
    temp: Number,
    bp: String,
    bp_left_systolic: Number,
    bp_left_diastolic: Number,
    bp_right_systolic: Number,
    bp_right_diastolic: Number,
    hr: Number,
    o2: Number,
    resp: Number,
    bp_systolic: Number,
    bp_diastolic: Number,
    avpu: { type: String }
  },
  triage: { type: Object, default: {} }, // will store the full structured AI output
  recommendations: { type: [String], default: [] }, // clinical recommendations from AI
  nurseNotes: { type: String, default: "" },
  status: { 
    type: String, 
    enum: ["pending", "confirmed", "completed", "unassigned"], 
    default: "unassigned" 
  },
  severity: { type: Number, default: 1 }, // keep as numeric for UI sorting
  isWaiting: { type: Boolean, default: true },
  submittedAt: Date,
  attendedAt: Date
}, { timestamps: true });

export const Encounter = mongoose.model("Encounter", encounterSchema);