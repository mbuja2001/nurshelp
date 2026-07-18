import mongoose from "mongoose";

const transcriptSchema = new mongoose.Schema({
  id: { type: Number },
  type: { type: String },
  text: { type: String }
}, { _id: false });

const aiSuggestedReportSchema = new mongoose.Schema({
  encounter: { type: mongoose.Schema.Types.ObjectId, ref: "Encounter", required: true },
  source: { type: String, default: 'ai' },
  aiMeta: { // metadata about AI suggestion (provider, model, confidence)
    provider: String,
    model: String,
    confidence: Number
  },
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
  transcript: { type: [transcriptSchema], default: [] },
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
  triage: { type: Object, default: {} },
  recommendations: { type: [String], default: [] },
  nurseNotes: { type: String, default: "" },
  status: {
    type: String,
    enum: ["pending", "suggested", "superseded", "archived"],
    default: "suggested"
  },
  severity: { type: Number, default: 1 },
  isWaiting: { type: Boolean, default: true },
  suggestedAt: { type: Date, default: Date.now }
}, { timestamps: true });

export const AiSuggestedReport = mongoose.model("AiSuggestedReport", aiSuggestedReportSchema);

export default AiSuggestedReport;
