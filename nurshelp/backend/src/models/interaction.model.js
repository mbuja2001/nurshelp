import mongoose, { Schema } from "mongoose";

const interactionSchema = new Schema(
    {
        encounter_id: { type: Schema.Types.ObjectId, ref: 'Encounter', required: true },
        nurse_note: String,
        ai_triage_summary: String, 
        transcript: String,      
    },
    { timestamps: true }
);

export const Interaction = mongoose.model("Interaction", interactionSchema);