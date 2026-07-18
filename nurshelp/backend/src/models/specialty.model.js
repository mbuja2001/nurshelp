import mongoose from 'mongoose';

const { Schema } = mongoose;

const specialtySchema = new Schema({
  // 1. UNIQUE IDENTIFIERS
  name: { type: String, required: true, unique: true, trim: true }, // e.g., "Cardiology"
  code: { type: String, required: true, unique: true, trim: true }, // e.g., NHS or SNOMED code

  // 2. HIERARCHY
  category: {
    type: String,
    enum: ['Medical', 'Surgical', 'Pathology', 'Radiology', 'Allied Health'],
    required: true
  },
  parentSpecialty: { type: Schema.Types.ObjectId, ref: 'Specialty', default: null },

  // 3. CLINICAL SCOPE
  description: { type: String },
  standardProcedures: [{ type: String }], // List of procedures typical for this specialty

  // 4. SYSTEM LOGIC
  isActive: { type: Boolean, default: true, index: true },
  requiresCredentialing: { type: Boolean, default: true }
}, { timestamps: true, collection: 'specialties' });

// No additional schema.index() for `name` or `code` — `unique: true` already creates indexes

export const Specialty = mongoose.model('Specialty', specialtySchema);
