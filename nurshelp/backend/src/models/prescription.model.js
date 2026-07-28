import mongoose from 'mongoose';

const prescriptionItemSchema = new mongoose.Schema(
  {
    medication_name: String,
    strength: String, // e.g., "500mg"
    form: String, // Tablets, Capsules, Liquid, etc.
    schedule: String, // Schedule 1-5 or OTC
    quantity: Number,
    unit: { type: String, default: 'Units' }, // Tablets, ml, etc.
    dosage: String, // e.g., "500mg"
    frequency: String, // e.g., "Every 6 hours"
    route: String, // Oral, IV, IM, SC, Topical, Inhalation
    duration: String, // e.g., "7 days"
    directions: String, // Full directions for patient
    refills: { type: Number, default: 0 },
    indication: String, // Clinical reason
    status: { type: String, enum: ['PENDING', 'ACCEPTED', 'DISPENSED', 'NOT_RECEIVED'], default: 'PENDING' },
    accepted_at: Date,
    estimated_received: Date,
    actual_received: Date,
    dispensed_by: mongoose.Schema.Types.ObjectId, // Pharmacist who dispensed
    dispensed_at: Date
  },
  { _id: true, timestamps: true }
);

const prescriptionSchema = new mongoose.Schema(
  {
    hospitalId: { type: mongoose.Schema.Types.ObjectId, ref: 'Hospital', required: true, index: true },
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter', required: true, index: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' },

    // Prescriber details
    prescriber: {
      id: mongoose.Schema.Types.ObjectId,
      name: String,
      email: String,
      role: String,
      specialty: String,
      hpcsa_number: String, // South African HPCSA registration
      practice_number: String,
      practice_address: String,
      phone: String
    },

    // Patient details
    patient_details: {
      name: String,
      id_number: String, // South African ID/DOB
      date_of_birth: Date,
      age: Number,
      address: String,
      medical_aid: String,
      medical_aid_number: String
    },

    // Prescription info
    prescription_number: { type: String, unique: true },
    date_issued: { type: Date, default: Date.now },
    valid_until: Date, // 30 days from issue for Schedule 3-4

    // Prescription items
    items: [prescriptionItemSchema],

    // Substitution & Constraints
    substitution_permitted: { type: Boolean, default: true },
    special_instructions: String,
    contraindications: String,

    // Overall status
    status: { type: String, enum: ['ISSUED', 'SUBMITTED', 'ACCEPTED', 'IN_PROGRESS', 'DISPENSED', 'COMPLETED', 'CANCELLED'], default: 'ISSUED', index: true },
    
    // Submission to pharmacy
    submitted_at: Date,
    submitted_to_pharmacy: Boolean,

    // Dispensing tracking
    all_accepted: Boolean,
    all_dispensed: Boolean,
    estimated_completion: Date,

    // Notes
    pharmacy_notes: String,
    cancellation_reason: String,
    cancelled_by: mongoose.Schema.Types.ObjectId,
    cancelled_at: Date
  },
  { timestamps: true }
);

// Auto-generate prescription number before save
prescriptionSchema.pre('save', async function (next) {
  if (!this.prescription_number) {
    const now = new Date();
    const timestamp = now.getTime().toString().slice(-8);
    this.prescription_number = `RX-${timestamp}-${Math.random().toString(36).substring(2, 6).toUpperCase()}`;
  }
  next();
});

export const Prescription = mongoose.models.Prescription || mongoose.model('Prescription', prescriptionSchema);
