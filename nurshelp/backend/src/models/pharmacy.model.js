import mongoose from 'mongoose';

/**
 * Pharmacy Management Models
 * Handles drug inventory, prescriptions, dispensing, and drug interactions
 */

const medicationSchema = new mongoose.Schema(
  {
    name: { type: String, required: true, unique: true, index: true },
    genericName: String,
    therapeuticClass: String,
    dosageForm: { type: String, enum: ['tablet', 'capsule', 'injection', 'liquid', 'patch', 'inhaler', 'other'] },
    strength: String,
    manufacturer: String,
    ndc: String, // National Drug Code
    upc: String,
    rxRequired: { type: Boolean, default: true },
    warning: [String], // Drug interaction warnings
    contraindications: [String],
    sideEffects: [String],
    dosageInstructions: String,
    maxDailyDose: Number,
    interactions: [
      {
        medicationId: mongoose.Schema.Types.ObjectId,
        severity: { type: String, enum: ['mild', 'moderate', 'severe'], index: true },
        description: String,
      },
    ],
  },
  { timestamps: true }
);

const pharmacyInventorySchema = new mongoose.Schema(
  {
    // Hospital context (multi-tenant isolation)
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },

    medicationId: { type: mongoose.Schema.Types.ObjectId, ref: 'Medication', required: true, index: true },
    facilityId: { type: String, required: true, index: true },
    quantity: { type: Number, required: true, min: 0 },
    unitPrice: { type: Number, required: true, min: 0 },
    totalValue: Number,
    minStockLevel: { type: Number, required: true },
    maxStockLevel: Number,
    reorderPoint: Number,
    reorderQuantity: Number,
    batchNumber: [
      {
        batchId: String,
        quantity: Number,
        expiryDate: Date,
        dateReceived: Date,
        manufacturer: String,
      },
    ],
    lastRestockDate: Date,
    expiringMedicines: Number,
    status: { type: String, enum: ['in-stock', 'low-stock', 'out-of-stock', 'recalled'], index: true },
  },
  { timestamps: true, indexes: [{ hospitalId: 1, facilityId: 1, medicationId: 1 }] }
);

const prescriptionItemSchema = new mongoose.Schema(
  {
    medicationId: { type: mongoose.Schema.Types.ObjectId, ref: 'Medication' },
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
    patientId: { type: String, required: true, index: true },
    physicianId: { type: String, required: true, index: true },

    medicationId: { type: mongoose.Schema.Types.ObjectId, ref: 'Medication' },
    medicationName: String,
    dosage: String, // e.g., "500mg twice daily"
    frequency: String,
    route: { type: String, enum: ['oral', 'IM', 'IV', 'subcutaneous', 'inhalation', 'topical', 'other'] },
    quantity: Number,
    duration: String, // e.g., "10 days"
    refills: { type: Number, default: 0 },
    refillsRemaining: { type: Number, default: 0 },

    // Formal prescription report fields
    prescription_number: { type: String, unique: true, sparse: true },
    date_issued: { type: Date, default: Date.now },
    valid_until: Date,
    prescriber: {
      id: mongoose.Schema.Types.ObjectId,
      name: String,
      email: String,
      role: String,
      specialty: String,
      hpcsa_number: String,
      practice_number: String,
      practice_address: String,
      phone: String
    },
    patient_details: {
      name: String,
      id_number: String,
      date_of_birth: Date,
      age: Number,
      address: String,
      medical_aid: String,
      medical_aid_number: String
    },
    items: [prescriptionItemSchema],
    substitution_permitted: { type: Boolean, default: true },
    special_instructions: String,
    contraindications: String,

    status: {
      type: String,
      enum: ['pending', 'approved', 'dispensed', 'completed', 'cancelled', 'suspended', 'ISSUED', 'SUBMITTED', 'IN_PROGRESS'],
      default: 'pending',
      index: true,
    },
    dateIssued: { type: Date, default: Date.now, index: true },
    expiryDate: Date,
    dispensedDate: Date,
    dispensedBy: mongoose.Schema.Types.ObjectId,

    // ePrescription fields
    isElectronic: { type: Boolean, default: false },
    ePrescritionId: { type: String, unique: true, sparse: true }, // Unique ID for eRX
    digitalSignature: {
      type: String,
      required: function() { return this.isElectronic; },
    },
    signatureTimestamp: Date,
    prescriptionVerificationCode: String, // For patient/pharmacy verification

    // Pharmacy routing
    routingPharmacies: [
      {
        pharmacyId: String,
        facilityId: String,
        pharmacyName: String,
        primaryRoute: { type: Boolean, default: false },
        status: { type: String, enum: ['offered', 'accepted', 'rejected', 'fulfilled'], default: 'offered' },
        routedDate: Date,
        acceptedDate: Date,
      },
    ],
    fulfilledByPharmacy: String,

    patientNotifications: [
      {
        notificationType: { type: String, enum: ['sent', 'routed', 'ready', 'dispensed', 'expired'], index: true },
        sentDate: Date,
        channel: { type: String, enum: ['email', 'sms', 'push', 'portal'] },
        status: { type: String, enum: ['pending', 'sent', 'read', 'failed'] },
      },
    ],
    requestedNotificationChannels: [{ type: String, enum: ['email', 'sms', 'push', 'portal'] }],

    allowSubstitution: { type: Boolean, default: true },
    substitutionRequests: [
      {
        requestedMedicationId: mongoose.Schema.Types.ObjectId,
        requestedMedicationName: String,
        reason: String,
        status: { type: String, enum: ['pending', 'approved', 'denied'] },
        reviewedBy: mongoose.Schema.Types.ObjectId,
        reviewDate: Date,
      },
    ],
    priorAuthorizationRequired: { type: Boolean, default: false },
    priorAuthorizationNumber: String,
    priorAuthorizationStatus: { type: String, enum: ['pending', 'approved', 'denied', 'expired'] },
    priorAuthExpiryDate: Date,

    interactionAlerts: [
      {
        medicationId: mongoose.Schema.Types.ObjectId,
        severity: String,
        description: String,
      },
    ],
    allergiesConflict: Boolean,
    dosageVerified: Boolean,
    pharmacistReview: Boolean,
    pharmacistReviewNotes: String,

    refillHistory: [
      {
        refillNumber: Number,
        refillDate: Date,
        dispensedDate: Date,
        refillStatus: { type: String, enum: ['pending', 'approved', 'dispensed', 'denied'] },
        facilityId: String,
        pharmacyId: String,
      },
    ],
    nextRefillEligibleDate: Date,
    autoRefillEnabled: { type: Boolean, default: false },

    instructions: String,
    sideEffectWarnings: [String],
    notes: String,
    contraindications: [String],
    pharmacy_notes: String,
    cancellation_reason: String,
    cancelled_by: mongoose.Schema.Types.ObjectId,
    cancelled_at: Date,
  },
  { timestamps: true, indexes: [{ patientId: 1, status: 1 }, { encounterId: 1 }, { ePrescritionId: 1 }, { 'routingPharmacies.status': 1 }] }
);

const ePrescriptionArchiveSchema = new mongoose.Schema(
  {
    originalPrescriptionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Prescription', required: true },
    archiveReason: { type: String, enum: ['cancelled', 'expired', 'superseded', 'patient-request'] },
    archivedDate: { type: Date, default: Date.now },
    archiveNotes: String,
    archiveUser: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff' },
    prescriptionSnapshot: mongoose.Schema.Types.Mixed, // Full prescription data at archive time
  },
  { timestamps: true, indexes: [{ originalPrescriptionId: 1 }, { archivedDate: -1 }] }
);

const priorAuthorizationSchema = new mongoose.Schema(
  {
    prescriptionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Prescription', required: true },
    medicationId: { type: mongoose.Schema.Types.ObjectId, ref: 'Medication', required: true },
    insuranceId: { type: mongoose.Schema.Types.ObjectId, ref: 'Insurance' },
    insuranceCompany: String,

    requestDate: { type: Date, default: Date.now },
    requestNumber: { type: String, unique: true, required: true },
    authorizationNumber: String,
    status: {
      type: String,
      enum: ['submitted', 'pending', 'approved', 'denied', 'expired', 'partial'],
      default: 'pending',
      index: true,
    },

    requestedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff' },
    reviewedBy: String, // Insurance reviewer
    reviewDate: Date,

    denialReason: String,
    denialCode: String,
    appealable: Boolean,

    expiryDate: Date,
    validityPeriod: Number, // Days

    clinicalJustification: String,
    requiredDocuments: [String],
    supportingDocuments: [String], // URLs to uploaded documents

    authorizationLimitations: {
      maxQuantity: Number, // Max pills/dose
      maxRefills: Number, // Max times can refill
      timeframeMonths: Number, // Valid for X months
    },

    notes: String,
  },
  { timestamps: true, indexes: [{ prescriptionId: 1 }, { requestNumber: 1 }, { status: 1 }, { insuranceId: 1 }] }
);

const pharmacySyncLogSchema = new mongoose.Schema(
  {
    prescriptionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Prescription', required: true },
    sourcePharmmacyId: String,
    targetPharmacyIds: [String],
    syncType: { type: String, enum: ['transfer', 'synchronization', 'consolidation'], required: true },
    syncDate: { type: Date, default: Date.now },
    syncStatus: { type: String, enum: ['initiated', 'in-progress', 'completed', 'failed'], default: 'initiated' },
    acknowledgedBy: mongoose.Schema.Types.ObjectId,
    acknowledgmentDate: Date,
    notes: String,
  },
  { timestamps: true, indexes: [{ prescriptionId: 1 }, { syncDate: -1 }] }
);

const pharmacyDispenseLogSchema = new mongoose.Schema(
  {
    prescriptionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Prescription', required: true, index: true },
    medicationId: { type: mongoose.Schema.Types.ObjectId, ref: 'Medication' },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' },

    quantity: Number,
    batchNumber: String,
    expiryDate: Date,
    dispensedDate: { type: Date, default: Date.now, index: true },
    dispensedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff' },
    facilityId: String,

    cost: Number,
    notes: String,
  },
  { timestamps: true }
);

/**
 * Simple Pharmaceutical Inventory for Hospital Registration
 * Stores initial drug inventory during hospital setup
 */
const pharmaceuticalSchema = new mongoose.Schema(
  {
    // Hospital context (multi-tenant isolation)
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },

    // Drug Information
    name: {
      type: String,
      required: true,
    },
    strength: String, // e.g., "500mg", "10ml", "2%"
    form: {
      type: String,
      enum: ['Tablet', 'Injection', 'Infusion', 'Liquid', 'Inhaler', 'Cream', 'Patch', 'tablet', 'capsule', 'injection', 'liquid', 'patch', 'inhaler', 'cream', 'drop', 'suspension', 'other'],
      required: true,
      set: (v) => v.charAt(0).toUpperCase() + v.slice(1).toLowerCase(),
    },
    route: {
      type: String,
      enum: ['PO', 'IV', 'IM', 'SC', 'PR', 'Inhalation', 'Topical', 'oral', 'inhalation', 'topical', 'transdermal', 'sublingual', 'other'],
      required: true,
    },

    // Inventory
    stock: {
      type: Number,
      required: true,
      min: 0,
    },
    minStockLevel: {
      type: Number,
      default: 10,
    },
    unitPrice: {
      type: Number,
      required: true,
      min: 0,
    },
    totalValue: Number, // Calculated: stock * unitPrice

    // Classification & Safety
    genericName: String,
    classification: String, // e.g., "Antibiotic", "Analgesic"
    requiresPrescription: {
      type: Boolean,
      default: true,
    },
    warnings: [String],
    contraindications: [String],

    // Metadata
    createdAt: {
      type: Date,
      default: Date.now,
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Manager',
    },
  },
  {
    timestamps: true,
    collection: 'pharmaceuticals',
    indexes: [
      { hospitalId: 1, name: 1 },
      { hospitalId: 1, stock: 1 },
      { hospitalId: 1, createdAt: -1 },
    ],
  }
);

// Calculate totalValue before save
pharmaceuticalSchema.pre('save', function(next) {
  this.totalValue = this.stock * this.unitPrice;
  next();
});

export const Medication = mongoose.models.Medication || mongoose.model('Medication', medicationSchema);
export const PharmacyInventory = mongoose.models.PharmacyInventory || mongoose.model('PharmacyInventory', pharmacyInventorySchema);
export const Prescription = mongoose.models.Prescription || mongoose.model('Prescription', prescriptionSchema);
export const PharmacyDispenseLog = mongoose.models.PharmacyDispenseLog || mongoose.model('PharmacyDispenseLog', pharmacyDispenseLogSchema);
export const EPrescriptionArchive = mongoose.models.EPrescriptionArchive || mongoose.model('EPrescriptionArchive', ePrescriptionArchiveSchema);
export const PriorAuthorization = mongoose.models.PriorAuthorization || mongoose.model('PriorAuthorization', priorAuthorizationSchema);
export const PharmacySyncLog = mongoose.models.PharmacySyncLog || mongoose.model('PharmacySyncLog', pharmacySyncLogSchema);
export const Pharmaceutical = mongoose.models.Pharmaceutical || mongoose.model('Pharmaceutical', pharmaceuticalSchema);
