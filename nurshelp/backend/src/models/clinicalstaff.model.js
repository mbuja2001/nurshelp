import mongoose, { Schema } from "mongoose";
import bcrypt from "bcrypt";

// Enterprise-grade Practitioner schema
const practitionerSchema = new Schema({
  // 1. Hospital context (assignment)
  hospitalId: {
    type: Schema.Types.ObjectId,
    ref: 'Hospital',
    required: true,
    index: true,
  },

  // 2. Identity & Authentication
  firstName: { type: String, required: true, trim: true },
  lastName: { type: String, required: true, trim: true },
  email: {
    type: String,
    required: true,
    lowercase: true,
    trim: true,
  },
  password: { type: String, required: true, select: false },

  // 3. System role (permissions)
  systemRole: {
    type: String,
    enum: ['PHYSICIAN', 'NURSE', 'ADMINISTRATIVE', 'TECHNICIAN', 'SUPERUSER', 'PHARMACIST', 'RADIOLOGIST', 'PATHOLOGIST'],
    required: true,
    index: true,
  },

  // 4. Clinical specialty / domain
  // Prefer a reference to the specialties collection; keep string for compatibility
  clinicalSpecialtyId: {
    type: Schema.Types.ObjectId,
    ref: 'Specialty',
    required: false,
    index: true,
  },
  clinicalSpecialty: {
    type: String,
    required: true,
    default: 'General Practice',
  },

  // 5. Professional credentials
  credentials: {
    title: { type: String, enum: ['MD', 'DO', 'RN', 'NP', 'PA', 'PhD'], required: true },
    licenseNumber: { type: String, required: true },
    npiNumber: { type: String, sparse: true },
  },

  // 6. Deployment / assignment within hospital
  assignment: {
    department: {
      type: String,
      enum: ['EMERGENCY', 'CARDIOLOGY', 'ICU', 'PEDIATRICS', 'RADIOLOGY', 'GENERAL_WARD'],
      required: true,
    },
    isOnCall: { type: Boolean, default: false },
  },

  // 6.1 Circle of Care: Active zone assignments (granular location tracking)
  // Used to determine which patients this staff member can access
  activeAssignments: [{
    zone: { type: String, enum: ['ED_TRIAGE', 'ED_RESUS', 'ED_ZONE_1', 'ED_ZONE_2', 'ICU_BED_1', 'ICU_BED_2', 'WARD_A', 'WARD_B'] },
    assignedAt: { type: Date, default: Date.now },
    shiftId: { type: Schema.Types.ObjectId, ref: 'Shift' },
  }],

  // 6.2 Break-Glass Access Log: Track when staff accessed patients outside their assigned department
  breakGlassAccess: [{
    patientId: { type: Schema.Types.ObjectId, ref: 'PatientEHRRecord' },
    accessedAt: { type: Date, default: Date.now },
    reason: { type: String, required: true }, // e.g., "Consultation", "Emergency escalation"
    approvedBy: { type: Schema.Types.ObjectId, ref: 'Practitioner' },
    auditLog: { type: Boolean, default: true },
  }],

  // 7. Status & metadata
  status: {
    type: String,
    enum: ['ACTIVE', 'ON_LEAVE', 'TERMINATED', 'PENDING_CREDENTIALING'],
    default: 'ACTIVE',
    index: true,
  },
  metadata: {
    createdBy: { type: Schema.Types.ObjectId, ref: 'Manager' },
    lastLogin: { type: Date },
  }
}, {
  timestamps: true,
  collection: 'practitioners',
});

// Password hashing (stronger salt rounds for production)
// REMOVED: Pre-save hook causes issues with dual model exports
// Password should be hashed at controller level before creation/update
// If you need to hash passwords in schema, use:
//   const hashedPwd = await bcrypt.hash(pwd, 12);
//   doc.password = hashedPwd;
//   await doc.save(); // won't re-hash

// Indexes
// Multi-tenant isolation: email must be unique per hospital, not globally
practitionerSchema.index({ hospitalId: 1, email: 1 }, { unique: true });
// Queries by hospital + specialty
practitionerSchema.index({ hospitalId: 1, clinicalSpecialty: 1, clinicalSpecialtyId: 1, status: 1 });
// Queries by hospital + name
practitionerSchema.index({ hospitalId: 1, lastName: 1, firstName: 1 });
// Zone-based access control
practitionerSchema.index({ 'activeAssignments.zone': 1, 'assignment.department': 1 });
// Email lookup (case-insensitive searches - used by auth login)
practitionerSchema.index({ email: 1 });

// Create model once and reuse
const PractitionerModel = mongoose.models.Practitioner || mongoose.model('Practitioner', practitionerSchema);

// CRITICAL: Also register under ClinicalStaff name for populate() lookups in NurseClockRecord
if (!mongoose.models.ClinicalStaff) {
  mongoose.model('ClinicalStaff', practitionerSchema);
}

// Keep backward-compatible exports: `ClinicalStaff` for existing code, and `Practitioner` as new name
export const Practitioner = PractitionerModel;
export const ClinicalStaff = PractitionerModel;
