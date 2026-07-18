import mongoose from 'mongoose';

const nurseProfileSchema = new mongoose.Schema(
  {
    nurseId: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    name: {
      type: String,
      required: true,
    },
    surname: {
      type: String,
      required: true,
    },
    licenseNumber: {
      type: String,
      required: true,
      unique: true,
    },
    licenseType: {
      type: String,
      enum: ['RN', 'PHC', 'SPECIALIST'],
      required: true,
      description: 'RN=Registered Nurse, PHC=Primary Health Care, SPECIALIST=Post-Basic Specialist'
    },
    nqfLevel: {
      type: Number,
      enum: [6, 7, 8],
      required: true,
      description: '6=Diploma RN, 7=Degree RN/Palliative, 8=Post-Basic Postgraduate'
    },
    specializations: {
      type: [String],
      enum: [
        /* Critical Care */
        'ICU Nurse', 'Emergency Nurse', 'Trauma Nurse',
        /* Obstetrics & Neonatal */
        'Midwife', 'Advanced Midwife', 'Neonatal Nurse', 'NICU Nurse',
        /* Medical & Surgical */
        'Theatre Nurse', 'Perioperative Nurse', 'Orthopaedic Nurse', 'Dialysis Nurse', 'Nephrology Nurse', 'Ophthalmic Nurse',
        /* Child & Mental Health */
        'Paediatric Nurse', 'Psychiatric Nurse', 'Mental Health Nurse',
        /* Oncology & Palliative */
        'Oncology Nurse', 'Palliative Care Nurse',
        /* Primary Health Care */
        'PHC Nurse', 'Community Nurse',
        /* General */
        'General Nurse'
      ],
      default: ['General Nurse']
    },
    primarySpecialty: {
      type: String,
      enum: [
        'Critical Care', 'Obstetrics', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 
        'Oncology', 'Palliative Care', 'Primary Health Care', 'General'
      ],
      default: 'General'
    },
    credentials: [
      {
        name: String, // e.g., "ICU Ventilator Certified", "ACLS", "PALS", "IV Therapy"
        verifiedDate: Date,
        expiryDate: Date,
        isActive: {
          type: Boolean,
          default: true,
        },
      },
    ],
    maxSatsLevel: {
      type: String,
      enum: ['green', 'yellow', 'red'],
      default: 'green',
      description: 'Maximum SATS risk level nurse can independently triage'
    },
    canTriageRed: {
      type: Boolean,
      default: false,
      description: 'SATS Red (Critical) - requires certification'
    },
    canTriageYellow: {
      type: Boolean,
      default: true,
      description: 'SATS Yellow (Medium Risk)'
    },
    canTriageGreen: {
      type: Boolean,
      default: true,
      description: 'SATS Green (Low Risk)'
    },
    maxBedsPerShift: {
      type: Number,
      default: 4, // How many patients they can manage
    },
    experienceYears: {
      type: Number,
      default: 0,
    },
    performanceScore: {
      type: Number,
      min: 0,
      max: 100,
      default: 75, // Based on feedback from physicians
    },
    certifications: [String], // Array of cert names for quick lookup
    isActive: {
      type: Boolean,
      default: true,
    },
    email: {
      type: String,
      required: true,
    },
    phone: String,
  },
  {
    timestamps: true,
  }
);

// Indexes for common queries
nurseProfileSchema.index({ nurseId: 1, specialty: 1 });
nurseProfileSchema.index({ licenseType: 1, maxESILevel: 1 });
nurseProfileSchema.index({ specialty: 1, isActive: 1 });
nurseProfileSchema.index({ email: 1 });

const NurseProfile = mongoose.model('NurseProfile', nurseProfileSchema);

export default NurseProfile;
