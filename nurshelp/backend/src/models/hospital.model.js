import mongoose from 'mongoose';

const hospitalSchema = new mongoose.Schema(
  {
    hospitalID: {
      type: String,
      required: true,
      unique: true,
      index: true,
      // Format: HOSP-{timestamp}
    },
    name: {
      type: String,
      required: true,
      trim: true,
      index: true,
    },
    country: {
      type: String,
      default: 'USA',
    },
    city: {
      type: String,
      trim: true,
    },
    state: {
      type: String,
      trim: true,
    },
    zipCode: {
      type: String,
      trim: true,
    },
    phone: {
      type: String,
      trim: true,
    },
    email: {
      type: String,
      trim: true,
      lowercase: true,
    },
    // Primary admin who registered the hospital
    adminManagerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Manager',
      required: false,
    },
    adminManagerName: {
      type: String,
      required: true,
    },
    adminManagerEmail: {
      type: String,
      required: true,
    },
    // Hospital status
    isActive: {
      type: Boolean,
      default: true,
      index: true,
    },
    // Department configuration for this hospital
    departments: [
      {
        name: {
          type: String,
          enum: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'General'],
          required: true,
        },
        isActive: {
          type: Boolean,
          default: true,
        },
      },
    ],
    // Subscription/License info
    subscription: {
      plan: {
        type: String,
        enum: ['FREE', 'BASIC', 'PROFESSIONAL', 'ENTERPRISE'],
        default: 'FREE',
      },
      maxManagers: {
        type: Number,
        default: 3,
      },
      maxStaff: {
        type: Number,
        default: 50,
      },
      maxPatients: {
        type: Number,
        default: 500,
      },
      startDate: {
        type: Date,
        default: Date.now,
      },
      expiryDate: {
        type: Date,
      },
    },
    // Audit
    createdAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
    collection: 'hospitals',
  }
);

// Index for querying by admin
hospitalSchema.index({ adminManagerId: 1 });
hospitalSchema.index({ isActive: 1, createdAt: -1 });

export const Hospital = mongoose.model('Hospital', hospitalSchema);
