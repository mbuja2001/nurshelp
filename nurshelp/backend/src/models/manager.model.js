import mongoose from 'mongoose';

const managerSchema = new mongoose.Schema(
  {
    // Hospital reference - each manager belongs to a hospital
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },
    managerID: {
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
    email: {
      type: String,
      required: true,
      lowercase: true,
      trim: true,
      index: true,
    },
    password: {
      type: String,
      required: true,
    },
    phone: {
      type: String,
      trim: true,
    },
    department: {
      type: String,
      enum: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'All'],
      default: 'All',
      description: 'Department managed by this manager'
    },
    role: {
      type: String,
      enum: ['MANAGER', 'SHIFT_MANAGER', 'DEPARTMENT_MANAGER', 'ADMIN'],
      required: true,
    },
    permissions: {
      canCreateShifts: { type: Boolean, default: true },
      canEditShifts: { type: Boolean, default: true },
      canDeleteShifts: { type: Boolean, default: true },
      canAssignStaff: { type: Boolean, default: true },
      canApproveLeaves: { type: Boolean, default: false },
      canManageRoles: { type: Boolean, default: false },
      canViewReports: { type: Boolean, default: true },
    },
    isActive: {
      type: Boolean,
      default: true,
      index: true,
    },
    lastLogin: {
      type: Date,
      default: null,
    },
  },
  { 
    timestamps: true,
    collection: 'managers',
  }
);

// Compound index for finding managers in a hospital
managerSchema.index({ hospitalId: 1, isActive: 1 });
// Unique compound index - email must be unique per hospital
managerSchema.index({ hospitalId: 1, email: 1 }, { unique: true });

export const Manager = mongoose.model('Manager', managerSchema);
