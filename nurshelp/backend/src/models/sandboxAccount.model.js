import mongoose from 'mongoose';

const sandboxAccountSchema = new mongoose.Schema({
  name: { type: String, required: true, trim: true },
  email: { type: String, required: true, lowercase: true, trim: true, unique: true, index: true },
  password: { type: String, required: true, select: false },
  isActive: { type: Boolean, default: true, index: true },
  lastLogin: { type: Date, default: null }
}, { timestamps: true, collection: 'sandboxaccounts' });

export const SandboxAccount = mongoose.model('SandboxAccount', sandboxAccountSchema);