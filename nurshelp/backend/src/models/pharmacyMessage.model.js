import mongoose from 'mongoose';

const pharmacyMessageSchema = new mongoose.Schema(
  {
    hospitalId: { type: mongoose.Schema.Types.ObjectId, ref: 'Hospital', index: true },
    fromUserId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true, index: true },
    fromName: { type: String },
    text: { type: String, required: true },
    status: { type: String, enum: ['PENDING', 'READ', 'RESOLVED'], default: 'PENDING', index: true },
    metadata: { type: Object, default: {} }
  },
  { timestamps: true }
);

export const PharmacyMessage = mongoose.model('PharmacyMessage', pharmacyMessageSchema);
