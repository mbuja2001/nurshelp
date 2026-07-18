import mongoose from 'mongoose';

const equipmentUsageSchema = new mongoose.Schema({
  equipmentName: { type: String, required: true, index: true, unique: true },
  count: { type: Number, default: 0 },
  lastUsedAt: { type: Date, default: null }
}, { timestamps: true });

export const EquipmentUsage = mongoose.model('EquipmentUsage', equipmentUsageSchema);

export default EquipmentUsage;
