#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';

dotenv.config();
const [encounterId, mobility, avpu] = process.argv.slice(2);
if (!encounterId || !mobility || !avpu) throw new Error('Usage: node scripts/backfill-encounter-tews-inputs.js <encounter-id> <mobility> <avpu>');
await mongoose.connect(process.env.MONGO_URI || process.env.MONGODB_URI);
try {
  const encounter = await Encounter.findById(encounterId);
  if (!encounter) throw new Error(`Encounter not found: ${encounterId}`);
  encounter.vitals = {
    ...(encounter.vitals?.toObject?.() || encounter.vitals || {}),
    mobility,
    avpu
  };
  encounter.markModified('vitals');
  await encounter.save();
  console.log(JSON.stringify({ encounterId, mobility, avpu, saved: true }, null, 2));
} finally {
  await mongoose.disconnect();
}
