#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';

dotenv.config();
const id = process.argv[2];
if (!id) throw new Error('Usage: node scripts/inspect-encounter-tews.js <encounter-id>');
await mongoose.connect(process.env.MONGO_URI || process.env.MONGODB_URI);
try {
  const encounter = await Encounter.findById(id).lean();
  console.log(JSON.stringify({
    id: encounter?._id,
    vitals: encounter?.vitals,
    triage: encounter?.triage,
    ai: encounter?.ai_triage_report,
    manual: encounter?.manual_triage_report,
    sourceMetadata: encounter?.sourceMetadata
  }, null, 2));
} finally {
  await mongoose.disconnect();
}
