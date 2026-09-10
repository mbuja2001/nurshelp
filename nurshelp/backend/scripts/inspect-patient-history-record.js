#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';

dotenv.config();
const batch = process.argv[2];
const mrn = process.argv[3];
if (!batch || !mrn) throw new Error('Usage: node scripts/inspect-patient-history-record.js <batch> <mrn>');
await mongoose.connect(process.env.MONGO_URI || process.env.MONGODB_URI);
try {
  const encounter = await Encounter.findOne({ seedBatch: batch, 'patient.id_number': mrn }).lean();
  console.log(JSON.stringify({
    id: encounter?._id,
    date: encounter?.date,
    updatedAt: encounter?.updatedAt,
    keys: Object.keys(encounter || {}),
    ai: encounter?.ai_triage_report,
    manual: encounter?.manual_triage_report,
    labs: encounter?.lab_requests,
    surgery: encounter?.surgery_records,
    discharge: encounter?.discharge_report,
    diagnosis: encounter?.diagnosis,
    diagnosedAt: encounter?.assigned_physician?.diagnosed_at
  }, null, 2));
} finally {
  await mongoose.disconnect();
}
