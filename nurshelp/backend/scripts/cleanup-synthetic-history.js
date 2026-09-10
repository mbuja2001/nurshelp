#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';
import PatientEHRRecord from '../src/models/patientEHRRecord.model.js';
import { Prescription } from '../src/models/pharmacy.model.js';

dotenv.config();

const keepBatch = process.argv[2];
if (!keepBatch) throw new Error('Usage: node scripts/cleanup-synthetic-history.js <batch-to-keep>');

const uri = process.env.MONGO_URI || process.env.MONGODB_URI;
if (!uri) throw new Error('Missing MONGO_URI or MONGODB_URI');

await mongoose.connect(uri);
try {
  const stale = await Encounter.find({ syntheticData: true, seedBatch: { $ne: keepBatch } }).select('_id seedBatch').lean();
  const encounterIds = stale.map((encounter) => encounter._id);
  const [encounters, projections, prescriptions] = await Promise.all([
    Encounter.deleteMany({ _id: { $in: encounterIds } }),
    PatientEHRRecord.deleteMany({ seedBatch: { $ne: keepBatch }, syntheticData: true }),
    Prescription.deleteMany({ encounterId: { $in: encounterIds } })
  ]);

  console.log(JSON.stringify({
    keptBatch: keepBatch,
    deletedEncounterCount: encounters.deletedCount,
    deletedProjectionCount: projections.deletedCount,
    deletedPrescriptionCount: prescriptions.deletedCount
  }, null, 2));
} finally {
  await mongoose.disconnect();
}