#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { ClinicalStaff } from '../src/models/clinicalstaff.model.js';

dotenv.config();

const uri = process.env.MONGO_URI || process.env.MONGODB_URI;
if (!uri) throw new Error('Missing MONGO_URI or MONGODB_URI');

const missingPosition = {
  $or: [
    { positionCode: null },
    { positionCode: { $exists: false } },
    { positionCode: '' },
  ],
};

await mongoose.connect(uri);
try {
  const count = await ClinicalStaff.countDocuments(missingPosition);
  console.log(`Found ${count} staff records without a positionCode.`);
  if (process.argv.includes('--apply')) {
    const result = await ClinicalStaff.updateMany(missingPosition, {
      $set: { positionCodeNeedsBackfill: true },
    });
    console.log(`Flagged ${result.modifiedCount} staff records for position backfill.`);
  } else {
    console.log('Dry run only. Re-run with --apply to persist backfill flags.');
  }
} finally {
  await mongoose.disconnect();
}
