#!/usr/bin/env node
import mongoose from 'mongoose';
import { config } from 'dotenv';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';

config();
await mongoose.connect(process.env.MONGO_URI);

console.log('=== FIXING SARAH DUPLICATES ===\n');

const CORRECT_SARAH_ID = '69eb614c00b7728234069a4f';
const ST_MERCY_ID = '69e60994224d3ac94f1399e8';

// Find all Sarahs
const allSarahs = await ClinicalStaff.find({ email: 'nurse.sarah@stmercy.hospital' });
console.log(`Found ${allSarahs.length} Sarah record(s)`);
console.log(`Correct Sarah (St Mercy): ${CORRECT_SARAH_ID}\n`);

// Delete all others
const toDelete = allSarahs.filter(s => s._id.toString() !== CORRECT_SARAH_ID);
console.log(`Deleting ${toDelete.length} duplicate record(s):`);

for (const sarah of toDelete) {
  await ClinicalStaff.deleteOne({ _id: sarah._id });
  console.log(`  ✓ Deleted: ${sarah._id} (Hospital: ${sarah.hospitalId})`);
}

// Verify
const remaining = await ClinicalStaff.find({ email: 'nurse.sarah@stmercy.hospital' });
console.log(`\n✅ RESULT: ${remaining.length} Sarah record(s) remain`);
if (remaining.length === 1) {
  console.log(`  ${remaining[0]._id} (St Mercy Hospital)`);
  console.log('\n✅ FIX COMPLETE: Login will now return correct Sarah');
}

await mongoose.connection.close();
