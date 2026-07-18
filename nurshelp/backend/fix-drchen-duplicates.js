#!/usr/bin/env node
/**
 * FIX: Remove duplicate Dr. Chen records and verify correct one is used
 */

import mongoose from 'mongoose';
import { config } from 'dotenv';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';

config();

await mongoose.connect(process.env.MONGO_URI);

console.log('=== FIXING DR. CHEN DUPLICATES ===\n');

// Find all Dr. Chen records
const allChens = await ClinicalStaff.find({ email: 'dr.chen@stmercy.hospital' });
console.log(`Found ${allChens.length} Dr. Chen records:`);
allChens.forEach((c, i) => {
  console.log(`  [${i}] ${c._id}`);
  console.log(`      Hospital: ${c.hospitalId}`);
  console.log(`      Status: ${c.status}`);
});
console.log();

// The CORRECT one is in St Mercy Hospital
const CORRECT_CHEN_ID = '69eb614c00b7728234069a4e';
const ST_MERCY_ID = '69e60994224d3ac94f1399e8';

const correctChen = await ClinicalStaff.findById(CORRECT_CHEN_ID);
console.log('✓ Correct Dr. Chen (St Mercy Hospital):');
console.log(`  ID: ${correctChen._id}`);
console.log(`  Hospital: ${correctChen.hospitalId}`);
console.log(`  Status: ${correctChen.status}\n`);

// Delete all others
const toDelete = allChens.filter(c => c._id.toString() !== CORRECT_CHEN_ID);
console.log(`Deleting ${toDelete.length} duplicate record(s):`);

for (const chen of toDelete) {
  await ClinicalStaff.deleteOne({ _id: chen._id });
  console.log(`  ✓ Deleted: ${chen._id} (Hospital: ${chen.hospitalId})`);
}

// Verify only one Chen remains
const remaining = await ClinicalStaff.find({ email: 'dr.chen@stmercy.hospital' });
console.log(`\n✓ RESULT: ${remaining.length} Dr. Chen record(s) remain`);
if (remaining.length === 1) {
  console.log(`  ${remaining[0]._id} (St Mercy Hospital)`);
  console.log('\n✅ FIX COMPLETE: Login will now return correct Dr. Chen');
  console.log('   Next: Retry the clock-in workflow');
}

await mongoose.connection.close();
