#!/usr/bin/env node
import mongoose from 'mongoose';
import { config } from 'dotenv';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';

config();
await mongoose.connect(process.env.MONGO_URI);

console.log('=== CHECKING SARAH RECORDS ===\n');

const sarahs = await ClinicalStaff.find({ email: 'nurse.sarah@stmercy.hospital' });
console.log(`Found ${sarahs.length} Sarah record(s):\n`);

sarahs.forEach((s, i) => {
  console.log(`[${i}] ID: ${s._id}`);
  console.log(`    Hospital: ${s.hospitalId}`);
  console.log(`    Name: ${s.firstName} ${s.lastName}`);
  console.log(`    Role: ${s.systemRole}`);
  console.log(`    Status: ${s.status}`);
  console.log();
});

// Expected: only the one in St Mercy (69e60994224d3ac94f1399e8)
const CORRECT_SARAH_ID = '69eb614c00b7728234069a4f';
const ST_MERCY_ID = '69e60994224d3ac94f1399e8';

const correctSarah = await ClinicalStaff.findById(CORRECT_SARAH_ID);
if (correctSarah) {
  console.log(`✓ Correct Sarah (St Mercy): ${CORRECT_SARAH_ID}`);
  console.log(`  Hospital: ${correctSarah.hospitalId}`);
} else {
  console.log(`✗ Correct Sarah not found with ID: ${CORRECT_SARAH_ID}`);
}

await mongoose.connection.close();
