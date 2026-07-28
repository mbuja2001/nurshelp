#!/usr/bin/env node
import mongoose from 'mongoose';
import { config } from 'dotenv';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';

config();
await mongoose.connect(process.env.MONGO_URI);

console.log('=== CHECKING ALL SEEDED STAFF FOR DUPLICATES ===\n');

const staffEmails = [
  'dr.martinez@stmercy.hospital',
  'nurse.mike@stmercy.hospital'
];

for (const email of staffEmails) {
  const staff = await ClinicalStaff.find({ email });
  console.log(`${email}: ${staff.length} record(s)`);
  if (staff.length > 1) {
    staff.forEach((s, i) => {
      console.log(`  [${i}] ${s._id} → ${s.hospitalId}`);
    });
  }
}

await mongoose.connection.close();
