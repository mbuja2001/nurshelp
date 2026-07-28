#!/usr/bin/env node
import mongoose from 'mongoose';
import { config } from 'dotenv';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';
import { Hospital } from './src/models/hospital.model.js';

config();

const MONGO_URI = process.env.MONGO_URI;
await mongoose.connect(MONGO_URI);

const ST_MERCY_ID = '69e60994224d3ac94f1399e8';

console.log('=== CHECKING STAFF BY HOSPITAL ===\n');

// Get hospitals
const hospitals = await Hospital.find().select('_id name');
console.log('Hospitals in DB:');
hospitals.forEach(h => {
  console.log(`  ${h._id} → ${h.name}`);
});
console.log();

// Check Dr. Chen in all hospitals
console.log('Dr. Chen locations:');
const chens = await ClinicalStaff.find({ email: 'dr.chen@stmercy.hospital' }).select('email firstName lastName hospitalId');
chens.forEach(c => {
  const hospName = hospitals.find(h => h._id.toString() === c.hospitalId.toString())?.name || 'Unknown';
  console.log(`  ${c._id} → Hospital: ${c.hospitalId} (${hospName})`);
});
console.log();

// Check all practitioners in St Mercy
console.log(`Staff in St Mercy Hospital (${ST_MERCY_ID}):`);
const staffInStMercy = await ClinicalStaff.find({ hospitalId: ST_MERCY_ID }).select('firstName lastName email systemRole hospitalId');
if (staffInStMercy.length === 0) {
  console.log('  (none)');
} else {
  staffInStMercy.forEach(s => {
    console.log(`  ${s._id}`);
    console.log(`    Name: ${s.firstName} ${s.lastName}`);
    console.log(`    Email: ${s.email}`);
    console.log(`    Role: ${s.systemRole}`);
  });
}

await mongoose.connection.close();
