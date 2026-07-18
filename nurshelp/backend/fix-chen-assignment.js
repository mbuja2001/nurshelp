#!/usr/bin/env node
/**
 * FIX: Update Dr. Chen's assignment from NURSE to PHYSICIAN
 */

import mongoose from 'mongoose';
import { config } from 'dotenv';
import { Shift } from './src/models/shift.model.js';

config();
await mongoose.connect(process.env.MONGO_URI);

const ST_MERCY_ID = '69e60994224d3ac94f1399e8';
const CHEN_ID = '69eb614c00b7728234069a4e';

console.log('=== FIXING DR. CHEN ASSIGNMENT ===\n');

// Find the shift with Chen assigned as NURSE
const today = new Date();
today.setHours(0, 0, 0, 0);
const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

const shift = await Shift.findOne({
  hospitalId: ST_MERCY_ID,
  'assignedStaff.staffID': CHEN_ID,
  date: { $gte: today, $lt: tomorrow }
});

if (!shift) {
  console.log('No shift found for Dr. Chen');
  process.exit(1);
}

console.log(`Found shift: ${shift.shiftName}`);
const chenAssignment = shift.assignedStaff.find(s => s.staffID?.toString() === CHEN_ID);

console.log(`Current assignment type: ${chenAssignment.type}`);

if (chenAssignment.type === 'NURSE') {
  // Update to PHYSICIAN
  chenAssignment.type = 'PHYSICIAN';
  
  // Also update teamAssignment if it exists
  if (shift.teamAssignment) {
    if (shift.teamAssignment.clinicalNurseId?.toString() === CHEN_ID) {
      console.log('Found Chen in clinicalNurseId, moving to physicianId...');
      shift.teamAssignment.physicianId = shift.teamAssignment.clinicalNurseId;
      shift.teamAssignment.clinicalNurseId = null;
    }
  }
  
  await shift.save();
  console.log(`\n✅ FIXED: Dr. Chen's role updated to PHYSICIAN in assignment`);
  console.log(`   Shift: ${shift.shiftName}`);
} else {
  console.log(`\n✓ Dr. Chen is already assigned as: ${chenAssignment.type}`);
}

// Verify
const updated = await Shift.findById(shift._id);
const updatedAssignment = updated.assignedStaff.find(s => s.staffID?.toString() === CHEN_ID);
console.log(`\nVerification: ${updatedAssignment.type}`);

if (updatedAssignment.type === 'PHYSICIAN') {
  console.log('✅ SUCCESS: Frontend will now render PHYSICIAN dashboard');
}

await mongoose.connection.close();
