#!/usr/bin/env node
import mongoose from 'mongoose';
import { config } from 'dotenv';
import { Shift } from './src/models/shift.model.js';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';

config();
await mongoose.connect(process.env.MONGO_URI);

const ST_MERCY_ID = '69e60994224d3ac94f1399e8';
const CHEN_ID = '69eb614c00b7728234069a4e';

console.log('=== DIAGNOSING SHIFT ASSIGNMENT SCHEMA ===\n');

// 1. Find Chen
const chen = await ClinicalStaff.findById(CHEN_ID);
console.log('Dr. Chen:');
console.log(`  ID: ${chen._id}`);
console.log(`  Name: ${chen.firstName} ${chen.lastName}`);
console.log(`  Role: ${chen.systemRole}`);
console.log(`  Hospital: ${chen.hospitalId}\n`);

// 2. Find today's shifts in St Mercy
const today = new Date();
today.setHours(0, 0, 0, 0);
const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

const shifts = await Shift.find({
  hospitalId: ST_MERCY_ID,
  date: { $gte: today, $lt: tomorrow }
});

console.log(`Shifts in St Mercy Hospital today: ${shifts.length}`);
shifts.forEach((shift, i) => {
  console.log(`\n  [${i+1}] ${shift.shiftName}`);
  console.log(`      Department: ${shift.department}`);
  console.log(`      Time: ${shift.shiftStart} - ${shift.shiftEnd}`);
  console.log(`      Assigned Staff: ${shift.assignedStaff.length}`);
  
  shift.assignedStaff.forEach((staff, j) => {
    console.log(`        [${j+1}] ${staff.name} (${staff.type})`);
    console.log(`            ID: ${staff.staffID}`);
    console.log(`            Status: ${staff.status}`);
  });
  
  // Check if Chen is assigned
  const chenAssigned = shift.assignedStaff.find(s => s.staffID?.toString() === CHEN_ID);
  if (chenAssigned) {
    console.log(`      ✓ Dr. Chen IS assigned to this shift as: ${chenAssigned.type}`);
  }
  
  // Check for teamAssignment field
  if (shift.teamAssignment) {
    console.log(`      Has teamAssignment field`);
  } else {
    console.log(`      ✗ NO teamAssignment field (uses old schema)`);
  }
});

// 3. Query like the frontend does
console.log(`\n\nQuerying shifts like frontend (using assignedStaff.staffID):`);
const queryResult = await Shift.findOne({
  hospitalId: ST_MERCY_ID,
  'assignedStaff.staffID': CHEN_ID,
  date: { $gte: today, $lt: tomorrow }
});

if (queryResult) {
  console.log(`✓ Found shift: ${queryResult.shiftName}`);
  const chenAssignment = queryResult.assignedStaff.find(s => s.staffID?.toString() === CHEN_ID);
  console.log(`  Chen's assigned type: ${chenAssignment.type}`);
  console.log(`  Shift has teamAssignment? ${queryResult.teamAssignment ? 'YES' : 'NO'}`);
} else {
  console.log(`✗ No shift found for Chen`);
}

console.log('\n=== DIAGNOSIS ===');
console.log('Issue: Frontend expects teamAssignment field with physicianId/clinicalNurseId');
console.log('But backend Shift model uses assignedStaff array with type="PHYSICIAN" or "NURSE"');
console.log('\nFix: Backend should either:');
console.log('1. Create/populate teamAssignment field from assignedStaff types');
console.log('2. Or update frontend to parse assignedStaff array instead of teamAssignment');
console.log('\nRecommendation: Option 1 - convert in getTodayAssignedShift controller');

await mongoose.connection.close();
