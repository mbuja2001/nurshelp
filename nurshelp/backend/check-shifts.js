#!/usr/bin/env node
import mongoose from 'mongoose';
import { config } from 'dotenv';
import { Shift } from './src/models/shift.model.js';

config();
await mongoose.connect(process.env.MONGO_URI);

const ST_MERCY_ID = '69e60994224d3ac94f1399e8';
const today = new Date();
today.setHours(0, 0, 0, 0);
const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

console.log(`=== SHIFTS IN ST MERCY HOSPITAL ===\n`);
console.log(`Searching for shifts on ${today.toDateString()}\n`);

// Get all shifts in St Mercy
const shifts = await Shift.find({ hospitalId: ST_MERCY_ID }).sort({ shiftDate: -1 }).limit(20);

console.log(`Total shifts in St Mercy: ${shifts.length}\n`);

// Show all shifts
console.log('Recent shifts:');
shifts.forEach((s, i) => {
  console.log(`  [${i+1}] ${s.shiftName} (ID: ${s.shiftID})`);
  console.log(`      Date: ${s.shiftDate.toDateString()}`);
  console.log(`      Time: ${s.shiftStart} - ${s.shiftEnd}`);
  console.log(`      Assigned: ${s.assignedStaff.length} / ${s.maxStaffRequired}`);
  if (s.assignedStaff.length > 0) {
    s.assignedStaff.forEach(st => {
      console.log(`        - ${st.name || 'Unknown'} (${st.staffID}) [${st.status}]`);
    });
  }
  console.log();
});

// Check today specifically
const todayShifts = shifts.filter(s => {
  const sDate = new Date(s.shiftDate);
  sDate.setHours(0, 0, 0, 0);
  return sDate.getTime() === today.getTime();
});

console.log(`\nShifts TODAY (${today.toDateString()}): ${todayShifts.length}`);
if (todayShifts.length > 0) {
  todayShifts.forEach(s => {
    console.log(`  ✓ ${s.shiftName} @ ${s.shiftStart}-${s.shiftEnd}`);
    s.assignedStaff.forEach(st => {
      console.log(`    └─ ${st.name} [${st.status}]`);
    });
  });
}

await mongoose.connection.close();
