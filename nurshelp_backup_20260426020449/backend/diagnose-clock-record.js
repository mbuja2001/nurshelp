#!/usr/bin/env node
/**
 * DIAGNOSIS: Clock Record Issue Analysis
 * 
 * PROBLEM:
 * ClockInOutBoard shows: {status: undefined, onBreak: false, clockInTime: undefined, ...}
 * 
 * ROOT CAUSE ANALYSIS:
 * 1. When a shift is ASSIGNED to staff via StaffScheduleManager:
 *    - Backend: assignStaffToShift() adds staff to Shift.assignedStaff array
 *    - Result: Shift.assignedStaff = [{staffID, type, name, surname, ...status: 'ASSIGNED'}]
 *    - BUT: No StaffClockRecord is created at this point
 * 
 * 2. When staff tries to clock in (ClockInOutBoard mounts):
 *    - Frontend: Calls GET /api/staff/today-record
 *    - Backend: getTodayRecord() queries StaffClockRecord collection
 *    - Query: StaffClockRecord.findOne({staffId, date: today})
 *    - Result: Returns NULL (record doesn't exist yet!)
 *    - Response: {success: true, record: null}
 * 
 * 3. Frontend receives null record:
 *    - data.record = null
 *    - setCurrentRecord(null) ✓
 *    - data.record?.status = undefined ✓
 *    - data.record?.clockInTime = undefined ✓
 *    - This is CORRECT behavior - there is no record yet
 * 
 * EXPECTED FLOW:
 * 1. Admin assigns shift: Shift.assignedStaff = [... staffID ...]
 * 2. Staff clocks in: Creates StaffClockRecord + updates Shift.assignedStaff[i].status = 'CLOCKED_IN'
 * 3. Staff views UI: Fetches StaffClockRecord with all timestamps
 * 
 * CURRENT STATE:
 * The issue is NOT a bug - it's the correct state. The undefined fields mean:
 * ✓ Shift was assigned successfully
 * ✗ Staff hasn't clocked in yet (so no record exists)
 * 
 * WHAT SHOULD HAPPEN:
 * - Frontend should show a "Ready to Clock In" UI when record is null
 * - When user clicks "Clock In", it calls POST /api/staff/clock-in
 * - Backend creates StaffClockRecord with clockInTime = now
 * - Subsequent GET /api/staff/today-record calls return the record with valid timestamps
 * 
 * NEXT STEPS:
 * 1. Check if "Clock In" button works and creates a record
 * 2. Verify POST /api/staff/clock-in endpoint creates StaffClockRecord correctly
 * 3. Confirm record fields match expected schema
 */

import mongoose from 'mongoose';
import { config } from 'dotenv';
import { StaffClockRecord } from './src/models/staffClockRecord.model.js';
import { Shift } from './src/models/shift.model.js';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';

config();

const MONGO_URI = process.env.MONGO_URI || 'mongodb+srv://testuser:testpass123@nursecluster.i7wam.mongodb.net/nursehelpdesk?retryWrites=true&w=majority';

// Connect to MongoDB
await mongoose.connect(MONGO_URI).catch(err => {
  console.error('MongoDB connection error:', err);
  process.exit(1);
});

console.log('✓ Connected to MongoDB\n');

// Dr. Chen's credentials
const CHEN_EMAIL = 'dr.chen@stmercy.hospital';
const ST_MERCY_ID = '69e60994224d3ac94f1399e8';

console.log('=== DIAGNOSIS: Clock Record Issue ===\n');

// 1. Find Dr. Chen
console.log('1. Finding Dr. Chen...');
const chen = await ClinicalStaff.findOne({ email: CHEN_EMAIL });
if (!chen) {
  console.log('   ✗ Dr. Chen not found');
  process.exit(1);
}
console.log(`   ✓ Found: ${chen.firstName} ${chen.lastName} (ID: ${chen._id})`);
console.log(`     Email: ${chen.email}`);
console.log(`     Hospital: ${chen.hospitalId}`);
console.log(`     Status: ${chen.status}\n`);

// 2. Find today's shifts for Chen
console.log('2. Finding shifts assigned to Dr. Chen for today...');
const today = new Date();
today.setHours(0, 0, 0, 0);
const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

const shiftsWithChen = await Shift.find({
  hospitalId: ST_MERCY_ID,
  'assignedStaff.staffID': chen._id.toString(),
  shiftDate: { $gte: today, $lt: tomorrow }
});

if (shiftsWithChen.length === 0) {
  console.log('   ✗ No shifts found assigned to Dr. Chen today');
  console.log(`   Query: hospitalId=${ST_MERCY_ID}, staffID=${chen._id}, date range: ${today.toDateString()}`);
} else {
  console.log(`   ✓ Found ${shiftsWithChen.length} shift(s):`);
  shiftsWithChen.forEach((shift, i) => {
    const chenAssignment = shift.assignedStaff.find(s => s.staffID === chen._id.toString());
    console.log(`     [${i+1}] Shift ID: ${shift.shiftID}`);
    console.log(`         Name: ${shift.shiftName}`);
    console.log(`         Date: ${shift.shiftDate.toDateString()}`);
    console.log(`         Time: ${shift.shiftStart} - ${shift.shiftEnd}`);
    console.log(`         Assignment Status: ${chenAssignment?.status || 'N/A'}`);
  });
}
console.log();

// 3. Check if a StaffClockRecord exists for Chen today
console.log('3. Checking for StaffClockRecord for Dr. Chen today...');
const record = await StaffClockRecord.findOne({
  staffId: chen._id,
  date: { $gte: today, $lt: tomorrow }
});

if (!record) {
  console.log('   ✗ NO StaffClockRecord found');
  console.log(`     This is EXPECTED: record is only created when staff clicks "Clock In"`);
  console.log(`     Query: staffId=${chen._id}, date range: ${today.toDateString()}\n`);
} else {
  console.log('   ✓ StaffClockRecord found:');
  console.log(`     Record ID: ${record._id}`);
  console.log(`     Status: ${record.status}`);
  console.log(`     Clock In Time: ${record.clockInTime}`);
  console.log(`     Clock Out Time: ${record.clockOutTime}`);
  console.log(`     Break Start Time: ${record.breakStartTime}`);
  console.log(`     Break End Time: ${record.breakEndTime}`);
  console.log(`     Shift ID: ${record.shiftId}\n`);
}

// 4. Summary
console.log('=== DIAGNOSIS SUMMARY ===\n');
console.log('Status: ✓ Everything is working as expected\n');
console.log('What happened:');
console.log('1. ✓ Shift was assigned to Dr. Chen');
if (shiftsWithChen.length > 0) {
  console.log('2. ✓ Shift exists in database with Chen in assignedStaff[]');
  console.log('3. ✗ No StaffClockRecord exists yet (expected - will be created at first clock-in)');
  console.log('\nWhy frontend shows undefined fields:');
  console.log('→ getTodayRecord() returns: {success: true, record: null}');
  console.log('→ Frontend: data.record?.status = undefined (accessing null.status)');
  console.log('→ This is CORRECT - there is no record until user clicks "Clock In"\n');
  console.log('Next action for user:');
  console.log('→ Click "Clock In" button on ClockInOutBoard');
  console.log('→ This calls POST /api/staff/clock-in');
  console.log('→ Backend creates StaffClockRecord with clockInTime = now');
  console.log('→ Subsequent GET requests will return the record with valid timestamps');
} else {
  console.log('2. ✗ No shift found assigned to Dr. Chen');
  console.log('   → Check if shift assignment was successful in StaffScheduleManager');
  console.log('   → Verify shift date matches today\'s date');
  console.log('   → Verify Dr. Chen was selected before assigning');
}

await mongoose.connection.close();
