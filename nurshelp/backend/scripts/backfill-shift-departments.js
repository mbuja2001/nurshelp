#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Hospital } from '../src/models/hospital.model.js';
import { Shift } from '../src/models/shift.model.js';
import { resolveConfiguredDepartment } from '../src/utils/hospitalDepartment.js';

dotenv.config();

const uri = process.env.MONGO_URI || process.env.MONGODB_URI;
if (!uri) throw new Error('Missing MONGO_URI or MONGODB_URI');

await mongoose.connect(uri);
try {
  const shifts = await Shift.find({
    $or: [
      { departmentId: null },
      { departmentId: { $exists: false } },
      { departmentName: null },
      { departmentName: { $exists: false } },
      { departmentName: '' },
    ],
  }).select('_id hospitalId department departmentId departmentName').lean();
  const hospitals = new Map();
  let matched = 0;
  const updates = [];

  for (const shift of shifts) {
    const hospitalId = String(shift.hospitalId);
    if (!hospitals.has(hospitalId)) {
      const hospital = await Hospital.findById(hospitalId).select('departments').lean();
      hospitals.set(hospitalId, hospital?.departments || []);
    }

    const configuredDepartment = resolveConfiguredDepartment(hospitals.get(hospitalId), shift.department);
    if (!configuredDepartment) continue;
    matched += 1;
    updates.push({
      updateOne: {
        filter: { _id: shift._id },
        update: {
          $set: {
            departmentId: configuredDepartment.departmentId,
            departmentName: shift.departmentName || configuredDepartment.departmentName,
          },
        },
      },
    });
  }

  console.log(`Found ${shifts.length} shifts missing department snapshot data; ${matched} match active configured departments.`);
  if (process.argv.includes('--apply') && updates.length) {
    const result = await Shift.bulkWrite(updates);
    console.log(`Backfilled department snapshots for ${result.modifiedCount} shifts.`);
  } else {
    console.log('Dry run only. Re-run with --apply to persist matching department snapshots.');
  }
  if (shifts.length !== matched) {
    console.warn(`${shifts.length - matched} legacy shifts did not match an active configured department; their department values were left unchanged.`);
  }
} finally {
  await mongoose.disconnect();
}
