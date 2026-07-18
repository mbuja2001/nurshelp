#!/usr/bin/env node
import mongoose from 'mongoose';
import { Manager } from './src/models/manager.model.js';
import { Hospital } from './src/models/hospital.model.js';
import dotenv from 'dotenv';

dotenv.config();

async function main() {
  try {
    console.log('🔗 Connecting to MongoDB...');
    await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/nursedb');
    console.log('✅ Connected\n');

    const managers = await Manager.find({}).lean();
    console.log(`📋 Total managers: ${managers.length}\n`);

    for (const mgr of managers) {
      console.log(`Manager: ${mgr.name} ${mgr.surname}`);
      console.log(`  Email: ${mgr.email}`);
      console.log(`  Role: ${mgr.role}`);
      console.log(`  Hospital ID: ${mgr.hospitalId}`);
      console.log(`  Active: ${mgr.isActive}`);
      console.log('');
    }

    if (managers.length === 0) {
      console.log('❌ No managers found in database');
    }

    process.exit(0);
  } catch (err) {
    console.error('❌ ERROR:', err.message);
    process.exit(1);
  }
}

main();
