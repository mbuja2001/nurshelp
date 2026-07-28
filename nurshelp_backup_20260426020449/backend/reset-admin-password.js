#!/usr/bin/env node
/**
 * Reset admin password for St Mercy Hospital
 */

import mongoose from 'mongoose';
import bcrypt from 'bcrypt';
import { Manager } from './src/models/manager.model.js';
import dotenv from 'dotenv';

dotenv.config();

async function main() {
  try {
    console.log('🔗 Connecting to MongoDB...');
    await mongoose.connect(process.env.MONGO_URI);
    console.log('✅ Connected\n');

    const adminEmail = 'alice.johnson@stmercy.hospital';
    const newPassword = 'password123';

    console.log(`🔐 Resetting password for ${adminEmail}`);
    console.log(`   New password: ${newPassword}\n`);

    const hashedPassword = await bcrypt.hash(newPassword, 12);

    const result = await Manager.findOneAndUpdate(
      { email: adminEmail },
      { password: hashedPassword },
      { new: true }
    );

    if (!result) {
      console.log(`❌ Manager ${adminEmail} not found!`);
      process.exit(1);
    }

    console.log(`✅ Password updated successfully!`);
    console.log(`\n📋 Manager Details:`);
    console.log(`   Name: ${result.name} ${result.surname}`);
    console.log(`   Email: ${result.email}`);
    console.log(`   Role: ${result.role}`);
    console.log(`   Hospital ID: ${result.hospitalId}`);

  } catch (err) {
    console.error('❌ ERROR:', err.message);
    process.exit(1);
  } finally {
    await mongoose.disconnect();
  }
}

main();
