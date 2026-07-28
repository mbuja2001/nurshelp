#!/usr/bin/env node
/**
 * Seed practitioners to St Mercy Hospital specifically
 */

import mongoose from 'mongoose';
import bcrypt from 'bcrypt';
import { Practitioner, ClinicalStaff } from './src/models/clinicalstaff.model.js';
import { Manager } from './src/models/manager.model.js';
import { Hospital } from './src/models/hospital.model.js';
import dotenv from 'dotenv';

dotenv.config();

const testPractitioners = [
  {
    email: 'dr.chen@stmercy.hospital',
    firstName: 'Chen',
    lastName: 'Wong',
    password: 'password123',
    systemRole: 'PHYSICIAN',
    clinicalSpecialty: 'Emergency Medicine',
    credentials: {
      title: 'MD',
      licenseNumber: 'MD-001-CHEN',
      npiNumber: 'NPI-001'
    },
    assignment: {
      department: 'EMERGENCY'
    }
  },
  {
    email: 'nurse.sarah@stmercy.hospital',
    firstName: 'Sarah',
    lastName: 'Johnson',
    password: 'password123',
    systemRole: 'NURSE',
    clinicalSpecialty: 'Emergency Nursing',
    credentials: {
      title: 'RN',
      licenseNumber: 'RN-002-JOHNSON',
      npiNumber: 'NPI-002'
    },
    assignment: {
      department: 'EMERGENCY'
    }
  },
  {
    email: 'dr.martinez@stmercy.hospital',
    firstName: 'Roberto',
    lastName: 'Martinez',
    password: 'password123',
    systemRole: 'PHYSICIAN',
    clinicalSpecialty: 'Internal Medicine',
    credentials: {
      title: 'MD',
      licenseNumber: 'MD-003-MARTINEZ',
      npiNumber: 'NPI-003'
    },
    assignment: {
      department: 'CARDIOLOGY'
    }
  },
  {
    email: 'nurse.mike@stmercy.hospital',
    firstName: 'Michael',
    lastName: 'Brown',
    password: 'password123',
    systemRole: 'NURSE',
    clinicalSpecialty: 'Cardiac Nursing',
    credentials: {
      title: 'RN',
      licenseNumber: 'RN-004-BROWN',
      npiNumber: 'NPI-004'
    },
    assignment: {
      department: 'CARDIOLOGY'
    }
  }
];

async function main() {
  try {
    console.log('🔗 Connecting to MongoDB...');
    await mongoose.connect(process.env.MONGO_URI);
    console.log('✅ Connected\n');

    // Find St Mercy Hospital
    console.log('🏥 Finding St Mercy Hospital...');
    const stMercyHosp = await Hospital.findOne({ name: /st\s+mercy/i });
    
    if (!stMercyHosp) {
      console.log('❌ St Mercy Hospital not found');
      process.exit(1);
    }

    console.log(`✅ Found: ${stMercyHosp.name}`);
    console.log(`   Hospital ID: ${stMercyHosp._id}`);
    console.log(`   Admin Manager: ${stMercyHosp.adminManagerEmail}\n`);

    // Clear existing practitioners for this hospital
    console.log('🗑️  Clearing existing practitioners for St Mercy Hospital...');
    const deleteResult = await ClinicalStaff.deleteMany({ hospitalId: stMercyHosp._id });
    console.log(`   Deleted: ${deleteResult.deletedCount} records\n`);

    // Hash passwords
    console.log(`📋 Hashing ${testPractitioners.length} passwords...`);
    const practitionersWithHashedPasswords = await Promise.all(
      testPractitioners.map(async (p) => ({
        ...p,
        password: await bcrypt.hash(p.password, 12),
        hospitalId: stMercyHosp._id,
        status: 'ACTIVE'
      }))
    );

    // Insert practitioners
    console.log(`📝 Inserting ${testPractitioners.length} practitioners...\n`);
    const inserted = await ClinicalStaff.insertMany(practitionersWithHashedPasswords);

    console.log('✅ SEEDING COMPLETE!\n');
    inserted.forEach((p, idx) => {
      console.log(`${idx + 1}. ${p.firstName} ${p.lastName}`);
      console.log(`   Email: ${p.email}`);
      console.log(`   Role: ${p.systemRole}`);
      console.log(`   Department: ${p.assignment?.department}`);
      console.log(`   ID: ${p._id}\n`);
    });

    // Verify the count
    const count = await ClinicalStaff.countDocuments({ hospitalId: stMercyHosp._id });
    console.log(`📊 Total staff in St Mercy Hospital: ${count}`);

  } catch (err) {
    console.error('❌ ERROR:', err.message);
    process.exit(1);
  } finally {
    await mongoose.disconnect();
  }
}

main();
