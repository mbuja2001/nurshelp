#!/usr/bin/env node
/**
 * Seed test practitioners (staff) with proper roles
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
    await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/nursedb');
    console.log('✅ Connected\n');

    // Get a manager to link practitioners to (use the first one)
    let manager = await Manager.findOne({}).lean();

    // If no manager exists, create a test hospital and an admin manager
    if (!manager) {
      console.log('⚠️  No managers found. Creating test hospital and admin manager...');

      const hospitalDoc = await Hospital.create({
        hospitalID: `HOSP-${Date.now()}`,
        name: 'St. Mercy Test Hospital',
        adminManagerName: 'Admin',
        adminManagerEmail: 'admin@stmercy.hospital',
        departments: [ { name: 'ED' }, { name: 'ICU' }, { name: 'CARDIOLOGY' } ]
      });

      const adminPassword = 'admin123';
      const hashed = await bcrypt.hash(adminPassword, 12);

      // Use insertMany to bypass any hooks
      const [mgr] = await Manager.insertMany([{
        hospitalId: hospitalDoc._id,
        managerID: `MGR-${Date.now()}`,
        name: 'Admin',
        surname: 'User',
        email: 'admin@stmercy.hospital',
        password: hashed,
        department: 'All',
        role: 'ADMIN',
        isActive: true,
        permissions: {
          canCreateShifts: true,
          canEditShifts: true,
          canDeleteShifts: true,
          canAssignStaff: true,
          canApproveLeaves: true,
          canManageRoles: true,
          canViewReports: true
        }
      }]);

      // Link admin to hospital
      hospitalDoc.adminManagerId = mgr._id;
      hospitalDoc.adminManagerName = `${mgr.name} ${mgr.surname}`;
      hospitalDoc.adminManagerEmail = mgr.email;
      await hospitalDoc.save();

      manager = mgr.toObject();

      console.log(`   ✅ Created admin manager: ${mgr.email} (password: ${adminPassword})`);
      console.log(`   ✅ Created hospital: ${hospitalDoc.name} (${hospitalDoc._id})`);
    }

    console.log(`📋 Seeding ${testPractitioners.length} test practitioners...`);
    console.log(`   Hospital ID: ${manager.hospitalId || manager.hospitalId}\n`);

    let created = 0;
    
    // Pre-hash all passwords before inserting
    const hashPromises = testPractitioners.map(p => 
      bcrypt.hash(p.password, 12).then(hash => ({
        ...p,
        password: hash
      }))
    );
    const practitioners = await Promise.all(hashPromises);

    // Insert with hospitalId and status
    const docsToInsert = practitioners.map(p => ({
      ...p,
      hospitalId: manager.hospitalId,
      status: 'ACTIVE',
      metadata: {
        createdBy: manager._id
      }
    }));

    try {
      // Use insertMany to bypass hooks
      const inserted = await Practitioner.insertMany(docsToInsert);
      for (const doc of inserted) {
        console.log(`   ✅ ${doc.email} (${doc.systemRole}) - ${doc.firstName} ${doc.lastName}`);
        console.log(`      ID: ${doc._id}`);
        console.log(`      Department: ${doc.assignment.department}`);
        created++;
      }
    } catch (err) {
      if (err.code === 11000) {
        // Duplicate key, try one by one
        for (const p of docsToInsert) {
          try {
            const existing = await Practitioner.findOne({ email: p.email });
            if (existing) {
              console.log(`   ⏭️  ${p.email} (already exists)`);
            } else {
              const doc = await Practitioner.insertMany([p]);
              console.log(`   ✅ ${p.email} (${p.systemRole}) - ${p.firstName} ${p.lastName}`);
              created++;
            }
          } catch (e) {
            console.error(`   ❌ ${p.email}: ${e.message}`);
          }
        }
      } else {
        throw err;
      }
    }

    console.log(`\n📊 Summary:`);
    console.log(`   Created: ${created}`);
    console.log(`   Skipped: ${testPractitioners.length - created}`);

    console.log('\n🔐 Test Credentials:');
    testPractitioners.forEach(p => {
      console.log(`   ${p.email} / password123`);
    });

    console.log('\n✨ Practitioners seeded successfully!');
    process.exit(0);
  } catch (err) {
    console.error('❌ ERROR:', err);
    process.exit(1);
  }
}

main();
