#!/usr/bin/env node
/**
 * Diagnostic script: List all users across Manager/ClinicalStaff/Practitioner
 * Helps identify why logins are failing
 */

import mongoose from 'mongoose';
import { Manager } from './src/models/manager.model.js';
import { ClinicalStaff, Practitioner } from './src/models/clinicalstaff.model.js';
import dotenv from 'dotenv';

dotenv.config();

async function main() {
  try {
    console.log('🔍 Connecting to MongoDB...');
    await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/nursedb');
    console.log('✅ Connected\n');

    // 1. Check Managers
    console.log('📋 MANAGERS:');
    const managers = await Manager.find({}, 'email name role').lean();
    if (managers.length === 0) {
      console.log('   (none found)');
    } else {
      managers.forEach(m => {
        console.log(`   - ${m.email} (${m.name}, Role: ${m.role})`);
      });
    }

    // 2. Check ClinicalStaff
    console.log('\n📋 CLINICAL STAFF (ClinicalStaff model):');
    const staff = await ClinicalStaff.find({}, 'email firstName lastName role systemRole').lean();
    if (staff.length === 0) {
      console.log('   (none found)');
    } else {
      staff.forEach(s => {
        const name = s.firstName ? `${s.firstName} ${s.lastName}` : '(no name)';
        const role = s.role || s.systemRole || '(no role)';
        console.log(`   - ${s.email} (${name}, Role: ${role})`);
      });
    }

    // 3. Check Practitioners (same collection as ClinicalStaff but via Practitioner model)
    console.log('\n📋 PRACTITIONERS (Practitioner model):');
    const practitioners = await Practitioner.find({}, 'email firstName lastName systemRole').lean();
    if (practitioners.length === 0) {
      console.log('   (none found)');
    } else {
      practitioners.forEach(p => {
        const name = p.firstName ? `${p.firstName} ${p.lastName}` : '(no name)';
        console.log(`   - ${p.email} (${name}, Role: ${p.systemRole})`);
      });
    }

    // 4. Look for dr.chen specifically
    console.log('\n🔎 Searching for dr.chen@stmercy.hospital:');
    const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const emailRegex = new RegExp(`^${escapeRegex('dr.chen@stmercy.hospital')}$`, 'i');

    const drChenManager = await Manager.findOne({ email: emailRegex }).lean();
    const drChenStaff = await ClinicalStaff.findOne({ email: emailRegex }).lean();
    const drChenPract = await Practitioner.findOne({ email: emailRegex }).lean();

    if (!drChenManager && !drChenStaff && !drChenPract) {
      console.log('   ❌ NOT FOUND in any collection');
    } else {
      if (drChenManager) console.log(`   ✅ Found in Manager: ${JSON.stringify(drChenManager, null, 2)}`);
      if (drChenStaff) console.log(`   ✅ Found in ClinicalStaff: ${JSON.stringify(drChenStaff, null, 2)}`);
      if (drChenPract) console.log(`   ✅ Found in Practitioner: ${JSON.stringify(drChenPract, null, 2)}`);
    }

    // 5. Summary
    console.log('\n📊 SUMMARY:');
    console.log(`   Total Managers: ${managers.length}`);
    console.log(`   Total ClinicalStaff: ${staff.length}`);
    console.log(`   Total Practitioners: ${practitioners.length}`);
    console.log(`   dr.chen@stmercy.hospital: ${drChenManager || drChenStaff || drChenPract ? '✅ EXISTS' : '❌ MISSING'}`);

    if (!drChenManager && !drChenStaff && !drChenPract && (managers.length > 0 || staff.length > 0)) {
      console.log('\n💡 Recommendation: Use one of the existing users above, or create dr.chen via the seeding script.');
    }

    process.exit(0);
  } catch (err) {
    console.error('❌ ERROR:', err);
    process.exit(1);
  }
}

main();
