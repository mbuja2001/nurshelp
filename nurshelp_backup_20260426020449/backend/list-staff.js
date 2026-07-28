import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();

// Import models
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';
import { Hospital } from './src/models/hospital.model.js';

const mongoUri = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function main() {
  try {
    console.log('🔌 Connecting to MongoDB...');
    await mongoose.connect(mongoUri);
    console.log('✅ Connected to MongoDB\n');

    // List all hospitals
    console.log('📋 ALL HOSPITALS:');
    console.log('='.repeat(80));
    const hospitals = await Hospital.find().lean();
    
    if (hospitals.length === 0) {
      console.log('❌ No hospitals found');
    } else {
      hospitals.forEach((h, idx) => {
        console.log(`${idx + 1}. ID: ${h._id}`);
        console.log(`   Name: ${h.name}`);
        console.log(`   Email: ${h.email}`);
        console.log();
      });
    }

    // For each hospital, list all staff
    console.log('\n📊 STAFF BY HOSPITAL:');
    console.log('='.repeat(80));
    
    for (const hospital of hospitals) {
      console.log(`\n🏥 ${hospital.name} (ID: ${hospital._id})`);
      console.log('-'.repeat(80));
      
      const staff = await ClinicalStaff.find({ hospitalId: hospital._id })
        .select('firstName lastName email systemRole clinicalSpecialty assignment status')
        .lean();
      
      if (staff.length === 0) {
        console.log('  ❌ No staff members found');
      } else {
        staff.forEach((s, idx) => {
          console.log(`  ${idx + 1}. ${s.firstName} ${s.lastName}`);
          console.log(`     Email: ${s.email}`);
          console.log(`     Role: ${s.systemRole || s.role || 'N/A'}`);
          console.log(`     Specialty: ${s.clinicalSpecialty || 'N/A'}`);
          console.log(`     Department: ${s.assignment?.department || 'N/A'}`);
          console.log(`     Status: ${s.status || 'ACTIVE'}`);
          console.log();
        });
      }
    }

    // Search for specific staff member
    console.log('\n🔍 SEARCH: dr.chen@stmercy.hospital');
    console.log('='.repeat(80));
    const drChen = await ClinicalStaff.findOne({ email: /dr\.chen/i })
      .populate('hospitalId', 'name email')
      .lean();
    
    if (drChen) {
      console.log('✅ FOUND!');
      console.log(`  Name: ${drChen.firstName} ${drChen.lastName}`);
      console.log(`  Email: ${drChen.email}`);
      console.log(`  Role: ${drChen.systemRole || drChen.role || 'N/A'}`);
      console.log(`  Hospital: ${drChen.hospitalId?.name || 'N/A'} (${drChen.hospitalId?._id || 'N/A'})`);
      console.log(`  Status: ${drChen.status || 'ACTIVE'}`);
    } else {
      console.log('❌ NOT FOUND in practitioners collection');
      
      // Try alternative collections
      const { Manager } = await import('./src/models/manager.model.js');
      const manager = await Manager.findOne({ email: /dr\.chen/i }).lean();
      if (manager) {
        console.log('⚠️  Found in Manager collection instead (might be a manager account)');
        console.log(`  Name: ${manager.name}`);
        console.log(`  Email: ${manager.email}`);
      }
    }

    console.log('\n✅ Done');
    process.exit(0);
  } catch (err) {
    console.error('❌ ERROR:', err.message);
    process.exit(1);
  }
}

main();
