import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();
const MONGO_URL = process.env.MONGO_URI;

async function listManagers() {
  try {
    await mongoose.connect(MONGO_URL);
    console.log('✅ Connected to MongoDB\n');

    // Import models
    const { Manager } = await import('./src/models/manager.model.js');
    const { Hospital } = await import('./src/models/hospital.model.js');
    const { Practitioner, ClinicalStaff } = await import('./src/models/clinicalstaff.model.js');

    // List all managers
    console.log('=' .repeat(70));
    console.log('📋 MANAGERS IN DATABASE');
    console.log('=' .repeat(70));
    
    const managers = await Manager.find().select('name surname email role hospitalId isActive');
    console.log(`Total Managers: ${managers.length}\n`);
    
    managers.forEach((mgr, idx) => {
      console.log(`${idx + 1}. ${mgr.name} ${mgr.surname}`);
      console.log(`   Email: ${mgr.email}`);
      console.log(`   Role: ${mgr.role}`);
      console.log(`   Hospital ID: ${mgr.hospitalId}`);
      console.log(`   Active: ${mgr.isActive}\n`);
    });

    // List hospitals associated with managers
    console.log('=' .repeat(70));
    console.log('🏥 HOSPITALS');
    console.log('=' .repeat(70));
    
    const hospitals = await Hospital.find().select('name adminManagerId adminManagerEmail');
    console.log(`Total Hospitals: ${hospitals.length}\n`);
    
    hospitals.forEach((hosp, idx) => {
      console.log(`${idx + 1}. ${hosp.name}`);
      console.log(`   ID: ${hosp._id}`);
      console.log(`   Admin Manager: ${hosp.adminManagerEmail}\n`);
    });

    // List staff (practitioners)
    console.log('=' .repeat(70));
    console.log('👨‍⚕️  PRACTITIONERS/STAFF');
    console.log('=' .repeat(70));
    
    const staff = await ClinicalStaff.find().select('firstName lastName email systemRole hospitalId status');
    console.log(`Total Staff: ${staff.length}\n`);
    
    // Group by hospital
    const staffByHospital = {};
    staff.forEach(s => {
      const hospId = s.hospitalId?.toString() || 'Unknown';
      if (!staffByHospital[hospId]) {
        staffByHospital[hospId] = [];
      }
      staffByHospital[hospId].push(s);
    });

    Object.entries(staffByHospital).forEach(([hospId, staffList]) => {
      console.log(`Hospital ${hospId} (${staffList.length} staff):`);
      staffList.forEach(s => {
        console.log(`  - ${s.firstName} ${s.lastName} (${s.email}) - Role: ${s.systemRole}, Status: ${s.status}`);
      });
      console.log();
    });

    // Find St Mercy Hospital specifically
    console.log('=' .repeat(70));
    console.log('🔍 ST MERCY HOSPITAL DETAILS');
    console.log('=' .repeat(70));
    
    const stMercyHosp = await Hospital.findOne({ name: /st\s+mercy/i });
    if (stMercyHosp) {
      console.log(`Hospital: ${stMercyHosp.name}`);
      console.log(`ID: ${stMercyHosp._id}`);
      console.log(`Admin Manager: ${stMercyHosp.adminManagerEmail}`);
      console.log(`Admin Manager ID: ${stMercyHosp.adminManagerId}`);
      
      // Find staff in this hospital
      const stMercyStaff = await ClinicalStaff.find({ hospitalId: stMercyHosp._id });
      console.log(`\nStaff in St Mercy (${stMercyStaff.length}):`);
      stMercyStaff.forEach(s => {
        console.log(`  - ${s.firstName} ${s.lastName} (${s.email}) - ${s.systemRole}`);
      });
    } else {
      console.log('❌ St Mercy Hospital not found');
    }

  } catch (err) {
    console.error('❌ ERROR:', err.message);
  } finally {
    await mongoose.disconnect();
  }
}

listManagers();
