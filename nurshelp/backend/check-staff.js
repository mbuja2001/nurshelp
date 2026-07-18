import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';
import { Manager } from './src/models/manager.model.js';
import { Hospital } from './src/models/hospital.model.js';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function checkStaff() {
  try {
    await mongoose.connect(MONGO_URI);
    
    // Find hospital
    const hospital = await Hospital.findOne({ name: 'St Mercy Hospital' });
    if (!hospital) {
      console.log('❌ Hospital "St Mercy Hospital" not found');
    } else {
      console.log(`✅ Found Hospital: ${hospital.name} (ID: ${hospital._id})`);
    }
    
    // Find physician
    const physician = await ClinicalStaff.findOne({ email: 'dr.chen@stmercy.hospital' });
    if (!physician) {
      console.log('❌ Physician "dr.chen@stmercy.hospital" not found');
    } else {
      console.log(`✅ Found Physician: ${physician.email} (${physician.name}) in hospital ${physician.hospitalId}`);
    }
    
    // Find admin
    const admin = await Manager.findOne({ email: 'alice.johnson@stmercy.hospital' });
    if (!admin) {
      console.log('❌ Admin "alice.johnson@stmercy.hospital" not found');
    } else {
      console.log(`✅ Found Admin: ${admin.email} (${admin.name}) in hospital ${admin.hospitalId}`);
    }
    
    console.log('\n');
    await mongoose.connection.close();
  } catch (err) {
    console.error('Error:', err.message);
  }
}

checkStaff();
