import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Hospital } from './src/models/hospital.model.js';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function checkHospitals() {
  try {
    await mongoose.connect(MONGO_URI);
    const hospitals = await Hospital.find({}, { name: 1, _id: 1 });
    console.log('\n📋 Available Hospitals:');
    hospitals.forEach((h, i) => {
      console.log(`  ${i + 1}. ${h.name}`);
    });
    console.log('\n');
    await mongoose.connection.close();
  } catch (err) {
    console.error('Error:', err.message);
  }
}

checkHospitals();
