import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Ward } from './src/models/ward.model.js';
import { BedSchedule } from './src/models/bedSchedule.model.js';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function checkBeds() {
  try {
    await mongoose.connect(MONGO_URI);
    
    const wards = await Ward.find({});
    console.log(`\n📋 Total Wards: ${wards.length}`);
    for (const w of wards) {
      console.log(`  Ward: ${w.wardName} (${w._id})`);
    }
    
    const beds = await BedSchedule.find({});
    console.log(`\n🛏️ Total BedSchedule Docs: ${beds.length}`);
    beds.slice(0, 5).forEach(b => {
      console.log(`  Bed: ${b.bedId} (wardId: ${b.wardId})`);
    });
    
    console.log('\n');
    await mongoose.connection.close();
  } catch (err) {
    console.error('Error:', err.message);
  }
}

checkBeds();
