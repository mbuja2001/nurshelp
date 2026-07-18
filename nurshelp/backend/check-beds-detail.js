import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Ward } from './src/models/ward.model.js';
import { BedSchedule } from './src/models/bedSchedule.model.js';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function checkBedsDetail() {
  try {
    await mongoose.connect(MONGO_URI);
    
    const beds = await BedSchedule.find({}).lean();
    const wardIds = [...new Set(beds.map(b => b.wardId.toString()))];
    
    console.log(`\n🛏️ Beds grouped by ward:`);
    for (const wId of wardIds) {
      const bedCount = beds.filter(b => b.wardId.toString() === wId).length;
      const ward = await Ward.findById(wId);
      console.log(`  Ward: ${ward?.wardName} (${wId}) - ${bedCount} beds`);
    }
    
    console.log('\n');
    await mongoose.connection.close();
  } catch (err) {
    console.error('Error:', err.message);
  }
}

checkBedsDetail();
