import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();

const encounterSchema = new mongoose.Schema({}, { strict: false });
const Encounter = mongoose.model('Encounter', encounterSchema, 'encounters');

async function fixStatus() {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/nurse-helpdesk');
    console.log('✅ Connected to MongoDB');

    // Find encounter with OFFICIAL_REPORT status
    const beforeEncounter = await Encounter.findOne({ status: 'OFFICIAL_REPORT' });
    if (beforeEncounter) {
      console.log('Found encounter with OFFICIAL_REPORT status:');
      console.log('  ID:', beforeEncounter._id);
      console.log('  Patient:', beforeEncounter.patient?.name);
      console.log('  Current Status:', beforeEncounter.status);
    }

    // Update the existing encounter with OFFICIAL_REPORT status to AWAITING_PHYSICIAN
    const result = await Encounter.updateOne(
      { status: 'OFFICIAL_REPORT' },
      { $set: { status: 'AWAITING_PHYSICIAN' } }
    );

    console.log('\n📝 Update result:');
    console.log('  Matched:', result.matchedCount);
    console.log('  Modified:', result.modifiedCount);

    // Verify the update
    const afterEncounter = await Encounter.findOne({ patient: { $exists: true } }).limit(1);
    console.log('\n✅ Verification - Encounter status now:', afterEncounter?.status);

    await mongoose.connection.close();
    console.log('\n✅ Done - Database connection closed');
    process.exit(0);
  } catch (err) {
    console.error('❌ Error:', err.message);
    process.exit(1);
  }
}

fixStatus();
