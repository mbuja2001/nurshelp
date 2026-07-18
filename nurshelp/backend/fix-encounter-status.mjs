import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();

async function fixStatus() {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/nurse-helpdesk');
    console.log('Connected to MongoDB');

    // Update the existing encounter with OFFICIAL_REPORT status to AWAITING_PHYSICIAN
    const result = await mongoose.connection.db.collection('encounters').updateOne(
      { status: 'OFFICIAL_REPORT' },
      { $set: { status: 'AWAITING_PHYSICIAN' } }
    );

    console.log('Updated encounters:', result.modifiedCount);
    
    // Verify the update
    const encounter = await mongoose.connection.db.collection('encounters').findOne({ status: 'AWAITING_PHYSICIAN' });
    console.log('Encounter now has status:', encounter?.status);

    await mongoose.connection.close();
    console.log('Done');
    process.exit(0);
  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  }
}

fixStatus();
