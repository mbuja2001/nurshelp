import dotenv from 'dotenv';
import { MongoClient, ObjectId } from 'mongodb';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function run() {
  const client = new MongoClient(MONGO_URI);
  try {
    await client.connect();
    const db = client.db();
    const nurses = db.collection('nurses');
    const shifts = db.collection('shifts');

    const email = 'dr.chen@stmercy.hospital';
    const doc = await nurses.findOne({ email });
    if (!doc) {
      console.log('Dr. Chen not found');
      return;
    }

    console.log('Before update: role =', doc.role);

    if (doc.role !== 'PHYSICIAN') {
      await nurses.updateOne({ _id: doc._id }, { $set: { role: 'PHYSICIAN' } });
      console.log('Updated role to PHYSICIAN');
    } else {
      console.log('Role already PHYSICIAN');
    }

    // Update any shift assignedStaff entries that reference this staff id
    const staffId = doc._id instanceof ObjectId ? doc._id : new ObjectId(doc._id);
    const shiftUpdateResult = await shifts.updateMany(
      { 'assignedStaff.staffID': staffId },
      { $set: { 'assignedStaff.$[elem].type': 'PHYSICIAN' } },
      { arrayFilters: [ { 'elem.staffID': staffId } ] }
    );

    console.log(`Shifts matched/modified: ${shiftUpdateResult.matchedCount}/${shiftUpdateResult.modifiedCount}`);

    // Verify final document
    const updated = await nurses.findOne({ email });
    console.log('After update:', { _id: updated._id, email: updated.email, role: updated.role, department: updated.department });

  } catch (err) {
    console.error('ERROR', err.message);
  } finally {
    await client.close();
  }
}

run();
