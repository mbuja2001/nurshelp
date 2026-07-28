import dotenv from 'dotenv';
import { MongoClient, ObjectId } from 'mongodb';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function run() {
  const client = new MongoClient(MONGO_URI);
  try {
    await client.connect();
    const db = client.db();
    const col = db.collection('nurses');
    const doc = await col.findOne({ email: 'dr.chen@stmercy.hospital' });
    if (!doc) {
      console.log('NOT FOUND');
    } else {
      const out = {
        _id: doc._id,
        name: doc.name,
        surname: doc.surname,
        email: doc.email,
        role: doc.role,
        department: doc.department,
        hospitalId: doc.hospitalId,
        createdBy: doc.createdBy,
        licenseNumber: doc.licenseNumber,
        createdAt: doc.createdAt || doc.dateCreated,
        raw: doc
      };
      console.log(JSON.stringify(out, null, 2));
    }
  } catch (err) {
    console.error('ERROR', err.message);
  } finally {
    await client.close();
  }
}

run();
