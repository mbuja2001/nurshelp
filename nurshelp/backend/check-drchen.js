import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function run() {
  try {
    await mongoose.connect(MONGO_URI);
    const doc = await ClinicalStaff.findOne({ email: 'dr.chen@stmercy.hospital' }).lean();
    if (!doc) {
      console.log('NOT FOUND: dr.chen@stmercy.hospital');
    } else {
      // Print concise diagnostic fields
      const out = {
        _id: doc._id,
        name: doc.name,
        surname: doc.surname,
        email: doc.email,
        role: doc.role || doc.type || null,
        department: doc.department || null,
        hospitalId: doc.hospitalId || null,
        createdBy: doc.createdBy || null,
        licenseNumber: doc.licenseNumber || null,
        createdAt: doc.createdAt || null,
        raw: doc
      };
      console.log(JSON.stringify(out, null, 2));
    }
    await mongoose.connection.close();
  } catch (err) {
    console.error('ERROR:', err.message);
    process.exit(1);
  }
}

run();
