import mongoose from 'mongoose';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve('./.env') });

const MONGO_URI = process.env.MONGO_URI;
await mongoose.connect(MONGO_URI);

const { Hospital } = await import('./src/models/hospital.model.js');
const { Ward } = await import('./src/models/ward.model.js');
const { BedSchedule } = await import('./src/models/bedSchedule.model.js');
const { Bed } = await import('./src/models/bed.model.js');

const latestHospitals = await Hospital.find().sort({ createdAt: -1 }).limit(10).lean();
console.log('HOSPITALS', latestHospitals.map(h => ({ id: String(h._id), name: h.name, hospitalID: h.hospitalID })));

const latest = latestHospitals[0];
if (!latest) {
  console.log('NO HOSPITALS');
  await mongoose.disconnect();
  process.exit(0);
}

const wardCount = await Ward.countDocuments({ hospitalId: latest._id });
const scheduleCount = await BedSchedule.countDocuments({});
const legacyCount = await Bed.countDocuments({ hospitalId: latest._id });
const wards = await Ward.find({ hospitalId: latest._id }).lean();
console.log('WARD_COUNT', wardCount);
console.log('SCHEDULE_COUNT', scheduleCount);
console.log('LEGACY_COUNT', legacyCount);
console.log('WARDS', wards.map(w => ({ id: String(w._id), dept: w.department, totalBeds: w.totalBeds, wardName: w.wardName })));
await mongoose.disconnect();
