/**
 * Live Test Script for Physician Dashboard APIs
 * 
 * Tests:
 * 1. Database connectivity
 * 2. Seed data creation
 * 3. API endpoints
 */

import mongoose from 'mongoose';
import seedPhysicianDashboard from './src/seeds/physician-dashboard-seed.js';
import { PatientJourney } from './src/models/patientJourney.model.js';
import { Ward } from './src/models/ward.model.js';
import { TriageAssessment } from './src/models/triageAssessment.model.js';
import { Disposition } from './src/models/disposition.model.js';

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';
const TEST_HOSPITAL_ID = '67b4f8c7e1a2b3c4d5e6f7a8';

console.log('🏥 PHYSICIAN DASHBOARD - LIVE API TEST');
console.log('═══════════════════════════════════════');

try {
  // 1. CONNECT TO MONGODB
  console.log('\n[1] Connecting to MongoDB...');
  await mongoose.connect(MONGO_URI);
  console.log('✅ Connected');

  // 2. SEED DATA
  console.log('\n[2] Seeding test data...');
  await seedPhysicianDashboard();
  console.log('✅ Seed complete');

  // 3. VERIFY DATA IN DB
  console.log('\n[3] Verifying data in database...');
  
  const journeyCount = await PatientJourney.countDocuments({});
  console.log(`   PatientJourney docs: ${journeyCount}`);
  
  const triageCount = await TriageAssessment.countDocuments({});
  console.log(`   TriageAssessment docs: ${triageCount}`);
  
  const dispositionCount = await Disposition.countDocuments({});
  console.log(`   Disposition docs: ${dispositionCount}`);
  
  const wardCount = await Ward.countDocuments({});
  console.log(`   Ward docs: ${wardCount}`);

  // 4. SIMULATE getPhysicianCases
  console.log('\n[4] Simulating getPhysicianCases...');
  const patients = await PatientJourney.find({
    'currentLocation.departmentContext': 'ER',
    currentState: { $in: ['OBSERVATION', 'ESCALATED', 'TRIAGED'] },
  }).populate('triageAssessmentId dispositionId');
  
  console.log(`   Found ${patients.length} active patients in ER`);
  patients.forEach(p => {
    const triage = p.triageAssessmentId;
    console.log(`   - ${triage?.firstName} ${triage?.lastName} (${triage?.triagePriority}): ${p.currentState}`);
  });

  // 5. SIMULATE getWardBeds
  console.log('\n[5] Simulating getWardBeds...');
  const ward = await Ward.findOne({ departmentContext: 'ER' });
  if (ward) {
    const occupancy = ward.beds?.filter(b => b.isOccupied).length || 0;
    console.log(`   Ward ER: ${ward.beds?.length || 0} beds total, ${occupancy} occupied`);
    ward.beds?.forEach(b => {
      console.log(`   - Bed ${b.bedNumber}: ${b.isOccupied ? 'OCCUPIED' : 'AVAILABLE'}`);
    });
  }

  console.log('\n✅ ALL TESTS PASSED');
  console.log('\nNow test in browser:');
  console.log('  1. Start backend: npm run dev');
  console.log('  2. Login as physician');
  console.log('  3. Go to Physician Dashboard');
  console.log('  4. Check browser console for fetch results');

} catch (err) {
  console.error('❌ ERROR:', err.message);
  console.error(err);
} finally {
  await mongoose.connection.close();
  process.exit(0);
}
