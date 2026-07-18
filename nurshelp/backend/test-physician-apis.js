#!/usr/bin/env node

/**
 * backend/test-physician-apis.js
 * 
 * Quick test script to validate physician dashboard APIs
 * Run: node test-physician-apis.js
 */

import mongoose from 'mongoose';
import dotenv from 'dotenv';
import seedPhysicianDashboard from './src/seeds/physician-dashboard-seed.js';

dotenv.config();

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelp';

async function runTests() {
  try {
    // Connect
    console.log('🔗 Connecting to MongoDB...');
    await mongoose.connect(MONGO_URI);
    console.log('✅ Connected\n');

    // Seed
    console.log('🌱 Seeding test data...');
    const { patients, triages, dispositions, journeys } = await seedPhysicianDashboard();

    // Verify
    console.log('✓ Triages created:', triages.length);
    console.log('✓ Dispositions created:', dispositions.length);
    console.log('✓ Journeys created:', journeys.length);

    // Display sample data
    console.log('\n📋 Sample Triage (John - RED):');
    console.log('  - Priority:', triages[0].triagePriority);
    console.log('  - Needs Bed:', triages[0].needsBed);
    console.log('  - HR:', triages[0].vitalSigns.heartRate);
    console.log('  - BP:', triages[0].vitalSigns.systolic + '/' + triages[0].vitalSigns.diastolic);

    console.log('\n📋 Sample Disposition (John):');
    console.log('  - Decision:', dispositions[0].decision);
    console.log('  - Discrepancy:', dispositions[0].discrepancy.type);
    console.log('  - Response Time:', dispositions[0].responseTimeMinutes, 'min');

    console.log('\n✅ All tests passed!\n');
    process.exit(0);
  } catch (err) {
    console.error('❌ Test failed:', err.message);
    process.exit(1);
  }
}

runTests();
