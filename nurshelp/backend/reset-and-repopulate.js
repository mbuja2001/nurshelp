/**
 * Clear Old Staff and Re-populate Hospital
 * 
 * IMPORTANT: This clears all ClinicalStaff accounts and re-runs the population
 * to ensure all passwords are properly hashed with the new fix.
 * 
 * Usage: node reset-and-repopulate.js
 */

import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import mongoose from 'mongoose';
import { spawn } from 'child_process';

// Load environment variables from .env file
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '.env') });

const MONGO_URI = process.env.MONGO_URI || 'mongodb+srv://mbuda:mbuda@cluster0.49acwqb.mongodb.net/?appName=Cluster0';

async function resetAndRepopulate() {
  try {
    console.log('\n╔════════════════════════════════════════════════╗');
    console.log('║  Reset Staff & Re-populate Hospital (FIXED)    ║');
    console.log('╚════════════════════════════════════════════════╝\n');

    // Step 1: Connect to MongoDB
    console.log('[RESET] Connecting to MongoDB...');
    await mongoose.connect(MONGO_URI);
    console.log('[RESET] ✅ Connected to MongoDB\n');

    // Step 2: Get collections
    const { ClinicalStaff } = await import('./src/models/clinicalstaff.model.js');
    const { Shift } = await import('./src/models/shift.model.js');

    // Step 3: Get counts before deletion
    const oldStaffCount = await ClinicalStaff.countDocuments();
    const oldShiftCount = await Shift.countDocuments();
    
    console.log(`[RESET] Current database state:`);
    console.log(`  - Staff accounts: ${oldStaffCount}`);
    console.log(`  - Shifts: ${oldShiftCount}\n`);

    // Step 4: Delete old staff accounts
    if (oldStaffCount > 0) {
      console.log(`[RESET] ⚠️  Deleting ${oldStaffCount} old staff accounts...`);
      const staffResult = await ClinicalStaff.deleteMany({});
      console.log(`[RESET] ✅ Deleted ${staffResult.deletedCount} staff records\n`);
    }

    // Step 5: Disconnect before running populate script
    await mongoose.disconnect();
    console.log('[RESET] Disconnected from MongoDB\n');

    // Step 6: Run populate-hospital-full.js
    console.log('[RESET] ═══════════════════════════════════════════');
    console.log('[RESET] Starting fresh hospital population...\n');

    return new Promise((resolve, reject) => {
      const populateProcess = spawn('node', ['populate-hospital-full.js'], {
        cwd: __dirname,  // Run in backend directory where populate-hospital-full.js is located
        stdio: 'inherit'  // Show all output from child process
      });

      populateProcess.on('close', (code) => {
        console.log('\n[RESET] ═══════════════════════════════════════════');
        if (code === 0) {
          console.log('[RESET] ✅ Population script completed successfully');
          console.log('[RESET]\n✨ Hospital re-populated with properly hashed passwords');
          console.log('[RESET] → All staff can now login\n');
          resolve();
        } else {
          reject(new Error(`Population script exited with code ${code}`));
        }
      });

      populateProcess.on('error', (err) => {
        reject(err);
      });
    });

  } catch (err) {
    console.error('\n[RESET] ❌ ERROR:', err.message);
    process.exit(1);
  }
}

resetAndRepopulate().catch((err) => {
  console.error('\n[RESET] ❌ Failed:', err.message);
  process.exit(1);
});
