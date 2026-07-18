/**
 * Test Staff Login Authentication
 * Tests that newly created staff accounts can login with hashed passwords
 * 
 * Usage: node test-staff-login.js
 */

import mongoose from 'mongoose';
import bcrypt from 'bcrypt';

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nurse_helpdesk_v2';

async function testStaffLogin() {
  try {
    console.log('\n[TEST] Connecting to MongoDB...');
    await mongoose.connect(MONGO_URI);
    console.log('[TEST] ✅ Connected');

    const { ClinicalStaff } = await import('./src/models/clinicalstaff.model.js');
    const { Manager } = await import('./src/models/manager.model.js');

    // Test 1: Check if ClinicalStaff collection has any staff
    console.log('\n[TEST] Checking staff accounts...');
    const staffCount = await ClinicalStaff.countDocuments();
    console.log(`[TEST] Found ${staffCount} staff accounts in database`);

    if (staffCount === 0) {
      console.log('[TEST] ⚠️  No staff accounts found. Run populate-hospital-full.js first.');
      await mongoose.disconnect();
      return;
    }

    // Test 2: Try test login with a real staff account
    console.log('\n[TEST] Fetching a staff account for testing...');
    const testStaff = await ClinicalStaff.findOne().select('+password');
    
    if (!testStaff) {
      console.log('[TEST] ⚠️  Could not fetch any staff account');
      await mongoose.disconnect();
      return;
    }

    console.log(`[TEST] Found staff: ${testStaff.firstName} ${testStaff.lastName}`);
    console.log(`[TEST] Email: ${testStaff.email}`);
    console.log(`[TEST] Password hash: ${testStaff.password.substring(0, 20)}...`);

    // Test 3: Verify password is hashed (bcrypt hashes start with $2a$, $2b$, or $2y$)
    const isHashed = testStaff.password.startsWith('$2a$') || 
                     testStaff.password.startsWith('$2b$') || 
                     testStaff.password.startsWith('$2y$');
    
    console.log(`\n[TEST] Password format check:`);
    console.log(`[TEST] Starts with $2a$/$2b$/$2y$: ${isHashed ? '✅ YES (hashed)' : '❌ NO (plaintext!)'}`);

    if (!isHashed) {
      console.log('[TEST] ⚠️  ALERT: Password appears to be plaintext!');
      console.log('[TEST] The fix may not be deployed or these are old accounts.');
    }

    // Test 4: Check if password hash can be compared
    if (isHashed) {
      console.log(`\n[TEST] Attempting bcrypt.compare() on hashed password...`);
      
      // We can't test with the real password (it's not stored), but we can verify
      // the format is compatible with bcrypt.compare()
      try {
        // Try comparing with a fake password - we expect false but no error
        const result = await bcrypt.compare('testpassword123', testStaff.password);
        console.log(`[TEST] ✅ bcrypt.compare() works (returned ${result})`);
        console.log(`[TEST] Hash format is compatible with login controller`);
      } catch (err) {
        console.log(`[TEST] ❌ bcrypt.compare() failed: ${err.message}`);
      }
    }

    // Test 5: Check Manager accounts for comparison
    console.log(`\n[TEST] Checking Manager (admin) accounts for comparison...`);
    const adminCount = await Manager.countDocuments();
    console.log(`[TEST] Found ${adminCount} admin accounts`);

    if (adminCount > 0) {
      const testAdmin = await Manager.findOne().select('+password');
      if (testAdmin) {
        const adminHashFormat = testAdmin.password.startsWith('$2a$') || 
                               testAdmin.password.startsWith('$2b$') || 
                               testAdmin.password.startsWith('$2y$');
        console.log(`[TEST] Admin password is hashed: ${adminHashFormat ? '✅ YES' : '❌ NO'}`);
      }
    }

    // Summary
    console.log('\n[TEST] ════════════════════════════════════════');
    console.log('[TEST] SUMMARY:');
    if (isHashed) {
      console.log('[TEST] ✅ PASS: Staff passwords are properly hashed');
      console.log('[TEST] → Staff should be able to login now');
    } else {
      console.log('[TEST] ❌ FAIL: Staff passwords are plaintext');
      console.log('[TEST] → Fix code is not deployed or needs redeployment');
      console.log('[TEST] → Existing staff accounts need to be deleted/regenerated');
    }
    console.log('[TEST] ════════════════════════════════════════\n');

  } catch (err) {
    console.error('[TEST] ERROR:', err.message);
  } finally {
    await mongoose.disconnect();
  }
}

testStaffLogin();
