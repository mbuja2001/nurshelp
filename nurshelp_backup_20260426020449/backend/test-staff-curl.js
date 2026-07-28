#!/usr/bin/env node

import { execSync } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';
import { join } from 'path';

const BACKEND_URL = 'http://localhost:5001';
const tmpDir = process.cwd();

function curlRequest(method, url, data = null, token = null) {
  let cmd = `curl -s -X ${method} "${url}" -H "Content-Type: application/json"`;
  
  if (token) {
    cmd += ` -H "Authorization: Bearer ${token}"`;
  }
  
  if (data) {
    const tmpFile = join(tmpDir, 'req-tmp.json');
    writeFileSync(tmpFile, JSON.stringify(data));
    cmd += ` -d @${tmpFile}`;
    try {
      const result = execSync(cmd, { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
      unlinkSync(tmpFile);
      return JSON.parse(result);
    } catch (err) {
      unlinkSync(tmpFile);
      throw err;
    }
  } else {
    const result = execSync(cmd, { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
    return JSON.parse(result);
  }
}

async function testStaffList() {
  try {
    console.log('🏥 TESTING STAFF LIST ENDPOINT FOR ST MERCY HOSPITAL');
    console.log('='.repeat(80));

    // Step 1: Login
    console.log('\n📋 STEP 1: Login as St Mercy Hospital Admin');
    console.log('-'.repeat(80));
    console.log('Email: alice.johnson@stmercy.hospital');
    console.log('Password: password123\n');

    const loginRes = curlRequest('POST', `${BACKEND_URL}/api/auth/login`, {
      email: 'alice.johnson@stmercy.hospital',
      password: 'password123'
    });

    console.log('Response:', JSON.stringify(loginRes, null, 2));

    if (!loginRes.token) {
      console.log('\n❌ LOGIN FAILED!');
      process.exit(1);
    }

    const token = loginRes.token;
    console.log('\n✅ LOGIN SUCCESSFUL!');
    console.log(`User: ${loginRes.user?.name}`);
    console.log(`Hospital: ${loginRes.user?.hospitalName}`);
    console.log(`Role: ${loginRes.user?.role}`);
    console.log(`Hospital ID: ${loginRes.user?.hospitalId}`);

    // Step 2: Get staff list
    console.log('\n📋 STEP 2: Query Staff List Endpoint');
    console.log('-'.repeat(80));
    console.log('URL: GET /api/manager/staff/list\n');

    const staffRes = curlRequest('GET', `${BACKEND_URL}/api/manager/staff/list`, null, token);

    console.log('Response:', JSON.stringify(staffRes, null, 2));

    if (!staffRes.success) {
      console.log('\n❌ STAFF QUERY FAILED!');
      process.exit(1);
    }

    console.log(`\n✅ STAFF QUERY SUCCESSFUL!`);
    console.log(`Total Staff Found: ${staffRes.count}`);

    if (staffRes.staff && staffRes.staff.length > 0) {
      console.log('\n👨‍⚕️  STAFF MEMBERS AT ST MERCY HOSPITAL:');
      staffRes.staff.forEach((staff, idx) => {
        console.log(`\n${idx + 1}. ${staff.firstName} ${staff.lastName}`);
        console.log(`   Email: ${staff.email}`);
        console.log(`   Role: ${staff.systemRole}`);
        console.log(`   Department: ${staff.assignment?.department || staff.department || 'N/A'}`);
        console.log(`   Status: ${staff.status}`);
      });
    } else {
      console.log('\n⚠️  NO STAFF MEMBERS FOUND!');
    }

    // Step 3: Test filters
    console.log('\n📋 STEP 3: Test Filters');
    console.log('-'.repeat(80));

    const filters = [
      { desc: 'NURSES', url: '/api/manager/staff/list?role=NURSE' },
      { desc: 'PHYSICIANS', url: '/api/manager/staff/list?role=PHYSICIAN' },
      { desc: 'Search: "Wong"', url: '/api/manager/staff/list?search=Wong' }
    ];

    for (const filter of filters) {
      const filterRes = curlRequest('GET', `${BACKEND_URL}${filter.url}`, null, token);
      console.log(`\n${filter.desc}:`);
      console.log(`  Count: ${filterRes.count}`);
      if (filterRes.staff && filterRes.staff.length > 0) {
        filterRes.staff.forEach(s => {
          console.log(`    - ${s.firstName} ${s.lastName} (${s.systemRole})`);
        });
      } else {
        console.log('    (No results)');
      }
    }

    console.log('\n' + '='.repeat(80));
    console.log('✅ TEST COMPLETE\n');

  } catch (err) {
    console.error('❌ ERROR:', err.message);
    process.exit(1);
  }
}

testStaffList();
