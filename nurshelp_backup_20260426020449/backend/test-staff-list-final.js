import { request } from 'undici';

const BACKEND_URL = 'http://localhost:5001';

async function makeRequest(url, method = 'GET', body = null, token = null) {
  const options = {
    path: new URL(url).pathname + new URL(url).search,
    method: method,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    }
  };

  if (body) {
    options.body = JSON.stringify(body);
  }

  try {
    const res = await request(url, options);
    let data = '';
    for await (const chunk of res.body) {
      data += chunk;
    }
    return {
      status: res.statusCode,
      data: JSON.parse(data)
    };
  } catch (err) {
    throw err;
  }
}

async function testStaffList() {
  try {
    console.log('🏥 TESTING STAFF LIST ENDPOINT FOR ST MERCY HOSPITAL');
    console.log('='.repeat(80));

    // Step 1: Login
    console.log('\n📋 STEP 1: Login as St Mercy Hospital Admin');
    console.log('-'.repeat(80));

    const loginRes = await makeRequest(`${BACKEND_URL}/api/auth/login`, 'POST', {
      email: 'alice.johnson@stmercy.hospital',
      password: 'password123'
    });

    console.log(`Status: ${loginRes.status}`);
    console.log(`Response:`, JSON.stringify(loginRes.data, null, 2));

    if (!loginRes.data.token) {
      console.log('\n❌ LOGIN FAILED!');
      process.exit(1);
    }

    const token = loginRes.data.token;
    console.log('\n✅ LOGIN SUCCESSFUL!');
    console.log(`User: ${loginRes.data.user?.name}`);
    console.log(`Hospital: ${loginRes.data.user?.hospitalName}`);
    console.log(`Role: ${loginRes.data.user?.role}`);
    console.log(`Hospital ID: ${loginRes.data.user?.hospitalId}`);

    // Step 2: Get staff list
    console.log('\n📋 STEP 2: Query Staff List Endpoint');
    console.log('-'.repeat(80));
    console.log(`URL: GET /api/manager/staff/list`);
    console.log(`Authorization: Bearer [token]\n`);

    const staffRes = await makeRequest(`${BACKEND_URL}/api/manager/staff/list`, 'GET', null, token);

    console.log(`Status: ${staffRes.status}`);
    console.log(`Response:`, JSON.stringify(staffRes.data, null, 2));

    if (!staffRes.data.success) {
      console.log('\n❌ STAFF QUERY FAILED!');
      process.exit(1);
    }

    console.log(`\n✅ STAFF QUERY SUCCESSFUL!`);
    console.log(`Total Staff Found: ${staffRes.data.count}`);

    if (staffRes.data.staff && staffRes.data.staff.length > 0) {
      console.log('\n👨‍⚕️  STAFF MEMBERS AT ST MERCY HOSPITAL:');
      staffRes.data.staff.forEach((staff, idx) => {
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
      const filterRes = await makeRequest(`${BACKEND_URL}${filter.url}`, 'GET', null, token);
      console.log(`\n${filter.desc}:`);
      console.log(`  Count: ${filterRes.data.count}`);
      if (filterRes.data.staff && filterRes.data.staff.length > 0) {
        filterRes.data.staff.forEach(s => {
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
