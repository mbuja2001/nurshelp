import dotenv from 'dotenv';

dotenv.config();

const BACKEND_URL = 'http://localhost:5001';

async function testStaffListQuery() {
  try {
    console.log('🏥 TESTING STAFF LIST ENDPOINT FOR ST MERCY HOSPITAL');
    console.log('=' .repeat(80));

    // Correct credentials for St Mercy Hospital admin
    const adminEmail = 'alice.johnson@stmercy.hospital';
    const adminPassword = 'password123'; // updated password

    // Step 1: Login as St Mercy admin
    console.log('\n📋 STEP 1: Login as St Mercy Hospital Admin');
    console.log('-'.repeat(80));
    console.log(`Email: ${adminEmail}`);
    console.log(`Password: ${adminPassword}\n`);
    
    const loginRes = await fetch(`${BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: adminEmail,
        password: adminPassword
      })
    });

    const loginData = await loginRes.json();
    console.log(`Response Status: ${loginRes.status}`);

    if (!loginData.token) {
      console.log('❌ Login failed!');
      console.log('Response:', JSON.stringify(loginData, null, 2));
      process.exit(1);
    }

    const token = loginData.token;
    console.log('✅ Login Successful!');
    console.log(`User: ${loginData.user?.name}`);
    console.log(`Role: ${loginData.user?.role}`);
    console.log(`Hospital: ${loginData.user?.hospitalName}`);
    console.log(`Hospital ID: ${loginData.user?.hospitalId}`);
    console.log(`Token: ${token.substring(0, 30)}...`);

    // Step 2: Query staff list endpoint
    console.log('\n📋 STEP 2: Query Staff List Endpoint');
    console.log('-'.repeat(80));
    
    const staffRes = await fetch(`${BACKEND_URL}/api/manager/staff/list`, {
      headers: { Authorization: `Bearer ${token}` }
    });

    console.log(`Response Status: ${staffRes.status}`);
    const staffData = await staffRes.json();
    console.log(`Response Body:`, JSON.stringify(staffData, null, 2));

    if (!staffData.success) {
      console.log('❌ Staff query failed!');
      process.exit(1);
    }

    console.log(`\n✅ Staff Query Successful!`);
    console.log(`Total Staff Found: ${staffData.count}`);

    if (staffData.staff && staffData.staff.length > 0) {
      console.log(`\n👨‍⚕️  Staff Members at St Mercy Hospital:`);
      staffData.staff.forEach((staff, idx) => {
        console.log(`\n${idx + 1}. ${staff.firstName} ${staff.lastName}`);
        console.log(`   Email: ${staff.email}`);
        console.log(`   Role: ${staff.systemRole}`);
        console.log(`   Department: ${staff.assignment?.department || staff.department || 'N/A'}`);
        console.log(`   Status: ${staff.status || 'N/A'}`);
      });
    } else {
      console.log('\n⚠️  No staff members found in St Mercy Hospital!');
      console.log('This is expected if no practitioners have been seeded to this hospital yet.');
    }

    // Step 3: Test filter endpoints
    console.log('\n📋 STEP 3: Test Filters');
    console.log('-'.repeat(80));

    const filters = [
      { desc: 'Filter by Role=NURSE', url: '?role=NURSE' },
      { desc: 'Filter by Role=PHYSICIAN', url: '?role=PHYSICIAN' },
      { desc: 'Search for "Wong"', url: '?search=Wong' }
    ];

    for (const filter of filters) {
      const filterUrl = `${BACKEND_URL}/api/manager/staff/list${filter.url}`;
      const filterRes = await fetch(filterUrl, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const filterData = await filterRes.json();
      
      console.log(`\n${filter.desc}:`);
      console.log(`  Status: ${filterRes.status}`);
      console.log(`  Count: ${filterData.count}`);
      
      if (filterData.staff && filterData.staff.length > 0) {
        filterData.staff.forEach(s => {
          console.log(`    - ${s.firstName} ${s.lastName} (${s.systemRole})`);
        });
      } else {
        console.log(`    (No results)`);
      }
    }

  } catch (err) {
    console.error('❌ ERROR:', err.message);
    process.exit(1);
  }

  process.exit(0);
}

testStaffListQuery();
