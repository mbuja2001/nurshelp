import fetch from 'node-fetch';

const BACKEND_URL = 'http://localhost:5001';

async function testStaffList() {
  try {
    console.log('🏥 Testing Staff List for St Mercy Hospital\n');
    console.log('=' .repeat(70));

    // Step 1: Login as admin
    console.log('\n📋 STEP 1: Login as Admin at St Mercy Hospital');
    console.log('-'.repeat(70));
    
    const loginRes = await fetch(`${BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: 'admin@stmercy.hospital',
        password: 'admin123'
      })
    });

    const loginData = await loginRes.json();
    console.log(`Status: ${loginRes.status}`);

    if (!loginData.token) {
      console.log('❌ Admin login failed!');
      console.log('Response:', JSON.stringify(loginData, null, 2));
      
      // Try fallback admin emails
      console.log('\n🔄 Trying alternative admin logins...');
      
      const altAdmins = [
        { email: 'john.smith@stmarys.hospital', password: 'manager123' },
        { email: 'admin.stmercy@gmail.com', password: 'admin123' }
      ];
      
      for (const admin of altAdmins) {
        console.log(`\nTrying: ${admin.email}`);
        const altRes = await fetch(`${BACKEND_URL}/api/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(admin)
        });
        const altData = await altRes.json();
        console.log(`Status: ${altRes.status}, Token: ${altData.token ? '✅ Yes' : '❌ No'}`);
        if (altData.token) {
          console.log(`✅ Found working admin!`);
          loginData.token = altData.token;
          loginData.user = altData.user;
          break;
        }
      }
      
      if (!loginData.token) {
        console.log('❌ No admin account found with working credentials');
        process.exit(1);
      }
    }

    console.log('✅ Admin Login Successful!');
    console.log(`User: ${loginData.user?.name}`);
    console.log(`Role: ${loginData.user?.role}`);
    console.log(`Hospital: ${loginData.user?.hospitalName}`);
    console.log(`Hospital ID: ${loginData.user?.hospitalId}`);
    const token = loginData.token;

    // Step 2: Query staff list
    console.log('\n📋 STEP 2: Query Staff List Endpoint');
    console.log('-'.repeat(70));
    
    const url = `${BACKEND_URL}/api/manager/staff/list`;
    console.log(`URL: ${url}`);
    console.log(`Auth: Bearer ${token.substring(0, 20)}...`);

    const staffRes = await fetch(url, {
      headers: { Authorization: `Bearer ${token}` }
    });

    console.log(`Status: ${staffRes.status}`);
    const staffData = await staffRes.json();

    if (staffData.success) {
      console.log(`\n✅ Staff Query Successful!`);
      console.log(`Total Staff Found: ${staffData.count}`);
      
      if (staffData.staff && staffData.staff.length > 0) {
        console.log(`\n📋 Staff Members:`);
        staffData.staff.forEach((staff, idx) => {
          console.log(`\n  ${idx + 1}. ${staff.firstName} ${staff.lastName}`);
          console.log(`     Email: ${staff.email}`);
          console.log(`     Role: ${staff.systemRole}`);
          console.log(`     Department: ${staff.assignment?.department || staff.department || 'N/A'}`);
          console.log(`     Status: ${staff.status || 'N/A'}`);
          console.log(`     Hospital ID: ${staff.hospitalId}`);
        });
      } else {
        console.log('\n⚠️  No staff members found!');
      }
    } else {
      console.log('❌ Staff Query Failed!');
      console.log('Response:', JSON.stringify(staffData, null, 2));
    }

    // Step 3: Test with filters
    console.log('\n📋 STEP 3: Test Staff List with Filters');
    console.log('-'.repeat(70));

    const filters = [
      { name: 'Role=NURSE', params: '?role=NURSE' },
      { name: 'Role=PHYSICIAN', params: '?role=PHYSICIAN' },
      { name: 'Search=Chen', params: '?search=Chen' }
    ];

    for (const filter of filters) {
      const filterUrl = `${BACKEND_URL}/api/manager/staff/list${filter.params}`;
      const filterRes = await fetch(filterUrl, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const filterData = await filterRes.json();
      console.log(`\n${filter.name}:`);
      console.log(`  Status: ${filterRes.status}`);
      console.log(`  Count: ${filterData.count || 0}`);
      if (filterData.staff && filterData.staff.length > 0) {
        filterData.staff.forEach(s => {
          console.log(`    - ${s.firstName} ${s.lastName} (${s.systemRole})`);
        });
      }
    }

  } catch (err) {
    console.error('❌ ERROR:', err.message);
  }
}

testStaffList();
