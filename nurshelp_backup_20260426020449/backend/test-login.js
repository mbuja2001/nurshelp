import fetch from 'node-fetch';

const BACKEND_URL = 'http://localhost:5001';

async function testLogin() {
  try {
    console.log('🔐 Testing login for dr.chen@stmercy.hospital\n');

    const loginRes = await fetch(`${BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: 'dr.chen@stmercy.hospital',
        password: 'doctor123'
      })
    });

    console.log(`📤 Response Status: ${loginRes.status}`);

    const loginData = await loginRes.json();
    console.log('📥 Response Body:', JSON.stringify(loginData, null, 2));

    if (loginData.token) {
      console.log('\n✅ LOGIN SUCCESSFUL!');
      console.log(`Token: ${loginData.token.substring(0, 50)}...`);
      console.log(`User: ${loginData.user?.name || 'N/A'}`);
      console.log(`Role: ${loginData.user?.role || 'N/A'}`);
      console.log(`Hospital: ${loginData.user?.hospitalName || 'N/A'}`);
      
      // Test a protected endpoint with the token
      console.log('\n🔍 Testing protected endpoint with token...');
      const shiftRes = await fetch(`${BACKEND_URL}/api/shifts/assigned`, {
        headers: { Authorization: `Bearer ${loginData.token}` }
      });
      console.log(`   GET /api/shifts/assigned Status: ${shiftRes.status}`);
      const shiftData = await shiftRes.json();
      console.log(`   Response:`, JSON.stringify(shiftData, null, 2).substring(0, 200) + '...');
    } else {
      console.log('\n❌ LOGIN FAILED');
      console.log(`Error: ${loginData.message}`);
    }
  } catch (err) {
    console.error('❌ ERROR:', err.message);
  }
}

testLogin();
