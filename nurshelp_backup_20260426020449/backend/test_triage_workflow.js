/**
 * Diagnostic Script: Test Manual Triage Workflow End-to-End
 * 
 * This script will:
 * 1. Log in as a nurse
 * 2. Submit a manual triage
 * 3. Check if encounter appears in the waiting queue
 * 4. Diagnose any issues in the data flow
 */

import fetch from 'node-fetch';
import jwt from 'jsonwebtoken';

const BACKEND_URL = 'http://localhost:5001';

async function main() {
  try {
    console.log('='.repeat(60));
    console.log('MANUAL TRIAGE WORKFLOW DIAGNOSTIC');
    console.log('='.repeat(60));

    // Step 1: Login
    console.log('\n[STEP 1] Logging in...');
    const loginRes = await fetch(`${BACKEND_URL}/api/nurses/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: 'nurse@test.com',
        password: 'nurse123'
      })
    });

    if (!loginRes.ok) {
      console.error('❌ Login failed:', loginRes.status);
      const data = await loginRes.json();
      console.error('Response:', data);
      return;
    }

    const loginData = await loginRes.json();
    const token = loginData.token;
    console.log('✅ Login successful');
    console.log(`   Token: ${token.substring(0, 20)}...`);

    // Decode token to see nurse ID
    const decoded = jwt.decode(token);
    console.log(`   Decoded JWT:`, decoded);
    console.log(`   Nurse ID in JWT: ${decoded.id}`);

    // Step 2: Submit manual triage
    console.log('\n[STEP 2] Submitting manual triage...');
    const triageRes = await fetch(`${BACKEND_URL}/api/triage`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        triageNurseId: decoded.id,
        patientId: 'PT-001',
        symptoms: ['fever', 'cough'],
        vitalSigns: {
          temperature: 38.5,
          heartRate: 100,
          respiratoryRate: 20,
          systolicBP: 130,
          diastolicBP: 85,
          oxygenSaturation: 95
        },
        manualSatsLevel: 'yellow',
        urgencyScore: 50,
        triageReasoningNotes: 'Test triage - fever and cough, moderate concern',
        recommendedDepartment: 'ED',
        immediateInterventions: ['Chest X-ray', 'Blood work'],
        assignedNurseId: decoded.id,
        assignedPhysicianId: 'phys-001',
        bedAssigned: null,
        patient: {
          name: 'Test Patient',
          surname: 'Diagnostic',
          id_number: 'PT-001'
        },
        vitals: {
          temp: 38.5,
          hr: 100,
          resp: 20,
          bp_systolic: 130,
          bp_diastolic: 85,
          o2: 95
        }
      })
    });

    if (!triageRes.ok) {
      console.error('❌ Triage submission failed:', triageRes.status);
      const data = await triageRes.json();
      console.error('Response:', data);
      return;
    }

    const triageData = await triageRes.json();
    console.log('✅ Triage submitted successfully');
    console.log(`   Response:`, JSON.stringify(triageData, null, 2));
    
    const encounterId = triageData.data?.triageRequestId || triageData.triageRequestId;
    console.log(`   Created Encounter ID: ${encounterId}`);

    // Step 3: Check waiting queue
    console.log('\n[STEP 3] Checking waiting encounters queue...');
    const queueRes = await fetch(`${BACKEND_URL}/api/encounters/waiting`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!queueRes.ok) {
      console.error('❌ Queue fetch failed:', queueRes.status);
      const data = await queueRes.json();
      console.error('Response:', data);
      return;
    }

    const queueData = await queueRes.json();
    console.log(`✅ Queue fetch successful`);
    console.log(`   Total encounters in queue: ${Array.isArray(queueData) ? queueData.length : queueData.length || 0}`);
    
    // Check if our encounter is in the queue
    const encounters = Array.isArray(queueData) ? queueData : queueData.encounters || [];
    const foundEncounter = encounters.find(e => e._id === encounterId);
    
    if (foundEncounter) {
      console.log('✅ ENCOUNTER FOUND IN QUEUE!');
      console.log(`   Encounter:`, JSON.stringify(foundEncounter, null, 2));
    } else {
      console.log('❌ ENCOUNTER NOT FOUND IN QUEUE');
      console.log(`   All encounters in queue:`, JSON.stringify(encounters.map(e => ({
        id: e._id,
        patient: e.patient?.name,
        nurse_id: e.nurse_id,
        isWaiting: e.isWaiting,
        status: e.status
      })), null, 2));
    }

    // Step 4: Fetch all encounters for this nurse
    console.log('\n[STEP 4] Checking all encounters for nurse...');
    const allRes = await fetch(`${BACKEND_URL}/api/encounters`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!allRes.ok) {
      console.error('❌ All encounters fetch failed:', allRes.status);
      return;
    }

    const allData = await allRes.json();
    console.log(`✅ All encounters fetch successful`);
    const allEncounters = Array.isArray(allData) ? allData : allData.encounters || [];
    console.log(`   Total encounters for nurse: ${allEncounters.length}`);
    
    const foundInAll = allEncounters.find(e => e._id === encounterId);
    if (foundInAll) {
      console.log('✅ ENCOUNTER FOUND IN ALL ENCOUNTERS');
    } else {
      console.log('❌ ENCOUNTER NOT FOUND IN ALL ENCOUNTERS');
    }

    console.log('\n' + '='.repeat(60));
    console.log('DIAGNOSTIC SUMMARY:');
    console.log('='.repeat(60));
    console.log(`✅ Login: SUCCESS`);
    console.log(`✅ Triage Submission: SUCCESS (ID: ${encounterId})`);
    console.log(`${foundEncounter ? '✅' : '❌'} Encounter in Waiting Queue: ${foundEncounter ? 'YES' : 'NO'}`);
    console.log(`${foundInAll ? '✅' : '❌'} Encounter in All Encounters: ${foundInAll ? 'YES' : 'NO'}`);

    if (!foundEncounter) {
      console.log('\n⚠️  ISSUE DETECTED:');
      console.log('   Encounter was created but not appearing in waiting queue.');
      console.log('   Possible causes:');
      console.log('   1. Nurse ID mismatch between creation and query');
      console.log('   2. isWaiting flag not set to true');
      console.log('   3. Status not set to "pending"');
      console.log('   4. Authentication issue in queue endpoint');
    }

  } catch (err) {
    console.error('❌ ERROR:', err.message);
    console.error(err.stack);
  }
}

main();
