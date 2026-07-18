/**
 * Hospital Population Script (Full Integration)
 * 
 * This script:
 * 1. Registers a new hospital with admin user
 * 2. Reads physicians_expanded.csv and registers ALL physicians (33+ specialties)
 * 3. Creates paired nurses for each physician
 * 4. Creates shifts for today and assigns physicians + nurses
 * 5. Outputs summary with credentials for testing
 * 
 * SUPPORTED SPECIALTIES (33+):
 *   Surgical: Orthopedics, Urology, ENT, General Surgery, Cardiothoracic Surgery,
 *            Plastic Surgery, Pediatric Surgery, Neurosurgery, Thoracic Surgery,
 *            Vascular Surgery, Trauma Surgery
 *   Medical: Pulmonology, Gastroenterology, Cardiology, Neurology, Nephrology,
 *           Rheumatology, Infectious Disease, Endocrinology, Geriatrics
 *   Diagnostic: Radiology, Interventional Radiology, Pathology, Toxicology
 *   Specialized: Pediatrics, OBGYN, Psychiatry, Oncology, Dermatology, Optometrist
 *   Critical: Critical Care, Emergency Medicine, General Medicine
 * 
 * Usage:
 *   node populate-hospital-full.js
 * 
 * Environment:
 *   VITE_BACKEND_URL or defaults to http://localhost:5001
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { parse as csvParse } from 'csv-parse/sync';

// ============ CONFIGURATION ============
const BACKEND_URL = process.env.VITE_BACKEND_URL || 'http://localhost:5001';
const CSV_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), 'ML', 'physicians_expanded.csv');

const HOSPITAL_NAME = 'Nurse Helpdesk Central Hospital';
const ADMIN_EMAIL = `admin-${Date.now()}@nursehelpdesk.local`;
const ADMIN_PASSWORD = 'AdminPassword123!';
const ADMIN_NAME = 'Administrator';
const ADMIN_SURNAME = 'System';

// Shift configuration
const SHIFT_START = '08:00';
const SHIFT_END = '16:00'; // 8-hour day shift
const MAX_STAFF_PER_SHIFT = 5;

// ============ LOGGING ============
const log = (msg) => console.log(`[POPULATE] ${msg}`);
const success = (msg) => console.log(`✓ ${msg}`);
const error = (msg) => console.error(`✗ ${msg}`);
const section = (title) => console.log(`\n${'='.repeat(60)}\n${title}\n${'='.repeat(60)}`);

// ============ HELPER FUNCTIONS ============

/**
 * Generate unique name with timestamp
 */
function generateUniqueName(prefix = '') {
  // Use timestamp + random to make names unique per run
  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substr(2, 5).toUpperCase();
  return `${prefix}${timestamp}${randomSuffix}`;
}

/**
 * Extract specialty abbreviation for email encoding
 */
function getSpecialtyAbbr(specialty) {
  const abbrevs = {
    'Orthopedics': 'ORT',
    'Urology': 'URO',
    'ENT': 'ENT',
    'General Surgery': 'GEN',
    'Cardiothoracic Surgery': 'CTS',
    'Plastic Surgery': 'PLS',
    'Pediatric Surgery': 'PED',
    'Neurosurgery': 'NSG',
    'Thoracic Surgery': 'THR',
    'Vascular Surgery': 'VAS',
    'Trauma Surgery': 'TRM',
    'Pulmonology': 'PUL',
    'Gastroenterology': 'GAS',
    'Cardiology': 'CAR',
    'Neurology': 'NEU',
    'Nephrology': 'NEP',
    'Rheumatology': 'RHE',
    'Infectious Disease': 'INF',
    'Endocrinology': 'END',
    'GeneralMedicine': 'GEM',
    'Geriatrics': 'GER',
    'Radiology': 'RAD',
    'Interventional Radiology': 'IRD',
    'Pediatrics': 'PED',
    'OBGYN': 'OBG',
    'Psychiatry': 'PSY',
    'Oncology': 'ONC',
    'Pathology': 'PAT',
    'Toxicology': 'TOX',
    'Dermatology': 'DER',
    'Optometrist': 'OPT',
    'Critical Care': 'CRC',
    'Emergency Medicine': 'EM'
  };
  return abbrevs[specialty] || specialty.substr(0, 3).toUpperCase();
}

/**
 * Make HTTP request
 */
async function makeRequest(method, endpoint, body = null, token = null) {
  try {
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
      }
    };

    if (token) {
      options.headers.Authorization = `Bearer ${token}`;
    }

    if (body) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(`${BACKEND_URL}${endpoint}`, options);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${data.message || data.error || 'Unknown error'}`);
    }

    return data;
  } catch (err) {
    throw new Error(`Request failed: ${err.message}`);
  }
}

/**
 * Register hospital and admin user
 */
async function registerHospital() {
  section('STEP 1: Register Hospital & Admin');
  
  log(`Registering hospital: "${HOSPITAL_NAME}"`);
  log(`Admin Email: ${ADMIN_EMAIL}`);

  try {
    const response = await makeRequest('POST', '/api/hospital/register', {
      hospitalName: HOSPITAL_NAME,
      managerName: ADMIN_NAME,
      managerSurname: ADMIN_SURNAME,
      managerEmail: ADMIN_EMAIL,
      managerPassword: ADMIN_PASSWORD,
      department: 'All'
    });

    if (!response.success) {
      throw new Error(response.message || 'Hospital registration failed');
    }

    success(`Hospital registered: ${response.hospital.name}`);
    success(`Hospital ID: ${response.hospital.hospitalID}`);
    success(`Admin: ${response.user.name} (${response.user.email})`);
    success(`Mongo Hospital ID: ${response.user.hospitalId}`);
    log(`Token includes hospitalId: ${response.user.hospitalId}\n`);

    return {
      hospId: response.hospital.hospitalID,
      token: response.token,
      adminId: response.user.id,
      hospitalMongoId: response.user.hospitalId
    };
  } catch (err) {
    error(`Hospital registration: ${err.message}`);
    throw err;
  }
}

/**
 * Read and parse physicians CSV
 */
function readPhysiciansCSV() {
  section('STEP 2: Load Physicians from CSV');

  try {
    if (!fs.existsSync(CSV_PATH)) {
      throw new Error(`CSV file not found: ${CSV_PATH}`);
    }

    const fileContent = fs.readFileSync(CSV_PATH, 'utf-8');
    const records = csvParse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      delimiter: ',',
      trim: true
    });

    // Filter out header row if it appears as data
    const filtered = records.filter(r => {
      const id = r.id || r.physician_id;
      return id && !isNaN(parseInt(id));
    });

    log(`Found ${filtered.length} physicians in CSV`);

    // Add line break after count
    console.log('');

    // Group by specialty for summary
    const specialtyGroups = {};
    filtered.forEach(r => {
      if (!specialtyGroups[r.specialty]) specialtyGroups[r.specialty] = 0;
      specialtyGroups[r.specialty]++;
    });
    
    log('📊 Specialty breakdown:');
    Object.entries(specialtyGroups).sort((a, b) => b[1] - a[1]).forEach(([spec, count]) => {
      console.log(`  • ${spec}: ${count} physician(s)`);
    });
    console.log('');

    return filtered.map((record, idx) => {
      // Parse availability_mask from CSV string like "[0, 1, 0, ...]"
      let availabilityMask = [];
      try {
        // Remove brackets and parse as array
        const maskStr = record.availability_mask
          .replace(/[\[\]]/g, '') // Remove brackets
          .split(',')
          .map(x => parseInt(x.trim()));
        availabilityMask = maskStr;
      } catch (e) {
        // Default to all available if parsing fails
        availabilityMask = Array(24).fill(1);
      }

      return {
        physicianId: (record.id || record.physician_id)?.trim(),
        physicianName: (record.name || record.physician_name)?.trim() || `Dr_${record.id || record.physician_id}`,
        specialty: record.specialty?.trim() || 'General Medicine',
        availabilityMask,
        workloadScore: parseFloat(record.workload_score) || 0.5,
        csvIndex: idx
      };
    });
  } catch (err) {
    error(`Failed to read CSV: ${err.message}`);
    throw err;
  }
}

/**
 * Generate nurse info paired with physician
 */
function generatePairedNurse(physician) {
  const nurseName = physician.physicianName.replace('Dr_', 'Nurse_');
  const [firstName, ...rest] = nurseName.split('_');
  const lastName = rest.join('_');
  
  return {
    nurseName,
    firstName,
    lastName,
    specialty: physician.specialty,
    department: mapSpecialtyToDepartment(physician.specialty)
  };
}

/**
 * Map medical specialty to hospital department
 * EXTENSIVE mapping covering all 33+ specialties
 */
function mapSpecialtyToDepartment(specialty) {
  const mapping = {
    // Surgical specialties → Operating Room / Surgical Floors
    'Orthopedics': 'OR',
    'Urology': 'OR',
    'ENT': 'OR',
    'General Surgery': 'OR',
    'Cardiothoracic Surgery': 'OR',
    'Plastic Surgery': 'OR',
    'Pediatric Surgery': 'OR',
    'Neurosurgery': 'OR',
    'Thoracic Surgery': 'OR',
    'Vascular Surgery': 'OR',
    'Trauma Surgery': 'OR',
    
    // Medical specialties → Medical-Surgical floors
    'Pulmonology': 'Medical-Surgical',
    'Gastroenterology': 'Medical-Surgical',
    'Cardiology': 'Medical-Surgical',
    'Neurology': 'Medical-Surgical',
    'Nephrology': 'Medical-Surgical',
    'Rheumatology': 'Medical-Surgical',
    'Infectious Disease': 'Medical-Surgical',
    'Endocrinology': 'Medical-Surgical',
    'GeneralMedicine': 'Medical-Surgical',
    'Geriatrics': 'Medical-Surgical',
    'Radiology': 'Medical-Surgical',
    'Interventional Radiology': 'Medical-Surgical',
    
    // Pediatrics department
    'Pediatrics': 'Pediatrics',
    
    // OB/GYN department
    'OBGYN': 'OB',
    
    // Mental Health
    'Psychiatry': 'Psychiatry',
    
    // Oncology - dedicated cancer center
    'Oncology': 'Medical-Surgical',
    
    // Lab/Diagnostic
    'Pathology': 'Medical-Surgical',
    'Toxicology': 'Medical-Surgical',
    
    // Specialized clinics
    'Dermatology': 'ED',
    'Optometrist': 'ED',
    
    // Critical care units
    'Critical Care': 'ICU',
    'Emergency Medicine': 'ED',
    
    // Fallback
    'specialty': 'ED' // header row
  };
  return mapping[specialty] || 'Medical-Surgical';
}

/**
 * Register a physician
 */
async function registerPhysician(physician, token) {
  try {
    // Email format: firstname_PHYS_specialty@hospital.local
    const specialtyAbbr = getSpecialtyAbbr(physician.specialty);
    const uniqueName = generateUniqueName('Dr');
    const email = `${uniqueName}_PHYS_${specialtyAbbr}@nursehelpdeskcentralhospital.local`.toLowerCase();
    const password = `PhysPassword_${uniqueName}!`;

    log(`  📤 Registering: ${email} (${physician.specialty})`);

    const response = await makeRequest('POST', '/api/manager/staff/register', {
      firstName: uniqueName,
      lastName: `${physician.specialty}`,
      email,
      password,
      role: 'PHYSICIAN',
      department: mapSpecialtyToDepartment(physician.specialty),
      specialty: physician.specialty
    }, token);

    if (!response.success) {
      throw new Error(response.message || 'Staff registration failed');
    }

    success(`  ✓ Registered with staff ID: ${response.staff.id} (Hospital: ${response.staff.hospitalId})`);

    return {
      staffId: response.staff.id,
      email,
      password,
      specialty: physician.specialty,
      role: 'PHYSICIAN',
      uniqueName
    };
  } catch (err) {
    error(`Physician registration (${physician.physicianName}): ${err.message}`);
    throw err;
  }
}

/**
 * Register a nurse
 */
async function registerNurse(specialty, department, token) {
  try {
    // Email format: firstname_NURSE_specialty@hospital.local
    const specialtyAbbr = getSpecialtyAbbr(specialty);
    const uniqueName = generateUniqueName('Nurse');
    const email = `${uniqueName}_NURSE_${specialtyAbbr}@nursehelpdeskcentralhospital.local`.toLowerCase();
    const password = `NursePassword_${uniqueName}!`;

    const response = await makeRequest('POST', '/api/manager/staff/register', {
      firstName: uniqueName,
      lastName: 'Nursing',
      email,
      password,
      role: 'NURSE',
      department,
      specialty: 'Nursing'
    }, token);

    if (!response.success) {
      throw new Error(response.message || 'Staff registration failed');
    }

    return {
      staffId: response.staff._id,
      email,
      password,
      role: 'NURSE',
      uniqueName,
      specialty,
      department
    };
  } catch (err) {
    error(`Nurse registration: ${err.message}`);
    throw err;
  }
}

/**
 * Create shift with assigned staff
 */
async function createShift(shiftData, token) {
  try {
    const response = await makeRequest('POST', '/api/shifts/create', shiftData, token);

    if (!response.success) {
      throw new Error(response.message || 'Shift creation failed');
    }

    return response.shift;
  } catch (err) {
    error(`Shift creation: ${err.message}`);
    throw err;
  }
}

/**
 * Get today's date in YYYY-MM-DD format
 */
function getTodayDateString() {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

/**
 * Main execution
 */
async function main() {
  try {
    section('NURSE HELPDESK HOSPITAL POPULATION SCRIPT');
    log('Backend URL: ' + BACKEND_URL);
    log('CSV Path: ' + CSV_PATH);
    log(`Shift Time: ${SHIFT_START} - ${SHIFT_END}`);
    console.log('');

    // Step 1: Register hospital
    const hospital = await registerHospital();
    const { hospId, token, adminId } = hospital;

    // Step 2: Load physicians from CSV
    const physicians = readPhysiciansCSV();

    // Step 3: Register physicians and nurses
    section('STEP 3: Register Physicians and Paired Nurses');
    
    const registeredStaff = {
      physicians: [],
      nurses: []
    };

    for (const physician of physicians) {
      // Register physician
      log(`Registering physician: ${physician.physicianName} (${physician.specialty})`);
      const physData = await registerPhysician(physician, token);
      registeredStaff.physicians.push({
        ...physData,
        physicianId: physician.physicianId
      });
      success(`Physician registered: ${physData.email}`);

      // Register paired nurse (one per physician)
      log(`  ↳ Registering paired nurse for ${physician.specialty}`);
      const nurseData = await registerNurse(
        physician.specialty,
        mapSpecialtyToDepartment(physician.specialty),
        token
      );
      registeredStaff.nurses.push(nurseData);
      success(`  ↳ Nurse registered: ${nurseData.email}`);
    }

    // Step 4: Create SINGLE unified shift and assign all staff
    section('STEP 4: Create Unified Shift & Assign All Staff');

    const todayDate = getTodayDateString();
    log(`Creating unified shift for: ${todayDate}`);
    log(`All physicians and nurses assigned to: ${SHIFT_START} - ${SHIFT_END}`);
    console.log('');

    // Assign ALL physicians + ALL nurses to the same shift
    const assignedStaff = [];

    // Add all physicians
    registeredStaff.physicians.forEach(phys => {
      assignedStaff.push({
        staffID: phys.staffId,
        type: 'PHYSICIAN',
        name: phys.uniqueName,
        surname: phys.specialty,
        specialty: phys.specialty,
        status: 'ASSIGNED'
      });
    });

    // Add all nurses
    registeredStaff.nurses.forEach(nurse => {
      assignedStaff.push({
        staffID: nurse.staffId,
        type: 'NURSE',
        name: nurse.uniqueName,
        surname: 'Nursing',
        specialty: 'Nursing',
        status: 'ASSIGNED'
      });
    });

    const unifiedShiftData = {
      shiftName: `Daily Shift - All Staff (${todayDate})`,
      department: 'ED', // Multi-department shift
      shiftStart: SHIFT_START,
      shiftEnd: SHIFT_END,
      date: todayDate,
      maxStaffRequired: assignedStaff.length,
      assignedStaff: assignedStaff
    };

    log(`Assigning ${registeredStaff.physicians.length} physicians + ${registeredStaff.nurses.length} nurses to unified shift`);
    const unifiedShift = await createShift(unifiedShiftData, token);
    const createdShifts = [{
      shiftId: unifiedShift.shiftID,
      shiftName: unifiedShift.shiftName,
      staffCount: assignedStaff.length
    }];

    success(`✓ Unified shift created with ${assignedStaff.length} staff members`);
    console.log('');

    // Step 5: Summary and credentials
    section('STEP 5: SUMMARY & CREDENTIALS');

    console.log('\n📋 HOSPITAL DETAILS:');
    console.log(`  Hospital Name: ${HOSPITAL_NAME}`);
    console.log(`  Hospital ID: ${hospId}`);
    console.log(`  Date Created: ${new Date().toISOString()}`);

    console.log('\n👤 ADMIN CREDENTIALS:');
    console.log(`  Email: ${ADMIN_EMAIL}`);
    console.log(`  Password: ${ADMIN_PASSWORD}`);
    console.log(`  Role: ADMIN`);

    console.log(`\n👨‍⚕️ REGISTERED PHYSICIANS: ${registeredStaff.physicians.length}`);
    
    // Group by specialty
    const physiciansBySpecialty = {};
    registeredStaff.physicians.forEach(p => {
      const specialty = p.specialty || 'Unknown';
      if (!physiciansBySpecialty[specialty]) {
        physiciansBySpecialty[specialty] = [];
      }
      physiciansBySpecialty[specialty].push(p);
    });
    
    log('Physicians by specialty:');
    Object.entries(physiciansBySpecialty).sort().forEach(([spec, physs]) => {
      console.log(`  • ${spec}: ${physs.length} physician(s)`);
      if (physs.length <= 2) {
        physs.forEach(p => console.log(`      - ${p.physicianName} (${p.email})`));
      } else {
        physs.slice(0, 2).forEach(p => console.log(`      - ${p.physicianName}`));
        console.log(`      ... and ${physs.length - 2} more`);
      }
    });

    console.log(`\n👩‍⚕️ REGISTERED NURSES: ${registeredStaff.nurses.length}`);
    console.log('  Sample credentials (first 5):');
    registeredStaff.nurses.slice(0, 5).forEach(nurse => {
      console.log(`    • ${nurse.nurseName}`);
      console.log(`      Email: ${nurse.email}`);
    });
    if (registeredStaff.nurses.length > 5) {
      console.log(`    ... and ${registeredStaff.nurses.length - 5} more`);
    }

    console.log(`\n📅 CREATED SHIFTS: ${createdShifts.length}`);
    createdShifts.forEach((shift, idx) => {
      console.log(`  ${idx + 1}. ${shift.shiftName}`);
      console.log(`     Shift ID: ${shift.shiftId}`);
      console.log(`     Staff Assigned: ${shift.staffCount}`);
    });

    console.log('\n' + '='.repeat(60));
    console.log('✓ HOSPITAL POPULATION COMPLETE');
    console.log('='.repeat(60));

    console.log('\n🔗 TESTING URLs:');
    console.log(`  Frontend: http://localhost:5173`);
    console.log(`  Backend: ${BACKEND_URL}`);
    console.log(`  Login with Admin: ${ADMIN_EMAIL} / ${ADMIN_PASSWORD}`);

    console.log('\n💾 Full credentials saved to: populate-hospital-credentials.json');

    // Save full credentials to file
    const outputPath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'populate-hospital-credentials.json');
    fs.writeFileSync(outputPath, JSON.stringify({
      hospital: { name: HOSPITAL_NAME, id: hospId },
      admin: { email: ADMIN_EMAIL, password: ADMIN_PASSWORD },
      physicians: registeredStaff.physicians,
      nurses: registeredStaff.nurses,
      shifts: createdShifts,
      createdAt: new Date().toISOString()
    }, null, 2));

    success(`Credentials file created: ${outputPath}`);

  } catch (err) {
    error(`Script execution failed: ${err.message}`);
    process.exit(1);
  }
}

// Run main
main().catch(err => {
  error(`Unhandled error: ${err.message}`);
  process.exit(1);
});
