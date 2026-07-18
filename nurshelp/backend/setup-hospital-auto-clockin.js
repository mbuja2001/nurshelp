/**
 * COMPLETE HOSPITAL SETUP WITH AUTO CLOCK-IN
 * 
 * Creates a full hospital infrastructure from scratch:
 * - Hospital registration with admin account
 * - Diverse staff across all departments (Physicians, Nurses, Radiologists, Pathologists, Pharmacists)
 * - Beds with equipment specifications
 * - Pharmaceutical inventory
 * - 3 shifts (morning, evening, night)
 * - Staff assigned to shifts with hospitalId
 * - Auto clock-in ALL staff for immediate testing
 * 
 * Enables:
 * - Triage of clocked-in staff
 * - Lab orders to radiologists/pathologists
 * - Medication inventory tracking
 * - Complete workflow testing
 * 
 * Usage: node setup-hospital-auto-clockin.js "Hospital Name"
 */

import fetch from 'node-fetch';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '.env') });

const API_BASE = process.env.API_BASE || 'http://localhost:5001/api';
const MONGO_URI = process.env.MONGO_URI;

// ============ LOGGING ============
const log = (msg) => console.log(`[SETUP] ${msg}`);
const success = (msg) => console.log(`✓ ${msg}`);
const error = (msg) => console.error(`✗ ${msg}`);
const section = (title) => console.log(`\n${'='.repeat(70)}\n${title}\n${'='.repeat(70)}`);

// ============ STAFF CATEGORIES ============
const PHYSICIANS = [
  // Critical Care - ICU
  { name: 'Dr. Sarah Chen', specialty: 'Internal Medicine', department: 'Medical Ward' },
  { name: 'Dr. Raj Patel', specialty: 'General Surgery', department: 'Surgical Ward' },
  { name: 'Dr. Maria Garcia', specialty: 'Emergency Medicine', department: 'ICU' },
  { name: 'Dr. James Wilson', specialty: 'Pediatrics', department: 'Pediatric Ward' },
  { name: 'Dr. Lisa Anderson', specialty: 'Cardiology', department: 'CICU' },
  { name: 'Dr. Robert Brown', specialty: 'Radiology', department: 'Radiology' },
  // Critical Care - NICU, PICU, SICU, Neuro ICU
  { name: 'Dr. Emily Davis', specialty: 'Neonatal Medicine', department: 'NICU' },
  { name: 'Dr. Michael Torres', specialty: 'Pediatric ICU', department: 'PICU' },
  { name: 'Dr. Jennifer Lee', specialty: 'Cardiothoracic Surgery', department: 'SICU' },
  { name: 'Dr. David Kumar', specialty: 'Neurosurgery', department: 'Neuro ICU' },
  // Step-Down Care - HDU, PCU
  { name: 'Dr. Amanda Wright', specialty: 'Pulmonology', department: 'HDU' },
  { name: 'Dr. Christopher Martinez', specialty: 'Nephrology', department: 'PCU' },
  // Surgical & Specialty Wards
  { name: 'Dr. Patricia Johnson', specialty: 'Orthopedics', department: 'Orthopaedic Ward' },
  { name: 'Dr. Daniel O\'Brien', specialty: 'Gastroenterology', department: 'Gastroenterology Ward' },
  { name: 'Dr. Susan Blake', specialty: 'Neurology', department: 'Neurology Ward' },
  { name: 'Dr. George Harris', specialty: 'Oncology', department: 'Oncology Ward' },
  { name: 'Dr. Diana Turner', specialty: 'Infectious Disease', department: 'Infectious Disease Ward' },
  { name: 'Dr. Paul Thompson', specialty: 'Geriatrics', department: 'Geriatric Ward' },
  { name: 'Dr. Helen White', specialty: 'OBGYN', department: 'Maternity Ward' },
  { name: 'Dr. Kevin Zhang', specialty: 'Psychiatry', department: 'Psychiatric Ward' },
  // Pathology & Support
  { name: 'Dr. Nancy Garcia', specialty: 'Clinical Pathology', department: 'Pathology' },
];

const NURSES = [
  // Emergency & Critical Care - ICU
  { name: 'Nurse Emma Thompson', specialty: 'Emergency Nursing', department: 'ICU', role: 'TRIAGE_NURSE' },
  { name: 'Nurse Michael Chen', specialty: 'Emergency Nursing', department: 'ICU', role: 'TRIAGE_NURSE' },
  { name: 'Nurse Amanda Foster', specialty: 'Critical Care Nursing', department: 'ICU', role: 'CLINICAL_NURSE' },
  // NICU, PICU, SICU, Neuro ICU
  { name: 'Nurse Jessica Lee', specialty: 'Neonatal Nursing', department: 'NICU', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Thomas Martinez', specialty: 'Pediatric Nursing', department: 'PICU', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Sophie Robinson', specialty: 'ICU Nursing', department: 'SICU', role: 'CLINICAL_NURSE' },
  { name: 'Nurse David Kumar', specialty: 'Neurosurgical Nursing', department: 'Neuro ICU', role: 'CLINICAL_NURSE' },
  // CICU - Cardiac
  { name: 'Nurse Helen White', specialty: 'Cardiac Nursing', department: 'CICU', role: 'CLINICAL_NURSE' },
  // Step-Down Care
  { name: 'Nurse Oliver Green', specialty: 'General Nursing', department: 'HDU', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Victoria Brown', specialty: 'General Nursing', department: 'PCU', role: 'CLINICAL_NURSE' },
  // Medical Ward
  { name: 'Nurse Rachel Murphy', specialty: 'General Nursing', department: 'Medical Ward', role: 'CLINICAL_NURSE' },
  // Surgical Ward
  { name: 'Nurse Marcus Johnson', specialty: 'Operating Room Nursing', department: 'Surgical Ward', role: 'CLINICAL_NURSE' },
  // Specialty Wards
  { name: 'Nurse Carla Smith', specialty: 'Orthopedic Nursing', department: 'Orthopaedic Ward', role: 'CLINICAL_NURSE' },
  { name: 'Nurse James Wilson', specialty: 'General Nursing', department: 'Gastroenterology Ward', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Patricia Davis', specialty: 'General Nursing', department: 'Neurology Ward', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Robert Miller', specialty: 'Oncology Nursing', department: 'Oncology Ward', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Linda Anderson', specialty: 'Infection Control Nursing', department: 'Infectious Disease Ward', role: 'CLINICAL_NURSE' },
  { name: 'Nurse George Taylor', specialty: 'Geriatric Nursing', department: 'Geriatric Ward', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Susan Harris', specialty: 'Maternity Nursing', department: 'Maternity Ward', role: 'CLINICAL_NURSE' },
  { name: 'Nurse Frank Jackson', specialty: 'Psychiatric Nursing', department: 'Psychiatric Ward', role: 'CLINICAL_NURSE' },
];

const RADIOLOGISTS = [
  { name: 'Dr. Peter Mitchell', specialty: 'X-Ray Technician', department: 'Radiology' },
  { name: 'Dr. Andrew Morris', specialty: 'CT Technician', department: 'Radiology' },
  { name: 'Dr. Jennifer Garcia', specialty: 'MRI Technician', department: 'Radiology' },
  { name: 'Dr. Marcus White', specialty: 'Ultrasound Technician', department: 'Radiology' },
];

const PATHOLOGISTS = [
  { name: 'Dr. Leonard Roberts', specialty: 'Lab Technician', department: 'Pathology' },
  { name: 'Dr. Victoria King', specialty: 'Hematology Technician', department: 'Pathology' },
];

const PHARMACISTS = [
  { name: 'Pharmacist Mark Johnson', specialty: 'Clinical Pharmacist', department: 'Pharmacy', role: 'PHARMACIST' },
  { name: 'Pharmacist Nancy Davis', specialty: 'Hospital Pharmacist', department: 'Pharmacy', role: 'PHARMACIST' },
];

// ============ DEPARTMENT MAPPING (From assign-shifts-with-teams.js) ============
const DEPARTMENTS = {
  // Critical Care Wards (High Acuity)
  'ICU': 'ICU',
  'NICU': 'NICU',
  'PICU': 'PICU',
  'CICU': 'CICU',
  'CCU': 'CICU',
  'SICU': 'SICU',
  'Neuro ICU': 'Neuro ICU',
  'Critical Care': 'ICU',
  
  // Step-Down & Progressive Care
  'HDU': 'HDU',
  'High Dependency Unit': 'HDU',
  'PCU': 'PCU',
  'Progressive Care Unit': 'PCU',
  
  // Medical and Surgical Wards
  'Medical Ward': 'Medical Ward',
  'Surgical Ward': 'Surgical Ward',
  'Orthopaedic Ward': 'Orthopaedic Ward',
  'Oncology Ward': 'Oncology Ward',
  'Neurology Ward': 'Neurology Ward',
  'Gastroenterology Ward': 'Gastroenterology Ward',
  
  // Demographic & Specialty Wards
  'Pediatric Ward': 'Pediatric Ward',
  'Geriatric Ward': 'Geriatric Ward',
  'Maternity Ward': 'Maternity Ward',
  'Psychiatric Ward': 'Psychiatric Ward',
  'Infectious Disease Ward': 'Infectious Disease Ward',
  
  // Support Departments
  'Radiology': 'Radiology',
  'Pathology': 'Pathology',
  'Pharmacy': 'Pharmacy',
};

// ============ INFRASTRUCTURE DATA ============
const BEDS = [
  // ICU Beds
  { bed_id: 'ICU-A-1', ward_unit: 'ICU', equipment_baseline: ['High-Acuity Monitor', 'Mechanical Ventilator', 'Defibrillator'] },
  { bed_id: 'ICU-A-2', ward_unit: 'ICU', equipment_baseline: ['High-Acuity Monitor', 'Mechanical Ventilator'] },
  { bed_id: 'ICU-B-1', ward_unit: 'ICU', equipment_baseline: ['High-Acuity Monitor', 'Defibrillator'] },
  { bed_id: 'ICU-B-2', ward_unit: 'ICU', equipment_baseline: ['High-Acuity Monitor'] },
  
  // CICU (Cardiac ICU)
  { bed_id: 'CICU-A-1', ward_unit: 'CICU', equipment_baseline: ['Cardiac Monitor', 'Defibrillator', 'Infusion Pump'] },
  { bed_id: 'CICU-A-2', ward_unit: 'CICU', equipment_baseline: ['Cardiac Monitor', 'Defibrillator'] },
  
  // Medical Ward
  { bed_id: 'MED-A-1', ward_unit: 'Medical Ward', equipment_baseline: ['High-Acuity Monitor', 'Oxygen Delivery'] },
  { bed_id: 'MED-A-2', ward_unit: 'Medical Ward', equipment_baseline: ['High-Acuity Monitor'] },
  { bed_id: 'MED-B-1', ward_unit: 'Medical Ward', equipment_baseline: ['High-Acuity Monitor'] },
  
  // Surgical Ward
  { bed_id: 'SURG-A-1', ward_unit: 'Surgical Ward', equipment_baseline: ['High-Acuity Monitor', 'Mechanical Ventilator'] },
  { bed_id: 'SURG-A-2', ward_unit: 'Surgical Ward', equipment_baseline: ['High-Acuity Monitor'] },
  
  // Pediatric Ward
  { bed_id: 'PED-A-1', ward_unit: 'Pediatric Ward', equipment_baseline: ['High-Acuity Monitor', 'Oxygen Delivery'] },
  { bed_id: 'PED-A-2', ward_unit: 'Pediatric Ward', equipment_baseline: ['High-Acuity Monitor'] },
  
  // Radiology Department (imaging stations)
  { bed_id: 'RAD-CT-1', ward_unit: 'Radiology', equipment_baseline: ['Portable Ultrasound'] },
  { bed_id: 'RAD-US-1', ward_unit: 'Radiology', equipment_baseline: ['Portable Ultrasound'] },
  
  // Pathology Lab
  { bed_id: 'PATH-LAB-1', ward_unit: 'Pathology', equipment_baseline: ['Blood Gas Analyzer'] },
  
  // Pharmacy
  { bed_id: 'PHARM-1', ward_unit: 'Pharmacy', equipment_baseline: ['Infusion Pump'] },
];

const MEDICATIONS = [
  { name: 'Amoxicillin', form: 'Tablet', route: 'PO', stock: 500, price: 5.00 },
  { name: 'Ceftriaxone', form: 'Injection', route: 'IV', stock: 300, price: 15.00 },
  { name: 'Doxycycline', form: 'Tablet', route: 'PO', stock: 400, price: 3.50 },
  { name: 'Acetaminophen', form: 'Tablet', route: 'PO', stock: 1000, price: 1.00 },
  { name: 'Ibuprofen', form: 'Tablet', route: 'PO', stock: 800, price: 2.00 },
  { name: 'Metformin', form: 'Tablet', route: 'PO', stock: 600, price: 2.50 },
  { name: 'Normal Saline', form: 'Injection', route: 'IV', stock: 200, price: 5.00 },
  { name: "Ringer's Lactate", form: 'Injection', route: 'IV', stock: 200, price: 6.00 },
];

// ============ API HELPERS ============
async function makeRequest(method, endpoint, data = null) {
  const options = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (data) options.body = JSON.stringify(data);
  
  const response = await fetch(`${API_BASE}${endpoint}`, options);
  const result = await response.json();
  
  if (!response.ok) {
    console.error(`Response body:`, result);
    throw new Error(`${method} ${endpoint}: ${result.error || result.message || response.statusText}`);
  }
  return result;
}

async function makeAuthenticatedRequest(method, endpoint, data = null, token) {
  const options = {
    method,
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
  };
  if (data) options.body = JSON.stringify(data);
  
  const response = await fetch(`${API_BASE}${endpoint}`, options);
  const result = await response.json();
  
  if (!response.ok) {
    console.error(`Response body:`, result);
    throw new Error(`${method} ${endpoint}: ${result.error || result.message || response.statusText}`);
  }
  return result;
}

// ============ MAIN SETUP FUNCTION ============
async function setupCompleteHospital(hospitalName) {
  try {
    section('🏥 COMPLETE HOSPITAL SETUP WITH AUTO CLOCK-IN');
    
    // CLEANUP: Delete all existing data for clean testing environment
    section('Step 0: Database Cleanup - Deleting All Existing Data');
    log('Connecting to MongoDB for database cleanup...');
    await mongoose.connect(MONGO_URI);
    
    // Import models for cleanup
    const HospitalModel = await import('./src/models/hospital.model.js').then(m => m.Hospital);
    const ManagerModel = await import('./src/models/manager.model.js').then(m => m.Manager);
    const ClinicalStaffModel = await import('./src/models/clinicalstaff.model.js').then(m => m.ClinicalStaff);
    const ShiftModel = await import('./src/models/shift.model.js').then(m => m.Shift);
    const NurseClockRecordModel = await import('./src/models/staffClockRecord.model.js').then(m => m.NurseClockRecord);
    const BedModel = await import('./src/models/bed.model.js').then(m => m.Bed);
    const PharmacyInventoryModel = await import('./src/models/pharmacy.model.js').then(m => m.PharmacyInventory);
    
    // Delete all documents from collections
    log('Deleting all hospitals...');
    const hospDeleted = await HospitalModel.deleteMany({});
    log(`  ✓ Deleted ${hospDeleted.deletedCount} hospitals`);
    
    log('Deleting all managers...');
    const mgrDeleted = await ManagerModel.deleteMany({});
    log(`  ✓ Deleted ${mgrDeleted.deletedCount} managers`);
    
    log('Deleting all clinical staff...');
    const staffDeleted = await ClinicalStaffModel.deleteMany({});
    log(`  ✓ Deleted ${staffDeleted.deletedCount} staff members`);
    
    log('Deleting all shifts...');
    const shiftDeleted = await ShiftModel.deleteMany({});
    log(`  ✓ Deleted ${shiftDeleted.deletedCount} shifts`);
    
    log('Deleting all clock records...');
    const clockDeleted = await NurseClockRecordModel.deleteMany({});
    log(`  ✓ Deleted ${clockDeleted.deletedCount} clock records`);
    
    log('Deleting all beds...');
    const bedDeleted = await BedModel.deleteMany({});
    log(`  ✓ Deleted ${bedDeleted.deletedCount} beds`);
    
    log('Deleting all pharmacy inventory...');
    const pharmDeleted = await PharmacyInventoryModel.deleteMany({});
    log(`  ✓ Deleted ${pharmDeleted.deletedCount} pharmacy records`);
    
    await mongoose.disconnect();
    log('✓ Database cleanup complete\n');
    
    section('Step 1: Hospital Registration');

    // 1. Register Hospital
    const hospitalRegistration = await makeRequest('POST', '/hospital/register', {
      hospitalName: hospitalName || `Hospital-${Date.now()}`,
      managerName: 'Admin',
      managerSurname: 'Account',
      managerEmail: 'admin@testhospital.local',
      managerPassword: 'Admin@123456',
      department: 'All'
    });

    const { token, hospital, user } = hospitalRegistration;
    success(`Hospital registered: ${hospital.name}`);
    success(`Hospital ID: ${hospital.hospitalID}`);
    success(`Admin Email: admin@testhospital.local`);
    success(`Admin Token acquired\n`);

    // 2. Create Staff - Physicians
    section('Step 2: Creating Staff Members');
    const allStaffIds = [];
    const staffCredentials = [];
    const roleMap = {
      'PHYSICIAN': 'MD',
      'TRIAGE_NURSE': 'RN',
      'CLINICAL_NURSE': 'RN',
      'RADIOLOGIST': 'MD',
      'PATHOLOGIST': 'MD',
      'PHARMACIST': 'PA',
    };

    // Register Physicians
    log('Creating Physicians...');
    for (const phys of PHYSICIANS) {
      const email = `${phys.name.toLowerCase().replace(/\s+/g, '.')}@testhospital.local`;
      const response = await makeAuthenticatedRequest('POST', '/manager/staff/register', {
        firstName: phys.name.split(' ')[1],
        lastName: phys.name.split(' ')[2] || 'Staff',
        email: email,
        password: 'Staff@123456',
        role: 'PHYSICIAN',
        specialty: phys.specialty,
        department: phys.department,
        credentials: {
          title: roleMap['PHYSICIAN'],
          license: `LIC-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
        }
      }, token);

      allStaffIds.push(response.staff.id);
      staffCredentials.push({ name: phys.name, email, password: 'Staff@123456', role: 'PHYSICIAN', specialty: phys.specialty });
      success(`  ✓ ${phys.name} (${phys.specialty})`);
    }

    // Register Nurses
    log('\nCreating Nurses...');
    for (const nurse of NURSES) {
      const email = `${nurse.name.toLowerCase().replace(/\s+/g, '.')}@testhospital.local`;
      const response = await makeAuthenticatedRequest('POST', '/manager/staff/register', {
        firstName: nurse.name.split(' ')[1],
        lastName: nurse.name.split(' ')[2] || 'Staff',
        email: email,
        password: 'Staff@123456',
        role: 'NURSE',
        specialty: nurse.specialty,
        department: nurse.department,
        credentials: {
          title: roleMap['NURSE'],
          license: `LIC-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
        }
      }, token);

      allStaffIds.push(response.staff.id);
      staffCredentials.push({ name: nurse.name, email, password: 'Staff@123456', role: 'NURSE', specialty: nurse.specialty, nsgRole: nurse.role || 'CLINICAL_NURSE' });
      success(`  ✓ ${nurse.name} (${nurse.specialty})`);
    }

    // Register Radiologists
    log('\nCreating Radiologists...');
    for (const rad of RADIOLOGISTS) {
      const email = `${rad.name.toLowerCase().replace(/\s+/g, '.')}@testhospital.local`;
      const response = await makeAuthenticatedRequest('POST', '/manager/staff/register', {
        firstName: rad.name.split(' ')[1],
        lastName: rad.name.split(' ')[2] || 'Staff',
        email: email,
        password: 'Staff@123456',
        role: 'RADIOLOGIST',
        specialty: rad.specialty,
        department: rad.department,
        credentials: {
          title: roleMap['RADIOLOGIST'],
          license: `LIC-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
        }
      }, token);

      allStaffIds.push(response.staff.id);
      staffCredentials.push({ name: rad.name, email, password: 'Staff@123456', role: 'RADIOLOGIST', specialty: rad.specialty });
      success(`  ✓ ${rad.name} (Radiology)`);
    }

    // Register Pathologists
    log('\nCreating Pathologists...');
    for (const path_staff of PATHOLOGISTS) {
      const email = `${path_staff.name.toLowerCase().replace(/\s+/g, '.')}@testhospital.local`;
      const response = await makeAuthenticatedRequest('POST', '/manager/staff/register', {
        firstName: path_staff.name.split(' ')[1],
        lastName: path_staff.name.split(' ')[2] || 'Staff',
        email: email,
        password: 'Staff@123456',
        role: 'PATHOLOGIST',
        specialty: path_staff.specialty,
        department: path_staff.department,
        credentials: {
          title: roleMap['PATHOLOGIST'],
          license: `LIC-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
        }
      }, token);

      allStaffIds.push(response.staff.id);
      staffCredentials.push({ name: path_staff.name, email, password: 'Staff@123456', role: 'PATHOLOGIST', specialty: path_staff.specialty });
      success(`  ✓ ${path_staff.name} (Pathology)`);
    }

    // Register Pharmacists
    log('\nCreating Pharmacists...');
    for (const pharm of PHARMACISTS) {
      const email = `${pharm.name.toLowerCase().replace(/\s+/g, '.')}@testhospital.local`;
      const response = await makeAuthenticatedRequest('POST', '/manager/staff/register', {
        firstName: pharm.name.split(' ')[1],
        lastName: pharm.name.split(' ')[2] || 'Staff',
        email: email,
        password: 'Staff@123456',
        role: 'PHARMACIST',
        specialty: pharm.specialty,
        department: pharm.department,
        credentials: {
          title: roleMap['PHARMACIST'],
          license: `LIC-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
        }
      }, token);

      allStaffIds.push(response.staff.id);
      staffCredentials.push({ name: pharm.name, email, password: 'Staff@123456', role: 'PHARMACIST', specialty: pharm.specialty });
      success(`  ✓ ${pharm.name} (Pharmacy)`);
    }

    success(`\n✓ Created ${allStaffIds.length} staff members\n`);

    // 3. Create Beds
    section('Step 3: Creating Hospital Beds');
    const bedsPayload = BEDS.map(bed => ({
      bed_id: bed.bed_id,
      ward_unit: bed.ward_unit,
      equipment_baseline: bed.equipment_baseline,
    }));

    await makeAuthenticatedRequest('POST', '/beds/batch', { beds: bedsPayload }, token);
    success(`✓ Created ${BEDS.length} beds across departments\n`);

    // 4. Create Pharmaceutical Inventory
    section('Step 4: Creating Pharmaceutical Inventory');
    const pharmPayload = MEDICATIONS.map(med => ({
      medicineID: med.name.toLowerCase().replace(/\s+/g, '-'),
      name: med.name,
      form: med.form,
      route: med.route,
      stock: med.stock,
      price: med.price,
    }));

    await makeAuthenticatedRequest('POST', '/pharmacy/batch', { pharmaceuticals: pharmPayload }, token);
    success(`✓ Created ${MEDICATIONS.length} medications in inventory\n`);

    // 5. Connect to MongoDB for Shift Creation
    section('Step 5: Creating Shifts & Auto Clock-in');
    log('Connecting to MongoDB...');
    await mongoose.connect(MONGO_URI);
    success('Connected to MongoDB\n');

    const { Shift } = await import('./src/models/shift.model.js');
    const { ClinicalStaff } = await import('./src/models/clinicalstaff.model.js');
    const { NurseClockRecord } = await import('./src/models/staffClockRecord.model.js');
    const { NurseSchedule } = await import('./src/models/nurseSchedule.model.js');

    // Register NurseSchedule model if not already registered
    if (!mongoose.models.NurseSchedule) {
      mongoose.model('NurseSchedule', NurseSchedule.schema || NurseSchedule);
    }

    // Use hospitalId from the registration response
    const hospitalObjectId = user.hospitalId;
    const managerId = user.id || user._id;

    // 6. Create Shifts
    const calculateDuration = (start, end) => {
      const [startHour] = start.split(':').map(Number);
      const [endHour] = end.split(':').map(Number);
      if (endHour > startHour) {
        return endHour - startHour;
      } else {
        // Night shift wraps around
        return (24 - startHour) + endHour;
      }
    };

    const shifts = [
      { name: 'Full Day Shift', start: '08:00', end: '23:55', color: '#3b82f6' },
    ];

    const createdShifts = [];
    log('Creating shifts...');
    for (const shift of shifts) {
      const shiftDoc = new Shift({
        hospitalId: hospitalObjectId,
        shiftID: `SHIFT-${Date.now()}-${Math.random().toString(36).substr(2, 5).toUpperCase()}`,
        shiftName: shift.name,
        shiftStart: shift.start,
        shiftEnd: shift.end,
        duration: calculateDuration(shift.start, shift.end),
        department: 'Nursing',
        createdBy: new mongoose.Types.ObjectId(managerId),
        color: shift.color,
        status: 'PUBLISHED',
        maxStaffRequired: 1,
        assignedStaff: [],
      });
      await shiftDoc.save();
      createdShifts.push(shiftDoc);
      success(`  ✓ ${shift.name}`);
    }

    log();

    // 7. Fetch created staff and assign to shifts
    log('Assigning staff to shifts...');
    const staffInHospital = await ClinicalStaff.find({ hospitalId: hospitalObjectId });
    success(`✓ Found ${staffInHospital.length} staff in hospital\n`);

    let staffAssigned = 0;
    for (const staff of staffInHospital) {
      // Distribute staff across shifts
      const shiftIndex = staffAssigned % createdShifts.length;
      const shift = createdShifts[shiftIndex];

      // Determine role based on specialty
      let assignmentType = 'NURSE';
      if (staff.systemRole === 'PHYSICIAN') assignmentType = 'PHYSICIAN';
      else if (staff.systemRole === 'TECHNICIAN') assignmentType = 'SPECIALIST';
      else if (staff.systemRole === 'ADMINISTRATIVE') assignmentType = 'SUPPORT';

      shift.assignedStaff.push({
        staffID: staff._id,
        type: assignmentType,
        name: staff.firstName || staff.name,
        surname: staff.lastName || 'Staff',
        specialty: staff.clinicalSpecialty || 'General',
        status: 'ASSIGNED',
      });

      staffAssigned++;
    }

    // Save all shifts with assignments
    for (const shift of createdShifts) {
      await shift.save();
    }

    success(`✓ Assigned ${staffAssigned} staff to shifts\n`);

    // 8. Auto Clock-in ALL Staff
    section('Step 6: Auto Clock-in All Staff');
    
    // DEBUG: Show staff breakdown by role
    const staffByRole = {};
    staffInHospital.forEach(staff => {
      const role = staff.systemRole || 'UNKNOWN';
      staffByRole[role] = (staffByRole[role] || 0) + 1;
    });
    
    log('Staff breakdown in database:');
    Object.entries(staffByRole).forEach(([role, count]) => {
      log(`  ${role}: ${count}`);
    });
    log();
    
    log('Detailed staff list:');
    staffInHospital.forEach((staff, idx) => {
      log(`  ${idx + 1}. ${staff.firstName} ${staff.lastName} - ${staff.systemRole} (${staff.clinicalSpecialty}) [${staff._id}]`);
    });
    log();
    
    log('Clocking in all staff members...');

    let clockedInCount = 0;
    const clockedInByRole = {};
    const failedClockIn = [];
    
    for (const staff of staffInHospital) {
      try {
        const shiftIndex = clockedInCount % createdShifts.length;
        const shift = createdShifts[shiftIndex];

        const clockRecord = new NurseClockRecord({
          staffId: staff._id,
          hospitalId: hospitalObjectId,
          shiftId: shift._id,
          date: new Date(),
          clockInTime: new Date(),
          status: 'CLOCKED_IN',
          clockOutTime: null,
          breakStartTime: null,
          breakEndTime: null,
          totalBreakTime: 0,
          duration: null,
          notes: `Auto clock-in for ${shift.shiftName}`,
        });

        await clockRecord.save();
        clockedInCount++;
        
        // Track by role
        const role = staff.systemRole || 'UNKNOWN';
        clockedInByRole[role] = (clockedInByRole[role] || 0) + 1;
      } catch (clockErr) {
        failedClockIn.push({ staff: `${staff.firstName} ${staff.lastName}`, error: clockErr.message });
        error(`  ✗ Failed to clock in ${staff.firstName} ${staff.lastName}: ${clockErr.message}`);
      }
    }

    success(`✓ Auto clocked-in ${clockedInCount} staff members\n`);
    
    // Show clock-in summary by role
    log('Clock-in summary by role:');
    Object.entries(clockedInByRole).forEach(([role, count]) => {
      log(`  ✓ ${role}: ${count} staff`);
    });
    
    if (failedClockIn.length > 0) {
      log('\nFailed clock-ins:');
      failedClockIn.forEach(fail => {
        log(`  ✗ ${fail.staff}: ${fail.error}`);
      });
    }
    log();

    // 9. Output Summary & Credentials
    section('✨ SETUP COMPLETE');

    console.log(`
╔═══════════════════════════════════════════════════════════════╗
║                    HOSPITAL SETUP COMPLETE                   ║
╚═══════════════════════════════════════════════════════════════╝

Hospital: ${hospital.name}
Hospital ID: ${hospital.hospitalID}

INFRASTRUCTURE:
  ✓ Staff: ${staffInHospital.length} total
    - Physicians: ${staffByRole['PHYSICIAN'] || 0}
    - Nurses: ${staffByRole['NURSE'] || 0}
    - Technicians: ${staffByRole['TECHNICIAN'] || 0}
    - Administrative: ${staffByRole['ADMINISTRATIVE'] || 0}
    - Other: ${staffByRole['OTHER'] || 0}
  ✓ Beds: ${BEDS.length} (across 7 departments)
  ✓ Medications: ${MEDICATIONS.length} in inventory
  ✓ Shifts: ${createdShifts.length} (Morning, Evening, Night)
  ✓ Clocked-In: ${clockedInCount} staff ready for testing
    - Physicians: ${clockedInByRole['PHYSICIAN'] || 0}
    - Nurses: ${clockedInByRole['NURSE'] || 0}
    - Technicians: ${clockedInByRole['TECHNICIAN'] || 0}
    - Administrative: ${clockedInByRole['ADMINISTRATIVE'] || 0}

═══════════════════════════════════════════════════════════════════

ADMIN CREDENTIALS:
  Email: admin@testhospital.local
  Password: Admin@123456

═══════════════════════════════════════════════════════════════════

TEST STAFF CREDENTIALS (Sample - all use password: Staff@123456):

PHYSICIANS:
`);

    staffCredentials.filter(s => s.role === 'PHYSICIAN').forEach(staff => {
      console.log(`  ${staff.email} (${staff.specialty})`);
    });

    console.log(`\nNURSES (Triage):
`);
    staffCredentials.filter(s => s.role === 'NURSE' && s.nsgRole === 'TRIAGE_NURSE').slice(0, 2).forEach(staff => {
      console.log(`  ${staff.email}`);
    });

    console.log(`\nNURSES (Clinical):
`);
    staffCredentials.filter(s => s.role === 'NURSE' && s.nsgRole === 'CLINICAL_NURSE').slice(0, 2).forEach(staff => {
      console.log(`  ${staff.email}`);
    });

    console.log(`\nRADIOLOGISTS:
`);
    staffCredentials.filter(s => s.role === 'RADIOLOGIST').forEach(staff => {
      console.log(`  ${staff.email}`);
    });

    console.log(`\nPATHOLOGISTS:
`);
    staffCredentials.filter(s => s.role === 'PATHOLOGIST').forEach(staff => {
      console.log(`  ${staff.email}`);
    });

    console.log(`\nPHARMACISTS:
`);
    staffCredentials.filter(s => s.role === 'PHARMACIST').forEach(staff => {
      console.log(`  ${staff.email}`);
    });

    console.log(`\n═══════════════════════════════════════════════════════════════════

READY FOR TESTING:
  ✓ Admin dashboard: Manage hospital infrastructure
  ✓ Triage: Create and triage patients
  ✓ Physician dashboard: View and manage cases
  ✓ Radiology lab: Receive and process imaging orders
  ✓ Pathology lab: Receive and process lab orders
  ✓ Pharmacy dashboard: Manage medication orders and inventory

NEXT STEPS:
  1. Login as admin → Verify hospital setup
  2. Login as physician → Test physician dashboard
  3. Login as triage nurse → Create test patient
  4. Send lab orders to radiologist/pathologist
  5. Verify medication inventory tracking
  6. Track staff clock-in status

═══════════════════════════════════════════════════════════════════
    `);

    // Save credentials to file for reference
    const credsFile = `hospital-${hospital.hospitalID}-credentials.json`;
    const fs = await import('fs');
    fs.writeFileSync(credsFile, JSON.stringify({
      hospital: {
        name: hospital.name,
        id: hospital.hospitalID,
      },
      admin: {
        email: 'admin@testhospital.local',
        password: 'Admin@123456',
      },
      staff: staffCredentials,
      api: {
        base: API_BASE,
      },
    }, null, 2));

    success(`✓ Credentials saved to: ${credsFile}`);

    await mongoose.disconnect();
    success('✓ Database disconnected');

  } catch (err) {
    error(err.message);
    console.error(err.stack);
    process.exit(1);
  }
}

// ============ RUN ============
const hospitalName = process.argv[2] || `Hospital-${Date.now()}`;
log(`Hospital Name: ${hospitalName}`);
setupCompleteHospital(hospitalName);
