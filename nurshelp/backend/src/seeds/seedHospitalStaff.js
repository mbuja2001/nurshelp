/**
 * Hospital Staff Database Seeding Script
 * Purpose: Clear all data and populate with realistic hospital staff
 * Run: node backend/src/seeds/seedHospitalStaff.js
 * 
 * This seeds:
 * - Managers (Shift Manager, Department Managers, Admin)
 * - Multiple types of Nurses (RN, LPN, Specialists)
 * - Specialist Physicians (various specialties)
 * - Sample shifts for demonstration
 * 
 * Clears:
 * - All Patients
 * - All Triage Records
 * - All Nurses
 * - All Physicians
 * - All Shifts
 * - All Clock Records
 * - All Schedules
 */

import mongoose from 'mongoose';
import bcrypt from 'bcrypt';
import dotenv from 'dotenv';

dotenv.config();

// Models
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import NurseProfile from '../models/nurseProfile.model.js';
import { Manager } from '../models/manager.model.js';
import { Shift } from '../models/shift.model.js';
import PatientEHRRecord from '../models/patientEHRRecord.model.js';
import { NurseSchedule } from '../models/nurseSchedule.model.js';
import { NurseClockRecord } from '../models/staffClockRecord.model.js';

const MONGO_URI = process.env.MONGO_URI;

// Connect to MongoDB
async function connectDB() {
  try {
    await mongoose.connect(MONGO_URI);
    console.log('✅ MongoDB connected for seeding');
  } catch (err) {
    console.error('❌ MongoDB connection error:', err);
    process.exit(1);
  }
}

// Clear all existing data
async function clearDatabase() {
  console.log('\n🧹 Clearing existing data...');
  try {
    await ClinicalStaff.deleteMany({});
    console.log('✓ Cleared clinical staff (nurses/physicians)');
    
    await NurseProfile.deleteMany({});
    console.log('✓ Cleared NurseProfiles');
    
    await Manager.deleteMany({});
    console.log('✓ Cleared Managers');
    
    await Shift.deleteMany({});
    console.log('✓ Cleared Shifts');
    
    await PatientEHRRecord.deleteMany({});
    console.log('✓ Cleared Patients');
    
    await NurseSchedule.deleteMany({});
    console.log('✓ Cleared Schedules');
    
    await NurseClockRecord.deleteMany({});
    console.log('✓ Cleared Clock Records');
    
  } catch (err) {
    console.error('❌ Error clearing database:', err);
    throw err;
  }
}

// Create Managers
async function seedManagers() {
  console.log('\n👨‍💼 Seeding Managers...');
  const managers = [
    {
      managerID: 'MGR-001',
      name: 'John',
      surname: 'Anderson',
      email: 'john.anderson@hospital.com',
      password: await bcrypt.hash('manager123', 10),
      phone: '555-0201',
      department: 'ED',
      role: 'SHIFT_MANAGER',
      permissions: {
        canCreateShifts: true,
        canEditShifts: true,
        canDeleteShifts: true,
        canAssignStaff: true,
        canApproveLeaves: false,
        canManageRoles: false,
        canViewReports: true,
      },
      isActive: true,
    },
    {
      managerID: 'MGR-002',
      name: 'Patricia',
      surname: 'Williams',
      email: 'patricia.williams@hospital.com',
      password: await bcrypt.hash('manager123', 10),
      phone: '555-0202',
      department: 'ICU',
      role: 'DEPARTMENT_MANAGER',
      permissions: {
        canCreateShifts: true,
        canEditShifts: true,
        canDeleteShifts: true,
        canAssignStaff: true,
        canApproveLeaves: true,
        canManageRoles: false,
        canViewReports: true,
      },
      isActive: true,
    },
    {
      managerID: 'MGR-003',
      name: 'David',
      surname: 'Brown',
      email: 'david.brown@hospital.com',
      password: await bcrypt.hash('admin123', 10),
      phone: '555-0203',
      department: 'All',
      role: 'ADMIN',
      permissions: {
        canCreateShifts: true,
        canEditShifts: true,
        canDeleteShifts: true,
        canAssignStaff: true,
        canApproveLeaves: true,
        canManageRoles: true,
        canViewReports: true,
      },
      isActive: true,
    },
    {
      managerID: 'MGR-004',
      name: 'Susan',
      surname: 'Miller',
      email: 'susan.miller@hospital.com',
      password: await bcrypt.hash('manager123', 10),
      phone: '555-0204',
      department: 'OR',
      role: 'SHIFT_MANAGER',
      permissions: {
        canCreateShifts: true,
        canEditShifts: true,
        canDeleteShifts: true,
        canAssignStaff: true,
        canApproveLeaves: false,
        canManageRoles: false,
        canViewReports: true,
      },
      isActive: true,
    },
  ];

  try {
    const created = await Manager.insertMany(managers);
    console.log(`✓ Created ${created.length} managers`);
    return created;
  } catch (err) {
    console.error('❌ Error seeding managers:', err);
    throw err;
  }
}

// Create Nurses with different specialties
async function seedNurses() {
  console.log('\n💉 Seeding Nurses...');
  
  const nurseColors = [
    // Emergency Nurses
    { name: 'Sarah', specialty: 'Emergency Nurse', department: 'ED', licenseType: 'RN', nqfLevel: 7 },
    { name: 'Michael', specialty: 'Trauma Nurse', department: 'ED', licenseType: 'RN', nqfLevel: 8 },
    { name: 'Jessica', specialty: 'Emergency Nurse', department: 'ED', licenseType: 'RN', nqfLevel: 6 },
    
    // ICU Nurses
    { name: 'Robert', specialty: 'ICU Nurse', department: 'ICU', licenseType: 'RN', nqfLevel: 8 },
    { name: 'Emily', specialty: 'Critical Care Nurse', department: 'ICU', licenseType: 'RN', nqfLevel: 7 },
    { name: 'David', specialty: 'ICU Nurse', department: 'ICU', licenseType: 'RN', nqfLevel: 7 },
    
    // Operating Room Nurses
    { name: 'Jennifer', specialty: 'Theatre Nurse', department: 'OR', licenseType: 'RN', nqfLevel: 8 },
    { name: 'Christopher', specialty: 'Perioperative Nurse', department: 'OR', licenseType: 'RN', nqfLevel: 7 },
    { name: 'Amanda', specialty: 'Theatre Nurse', department: 'OR', licenseType: 'RN', nqfLevel: 6 },
    
    // Obstetrics Nurses
    { name: 'Nicole', specialty: 'Midwife', department: 'OB', licenseType: 'RN', nqfLevel: 8 },
    { name: 'Lauren', specialty: 'Neonatal Nurse', department: 'OB', licenseType: 'RN', nqfLevel: 7 },
    { name: 'Karen', specialty: 'Midwife', department: 'OB', licenseType: 'RN', nqfLevel: 7 },
    
    // Medical-Surgical Nurses
    { name: 'James', specialty: 'Dialysis Nurse', department: 'Medical-Surgical', licenseType: 'RN', nqfLevel: 7 },
    { name: 'Maria', specialty: 'Oncology Nurse', department: 'Medical-Surgical', licenseType: 'RN', nqfLevel: 8 },
    { name: 'Lisa', specialty: 'General Nurse', department: 'Medical-Surgical', licenseType: 'RN', nqfLevel: 6 },
    
    // Pediatric Nurses
    { name: 'Thomas', specialty: 'Paediatric Nurse', department: 'Pediatrics', licenseType: 'RN', nqfLevel: 7 },
    { name: 'Angela', specialty: 'Neonatal Nurse', department: 'Pediatrics', licenseType: 'RN', nqfLevel: 8 },
    { name: 'Steven', specialty: 'Paediatric Nurse', department: 'Pediatrics', licenseType: 'RN', nqfLevel: 7 },
    
    // Psychiatric Nurses
    { name: 'Patricia', specialty: 'Mental Health Nurse', department: 'Psychiatry', licenseType: 'RN', nqfLevel: 7 },
    { name: 'Mark', specialty: 'Psychiatric Nurse', department: 'Psychiatry', licenseType: 'RN', nqfLevel: 7 },
    { name: 'Sandra', specialty: 'Mental Health Nurse', department: 'Psychiatry', licenseType: 'RN', nqfLevel: 6 },
  ];

  const nurses = [];
  for (let i = 0; i < nurseColors.length; i++) {
    const nc = nurseColors[i];
    const nurseID = `NURSE-${String(i + 1).padStart(3, '0')}`;
    const email = `${nc.name.toLowerCase()}.${i + 1}@hospital.com`;
    const password = await bcrypt.hash('nurse123', 10);

    nurses.push({
      name: nc.name,
      surname: 'Nurse-' + (i + 1),
      email,
      password,
      role: 'NURSE',
      department: nc.department,
      specialty: nc.specialty,
      licenseType: nc.licenseType,
      isActive: true,
    });
  }

  try {
    const created = await ClinicalStaff.insertMany(nurses);
    console.log(`✓ Created ${created.length} nurses`);
    return created;
  } catch (err) {
    console.error('❌ Error seeding nurses:', err);
    throw err;
  }
}

// Create Specialist Physicians
async function seedPhysicians() {
  console.log('\n👨‍⚕️ Seeding Physicians...');
  
  const physicianSpecialties = [
    // Emergency Medicine
    { name: 'Dr. William', specialty: 'Emergency Medicine', department: 'ED' },
    { name: 'Dr. Elizabeth', specialty: 'Emergency Medicine', department: 'ED' },
    
    // Critical Care / Intensive Care
    { name: 'Dr. Joseph', specialty: 'Critical Care', department: 'ICU' },
    { name: 'Dr. Barbara', specialty: 'Critical Care', department: 'ICU' },
    
    // Surgery / Operating Room
    { name: 'Dr. Richard', specialty: 'General Surgery', department: 'OR' },
    { name: 'Dr. Susan', specialty: 'Orthopaedic Surgery', department: 'OR' },
    { name: 'Dr. Charles', specialty: 'Cardiovascular Surgery', department: 'OR' },
    
    // Obstetrics & Gynecology
    { name: 'Dr. Margaret', specialty: 'Obstetrics & Gynecology', department: 'OB' },
    { name: 'Dr. Donald', specialty: 'Obstetrics & Gynecology', department: 'OB' },
    
    // Internal Medicine
    { name: 'Dr. Karen', specialty: 'Cardiology', department: 'Medical-Surgical' },
    { name: 'Dr. Edward', specialty: 'Pulmonology', department: 'Medical-Surgical' },
    { name: 'Dr. Nancy', specialty: 'Gastroenterology', department: 'Medical-Surgical' },
    { name: 'Dr. Ronald', specialty: 'Nephrology', department: 'Medical-Surgical' },
    
    // Pediatrics
    { name: 'Dr. Lisa', specialty: 'Pediatrics', department: 'Pediatrics' },
    { name: 'Dr. Paul', specialty: 'Neonatology', department: 'Pediatrics' },
    
    // Psychiatry
    { name: 'Dr. Mary', specialty: 'Psychiatry', department: 'Psychiatry' },
    { name: 'Dr. Mark', specialty: 'Psychiatry', department: 'Psychiatry' },
  ];

  const physicians = [];
  for (let i = 0; i < physicianSpecialties.length; i++) {
    const ps = physicianSpecialties[i];
    const physicianID = `PHY-${String(i + 1).padStart(3, '0')}`;
    const email = `${ps.name.toLowerCase().replace('dr. ', '')}@hospital.com`;
    const password = await bcrypt.hash('physician123', 10);

    physicians.push({
      physicianID,
      name: ps.name,
      surname: 'MD',
      email,
      password,
      role: 'PHYSICIAN',
      department: ps.department,
      specialty: ps.specialty,
      licenseNumber: `LIC-${String(i + 1).padStart(5, '0')}`,
      licenseType: 'SPECIALIST',
      isActive: true,
    });
  }

  try {
    const created = await ClinicalStaff.insertMany(physicians); // Using ClinicalStaff model for physicians too
    console.log(`✓ Created ${created.length} physicians`);
    return created;
  } catch (err) {
    console.error('❌ Error seeding physicians:', err);
    throw err;
  }
}

// Run all seeding
async function seedDatabase() {
  try {
    await connectDB();
    await clearDatabase();
    
    const managers = await seedManagers();
    const nurses = await seedNurses();
    const physicians = await seedPhysicians();

    console.log('\n✅ Database seeding completed successfully!');
    console.log(`\n📊 Summary:`);
    console.log(`  - Managers: ${managers.length}`);
    console.log(`  - Nurses: ${nurses.length}`);
    console.log(`  - Physicians: ${physicians.length}`);
    console.log(`\n🔐 Default Credentials:`);
    console.log(`  Managers: password='manager123' or 'admin123'`);
    console.log(`  Nurses: password='nurse123'`);
    console.log(`  Physicians: password='physician123'`);

  } catch (err) {
    console.error('❌ Seeding failed:', err);
    process.exit(1);
  } finally {
    await mongoose.connection.close();
  }
}

seedDatabase();
