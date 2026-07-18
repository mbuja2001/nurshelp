#!/usr/bin/env node
/**
 * Hospital System Population Script
 * 
 * Creates a complete hospital with:
 * - Multiple departments (ED, ICU, OR, OB, Medical-Surgical, Pediatrics, Psychiatry)
 * - Realistic staff roster (nurses and physicians per department)
 * - Shift schedules with role-based team assignments
 * - Dynamic roster for ESI_Engine to query active physicians
 * 
 * Usage:
 *   node populate-hospital-system.js
 * 
 * This script:
 * 1. Registers a hospital named "St. Mercy Teaching Hospital"
 * 2. Creates departments with staffing models
 * 3. Populates clinical staff (nurses and physicians)
 * 4. Creates shifts with team assignments for ESI_Engine integration
 */

import mongoose from 'mongoose';
import { Hospital } from './backend/src/models/hospital.model.js';
import { Manager } from './backend/src/models/manager.model.js';
import { ClinicalStaff } from './backend/src/models/clinicalstaff.model.js';
import { Shift } from './backend/src/models/shift.model.js';
import bcrypt from 'bcrypt';
import dotenv from 'dotenv';

dotenv.config();

// ============================================
// CONNECTION SETUP
// ============================================

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/nursehelpdesk';

console.log('🔗 Connecting to MongoDB:', MONGODB_URI);

try {
  await mongoose.connect(MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });
  console.log('✓ MongoDB connected');
} catch (err) {
  console.error('❌ MongoDB connection failed:', err.message);
  process.exit(1);
}

// ============================================
// DEPARTMENT & STAFFING CONFIGURATION
// ============================================

const DEPARTMENTS = {
  ED: {
    name: 'ED',
    fullName: 'Emergency Department',
    nurses: 4,
    physicians: 2,
    specialties: ['Emergency Medicine', 'Trauma Surgery', 'Critical Care']
  },
  ICU: {
    name: 'ICU',
    fullName: 'Intensive Care Unit',
    nurses: 3,
    physicians: 2,
    specialties: ['Critical Care', 'Pulmonology', 'Cardiology']
  },
  OR: {
    name: 'OR',
    fullName: 'Operating Room',
    nurses: 3,
    physicians: 2,
    specialties: ['General Surgery', 'Orthopedic Surgery', 'Vascular Surgery']
  },
  OB: {
    name: 'OB',
    fullName: 'Obstetrics & Gynaecology',
    nurses: 2,
    physicians: 2,
    specialties: ['Obstetrics', 'Gynecology', 'Maternal-Fetal Medicine']
  },
  'Medical-Surgical': {
    name: 'Medical-Surgical',
    fullName: 'Medical-Surgical Ward',
    nurses: 3,
    physicians: 2,
    specialties: ['Internal Medicine', 'Gastroenterology', 'Nephrology']
  },
  Pediatrics: {
    name: 'Pediatrics',
    fullName: 'Pediatrics Department',
    nurses: 2,
    physicians: 1,
    specialties: ['Pediatrics', 'Neonatology', 'Pediatric Surgery']
  },
  Psychiatry: {
    name: 'Psychiatry',
    fullName: 'Psychiatry Department',
    nurses: 2,
    physicians: 1,
    specialties: ['Psychiatry', 'Behavioral Medicine']
  }
};

// Staff name pools for realistic naming
const FIRST_NAMES = {
  NURSE: ['Sarah', 'Emma', 'Rachel', 'Jessica', 'Michael', 'James', 'David', 'Robert'],
  PHYSICIAN: ['Dr. Chen', 'Dr. Patel', 'Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis', 'Dr. Miller']
};

const LAST_NAMES = {
  NURSE: ['Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez'],
  PHYSICIAN: ['Chen', 'Patel', 'Smith', 'Johnson', 'Williams', 'Brown', 'Davis', 'Miller']
};

// ============================================
// HELPER FUNCTIONS
// ============================================

function generateEmail(firstName, lastName, department) {
  const fname = firstName.toLowerCase().replace(/\s+/g, '');
  const lname = lastName.toLowerCase().replace(/\s+/g, '');
  const deptCode = department.toLowerCase().replace(/[^a-z]/g, '') || 'gen';
  return `${fname}.${lname}.${deptCode}@stmercy.com`;
}

function generateStaffName(role, index, department) {
  const firstNames = FIRST_NAMES[role] || FIRST_NAMES.NURSE;
  const lastNames = LAST_NAMES[role] || LAST_NAMES.NURSE;
  
  const firstName = firstNames[index % firstNames.length];
  const lastName = lastNames[(index + index) % lastNames.length];
  
  return { firstName, lastName };
}

function getSpecialtyForRole(role, department, index) {
  const specialties = DEPARTMENTS[department]?.specialties || ['General Practice'];
  return specialties[index % specialties.length];
}

async function generateShiftTimes() {
  // Standard hospital shift times
  return [
    { name: '08:00 - 16:00', start: '08:00', end: '16:00', duration: 8 },
    { name: '16:00 - 00:00', start: '16:00', end: '00:00', duration: 8 },
    { name: '00:00 - 08:00', start: '00:00', end: '08:00', duration: 8 },
    { name: 'On-call', start: '00:00', end: '23:59', duration: 24 }
  ];
}

function getDayOfWeek(date) {
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  return days[date.getDay()];
}

// ============================================
// MAIN POPULATION LOGIC
// ============================================

async function populateHospitalSystem() {
  console.log('\n' + '='.repeat(70));
  console.log('🏥 HOSPITAL SYSTEM POPULATION');
  console.log('='.repeat(70) + '\n');

  try {
    // ---- STEP 1: Register Hospital ----
    console.log('📋 STEP 1: Registering Hospital...');
    
    const hospitalExists = await Hospital.findOne({ name: 'St. Mercy Teaching Hospital' });
    let hospital;
    let manager;

    if (hospitalExists) {
      console.log('  ℹ Hospital already exists. Clearing old data...');
      hospital = hospitalExists;
      
      // Clear old staff and shifts
      await ClinicalStaff.deleteMany({ hospitalId: hospital._id });
      await Shift.deleteMany({ hospitalId: hospital._id });
      console.log('  ✓ Old data cleared');
    } else {
      const hashedPassword = await bcrypt.hash('admin123', 10);
      
      hospital = new Hospital({
        hospitalID: `HOSP-${Date.now()}`,
        name: 'St. Mercy Teaching Hospital',
        country: 'USA',
        city: 'San Francisco',
        state: 'CA',
        zipCode: '94107',
        phone: '(415) 555-4100',
        email: 'admin@stmercy.com',
        adminManagerName: 'John Administrator',
        adminManagerEmail: 'admin@stmercy.com',
        isActive: true,
        departments: Object.keys(DEPARTMENTS).map(deptKey => ({
          name: deptKey,
          isActive: true
        })),
        subscription: {
          plan: 'ENTERPRISE',
          maxManagers: 10,
          maxStaff: 500,
          maxPatients: 5000,
          startDate: new Date(),
          expiryDate: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000)
        }
      });

      await hospital.save();
      console.log('  ✓ Hospital created:', hospital.hospitalID);

      // Create admin manager
      manager = new Manager({
        hospitalId: hospital._id,
        managerID: `MGR-${Date.now()}`,
        name: 'John',
        surname: 'Administrator',
        email: 'admin@stmercy.com',
        password: hashedPassword,
        department: 'All',
        role: 'ADMIN',
        permissions: {
          canCreateShifts: true,
          canEditShifts: true,
          canDeleteShifts: true,
          canAssignStaff: true,
          canApproveLeaves: true,
          canManageRoles: true,
          canViewReports: true
        },
        isActive: true
      });

      await manager.save();
      hospital.adminManagerId = manager._id;
      await hospital.save();
      console.log('  ✓ Admin manager created');
    }

    // ---- STEP 2: Create Clinical Staff ----
    console.log('\n📊 STEP 2: Creating Clinical Staff...');

    const createdStaff = {};
    let totalStaffCount = 0;

    for (const [deptKey, deptConfig] of Object.entries(DEPARTMENTS)) {
      console.log(`\n  ${deptConfig.fullName}:`);
      createdStaff[deptKey] = { nurses: [], physicians: [] };

      // Create nurses
      for (let i = 0; i < deptConfig.nurses; i++) {
        const { firstName, lastName } = generateStaffName('NURSE', i, deptKey);
        const email = generateEmail(firstName, lastName, deptKey);
        const specialty = getSpecialtyForRole('NURSE', deptKey, i);

        const staff = new ClinicalStaff({
          hospitalId: hospital._id,
          firstName,
          lastName,
          email,
          password: await bcrypt.hash('nurse123', 10),
          systemRole: 'NURSE',
          clinicalSpecialty: specialty,
          credentials: {
            title: 'RN',
            licenseNumber: `RN-${deptKey}-${i + 1}`,
            npiNumber: `NPI${Math.random().toString().substr(2, 10)}`
          },
          assignment: {
            department: deptKey === 'Medical-Surgical' ? 'EMERGENCY' : deptKey.toUpperCase().replace(/[^A-Z]/g, ''),
            isOnCall: false
          },
          status: 'ACTIVE',
          metadata: { createdBy: hospital.adminManagerId }
        });

        // Use Map-style department mapping
        const deptMap = {
          'ED': 'EMERGENCY',
          'EMERGENCY': 'EMERGENCY',
          'CARDIOLOGY': 'CARDIOLOGY',
          'ICU': 'ICU',
          'OR': 'GENERAL_WARD',
          'OB': 'GENERAL_WARD',
          'MEDICAL-SURGICAL': 'GENERAL_WARD',
          'MEDICAL_SURGICAL': 'GENERAL_WARD',
          'PEDIATRICS': 'PEDIATRICS',
          'RADIOLOGY': 'RADIOLOGY',
          'GENERAL': 'GENERAL_WARD',
          'Pediatrics': 'PEDIATRICS',
          'Psychiatry': 'GENERAL_WARD'
        };

        staff.assignment.department = deptMap[deptKey] || 'GENERAL_WARD';

        await staff.save();
        createdStaff[deptKey].nurses.push(staff);
        totalStaffCount++;
      }

      // Create physicians
      for (let i = 0; i < deptConfig.physicians; i++) {
        const { firstName, lastName } = generateStaffName('PHYSICIAN', i, deptKey);
        const email = generateEmail(firstName, lastName, deptKey);
        const specialty = getSpecialtyForRole('PHYSICIAN', deptKey, i);

        const staff = new ClinicalStaff({
          hospitalId: hospital._id,
          firstName,
          lastName,
          email,
          password: await bcrypt.hash('physician123', 10),
          systemRole: 'PHYSICIAN',
          clinicalSpecialty: specialty,
          credentials: {
            title: 'MD',
            licenseNumber: `MD-${deptKey}-${i + 1}`,
            npiNumber: `NPI${Math.random().toString().substr(2, 10)}`
          },
          assignment: {
            department: deptKey === 'Medical-Surgical' ? 'EMERGENCY' : deptKey.toUpperCase().replace(/[^A-Z]/g, ''),
            isOnCall: true  // Physicians are typically on-call
          },
          status: 'ACTIVE',
          metadata: { createdBy: hospital.adminManagerId }
        });

        const deptMap = {
          'ED': 'EMERGENCY',
          'EMERGENCY': 'EMERGENCY',
          'CARDIOLOGY': 'CARDIOLOGY',
          'ICU': 'ICU',
          'OR': 'GENERAL_WARD',
          'OB': 'GENERAL_WARD',
          'MEDICAL-SURGICAL': 'GENERAL_WARD',
          'MEDICAL_SURGICAL': 'GENERAL_WARD',
          'PEDIATRICS': 'PEDIATRICS',
          'RADIOLOGY': 'RADIOLOGY',
          'GENERAL': 'GENERAL_WARD',
          'Pediatrics': 'PEDIATRICS',
          'Psychiatry': 'GENERAL_WARD'
        };

        staff.assignment.department = deptMap[deptKey] || 'GENERAL_WARD';

        await staff.save();
        createdStaff[deptKey].physicians.push(staff);
        totalStaffCount++;
      }

      console.log(`    ✓ ${deptConfig.nurses} nurses, ${deptConfig.physicians} physicians`);
    }

    console.log(`\n  Total staff created: ${totalStaffCount}`);

    // ---- STEP 3: Create Shifts with Team Assignments ----
    console.log('\n📅 STEP 3: Creating Shifts...');

    const shiftTimes = await generateShiftTimes();
    const baseDate = new Date();
    baseDate.setDate(baseDate.getDate() + 1); // Start from tomorrow
    baseDate.setHours(0, 0, 0, 0);

    let totalShiftsCount = 0;

    for (const [deptKey, deptConfig] of Object.entries(DEPARTMENTS)) {
      console.log(`\n  ${deptConfig.fullName}:`);

      // Create 7 days of shifts (1 week)
      for (let dayOffset = 0; dayOffset < 7; dayOffset++) {
        const shiftDate = new Date(baseDate);
        shiftDate.setDate(shiftDate.getDate() + dayOffset);
        const dayOfWeek = getDayOfWeek(shiftDate);

        // Create shifts for this day
        for (const shiftTime of shiftTimes.slice(0, 3)) { // 3 shifts per day (morning, evening, night)
          const shift = new Shift({
            hospitalId: hospital._id,
            shiftID: `SHIFT-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
            shiftName: `${deptConfig.name} ${shiftTime.name}`,
            department: deptKey,
            shiftStart: shiftTime.start,
            shiftEnd: shiftTime.end,
            duration: shiftTime.duration,
            date: shiftDate,
            dayOfWeek,
            recurrence: 'WEEKLY',
            recurrenceEndDate: new Date(shiftDate.getTime() + 90 * 24 * 60 * 60 * 1000), // 90 days
            maxStaffRequired: deptConfig.nurses + 1, // nurses + 1 physician
            status: 'PUBLISHED',
            createdBy: hospital.adminManagerId,
            notes: `Regular ${dayOfWeek} shift for ${deptConfig.fullName}`,
            
            // Team Assignment (KEY: for dashboard role assignment)
            teamAssignment: {
              triageNurseId: createdStaff[deptKey].nurses[0]?._id || null,
              clinicalNurseId: createdStaff[deptKey].nurses[1]?._id || null,
              clinicalNurseContext: deptKey === 'ED' ? 'CLINICAL_NURSE_ER' : 
                                    deptKey === 'ICU' ? 'CLINICAL_NURSE_ICU' : 
                                    'CLINICAL_NURSE_SURGICAL',
              physicianId: createdStaff[deptKey].physicians[0]?._id || null,
              physicianContext: deptKey === 'ED' ? 'PHYSICIAN_ER' :
                             deptKey === 'ICU' ? 'PHYSICIAN_ICU' :
                             'PHYSICIAN_SURGICAL',
              physicianSpecialty: deptConfig.specialties[0],
              teamAssignedAt: new Date(),
              assignedBy: hospital.adminManagerId
            },

            shiftContext: deptKey === 'ED' ? 'TRIAGE' : deptKey === 'OR' ? 'SURGICAL' : deptKey === 'ICU' ? 'ICU' : 'WARD',

            // Assign staff to shift
            assignedStaff: [
              ...createdStaff[deptKey].nurses.slice(0, deptConfig.nurses).map((nurse, idx) => ({
                staffID: nurse._id,
                type: 'NURSE',
                name: nurse.firstName,
                surname: nurse.lastName,
                specialty: nurse.clinicalSpecialty,
                status: 'ASSIGNED'
              })),
              {
                staffID: createdStaff[deptKey].physicians[0]?._id,
                type: 'PHYSICIAN',
                name: createdStaff[deptKey].physicians[0]?.firstName,
                surname: createdStaff[deptKey].physicians[0]?.lastName,
                specialty: createdStaff[deptKey].physicians[0]?.clinicalSpecialty,
                status: 'ASSIGNED'
              }
            ].filter(s => s.staffID) // Remove null assignments
          });

          await shift.save();
          totalShiftsCount++;
        }
      }

      console.log(`    ✓ ${7 * 3} shifts created (7 days × 3 shifts/day)`);
    }

    console.log(`\n  Total shifts created: ${totalShiftsCount}`);

    // ---- STEP 4: Create ESI_Engine Roster Query Function ----
    console.log('\n🔍 STEP 4: ESI_Engine Integration Summary');
    console.log('  ✓ Active physician roster is live in MongoDB');
    console.log('  ✓ Query template:');
    console.log('    ClinicalStaff.find({');
    console.log('      hospitalId: <hospitalId>,');
    console.log('      systemRole: "PHYSICIAN",');
    console.log('      assignment.isOnCall: true,');
    console.log('      status: "ACTIVE"');
    console.log('    })');

    // ---- SUMMARY ----
    console.log('\n' + '='.repeat(70));
    console.log('✅ POPULATION COMPLETE');
    console.log('='.repeat(70));
    console.log(`\nHospital: ${hospital.name}`);
    console.log(`Hospital ID: ${hospital._id}`);
    console.log(`Manager: admin@stmercy.com / admin123`);
    console.log(`\nStaff Created: ${totalStaffCount}`);
    console.log(`Shifts Created: ${totalShiftsCount}`);
    console.log(`\nDepartments: ${Object.keys(DEPARTMENTS).join(', ')}`);
    console.log('\nCredentials for Staff Login:');
    console.log('  Nurses: password = "nurse123"');
    console.log('  Physicians: password = "physician123"');
    console.log('\nStaff Email Format: firstname.lastname.dept@stmercy.com');
    console.log('\nNext Step: Update ESI_Engine to query live physician roster');
    console.log('           instead of using hardcoded physicians.csv\n');

  } catch (err) {
    console.error('❌ Population failed:', err);
    console.error(err.stack);
  } finally {
    await mongoose.connection.close();
    console.log('🔌 MongoDB disconnected\n');
  }
}

// ============================================
// RUN POPULATION
// ============================================

populateHospitalSystem().catch(console.error);
