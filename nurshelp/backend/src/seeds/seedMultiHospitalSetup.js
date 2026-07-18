/**
 * Multi-Hospital Setup Seed Script
 * 
 * Creates:
 * - 1 Hospital (St. Mary's Medical Center)
 * - 1 Admin Manager with a shift
 * - Multiple Nurses/Physicians with shifts
 * - All staff are assigned shifts (no self-selection)
 * 
 * Staff can ONLY:
 * - Login with credentials
 * - Clock in/out
 * - Request breaks (1 hour pre-defined)
 * 
 * Staff CANNOT:
 * - Change their shifts
 * - Create shifts
 * - Modify their role/department
 */

import mongoose from "mongoose";
import bcrypt from "bcrypt";
import dotenv from "dotenv";
import { Hospital } from "../models/hospital.model.js";
import { Manager } from "../models/manager.model.js";
import { ClinicalStaff } from "../models/clinicalstaff.model.js";
import { Shift } from "../models/shift.model.js";
import { NurseClockRecord } from "../models/staffClockRecord.model.js";

dotenv.config();

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/nursehelpdesk";

async function seedDatabase() {
  try {
    console.log("🔄 Connecting to MongoDB...");
    await mongoose.connect(MONGO_URI);
    console.log("✅ Connected");

    // ============================================
    // CLEAR ALL DATA
    // ============================================
    console.log("\n🧹 Clearing all data...");
    
    const collections = [
      "hospitals",
      "managers",
      "nurses",
      "shifts",
      "nurseClockRecords",
      "nurseclockrecords",
      "patients",
      "patientehrrecords",
      "encounters",
      "triagerequests",
      "wardassignments"
    ];

    for (const collectionName of collections) {
      try {
        await mongoose.connection.collection(collectionName).deleteMany({});
        console.log(`  ✓ Cleared ${collectionName}`);
      } catch (err) {
        // Collection might not exist, that's fine
      }
    }

    // ============================================
    // CREATE ADMIN MANAGER (FIRST - Hospital needs this)
    // ============================================
    console.log("\n👨‍💼 Creating Hospital Admin Manager...");

    // Create a temporary hospital ID for manager reference
    const tempHospitalId = new mongoose.Types.ObjectId();

    const hashedAdminPassword = await bcrypt.hash("admin123", 10);
    
    const adminManager = new Manager({
      hospitalId: tempHospitalId,
      managerID: `MGR-${Date.now()}`,
      name: "John",
      surname: "Smith",
      email: "john.smith@stmarys.hospital",
      password: hashedAdminPassword,
      department: "All",
      role: "ADMIN",
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
    });

    const adminManagerDoc = await adminManager.save();
    console.log(`✓ Admin Manager created: ${adminManagerDoc.name} (${adminManagerDoc.email})`);

    // ============================================
    // CREATE HOSPITAL
    // ============================================
    console.log("\n🏥 Creating Hospital...");

    const hospital = new Hospital({
      hospitalID: `HOSP-${Date.now()}`,
      name: "St. Mary's Medical Center",
      country: "South Africa",
      city: "Cape Town",
      state: "Western Cape",
      zipCode: "8000",
      phone: "+27 21 917 3000",
      email: "admin@stmarys.hospital",
      adminManagerId: adminManagerDoc._id,
      adminManagerName: adminManagerDoc.name,
      adminManagerEmail: adminManagerDoc.email,
      departments: [
        { name: "ED", isActive: true },
        { name: "ICU", isActive: true },
        { name: "OR", isActive: true },
        { name: "OB", isActive: true },
      ],
      isActive: true,
    });

    const hospitalDoc = await hospital.save();
    console.log(`✓ Hospital created: ${hospitalDoc.name}`);

    // Update manager's hospitalId to reference the actual hospital
    adminManagerDoc.hospitalId = hospitalDoc._id;
    await adminManagerDoc.save();
    console.log(`✓ Updated manager's hospitalId reference`);

    // ============================================
    // CREATE SHIFTS FOR STAFF
    // ============================================
    console.log("\n📅 Creating Shifts...");

    // ============================================
    // CREATE NURSES WITH SHIFTS
    // ============================================
    console.log("\n👩‍⚕️ Creating Nurses...");

    const nurseData = [
      { name: "Sarah", surname: "Johnson", specialty: "Emergency Medicine", department: "ED" },
      { name: "Emily", surname: "Williams", specialty: "Critical Care", department: "ICU" },
      { name: "Jessica", surname: "Brown", specialty: "Surgical Nursing", department: "OR" },
      { name: "Amanda", surname: "Davis", specialty: "Obstetric Nursing", department: "OB" },
      { name: "Lisa", surname: "Rodriguez", specialty: "Emergency Medicine", department: "ED" },
    ];

    const nurses = [];
    const shiftTimes = [
      { start: "08:00", end: "16:00", name: "Day" },
      { start: "16:00", end: "00:00", name: "Evening" },
      { start: "00:00", end: "08:00", name: "Night" },
    ];

    for (let i = 0; i < nurseData.length; i++) {
      const nData = nurseData[i];
      const email = `${nData.name.toLowerCase()}.${nData.surname.toLowerCase()}@stmarys.hospital`;

      const nurse = new ClinicalStaff({
        hospitalId: hospitalDoc._id,
        name: nData.name,
        surname: nData.surname,
        email,
        password: "nurse123",
        role: "NURSE",
        department: nData.department,
        specialty: nData.specialty,
        licenseNumber: `LIC-${1000 + i}`,
        isActive: true,
        createdBy: adminManagerDoc._id,
      });

      const nurseDoc = await nurse.save();
      nurses.push(nurseDoc);
      console.log(`  ✓ Created: ${nurseDoc.name} ${nurseDoc.surname} (${nurseDoc.department})`);

      // Create shift for this nurse
      const shiftTime = shiftTimes[i % shiftTimes.length];
      const nurseShift = new Shift({
        hospitalId: hospitalDoc._id,
        shiftID: `SHIFT-${nurseDoc._id}`,
        shiftName: `${nData.name}'s ${shiftTime.name} Shift`,
        department: nData.department,
        shiftStart: shiftTime.start,
        shiftEnd: shiftTime.end,
        duration: 8,
        date: new Date(),
        recurrence: "WEEKLY",
        recurrenceEndDate: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
        maxStaffRequired: 1,
        status: "PUBLISHED",
        createdBy: adminManagerDoc._id,
        assignedStaff: [
          {
            staffID: nurseDoc._id,
            type: "NURSE",
            name: nurseDoc.name,
            surname: nurseDoc.surname,
            specialty: nurseDoc.specialty,
            status: "ASSIGNED",
          },
        ],
      });

      await nurseShift.save();
      console.log(`    ✓ Shift assigned: ${shiftTime.name} (${shiftTime.start}-${shiftTime.end})`);
    }

    // ============================================
    // CREATE PHYSICIANS
    // ============================================
    console.log("\n👨‍⚕️ Creating Physicians...");

    const physicianData = [
      { name: "Dr. Michael", surname: "Johnson", specialty: "Emergency Medicine", department: "ED" },
      { name: "Dr. Robert", surname: "Williams", specialty: "Critical Care", department: "ICU" },
      { name: "Dr. James", surname: "Brown", specialty: "Surgery", department: "OR" },
    ];

    for (let i = 0; i < physicianData.length; i++) {
      const pData = physicianData[i];
      const email = `${pData.name.toLowerCase().replace(" ", ".")}.${pData.surname.toLowerCase()}@stmarys.hospital`;

      const physician = new ClinicalStaff({
        hospitalId: hospitalDoc._id,
        name: pData.name,
        surname: pData.surname,
        email,
        password: "physician123",
        role: "PHYSICIAN",
        department: pData.department,
        specialty: pData.specialty,
        licenseNumber: `LIC-MD-${2000 + i}`,
        isActive: true,
        createdBy: adminManagerDoc._id,
      });

      const physicianDoc = await physician.save();
      nurses.push(physicianDoc);
      console.log(`  ✓ Created: ${physicianDoc.name} ${physicianDoc.surname} (${physicianDoc.department})`);

      // Create shift
      const physicianShift = new Shift({
        hospitalId: hospitalDoc._id,
        shiftID: `SHIFT-${physicianDoc._id}`,
        shiftName: `${pData.name}'s Day Shift`,
        department: pData.department,
        shiftStart: "08:00",
        shiftEnd: "17:00",
        duration: 9,
        date: new Date(),
        recurrence: "WEEKLY",
        recurrenceEndDate: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
        maxStaffRequired: 1,
        status: "PUBLISHED",
        createdBy: adminManagerDoc._id,
        assignedStaff: [
          {
            staffID: physicianDoc._id,
            type: "PHYSICIAN",
            name: physicianDoc.name,
            surname: physicianDoc.surname,
            specialty: physicianDoc.specialty,
            status: "ASSIGNED",
          },
        ],
      });

      await physicianShift.save();
      console.log(`    ✓ Shift assigned: Day (08:00-17:00)`);
    }

    // ============================================
    // SUMMARY
    // ============================================
    console.log("\n" + "=".repeat(60));
    console.log("✅ DATABASE SEEDING COMPLETED SUCCESSFULLY");
    console.log("=".repeat(60));

    console.log("\n📋 Hospital Setup:");
    console.log(`  Hospital: ${hospitalDoc.name}`);
    console.log(`  Hospital ID: ${hospitalDoc._id}`);

    console.log("\n👨‍💼 Admin Manager:");
    console.log(`  Name: ${adminManagerDoc.name} ${adminManagerDoc.surname}`);
    console.log(`  Email: ${adminManagerDoc.email}`);
    console.log(`  Password: admin123`);
    console.log(`  Role: ADMIN (can create shifts)`);

    console.log("\n👩‍⚕️ Nurses (5 created):");
    nurseData.forEach((nData, idx) => {
      console.log(`  ${idx + 1}. ${nData.name} ${nData.surname}`);
      console.log(`     Email: ${nData.name.toLowerCase()}.${nData.surname.toLowerCase()}@stmarys.hospital`);
      console.log(`     Password: nurse123`);
      console.log(`     Department: ${nData.department}`);
      console.log(`     Status: CAN ONLY CLOCK IN/OUT/BREAK`);
    });

    console.log("\n👨‍⚕️ Physicians (3 created):");
    physicianData.forEach((pData, idx) => {
      console.log(`  ${idx + 1}. ${pData.name} ${pData.surname}`);
      console.log(`     Email: ${pData.name.toLowerCase().replace(" ", ".")}.${pData.surname.toLowerCase()}@stmarys.hospital`);
      console.log(`     Password: physician123`);
      console.log(`     Department: ${pData.department}`);
      console.log(`     Status: CAN ONLY CLOCK IN/OUT/BREAK`);
    });

    console.log("\n📅 Shift Management:");
    console.log("  ✓ Manager creates and manages all shifts");
    console.log("  ✓ Each nurse has an assigned shift (rotating)");
    console.log("  ✓ Each physician has a Day shift (08:00-17:00)");
    console.log("  ✓ All shifts are PUBLISHED - can be clocked in");

    console.log("\n🔒 Staff Permissions:");
    console.log("  ✓ Staff can LOGIN with their credentials");
    console.log("  ✓ Staff can CLOCK IN to their assigned shift");
    console.log("  ✓ Staff can CLOCK OUT from their shift");
    console.log("  ✓ Staff can REQUEST BREAK (1 hour pre-defined)");
    console.log("  ✗ Staff CANNOT create/change shifts");
    console.log("  ✗ Staff CANNOT modify their department");
    console.log("  ✗ Staff CANNOT approve leaves");

    await mongoose.connection.close();
    console.log("\n✅ Database seeding completed and connection closed");
    process.exit(0);
  } catch (err) {
    console.error("❌ Seeding error:", err.message);
    console.error(err);
    process.exit(1);
  }
}

seedDatabase();
