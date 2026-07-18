/**
 * Available Beds Seeding Script
 * 
 * Populates beds for all wards with AVAILABLE status
 * Ready for patient assignment
 */

import mongoose from "mongoose";
import dotenv from "dotenv";
import { Hospital } from "../models/hospital.model.js";
import { Ward } from "../models/ward.model.js";
import { BedSchedule } from "../models/bedSchedule.model.js";

dotenv.config();

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/nursehelpdesk";

const BED_TEMPLATES = {
  ED: [
    { roomNumber: "ED-01", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "ED-02", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "ED-03", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "ED-04", bedType: "MONITORED", features: { hasMonitor: true, hasVentilator: false } },
    { roomNumber: "ED-05", bedType: "MONITORED", features: { hasMonitor: true, hasVentilator: false } },
    { roomNumber: "ED-RESUS-01", bedType: "INTENSIVE", features: { hasMonitor: true, hasVentilator: true } },
    { roomNumber: "ED-TRAUMA-01", bedType: "INTENSIVE", features: { hasMonitor: true, hasVentilator: true } },
  ],
  ICU: [
    { roomNumber: "ICU-01", bedType: "INTENSIVE", features: { hasMonitor: true, hasVentilator: true, hasDialysis: false } },
    { roomNumber: "ICU-02", bedType: "INTENSIVE", features: { hasMonitor: true, hasVentilator: true, hasDialysis: false } },
    { roomNumber: "ICU-03", bedType: "INTENSIVE", features: { hasMonitor: true, hasVentilator: true, hasDialysis: true } },
    { roomNumber: "ICU-04", bedType: "INTENSIVE", features: { hasMonitor: true, hasVentilator: true, hasDialysis: true } },
    { roomNumber: "ICU-05", bedType: "HIGH_DEPENDENCY", features: { hasMonitor: true, hasVentilator: false } },
    { roomNumber: "ICU-06", bedType: "HIGH_DEPENDENCY", features: { hasMonitor: true, hasVentilator: false } },
  ],
  GENERAL: [
    { roomNumber: "GEN-01", bedType: "STANDARD", features: { hasMonitor: false } },
    { roomNumber: "GEN-02", bedType: "STANDARD", features: { hasMonitor: false } },
    { roomNumber: "GEN-03", bedType: "STANDARD", features: { hasMonitor: false } },
    { roomNumber: "GEN-04", bedType: "STANDARD", features: { hasMonitor: false } },
    { roomNumber: "GEN-05", bedType: "MONITORED", features: { hasMonitor: true } },
    { roomNumber: "GEN-06", bedType: "MONITORED", features: { hasMonitor: true } },
    { roomNumber: "GEN-ISO-01", bedType: "ISOLATION", features: { hasMonitor: true, canIsolate: true } },
  ],
  PEDIATRICS: [
    { roomNumber: "PED-01", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "PED-02", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "PED-03", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "PED-04", bedType: "MONITORED", features: { hasMonitor: true } },
    { roomNumber: "PED-ICU-01", bedType: "INTENSIVE", features: { hasMonitor: true, hasVentilator: true } },
  ],
  MATERNITY: [
    { roomNumber: "MAT-LABOR-01", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "MAT-LABOR-02", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "MAT-DELIVERY-01", bedType: "STANDARD", features: { hasMonitor: true } },
    { roomNumber: "MAT-POSTPARTUM-01", bedType: "STANDARD", features: { hasMonitor: false } },
    { roomNumber: "MAT-POSTPARTUM-02", bedType: "STANDARD", features: { hasMonitor: false } },
  ],
};

async function seedBeds() {
  try {
    console.log("🔄 Connecting to MongoDB...");
    await mongoose.connect(MONGO_URI);
    console.log("✅ Connected");

    // Get hospital
    const hospital = await Hospital.findOne();
    if (!hospital) {
      console.error("❌ No hospital found! Run seedMultiHospitalSetup.js first");
      process.exit(1);
    }
    console.log(`\n🏥 Using Hospital: ${hospital.name}`);

    // Get all wards or create them
    let wards = await Ward.find({ hospitalId: hospital._id });
    
    if (wards.length === 0) {
      console.log("\n🏢 No wards found. Creating default wards...");
      
      const wardTemplates = [
        { wardName: "Emergency Department - Ward A", wardId: "ED-Ward-A", department: "ED", floor: 1, wing: "Main", bedCount: 7 },
        { wardName: "Intensive Care Unit - Ward A", wardId: "ICU-Ward-A", department: "ICU", floor: 2, wing: "Main", bedCount: 6 },
        { wardName: "General Ward - Ward A", wardId: "GEN-Ward-A", department: "GENERAL", floor: 3, wing: "Main", bedCount: 7 },
        { wardName: "Pediatrics - Ward A", wardId: "PED-Ward-A", department: "PEDIATRICS", floor: 2, wing: "East", bedCount: 5 },
        { wardName: "Maternity - Ward A", wardId: "MAT-Ward-A", department: "MATERNITY", floor: 3, wing: "East", bedCount: 5 },
      ];
      
      for (const template of wardTemplates) {
        const newWard = new Ward({
          wardId: template.wardId,
          wardName: template.wardName,
          department: template.department,
          hospitalId: hospital._id,
          location: {
            floor: template.floor,
            wing: template.wing,
            area: ""
          },
          totalBeds: template.bedCount,
          occupiedBeds: 0,
          availableBeds: 0,
          cleaningBeds: 0,
          maintenanceBeds: 0,
          isActive: true
        });
        await newWard.save();
        console.log(`  ✓ Created ward: ${newWard.wardName}`);
      }
      
      wards = await Ward.find({ hospitalId: hospital._id });
    }
    
    console.log(`\n🏢 Found ${wards.length} wards`);

    // Clear existing beds
    console.log("\n🧹 Clearing existing beds...");
    await BedSchedule.deleteMany({});

    let totalBedsCreated = 0;

    // Create beds for each ward
    for (const ward of wards) {
      const bedTemplates = BED_TEMPLATES[ward.department] || BED_TEMPLATES.GENERAL;
      console.log(`\n📋 Creating beds for ${ward.wardName} (${ward.department})...`);

      const bedsForWard = [];

      for (let i = 0; i < bedTemplates.length; i++) {
        const template = bedTemplates[i];
        const bedId = `${ward.wardId}-${String(i + 1).padStart(2, "0")}`;

        const bed = new BedSchedule({
          wardId: ward._id,
          bedId: bedId,
          roomNumber: template.roomNumber,
          department: ward.department,
          bedType: template.bedType,
          status: "AVAILABLE", // ✅ ALL BEDS AVAILABLE
          statusChangedAt: new Date(),
          currentEncounterId: null,
          assignedNurseId: null,
          occupancyStartTime: null,
          expectedDischargeTime: null,
          acuityLevel: "MODERATE",
          hasMonitor: template.features.hasMonitor || false,
          hasVentilator: template.features.hasVentilator || false,
          hasDialysis: template.features.hasDialysis || false,
          canIsolate: template.features.canIsolate || false,
        });

        await bed.save();
        bedsForWard.push(bed._id);
        totalBedsCreated++;

        console.log(`  ✓ Created bed: ${template.roomNumber} (${template.bedType})`);
      }

      // Update ward with bed references
      ward.beds = bedsForWard;
      ward.totalBeds = bedsForWard.length;
      ward.availableBeds = bedsForWard.length;
      ward.occupiedBeds = 0;
      ward.cleaningBeds = 0;
      ward.maintenanceBeds = 0;
      await ward.save();

      console.log(`  ✅ Ward updated: ${bedsForWard.length} beds linked`);
    }

    console.log(`\n✨ Successfully created ${totalBedsCreated} available beds across all wards`);
    console.log(`\n📊 Summary:`);
    console.log(`  - Hospital: ${hospital.name}`);
    console.log(`  - Wards: ${wards.length}`);
    console.log(`  - Total Beds: ${totalBedsCreated}`);
    console.log(`  - All Status: AVAILABLE ✅`);

    await mongoose.disconnect();
    console.log("\n✅ Database seeding complete!");
    process.exit(0);

  } catch (error) {
    console.error("❌ Seeding error:", error);
    process.exit(1);
  }
}

seedBeds();
