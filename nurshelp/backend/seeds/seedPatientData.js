#!/usr/bin/env node
/**
 * Seed Script: Patient Data with Bed Assignments
 * Creates realistic patient encounters and assigns them to available beds
 * for physician management and monitoring
 */

import mongoose from "mongoose";
import dotenv from "dotenv";
import { Encounter } from "../src/models/encounter.model.js";
import PatientEHRRecord from "../src/models/patientEHRRecord.model.js";
import { BedSchedule } from "../src/models/bedSchedule.model.js";
import { Ward } from "../src/models/ward.model.js";

dotenv.config();

const MONGODB_URI = process.env.MONGO_URI || process.env.MONGODB_URI || "mongodb://localhost:27017/nurse_helpdesk";

// Sample realistic patient data
const SAMPLE_PATIENTS = [
  {
    name: "John",
    surname: "Miller",
    id_number: "P001",
    gender: "M",
    age: 65,
    height_cm: 180,
    symptoms: "Chest pain, shortness of breath",
    duration: "2 hours",
    painLevel: "8/10",
    history: "Hypertension, Type 2 Diabetes",
    esiLevel: 2,
    department: "ER",
    acuityScore: 85,
    vitals: {
      temp: 37.2,
      hr: 110,
      bp_systolic: 155,
      bp_diastolic: 95,
      resp: 22,
      o2: 94
    }
  },
  {
    name: "Maria",
    surname: "Garcia",
    id_number: "P002",
    gender: "F",
    age: 42,
    height_cm: 165,
    symptoms: "Severe abdominal pain, nausea, vomiting",
    duration: "6 hours",
    painLevel: "9/10",
    history: "Appendicitis family history",
    esiLevel: 2,
    department: "ER",
    acuityScore: 88,
    vitals: {
      temp: 38.5,
      hr: 105,
      bp_systolic: 140,
      bp_diastolic: 85,
      resp: 20,
      o2: 97
    }
  },
  {
    name: "Robert",
    surname: "Chen",
    id_number: "P003",
    gender: "M",
    age: 55,
    height_cm: 175,
    symptoms: "Acute stroke symptoms, left arm weakness",
    duration: "1 hour",
    painLevel: "N/A",
    history: "Atrial fibrillation, on warfarin",
    esiLevel: 1,
    department: "ER",
    acuityScore: 95,
    vitals: {
      temp: 36.8,
      hr: 98,
      bp_systolic: 165,
      bp_diastolic: 105,
      resp: 18,
      o2: 98
    }
  },
  {
    name: "Sarah",
    surname: "Johnson",
    id_number: "P004",
    gender: "F",
    age: 38,
    height_cm: 170,
    symptoms: "Asthma exacerbation, wheezing",
    duration: "4 hours",
    painLevel: "6/10",
    history: "Moderate persistent asthma, allergies",
    esiLevel: 3,
    department: "ER",
    acuityScore: 72,
    vitals: {
      temp: 37.0,
      hr: 115,
      bp_systolic: 130,
      bp_diastolic: 80,
      resp: 28,
      o2: 90
    }
  },
  {
    name: "David",
    surname: "Williams",
    id_number: "P005",
    gender: "M",
    age: 72,
    height_cm: 172,
    symptoms: "Fall with hip pain, unable to bear weight",
    duration: "30 minutes",
    painLevel: "8/10",
    history: "Osteoporosis, on blood thinners",
    esiLevel: 2,
    department: "ER",
    acuityScore: 82,
    vitals: {
      temp: 36.9,
      hr: 88,
      bp_systolic: 145,
      bp_diastolic: 88,
      resp: 16,
      o2: 97
    }
  },
  {
    name: "Lisa",
    surname: "Thompson",
    id_number: "P006",
    gender: "F",
    age: 48,
    height_cm: 168,
    symptoms: "Acute migraine, photophobia, nausea",
    duration: "3 hours",
    painLevel: "8/10",
    history: "Chronic migraines with aura",
    esiLevel: 3,
    department: "ER",
    acuityScore: 65,
    vitals: {
      temp: 37.1,
      hr: 102,
      bp_systolic: 135,
      bp_diastolic: 82,
      resp: 17,
      o2: 98
    }
  },
  {
    name: "Michael",
    surname: "Anderson",
    id_number: "P007",
    gender: "M",
    age: 51,
    height_cm: 178,
    symptoms: "Delirium, confusion, fever",
    duration: "12 hours",
    painLevel: "N/A",
    history: "COPD, recent hospitalization",
    esiLevel: 2,
    department: "ICU",
    acuityScore: 90,
    vitals: {
      temp: 39.5,
      hr: 120,
      bp_systolic: 175,
      bp_diastolic: 105,
      resp: 25,
      o2: 87
    }
  },
  {
    name: "Jennifer",
    surname: "Martinez",
    id_number: "P008",
    gender: "F",
    age: 29,
    height_cm: 162,
    symptoms: "Laceration on left forearm, heavy bleeding",
    duration: "20 minutes",
    painLevel: "7/10",
    history: "No significant past medical history",
    esiLevel: 3,
    department: "ER",
    acuityScore: 68,
    vitals: {
      temp: 36.7,
      hr: 105,
      bp_systolic: 128,
      bp_diastolic: 78,
      resp: 18,
      o2: 99
    }
  }
];

async function seedPatientData() {
  try {
    console.log("🔌 Connecting to MongoDB...");
    await mongoose.connect(MONGODB_URI);
    console.log("✅ Connected to MongoDB");

    // Clear existing data (optional - comment out to preserve)
    console.log("\n🧹 Clearing existing patient data...");
    await Encounter.deleteMany({});
    await PatientEHRRecord.deleteMany({});
    console.log("✅ Cleared existing data");

    // Get available beds or create them
    console.log("\n🛏️  Setting up beds...");
    let beds = await BedSchedule.find({ status: "AVAILABLE" }).limit(SAMPLE_PATIENTS.length);
    
    if (beds.length < SAMPLE_PATIENTS.length) {
      console.log(`⚠️  Found ${beds.length} available beds, creating more...`);
      
      // Get or create ER ward
      let erWard = await Ward.findOne({ name: "Emergency Room" });
      if (!erWard) {
        console.log("Creating ER ward...");
        erWard = await Ward.create({
          name: "Emergency Room",
          abbreviation: "ER",
          hospitalId: null,
          type: "EMERGENCY",
          location: "Main Building, Level 1",
          totalBeds: 12,
          occupiedBeds: 0,
          availableBeds: 12,
          cleaningBeds: 0,
          maintenanceBeds: 0
        });
      }

      // Get or create ICU ward
      let icuWard = await Ward.findOne({ name: "Intensive Care Unit" });
      if (!icuWard) {
        console.log("Creating ICU ward...");
        icuWard = await Ward.create({
          name: "Intensive Care Unit",
          abbreviation: "ICU",
          hospitalId: null,
          type: "ICU",
          location: "Main Building, Level 3",
          totalBeds: 6,
          occupiedBeds: 0,
          availableBeds: 6,
          cleaningBeds: 0,
          maintenanceBeds: 0
        });
      }

      // Create beds for ER
      const erBeds = [];
      for (let i = 1; i <= 8; i++) {
        erBeds.push({
          wardId: erWard._id,
          bedId: `ER-BED-${String(i).padStart(2, "0")}`,
          roomNumber: `ER-${i}`,
          department: "ER",
          bedType: "MONITORED",
          status: "AVAILABLE",
          hasMonitor: true,
          hasVentilator: false,
          canIsolate: false,
          acuityLevel: "MODERATE"
        });
      }

      // Create beds for ICU
      const icuBeds = [];
      for (let i = 1; i <= 4; i++) {
        icuBeds.push({
          wardId: icuWard._id,
          bedId: `ICU-BED-${String(i).padStart(2, "0")}`,
          roomNumber: `ICU-${i}`,
          department: "ICU",
          bedType: "INTENSIVE",
          status: "AVAILABLE",
          hasMonitor: true,
          hasVentilator: true,
          canIsolate: true,
          acuityLevel: "CRITICAL"
        });
      }

      const createdBeds = await BedSchedule.insertMany([...erBeds, ...icuBeds]);
      beds.push(...createdBeds);
      console.log(`✅ Created ${createdBeds.length} new beds`);
    }

    // Create patient encounters and EHR records
    console.log("\n👥 Creating patient encounters and assigning to beds...");
    const createdEncounters = [];
    const createdEHRs = [];

    for (let i = 0; i < SAMPLE_PATIENTS.length; i++) {
      const patientData = SAMPLE_PATIENTS[i];
      const bed = beds[i];

      if (!bed) {
        console.warn(`⚠️  No bed available for patient ${patientData.name} ${patientData.surname}`);
        continue;
      }

      // Create encounter
      const encounter = await Encounter.create({
        patient: {
          id_number: patientData.id_number,
          name: patientData.name,
          surname: patientData.surname,
          gender: patientData.gender,
          age: patientData.age,
          height_cm: patientData.height_cm,
          symptoms: patientData.symptoms,
          duration: patientData.duration,
          painLevel: patientData.painLevel,
          history: patientData.history
        },
        vitals: patientData.vitals,
        status: "confirmed",
        severity: patientData.esiLevel,
        submittedAt: new Date(),
        attendedAt: new Date()
      });

      // Create EHR record
      const ehrRecord = await PatientEHRRecord.create({
        encounterId: encounter._id.toString(),
        patientId: patientData.id_number,
        name: patientData.name,
        surname: patientData.surname,
        age: patientData.age,
        gender: patientData.gender,
        mrn: `MRN-${patientData.id_number}`,
        currentBedId: bed.bedId,
        currentDepartment: patientData.department,
        admissionTime: new Date(),
        esiLevel: patientData.esiLevel,
        acuityScore: patientData.acuityScore,
        latestVitals: {
          temperature: patientData.vitals.temp,
          heartRate: patientData.vitals.hr,
          systolicBP: patientData.vitals.bp_systolic,
          diastolicBP: patientData.vitals.bp_diastolic,
          respiratoryRate: patientData.vitals.resp,
          o2Saturation: patientData.vitals.o2,
          recordedAt: new Date()
        },
        status: "IN_PROGRESS"
      });

      // Update bed to OCCUPIED
      await BedSchedule.findByIdAndUpdate(bed._id, {
        status: "OCCUPIED",
        currentEncounterId: encounter._id,
        occupancyStartTime: new Date(),
        statusChangedAt: new Date()
      });

      createdEncounters.push(encounter);
      createdEHRs.push(ehrRecord);

      console.log(
        `✅ Created: ${patientData.name} ${patientData.surname} → Bed ${bed.bedId} (ESI ${patientData.esiLevel})`
      );
    }

    // Summary
    console.log("\n📊 Seeding Complete!");
    console.log(`✅ Created ${createdEncounters.length} encounters`);
    console.log(`✅ Created ${createdEHRs.length} EHR records`);
    console.log(`✅ Assigned ${createdEncounters.length} patients to available beds`);
    console.log("\n🎯 Patients are now ready for physician management!");

    await mongoose.connection.close();
    console.log("\n✅ Done!");
  } catch (error) {
    console.error("❌ Error seeding data:", error);
    process.exit(1);
  }
}

seedPatientData();
