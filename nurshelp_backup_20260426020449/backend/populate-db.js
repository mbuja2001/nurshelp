/**
 * Database Population Script
 * Directly populates MongoDB with test data for physician dashboard
 */

import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Hospital } from './src/models/hospital.model.js';
import { ClinicalStaff } from './src/models/clinicalstaff.model.js';
import { TriageAssessment } from './src/models/triageAssessment.model.js';
import { Disposition } from './src/models/disposition.model.js';
import { PatientJourney } from './src/models/patientJourney.model.js';
import { Ward } from './src/models/ward.model.js';

dotenv.config();

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

async function populateDatabase() {
  try {
    console.log('🔌 Connecting to MongoDB...');
    await mongoose.connect(MONGO_URI);
    console.log('✅ Connected to MongoDB\n');

    // 1. Find Existing Hospital
    console.log('📋 Finding Hospital...');
    let hospital = await Hospital.findOne({ name: 'St Mercy Hospital' });
    if (!hospital) {
      console.error('❌ Hospital "St Mercy Hospital" not found in database');
      throw new Error('Hospital not found');
    }
    console.log('✅ Hospital found:', hospital.name, '(' + hospital._id + ')');

    // 2. Find Existing Manager and Physician
    console.log('\n👨‍⚕️ Finding Staff...');

    // Find existing manager
    const { Manager } = await import('./src/models/manager.model.js');
    let manager = await Manager.findOne({ email: 'alice.johnson@stmercy.hospital' });
    if (!manager) {
      console.error('❌ Manager alice.johnson@stmercy.hospital not found');
      throw new Error('Manager not found');
    }
    console.log('✅ Manager found:', manager.email);

    // Find existing physician
    let physician = await ClinicalStaff.findOne({ email: 'dr.chen@stmercy.hospital' });
    if (!physician) {
      console.error('❌ Physician dr.chen@stmercy.hospital not found');
      throw new Error('Physician not found');
    }
    console.log('✅ Physician found:', physician.email);

    // Find or create triage nurse
    let triageNurse = await ClinicalStaff.findOne({ email: 'nurse.triage@stmercy.hospital' });
    if (!triageNurse) {
      triageNurse = await ClinicalStaff.create({
        hospitalId: hospital._id,
        name: 'Sarah',
        surname: 'Johnson',
        email: 'nurse.triage@stmercy.hospital',
        password: 'nurse123',
        role: 'NURSE',
        licenseNumber: 'RN-001',
        createdBy: manager._id,
        department: 'ED',
      });
      console.log('✅ Triage nurse created:', triageNurse.email);
    } else {
      console.log('✅ Triage nurse exists:', triageNurse.email);
    }

    // 3. Create Test Patients & Triages
    console.log('\n🏥 Creating Triages...');
    const patientData = [
      { id: 1, firstName: 'John', lastName: 'Doe', age: 45, gender: 'M', priority: 'RED', complaint: 'Chest pain', needsBed: true },
      { id: 2, firstName: 'Sarah', lastName: 'Smith', age: 32, gender: 'F', priority: 'GREEN', complaint: 'Ankle sprain', needsBed: false },
      { id: 3, firstName: 'Michael', lastName: 'Jones', age: 55, gender: 'M', priority: 'YELLOW', complaint: 'Viral fever', needsBed: false },
      { id: 4, firstName: 'Maria', lastName: 'Garcia', age: 72, gender: 'F', priority: 'YELLOW', complaint: 'Head trauma', needsBed: true },
    ];

    const triages = [];
    const patientIds = [];

    for (const pData of patientData) {
      const patientId = new mongoose.Types.ObjectId();
      patientIds.push(patientId);

      let triage = await TriageAssessment.findOne({
        hospitalId: hospital._id,
        firstName: pData.firstName,
        lastName: pData.lastName,
      });

      if (!triage) {
        triage = await TriageAssessment.create({
          hospitalId: hospital._id,
          patientId,
          triageNurseId: triageNurse._id,
          firstName: pData.firstName,
          lastName: pData.lastName,
          age: pData.age,
          gender: pData.gender,
          chiefComplaint: pData.complaint,
          vitalSigns: {
            heartRate: 70 + Math.random() * 40,
            systolic: 110 + Math.random() * 50,
            diastolic: 70 + Math.random() * 30,
            temperature: 36.5 + Math.random() * 2,
            respiratoryRate: 14 + Math.random() * 10,
          },
          triagePriority: pData.priority,
          needsBed: pData.needsBed,
          status: pData.needsBed ? 'AWAITING_PHYSICIAN_DECISION' : 'IN_OBSERVATION',
          triageCompletedAt: new Date(),
        });
        console.log(`✅ Triage created: ${pData.firstName} ${pData.lastName} (${pData.priority})`);
      } else {
        console.log(`✅ Triage already exists: ${pData.firstName} ${pData.lastName}`);
      }
      triages.push(triage);
    }

    // 4. Create Dispositions
    console.log('\n📝 Creating Dispositions...');
    const dispositionData = [
      { patientIdx: 0, decision: 'ADMITTED', discrepancy: 'TRUE_POSITIVE' },
      { patientIdx: 1, decision: 'DISCHARGED', discrepancy: 'TRUE_NEGATIVE' },
      { patientIdx: 2, decision: 'DISCHARGED', discrepancy: 'TRUE_NEGATIVE' },
      { patientIdx: 3, decision: 'ADMITTED', discrepancy: 'FALSE_NEGATIVE' },
    ];

    for (const dData of dispositionData) {
      let disposition = await Disposition.findOne({
        patientId: patientIds[dData.patientIdx],
        hospitalId: hospital._id,
      });

      if (!disposition) {
        disposition = await Disposition.create({
          hospitalId: hospital._id,
          patientId: patientIds[dData.patientIdx],
          triageAssessmentId: triages[dData.patientIdx]._id,
          physicianId: physician._id,
          triageRecommendation: !!triages[dData.patientIdx].needsBed,
          decision: dData.decision,
          discrepancy: {
            hasDiscrepancy: dData.discrepancy !== 'TRUE_POSITIVE' && dData.discrepancy !== 'TRUE_NEGATIVE',
            type: dData.discrepancy,
            reason: dData.discrepancy === 'FALSE_NEGATIVE' ? 'Missed warfarin and confusion risk' : 'Match',
            severity: dData.discrepancy.startsWith('FALSE') ? 'HIGH' : 'LOW',
          },
          status: 'COMPLETED',
          decisionMadeAt: new Date(),
        });
        console.log(`✅ Disposition: ${patientData[dData.patientIdx].firstName} → ${dData.decision}`);
      } else {
        console.log(`✅ Disposition already exists for ${patientData[dData.patientIdx].firstName}`);
      }
    }

    // 5. Create Patient Journeys
    console.log('\n🛣️ Creating Patient Journeys...');
    const journeyData = [
      { patientIdx: 0, state: 'OBSERVATION' },
      { patientIdx: 1, state: 'TRIAGED' },
      { patientIdx: 2, state: 'ESCALATED' },
      { patientIdx: 3, state: 'OBSERVATION' },
    ];

    for (const jData of journeyData) {
      let journey = await PatientJourney.findOne({
        patientId: patientIds[jData.patientIdx],
        hospitalId: hospital._id,
      });

      if (!journey) {
        journey = await PatientJourney.create({
          hospitalId: hospital._id,
          patientId: patientIds[jData.patientIdx],
          currentState: jData.state,
          currentLocation: {
            locationType: jData.patientIdx < 2 ? 'OBSERVATION_ROOM' : 'TRIAGE_DESK',
            departmentContext: 'ER',
            observationRoomNumber: jData.patientIdx < 2 ? `OBS_${jData.patientIdx + 1}` : null,
          },
          arrivedAt: new Date(Date.now() - 30 * 60000),
          triageCompletedAt: new Date(Date.now() - 20 * 60000),
          triageAssessmentId: triages[jData.patientIdx]._id,
        });
        console.log(`✅ Journey created: ${patientData[jData.patientIdx].firstName} (${jData.state})`);
      } else {
        console.log(`✅ Journey already exists for ${patientData[jData.patientIdx].firstName}`);
      }
    }

    // 6. Create Ward and BedSchedule entries
    console.log('\n🛏️ Creating Ward with Beds...');
    const { BedSchedule } = await import('./src/models/bedSchedule.model.js');
    
    // Try to find any ED ward with beds
    let ward = null;
    let bedIds = [];
    
    const edWards = await Ward.find({ wardName: { $regex: 'Emergency', $options: 'i' } }).sort({ createdAt: -1 });
    for (const w of edWards) {
      const beds = await BedSchedule.find({ wardId: w._id });
      if (beds.length > 0) {
        ward = w;
        bedIds = beds.map(b => b._id);
        console.log(`✅ Found ED ward with ${beds.length} beds`);
        break;
      }
    }
    
    // If no ED ward with beds found, create new one
    if (!ward) {
      ward = await Ward.create({
        wardId: `ED-WARD-${Date.now()}`,
        wardName: 'Emergency Department',
        department: 'ED',
        hospitalId: hospital._id,
        location: { floor: 1, wing: 'Main' },
        totalBeds: 9,
        occupiedBeds: 2,
        availableBeds: 7,
      });

      bedIds = [];

      // ICU beds (5)
      for (let i = 1; i <= 5; i++) {
        const bedId = `ED-ICU-0${i}`;
        try {
          const bedDoc = await BedSchedule.create({
            wardId: ward._id,
            bedId,
            roomNumber: `ICU_${i}`,
            department: 'ED',
            bedType: 'INTENSIVE',
            status: i <= 2 ? 'OCCUPIED' : 'AVAILABLE',
            currentEncounterId: i <= 2 ? patientIds[i - 1] : null,
            occupancyStartTime: i <= 2 ? new Date() : null,
          });
          bedIds.push(bedDoc._id);
        } catch (err) {
          if (err.code === 11000) {
            const existing = await BedSchedule.findOne({ bedId });
            if (existing) bedIds.push(existing._id);
          } else throw err;
        }
      }

      // Observation beds (2)
      for (let i = 1; i <= 2; i++) {
        const bedId = `ED-OBS-0${i}`;
        try {
          const bedDoc = await BedSchedule.create({
            wardId: ward._id,
            bedId,
            roomNumber: `OBS_${i}`,
            department: 'ED',
            bedType: 'MONITORED',
            status: 'AVAILABLE',
          });
          bedIds.push(bedDoc._id);
        } catch (err) {
          if (err.code === 11000) {
            const existing = await BedSchedule.findOne({ bedId });
            if (existing) bedIds.push(existing._id);
          } else throw err;
        }
      }

      // Standard beds (2)
      for (let i = 1; i <= 2; i++) {
        const bedId = `ED-STD-0${i}`;
        try {
          const bedDoc = await BedSchedule.create({
            wardId: ward._id,
            bedId,
            roomNumber: `STD_${i}`,
            department: 'ED',
            bedType: 'STANDARD',
            status: 'AVAILABLE',
          });
          bedIds.push(bedDoc._id);
        } catch (err) {
          if (err.code === 11000) {
            const existing = await BedSchedule.findOne({ bedId });
            if (existing) bedIds.push(existing._id);
          } else throw err;
        }
      }

      // Attach bed ids to ward and update counts
      ward.beds = bedIds;
      ward.totalBeds = bedIds.length;
      ward.occupiedBeds = bedIds.filter(id => id).length > 0 ? 2 : 0;
      ward.availableBeds = ward.totalBeds - ward.occupiedBeds;
      await ward.save();

      console.log('✅ Ward created with 9 beds (2 occupied, 7 available)');
    } else {
      // Ward exists - ensure beds are linked
      ward.beds = bedIds;
      await ward.save();
    }

    // Summary
    console.log('\n' + '='.repeat(50));
    console.log('✅ DATABASE POPULATION COMPLETE!');
    console.log('='.repeat(50));
    console.log('\n📊 Summary:');
    console.log(`  Hospital: ${hospital.name} (${hospital._id})`);
    console.log(`  Manager: ${manager.email}`);
    console.log(`  Physician: ${physician.name} (${physician.email})`);
    console.log(`  Patients: ${patientData.length}`);
    console.log(`  Triages: ${triages.length}`);
    console.log(`  Dispositions: ${dispositionData.length}`);
    console.log(`  Ward Beds: ${ward.beds.length} total (2 occupied, 7 available)`);
    console.log('\n🔑 Test Credentials:');
    console.log(`  Email: ${physician.email}`);
    console.log(`  Password: (use your existing password)`);
    console.log('\n💡 Next Steps:');
    console.log('  1. Start the backend: npm run dev');
    console.log('  2. Login with existing credentials');
    console.log('  3. Navigate to Physician Dashboard');
    console.log('  4. You should see 4 patients in the 3-lane view');
    console.log('\n');

    await mongoose.connection.close();
    console.log('🔌 Database connection closed');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

populateDatabase();
