/**
 * seeds/physician-dashboard-seed.js
 * 
 * Seed data for full hospital day simulation:
 * - 4 patients: John (RED), Sarah (GREEN), Michael (YELLOW), Maria (YELLOW)
 * - TriageAssessments with needsBed predictions
 * - Dispositions with physician decisions
 * - PatientJourneys tracking state
 * - Ward beds with occupancy
 */

import mongoose from 'mongoose';
import { TriageAssessment } from '../models/triageAssessment.model.js';
import { Disposition } from '../models/disposition.model.js';
import { PatientJourney } from '../models/patientJourney.model.js';
import { Ward } from '../models/ward.model.js';

// Mock patient data - these would normally be created via patient registration
const mockPatients = [
  {
    _id: new mongoose.Types.ObjectId(),
    firstName: 'John',
    lastName: 'Smith',
    age: 59,
    gender: 'M',
    dob: '1965-03-15',
  },
  {
    _id: new mongoose.Types.ObjectId(),
    firstName: 'Sarah',
    lastName: 'Wong',
    age: 34,
    gender: 'F',
    dob: '1992-07-22',
  },
  {
    _id: new mongoose.Types.ObjectId(),
    firstName: 'Michael',
    lastName: 'Chen',
    age: 48,
    gender: 'M',
    dob: '1978-11-05',
  },
  {
    _id: new mongoose.Types.ObjectId(),
    firstName: 'Maria',
    lastName: 'Rodriguez',
    age: 71,
    gender: 'F',
    dob: '1955-08-12',
  },
];

// Mock staff IDs
const mockStaffIds = {
  triageNurse: new mongoose.Types.ObjectId(), // Alice
  clinicalNurse: new mongoose.Types.ObjectId(), // Bob
  physician: new mongoose.Types.ObjectId(), // Dr. Chen
};

const mockHospitalId = new mongoose.Types.ObjectId();
const mockShiftId = new mongoose.Types.ObjectId();

export async function seedPhysicianDashboard() {
  try {
    console.log('\n🌱 Starting Physician Dashboard Seed...\n');

    // Clear existing data
    await TriageAssessment.deleteMany({ hospitalId: mockHospitalId });
    await Disposition.deleteMany({ hospitalId: mockHospitalId });
    await PatientJourney.deleteMany({ hospitalId: mockHospitalId });

    const shiftDate = new Date('2026-04-20');

    // ============ PATIENT 1: JOHN (RED, ADMITTED) ============
    console.log('📋 Creating JOHN (RED - Cardiac, ADMITTED)...');
    const johnTriageTime = new Date(shiftDate.getTime() + 7.25 * 60 * 60 * 1000); // 7:15 AM
    const johnTriage = await TriageAssessment.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[0]._id,
      triageNurseId: mockStaffIds.triageNurse,
      shiftId: mockShiftId,
      chiefComplaint: 'Acute chest pain for 2 hours',
      symptoms: ['chest pain', 'shortness of breath', 'nausea'],
      medicalHistory: 'Hypertension x10 years, smoking history',
      medications: ['Lisinopril 10mg', 'Aspirin 81mg daily'],
      allergies: ['NKDA'],
      vitalSigns: {
        temperature: 37.2,
        heartRate: 102,
        respiratoryRate: 20,
        systolic: 158,
        diastolic: 92,
        oxygenSaturation: 96,
      },
      triagePriority: 'RED',
      triageReasoning: 'Acute onset chest pain radiating to arm, elevated vitals',
      needsBed: true, // ⭐ CRITICAL: Triage says NEEDS BED
      bedNeedReasoning: 'Likely needs cardiac workup, probable admission',
      arrivedAt: johnTriageTime,
      triageStartedAt: new Date(johnTriageTime.getTime() + 3 * 60 * 1000), // 3 min later
      triageCompletedAt: new Date(johnTriageTime.getTime() + 4.5 * 60 * 1000), // 4.5 min
      status: 'DISPOSED',
    });

    const johnDispositionTime = new Date(johnTriageTime.getTime() + 10 * 60 * 1000); // 7:25 AM
    const johnDisposition = await Disposition.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[0]._id,
      triageAssessmentId: johnTriage._id,
      physicianId: mockStaffIds.physician,
      shiftId: mockShiftId,
      decision: 'ADMITTED', // ⭐ Physician ADMITS (matches triage: TRUE_POSITIVE)
      admissionDetails: {
        wardName: 'ICU/CCU',
        bedNumber: 'ICU_05',
        bedType: 'CRITICAL_CARE',
        icuRequired: true,
      },
      notes: 'ACS protocol, troponin levels needed, continuous monitoring',
      triageRecommendation: true,
      discrepancy: {
        hasDiscrepancy: false,
        type: 'TRUE_POSITIVE', // ✓ Correct prediction
      },
      triageCompletedAt: johnTriage.triageCompletedAt,
      physicianReviewStartedAt: new Date(johnTriageTime.getTime() + 7 * 60 * 1000),
      decisionMadeAt: johnDispositionTime,
      responseTimeMinutes: 9.5,
      status: 'COMPLETED',
    });

    const johnJourney = await PatientJourney.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[0]._id,
      shiftId: mockShiftId,
      currentState: 'ADMITTED',
      currentLocation: {
        locationType: 'WARD_BED',
        departmentContext: 'ER',
        wardName: 'ICU/CCU',
        bedNumber: 'ICU_05',
      },
      arrivedAt: johnTriageTime,
      triageCompletedAt: johnTriage.triageCompletedAt,
      admittedAt: johnDispositionTime,
      triageAssessmentId: johnTriage._id,
      dispositionId: johnDisposition._id,
      isActive: false,
    });

    // ============ PATIENT 2: SARAH (GREEN, DISCHARGED) ============
    console.log('📋 Creating SARAH (GREEN - Ankle Sprain, DISCHARGED)...');
    const sarahTriageTime = new Date(shiftDate.getTime() + 7.583 * 60 * 60 * 1000); // 7:35 AM
    const sarahTriage = await TriageAssessment.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[1]._id,
      triageNurseId: mockStaffIds.triageNurse,
      shiftId: mockShiftId,
      chiefComplaint: 'Ankle sprain from fall at home',
      symptoms: ['ankle pain', 'swelling', 'difficulty walking'],
      medicalHistory: 'No medical conditions',
      medications: [],
      allergies: ['NKDA'],
      vitalSigns: {
        temperature: 36.8,
        heartRate: 78,
        respiratoryRate: 16,
        systolic: 120,
        diastolic: 80,
        oxygenSaturation: 98,
      },
      triagePriority: 'GREEN',
      triageReasoning: 'Ankle sprain, vital signs normal, no serious injury',
      needsBed: false, // ⭐ Triage says NO BED
      bedNeedReasoning: 'Can be managed in observation, likely discharged same day',
      arrivedAt: sarahTriageTime,
      triageStartedAt: new Date(sarahTriageTime.getTime() + 2 * 60 * 1000),
      triageCompletedAt: new Date(sarahTriageTime.getTime() + 3 * 60 * 1000),
      status: 'DISPOSED',
    });

    const sarahDisposition = await Disposition.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[1]._id,
      triageAssessmentId: sarahTriage._id,
      physicianId: mockStaffIds.physician,
      shiftId: mockShiftId,
      decision: 'DISCHARGED', // ⭐ Physician DISCHARGES (matches triage: TRUE_NEGATIVE)
      notes: 'Ankle sprain, stable. X-ray shows no fracture. Discharged with pain management instructions.',
      triageRecommendation: false,
      discrepancy: {
        hasDiscrepancy: false,
        type: 'TRUE_NEGATIVE', // ✓ Correct prediction
      },
      triageCompletedAt: sarahTriage.triageCompletedAt,
      decisionMadeAt: new Date(shiftDate.getTime() + 9 * 60 * 60 * 1000), // 9:00 AM
      responseTimeMinutes: 85,
      status: 'COMPLETED',
    });

    const sarahJourney = await PatientJourney.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[1]._id,
      shiftId: mockShiftId,
      currentState: 'DISCHARGED',
      currentLocation: {
        locationType: 'DISCHARGED',
        departmentContext: 'ER',
      },
      arrivedAt: sarahTriageTime,
      triageCompletedAt: sarahTriage.triageCompletedAt,
      dischargedAt: new Date(shiftDate.getTime() + 9 * 60 * 60 * 1000),
      triageAssessmentId: sarahTriage._id,
      dispositionId: sarahDisposition._id,
      isActive: false,
    });

    // ============ PATIENT 3: MICHAEL (YELLOW, DISCHARGED) ============
    console.log('📋 Creating MICHAEL (YELLOW - Viral Illness, DISCHARGED)...');
    const michaelTriageTime = new Date(shiftDate.getTime() + 8.25 * 60 * 60 * 1000); // 8:15 AM
    const michaelTriage = await TriageAssessment.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[2]._id,
      triageNurseId: mockStaffIds.triageNurse,
      shiftId: mockShiftId,
      chiefComplaint: 'Mild fever, body aches, no respiratory distress',
      symptoms: ['fever', 'body aches', 'fatigue'],
      medicalHistory: 'No significant history',
      medications: [],
      allergies: ['NKDA'],
      vitalSigns: {
        temperature: 38.2,
        heartRate: 85,
        respiratoryRate: 18,
        systolic: 125,
        diastolic: 81,
        oxygenSaturation: 97,
      },
      triagePriority: 'YELLOW',
      triageReasoning: 'Appears to be viral infection, vitals stable',
      needsBed: false, // ⭐ Triage says NO BED
      bedNeedReasoning: 'Appears viral, vitals stable, safe to discharge with rest',
      arrivedAt: michaelTriageTime,
      triageStartedAt: new Date(michaelTriageTime.getTime() + 2 * 60 * 1000),
      triageCompletedAt: new Date(michaelTriageTime.getTime() + 3 * 60 * 1000),
      status: 'DISPOSED',
    });

    const michaelDisposition = await Disposition.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[2]._id,
      triageAssessmentId: michaelTriage._id,
      physicianId: mockStaffIds.physician,
      shiftId: mockShiftId,
      decision: 'DISCHARGED', // ⭐ Physician DISCHARGES (matches triage: TRUE_NEGATIVE)
      notes: 'Likely viral infection. Stable vitals. Discharged with symptomatic care instructions.',
      triageRecommendation: false,
      discrepancy: {
        hasDiscrepancy: false,
        type: 'TRUE_NEGATIVE', // ✓ Correct prediction
      },
      triageCompletedAt: michaelTriage.triageCompletedAt,
      decisionMadeAt: new Date(shiftDate.getTime() + 9.583 * 60 * 60 * 1000), // 9:35 AM
      responseTimeMinutes: 76,
      status: 'COMPLETED',
    });

    const michaelJourney = await PatientJourney.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[2]._id,
      shiftId: mockShiftId,
      currentState: 'DISCHARGED',
      currentLocation: {
        locationType: 'DISCHARGED',
        departmentContext: 'ER',
      },
      arrivedAt: michaelTriageTime,
      triageCompletedAt: michaelTriage.triageCompletedAt,
      dischargedAt: new Date(shiftDate.getTime() + 9.583 * 60 * 60 * 1000),
      triageAssessmentId: michaelTriage._id,
      dispositionId: michaelDisposition._id,
      isActive: false,
    });

    // ============ PATIENT 4: MARIA (YELLOW, ADMITTED - DISCREPANCY) ============
    console.log('📋 Creating MARIA (YELLOW - Confusion, ADMITTED - FALSE_NEGATIVE)...');
    const mariaTriageTime = new Date(shiftDate.getTime() + 11.5 * 60 * 60 * 1000); // 11:30 AM
    const mariaTriage = await TriageAssessment.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[3]._id,
      triageNurseId: mockStaffIds.triageNurse,
      shiftId: mockShiftId,
      chiefComplaint: 'Mild headache, confusion for 2 hours',
      symptoms: ['headache', 'confusion'],
      medicalHistory: 'On warfarin (blood thinner)',
      medications: ['Warfarin'],
      allergies: ['NKDA'],
      vitalSigns: {
        temperature: 37.5,
        heartRate: 88,
        respiratoryRate: 18,
        systolic: 138,
        diastolic: 85,
        oxygenSaturation: 96,
      },
      triagePriority: 'YELLOW',
      triageReasoning: 'Mild symptoms, vitals stable',
      needsBed: false, // ⭐ Triage says NO BED (MISSED the warfarin + head trauma concern)
      bedNeedReasoning: 'Mild symptoms, vitals stable, can observe and discharge',
      arrivedAt: mariaTriageTime,
      triageStartedAt: new Date(mariaTriageTime.getTime() + 2 * 60 * 1000),
      triageCompletedAt: new Date(mariaTriageTime.getTime() + 3 * 60 * 1000),
      status: 'DISPOSED',
    });

    const mariaDispositionTime = new Date(shiftDate.getTime() + 11.833 * 60 * 60 * 1000); // 11:50 AM
    const mariaDisposition = await Disposition.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[3]._id,
      triageAssessmentId: mariaTriage._id,
      physicianId: mockStaffIds.physician,
      shiftId: mockShiftId,
      decision: 'ADMITTED', // ⭐ Physician ADMITS (DISAGREES with triage: FALSE_NEGATIVE)
      admissionDetails: {
        wardName: 'ICU',
        bedNumber: 'ICU_03',
        bedType: 'CRITICAL_CARE',
        icuRequired: true,
      },
      notes: 'Warfarin + confusion + head trauma → possible intracranial bleed. CT head + neuro monitoring required.',
      triageRecommendation: false,
      discrepancy: {
        hasDiscrepancy: true,
        type: 'FALSE_NEGATIVE', // ✗ Triage missed this admission need
        reason: 'Warfarin + confusion + head trauma not fully assessed in triage. Condition deteriorated.',
        severity: 'HIGH',
      },
      triageCompletedAt: mariaTriage.triageCompletedAt,
      physicianReviewStartedAt: new Date(mariaTriageTime.getTime() + 15 * 60 * 1000),
      decisionMadeAt: mariaDispositionTime,
      responseTimeMinutes: 17,
      status: 'COMPLETED',
    });

    const mariaJourney = await PatientJourney.create({
      hospitalId: mockHospitalId,
      patientId: mockPatients[3]._id,
      shiftId: mockShiftId,
      currentState: 'ADMITTED',
      currentLocation: {
        locationType: 'WARD_BED',
        departmentContext: 'ER',
        wardName: 'ICU',
        bedNumber: 'ICU_03',
      },
      arrivedAt: mariaTriageTime,
      triageCompletedAt: mariaTriage.triageCompletedAt,
      escalatedAt: new Date(mariaTriageTime.getTime() + 15 * 60 * 1000),
      admittedAt: mariaDispositionTime,
      triageAssessmentId: mariaTriage._id,
      dispositionId: mariaDisposition._id,
      isActive: false,
    });

    // ============ CREATE WARD WITH BEDS ============
    console.log('🛏️ Creating ward beds...');
    await Ward.updateOne(
      { hospitalId: mockHospitalId, departmentContext: 'ER' },
      {
        $set: {
          hospitalId: mockHospitalId,
          departmentContext: 'ER',
          wardName: 'Emergency Department',
          beds: [
            { bedNumber: 'ICU_01', bedType: 'CRITICAL_CARE', isOccupied: false },
            { bedNumber: 'ICU_02', bedType: 'CRITICAL_CARE', isOccupied: false },
            { bedNumber: 'ICU_03', bedType: 'CRITICAL_CARE', isOccupied: true, patientId: mockPatients[3]._id }, // Maria
            { bedNumber: 'ICU_04', bedType: 'CRITICAL_CARE', isOccupied: false },
            { bedNumber: 'ICU_05', bedType: 'CRITICAL_CARE', isOccupied: true, patientId: mockPatients[0]._id }, // John
            { bedNumber: 'OBS_01', bedType: 'OBSERVATION', isOccupied: false },
            { bedNumber: 'OBS_02', bedType: 'OBSERVATION', isOccupied: false },
            { bedNumber: 'STD_01', bedType: 'STANDARD', isOccupied: false },
            { bedNumber: 'STD_02', bedType: 'STANDARD', isOccupied: false },
          ],
        },
      },
      { upsert: true }
    );

    console.log('\n✅ Physician Dashboard Seed Complete!\n');
    console.log('📊 Seeded Data:');
    console.log(`   - ${mockPatients.length} patients with triages and dispositions`);
    console.log('   - 2 admissions (John-RED, Maria-YELLOW)');
    console.log('   - 2 discharges (Sarah-GREEN, Michael-YELLOW)');
    console.log('   - 1 discrepancy (Maria - FALSE_NEGATIVE)');
    console.log('   - 9 beds (5 ICU, 2 OBS, 2 STD)');
    console.log('   - Occupancy: 2/9 beds occupied\n');

    return {
      patients: mockPatients,
      triages: [johnTriage, sarahTriage, michaelTriage, mariaTriage],
      dispositions: [johnDisposition, sarahDisposition, michaelDisposition, mariaDisposition],
      journeys: [johnJourney, sarahJourney, michaelJourney, mariaJourney],
    };
  } catch (err) {
    console.error('❌ Seed error:', err);
    throw err;
  }
}

export default seedPhysicianDashboard;
