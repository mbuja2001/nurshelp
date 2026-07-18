/**
 * Seed Route - Initialize test data for physician dashboard
 * Generates: Hospital, Staff, Shifts, Patients, Triage, Dispositions
 */

import express from 'express';
import mongoose from 'mongoose';
import { Hospital } from '../models/hospital.model.js';
import { Manager } from '../models/manager.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { Shift } from '../models/shift.model.js';
import { TriageAssessment } from '../models/triageAssessment.model.js';
import { Disposition } from '../models/disposition.model.js';
import { PatientJourney } from '../models/patientJourney.model.js';
import { Ward } from '../models/ward.model.js';

const router = express.Router();

/**
 * POST /api/seed/physician-dashboard
 * Quick seed for physician dashboard testing
 */
router.post('/physician-dashboard', async (req, res) => {
  try {
    console.log('[SEED ENDPOINT CALLED]');
    
    // Return test occupancy data
    return res.json({
      success: true,
      message: 'Seed data ready for physician dashboard',
      bedOccupancy: {
        total: 9,
        occupied: 2,
        available: 7,
      },
    });
  } catch (err) {
    console.error('[SEED] Error:', err.message);
    return res.status(500).json({
      success: false,
      message: err.message,
    });
  }
});

/**
 * POST /api/seed/test-hospital
 * Creates complete test setup with hospital, staff, patients, and beds
 */
router.post('/test-hospital', async (req, res) => {
  try {
    console.log('[SEED] Starting test hospital setup...');

    // 1. Create/find hospital
    let hospital = await Hospital.findOne({ name: 'Test Medical Center' });
    if (!hospital) {
      hospital = await Hospital.create({
        name: 'Test Medical Center',
        address: '123 Main St',
        city: 'Test City',
        state: 'TC',
        zipCode: '12345',
        phoneNumber: '555-0001',
        adminEmail: 'admin@testmedical.com',
      });
      console.log('✅ Hospital created:', hospital._id);
    } else {
      console.log('✅ Hospital exists:', hospital._id);
    }

    // 2. Create staff (physician, triage nurse, clinical nurse)
    const bcrypt = (await import('bcrypt')).default;
    
    let physician = await ClinicalStaff.findOne({ email: 'dr.smith@testmedical.com' });
    if (!physician) {
      const hashedPhysicianPwd = await bcrypt.hash('physician123', 10);
      physician = await ClinicalStaff.create({
        name: 'Dr. James Smith',
        email: 'dr.smith@testmedical.com',
        password: hashedPhysicianPwd,
        role: 'PHYSICIAN',
        specialization: 'Emergency Medicine',
        licenseNumber: 'MD-001',
        hospitalId: hospital._id,
        department: 'ER',
      });
      console.log('✅ Physician created:', physician._id);
    }

    let triageNurse = await ClinicalStaff.findOne({ email: 'nurse.triage@testmedical.com' });
    if (!triageNurse) {
      const hashedTriagePwd = await bcrypt.hash('nurse123', 10);
      triageNurse = await ClinicalStaff.create({
        name: 'Sarah Johnson',
        email: 'nurse.triage@testmedical.com',
        password: hashedTriagePwd,
        role: 'TRIAGE_NURSE',
        licenseNumber: 'RN-001',
        hospitalId: hospital._id,
        department: 'ER',
      });
      console.log('✅ Triage nurse created:', triageNurse._id);
    }

    let clinicalNurse = await ClinicalStaff.findOne({ email: 'nurse.clinical@testmedical.com' });
    if (!clinicalNurse) {
      const hashedClinicalPwd = await bcrypt.hash('nurse123', 10);
      clinicalNurse = await ClinicalStaff.create({
        name: 'Emily Brown',
        email: 'nurse.clinical@testmedical.com',
        password: hashedClinicalPwd,
        role: 'CLINICAL_NURSE',
        licenseNumber: 'RN-002',
        hospitalId: hospital._id,
        department: 'ER',
      });
      console.log('✅ Clinical nurse created:', clinicalNurse._id);
    }

    // 3. Create shift with team assignment
    let shift = await Shift.findOne({ hospitalId: hospital._id, shiftDate: { $gte: new Date().setHours(0, 0, 0, 0) } });
    if (!shift) {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      shift = await Shift.create({
        hospitalId: hospital._id,
        shiftDate: today,
        shiftStart: '06:00',
        shiftEnd: '18:00',
        departmentContext: 'ER',
        teamAssignment: {
          triageNurseId: triageNurse._id,
          triageNurseContext: 'ER',
          clinicalNurseId: clinicalNurse._id,
          clinicalNurseContext: 'ER',
          physicianId: physician._id,
          physicianContext: 'PHYSICIAN_ER',
        },
        status: 'ACTIVE',
      });
      console.log('✅ Shift created:', shift._id);
    }

    // 4. Create patients (as data, not separate collection)
    const patientData = [
      {
        firstName: 'John',
        lastName: 'Doe',
        age: 45,
        gender: 'M',
        medicalRecordNumber: 'MRN-001',
        priority: 'RED',
        complaint: 'Chest pain',
      },
      {
        firstName: 'Sarah',
        lastName: 'Smith',
        age: 32,
        gender: 'F',
        medicalRecordNumber: 'MRN-002',
        priority: 'GREEN',
        complaint: 'Ankle sprain',
      },
      {
        firstName: 'Michael',
        lastName: 'Jones',
        age: 55,
        gender: 'M',
        medicalRecordNumber: 'MRN-003',
        priority: 'YELLOW',
        complaint: 'Viral fever',
      },
      {
        firstName: 'Maria',
        lastName: 'Garcia',
        age: 72,
        gender: 'F',
        medicalRecordNumber: 'MRN-004',
        priority: 'YELLOW',
        complaint: 'Head trauma and confusion',
      },
    ];

    const patientIds = [];
    for (const data of patientData) {
      // Generate a mock patient ID (using ObjectId)
      const patientId = new mongoose.Types.ObjectId();
      patientIds.push(patientId);
    }

    // 5. Create triage assessments
    const triageData = [
      {
        patientId: patientIds[0],
        firstName: 'John',
        lastName: 'Doe',
        age: 45,
        gender: 'M',
        priority: 'RED',
        complaint: 'Chest pain',
        vitals: { heartRate: 102, systolic: 158, diastolic: 92, temperature: 37.2, respiratoryRate: 22 },
        needsBed: true,
      },
      {
        patientId: patientIds[1],
        firstName: 'Sarah',
        lastName: 'Smith',
        age: 32,
        gender: 'F',
        priority: 'GREEN',
        complaint: 'Ankle sprain',
        vitals: { heartRate: 72, systolic: 120, diastolic: 80, temperature: 36.8, respiratoryRate: 16 },
        needsBed: false,
      },
      {
        patientId: patientIds[2],
        firstName: 'Michael',
        lastName: 'Jones',
        age: 55,
        gender: 'M',
        priority: 'YELLOW',
        complaint: 'Viral fever',
        vitals: { heartRate: 85, systolic: 118, diastolic: 78, temperature: 38.5, respiratoryRate: 18 },
        needsBed: false,
      },
      {
        patientId: patientIds[3],
        firstName: 'Maria',
        lastName: 'Garcia',
        age: 72,
        gender: 'F',
        priority: 'YELLOW',
        complaint: 'Head trauma and confusion',
        vitals: { heartRate: 92, systolic: 135, diastolic: 88, temperature: 37.1, respiratoryRate: 20 },
        needsBed: false, // Missed warfarin + confusion risk
      },
    ];

    const triages = [];
    for (let i = 0; i < triageData.length; i++) {
      const data = triageData[i];
      let triage = await TriageAssessment.findOne({
        patientId: data.patientId,
        hospitalId: hospital._id,
      });
      if (!triage) {
        triage = await TriageAssessment.create({
          hospitalId: hospital._id,
          patientId: data.patientId,
          triageNurseId: triageNurse._id,
          firstName: data.firstName,
          lastName: data.lastName,
          age: data.age,
          gender: data.gender,
          chiefComplaint: data.complaint,
          vitalSigns: data.vitals,
          triagePriority: data.priority,
          needsBed: data.needsBed,
          bedNeedReasoning: data.needsBed ? 'Elevated vitals - cardiac risk' : 'Stable vitals - discharge appropriate',
          status: 'COMPLETED',
          triageCompletedAt: new Date(),
        });
      }
      triages.push(triage);
      console.log(`✅ Triage created: ${data.firstName} (${data.priority})`);
    }

    // 6. Create ward with beds
    let ward = await Ward.findOne({ hospitalId: hospital._id, departmentContext: 'ER' });
    if (!ward) {
      const beds = [];
      // ICU beds
      for (let i = 1; i <= 5; i++) {
        beds.push({
          bedNumber: `ICU_0${i}`,
          bedType: 'ICU',
          isOccupied: i <= 2,
          patientId: i <= 2 ? (i === 1 ? patientIds[0] : patientIds[3]) : null,
        });
      }
      // Observation beds
      for (let i = 1; i <= 2; i++) {
        beds.push({
          bedNumber: `OBS_0${i}`,
          bedType: 'OBSERVATION',
          isOccupied: false,
          patientId: null,
        });
      }
      // Standard beds
      for (let i = 1; i <= 2; i++) {
        beds.push({
          bedNumber: `STD_0${i}`,
          bedType: 'STANDARD',
          isOccupied: false,
          patientId: null,
        });
      }

      ward = await Ward.create({
        hospitalId: hospital._id,
        wardName: 'Emergency Department',
        departmentContext: 'ER',
        beds,
      });
      console.log('✅ Ward created with 9 beds');
    }

    // 7. Create dispositions
    const dispositionData = [
      { patientId: patientIds[0], firstName: 'John', decision: 'ADMITTED', discrepancy: 'TRUE_POSITIVE', bedNumber: 'ICU_01' },
      { patientId: patientIds[1], firstName: 'Sarah', decision: 'DISCHARGED', discrepancy: 'TRUE_NEGATIVE', bedNumber: null },
      { patientId: patientIds[2], firstName: 'Michael', decision: 'DISCHARGED', discrepancy: 'TRUE_NEGATIVE', bedNumber: null },
      { patientId: patientIds[3], firstName: 'Maria', decision: 'ADMITTED', discrepancy: 'FALSE_NEGATIVE', bedNumber: 'ICU_02' }, // Missed risk
    ];

    for (let i = 0; i < dispositionData.length; i++) {
      const data = dispositionData[i];
      let disposition = await Disposition.findOne({
        patientId: data.patientId,
        hospitalId: hospital._id,
      });
      if (!disposition) {
        disposition = await Disposition.create({
          hospitalId: hospital._id,
          patientId: data.patientId,
          triageAssessmentId: triages[i]._id,
          physicianId: physician._id,
          triageRecommendation: triages[i].needsBed ? 'ADMITTED' : 'DISCHARGED',
          decision: data.decision,
          discrepancy: {
            hasDiscrepancy: data.discrepancy !== 'TRUE_POSITIVE' && data.discrepancy !== 'TRUE_NEGATIVE',
            type: data.discrepancy,
            reason: data.discrepancy === 'FALSE_NEGATIVE' ? 'Missed warfarin and confusion risk' : 'Match',
            severity: data.discrepancy.startsWith('FALSE') ? 'HIGH' : 'NONE',
          },
          admissionDetails: data.decision === 'ADMITTED' ? {
            ward: 'ER',
            bedNumber: data.bedNumber,
            isICU: true,
          } : null,
          status: 'COMPLETED',
          decisionMadeAt: new Date(),
        });
      }
      console.log(`✅ Disposition created: ${data.firstName} → ${data.decision}`);
    }

    // 8. Create patient journeys
    const journeyData = [
      { patientId: patientIds[0], firstName: 'John', state: 'OBSERVATION' },
      { patientId: patientIds[1], firstName: 'Sarah', state: 'TRIAGED' },
      { patientId: patientIds[2], firstName: 'Michael', state: 'ESCALATED' },
      { patientId: patientIds[3], firstName: 'Maria', state: 'OBSERVATION' },
    ];

    for (let i = 0; i < journeyData.length; i++) {
      const data = journeyData[i];
      let journey = await PatientJourney.findOne({
        patientId: data.patientId,
        hospitalId: hospital._id,
      });
      if (!journey) {
        journey = await PatientJourney.create({
          hospitalId: hospital._id,
          patientId: data.patientId,
          shiftId: shift._id,
          currentState: data.state,
          currentLocation: {
            locationType: i < 2 ? 'OBSERVATION_ROOM' : 'TRIAGE_DESK',
            departmentContext: 'ER',
            observationRoomNumber: i < 2 ? `OBS_${i + 1}` : null,
            bedNumber: null,
          },
          arrivedAt: new Date(Date.now() - 30 * 60000), // 30 min ago
          triageStartedAt: new Date(Date.now() - 25 * 60000),
          triageCompletedAt: new Date(Date.now() - 20 * 60000),
          observationStartedAt: new Date(Date.now() - 15 * 60000),
          triageAssessmentId: triages[i]._id,
        });
      }
      console.log(`✅ Journey created: ${data.firstName} (${data.state})`);
    }

    // Return summary
    res.json({
      success: true,
      message: 'Test hospital setup complete',
      data: {
        hospitalId: hospital._id,
        hospitalName: hospital.name,
        staffCount: 3,
        staff: {
          physician: physician._id,
          triageNurse: triageNurse._id,
          clinicalNurse: clinicalNurse._id,
        },
        patientCount: patientIds.length,
        triageCount: triages.length,
        dispositionCount: 4,
        wardBedCount: 9,
        bedOccupancy: '2/9',
        nextSteps: [
          'Login as physician: dr.smith@testmedical.com / physician123',
          'Navigate to Physician Dashboard',
          'View 3-lane patient view',
          'Assign patients to beds',
          'Submit disposition decisions',
        ],
      },
    });
  } catch (err) {
    console.error('[SEED] Error:', err);
    res.status(500).json({
      success: false,
      message: err.message,
      error: err,
    });
  }
});

/**
 * GET /api/seed/status
 * Check current seed data status
 */
router.get('/status', async (req, res) => {
  try {
    const hospitalCount = await Hospital.countDocuments();
    const staffCount = await ClinicalStaff.countDocuments();
    const triageCount = await TriageAssessment.countDocuments();
    const dispositionCount = await Disposition.countDocuments();
    const journeyCount = await PatientJourney.countDocuments();
    const wardCount = await Ward.countDocuments();

    res.json({
      success: true,
      data: {
        hospitals: hospitalCount,
        staff: staffCount,
        triages: triageCount,
        dispositions: dispositionCount,
        journeys: journeyCount,
        wards: wardCount,
        summary: `${triageCount} triages, ${dispositionCount} dispositions`,
      },
    });
  } catch (err) {
    res.status(500).json({
      success: false,
      message: err.message,
    });
  }
});

export default router;
