#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';
import PatientEHRRecord from '../src/models/patientEHRRecord.model.js';
import { Hospital } from '../src/models/hospital.model.js';
import { Prescription } from '../src/models/pharmacy.model.js';
import { buildDischargeReportData } from '../src/utils/dischargeReportBuilder.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const DEFAULT_COUNT = 10;
const DEFAULT_ENCOUNTERS = 12;
const DEFAULT_ACTIVE_QUEUE = 0;
const DEFAULT_BATCH = `history-stress-${new Date().toISOString().slice(0, 10)}`;
const BASE_DATE = new Date('2026-08-25T12:00:00.000Z');

const scenarios = [
  { symptoms: 'chest pain and shortness of breath', history: 'hypertension', status: 'reported', critical: { allergies: ['Peanuts'], adverseReactions: ['Anaphylaxis'], medications: ['Warfarin'], conditions: ['Severe haemophilia A'], implantsOrProcedures: ['Recent splenectomy'] }, esi: 2 },
  { symptoms: 'fever and productive cough', history: 'COPD', status: 'none_established', critical: {}, esi: 3 },
  { symptoms: 'abdominal pain and vomiting', history: 'previous appendicitis', status: 'reported', critical: { allergies: ['Penicillin'], adverseReactions: ['Severe rash'], medications: [], conditions: [], implantsOrProcedures: [] }, esi: 3 },
  { symptoms: 'weakness and dizziness', history: 'atrial fibrillation', status: 'reported', critical: { allergies: [], adverseReactions: [], medications: ['Apixaban'], conditions: ['Atrial fibrillation'], implantsOrProcedures: [] }, esi: 2 }
];

const priorityForEsi = { 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' };
const addMinutes = (date, minutes) => new Date(date.getTime() + minutes * 60 * 1000);

const getArg = (name, fallback) => {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
};

const numberArg = (name, fallback) => Math.max(1, Number.parseInt(getArg(name, fallback), 10) || fallback);
const nonNegativeNumberArg = (name, fallback) => Math.max(0, Number.parseInt(getArg(name, fallback), 10) || fallback);
const hasFlag = (name) => process.argv.includes(`--${name}`);

const makeEncounter = ({ patientIndex, encounterIndex, batch, hospitalId, count, encountersPerPatient, activeQueue }) => {
  const scenario = scenarios[(patientIndex + encounterIndex) % scenarios.length];
  const patientId = `SYNTH-HISTORY-${String(patientIndex + 1).padStart(6, '0')}`;
  const encounterId = `${batch}-${patientId}-${String(encounterIndex + 1).padStart(3, '0')}`;
  const date = new Date(BASE_DATE.getTime() - encounterIndex * 10 * 24 * 60 * 60 * 1000 - (patientIndex % 5) * 60 * 60 * 1000);
  const criticalHistory = scenario.status === 'reported' ? scenario.critical : {};
  const isActiveQueuePatient = patientIndex * encountersPerPatient + encounterIndex < activeQueue;
  const priority = priorityForEsi[scenario.esi];
  const aiTriageReport = {
    esi_level: scenario.esi,
    triageCategory: priority,
    calculated_priority: priority,
    clinical_summary: `Synthetic AI triage for ${scenario.symptoms}.`,
    differential_diagnoses: [scenario.history, scenario.symptoms],
    recommendations: ['Complete physician review', 'Review linked laboratory reports'],
    generated_at: addMinutes(date, 1),
    generated_by: 'synthetic-ai-triage'
  };
  const manualTriageReport = {
    esi_level: scenario.esi,
    selected_priority: priority,
    satsLevel: priority,
    notes: 'Synthetic manual confirmation of the AI triage result.',
    confirmed_at: addMinutes(date, 2),
    confirmed_by: 'synthetic-triage-nurse'
  };
  const labRequests = [
    {
      specialty: 'RADIOLOGY',
      lab_type: 'Chest X-ray',
      description: `Evaluate ${scenario.symptoms}.`,
      priority: scenario.esi <= 2 ? 'URGENT' : 'MEDIUM',
      requested_by: 'Dr. Synthetic Physician',
      requested_at: addMinutes(date, 4),
      assigned_technician: 'Synthetic Radiologist',
      assigned_at: addMinutes(date, 4),
      report_content: 'No acute cardiopulmonary abnormality identified in this synthetic report.',
      clinical_impression: 'Stable radiographic findings for historical fixture.',
      recommendations: ['Correlate with clinical findings.'],
      submitted_at: addMinutes(date, 5),
      submitted_by: 'Synthetic Radiologist',
      status: 'REVIEWED',
      reviewed_at: addMinutes(date, 5),
      reviewed_by: 'Dr. Synthetic Physician',
      review_notes: 'Radiology report reviewed and filed in the encounter record.',
      created_at: addMinutes(date, 4),
      updated_at: addMinutes(date, 5)
    },
    {
      specialty: 'PATHOLOGY',
      lab_type: 'Complete blood count',
      description: `Baseline blood work for ${scenario.history}.`,
      priority: 'MEDIUM',
      requested_by: 'Dr. Synthetic Physician',
      requested_at: addMinutes(date, 4),
      assigned_technician: 'Synthetic Pathologist',
      assigned_at: addMinutes(date, 4),
      report_content: 'Hemoglobin, white cell count, and platelets within expected synthetic reference ranges.',
      clinical_impression: 'No critical laboratory abnormality in this historical fixture.',
      recommendations: ['Continue routine clinical correlation.'],
      submitted_at: addMinutes(date, 5),
      submitted_by: 'Synthetic Pathologist',
      status: 'REVIEWED',
      reviewed_at: addMinutes(date, 5),
      reviewed_by: 'Dr. Synthetic Physician',
      review_notes: 'Pathology report reviewed and filed in the encounter record.',
      created_at: addMinutes(date, 4),
      updated_at: addMinutes(date, 5),
    }
  ];

  return {
    hospitalId,
    date,
    encounterId,
    patient: {
      id_number: patientId,
      name: `Synthetic${patientIndex + 1}`,
      surname: 'HistoryPatient',
      gender: patientIndex % 2 ? 'F' : 'M',
      age: 18 + (patientIndex % 70),
      height_cm: 155 + (patientIndex % 40),
      symptoms: scenario.symptoms,
      duration: `${encounterIndex + 1} hours`,
      painLevel: `${Math.min(10, 3 + (patientIndex + encounterIndex) % 8)}/10`,
      history: scenario.history,
      criticalHistoryStatus: scenario.status,
      criticalHistory,
      criticalHistoryReview: {
        sourceEncounterId: encounterId,
        sourceTimestamp: date,
        confirmed: scenario.status === 'reported',
        reviewedAt: date,
        reviewedBy: 'synthetic-seed-reviewer',
        reviewReason: 'Synthetic fixture for context-stuffing validation'
      }
    },
    vitals: {
      temp: 36.5 + ((patientIndex + encounterIndex) % 30) / 10,
      hr: 70 + ((patientIndex * 3 + encounterIndex * 5) % 70),
      resp: 14 + ((patientIndex + encounterIndex) % 16),
      bp_systolic: 105 + ((patientIndex * 7 + encounterIndex) % 65),
      bp_diastolic: 65 + ((patientIndex + encounterIndex) % 35),
      o2: 91 + ((patientIndex + encounterIndex) % 9),
      recordedAt: addMinutes(date, 1),
    },
    triage: {
      esi_level: scenario.esi,
      aiTriageReport,
      manualTriageReport,
      synthetic: true,
      seedBatch: batch
    },
    ai_triage_report: aiTriageReport,
    manual_triage_report: manualTriageReport,
    severity: scenario.esi,
    syntheticData: true,
    seedBatch: batch,
    seedPatientCount: count,
    diagnosis: `${scenario.history}; evaluated for ${scenario.symptoms}.`,
    treatment_plan: 'Historical treatment completed; follow-up with the primary care team as clinically indicated.',
    physician_notes: 'Synthetic historical encounter with linked specialist reports and discharge documentation.',
    assigned_physician: {
      id: null,
      name: 'Dr. Synthetic Physician',
      specialty: 'Internal Medicine',
      diagnosed_at: addMinutes(date, 3)
    },
    medication_administrations: [{
      medication_name: scenario.esi <= 2 ? 'Acetaminophen' : 'Amoxicillin',
      dosage: scenario.esi <= 2 ? '500 mg' : '500 mg',
      frequency: 'Every 8 hours as needed',
      route: 'Oral',
      indication: scenario.symptoms,
      administered_at: addMinutes(date, 6),
      recorded_at: addMinutes(date, 6),
      status: 'ADMINISTERED',
      administered_by: 'Synthetic Nurse',
      prescribed_by_role: 'PHYSICIAN',
      patient_response: 'Tolerated without documented adverse reaction.'
    }],
    lab_requests: labRequests,
    surgery_records: [{
      procedure_name: 'Diagnostic evaluation and observation',
      surgery_type: 'ELECTIVE',
      rationale: 'Synthetic historical procedure record for report navigation.',
      priority: 'MEDIUM',
      scheduled_at: addMinutes(date, 6),
      surgeon: 'Dr. Synthetic Surgeon',
      surgical_team: ['Synthetic Operating Room Team'],
      requested_by: 'Dr. Synthetic Physician',
      requested_at: addMinutes(date, 6),
      status: 'COMPLETED',
      performed_at: addMinutes(date, 7),
      outcome: 'Completed without complication.',
      findings: 'No unexpected finding in this synthetic historical record.',
      complications: 'None documented.',
      notes: 'Synthetic surgery report linked to the encounter.',
      recorded_by: 'Dr. Synthetic Surgeon',
      recorded_at: addMinutes(date, 7),
      updated_at: addMinutes(date, 7),
    }],
    status: isActiveQueuePatient ? 'TRIAGE_IN_PROGRESS' : 'DISCHARGED',
    isWaiting: isActiveQueuePatient,
    specialistAttended: !isActiveQueuePatient,
    submittedAt: date,
    attendedAt: isActiveQueuePatient ? null : date,
    sourceMetadata: {
      synthetic: true,
      seedBatch: batch,
      recordType: isActiveQueuePatient ? 'ACTIVE_QUEUE' : 'HISTORICAL_ENCOUNTER'
    }
  };
};

async function main() {
  const uri = process.env.MONGO_URI || process.env.MONGODB_URI;
  if (!uri) {
    throw new Error('Missing MongoDB connection string. Set MONGO_URI or MONGODB_URI in backend/.env');
  }

  const count = numberArg('patients', DEFAULT_COUNT);
  const encountersPerPatient = numberArg('encounters-per-patient', DEFAULT_ENCOUNTERS);
  const batch = getArg('batch', DEFAULT_BATCH);
  const activeQueue = Math.min(nonNegativeNumberArg('active-queue', DEFAULT_ACTIVE_QUEUE), count * encountersPerPatient);
  const hospitalArg = getArg('hospital-id', process.env.HOSPITAL_ID || '');
  const createProjection = !hasFlag('no-projection');

  if (!process.env.ALLOW_SYNTHETIC_SEED && process.env.NODE_ENV === 'production') {
    throw new Error('Synthetic seeding is blocked in production. Set an explicit non-production environment.');
  }

  await mongoose.connect(uri);
  try {
    let hospitalId = hospitalArg || null;
    if (hospitalId && !mongoose.Types.ObjectId.isValid(hospitalId)) {
      const hospital = await Hospital.findOne({ hospitalID: hospitalId }).select('_id').lean();
      hospitalId = hospital?._id || hospitalId;
    }
    if (!hospitalId) {
      const hospital = await Hospital.findOne({ isActive: true }).select('_id').lean();
      hospitalId = hospital?._id || null;
    }
    if (!hospitalId) throw new Error('No hospital found. Pass --hospital-id or configure an active hospital.');

    const existing = await Encounter.countDocuments({ seedBatch: batch });
    if (existing > 0) throw new Error(`Seed batch ${batch} already exists (${existing} records). Choose another batch or remove it explicitly.`);

    const documents = [];
    for (let patientIndex = 0; patientIndex < count; patientIndex += 1) {
      for (let encounterIndex = 0; encounterIndex < encountersPerPatient; encounterIndex += 1) {
        documents.push(makeEncounter({ patientIndex, encounterIndex, batch, hospitalId, count, encountersPerPatient, activeQueue }));
      }
    }

    const inserted = await Encounter.insertMany(documents, { ordered: true });
    const historicalEncounters = inserted.filter((encounter) => encounter.status === 'DISCHARGED');
    await Promise.all(historicalEncounters.map(async (encounter) => {
      encounter.discharge_report = buildDischargeReportData({
        encounter,
        overrides: {
          authorId: 'synthetic-discharge-clinician',
          signedOffBy: 'Dr. Synthetic Physician',
          followUpInstructions: 'Routine follow-up recommended; return for worsening symptoms.',
          notes: 'Synthetic discharge report linked to the historical encounter.'
        }
      });
      encounter.discharge_report.dischargeDate = addMinutes(encounter.date, 9);
      encounter.discharge_report.createdAt = addMinutes(encounter.date, 9);
      encounter.discharge_report.updatedAt = addMinutes(encounter.date, 9);
      encounter.markModified('discharge_report');
      await encounter.save();
    }));

    const prescriptions = await Prescription.insertMany(historicalEncounters.map((encounter, index) => ({
      hospitalId,
      encounterId: encounter._id,
      patientId: encounter.patient.id_number,
      physicianId: 'synthetic-physician',
      medicationName: encounter.medication_administrations?.[0]?.medication_name || 'Acetaminophen',
      patient_details: {
        name: `${encounter.patient.name} ${encounter.patient.surname}`,
        id_number: encounter.patient.id_number,
        age: encounter.patient.age
      },
      prescription_number: `SYN-${batch}-${String(index + 1).padStart(4, '0')}`,
      date_issued: addMinutes(encounter.date, 8),
      valid_until: new Date(encounter.date.getTime() + 30 * 24 * 60 * 60 * 1000),
      prescriber: { name: 'Dr. Synthetic Physician', role: 'PHYSICIAN', specialty: 'Internal Medicine' },
      items: [{
        medication_name: encounter.medication_administrations?.[0]?.medication_name || 'Acetaminophen',
        strength: '500 mg',
        form: 'Tablet',
        quantity: 20,
        unit: 'tablets',
        dosage: '500 mg',
        frequency: 'Every 8 hours as needed',
        route: 'oral',
        duration: '7 days',
        directions: 'Take one tablet by mouth every 8 hours as needed.',
        indication: encounter.patient.symptoms,
        status: 'DISPENSED',
        dispensed_at: encounter.date
      }],
      status: 'dispensed',
      submitted_at: addMinutes(encounter.date, 8),
      submitted_to_pharmacy: true,
      all_accepted: true,
      all_dispensed: true,
      pharmacy_notes: 'Synthetic historical ePrescription fulfilled for report navigation.'
    })), { ordered: true });
    let projections = 0;
    if (createProjection) {
      const projectionDocuments = documents.map((encounter) => ({
        hospitalId,
        encounterId: encounter.encounterId,
        patientId: encounter.patient.id_number,
        name: encounter.patient.name,
        surname: encounter.patient.surname,
        age: encounter.patient.age,
        gender: encounter.patient.gender,
        height_cm: encounter.patient.height_cm,
        mrn: encounter.patient.id_number,
        date: encounter.date,
        syntheticData: true,
        seedBatch: batch,
        admissionTime: encounter.date,
        esiLevel: encounter.triage.esi_level,
        acuityScore: Math.max(0, 100 - encounter.triage.esi_level * 10),
        chiefComplaint: encounter.patient.symptoms,
        symptoms: encounter.patient.symptoms,
        duration: encounter.patient.duration,
        history: encounter.patient.history,
        painLevel: encounter.patient.painLevel,
        criticalHistoryStatus: encounter.patient.criticalHistoryStatus,
        criticalHistory: encounter.patient.criticalHistory,
        criticalHistoryReview: encounter.patient.criticalHistoryReview,
        latestVitals: {
          temperature: encounter.vitals.temp,
          heartRate: encounter.vitals.hr,
          respiratoryRate: encounter.vitals.resp,
          bpLeft: { systolic: encounter.vitals.bp_systolic, diastolic: encounter.vitals.bp_diastolic },
          o2Saturation: encounter.vitals.o2,
          recordedAt: encounter.vitals.recordedAt
        },
        sourceMetadata: { synthetic: true, seedBatch: batch, source: 'seed-patient-history-stress' }
      }));
      const createdProjections = await PatientEHRRecord.insertMany(projectionDocuments, { ordered: true });
      projections = createdProjections.length;
    }
    console.log(JSON.stringify({ synthetic: true, batch, hospitalId: String(hospitalId), patients: count, encountersPerPatient, activeQueue, historicalEncounters: historicalEncounters.length, inserted: inserted.length, prescriptions: prescriptions.length, projections, projectionEnabled: createProjection }, null, 2));
  } finally {
    await mongoose.disconnect();
  }
}

main().catch((error) => {
  console.error(`[seed-patient-history-stress] ${error.message}`);
  process.exitCode = 1;
});
