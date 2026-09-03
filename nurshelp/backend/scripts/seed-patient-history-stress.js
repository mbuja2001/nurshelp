#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';
import PatientEHRRecord from '../src/models/patientEHRRecord.model.js';
import { Hospital } from '../src/models/hospital.model.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const DEFAULT_COUNT = 10;
const DEFAULT_ENCOUNTERS = 12;
const DEFAULT_BATCH = `history-stress-${new Date().toISOString().slice(0, 10)}`;
const BASE_DATE = new Date('2026-08-25T12:00:00.000Z');

const scenarios = [
  { symptoms: 'chest pain and shortness of breath', history: 'hypertension', status: 'reported', critical: { allergies: ['Peanuts'], adverseReactions: ['Anaphylaxis'], medications: ['Warfarin'], conditions: ['Severe haemophilia A'], implantsOrProcedures: ['Recent splenectomy'] }, esi: 2 },
  { symptoms: 'fever and productive cough', history: 'COPD', status: 'none_established', critical: {}, esi: 3 },
  { symptoms: 'abdominal pain and vomiting', history: 'previous appendicitis', status: 'reported', critical: { allergies: ['Penicillin'], adverseReactions: ['Severe rash'], medications: [], conditions: [], implantsOrProcedures: [] }, esi: 3 },
  { symptoms: 'weakness and dizziness', history: 'atrial fibrillation', status: 'reported', critical: { allergies: [], adverseReactions: [], medications: ['Apixaban'], conditions: ['Atrial fibrillation'], implantsOrProcedures: [] }, esi: 2 }
];

const getArg = (name, fallback) => {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
};

const numberArg = (name, fallback) => Math.max(1, Number.parseInt(getArg(name, fallback), 10) || fallback);
const hasFlag = (name) => process.argv.includes(`--${name}`);

const makeEncounter = ({ patientIndex, encounterIndex, batch, hospitalId, count }) => {
  const scenario = scenarios[(patientIndex + encounterIndex) % scenarios.length];
  const patientId = `SYNTH-HISTORY-${String(patientIndex + 1).padStart(6, '0')}`;
  const encounterId = `${batch}-${patientId}-${String(encounterIndex + 1).padStart(3, '0')}`;
  const date = new Date(BASE_DATE.getTime() - encounterIndex * 10 * 24 * 60 * 60 * 1000 - (patientIndex % 5) * 60 * 60 * 1000);
  const criticalHistory = scenario.status === 'reported' ? scenario.critical : {};

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
      recordedAt: date
    },
    triage: { esi_level: scenario.esi, synthetic: true, seedBatch: batch },
    severity: scenario.esi,
    syntheticData: true,
    seedBatch: batch,
    seedPatientCount: count,
    status: 'TRIAGE_IN_PROGRESS',
    submittedAt: date,
    attendedAt: date
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
  const hospitalArg = getArg('hospital-id', process.env.HOSPITAL_ID || '');
  const createProjection = hasFlag('projection');

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
        documents.push(makeEncounter({ patientIndex, encounterIndex, batch, hospitalId, count }));
      }
    }

    const inserted = await Encounter.insertMany(documents, { ordered: true });
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
    console.log(JSON.stringify({ synthetic: true, batch, hospitalId: String(hospitalId), patients: count, encountersPerPatient, inserted: inserted.length, projections, projectionEnabled: createProjection }, null, 2));
  } finally {
    await mongoose.disconnect();
  }
}

main().catch((error) => {
  console.error(`[seed-patient-history-stress] ${error.message}`);
  process.exitCode = 1;
});
