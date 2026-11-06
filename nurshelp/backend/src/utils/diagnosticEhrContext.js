import { buildLongitudinalHistoryContext } from './longitudinalPatientHistory.js';

const historyRecord = (encounter) => ({
  _id: String(encounter._id),
  date: encounter.date,
  status: encounter.status,
  diagnoses: encounter.diagnosis ? [encounter.diagnosis] : [],
  symptoms: encounter.patient?.symptoms || '',
  interventions: encounter.interventions || [],
  disposition: encounter.disposition || '',
  vitals: encounter.vitals || {},
  SATS_reasoning: encounter.SATS_reasoning || {},
  patient: {
    age: encounter.patient?.age,
    gender: encounter.patient?.gender,
    history: encounter.patient?.history || '',
    criticalHistoryStatus: encounter.patient?.criticalHistoryStatus || '',
    criticalHistory: encounter.patient?.criticalHistory || {}
  }
});

export const buildDiagnosticEhrContext = ({
  encounter,
  historicalEncounters = [],
  capturedAt = new Date()
}) => {
  const patient = encounter.patient || {};
  const currentEncounter = {
    _id: String(encounter._id),
    date: encounter.date,
    status: encounter.status,
    diagnoses: encounter.diagnosis ? [encounter.diagnosis] : [],
    symptoms: patient.symptoms || '',
    vitals: encounter.vitals || {},
    triage: {
      triageCategory: encounter.triage?.triageCategory,
      priority_code: encounter.triage?.priority_code,
      esi_level: encounter.triage?.esi_level
    },
    patient: {
      id_number: patient.id_number || '',
      name: patient.name || '',
      surname: patient.surname || '',
      age: patient.age ?? null,
      gender: patient.gender || '',
      history: patient.history || '',
      criticalHistoryStatus: patient.criticalHistoryStatus || '',
      criticalHistory: patient.criticalHistory || {}
    }
  };
  const history = buildLongitudinalHistoryContext({
    encounters: historicalEncounters.map(historyRecord),
    currentEncounter,
    asOf: capturedAt,
    maxCharacters: 6000
  });

  return {
    version: 1,
    capturedAt,
    patient: {
      id: patient.id_number || '',
      name: [patient.name, patient.surname].filter(Boolean).join(' '),
      age: patient.age ?? null,
      gender: patient.gender || ''
    },
    ...history.payload
  };
};
