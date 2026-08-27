import { buildHistoryWindows, buildInferenceHistoryPayload } from '../src/utils/longitudinalPatientHistory.js';

const asOf = new Date('2026-08-25T12:00:00.000Z');

export const syntheticPatient = {
  id: 'SYNTH-001',
  name: 'Synthetic Patient',
  encounters: [
    {
      id: 'enc-current',
      date: '2026-08-25T10:00:00.000Z',
      symptoms: ['trauma', 'bleeding'],
      diagnoses: ['traumatic injury'],
      interventions: ['hemorrhage control'],
      vitals: { systolic_bp: 72, hr: 138, resp: 28 },
      mobility: 'Immobile',
      traumaPresent: true,
      tews: { total: 9, priority: 'red' },
      disposition: 'ICU admission'
    },
    {
      id: 'enc-recent',
      date: '2026-08-23T08:00:00.000Z',
      symptoms: ['trauma', 'pain'],
      diagnoses: ['traumatic injury'],
      interventions: ['imaging'],
      vitals: { systolic_bp: 94, hr: 112, resp: 22 },
      mobility: 'With help',
      traumaPresent: true,
      tews: { total: 6, priority: 'orange' },
      disposition: 'Emergency discharge'
    },
    {
      id: 'enc-10d',
      date: '2026-08-15T08:00:00.000Z',
      symptoms: ['hypertension'],
      diagnoses: ['hypertension'],
      interventions: ['medication adjustment'],
      vitals: { systolic_bp: 128, hr: 82, resp: 16 },
      mobility: 'Walks',
      traumaPresent: false,
      tews: { total: 2, priority: 'green' },
      disposition: 'Discharged'
    },
    {
      id: 'enc-30d',
      date: '2026-07-26T08:00:00.000Z',
      symptoms: ['hypertension'],
      diagnoses: ['hypertension'],
      interventions: ['medication adjustment'],
      vitals: { systolic_bp: 136, hr: 88, resp: 18 },
      mobility: 'Walks',
      traumaPresent: false,
      tews: { total: 3, priority: 'yellow' },
      disposition: 'Discharged'
    },
    {
      id: 'enc-60d',
      date: '2026-06-26T08:00:00.000Z',
      symptoms: ['trauma'],
      diagnoses: ['traumatic injury'],
      interventions: ['wound care'],
      vitals: { systolic_bp: 110, hr: 96, resp: 20 },
      mobility: 'With help',
      traumaPresent: true,
      tews: { total: 7, priority: 'red' },
      disposition: 'Admission'
    },
    {
      id: 'enc-old',
      date: '2026-01-01T08:00:00.000Z',
      symptoms: ['old complaint'],
      diagnoses: ['resolved condition'],
      interventions: ['observation'],
      vitals: { systolic_bp: 120, hr: 80, resp: 16 },
      mobility: 'Walks',
      traumaPresent: false,
      tews: { total: 1, priority: 'green' },
      disposition: 'Discharged'
    },
    {
      id: 'enc-critical',
      date: '2024-02-01T10:00:00.000Z',
      symptoms: ['allergic reaction'],
      diagnoses: ['anaphylaxis'],
      interventions: ['epinephrine'],
      vitals: { systolic_bp: 82, hr: 118, resp: 26 },
      mobility: 'Immobile',
      traumaPresent: false,
      criticalFlag: true,
      criticalFacts: [{ type: 'allergy', code: 'penicillin', description: 'Penicillin anaphylaxis', severity: 'life-threatening', verified: true }],
      tews: { total: 8, priority: 'red' },
      disposition: 'ICU admission'
    }
  ]
};

export const runSimulation = () => {
  const { encounters, ...patient } = syntheticPatient;
  const windows = buildHistoryWindows({ encounters, currentEncounter: encounters[0], asOf });
  const payload = buildInferenceHistoryPayload(windows, { ...patient, ...encounters[0] }, 12000);
  return { windows, payload };
};

export const formatSimulationTimestamp = (value) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Invalid date';
  return {
    iso: date.toISOString(),
    date: date.toISOString().slice(0, 10),
    time: date.toISOString().slice(11, 19),
    timezone: 'UTC'
  };
};

if (import.meta.url === `file://${process.argv[1].replaceAll('\\', '/')}`) {
  const { windows, payload } = runSimulation();
  console.log(JSON.stringify({
    recent_encounters: windows.recent_window.encounters.map((encounter) => ({
      id: encounter.id,
      occurred_at: formatSimulationTimestamp(encounter.date)
    })),
    trend_window: {
      from: formatSimulationTimestamp(windows.trend_window.from),
      to: formatSimulationTimestamp(windows.trend_window.to)
    },
    trend_summary: windows.trend_window.summary,
    relevant_encounters: windows.relevant_history.map((event) => event.id),
    inference_payload_characters: JSON.stringify(payload).length,
    truncated: payload.metadata.truncated
  }, null, 2));
}
