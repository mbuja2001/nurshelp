import test from 'node:test';
import assert from 'node:assert/strict';
import { buildDiagnosticEhrContext } from './diagnosticEhrContext.js';

test('captures current encounter and prior discharged EHR context for diagnostic requests', () => {
  const capturedAt = new Date('2026-10-13T12:00:00.000Z');
  const context = buildDiagnosticEhrContext({
    capturedAt,
    encounter: {
      _id: 'current-encounter',
      date: capturedAt,
      status: 'AWAITING_PHYSICIAN',
      diagnosis: 'Pneumonia',
      patient: {
        id_number: 'SYN-patient-1',
        name: 'Amina',
        surname: 'Okafor',
        age: 42,
        gender: 'F',
        symptoms: 'Shortness of breath'
      },
      vitals: { hr: 96 }
    },
    historicalEncounters: [{
      _id: 'prior-encounter',
      date: new Date('2026-09-20T12:00:00.000Z'),
      status: 'DISCHARGED',
      diagnosis: 'Asthma exacerbation',
      patient: { symptoms: 'Wheezing', criticalHistory: { allergies: ['Penicillin'] } }
    }]
  });

  assert.equal(context.version, 1);
  assert.equal(context.capturedAt, capturedAt);
  assert.equal(context.patient.id, 'SYN-patient-1');
  assert.equal(context.current_encounter._id, 'current-encounter');
  assert.equal(context.metadata.source_encounter_count, 1);
  assert.ok(context.relevant_history.some((visit) => visit.diagnoses.includes('Asthma exacerbation')));
  assert.ok(context.relevant_history.some((visit) => visit.patient.criticalHistory.allergies.includes('Penicillin')));
});
