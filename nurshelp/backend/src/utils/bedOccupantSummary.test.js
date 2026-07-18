import test from 'node:test';
import assert from 'node:assert/strict';
import { buildOccupantSummary, extractPatientDisplayName } from './bedOccupantSummary.js';

test('extractPatientDisplayName prefers explicit patient name fields', () => {
  const encounter = {
    patientName: 'Jane Doe',
    patient: { name: 'Wrong Name', firstName: 'Wrong' },
    firstName: 'Also Wrong'
  };

  assert.equal(extractPatientDisplayName(encounter), 'Jane Doe');
});

test('buildOccupantSummary falls back to Unknown when no patient name is available', () => {
  const summary = buildOccupantSummary({ currentEncounterId: 'enc-123' }, null);

  assert.equal(summary._id, 'enc-123');
  assert.equal(summary.patientName, 'Unknown');
});

test('buildOccupantSummary uses nested patient info and diagnosis', () => {
  const summary = buildOccupantSummary(
    { currentEncounterId: 'enc-123' },
    {
      _id: 'enc-123',
      patient: { name: 'John Smith', age: 40 },
      diagnosis: 'Chest pain',
      severity: 2,
      triage: { aiTriageReport: { esi_level: 1 } },
      status: 'DIAGNOSED'
    }
  );

  assert.equal(summary.patientName, 'John Smith');
  assert.equal(summary.diagnosis, 'Chest pain');
  assert.equal(summary.severity, 2);
  assert.equal(summary.status, 'DIAGNOSED');
});

test('buildOccupantSummary includes persisted vitals for the bed modal', () => {
  const summary = buildOccupantSummary(
    { currentEncounterId: 'enc-123' },
    {
      _id: 'enc-123',
      patient: { name: 'Alice Brown', age: 31 },
      vitals: { temp: 37.2, hr: 98, o2: 96, bp: '120/78' },
      diagnosis: 'Asthma exacerbation'
    }
  );

  assert.deepEqual(summary.vitals, { temp: 37.2, hr: 98, o2: 96, bp: '120/78' });
  assert.equal(summary.diagnosis, 'Asthma exacerbation');
});

test('buildOccupantSummary preserves the latest persisted diagnosis and vitals for clinical review', () => {
  const summary = buildOccupantSummary(
    { currentEncounterId: 'enc-456' },
    {
      _id: 'enc-456',
      patientName: 'Grace Hopper',
      vitals: { temp: 38.1, hr: 112, o2: 94, bp: '102/68' },
      diagnosis: 'Septic shock'
    }
  );

  assert.equal(summary.patientName, 'Grace Hopper');
  assert.equal(summary.diagnosis, 'Septic shock');
  assert.deepEqual(summary.vitals, { temp: 38.1, hr: 112, o2: 94, bp: '102/68' });
});
