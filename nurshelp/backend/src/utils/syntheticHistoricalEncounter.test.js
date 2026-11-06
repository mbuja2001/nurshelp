import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createSyntheticHistoricalEncounter,
  historicalEncounterDate,
  normalizeHistoricalDays
} from './syntheticHistoricalEncounter.js';

test('normalizes the explicit 90-day default and rejects invalid history windows', () => {
  assert.equal(normalizeHistoricalDays(), 90);
  assert.equal(normalizeHistoricalDays('365'), 365);
  assert.throws(() => normalizeHistoricalDays(0), RangeError);
  assert.throws(() => normalizeHistoricalDays(3651), RangeError);
  assert.throws(() => normalizeHistoricalDays(1.5), RangeError);
});

test('places synthetic encounter dates strictly inside the selected historical window', () => {
  const now = new Date('2026-10-13T12:00:00.000Z');
  const start = now.getTime() - 90 * 24 * 60 * 60 * 1000;
  const dates = Array.from({ length: 4 }, (_, index) => historicalEncounterDate({
    index,
    count: 4,
    historicalDays: 90,
    now
  }));

  assert.equal(dates[0].getTime(), now.getTime() - 23 * 24 * 60 * 60 * 1000);
  assert.equal(dates.at(-1).getTime(), start);
  assert.ok(dates.every((date) => date.getTime() >= start && date.getTime() < now.getTime()));
});

test('creates discharged historical records without assigning staff or creating triage state', () => {
  const encounter = createSyntheticHistoricalEncounter({
    hospitalId: 'hospital-1',
    actorId: 'admin-1',
    batch: 'admin-123',
    encounterDate: new Date('2026-10-01T12:00:00.000Z'),
    patientId: 'SYN-admin-123-1',
    patientName: 'Synthetic Patient',
    complaint: 'Fever and generalized weakness'
  });

  assert.equal(encounter.status, 'DISCHARGED');
  assert.equal(encounter.isWaiting, false);
  assert.equal(encounter.syntheticData, true);
  assert.equal(encounter.patient.id_number, 'SYN-admin-123-1');
  assert.equal(encounter.status_history[0].status, 'DISCHARGED');
  assert.equal('nurse_id' in encounter, false);
  assert.equal('assignedNurseId' in encounter, false);
  assert.equal('assignedPhysicianId' in encounter, false);
  assert.equal('triage' in encounter, false);
  assert.equal('ai_triage_report' in encounter, false);
  assert.equal('clinical_handovers' in encounter, false);
});
