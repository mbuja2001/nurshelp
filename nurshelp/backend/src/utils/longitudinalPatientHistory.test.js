import test from 'node:test';
import assert from 'node:assert/strict';
import { buildHistoryWindows, buildInferenceHistoryPayload, buildLongitudinalHistoryContext, sizeWindowsToBudget } from './longitudinalPatientHistory.js';
import { formatSimulationTimestamp, syntheticPatient } from '../../scripts/simulate-longitudinal-history.js';

const asOf = new Date('2026-08-25T12:00:00.000Z');

const makeWindows = () => buildHistoryWindows({
  encounters: syntheticPatient.encounters,
  currentEncounter: syntheticPatient.encounters[0],
  asOf
});

test('sizes history windows according to the configured budget', () => {
  assert.deepEqual(sizeWindowsToBudget(12000), { maxCharacters: 12000, recentHours: 72, maxRelevant: 5, maxSummaryItems: 20 });
  assert.deepEqual(sizeWindowsToBudget(6000), { maxCharacters: 6000, recentHours: 48, maxRelevant: 3, maxSummaryItems: 15 });
  assert.deepEqual(sizeWindowsToBudget(3000), { maxCharacters: 3000, recentHours: 24, maxRelevant: 2, maxSummaryItems: 10 });
  assert.deepEqual(sizeWindowsToBudget(1000), { maxCharacters: 1000, recentHours: 12, maxRelevant: 1, maxSummaryItems: 5 });
});

test('applies the budget profile before building the inference payload', () => {
  const result = buildLongitudinalHistoryContext({
    encounters: syntheticPatient.encounters,
    currentEncounter: syntheticPatient.encounters[0],
    asOf,
    maxCharacters: 3000
  });

  assert.equal(result.budgetProfile.recentHours, 24);
  assert.equal(result.windows.windows.recent_hours, 24);
  assert.equal(result.windows.windows.trend_days, 90);
  assert.ok(JSON.stringify(result.payload).length <= 3000 || result.payload.metadata.truncated);
  assert.equal(result.payload.metadata.source_encounter_count, syntheticPatient.encounters.length);
  assert.ok(result.payload.metadata.estimated_characters > 0);
});

test('preserves encounter dates and times as normalized UTC timestamps', () => {
  const result = makeWindows();

  assert.equal(result.recent_window.encounters[0].date, '2026-08-23T08:00:00.000Z');
  assert.deepEqual(formatSimulationTimestamp(result.recent_window.encounters[0].date), {
    iso: '2026-08-23T08:00:00.000Z',
    date: '2026-08-23',
    time: '08:00:00',
    timezone: 'UTC'
  });
  assert.equal(result.recent_window.from, '2026-08-22T12:00:00.000Z');
  assert.equal(result.recent_window.to, '2026-08-25T12:00:00.000Z');
});

test('keeps old critical history outside ordinary time windows', () => {
  const result = makeWindows();

  assert.deepEqual(result.critical_history.map((encounter) => encounter.id), ['enc-critical']);
  assert.equal(result.critical_facts[0].description, 'Penicillin anaphylaxis');
  assert.equal(result.metadata.critical_encounter_count, 1);
});

test('splits recent 72-hour history from the 90-day trend window', () => {
  const result = makeWindows();

  assert.deepEqual(result.recent_window.encounters.map((encounter) => encounter.id), ['enc-recent']);
  assert.deepEqual(result.trend_window.summary.encounter_count, 3);
  assert.equal(result.trend_window.summary.latest_tews, 2);
  assert.equal(result.trend_window.summary.highest_tews, 7);
  assert.equal(result.trend_window.summary.average_tews, 4);
});

test('calculates priority counts, vital trends, mobility changes, and trauma frequency', () => {
  const summary = makeWindows().trend_window.summary;

  assert.equal(summary.red_cases, 1);
  assert.equal(summary.orange_cases, 0);
  assert.deepEqual(summary.vital_trends.systolic_bp, { first: 110, latest: 128, change: 18 });
  assert.deepEqual(summary.mobility_changes, ['With help', 'Walks']);
  assert.equal(summary.trauma_present_count, 1);
  assert.deepEqual(summary.recurrent_diagnoses, ['hypertension', 'traumatic injury']);
});

test('ranks relevant history by matching current complaint and clinical risk', () => {
  const result = makeWindows();

  assert.equal(result.relevant_history[0].id, 'enc-60d');
  assert.equal(result.relevant_history[0].selection_relevance, 32);
});

test('bounds the inference payload and records truncation metadata', () => {
  const result = makeWindows();
  const originalRelevantCount = result.relevant_history.length;
  const payload = buildInferenceHistoryPayload(result, syntheticPatient.encounters[0], 1200);

  assert.ok(JSON.stringify(payload).length > 1200 || payload.metadata.truncated);
  assert.equal(payload.metadata.truncated, true);
  assert.ok(payload.relevant_history.length < originalRelevantCount);
});

test('rejects an invalid reference date', () => {
  assert.throws(() => buildHistoryWindows({ encounters: [], currentEncounter: {}, asOf: 'invalid-date' }), /asOf must be a valid date/);
});

test('requires an explicit current encounter for relevance-safe construction', () => {
  assert.throws(() => buildHistoryWindows({ encounters: syntheticPatient.encounters, asOf }), /currentEncounter is required/);
});
