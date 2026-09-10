import test from 'node:test';
import assert from 'node:assert/strict';
import { canViewSurgery, sortOperationsEvents } from './operationsFeed.js';

test('operations alerts sort by severity, wait, creation time, then id', () => {
  const events = sortOperationsEvents([
    { id: 'info-new', severity: 'INFO', createdAt: '2026-09-04T12:00:00Z' },
    { id: 'critical-old', severity: 'CRITICAL', createdAt: '2026-09-04T11:00:00Z' },
    { id: 'high-wait', severity: 'HIGH', priority: { waitMs: 600000 }, createdAt: '2026-09-04T10:00:00Z' },
    { id: 'high-new', severity: 'HIGH', priority: { waitMs: 0 }, createdAt: '2026-09-04T12:01:00Z' }
  ]);

  assert.deepEqual(events.map((event) => event.id), ['critical-old', 'high-wait', 'high-new', 'info-new']);
});

test('surgery visibility requires assignment or explicit team involvement', () => {
  const encounter = {
    assigned_clinicians: [{ id: 'clinician-a', role: 'PHYSICIAN' }],
    assignedPhysicianId: 'physician-a'
  };
  const surgery = { surgeon: 'Dr. Operating Lead', surgical_team: ['Nurse Theatre A'] };

  assert.equal(canViewSurgery(encounter, surgery, { _id: 'clinician-a', role: 'PHYSICIAN' }), true);
  assert.equal(canViewSurgery(encounter, surgery, { _id: 'clinician-b', role: 'PHYSICIAN', name: 'Dr. Unrelated' }), false);
  assert.equal(canViewSurgery(encounter, surgery, { _id: 'clinician-b', role: 'PHYSICIAN', name: 'Dr. Operating Lead' }), true);
  assert.equal(canViewSurgery(encounter, surgery, { _id: 'manager-a', role: 'MANAGER' }), true);
});
