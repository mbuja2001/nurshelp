import test from 'node:test';
import assert from 'node:assert/strict';
import { canAccessEncounter } from './encounterAccess.js';

test('allows access when the user is the assigned physician', () => {
  const encounter = {
    nurse_id: 'nurse-1',
    assignedPhysicianId: 'physician-2',
    assigned_clinicians: [],
    clinical_handovers: []
  };

  const user = { _id: 'physician-2', role: 'PHYSICIAN' };

  assert.equal(canAccessEncounter(encounter, user), true);
});

test('allows access when the user is listed in clinical handovers', () => {
  const encounter = {
    nurse_id: 'nurse-1',
    assignedPhysicianId: null,
    assigned_clinicians: [],
    clinical_handovers: [{ clinician_id: 'physician-9' }]
  };

  const user = { _id: 'physician-9', role: 'PHYSICIAN' };

  assert.equal(canAccessEncounter(encounter, user), true);
});

test('allows a physician to view a hospitalized encounter when it is part of the active clinical workflow', () => {
  const encounter = {
    nurse_id: 'nurse-1',
    assignedPhysicianId: null,
    assigned_clinicians: [],
    clinical_handovers: [],
    status: 'HOSPITALIZED'
  };

  const user = { _id: 'physician-99', role: 'PHYSICIAN' };

  assert.equal(canAccessEncounter(encounter, user), true);
});

test('allows access when the user is the triage-assigned physician', () => {
  const encounter = {
    nurse_id: 'nurse-1',
    assignedPhysicianId: null,
    assigned_physician: { id: 'physician-5', name: 'Dr. Adams' },
    assigned_clinicians: [],
    clinical_handovers: [],
    status: 'TRIAGE_IN_PROGRESS'
  };

  const user = { _id: 'physician-5', role: 'PHYSICIAN' };

  assert.equal(canAccessEncounter(encounter, user), true);
});

test('denies access when the user is not assigned to the encounter', () => {
  const encounter = {
    nurse_id: 'nurse-1',
    assignedPhysicianId: 'physician-2',
    assigned_clinicians: [],
    clinical_handovers: [],
    status: 'TRIAGE_IN_PROGRESS'
  };

  const user = { _id: 'physician-99', role: 'PHYSICIAN' };

  assert.equal(canAccessEncounter(encounter, user), false);
});
