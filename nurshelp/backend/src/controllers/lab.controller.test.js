import test from 'node:test';
import assert from 'node:assert/strict';
import { canAccessLabEncounter } from './lab.controller.js';

test('allows same-hospital physicians to access encounter lab requests', () => {
  const encounter = {
    _id: 'enc-1',
    hospitalId: 'hospital-1',
    assignedPhysicianId: 'physician-1',
    assigned_clinicians: [],
    clinical_handovers: []
  };

  const req = {
    user: {
      id: 'physician-1',
      hospitalId: 'hospital-1',
      role: 'PHYSICIAN'
    }
  };

  assert.equal(canAccessLabEncounter(encounter, req), true);
});

test('denies access when hospital does not match', () => {
  const encounter = {
    _id: 'enc-2',
    hospitalId: 'hospital-1',
    assignedPhysicianId: 'physician-2',
    assigned_clinicians: [],
    clinical_handovers: []
  };

  const req = {
    user: {
      id: 'physician-2',
      hospitalId: 'hospital-2',
      role: 'PHYSICIAN'
    }
  };

  assert.equal(canAccessLabEncounter(encounter, req), false);
});

test('allows same-hospital specialists to access encounter lab requests', () => {
  const encounter = {
    _id: 'enc-3',
    hospitalId: 'hospital-1',
    assignedPhysicianId: 'physician-2',
    assigned_clinicians: [],
    clinical_handovers: []
  };

  const req = {
    user: {
      id: 'specialist-1',
      hospitalId: 'hospital-1',
      role: 'RADIOLOGIST'
    }
  };

  assert.equal(canAccessLabEncounter(encounter, req), true);
});
