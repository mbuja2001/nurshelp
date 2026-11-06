import test from 'node:test';
import assert from 'node:assert/strict';
import { Encounter } from '../models/encounter.model.js';
import {
  acknowledgeLabRequest,
  canAccessLabEncounter,
  canManageSpecialtyRequest
} from './lab.controller.js';

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

test('only the matching specialist role can manage a specialty request', () => {
  assert.equal(canManageSpecialtyRequest({ user: { role: 'RADIOLOGIST' } }, 'RADIOLOGY'), true);
  assert.equal(canManageSpecialtyRequest({ user: { role: 'PATHOLOGIST' } }, 'PATHOLOGY'), true);
  assert.equal(canManageSpecialtyRequest({ user: { role: 'PATHOLOGIST' } }, 'RADIOLOGY'), false);
  assert.equal(canManageSpecialtyRequest({ user: { role: 'RADIOLOGY_TECHNICIAN' } }, 'RADIOLOGY'), false);
});

test('allows a radiologist to accept a pending radiology request', async () => {
  const labRequest = {
    request_id: { toString: () => 'request-1' },
    specialty: 'RADIOLOGY',
    lab_type: 'Chest X-ray',
    status: 'PENDING'
  };
  const encounter = {
    hospitalId: 'hospital-1',
    assignedPhysicianId: 'physician-1',
    lab_requests: [labRequest],
    status_history: [],
    save: async () => {}
  };
  const originalFindById = Encounter.findById;
  Encounter.findById = async () => encounter;

  const response = {
    statusCode: null,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(body) {
      this.body = body;
      return this;
    }
  };

  try {
    await acknowledgeLabRequest({
      params: { id: 'encounter-1', requestId: 'request-1' },
      user: {
        id: 'radiologist-1',
        hospitalId: 'hospital-1',
        role: 'RADIOLOGIST',
        name: 'QA Radiologist'
      }
    }, response);
  } finally {
    Encounter.findById = originalFindById;
  }

  assert.equal(response.statusCode, 200);
  assert.equal(response.body.status, 'ACKNOWLEDGED');
  assert.equal(labRequest.assigned_technician, 'QA Radiologist');
});
