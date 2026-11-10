import test from 'node:test';
import assert from 'node:assert/strict';
import { Encounter } from '../models/encounter.model.js';
import {
  acknowledgeLabRequest,
  canAccessLabEncounter,
  canManageSpecialtyRequest,
  recordRadiologyWorkflowEvent,
  getLabRequestsForSpecialty
} from './lab.controller.js';

const makeResponse = () => ({
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
});

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

test('records an ordered, auditable radiology status from an assigned specialty technician', async () => {
  const labRequest = {
    request_id: { toString: () => 'request-1' },
    specialty: 'RADIOLOGY',
    status: 'ACKNOWLEDGED',
    imagingWorkflow: { stage: undefined, events: [] }
  };
  const encounter = {
    hospitalId: 'hospital-1',
    lab_requests: [labRequest],
    status_history: [],
    save: async () => {}
  };
  const originalFindById = Encounter.findById;
  Encounter.findById = async () => encounter;
  const response = makeResponse();

  try {
    await recordRadiologyWorkflowEvent({
      params: { encounterId: 'encounter-1', requestId: 'request-1' },
      body: { stage: 'PATIENT_ARRIVED' },
      user: {
        id: 'technician-1',
        hospitalId: 'hospital-1',
        role: 'TECHNICIAN',
        specialty: 'Radiology',
        name: 'QA Technician'
      }
    }, response);
  } finally {
    Encounter.findById = originalFindById;
  }

  assert.equal(response.statusCode, 200);
  assert.equal(labRequest.imagingWorkflow.stage, 'PATIENT_ARRIVED');
  assert.equal(labRequest.imagingWorkflow.events[0].recorded_by, 'QA Technician');
  assert.equal(labRequest.imagingWorkflow.events[0].source, 'MANUAL');
  assert.equal(encounter.status_history[0].status, 'RADIOLOGY_WORKFLOW_PATIENT_ARRIVED');
});

test('rejects skipped radiology stages and non-HTTPS viewer URLs', async () => {
  const labRequest = {
    request_id: { toString: () => 'request-1' },
    specialty: 'RADIOLOGY',
    status: 'ACKNOWLEDGED',
    imagingWorkflow: { stage: undefined, events: [] }
  };
  const encounter = {
    hospitalId: 'hospital-1',
    lab_requests: [labRequest],
    status_history: [],
    save: async () => {}
  };
  const originalFindById = Encounter.findById;
  Encounter.findById = async () => encounter;
  const technician = {
    params: { encounterId: 'encounter-1', requestId: 'request-1' },
    user: {
      id: 'technician-1',
      hospitalId: 'hospital-1',
      role: 'TECHNICIAN',
      specialty: 'Radiology',
      name: 'QA Technician'
    }
  };

  try {
    const skippedStageResponse = makeResponse();
    await recordRadiologyWorkflowEvent({
      ...technician,
      body: { stage: 'IMAGES_AVAILABLE' }
    }, skippedStageResponse);
    assert.equal(skippedStageResponse.statusCode, 409);

    const insecureUrlResponse = makeResponse();
    await recordRadiologyWorkflowEvent({
      ...technician,
      body: { stage: 'IMAGES_AVAILABLE', viewer_url: 'http://viewer.example/study' }
    }, insecureUrlResponse);
    assert.equal(insecureUrlResponse.statusCode, 400);
    assert.equal(labRequest.imagingWorkflow.events.length, 0);
  } finally {
    Encounter.findById = originalFindById;
  }
});

test('exposes only workflow details to a radiology technician queue', async () => {
  const encounter = {
    _id: 'encounter-1',
    hospitalId: 'hospital-1',
    patient: { name: 'Synthetic Patient', age: 38, id_number: 'QA-001' },
    diagnosis: 'Sensitive diagnosis',
    vitals: { temperature: 38 },
    lab_requests: [{
      request_id: { toString: () => 'request-1' },
      specialty: 'RADIOLOGY',
      status: 'ACKNOWLEDGED',
      lab_type: 'Chest X-ray',
      report_content: 'Restricted report text',
      clinical_impression: 'Restricted impression',
      ehrContext: { summary: 'Restricted history' },
      imagingWorkflow: { stage: 'PATIENT_ARRIVED', events: [] }
    }]
  };
  const originalFind = Encounter.find;
  Encounter.find = () => ({ lean: async () => [encounter] });
  const response = makeResponse();

  try {
    await getLabRequestsForSpecialty({
      query: { specialty: 'RADIOLOGY' },
      user: {
        id: 'technician-1',
        hospitalId: 'hospital-1',
        role: 'TECHNICIAN',
        specialty: 'Radiology'
      }
    }, response);
  } finally {
    Encounter.find = originalFind;
  }

  assert.equal(response.statusCode, 200);
  const [queueItem] = response.body.lab_requests;
  assert.equal(queueItem.patientId, 'QA-001');
  assert.equal(queueItem.imagingWorkflow.stage, 'PATIENT_ARRIVED');
  assert.equal('report_content' in queueItem, false);
  assert.equal('clinical_impression' in queueItem, false);
  assert.equal('ehrContext' in queueItem, false);
  assert.equal('diagnosis' in queueItem, false);
  assert.equal('vitals' in queueItem, false);
});
