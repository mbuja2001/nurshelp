import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPendingRequestFromPrescription, mapPrescriptionStatusForDecision } from './pharmacy.controller.js';

test('maps approval decisions to valid prescription statuses', () => {
  assert.equal(mapPrescriptionStatusForDecision('APPROVED'), 'approved');
  assert.equal(mapPrescriptionStatusForDecision('DECLINED'), 'cancelled');
  assert.equal(mapPrescriptionStatusForDecision('DISPENSED'), 'dispensed');
});

test('buildPendingRequestFromPrescription maps a physician prescription into a pharmacy request payload', () => {
  const prescription = {
    _id: 'presc-123',
    encounterId: 'enc-456',
    patientId: 'P-001',
    medicationName: 'Metformin',
    dosage: '500mg',
    frequency: 'BD',
    route: 'Oral',
    special_instructions: 'Take with food',
    patient_details: {
      name: 'Jane Doe',
      id_number: 'MRN-100'
    },
    prescriber: {
      name: 'Dr. Patel'
    },
    items: [
      {
        medication_name: 'Metformin',
        dosage: '500mg',
        frequency: 'BD',
        route: 'Oral',
        indication: 'Diabetes control'
      }
    ],
    date_issued: '2025-01-01T00:00:00.000Z'
  };

  const result = buildPendingRequestFromPrescription(prescription);

  assert.equal(result._id, 'presc-123');
  assert.equal(result.encounterId, 'enc-456');
  assert.equal(result.patientName, 'Jane Doe');
  assert.equal(result.patientId, 'MRN-100');
  assert.equal(result.medicationName, 'Metformin');
  assert.equal(result.dosage, '500mg');
  assert.equal(result.frequency, 'BD');
  assert.equal(result.route, 'Oral');
  assert.equal(result.indication, 'Diabetes control');
  assert.equal(result.requestedBy, 'Dr. Patel');
  assert.equal(result.prescriptionId, 'presc-123');
  assert.equal(result.status, 'PENDING');
});
