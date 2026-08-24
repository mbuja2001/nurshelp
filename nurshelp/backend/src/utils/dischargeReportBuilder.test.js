import test from 'node:test';
import assert from 'node:assert/strict';
import { buildDischargeReportData } from './dischargeReportBuilder.js';

test('builds a discharge report payload from encounter data', () => {
  const encounter = {
    _id: 'enc-1',
    patient: { id_number: 'MRN-100', name: 'Jane', surname: 'Doe' },
    diagnosis: 'Community-acquired pneumonia',
    treatment_plan: 'Oral antibiotics and rest',
    medication_administrations: [
      { medication_name: 'Amoxicillin', dosage: '500mg', frequency: 'PO BID', route: 'Oral' }
    ],
    surgery_records: [
      { procedure_name: 'Laparoscopy', performed_at: '2026-08-01T10:00:00.000Z' }
    ],
    discharge_scheduled_at: '2026-08-02T12:00:00.000Z',
    assigned_physician: { name: 'Dr Patel' }
  };

  const clinician = {
    id: 'clinician-42',
    name: 'Dr Patel',
    role: 'PHYSICIAN',
    specialty: 'Internal Medicine',
    credentials: {
      title: 'MBChB',
      licenseNumber: 'MP-2024-77',
      npiNumber: '1234567890'
    }
  };

  const report = buildDischargeReportData({
    encounter,
    clinician,
    overrides: {
      followUpInstructions: 'Follow up with PCP in 7 days'
    }
  });

  assert.ok(report.reportId.startsWith('DR-'));
  assert.equal(report.patientId, 'MRN-100');
  assert.equal(report.primaryDiagnosis, 'Community-acquired pneumonia');
  assert.equal(report.procedures[0].procedureName, 'Laparoscopy');
  assert.equal(report.medications[0].name, 'Amoxicillin');
  assert.equal(report.followUpInstructions, 'Follow up with PCP in 7 days');
  assert.equal(report.authorId, 'Dr Patel');
  assert.equal(report.signedOffBy, 'Dr Patel');
  assert.ok(report.clinicianSignature);
  assert.equal(report.clinicianSignature.clinicianRole, 'PHYSICIAN');
  assert.equal(report.clinicianSignature.qualification, 'MBChB');
  assert.equal(report.clinicianSignature.licenseNumber, 'MP-2024-77');
  assert.equal(report.clinicianSignature.npiNumber, '1234567890');
  assert.ok(report.clinicianSignature.hash);
  assert.equal(report.clinicianSignature.documentType, 'DISCHARGE_REPORT');
  assert.equal(report.status, 'draft');
});
