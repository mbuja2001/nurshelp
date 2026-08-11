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

  const report = buildDischargeReportData({
    encounter,
    overrides: { followUpInstructions: 'Follow up with PCP in 7 days' }
  });

  assert.ok(report.reportId.startsWith('DR-'));
  assert.equal(report.patientId, 'MRN-100');
  assert.equal(report.primaryDiagnosis, 'Community-acquired pneumonia');
  assert.equal(report.procedures[0].procedureName, 'Laparoscopy');
  assert.equal(report.medications[0].name, 'Amoxicillin');
  assert.equal(report.followUpInstructions, 'Follow up with PCP in 7 days');
  assert.equal(report.authorId, 'Dr Patel');
  assert.equal(report.status, 'draft');
});
