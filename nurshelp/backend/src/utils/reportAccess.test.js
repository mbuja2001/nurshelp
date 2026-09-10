import test from 'node:test';
import assert from 'node:assert/strict';
import { ACCESS, REPORT_TYPES, getReportAccessDecision } from './reportAccess.js';

const encounter = {
  hospitalId: 'hospital-1',
  nurse_id: 'triage-1',
  assignedNurseId: 'clinical-1',
  assignedPhysicianId: 'physician-1',
  assigned_clinicians: [{ id: 'physician-1', role: 'PHYSICIAN' }]
};

const user = (id, role, hospitalId = 'hospital-1') => ({ _id: id, role, hospitalId });

test('triage nurse can read assigned triage report but not discharge', () => {
  const triage = getReportAccessDecision({
    user: user('triage-1', 'TRIAGE_NURSE'),
    encounter,
    reportType: REPORT_TYPES.TRIAGE
  });
  const discharge = getReportAccessDecision({
    user: user('triage-1', 'TRIAGE_NURSE'),
    encounter,
    reportType: REPORT_TYPES.DISCHARGE
  });

  assert.equal(triage.allowed, true);
  assert.equal(triage.visibility, ACCESS.FULL);
  assert.equal(discharge.allowed, false);
});

test('clinical nurse gets full read access but cannot edit prescriptions', () => {
  const read = getReportAccessDecision({
    user: user('clinical-1', 'CLINICAL_NURSE'),
    encounter,
    reportType: REPORT_TYPES.PRESCRIPTION
  });
  const edit = getReportAccessDecision({
    user: user('clinical-1', 'CLINICAL_NURSE'),
    encounter,
    reportType: REPORT_TYPES.PRESCRIPTION,
    action: 'edit'
  });

  assert.equal(read.allowed, true);
  assert.equal(read.visibility, ACCESS.FULL);
  assert.equal(edit.allowed, false);
});

test('assigned physician can edit surgery and prescription reports', () => {
  for (const reportType of [REPORT_TYPES.SURGERY, REPORT_TYPES.PRESCRIPTION]) {
    const decision = getReportAccessDecision({
      user: user('physician-1', 'PHYSICIAN'),
      encounter,
      reportType,
      action: 'edit'
    });
    assert.equal(decision.allowed, true);
  }
});

test('same-hospital but unassigned physician is denied', () => {
  const decision = getReportAccessDecision({
    user: user('physician-2', 'PHYSICIAN'),
    encounter,
    reportType: REPORT_TYPES.TRIAGE
  });

  assert.equal(decision.allowed, false);
  assert.equal(decision.reason, 'encounter_assignment_required');
});

test('cross-hospital access is denied even for an assigned id', () => {
  const decision = getReportAccessDecision({
    user: user('physician-1', 'PHYSICIAN', 'hospital-2'),
    encounter,
    reportType: REPORT_TYPES.SURGERY
  });

  assert.equal(decision.allowed, false);
  assert.equal(decision.reason, 'hospital_mismatch');
});
