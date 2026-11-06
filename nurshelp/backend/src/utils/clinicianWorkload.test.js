import test from 'node:test';
import assert from 'node:assert/strict';
import { buildCapacityAlerts, calculateClinicianWorkload, validateAcuityWeightOrdering, workloadQuery } from './clinicianWorkload.js';
import { buildCapacityAuditDetails, queuePriority, updateAcuityWeights, updateCapacity } from '../controllers/operations.controller.js';

const encounter = (id, status, assigned_clinicians, extra = {}) => ({ _id: id, status, assigned_clinicians, ...extra });

const physician = (id, name = id) => ({ id, name, role: 'PHYSICIAN' });
const triageNurse = (id, name = id) => ({ id, name, role: 'TRIAGE_NURSE' });
const clinicalNurse = (id, name = id) => ({ id, name, role: 'CLINICAL_NURSE' });

test('counts every assigned clinician but only once per encounter', () => {
  const result = calculateClinicianWorkload([
    encounter('e1', 'AWAITING_PHYSICIAN', [physician('p1'), physician('p1'), clinicalNurse('n1')]),
    encounter('e2', 'HOSPITALIZED', [physician('p1'), clinicalNurse('n1')])
  ]);

  assert.equal(result.lanes.physicians.find((entry) => entry.clinicianId === 'p1').activeCases, 1);
  assert.equal(result.lanes.clinicalNurses.find((entry) => entry.clinicianId === 'n1').activeCases, 1);
});

test('separates role lanes and applies role-specific statuses', () => {
  const result = calculateClinicianWorkload([
    encounter('triage', 'TRIAGE_IN_PROGRESS', [triageNurse('tn1')], { isWaiting: true }),
    encounter('ward', 'HOSPITALIZED', [clinicalNurse('cn1')]),
    encounter('doctor', 'AWAITING_PHYSICIAN', [physician('p1')])
  ]);

  assert.deepEqual(result.lanes.triageNurses.map((entry) => entry.clinicianId), ['tn1']);
  assert.deepEqual(result.lanes.clinicalNurses.map((entry) => entry.clinicianId), ['cn1']);
  assert.deepEqual(result.lanes.physicians.map((entry) => entry.clinicianId), ['p1']);
});

test('counts the triage nurse recorded on an encounter separately from downstream assignments', () => {
  const result = calculateClinicianWorkload([
    encounter('triage-owned', 'AWAITING_PHYSICIAN', [physician('p1')], {
      isWaiting: true,
      nurse_id: 'tn1',
      triage: { nurse_name: 'Triage Nurse' }
    })
  ]);

  assert.equal(result.lanes.triageNurses[0].clinicianId, 'tn1');
  assert.equal(result.lanes.triageNurses[0].activeCases, 1);
});

test('excludes discharged encounters and preserves hospital scoping in query', () => {
  const result = calculateClinicianWorkload([
    encounter('discharged', 'DISCHARGED', [physician('p1')]),
    encounter('active', 'AWAITING_PHYSICIAN', [physician('p1')])
  ]);
  assert.equal(result.lanes.physicians[0].activeCases, 1);
  assert.deepEqual(workloadQuery('hospital-a'), {
    hospitalId: 'hospital-a',
    specialistAttended: { $ne: true },
    status: { $ne: 'DISCHARGED' },
    $or: [{ isWaiting: true }, { status: { $in: ['TRIAGE_IN_PROGRESS', 'AWAITING_PHYSICIAN', 'ACKNOWLEDGED', 'IN_PROGRESS', 'DIAGNOSED', 'SEEN', 'HOSPITALIZED'] } }]
  });
});

test('requires stable clinician ids and does not merge name-only records', () => {
  const result = calculateClinicianWorkload([
    encounter('e1', 'AWAITING_PHYSICIAN', [{ name: 'Same Name', role: 'PHYSICIAN' }]),
    encounter('e2', 'AWAITING_PHYSICIAN', [physician('p1', 'Same Name')])
  ]);
  assert.deepEqual(result.lanes.physicians.map((entry) => entry.clinicianId), ['p1']);
});

test('active case total is unique by encounter across lanes', () => {
  const result = calculateClinicianWorkload([
    encounter('e1', 'HOSPITALIZED', [physician('p1'), clinicalNurse('n1')])
  ]);
  assert.equal(result.totals.activeCases, 1);
});

test('one assigned physician without capacity is unconfigured, not 100% capacity', () => {
  const result = calculateClinicianWorkload([
    encounter('physician-case', 'AWAITING_PHYSICIAN', [physician('p1', 'Dr One')])
  ]);
  assert.equal(result.lanes.physicians[0].activeCases, 1);
  assert.equal(result.lanes.physicians[0].activeCaseShare, 100);
  assert.equal(result.lanes.physicians[0].capacity, null);
  assert.equal(result.lanes.physicians[0].workloadPercent, null);
  assert.equal(result.lanes.physicians[0].workloadStatus, 'Unconfigured');
});

test('configured capacity produces a true workload percentage', () => {
  const result = calculateClinicianWorkload([
    encounter('physician-case', 'AWAITING_PHYSICIAN', [{ ...physician('p1', 'Dr One'), capacity: 4 }])
  ]);
  assert.equal(result.lanes.physicians[0].workloadPercent, 25);
  assert.equal(result.lanes.physicians[0].activeCaseShare, 100);
});

test('includes a zero-case clinician from the roster and resolves capacity policy', () => {
  const result = calculateClinicianWorkload([], {
    roster: [{ id: 'p1', name: 'Dr Zero', role: 'PHYSICIAN', department: 'ED' }],
    defaultCapacityByRole: { PHYSICIAN: 4 }
  });
  const entry = result.lanes.physicians[0];
  assert.equal(entry.activeCases, 0);
  assert.equal(entry.capacity, 4);
  assert.equal(entry.capacitySource, 'hospital default');
  assert.equal(entry.workloadPercent, 0);
  assert.equal(entry.workloadStatus, 'Configured');
});

test('acuity weights drive true workload percentage, not raw case count', () => {
  const result = calculateClinicianWorkload([
    encounter('red-case', 'AWAITING_PHYSICIAN', [physician('p1', 'Dr One')], { triage: { priority_code: 'RED' } })
  ], { acuityWeightByPriority: { RED: 2, ORANGE: 1, YELLOW: 1, GREEN: 1 }, defaultCapacityByRole: { PHYSICIAN: 4 } });
  const entry = result.lanes.physicians[0];
  assert.equal(entry.activeCases, 1);
  assert.equal(entry.activeWorkload, 2);
  assert.equal(entry.workloadPercent, 50);
});

test('numeric and lowercase priority codes are coerced into the correct acuity bucket', () => {
  const numeric = calculateClinicianWorkload([
    encounter('red-numeric', 'AWAITING_PHYSICIAN', [physician('p1', 'Numeric')], { triage: { priority_code: 1 } })
  ], { acuityWeightByPriority: { RED: 2, ORANGE: 1, YELLOW: 1, GREEN: 1 }, defaultCapacityByRole: { PHYSICIAN: 4 } });
  const lowercase = calculateClinicianWorkload([
    encounter('red-lower', 'AWAITING_PHYSICIAN', [physician('p2', 'Lowercase')], { triage: { satsCategory: 'red' } })
  ], { acuityWeightByPriority: { RED: 2, ORANGE: 1, YELLOW: 1, GREEN: 1 }, defaultCapacityByRole: { PHYSICIAN: 4 } });

  assert.equal(numeric.lanes.physicians[0].activeWorkload, 2);
  assert.equal(lowercase.lanes.physicians[0].activeWorkload, 2);
  assert.equal(numeric.lanes.physicians[0].workloadPercent, 50);
  assert.equal(lowercase.lanes.physicians[0].workloadPercent, 50);
});

test('reads nested triage manual report acuity', () => {
  const result = calculateClinicianWorkload([
    encounter('nested-orange', 'AWAITING_PHYSICIAN', [physician('p1')], {
      triage: { manualTriageReport: { satsLevel: 'orange' } }
    })
  ], { acuityWeightByPriority: { ORANGE: 2 }, defaultCapacityByRole: { PHYSICIAN: 4 } });

  assert.equal(result.lanes.physicians[0].acuityByPriority.ORANGE, 1);
  assert.equal(result.lanes.physicians[0].activeWorkload, 2);
});

test('manual SATS category takes precedence over numeric encounter severity', () => {
  assert.equal(queuePriority({
    severity: 1,
    manual_triage_report: { satsLevel: 'GREEN' },
    ai_triage_report: { calculated_priority: 'YELLOW' }
  }).code, 'GREEN');

  const result = calculateClinicianWorkload([
    encounter('manual-green', 'AWAITING_PHYSICIAN', [physician('p1')], {
      severity: 1,
      manual_triage_report: { satsLevel: 'GREEN' },
      ai_triage_report: { calculated_priority: 'YELLOW' }
    })
  ], { acuityWeightByPriority: { GREEN: 1, RED: 3 }, defaultCapacityByRole: { PHYSICIAN: 4 } });

  assert.equal(result.lanes.physicians[0].acuityByPriority.GREEN, 1);
  assert.equal(result.lanes.physicians[0].acuityByPriority.RED || 0, 0);
  assert.equal(result.lanes.physicians[0].activeWorkload, 1);
});

test('capacity precedence resolves clinician, department, then hospital defaults', () => {
  const result = calculateClinicianWorkload([
    encounter('clinician', 'AWAITING_PHYSICIAN', [{ ...physician('p1', 'Override'), department: 'ED', workloadCapacity: 4 }]),
    encounter('department', 'AWAITING_PHYSICIAN', [{ ...physician('p2', 'Department'), department: 'ED' }]),
    encounter('hospital', 'AWAITING_PHYSICIAN', [{ ...physician('p3', 'Hospital'), department: 'ICU' }]),
    encounter('none', 'AWAITING_PHYSICIAN', [{ ...physician('p4', 'Unconfigured'), department: 'WARD' }])
  ], {
    departmentShiftConfig: { ED: { capacityByRole: { PHYSICIAN: 6 } } },
    defaultCapacityByRole: { PHYSICIAN: 8 }
  });

  const entries = Object.fromEntries(result.lanes.physicians.map((entry) => [entry.clinicianId, entry]));
  assert.equal(entries.p1.capacity, 4);
  assert.equal(entries.p1.capacitySource, 'clinician override');
  assert.equal(entries.p2.capacity, 6);
  assert.equal(entries.p2.capacitySource, 'department default');
  assert.equal(entries.p3.capacity, 8);
  assert.equal(entries.p3.capacitySource, 'hospital default');
  assert.equal(entries.p4.capacity, 8);
  assert.equal(entries.p4.capacitySource, 'hospital default');
});

test('capacity alerts distinguish configured congestion from unconfigured capacity', () => {
  const alerts = buildCapacityAlerts([
    { clinicianId: 'p1', name: 'Configured', workloadStatus: 'Configured', workloadPercent: 100, capacity: 4, role: 'PHYSICIAN' },
    { clinicianId: 'p2', name: 'Missing', workloadStatus: 'Unconfigured', workloadPercent: null, capacity: null, activeCases: 1, role: 'PHYSICIAN' }
  ]);
  assert.deepEqual(alerts.map((alert) => alert.type), ['QUEUE_CONGESTION', 'CAPACITY_UNCONFIGURED']);
});

test('acuity weights turn two cases into a dynamic workload score', () => {
  const result = calculateClinicianWorkload([
    encounter('red', 'AWAITING_PHYSICIAN', [{ ...physician('p1'), department: 'ICU' }], { severity: 1 }),
    encounter('green', 'AWAITING_PHYSICIAN', [{ ...physician('p1'), department: 'ICU' }], { severity: 4 })
  ], {
    acuityWeightByPriority: { RED: 2, GREEN: 0.5 },
    defaultCapacityByRole: { PHYSICIAN: 4 }
  });
  const workload = result.lanes.physicians[0];
  assert.equal(workload.activeCases, 2);
  assert.equal(workload.activeWorkload, 2.5);
  assert.equal(workload.workloadPercent, 63);
});

test('requires non-increasing acuity weights from RED to GREEN', () => {
  const first = validateAcuityWeightOrdering({ RED: 3, ORANGE: 2, YELLOW: 2, GREEN: 1 });
  const second = validateAcuityWeightOrdering({ RED: 1, ORANGE: 2, YELLOW: 1, GREEN: 0.5 });

  assert.equal(first.valid, true);
  assert.equal(second.valid, false);
  assert.match(second.message, /RED.*ORANGE.*YELLOW.*GREEN/i);
});

test('rejects hospital mismatch during a write attempt', async () => {
  const req = {
    user: { hospitalId: 'hospital-a', _id: 'user-1', email: 'admin@hospital-a.test', role: 'ADMIN' },
    body: { hospitalId: 'hospital-b', level: 'hospital', role: 'PHYSICIAN', capacity: 8 }
  };
  const res = createJsonResponse();

  await updateCapacity(req, res);

  assert.equal(res.statusCode, 403);
  assert.match(res.payload.message, /Authenticated hospital does not match/i);
});

test('rejects invalid acuity ordering at the controller level', async () => {
  const req = {
    user: { hospitalId: 'hospital-a', _id: 'user-1', email: 'admin@hospital-a.test', role: 'ADMIN' },
    body: { weights: { RED: 1, ORANGE: 2, YELLOW: 1, GREEN: 0.5 } }
  };
  const res = createJsonResponse();

  await updateAcuityWeights(req, res);

  assert.equal(res.statusCode, 400);
  assert.match(res.payload.message, /RED.*ORANGE.*YELLOW.*GREEN/i);
});

test('includes actor, hospital, role, old/new values and timestamp in the audit payload', () => {
  const details = buildCapacityAuditDetails({
    actor: 'admin@hospital-a.test',
    hospitalId: 'hospital-a',
    level: 'hospital',
    target: 'PHYSICIAN',
    role: 'PHYSICIAN',
    oldValue: 6,
    newValue: 8
  });

  assert.equal(details.actor, 'admin@hospital-a.test');
  assert.equal(details.hospitalId, 'hospital-a');
  assert.equal(details.role, 'PHYSICIAN');
  assert.equal(details.oldValue, 6);
  assert.equal(details.newValue, 8);
  assert.ok(details.timestamp);
});

const createJsonResponse = () => ({
  statusCode: null,
  payload: null,
  status(code) {
    this.statusCode = code;
    return this;
  },
  json(data) {
    this.payload = data;
    return this;
  }
});
