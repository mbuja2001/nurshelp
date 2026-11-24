import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import mongoose from 'mongoose';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { Appointment } from '../models/hmsMisc.model.js';
import { Encounter } from '../models/encounter.model.js';
import { getCalendarEvents } from './calendar.controller.js';
import {
  createSurgeryRecord,
  getSurgeriesForPhysician,
  updateSurgeryRecord
} from './encounter.controller.js';

const originalEncounterFind = Encounter.find;
const originalEncounterFindById = Encounter.findById;
const originalAppointmentFind = Appointment.find;

afterEach(() => {
  Encounter.find = originalEncounterFind;
  Encounter.findById = originalEncounterFindById;
  Appointment.find = originalAppointmentFind;
});

const queryResult = (result) => ({
  select() { return this; },
  lean: async () => result
});

const createResponse = () => ({
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

const PHASE_ZERO_ALLOWED_ROLES = new Set(['PHYSICIAN', 'DOCTOR', 'SURGEON']);
const LEGACY_ROLE_ALIASES = [
  'DOCTOR',
  'SURGEON',
  'SPECIALIST',
  'PHYSICIAN_ER',
  'TRIAGE_NURSE',
  'TRIAGE',
  'TRIAGE_NURSE_ER',
  'CLINICAL_NURSE',
  'CLINICAL_NURSE_ER',
  'CLINICAL_NURSE_WARD',
  'ADMIN',
  'MANAGER',
  'SHIFT_MANAGER',
  'DEPARTMENT_MANAGER',
  'THEATRE_PREOP_LEAD',
  'RESEARCH_STAFF',
  'OTHER_SUPPORT'
];
const PHASE_ZERO_ROLE_MATRIX = [
  ...ClinicalStaff.schema.path('systemRole').enumValues
    .map((role) => [role, PHASE_ZERO_ALLOWED_ROLES.has(role)]),
  ...LEGACY_ROLE_ALIASES.map((role) => [role, PHASE_ZERO_ALLOWED_ROLES.has(role)]),
  ['UNKNOWN', false],
  ['', false],
  [null, false]
];

test('calendar returns only explicitly assigned surgeries and same-hospital appointments', async () => {
  const userId = new mongoose.Types.ObjectId();
  const otherStaffId = new mongoose.Types.ObjectId();
  const hospitalId = new mongoose.Types.ObjectId();
  const otherHospitalId = new mongoose.Types.ObjectId();
  const start = new Date('2026-11-23T10:00:00.000Z');
  const end = new Date('2026-11-23T11:00:00.000Z');
  let encounterQuery;
  let appointmentQuery;

  Encounter.find = (query) => {
    encounterQuery = query;
    return queryResult([{
      _id: new mongoose.Types.ObjectId(),
      hospitalId,
      surgery_records: [
        {
          _id: new mongoose.Types.ObjectId(),
          surgeon_id: userId,
          procedure_name: 'Assigned procedure',
          scheduled_at: start,
          scheduled_end: end,
          status: 'SCHEDULED'
        },
        {
          _id: new mongoose.Types.ObjectId(),
          surgeon_id: otherStaffId,
          surgeon: 'Assigned procedure',
          procedure_name: 'Unassigned procedure',
          scheduled_at: start,
          scheduled_end: end,
          status: 'SCHEDULED'
        },
        {
          _id: new mongoose.Types.ObjectId(),
          surgeon_id: userId,
          procedure_name: 'Missing duration',
          scheduled_at: start,
          status: 'SCHEDULED'
        }
      ]
    }]);
  };
  Appointment.find = (query) => {
    appointmentQuery = query;
    return queryResult([
      {
        _id: new mongoose.Types.ObjectId(),
        hospitalId,
        physicianId: userId,
        appointmentType: 'Follow-up',
        scheduledDateTime: start,
        duration: 30,
        status: 'scheduled'
      },
      {
        _id: new mongoose.Types.ObjectId(),
        hospitalId: otherHospitalId,
        physicianId: userId,
        scheduledDateTime: start,
        duration: 30,
        status: 'scheduled'
      },
      {
        _id: new mongoose.Types.ObjectId(),
        hospitalId,
        physicianId: otherStaffId,
        scheduledDateTime: start,
        duration: 30,
        status: 'scheduled'
      }
    ]);
  };

  const res = createResponse();
  await getCalendarEvents({
    hospitalId,
    user: { _id: userId, hospitalId, role: 'PHYSICIAN' },
    query: { startDate: start.toISOString(), endDate: end.toISOString() }
  }, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.events.length, 2);
  assert.deepEqual(res.body.events.map((event) => event.type).sort(), ['appointment', 'surgery']);
  assert.match(res.body.events.find((event) => event.type === 'surgery').title, /Assigned procedure/);
  assert.ok(encounterQuery.surgery_records.$elemMatch.surgeon_id.equals(userId));
  assert.ok(appointmentQuery.physicianId.equals(userId));
  assert.ok(appointmentQuery.hospitalId.equals(hospitalId));
  assert.ok(res.body.disabledCategories.includes('lab'));
});

test('calendar returns no patient events for non-clinical roles without querying records', async () => {
  const hospitalId = new mongoose.Types.ObjectId();
  let queried = false;
  Encounter.find = () => {
    queried = true;
    throw new Error('must not query');
  };
  Appointment.find = () => {
    queried = true;
    throw new Error('must not query');
  };

  const res = createResponse();
  await getCalendarEvents({
    hospitalId,
    user: { _id: new mongoose.Types.ObjectId(), hospitalId, role: 'ADMINISTRATIVE' },
    query: {}
  }, res);

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body.events, []);
  assert.equal(queried, false);
});

test('calendar rejects missing staff identity and mismatched hospital context before querying records', async () => {
  const hospitalId = new mongoose.Types.ObjectId();
  const userId = new mongoose.Types.ObjectId();
  let queried = false;
  Encounter.find = Appointment.find = () => {
    queried = true;
    throw new Error('must not query');
  };

  const unauthenticatedResponse = createResponse();
  await getCalendarEvents({ hospitalId, query: {} }, unauthenticatedResponse);
  assert.equal(unauthenticatedResponse.statusCode, 401);

  const missingHospitalResponse = createResponse();
  await getCalendarEvents({
    user: { _id: userId, role: 'PHYSICIAN' },
    query: {}
  }, missingHospitalResponse);
  assert.equal(missingHospitalResponse.statusCode, 401);

  const mismatchedContextResponse = createResponse();
  await getCalendarEvents({
    hospitalId,
    user: { _id: userId, hospitalId: new mongoose.Types.ObjectId(), role: 'PHYSICIAN' },
    query: {}
  }, mismatchedContextResponse);
  assert.equal(mismatchedContextResponse.statusCode, 403);
  assert.equal(queried, false);
});

test('calendar rejects malformed date ranges instead of silently using defaults', async () => {
  const hospitalId = new mongoose.Types.ObjectId();
  const res = createResponse();

  await getCalendarEvents({
    hospitalId,
    user: { _id: new mongoose.Types.ObjectId(), hospitalId, role: 'PHYSICIAN' },
    query: { startDate: 'not-a-date', endDate: '2026-11-24T00:00:00.000Z' }
  }, res);

  assert.equal(res.statusCode, 400);
  assert.equal(res.body.success, false);
});

test('scheduled surgery list returns only records with an explicit surgeon identity', async () => {
  const userId = new mongoose.Types.ObjectId();
  const hospitalId = new mongoose.Types.ObjectId();
  const otherStaffId = new mongoose.Types.ObjectId();
  let query;
  Encounter.find = (filter) => {
    query = filter;
    return queryResult([{
      _id: new mongoose.Types.ObjectId(),
      hospitalId,
      patient: { name: 'Assigned patient' },
      surgery_records: [
        { _id: new mongoose.Types.ObjectId(), surgeon_id: userId, procedure_name: 'Assigned procedure', status: 'SCHEDULED' },
        { _id: new mongoose.Types.ObjectId(), surgeon_id: otherStaffId, procedure_name: 'Unassigned procedure', status: 'SCHEDULED' }
      ]
    }]);
  };

  const res = createResponse();
  await getSurgeriesForPhysician({
    query: {},
    hospitalId,
    user: { _id: userId, hospitalId, role: 'PHYSICIAN' }
  }, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.count, 1);
  assert.equal(res.body.surgeries[0].procedure_name, 'Assigned procedure');
  assert.ok(query.hospitalId.equals(hospitalId));
  assert.ok(query.surgery_records.$elemMatch.surgeon_id.equals(userId));
});

test('assigned-capable roles still receive no events when they have no persisted assignment', async () => {
  const userId = new mongoose.Types.ObjectId();
  const otherStaffId = new mongoose.Types.ObjectId();
  const hospitalId = new mongoose.Types.ObjectId();
  const start = new Date('2026-11-23T10:00:00.000Z');
  const end = new Date('2026-11-23T11:00:00.000Z');

  Encounter.find = () => queryResult([{
    _id: new mongoose.Types.ObjectId(),
    hospitalId,
    patient: { name: 'Unassigned patient' },
    surgery_records: [{
      _id: new mongoose.Types.ObjectId(),
      surgeon_id: otherStaffId,
      surgeon: 'Dr Assigned',
      recipientIds: [userId],
      procedure_name: 'Unassigned surgery',
      scheduled_at: start,
      scheduled_end: end,
      status: 'SCHEDULED'
    }]
  }]);
  Appointment.find = () => queryResult([{
    _id: new mongoose.Types.ObjectId(),
    hospitalId,
    physicianId: otherStaffId,
    recipientIds: [userId],
    scheduledDateTime: start,
    duration: 30,
    status: 'scheduled'
  }]);

  for (const role of ['PHYSICIAN', 'DOCTOR', 'SURGEON']) {
    const request = {
      hospitalId,
      user: { _id: userId, hospitalId, role },
      query: { startDate: start.toISOString(), endDate: end.toISOString() }
    };
    const calendarResponse = createResponse();
    const surgeryListResponse = createResponse();
    await getCalendarEvents(request, calendarResponse);
    await getSurgeriesForPhysician(request, surgeryListResponse);

    assert.deepEqual(calendarResponse.body.events, [], `calendar assignment denial for ${role}`);
    assert.equal(surgeryListResponse.body.count, 0, `surgery-list assignment denial for ${role}`);
  }
});

test('every disabled calendar category returns no events and performs no patient-record query', async () => {
  const hospitalId = new mongoose.Types.ObjectId();
  let queried = false;
  Encounter.find = Appointment.find = () => {
    queried = true;
    throw new Error('disabled category must not query patient records');
  };
  const disabledCategories = [
    'meeting',
    'research',
    'lab',
    'med',
    'diagnosed',
    'discharged',
    'administered',
    'handoff',
    'radiology',
    'pathology'
  ];

  for (const category of disabledCategories) {
    const response = createResponse();
    await getCalendarEvents({
      hospitalId,
      user: { _id: new mongoose.Types.ObjectId(), hospitalId, role: 'PHYSICIAN' },
      query: {
        startDate: '2026-11-23T00:00:00.000Z',
        endDate: '2026-11-24T00:00:00.000Z',
        category
      }
    }, response);

    assert.equal(response.statusCode, 200, `disabled category status: ${category}`);
    assert.deepEqual(response.body.events, [], `disabled category events: ${category}`);
    assert.deepEqual(response.body.disabledCategories, disabledCategories);
  }
  assert.equal(queried, false);
});

test('legacy surgery create and update endpoints reject writes without reading or mutating encounters', async () => {
  Encounter.findById = () => {
    throw new Error('legacy write must not read an encounter');
  };
  const createRes = createResponse();
  const updateRes = createResponse();

  await createSurgeryRecord({ params: { id: new mongoose.Types.ObjectId() } }, createRes);
  await updateSurgeryRecord({
    params: { id: new mongoose.Types.ObjectId(), surgeryId: new mongoose.Types.ObjectId() }
  }, updateRes);

  for (const response of [createRes, updateRes]) {
    assert.equal(response.statusCode, 410);
    assert.equal(response.body.code, 'LEGACY_SURGERY_WRITES_DISABLED');
  }
});

test('TS-AUTH-P0-MATRIX: role/resource/action/hospital matrix enforces only assigned same-hospital reads', async () => {
  const userId = new mongoose.Types.ObjectId();
  const otherStaffId = new mongoose.Types.ObjectId();
  const hospitalId = new mongoose.Types.ObjectId();
  const otherHospitalId = new mongoose.Types.ObjectId();
  const start = new Date('2026-11-23T10:00:00.000Z');
  const end = new Date('2026-11-23T11:00:00.000Z');
  const assignedEncounter = {
    _id: new mongoose.Types.ObjectId(),
    hospitalId,
    patient: { name: 'Assigned patient' },
    surgery_records: [{
      _id: new mongoose.Types.ObjectId(),
      surgeon_id: userId,
      procedure_name: 'Assigned surgery',
      scheduled_at: start,
      scheduled_end: end,
      status: 'SCHEDULED'
    }]
  };
  const sameHospitalUnassignedEncounter = {
    _id: new mongoose.Types.ObjectId(),
    hospitalId,
    patient: { name: 'Unassigned patient' },
    surgery_records: [{
      _id: new mongoose.Types.ObjectId(),
      surgeon_id: otherStaffId,
      surgeon: 'Dr Assigned',
      recipientIds: [userId],
      procedure_name: 'Unassigned surgery',
      scheduled_at: start,
      scheduled_end: end,
      status: 'SCHEDULED'
    }]
  };
  const otherHospitalEncounter = {
    ...assignedEncounter,
    _id: new mongoose.Types.ObjectId(),
    hospitalId: otherHospitalId
  };
  const assignedAppointment = {
    _id: new mongoose.Types.ObjectId(),
    hospitalId,
    physicianId: userId,
    scheduledDateTime: start,
    duration: 30,
    status: 'scheduled'
  };
  const otherStaffAppointment = {
    ...assignedAppointment,
    _id: new mongoose.Types.ObjectId(),
    physicianId: otherStaffId,
    recipientIds: [userId]
  };
  const otherHospitalAppointment = {
    ...assignedAppointment,
    _id: new mongoose.Types.ObjectId(),
    hospitalId: otherHospitalId
  };
  const missingHospitalAppointment = {
    ...assignedAppointment,
    _id: new mongoose.Types.ObjectId(),
    hospitalId: null
  };

  for (const [role, allowed] of PHASE_ZERO_ROLE_MATRIX) {
    const observedFilters = [];
    Encounter.find = (filter) => {
      observedFilters.push({ resource: 'surgery', filter });
      return queryResult([assignedEncounter, sameHospitalUnassignedEncounter, otherHospitalEncounter]);
    };
    Appointment.find = (filter) => {
      observedFilters.push({ resource: 'appointment', filter });
      return queryResult([
        assignedAppointment,
        otherStaffAppointment,
        otherHospitalAppointment,
        missingHospitalAppointment
      ]);
    };

    const request = {
      hospitalId,
      user: { _id: userId, hospitalId, role },
      userRole: role,
      query: {
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        userId: otherStaffId.toString(),
        hospitalId: otherHospitalId.toString(),
        role: allowed ? 'ADMINISTRATIVE' : 'PHYSICIAN',
        recipientIds: [userId.toString()]
      }
    };
    const calendarResponse = createResponse();
    const surgeryListResponse = createResponse();
    await getCalendarEvents(request, calendarResponse);
    await getSurgeriesForPhysician(request, surgeryListResponse);

    assert.equal(calendarResponse.statusCode, 200, `calendar status for role ${role}`);
    assert.equal(surgeryListResponse.statusCode, 200, `surgery-list status for role ${role}`);
    assert.equal(calendarResponse.body.events.length, allowed ? 2 : 0, `calendar events for role ${role}`);
    assert.equal(surgeryListResponse.body.count, allowed ? 1 : 0, `surgery-list count for role ${role}`);
    assert.deepEqual(
      calendarResponse.body.events.map((event) => event.type).sort(),
      allowed ? ['appointment', 'surgery'] : [],
      `calendar resources for role ${role}`
    );
    assert.ok(calendarResponse.body.disabledCategories.includes('lab'));

    if (allowed) {
      assert.equal(observedFilters.length, 3, `query count for role ${role}`);
      assert.ok(observedFilters.every(({ filter }) => filter.hospitalId.equals(hospitalId)));
      assert.ok(observedFilters.some(({ resource, filter }) => (
        resource === 'surgery'
        && filter.surgery_records.$elemMatch.surgeon_id.equals(userId)
      )));
      assert.ok(observedFilters.some(({ resource, filter }) => (
        resource === 'appointment'
        && filter.physicianId.equals(userId)
      )));
      assert.equal(
        calendarResponse.body.events.some((event) => event.title.includes('Unassigned')),
        false
      );
    } else {
      assert.deepEqual(observedFilters, [], `denied roles must not query patient records: ${role}`);
    }
  }
});
