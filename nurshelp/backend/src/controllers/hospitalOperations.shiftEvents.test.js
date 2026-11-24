import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { AuditLog } from '../models/auditLog.model.js';
import { Shift } from '../models/shift.model.js';
import { recordHospitalShiftEvent } from './hospitalOperations.controller.js';

const originalAuditCreate = AuditLog.create;
const originalShiftFindOne = Shift.findOne;
const originalShiftFindOneAndUpdate = Shift.findOneAndUpdate;

afterEach(() => {
  AuditLog.create = originalAuditCreate;
  Shift.findOne = originalShiftFindOne;
  Shift.findOneAndUpdate = originalShiftFindOneAndUpdate;
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
  },
});

const getTomorrow = () => {
  const date = new Date();
  date.setDate(date.getDate() + 1);
  return date;
};

describe('hospital shift lifecycle events', () => {
  it('starts a persisted shift referenced by its database ID', async () => {
    const now = new Date();
    const scheduledEnd = new Date(now.getTime() + 8 * 60 * 60 * 1000);
    const formatTime = (date) => `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
    const shift = {
      _id: 'shift-document-id',
      shiftID: 'SHIFT-001',
      shiftName: 'Hospital Day Shift',
      status: 'PUBLISHED',
      date: new Date(),
      shiftStart: formatTime(now),
      shiftEnd: formatTime(scheduledEnd),
      assignedStaff: [{ status: 'ASSIGNED' }],
      save: async () => {},
    };
    let query;
    Shift.findOne = async (value) => {
      query = value;
      return shift;
    };
    AuditLog.create = async (event) => ({ _id: 'audit-1', ...event });
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'STARTED', operatingDate: shift.date.toISOString().slice(0, 10), shift: { id: shift._id, shiftID: shift.shiftID } },
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(response.body.success, true);
    assert.deepEqual(query, { hospitalId: 'hospital-1', _id: shift._id });
    assert.equal(shift.status, 'IN_PROGRESS');
  });

  it('rejects starting a published shift after its scheduled end', async () => {
    const shift = {
      _id: 'missed-shift',
      shiftID: 'SHIFT-MISSED',
      shiftName: 'Missed shift',
      status: 'PUBLISHED',
      date: new Date('2000-01-01T00:00:00.000Z'),
      shiftStart: '08:00',
      shiftEnd: '16:00',
      assignedStaff: [{ status: 'ASSIGNED' }],
      save: async () => {},
    };
    Shift.findOne = async () => shift;
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'STARTED', shift: { id: shift._id } },
    }, response);

    assert.equal(response.statusCode, 409);
    assert.match(response.body.message, /past its scheduled end/i);
    assert.equal(shift.status, 'PUBLISHED');
  });

  it('rejects starting a published shift more than five minutes before its scheduled start', async () => {
    const now = new Date();
    const scheduledStart = new Date(now.getTime() + 6 * 60 * 1000);
    const date = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const shift = {
      _id: 'not-ready-shift',
      shiftID: 'SHIFT-NOT-READY',
      shiftName: 'Not-ready shift',
      status: 'PUBLISHED',
      date,
      shiftStart: `${String(scheduledStart.getHours()).padStart(2, '0')}:${String(scheduledStart.getMinutes()).padStart(2, '0')}`,
      shiftEnd: `${String((scheduledStart.getHours() + 8) % 24).padStart(2, '0')}:${String(scheduledStart.getMinutes()).padStart(2, '0')}`,
      assignedStaff: [{ status: 'ASSIGNED' }],
      save: async () => {},
    };
    Shift.findOne = async () => shift;
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'STARTED', shift: { id: shift._id } },
    }, response);

    assert.equal(response.statusCode, 409);
    assert.match(response.body.message, /within five minutes/i);
    assert.equal(shift.status, 'PUBLISHED');
  });

  it('allows starting a published shift at the five-minute early boundary', async () => {
    const now = new Date();
    const scheduledStart = new Date(now.getTime() + 5 * 60 * 1000);
    const date = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const shift = {
      _id: 'early-start-shift',
      shiftID: 'SHIFT-EARLY-START',
      shiftName: 'Early-start shift',
      status: 'PUBLISHED',
      date,
      shiftStart: `${String(scheduledStart.getHours()).padStart(2, '0')}:${String(scheduledStart.getMinutes()).padStart(2, '0')}`,
      shiftEnd: `${String((scheduledStart.getHours() + 8) % 24).padStart(2, '0')}:${String(scheduledStart.getMinutes()).padStart(2, '0')}`,
      assignedStaff: [{ status: 'ASSIGNED' }],
      save: async () => {},
    };
    Shift.findOne = async () => shift;
    AuditLog.create = async (event) => ({ _id: 'audit-early-start', ...event });
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'STARTED', shift: { id: shift._id } },
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(shift.status, 'IN_PROGRESS');
  });

  it('does not allow an active shift to be started a second time', async () => {
    const shift = {
      _id: 'active-shift',
      shiftID: 'SHIFT-ACTIVE',
      shiftName: 'Active shift',
      status: 'IN_PROGRESS',
      assignedStaff: [{ status: 'ASSIGNED' }],
    };
    Shift.findOne = async () => shift;
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'STARTED', shift: { id: shift._id } },
    }, response);

    assert.equal(response.statusCode, 409);
    assert.match(response.body.message, /only published shifts/i);
    assert.equal(shift.status, 'IN_PROGRESS');
  });

  it('does not allow a client-supplied future operating date to start a future shift', async () => {
    const shift = {
      _id: 'future-shift',
      shiftID: 'SHIFT-FUTURE',
      shiftName: 'Future shift',
      status: 'PUBLISHED',
      date: getTomorrow(),
      shiftStart: '08:00',
      shiftEnd: '16:00',
      assignedStaff: [{ status: 'ASSIGNED' }],
      save: async () => {},
    };
    Shift.findOne = async () => shift;
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'STARTED', operatingDate: shift.date.toISOString().slice(0, 10), shift: { id: shift._id } },
    }, response);

    assert.equal(response.statusCode, 409);
    assert.match(response.body.message, /scheduled date/i);
    assert.equal(shift.status, 'PUBLISHED');
  });

  it('requires confirmation and a reason to stop an active shift early', async () => {
    const shift = {
      _id: 'active-shift',
      shiftID: 'SHIFT-ACTIVE',
      shiftName: 'Active shift',
      status: 'IN_PROGRESS',
      date: getTomorrow(),
      shiftStart: '08:00',
      shiftEnd: '16:00',
      assignedStaff: [{ status: 'ASSIGNED' }],
    };
    Shift.findOne = async () => shift;
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'CANCELLED', shift: { id: shift._id }, earlyStopConfirmed: true },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.match(response.body.message, /reason/i);
  });

  it('requires explicit confirmation even when an early-stop reason is supplied', async () => {
    const shift = {
      _id: 'active-shift',
      shiftID: 'SHIFT-ACTIVE',
      shiftName: 'Active shift',
      status: 'IN_PROGRESS',
      date: getTomorrow(),
      shiftStart: '08:00',
      shiftEnd: '16:00',
      assignedStaff: [{ status: 'ASSIGNED' }],
    };
    Shift.findOne = async () => shift;
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'CANCELLED', shift: { id: shift._id }, reason: 'Unexpected closure' },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.match(response.body.message, /confirmation/i);
  });

  it('records an active early stop as cancelled with an audited reason', async () => {
    const shift = {
      _id: 'active-shift',
      shiftID: 'SHIFT-ACTIVE',
      shiftName: 'Active shift',
      department: null,
      specialty: null,
      wardId: null,
      status: 'IN_PROGRESS',
      date: getTomorrow(),
      shiftStart: '08:00',
      shiftEnd: '16:00',
      assignedStaff: [{ status: 'ASSIGNED' }],
    };
    Shift.findOne = async () => shift;
    Shift.findOneAndUpdate = async (filter, update) => {
      assert.deepEqual(filter, { _id: shift._id, hospitalId: 'hospital-1', status: 'IN_PROGRESS' });
      Object.assign(shift, update.$set);
      return shift;
    };
    AuditLog.create = async (event) => ({ _id: 'audit-cancel', ...event });
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: {
        event: 'CANCELLED',
        shift: { id: shift._id },
        earlyStopConfirmed: true,
        reason: 'Emergency service disruption',
      },
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(shift.status, 'CANCELLED');
    assert.equal(shift.cancellationReason, 'Emergency service disruption');
    assert.equal(response.body.event.action, 'HOSPITAL_SHIFT_CANCELLED');
    assert.equal(response.body.event.reason, 'Emergency service disruption');
  });

  it('rejects lifecycle events without a persisted shift reference', async () => {
    const response = createResponse();

    await recordHospitalShiftEvent({
      hospitalId: 'hospital-1',
      user: { id: 'manager-1' },
      body: { event: 'STARTED', shift: { shiftName: 'Hospital Day Shift' } },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.message, 'A persisted shift ID is required');
  });
});
