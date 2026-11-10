import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { AuditLog } from '../models/auditLog.model.js';
import { Shift } from '../models/shift.model.js';
import { recordHospitalShiftEvent } from './hospitalOperations.controller.js';

const originalAuditCreate = AuditLog.create;
const originalShiftFindOne = Shift.findOne;

afterEach(() => {
  AuditLog.create = originalAuditCreate;
  Shift.findOne = originalShiftFindOne;
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

describe('hospital shift lifecycle events', () => {
  it('starts a persisted shift referenced by its database ID', async () => {
    const shift = {
      _id: 'shift-document-id',
      shiftID: 'SHIFT-001',
      shiftName: 'Hospital Day Shift',
      status: 'PUBLISHED',
      date: new Date(),
      shiftStart: '08:00',
      shiftEnd: '16:00',
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
      body: { event: 'STARTED', shift: { id: shift._id, shiftID: shift.shiftID } },
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(response.body.success, true);
    assert.deepEqual(query, { hospitalId: 'hospital-1', _id: shift._id });
    assert.equal(shift.status, 'IN_PROGRESS');
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
