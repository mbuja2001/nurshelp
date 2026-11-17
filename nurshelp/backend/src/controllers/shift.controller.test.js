import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import mongoose from 'mongoose';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { Shift } from '../models/shift.model.js';
import { createShift, updateShift } from './shift.controller.js';

const originalClinicalStaffFind = ClinicalStaff.find;
const originalShiftFindOne = Shift.findOne;
const originalShiftFindOneAndUpdate = Shift.findOneAndUpdate;
const originalShiftSave = Shift.prototype.save;
const hospitalId = new mongoose.Types.ObjectId();
const staffId = new mongoose.Types.ObjectId();

afterEach(() => {
  ClinicalStaff.find = originalClinicalStaffFind;
  Shift.findOne = originalShiftFindOne;
  Shift.findOneAndUpdate = originalShiftFindOneAndUpdate;
  Shift.prototype.save = originalShiftSave;
});

const responseStub = () => ({
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

const getFutureShiftDate = () => {
  const date = new Date();
  date.setDate(date.getDate() + 1);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
};

describe('shift creation API validation', () => {
  it('returns the position-specific code for an invalid shift duty', async () => {
    ClinicalStaff.find = () => ({
      lean: async () => [{
        _id: staffId,
        hospitalId,
        status: 'ACTIVE',
        systemRole: 'NURSE',
        positionCode: 'TRIAGE_NURSE',
        firstName: 'Triage',
        lastName: 'Nurse',
      }],
    });

    const response = responseStub();
    await createShift({
      user: { id: 'manager-1' },
      hospitalId,
      body: {
        shiftName: 'ED Day Shift',
        shiftStart: '08:00',
        shiftEnd: '16:00',
        date: getFutureShiftDate(),
        maxStaffRequired: 1,
        assignedStaff: [{ staffID: staffId, assignmentRole: 'CLINICAL_NURSE' }],
      },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.code, 'DUTY_NOT_PERMITTED_FOR_POSITION');
  });

  it('requires an explicit duty from a legacy nurse and marks the record for backfill', async () => {
    ClinicalStaff.find = () => ({
      lean: async () => [{
        _id: staffId,
        hospitalId,
        status: 'ACTIVE',
        systemRole: 'NURSE',
        positionCode: null,
        firstName: 'Legacy',
        lastName: 'Nurse',
      }],
    });

    const response = responseStub();
    await createShift({
      user: { id: 'manager-1' },
      hospitalId,
      body: {
        shiftName: 'ED Day Shift',
        shiftStart: '08:00',
        shiftEnd: '16:00',
        date: getFutureShiftDate(),
        maxStaffRequired: 1,
        assignedStaff: [{ staffID: staffId }],
      },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.code, 'SHIFT_DUTY_REQUIRED');
    assert.equal(response.body.positionCodeNeedsBackfill, true);
  });

  it('rejects a shift whose scheduled start time has already passed', async () => {
    const response = responseStub();

    await createShift({
      user: { id: 'manager-1' },
      hospitalId,
      body: {
        shiftName: 'Past shift',
        shiftStart: '08:00',
        shiftEnd: '16:00',
        date: '2000-01-01',
        maxStaffRequired: 1,
        assignedStaff: [],
      },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.code, 'SHIFT_START_TIME_PASSED');
  });

  it('requires a valid dated shift schedule', async () => {
    const response = responseStub();

    await createShift({
      user: { id: 'manager-1' },
      hospitalId,
      body: {
        shiftName: 'Undated shift',
        shiftStart: '08:00',
        shiftEnd: '16:00',
        maxStaffRequired: 1,
        assignedStaff: [],
      },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.code, 'INVALID_SHIFT_SCHEDULE');
  });

  it('rejects moving an existing shift to a start time that has passed', async () => {
    Shift.findOne = async () => ({
      status: 'PUBLISHED',
      date: new Date('2030-10-14T00:00:00.000Z'),
      shiftStart: '08:00',
      shiftEnd: '16:00',
    });
    const response = responseStub();

    await updateShift({
      params: { shiftID: 'SHIFT-001' },
      hospitalId,
      body: { date: '2000-01-01' },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.equal(response.body.code, 'SHIFT_START_TIME_PASSED');
  });

  it('creates a hospital-wide shift and assigns staff without a department', async () => {
    ClinicalStaff.find = () => ({
      lean: async () => [{
        _id: staffId,
        hospitalId,
        status: 'ACTIVE',
        systemRole: 'NURSE',
        positionCode: 'TRIAGE_NURSE',
        firstName: 'Triage',
        lastName: 'Nurse',
      }],
    });
    Shift.prototype.save = async function () {
      this._id = new mongoose.Types.ObjectId();
      return this;
    };

    const response = responseStub();
    await createShift({
      user: { id: 'manager-1' },
      hospitalId,
      body: {
        shiftName: 'Hospital Day Shift',
        shiftStart: '08:00',
        shiftEnd: '16:00',
        date: getFutureShiftDate(),
        maxStaffRequired: 1,
        assignedStaff: [{ staffID: staffId, assignmentRole: 'TRIAGE_NURSE' }],
      },
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(response.body.success, true);
    assert.equal(response.body.shift.department, null);
    assert.equal(response.body.shift.assignedStaff.length, 1);
  });

  it('returns the original shift when an idempotency key is retried', async () => {
    const existingShift = { _id: new mongoose.Types.ObjectId(), shiftID: 'SHIFT-EXISTING' };
    Shift.findOne = (query) => ({
      lean: async () => query.idempotencyKey === 'request-123' ? existingShift : null,
    });

    const response = responseStub();
    await createShift({
      user: { id: 'manager-1' },
      hospitalId,
      body: { idempotencyKey: 'request-123' },
    }, response);

    assert.equal(response.statusCode, 200);
    assert.equal(response.body.success, true);
    assert.equal(response.body.shift, existingShift);
  });
});
