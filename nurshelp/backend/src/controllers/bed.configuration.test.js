import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { BED_EQUIPMENT_CATALOG, BED_EQUIPMENT_OPTIONS } from '../config/bedEquipmentCatalog.js';
import { Bed } from '../models/bed.model.js';
import { BedSchedule } from '../models/bedSchedule.model.js';
import { Ward } from '../models/ward.model.js';
import { createBedConfiguration, getBedConfigurations, updateBedConfiguration } from './bed.controller.js';

const originalBedFind = Bed.find;
const originalBedFindOne = Bed.findOne;
const originalBedFindOneAndUpdate = Bed.findOneAndUpdate;
const originalBedCreate = Bed.create;
const originalBedScheduleFind = BedSchedule.find;
const originalBedScheduleFindOne = BedSchedule.findOne;
const originalBedScheduleFindOneAndUpdate = BedSchedule.findOneAndUpdate;
const originalBedScheduleCreate = BedSchedule.create;
const originalWardFind = Ward.find;
const originalWardFindOne = Ward.findOne;
const originalWardUpdateOne = Ward.updateOne;

afterEach(() => {
  Bed.find = originalBedFind;
  Bed.findOne = originalBedFindOne;
  Bed.findOneAndUpdate = originalBedFindOneAndUpdate;
  Bed.create = originalBedCreate;
  BedSchedule.find = originalBedScheduleFind;
  BedSchedule.findOne = originalBedScheduleFindOne;
  BedSchedule.findOneAndUpdate = originalBedScheduleFindOneAndUpdate;
  BedSchedule.create = originalBedScheduleCreate;
  Ward.find = originalWardFind;
  Ward.findOne = originalWardFindOne;
  Ward.updateOne = originalWardUpdateOne;
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

describe('admin bed configuration', () => {
  it('exposes active hospital wards and the predefined equipment catalogue', async () => {
    Bed.find = () => ({ sort() { return this; }, lean: async () => [] });
    BedSchedule.find = () => ({ lean: async () => [] });
    Ward.find = () => ({
      sort() { return this; },
      select() { return this; },
      lean: async () => [{ _id: 'ward-1', wardName: 'Critical Care East', rooms: [{ roomNumber: 'E101', roomName: 'East room' }], isActive: true }],
    });
    const response = responseStub();

    await getBedConfigurations({ hospitalId: 'hospital-1' }, response);

    assert.equal(response.body.success, true);
    assert.equal(response.body.wards[0].wardName, 'Critical Care East');
    assert.equal(response.body.wards[0].rooms[0].roomNumber, 'E101');
    assert.ok(response.body.equipmentCatalog.length > 0);
    assert.ok(response.body.equipmentCatalog.some((group) => group.items.includes('Mechanical Ventilator')));
  });

  it('creates a bed only under an existing active ward and stores selected catalogue equipment', async () => {
    const selectedEquipment = ['Mechanical Ventilator', 'Pulse Oximeter'];
    const configuredWard = { _id: 'ward-1', wardName: 'Critical Care East', department: 'ICU', rooms: [{ roomNumber: 'E101', roomName: 'East room' }] };
    let bedRecord;
    let scheduleRecord;
    Bed.findOne = async () => null;
    Ward.findOne = async (query) => query._id === configuredWard._id ? configuredWard : null;
    Bed.create = async (document) => {
      bedRecord = document;
      return { ...document, toObject: () => document };
    };
    BedSchedule.create = async (document) => {
      scheduleRecord = document;
      return { _id: 'schedule-1', ...document };
    };
    Ward.updateOne = async () => ({ acknowledged: true });
    const response = responseStub();

    await createBedConfiguration({
      hospitalId: 'hospital-1',
      user: { _id: 'manager-1' },
      body: { bed_id: 'ICU-E-01', wardId: 'ward-1', room_number: 'E101', equipment_baseline: selectedEquipment },
    }, response);

    assert.equal(response.statusCode, 201);
    assert.equal(bedRecord.ward_unit, configuredWard.wardName);
    assert.equal(bedRecord.room_number, 'E101');
    assert.deepEqual(bedRecord.equipment_baseline, selectedEquipment);
    assert.equal(scheduleRecord.wardId, configuredWard._id);
    assert.equal(scheduleRecord.roomNumber, 'E101');
    assert.equal(scheduleRecord.department, configuredWard.department);
  });

  it('rejects free-text equipment, missing rooms, and unconfigured ward selections', async () => {
    const freeTextResponse = responseStub();
    await createBedConfiguration({
      hospitalId: 'hospital-1',
      body: { bed_id: 'ICU-E-02', wardId: 'ward-1', room_number: 'E101', equipment_baseline: ['Custom equipment'] },
    }, freeTextResponse);
    assert.equal(freeTextResponse.statusCode, 400);
    assert.match(freeTextResponse.body.message, /predefined/i);

    Bed.findOne = async () => null;
    Ward.findOne = async () => ({ _id: 'ward-1', wardName: 'Critical Care East', department: 'ICU', rooms: [] });
    const missingRoomResponse = responseStub();
    await createBedConfiguration({
      hospitalId: 'hospital-1',
      body: { bed_id: 'ICU-E-03', wardId: 'ward-1', equipment_baseline: [] },
    }, missingRoomResponse);
    assert.equal(missingRoomResponse.statusCode, 400);
    assert.match(missingRoomResponse.body.message, /room configured/i);

    Ward.findOne = async () => null;
    const unconfiguredWardResponse = responseStub();
    await createBedConfiguration({
      hospitalId: 'hospital-1',
      body: { bed_id: 'ICU-E-04', wardId: 'missing-ward', equipment_baseline: [] },
    }, unconfiguredWardResponse);
    assert.equal(unconfiguredWardResponse.statusCode, 400);
    assert.match(unconfiguredWardResponse.body.message, /active ward/i);
  });

  it('rejects a room that is not configured under the selected ward', async () => {
    Bed.findOne = async () => null;
    Ward.findOne = async () => ({ _id: 'ward-1', wardName: 'Critical Care East', department: 'ICU', rooms: [{ roomNumber: 'E101' }] });
    const response = responseStub();

    await createBedConfiguration({
      hospitalId: 'hospital-1',
      body: { bed_id: 'ICU-E-05', wardId: 'ward-1', room_number: 'E102', equipment_baseline: [] },
    }, response);

    assert.equal(response.statusCode, 400);
    assert.match(response.body.message, /room configured under the selected ward/i);
  });

  it('keeps edited beds in a room configured under their selected ward', async () => {
    const existingBed = { _id: 'bed-doc-1', bed_id: 'ICU-E-01', room_number: 'E101', equipment_baseline: [], toObject() { return this; } };
    const existingSchedule = { _id: 'schedule-1', bedId: 'ICU-E-01', wardId: 'ward-1', roomNumber: 'E101', status: 'AVAILABLE' };
    const configuredWard = { _id: 'ward-1', wardName: 'Critical Care East', department: 'ICU', rooms: [{ roomNumber: 'E101' }] };
    Bed.findOne = async () => existingBed;
    BedSchedule.findOne = async () => existingSchedule;
    Ward.findOne = async () => configuredWard;
    Bed.findOneAndUpdate = async (_query, update) => ({ ...existingBed, ...update.$set, toObject() { return this; } });
    BedSchedule.findOneAndUpdate = async (_query, update) => ({ ...existingSchedule, ...update.$set });
    const response = responseStub();

    await updateBedConfiguration({
      hospitalId: 'hospital-1',
      params: { bedId: 'ICU-E-01' },
      body: { wardId: 'ward-1', room_number: 'E101' },
    }, response);

    assert.equal(response.body.success, true);
    assert.equal(response.body.bed.room_number, 'E101');
    assert.equal(response.body.bed.schedule.roomNumber, 'E101');
  });

  it('keeps the catalog unique and aligned with the bed model enum', () => {
    assert.equal(new Set(BED_EQUIPMENT_OPTIONS).size, BED_EQUIPMENT_OPTIONS.length);
    assert.deepEqual(Bed.schema.path('equipment_baseline').embeddedSchemaType.enumValues, BED_EQUIPMENT_OPTIONS);
    assert.ok(BED_EQUIPMENT_CATALOG.some((group) => group.category === 'Neonatal and pediatric care'));
  });
});
