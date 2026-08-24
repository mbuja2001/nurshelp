import test from 'node:test';
import assert from 'node:assert/strict';

import { Ward } from '../models/ward.model.js';
import {
  normalizeWardDepartment,
  buildWardDefinitionsFromBeds,
  inferBedType,
} from './bedWardBootstrap.js';

test('normalizeWardDepartment supports common ward labels', () => {
  assert.equal(normalizeWardDepartment('ICU'), 'ICU');
  assert.equal(normalizeWardDepartment('Medical Ward'), 'GENERAL');
  assert.equal(normalizeWardDepartment('Pediatric Ward'), 'PEDIATRICS');
  assert.equal(normalizeWardDepartment('Emergency Department'), 'ED');
});

test('buildWardDefinitionsFromBeds groups bed payloads by ward', () => {
  const wards = buildWardDefinitionsFromBeds([
    { bed_id: 'ICU-01', ward_unit: 'ICU', equipment_baseline: ['High-Acuity Monitor', 'Mechanical Ventilator'] },
    { bed_id: 'ICU-02', ward_unit: 'ICU', equipment_baseline: ['High-Acuity Monitor'] },
    { bed_id: 'GEN-01', ward_unit: 'Medical Ward', equipment_baseline: ['Oxygen Delivery'] },
  ]);

  assert.equal(wards.length, 2);
  assert.deepEqual(wards.map((w) => w.department).sort(), ['GENERAL', 'ICU']);
  assert.equal(wards.find((w) => w.department === 'ICU').bedCount, 2);
  assert.equal(wards.find((w) => w.department === 'GENERAL').bedCount, 1);
});

test('inferBedType identifies ICU beds from equipment features', () => {
  assert.equal(
    inferBedType({ ward_unit: 'ICU', equipment_baseline: ['Mechanical Ventilator', 'Defibrillator'] }),
    'INTENSIVE'
  );
  assert.equal(
    inferBedType({ ward_unit: 'Medical Ward', equipment_baseline: ['Oxygen Delivery'] }),
    'STANDARD'
  );
});

test('Ward schema accepts all real clinical departments used by the bootstrap flow', () => {
  const ward = new Ward({
    wardId: 'CICU-Ward-A',
    wardName: 'Cardiac ICU - Ward A',
    department: 'CICU',
    hospitalId: '507f1f77bcf86cd799439011',
    location: { floor: 2, wing: 'Main', area: '' },
    totalBeds: 2,
    occupiedBeds: 0,
    availableBeds: 2,
    cleaningBeds: 0,
    maintenanceBeds: 0,
    isActive: true,
  });

  assert.equal(ward.validateSync(), undefined);

  const neuroWard = new Ward({
    wardId: 'NEURO-Ward-A',
    wardName: 'Neuro ICU - Ward A',
    department: 'NEURO',
    hospitalId: '507f1f77bcf86cd799439011',
    location: { floor: 3, wing: 'Main', area: '' },
    totalBeds: 2,
    occupiedBeds: 0,
    availableBeds: 2,
    cleaningBeds: 0,
    maintenanceBeds: 0,
    isActive: true,
  });

  assert.equal(neuroWard.validateSync(), undefined);
});
