import test from 'node:test';
import assert from 'node:assert/strict';
import { buildNotificationOwnershipFilter } from './notificationOwnership.js';

test('notification mutation filter scopes by id, clinician, and hospital', () => {
  assert.deepEqual(
    buildNotificationOwnershipFilter({
      notificationId: 'notification-a',
      clinicianId: 'clinician-a',
      hospitalId: 'hospital-a'
    }),
    {
      _id: 'notification-a',
      clinician_id: 'clinician-a',
      hospital_id: 'hospital-a'
    }
  );
});

test('a different clinician or hospital cannot match the ownership filter', () => {
  const filter = buildNotificationOwnershipFilter({
    notificationId: 'notification-a',
    clinicianId: 'clinician-a',
    hospitalId: 'hospital-a'
  });

  assert.notDeepEqual(filter, {
    _id: 'notification-a',
    clinician_id: 'clinician-b',
    hospital_id: 'hospital-a'
  });
  assert.notDeepEqual(filter, {
    _id: 'notification-a',
    clinician_id: 'clinician-a',
    hospital_id: 'hospital-b'
  });
});
