import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeSupportTicketInput, normalizeSupportTicketUpdate } from './supportTicketValidation.js';

test('normalizes only approved ticket fields and excludes sensitive or extra input', () => {
  const result = normalizeSupportTicketInput({
    title: ' Dashboard unavailable ',
    description: 'The assigned shift dashboard did not load.',
    category: 'system_error',
    severity: 'high',
    pagePath: '/shift-dashboard',
    failedRequest: '/api/dashboard/config',
    primaryStatus: 500,
    fallbackStatus: 403,
    errorCode: 'DASHBOARD_CONFIG_FAILED',
    errorMessage: 'Dashboard configuration could not be loaded',
    occurredAt: '2026-11-12T10:00:00.000Z',
    token: 'must-not-be-persisted',
    patientId: 'must-not-be-persisted',
    headers: { authorization: 'must-not-be-persisted' }
  });

  assert.equal(result.error, undefined);
  assert.equal(result.value.title, 'Dashboard unavailable');
  assert.equal(result.value.category, 'SYSTEM_ERROR');
  assert.equal(result.value.primaryStatus, 500);
  assert.equal(result.value.fallbackStatus, 403);
  assert.equal('token' in result.value, false);
  assert.equal('patientId' in result.value, false);
  assert.equal('headers' in result.value, false);
});

test('rejects unsafe page paths and invalid HTTP statuses', () => {
  const unsafePath = normalizeSupportTicketInput({
    title: 'Dashboard unavailable',
    description: 'The dashboard could not load.',
    category: 'SYSTEM_ERROR',
    severity: 'HIGH',
    pagePath: '/shift-dashboard?token=secret'
  });
  assert.match(unsafePath.error, /pathname/);

  const invalidStatus = normalizeSupportTicketInput({
    title: 'Dashboard unavailable',
    description: 'The dashboard could not load.',
    category: 'SYSTEM_ERROR',
    severity: 'HIGH',
    fallbackStatus: 99
  });
  assert.match(invalidStatus.error, /HTTP statuses/);
});

test('allows only known update fields and valid lifecycle states', () => {
  const result = normalizeSupportTicketUpdate({
    status: 'in_progress',
    assignToMe: true,
    resolutionNote: 'Investigating',
    hospitalId: 'must-not-be-persisted'
  });

  assert.equal(result.error, undefined);
  assert.deepEqual(result.value, {
    status: 'IN_PROGRESS',
    assignToMe: true,
    resolutionNote: 'Investigating'
  });
  assert.match(normalizeSupportTicketUpdate({ status: 'CANCELLED' }).error, /status/);
});
