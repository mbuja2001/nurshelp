import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeEhrProtocols, resolveEhrIntegrationAccess } from './ehrIntegrationAccess.js';

test('normalizes EHR protocol lists to supported unique protocol identifiers', () => {
  assert.deepEqual(normalizeEhrProtocols(['hl7v2', 'hl7v2', 'fhir-r4', 'cda', null]), ['hl7v2', 'fhir-r4']);
  assert.deepEqual(normalizeEhrProtocols('hl7v2'), []);
});

test('allows sandbox admin fallback only when no account policy exists', () => {
  assert.equal(resolveEhrIntegrationAccess(null, 'SANDBOX').canEnableSandboxAccess, true);
  assert.equal(resolveEhrIntegrationAccess(null, undefined).canEnableSandboxAccess, true);
  assert.equal(resolveEhrIntegrationAccess({ source: 'ACCOUNT_OWNER', enabledProtocols: ['fhir-r4'] }, 'SANDBOX').canEnableSandboxAccess, false);
  assert.equal(resolveEhrIntegrationAccess(null, 'PRODUCTION').canEnableSandboxAccess, false);
});

test('reports sandbox admin access as an attestation, not as a live connection', () => {
  const access = resolveEhrIntegrationAccess({
    source: 'HOSPITAL_ADMIN_SANDBOX',
    enabledProtocols: ['hl7v2'],
    attestedByName: 'QA administrator',
    reason: 'Licensed test environment is not yet connected',
  }, 'SANDBOX');

  assert.equal(access.isSandboxConfiguration, true);
  assert.deepEqual(access.enabledProtocols, ['hl7v2']);
  assert.equal(access.attestedByName, 'QA administrator');
  assert.equal(access.canEnableSandboxAccess, false);
});

test('production environments never inherit sandbox EHR access', () => {
  const access = resolveEhrIntegrationAccess({
    source: 'ACCOUNT_OWNER',
    enabledProtocols: ['hl7v2', 'fhir-r4'],
  }, 'PRODUCTION');

  assert.deepEqual(access.enabledProtocols, []);
  assert.equal(access.canEnableSandboxAccess, false);
  assert.equal(access.isSandboxConfiguration, false);
});
