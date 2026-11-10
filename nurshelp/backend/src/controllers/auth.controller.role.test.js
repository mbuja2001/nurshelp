import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { deriveTokenRole } from './auth.controller.js';

describe('deriveTokenRole', () => {
  it('preserves technician identity regardless of specialty', () => {
    assert.equal(deriveTokenRole('TECHNICIAN'), 'TECHNICIAN');
  });

  it('preserves specialist roles only when they are explicitly assigned', () => {
    assert.equal(deriveTokenRole('RADIOLOGIST'), 'RADIOLOGIST');
    assert.equal(deriveTokenRole('PATHOLOGIST'), 'PATHOLOGIST');
    assert.equal(deriveTokenRole('PHARMACIST'), 'PHARMACIST');
  });

  it('keeps support staff out of the administrator role', () => {
    assert.equal(deriveTokenRole('SUPPORT'), 'SUPPORT');
    assert.equal(deriveTokenRole('ADMINISTRATIVE'), 'ADMIN');
  });
});
