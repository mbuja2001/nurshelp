import { describe, it, expect } from 'vitest';
import { normalizeRoleLabel } from './jwt';

describe('normalizeRoleLabel', () => {
  it('maps technician roles to a pharmacist dashboard when specialty is pharmacy', () => {
    expect(normalizeRoleLabel('TECHNICIAN', 'Pharmacy')).toBe('PHARMACIST');
  });

  it('maps technician roles to a radiologist dashboard when specialty is radiology', () => {
    expect(normalizeRoleLabel('TECHNICIAN', 'Radiology')).toBe('RADIOLOGIST');
  });

  it('maps technician roles to a pathologist dashboard when specialty is pathology', () => {
    expect(normalizeRoleLabel('TECHNICIAN', 'Pathology')).toBe('PATHOLOGIST');
  });
});
