import { describe, it, expect } from 'vitest';
import { getDashboardRoute, normalizeRoleLabel, resolveTechnicianSpecialty } from './jwt';

describe('normalizeRoleLabel', () => {
  it('keeps pharmacy technicians distinct from pharmacists', () => {
    expect(normalizeRoleLabel('TECHNICIAN', 'Pharmacy')).toBe('TECHNICIAN');
    expect(normalizeRoleLabel('PHARMACY_TECHNICIAN')).toBe('TECHNICIAN');
  });

  it('keeps radiology technicians distinct from radiologists', () => {
    expect(normalizeRoleLabel('TECHNICIAN', 'Radiology')).toBe('TECHNICIAN');
    expect(normalizeRoleLabel('RADIOLOGY_TECHNICIAN')).toBe('TECHNICIAN');
  });

  it('keeps pathology technicians distinct from pathologists', () => {
    expect(normalizeRoleLabel('TECHNICIAN', 'Pathology')).toBe('TECHNICIAN');
    expect(normalizeRoleLabel('PATHOLOGY_TECHNICIAN')).toBe('TECHNICIAN');
  });

  it('does not infer specialist privileges from a technician specialty', () => {
    expect(normalizeRoleLabel('TECHNICIAN', 'Laboratory')).toBe('TECHNICIAN');
    expect(resolveTechnicianSpecialty('TECHNICIAN', 'Laboratory')).toBeNull();
    expect(getDashboardRoute('TECHNICIAN', 'Laboratory')).toBe('TECHNICIAN_UNCONFIGURED');
  });

  it.each([
    ['Pharmacy', 'PHARMACY'],
    ['Radiology', 'RADIOLOGY'],
    ['Clinical Pathology', 'PATHOLOGY']
  ])('resolves %s to a specialty-specific technician workspace', (specialty, expected) => {
    expect(resolveTechnicianSpecialty('TECHNICIAN', specialty)).toBe(expected);
    expect(getDashboardRoute('TECHNICIAN', specialty)).toBe('TECHNICIAN');
  });

  it('does not infer a technician specialty for a non-technician identity', () => {
    expect(resolveTechnicianSpecialty('NURSE', 'Radiology')).toBeNull();
  });

  it('still resolves explicitly assigned specialists by specialty', () => {
    expect(normalizeRoleLabel('SPECIALIST', 'Radiology')).toBe('RADIOLOGIST');
  });
});
