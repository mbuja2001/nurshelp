import { describe, expect, it } from 'vitest';
import {
  getConfiguredDepartmentSpecialties,
  resolveConfiguredDepartmentName,
  resolveStaffPositionCode
} from './hospitalStaffOnboarding';

describe('configured staff onboarding options', () => {
  it('returns unique specialties configured for the selected active department', () => {
    const departments = [
      { name: 'ED', isActive: true, specialties: [{ name: 'Emergency Medicine' }, 'Emergency Medicine'] },
      { name: 'ICU', isActive: true, specialties: [{ name: 'Critical Care Nursing' }] },
    ];

    expect(getConfiguredDepartmentSpecialties(departments, 'ed')).toEqual(['Emergency Medicine']);
  });

  it('does not expose specialties from inactive or unconfigured departments', () => {
    const departments = [{ name: 'ED', isActive: false, specialties: ['Emergency Medicine'] }];

    expect(getConfiguredDepartmentSpecialties(departments, 'ED')).toEqual([]);
    expect(getConfiguredDepartmentSpecialties(departments, 'ICU')).toEqual([]);
  });

  it('resolves known legacy staff positions without losing more specific position titles', () => {
    expect(resolveStaffPositionCode({ systemRole: 'NURSE', positionTitle: 'Clinical nurse' })).toBe('CLINICAL_NURSE');
    expect(resolveStaffPositionCode({ systemRole: 'NURSE' })).toBe('');
    expect(resolveStaffPositionCode({ systemRole: 'PHYSICIAN' })).toBe('PHYSICIAN');
  });

  it('maps supported legacy department values to the active hospital department', () => {
    const departments = [
      { name: 'ED', isActive: true },
      { name: 'General', isActive: true },
    ];

    expect(resolveConfiguredDepartmentName(departments, 'EMERGENCY')).toBe('ED');
    expect(resolveConfiguredDepartmentName(departments, 'GENERAL_WARD')).toBe('General');
    expect(resolveConfiguredDepartmentName(departments, 'RADIOLOGY')).toBe('');
  });
});
