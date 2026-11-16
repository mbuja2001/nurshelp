import { describe, expect, it } from 'vitest';
import {
  getAllowedShiftDuties,
  HOSPITAL_STAFF_ROLES,
  resolveShiftAssignmentDuty,
} from './hospitalStaffRoleCatalog';

describe('shared staff shift duties', () => {
  it('provides allowed duties for every onboarding position', () => {
    HOSPITAL_STAFF_ROLES.forEach((role) => {
      expect(getAllowedShiftDuties(role.code, role.systemRole).length).toBeGreaterThan(0);
    });
  });

  it('offers only the triage duty to a triage nurse and resets an invalid selection', () => {
    const duties = getAllowedShiftDuties('TRIAGE_NURSE', 'NURSE');
    expect(duties).toEqual(['TRIAGE_NURSE']);
    expect(resolveShiftAssignmentDuty({ positionCode: 'TRIAGE_NURSE', systemRole: 'NURSE' }, 'CLINICAL_NURSE')).toBe('TRIAGE_NURSE');
  });

  it('requires an explicit duty for legacy nurses and registered nurses with two choices', () => {
    expect(resolveShiftAssignmentDuty({ systemRole: 'NURSE' })).toBe('');
    expect(resolveShiftAssignmentDuty({ positionCode: 'REGISTERED_NURSE', systemRole: 'NURSE' })).toBe('');
    expect(getAllowedShiftDuties('', 'NURSE')).toEqual(['TRIAGE_NURSE', 'CLINICAL_NURSE']);
  });

  it('defaults legacy specialist roles to their specialty duty only', () => {
    const specializedDuties = [
      ['PHYSICIAN', 'PHYSICIAN'],
      ['SPECIALIST', 'SPECIALIST'],
      ['RADIOLOGIST', 'RADIOLOGIST'],
      ['PATHOLOGIST', 'PATHOLOGIST'],
      ['PHARMACIST', 'PHARMACIST'],
      ['TECHNICIAN', 'TECHNICIAN'],
    ];

    specializedDuties.forEach(([systemRole, duty]) => {
      const staff = { systemRole };
      expect(getAllowedShiftDuties('', systemRole)).toEqual([duty]);
      expect(resolveShiftAssignmentDuty(staff)).toBe(duty);
      expect(resolveShiftAssignmentDuty(staff, 'SUPPORT')).toBe(duty);
    });
  });

  it('uses the pharmacist duty for pharmacist positions and replaces stale incompatible selections', () => {
    const pharmacist = { positionCode: 'PHARMACIST', systemRole: 'PHARMACIST' };
    expect(getAllowedShiftDuties(pharmacist.positionCode, pharmacist.systemRole)).toEqual(['PHARMACIST']);
    expect(resolveShiftAssignmentDuty(pharmacist)).toBe('PHARMACIST');
    expect(resolveShiftAssignmentDuty(pharmacist, 'SUPPORT')).toBe('PHARMACIST');
  });
});
