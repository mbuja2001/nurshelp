import { describe, expect, it } from 'vitest';
import { canAssignSelfAsDownstreamNurse, getEffectiveDutyRole } from './assignmentValidation';

describe('downstream nurse assignment validation', () => {
  it('blocks a triage nurse from assigning themself as the clinical nurse', () => {
    expect(canAssignSelfAsDownstreamNurse('staff-1', 'staff-1', 'TRIAGE_NURSE')).toBe(false);
    expect(canAssignSelfAsDownstreamNurse('staff-1', 'staff-1', null)).toBe(false);
  });

  it('allows self-assignment only when the effective shift duty is clinical nurse', () => {
    expect(canAssignSelfAsDownstreamNurse('staff-1', 'staff-1', 'CLINICAL_NURSE')).toBe(true);
    expect(canAssignSelfAsDownstreamNurse('staff-1', 'staff-2', 'TRIAGE_NURSE')).toBe(true);
  });

  it('normalizes dashboard contexts to effective shift roles', () => {
    expect(getEffectiveDutyRole('TRIAGE_NURSE')).toBe('TRIAGE_NURSE');
    expect(getEffectiveDutyRole('CLINICAL_NURSE_ER')).toBe('CLINICAL_NURSE');
    expect(getEffectiveDutyRole('NO_SHIFT')).toBe(null);
  });
});
