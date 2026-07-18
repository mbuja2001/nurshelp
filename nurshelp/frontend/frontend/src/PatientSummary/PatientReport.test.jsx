import { computeDisplayBP } from './PatientReport';

describe('computeDisplayBP helper', () => {
  it('shows systolic/diastolic when available in vitals', () => {
    const vitals = { bp_systolic: 120, bp_diastolic: 80 };
    expect(computeDisplayBP(vitals, {})).toBe('120/80');
  });

  it('falls back to vitals.bp string when numbers missing', () => {
    const vitals = { bp: '110/70' };
    expect(computeDisplayBP(vitals, {})).toBe('110/70');
  });

  it('uses triage.vitals_parsed when vitals object lacks values', () => {
    const triage = { vitals_parsed: { bp_systolic: 95, bp_diastolic: 60 } };
    expect(computeDisplayBP({}, triage)).toBe('95/60');
  });

  it('returns -- when no BP information is present anywhere', () => {
    expect(computeDisplayBP({}, {})).toBe('--');
  });
});
