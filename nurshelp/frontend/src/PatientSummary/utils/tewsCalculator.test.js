import { describe, expect, it } from 'vitest';
import { calculateTews, getInitialTewsProfile } from './tewsCalculator';

describe('TEWS calculator', () => {
  it('matches the manual adult scoring fields and excludes trauma from total', () => {
    const result = calculateTews('adult', {
      temperature: 39,
      heartRate: 120,
      respiratoryRate: 24,
      systolicBP: 75,
      avpu: 'Alert',
      mobility: 'Walks',
      traumaPresent: true
    });

    expect(result.components).toEqual({ temperature: 2, heartRate: 1, respiratoryRate: 1, systolicBP: 1, avpu: 0, mobility: 0, trauma: 1 });
    expect(result.total).toBe(5);
    expect(result.traumaModifier).toBe(1);
    expect(result.priority).toBe('orange');
  });

  it('uses the same adult profile split as manual triage', () => {
    expect(getInitialTewsProfile({ age: 13 })).toBe('adult');
    expect(getInitialTewsProfile({ age: 12, height_cm: 90 })).toBe('child-younger');
  });
});
