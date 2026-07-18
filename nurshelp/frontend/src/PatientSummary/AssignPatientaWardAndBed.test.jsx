import { describe, expect, it } from 'vitest';
import { getPatientDisplayName, getEncounterIdentifier, getBedLabel } from './assignWardBedUtils';

describe('assignWardBedUtils', () => {
  it('returns the patient display name from common encounter shapes', () => {
    expect(getPatientDisplayName({ patientName: 'Ada' })).toBe('Ada');
    expect(getPatientDisplayName({ patient: { name: 'Grace' } })).toBe('Grace');
    expect(getPatientDisplayName({ firstName: 'Linus' })).toBe('Linus');
  });

  it('returns the encounter identifier from common fields', () => {
    expect(getEncounterIdentifier({ _id: 'enc-1' })).toBe('enc-1');
    expect(getEncounterIdentifier({ id: 'enc-2' })).toBe('enc-2');
  });

  it('returns the bed label from the available bed shapes', () => {
    expect(getBedLabel({ bedId: 'A-01' })).toBe('A-01');
    expect(getBedLabel({ bed_number: 'B-02' })).toBe('B-02');
  });
});
