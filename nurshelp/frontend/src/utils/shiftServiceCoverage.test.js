import { describe, expect, it } from 'vitest';
import { canonicalizeServiceCoverage, inferStaffServiceCoverage } from './shiftServiceCoverage';

const services = ['Emergency Medicine', 'Emergency Nursing', 'Radiology', 'Clinical Pathology', 'Lab Technician', 'Clinical Pharmacist'];

describe('shift service coverage', () => {
  it.each([
    ['Radiology Technician', 'Radiology'],
    ['Clinical Pathology', 'Clinical Pathology'],
    ['Laboratory Technician', 'Lab Technician'],
    ['Pharmacist', 'Clinical Pharmacist'],
  ])('maps %s to the configured %s service', (input, expected) => {
    expect(canonicalizeServiceCoverage(input, services)).toBe(expected);
  });

  it('uses the person assignment service independently of the ED shift specialty', () => {
    expect(inferStaffServiceCoverage(
      { clinicalSpecialty: 'Radiology Technician' },
      'SUPPORT',
      services
    )).toBe('Radiology');
  });

  it('fails closed when the person service is not configured in the hospital', () => {
    expect(canonicalizeServiceCoverage('Radiology', ['Emergency Medicine'])).toBe('');
  });
});
