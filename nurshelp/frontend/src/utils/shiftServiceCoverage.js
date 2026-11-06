const SERVICE_ALIASES = [
  { pattern: /RADIOLOGY|RADIOLOGIST|IMAGING/i, specialty: 'Radiology' },
  { pattern: /PATHOLOGY|PATHOLOGIST|HISTOLOGY/i, specialty: 'Clinical Pathology' },
  { pattern: /\bLAB\b|LABORATORY|HEMATOLOGY|LAB SCIENTIST/i, specialty: 'Lab Technician' },
  { pattern: /PHARMACY|PHARMACIST/i, specialty: 'Clinical Pharmacist' },
];

export const canonicalizeServiceCoverage = (value, configuredSpecialties = []) => {
  const requested = String(value || '').trim();
  if (!requested) return '';

  const directMatch = configuredSpecialties.find((name) => String(name).toLowerCase() === requested.toLowerCase());
  if (directMatch) return directMatch;

  const alias = SERVICE_ALIASES.find(({ pattern }) => pattern.test(requested));
  return alias
    ? configuredSpecialties.find((name) => String(name).toLowerCase() === alias.specialty.toLowerCase()) || ''
    : '';
};

export const inferStaffServiceCoverage = (staff, assignmentRole, configuredSpecialties = []) => {
  const role = String(assignmentRole || '').toUpperCase();
  const roleSpecialty = {
    RADIOLOGIST: 'Radiology',
    PATHOLOGIST: 'Clinical Pathology',
    PHARMACIST: 'Clinical Pharmacist',
    TRIAGE_NURSE: 'Emergency Nursing',
    CLINICAL_NURSE: 'Emergency Nursing',
    PHYSICIAN: 'Emergency Medicine',
  }[role];
  return canonicalizeServiceCoverage(
    staff?.clinicalSpecialty || staff?.specialty || roleSpecialty,
    configuredSpecialties
  ) || canonicalizeServiceCoverage(roleSpecialty, configuredSpecialties);
};
