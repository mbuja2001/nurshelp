import mongoose from 'mongoose';
import { Hospital } from '../models/hospital.model.js';
import { Ward } from '../models/ward.model.js';

const SERVICE_ALIASES = [
  { pattern: /RADIOLOGY|RADIOLOGIST|IMAGING/i, specialty: 'Radiology' },
  { pattern: /PATHOLOGY|PATHOLOGIST|HISTOLOGY/i, specialty: 'Clinical Pathology' },
  { pattern: /\bLAB\b|LABORATORY|HEMATOLOGY|LAB SCIENTIST/i, specialty: 'Lab Technician' },
  { pattern: /PHARMACY|PHARMACIST/i, specialty: 'Clinical Pharmacist' },
];

const specialtyNames = (departments = []) => departments
  .filter((department) => department.isActive !== false)
  .flatMap((department) => (department.specialties || []).map(
    (item) => typeof item === 'string' ? item : item.name
  ))
  .filter(Boolean);

export const canonicalizeServiceCoverage = (value, configuredSpecialties = []) => {
  const requested = String(value || '').trim();
  if (!requested) return null;

  const configured = specialtyNames(configuredSpecialties.map((name) => ({ specialties: [name] })));
  const directMatch = configured.find((name) => name.toLowerCase() === requested.toLowerCase());
  if (directMatch) return directMatch;

  const alias = SERVICE_ALIASES.find(({ pattern }) => pattern.test(requested));
  return alias
    ? configured.find((name) => name.toLowerCase() === alias.specialty.toLowerCase()) || null
    : null;
};

export const inferAssignmentServiceCoverage = ({ staff, assignmentRole, configuredSpecialties = [] }) => {
  const role = String(assignmentRole || '').toUpperCase();
  const roleService = {
    RADIOLOGIST: 'Radiology',
    PATHOLOGIST: 'Clinical Pathology',
    PHARMACIST: 'Clinical Pharmacist',
    TRIAGE_NURSE: 'Emergency Nursing',
    CLINICAL_NURSE: 'Emergency Nursing',
    PHYSICIAN: 'Emergency Medicine',
  }[role];

  return canonicalizeServiceCoverage(
    staff?.clinicalSpecialty || staff?.specialty || roleService,
    configuredSpecialties
  ) || canonicalizeServiceCoverage(roleService, configuredSpecialties);
};

export const getConfiguredHospitalServiceSpecialties = async (hospitalId) => {
  const hospital = await Hospital.findById(hospitalId).select('departments').lean();
  return hospital ? specialtyNames(hospital.departments || []) : [];
};

export const validateHospitalShiftCoverage = async ({ hospitalId, departmentName, specialty, wardId }) => {
  const hospital = await Hospital.findById(hospitalId).select('departments').lean();
  if (!hospital) return 'Hospital not found';

  const department = (hospital.departments || []).find(
    (item) => item.name === departmentName && item.isActive !== false
  );
  if (!department) return `${departmentName} is not an active department in this hospital`;

  const configured = (department.specialties || []).map(
    (item) => typeof item === 'string' ? item : item.name
  );
  if (!specialty || !configured.includes(specialty)) {
    return `Choose a specialty configured for ${departmentName}`;
  }

  if (wardId) {
    if (!mongoose.isValidObjectId(wardId)) return 'Select a valid hospital ward';
    const ward = await Ward.findOne({ _id: wardId, hospitalId, isActive: true }).lean();
    if (!ward) return 'The selected ward is inactive or belongs to another hospital';
    const wardDepartment = ward.departmentName || ({
      MATERNITY: 'OB',
      PEDIATRICS: 'Pediatrics',
      PSYCH: 'Psychiatry',
      GENERAL: 'General',
    }[ward.department] || ward.department);
    if (wardDepartment !== departmentName) {
      return `The selected ward is not part of ${departmentName}`;
    }
  }

  return null;
};
