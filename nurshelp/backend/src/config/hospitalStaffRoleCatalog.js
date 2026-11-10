export {
  allowedShiftDuties,
  getAllowedShiftDuties,
  isShiftDutyAllowed,
  legacySystemRoleDuties,
  resolveShiftAssignmentDuty,
  SHIFT_DUTY_LABELS
} from '../../../shared/hospitalStaffRoleCatalog.js';

export const HOSPITAL_STAFF_ROLES = [
  { code: 'PHYSICIAN', label: 'Physician', category: 'Licensed clinical', systemRole: 'PHYSICIAN', defaultTitle: 'MD', requiresCredentials: true },
  { code: 'REGISTERED_NURSE', label: 'Registered nurse', category: 'Licensed clinical', systemRole: 'NURSE', defaultTitle: 'RN', requiresCredentials: true },
  { code: 'TRIAGE_NURSE', label: 'Triage nurse', category: 'Licensed clinical', systemRole: 'NURSE', defaultTitle: 'RN', requiresCredentials: true },
  { code: 'CLINICAL_NURSE', label: 'Clinical nurse', category: 'Licensed clinical', systemRole: 'NURSE', defaultTitle: 'RN', requiresCredentials: true },
  { code: 'RADIOLOGIST', label: 'Radiologist', category: 'Licensed clinical', systemRole: 'RADIOLOGIST', defaultTitle: 'MD', requiresCredentials: true },
  { code: 'PATHOLOGIST', label: 'Pathologist', category: 'Licensed clinical', systemRole: 'PATHOLOGIST', defaultTitle: 'PhD', requiresCredentials: true },
  { code: 'PHARMACIST', label: 'Pharmacist', category: 'Licensed clinical', systemRole: 'PHARMACIST', defaultTitle: 'PhD', requiresCredentials: true },
  { code: 'TECHNICIAN', label: 'Clinical / diagnostic technician', category: 'Clinical support', systemRole: 'TECHNICIAN', defaultTitle: '', requiresCredentials: false },
  { code: 'ADMINISTRATIVE_SUPPORT', label: 'Administrative support (non-manager)', category: 'Administrative support', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'CERTIFIED_NURSING_ASSISTANT', label: 'Certified nursing assistant (CNA)', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'NURSE_AIDE', label: 'Nurse aide', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'PATIENT_CARE_TECHNICIAN', label: 'Patient care technician (PCT)', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'GERIATRIC_CARE_ASSISTANT', label: 'Geriatric care assistant', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'PEDIATRIC_CARE_ASSISTANT', label: 'Pediatric care assistant', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'BEHAVIORAL_HEALTH_AIDE', label: 'Psychiatric / behavioral health aide', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'UNIT_SECRETARY', label: 'Unit secretary', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'WARD_CLERK', label: 'Ward clerk', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'HOSPITAL_ORDERLY', label: 'Hospital orderly', category: 'Nursing, ward & direct care', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'PHYSICAL_THERAPY_ASSISTANT', label: 'Physical therapy assistant (PTA)', category: 'Therapy & allied health', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'OCCUPATIONAL_THERAPY_ASSISTANT', label: 'Occupational therapy assistant (OTA)', category: 'Therapy & allied health', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'SPEECH_LANGUAGE_PATHOLOGY_ASSISTANT', label: 'Speech-language pathology assistant (SLPA)', category: 'Therapy & allied health', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'DIETETIC_ASSISTANT', label: 'Dietetic assistant', category: 'Therapy & allied health', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'DIETARY_AIDE', label: 'Dietary aide', category: 'Therapy & allied health', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'PHARMACY_ASSISTANT', label: 'Pharmacy assistant', category: 'Therapy & allied health', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'MEDICAL_ASSISTANT', label: 'Medical assistant (MA)', category: 'Procedural, surgical & clinic support', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'DENTAL_ASSISTANT', label: 'Dental assistant', category: 'Procedural, surgical & clinic support', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'ANESTHESIA_ASSISTANT', label: 'Anesthesia assistant', category: 'Procedural, surgical & clinic support', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'ANESTHESIA_TECHNOLOGIST_ASSISTANT', label: 'Anesthesia technologist assistant', category: 'Procedural, surgical & clinic support', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'ENDOSCOPY_GI_ASSISTANT', label: 'Endoscopy / GI assistant', category: 'Procedural, surgical & clinic support', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'CENTRAL_STERILE_SUPPLY_ASSISTANT', label: 'Central sterile supply assistant', category: 'Procedural, surgical & clinic support', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'PATIENT_TRANSPORTER', label: 'Patient transporter', category: 'Transport, logistics & care coordination', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'HOSPITAL_PORTER', label: 'Hospital porter', category: 'Transport, logistics & care coordination', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'MEDICAL_SOCIAL_WORK_ASSISTANT', label: 'Medical social work assistant', category: 'Transport, logistics & care coordination', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'DISCHARGE_PLANNER', label: 'Discharge planner', category: 'Transport, logistics & care coordination', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'PATIENT_NAVIGATOR', label: 'Patient navigator', category: 'Transport, logistics & care coordination', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false },
  { code: 'LANGUAGE_INTERPRETER', label: 'Language interpreter', category: 'Transport, logistics & care coordination', systemRole: 'SUPPORT', defaultTitle: '', requiresCredentials: false }
];

export const findHospitalStaffRole = (code) => HOSPITAL_STAFF_ROLES.find((role) => role.code === String(code || '').trim().toUpperCase());

export const legacyHospitalStaffRole = (role) => {
  const canonicalRole = String(role || '').trim().toUpperCase();
  if (canonicalRole === 'ADMINISTRATIVE') return null;
  const legacyCode = {
    NURSE: 'REGISTERED_NURSE',
    PHYSICIAN: 'PHYSICIAN',
    TECHNICIAN: 'TECHNICIAN',
    PHARMACIST: 'PHARMACIST',
    RADIOLOGIST: 'RADIOLOGIST',
    PATHOLOGIST: 'PATHOLOGIST'
  }[canonicalRole];
  return findHospitalStaffRole(legacyCode || canonicalRole);
};
