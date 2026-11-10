export const allowedShiftDuties = Object.freeze({
  PHYSICIAN: Object.freeze(['PHYSICIAN']),
  REGISTERED_NURSE: Object.freeze(['TRIAGE_NURSE', 'CLINICAL_NURSE']),
  TRIAGE_NURSE: Object.freeze(['TRIAGE_NURSE']),
  CLINICAL_NURSE: Object.freeze(['CLINICAL_NURSE']),
  RADIOLOGIST: Object.freeze(['RADIOLOGIST']),
  PATHOLOGIST: Object.freeze(['PATHOLOGIST']),
  PHARMACIST: Object.freeze(['PHARMACIST']),
  TECHNICIAN: Object.freeze(['TECHNICIAN']),
  ADMINISTRATIVE_SUPPORT: Object.freeze(['SUPPORT']),
  CERTIFIED_NURSING_ASSISTANT: Object.freeze(['SUPPORT']),
  NURSE_AIDE: Object.freeze(['SUPPORT']),
  PATIENT_CARE_TECHNICIAN: Object.freeze(['SUPPORT']),
  GERIATRIC_CARE_ASSISTANT: Object.freeze(['SUPPORT']),
  PEDIATRIC_CARE_ASSISTANT: Object.freeze(['SUPPORT']),
  BEHAVIORAL_HEALTH_AIDE: Object.freeze(['SUPPORT']),
  UNIT_SECRETARY: Object.freeze(['SUPPORT']),
  WARD_CLERK: Object.freeze(['SUPPORT']),
  HOSPITAL_ORDERLY: Object.freeze(['SUPPORT']),
  PHYSICAL_THERAPY_ASSISTANT: Object.freeze(['SUPPORT']),
  OCCUPATIONAL_THERAPY_ASSISTANT: Object.freeze(['SUPPORT']),
  SPEECH_LANGUAGE_PATHOLOGY_ASSISTANT: Object.freeze(['SUPPORT']),
  DIETETIC_ASSISTANT: Object.freeze(['SUPPORT']),
  DIETARY_AIDE: Object.freeze(['SUPPORT']),
  PHARMACY_ASSISTANT: Object.freeze(['SUPPORT']),
  MEDICAL_ASSISTANT: Object.freeze(['SUPPORT']),
  DENTAL_ASSISTANT: Object.freeze(['SUPPORT']),
  ANESTHESIA_ASSISTANT: Object.freeze(['SUPPORT']),
  ANESTHESIA_TECHNOLOGIST_ASSISTANT: Object.freeze(['SUPPORT']),
  ENDOSCOPY_GI_ASSISTANT: Object.freeze(['SUPPORT']),
  CENTRAL_STERILE_SUPPLY_ASSISTANT: Object.freeze(['SUPPORT']),
  PATIENT_TRANSPORTER: Object.freeze(['SUPPORT']),
  HOSPITAL_PORTER: Object.freeze(['SUPPORT']),
  MEDICAL_SOCIAL_WORK_ASSISTANT: Object.freeze(['SUPPORT']),
  DISCHARGE_PLANNER: Object.freeze(['SUPPORT']),
  PATIENT_NAVIGATOR: Object.freeze(['SUPPORT']),
  LANGUAGE_INTERPRETER: Object.freeze(['SUPPORT']),
});

export const legacySystemRoleDuties = Object.freeze({
  NURSE: Object.freeze(['TRIAGE_NURSE', 'CLINICAL_NURSE']),
  PHYSICIAN: Object.freeze(['PHYSICIAN', 'SPECIALIST']),
  SPECIALIST: Object.freeze(['SPECIALIST']),
  RADIOLOGIST: Object.freeze(['RADIOLOGIST', 'SPECIALIST']),
  PATHOLOGIST: Object.freeze(['PATHOLOGIST', 'SPECIALIST']),
  PHARMACIST: Object.freeze(['PHARMACIST', 'SPECIALIST']),
  TECHNICIAN: Object.freeze(['TECHNICIAN', 'SUPPORT']),
  SUPPORT: Object.freeze(['SUPPORT']),
  SUPERUSER: Object.freeze(['SUPPORT']),
  ADMINISTRATIVE: Object.freeze(['SUPPORT']),
});

export const SHIFT_DUTY_LABELS = Object.freeze({
  TRIAGE_NURSE: 'Triage nurse',
  CLINICAL_NURSE: 'Clinical nurse',
  PHYSICIAN: 'Physician',
  SPECIALIST: 'Specialist',
  RADIOLOGIST: 'Radiologist',
  PATHOLOGIST: 'Pathologist',
  PHARMACIST: 'Pharmacist',
  TECHNICIAN: 'Technician',
  SUPPORT: 'Support',
});

export const getAllowedShiftDuties = (positionCode, systemRole) => {
  const normalizedPositionCode = String(positionCode || '').trim().toUpperCase();
  if (normalizedPositionCode) return allowedShiftDuties[normalizedPositionCode] || [];

  const normalizedSystemRole = String(systemRole || '').trim().toUpperCase().replace(/[ -]+/g, '_');
  return legacySystemRoleDuties[normalizedSystemRole] || [];
};

export const resolveShiftAssignmentDuty = (staff, selectedDuty = '') => {
  const positionCode = String(staff?.positionCode || '').trim();
  const allowedDuties = getAllowedShiftDuties(positionCode, staff?.systemRole || staff?.role);
  if (allowedDuties.includes(selectedDuty)) return selectedDuty;
  return positionCode && allowedDuties.length === 1 ? allowedDuties[0] : '';
};

export const isShiftDutyAllowed = (staff, duty) => (
  getAllowedShiftDuties(staff?.positionCode, staff?.systemRole || staff?.role).includes(duty)
);
