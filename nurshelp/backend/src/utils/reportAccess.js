const REPORT_TYPES = Object.freeze({
  TRIAGE: 'TRIAGE',
  CLINICAL_NOTE: 'CLINICAL_NOTE',
  SURGERY: 'SURGERY',
  PRESCRIPTION: 'PRESCRIPTION',
  LAB: 'LAB',
  DISCHARGE: 'DISCHARGE'
});

const ACCESS = Object.freeze({
  NONE: 'none',
  SUMMARY: 'summary',
  FULL: 'full'
});

const ROLE_POLICIES = Object.freeze({
  TRIAGE_NURSE: {
    TRIAGE: { read: ACCESS.FULL, edit: true },
    CLINICAL_NOTE: { read: ACCESS.SUMMARY, edit: false },
    SURGERY: { read: ACCESS.SUMMARY, edit: false },
    PRESCRIPTION: { read: ACCESS.SUMMARY, edit: false },
    LAB: { read: ACCESS.SUMMARY, edit: false },
    DISCHARGE: { read: ACCESS.NONE, edit: false }
  },
  NURSE: {
    TRIAGE: { read: ACCESS.FULL, edit: false },
    CLINICAL_NOTE: { read: ACCESS.FULL, edit: false },
    SURGERY: { read: ACCESS.FULL, edit: false },
    PRESCRIPTION: { read: ACCESS.FULL, edit: false },
    LAB: { read: ACCESS.FULL, edit: false },
    DISCHARGE: { read: ACCESS.FULL, edit: false }
  },
  CLINICAL_NURSE: {
    TRIAGE: { read: ACCESS.FULL, edit: false },
    CLINICAL_NOTE: { read: ACCESS.FULL, edit: false },
    SURGERY: { read: ACCESS.FULL, edit: false },
    PRESCRIPTION: { read: ACCESS.FULL, edit: false },
    LAB: { read: ACCESS.FULL, edit: false },
    DISCHARGE: { read: ACCESS.FULL, edit: false }
  },
  PHYSICIAN: {
    TRIAGE: { read: ACCESS.FULL, edit: true },
    CLINICAL_NOTE: { read: ACCESS.FULL, edit: true },
    SURGERY: { read: ACCESS.FULL, edit: true },
    PRESCRIPTION: { read: ACCESS.FULL, edit: true },
    LAB: { read: ACCESS.FULL, edit: true },
    DISCHARGE: { read: ACCESS.FULL, edit: true }
  },
  PHARMACIST: {
    PRESCRIPTION: { read: ACCESS.FULL, edit: true }
  },
  ADMIN: Object.fromEntries(Object.values(REPORT_TYPES).map((type) => [type, { read: ACCESS.FULL, edit: true }]))
});

const idString = (value) => {
  if (!value) return null;
  if (typeof value === 'object' && value._id) return String(value._id);
  return String(value);
};

const roleString = (user) => {
  const role = String(user?.role || '').toUpperCase().trim();
  if (['DOCTOR', 'SURGEON', 'SPECIALIST'].includes(role)) return 'PHYSICIAN';
  if (['CLINICAL_NURSE_ER', 'CLINICAL_NURSE_WARD'].includes(role)) return 'CLINICAL_NURSE';
  if (['TRIAGE', 'TRIAGE_NURSE_ER'].includes(role)) return 'TRIAGE_NURSE';
  if (['MANAGER', 'SHIFT_MANAGER', 'DEPARTMENT_MANAGER'].includes(role)) return 'ADMIN';
  return role;
};

const isManagerRole = (role) => ['ADMIN', 'MANAGER', 'SHIFT_MANAGER', 'DEPARTMENT_MANAGER'].includes(role);

const isSameHospital = (user, encounter) => {
  const userHospital = idString(user?.hospitalId);
  const encounterHospital = idString(encounter?.hospitalId);
  return Boolean(userHospital && encounterHospital && userHospital === encounterHospital);
};

const assignedIds = (encounter) => [
  encounter?.nurse_id,
  encounter?.assignedNurseId,
  encounter?.assigned_physician?.id,
  encounter?.assignedPhysicianId,
  encounter?.triage?.assigned_physician?.id,
  encounter?.triage?.assigned_nurse?.id,
  encounter?.triage?.triage_nurse_id,
  encounter?.original_triage_nurse,
  ...(encounter?.assigned_clinicians || []).map((item) => item?.id),
  ...(encounter?.manually_selected_clinicians || []).map((item) => item?.id),
  ...(encounter?.clinical_handovers || []).map((item) => item?.clinician_id)
].map(idString).filter(Boolean);

const isAssignedToEncounter = (user, encounter) => {
  const userId = idString(user?._id || user?.id);
  return Boolean(userId && assignedIds(encounter).includes(userId));
};

export const getReportAccessDecision = ({ user, encounter, reportType, action = 'read' }) => {
  const type = String(reportType || '').toUpperCase();
  const role = roleString(user);
  const policy = ROLE_POLICIES[role]?.[type];

  if (!user || !encounter) return { allowed: false, reason: 'missing_context', visibility: ACCESS.NONE };
  if (!isSameHospital(user, encounter)) return { allowed: false, reason: 'hospital_mismatch', visibility: ACCESS.NONE };
  if (!policy || policy.read === ACCESS.NONE) return { allowed: false, reason: 'role_report_denied', visibility: ACCESS.NONE };
  if (action !== 'read' && !policy.edit) return { allowed: false, reason: 'edit_denied', visibility: policy.read };
  if (!isManagerRole(role) && !isAssignedToEncounter(user, encounter)) {
    return { allowed: false, reason: 'encounter_assignment_required', visibility: policy.read };
  }

  return { allowed: true, reason: 'allowed', visibility: policy.read, role, reportType: type, action };
};

export const canAccessReport = (context) => getReportAccessDecision(context).allowed;
export const getReportVisibility = (context) => getReportAccessDecision(context).visibility;
export { ACCESS, REPORT_TYPES, ROLE_POLICIES, isAssignedToEncounter, isSameHospital };
