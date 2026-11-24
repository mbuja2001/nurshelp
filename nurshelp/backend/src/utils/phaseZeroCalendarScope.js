const PHASE_ZERO_ROLES = new Set(['PHYSICIAN', 'DOCTOR', 'SURGEON']);

export const PHASE_ZERO_DISABLED_CATEGORIES = [
  'meeting',
  'research',
  'lab',
  'med',
  'diagnosed',
  'discharged',
  'administered',
  'handoff',
  'radiology',
  'pathology'
];

export const identityValue = (value) => {
  if (value && typeof value === 'object') {
    if (typeof value.toHexString === 'function') return value;
    return value._id || value.id || null;
  }
  return value || null;
};

export const hasPhaseZeroClinicalRole = (role) => PHASE_ZERO_ROLES.has(String(role || '').toUpperCase());

export const isAssignedSurgeon = (surgery, userId) => (
  Boolean(surgery?.surgeon_id)
  && String(identityValue(surgery.surgeon_id)) === String(identityValue(userId))
);
