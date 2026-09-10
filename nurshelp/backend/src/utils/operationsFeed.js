const severityRank = {
  CRITICAL: 5,
  HIGH: 4,
  MEDIUM: 3,
  LOW: 2,
  INFO: 1
};

export const sortOperationsEvents = (events = []) => [...events].sort((left, right) => {
  const severityDifference = (severityRank[right.severity] || 0) - (severityRank[left.severity] || 0);
  if (severityDifference !== 0) return severityDifference;

  const waitDifference = (right.priority?.waitMs || 0) - (left.priority?.waitMs || 0);
  if (waitDifference !== 0) return waitDifference;

  const createdDifference = new Date(right.createdAt).getTime() - new Date(left.createdAt).getTime();
  if (createdDifference !== 0) return createdDifference;

  return String(left.id).localeCompare(String(right.id));
});

const asId = (value) => value === undefined || value === null ? null : String(value);

const textMatchesStaff = (value, staff) => {
  if (!value || !staff) return false;
  const haystack = String(value).toLowerCase();
  return [staff.name, staff.email].filter(Boolean).some((candidate) => haystack.includes(String(candidate).toLowerCase()));
};

export const canViewSurgery = (encounter, surgery, user) => {
  const role = String(user?.role || '').toUpperCase();
  if (role === 'ADMIN' || role === 'MANAGER') return true;

  const clinicianId = asId(user?._id || user?.id);
  const assignedIds = [
    ...(encounter?.assigned_clinicians || []).map((clinician) => clinician?.id || clinician?._id),
    encounter?.assignedPhysicianId,
    encounter?.assignedNurseId,
    encounter?.nurse_id
  ].map(asId).filter(Boolean);

  if (clinicianId && assignedIds.includes(clinicianId)) return true;
  if (textMatchesStaff(surgery?.surgeon, user)) return true;
  return (surgery?.surgical_team || []).some((member) => textMatchesStaff(member, user));
};
