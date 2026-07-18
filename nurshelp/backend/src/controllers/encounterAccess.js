export const canAccessEncounter = (encounter, user) => {
  if (!encounter || !user?._id) return false;

  if (encounter.nurse_id?.toString() === user._id.toString()) {
    return true;
  }

  if (encounter.assignedPhysicianId?.toString() === user._id.toString()) {
    return true;
  }

  const normalizedRole = user.role?.toUpperCase();
  const isPhysicianRole = ["PHYSICIAN", "DOCTOR", "SPECIALIST"].includes(normalizedRole);
  const activeClinicalStatuses = ["HOSPITALIZED", "DIAGNOSED", "SEEN", "AWAITING_PHYSICIAN", "OFFICIAL_REPORT", "DISCHARGED"];

  if (isPhysicianRole && activeClinicalStatuses.includes(encounter.status)) {
    return true;
  }

  const assignedClinicianIds = (encounter.assigned_clinicians || [])
    .map((clinician) => clinician.id?.toString())
    .filter(Boolean);

  if (assignedClinicianIds.includes(user._id.toString())) {
    return true;
  }

  const handoverClinicianIds = (encounter.clinical_handovers || [])
    .map((handover) => handover.clinician_id?.toString())
    .filter(Boolean);

  return handoverClinicianIds.includes(user._id.toString());
};
