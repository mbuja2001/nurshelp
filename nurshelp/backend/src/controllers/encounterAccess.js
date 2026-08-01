export const canAccessEncounter = (a, b) => {
  // Support either signature: (encounter, user) OR (user, encounter)
  let encounter = a;
  let user = b;
  if (a && a.role && b && (b.patient || b.surgery_records || b.assigned_physician)) {
    // caller passed (user, encounter)
    user = a;
    encounter = b;
  }
  if (!encounter || !user?._id) return false;

  if (encounter.nurse_id?.toString() === user._id.toString()) {
    return true;
  }

  if (encounter.assignedPhysicianId?.toString() === user._id.toString()) {
    return true;
  }

  if (encounter.assigned_physician?.id?.toString() === user._id.toString()) {
    return true;
  }

  if (encounter.triage?.assigned_physician?.id?.toString() === user._id.toString()) {
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
