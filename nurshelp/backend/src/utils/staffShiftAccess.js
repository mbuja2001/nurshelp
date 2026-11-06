const sameId = (left, right) => String(left?._id || left || '') === String(right?._id || right || '');

export const isStaffAssignedToShift = (shift, staffId, hospitalId) => {
  if (!shift || !staffId || !sameId(shift.hospitalId, hospitalId)) return false;

  const assignedInRoster = (shift.assignedStaff || []).some((assignment) =>
    sameId(assignment.staffID, staffId) && assignment.status !== 'ABSENT'
  );
  const team = shift.teamAssignment || {};
  const assignedToTeam = ['triageNurseId', 'clinicalNurseId', 'physicianId']
    .some((key) => sameId(team[key], staffId));

  return assignedInRoster || assignedToTeam;
};

export const isShiftApprovedForClockIn = (shift) => shift?.status === 'IN_PROGRESS';
