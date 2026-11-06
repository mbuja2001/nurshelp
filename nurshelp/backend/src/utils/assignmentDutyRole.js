import { Shift } from '../models/shift.model.js';

export const resolveShiftAssignmentRole = (shift, staffId) => {
  const staffKey = String(staffId || '');
  if (!staffKey) return null;

  const rosterAssignment = (shift?.assignedStaff || []).find((assignment) => {
    const assignedStaffId = assignment.staffID?._id || assignment.staffID;
    return String(assignedStaffId || '') === staffKey;
  });
  if (rosterAssignment?.assignmentRole) {
    return String(rosterAssignment.assignmentRole).toUpperCase();
  }

  const teamAssignment = shift?.teamAssignment || {};
  if (String(teamAssignment.triageNurseId?._id || teamAssignment.triageNurseId || '') === staffKey) return 'TRIAGE_NURSE';
  if (String(teamAssignment.clinicalNurseId?._id || teamAssignment.clinicalNurseId || '') === staffKey) return 'CLINICAL_NURSE';
  if (String(teamAssignment.physicianId?._id || teamAssignment.physicianId || '') === staffKey) return 'PHYSICIAN';
  return null;
};

export const getTodayShiftAssignmentRole = async ({ hospitalId, staffId }) => {
  if (!hospitalId || !staffId) return null;

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const staffKey = String(staffId);
  const shift = await Shift.findOne({
    hospitalId,
    date: { $gte: today, $lt: tomorrow },
    $or: [
      { 'assignedStaff.staffID': staffId },
      { 'teamAssignment.triageNurseId': staffId },
      { 'teamAssignment.clinicalNurseId': staffId },
      { 'teamAssignment.physicianId': staffId }
    ]
  })
    .sort({ 'teamAssignment.teamAssignedAt': -1, _id: -1 })
    .select('assignedStaff teamAssignment')
    .lean();

  return resolveShiftAssignmentRole(shift, staffKey);
};

export const isSelfAssignmentAllowed = (actorId, assigneeId, effectiveDutyRole) => {
  if (!actorId || !assigneeId || String(actorId) !== String(assigneeId)) return true;
  return String(effectiveDutyRole || '').toUpperCase() === 'CLINICAL_NURSE';
};
