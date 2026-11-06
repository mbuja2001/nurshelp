import { normalizeRoleLabel, resolveTechnicianSpecialty } from './jwt';

export const resolveShiftDashboardContext = ({ userId, role, specialty, shift }) => {
  const userKey = String(userId || '');
  const roster = (shift?.assignedStaff || []).find((assignment) => String(assignment.staffID?._id || assignment.staffID) === userKey);
  const assignmentRole = String(roster?.assignmentRole || '').toUpperCase();
  const normalizedRole = normalizeRoleLabel(role);

  if (roster && normalizedRole === 'TECHNICIAN') {
    if (assignmentRole !== 'SUPPORT') return 'TECHNICIAN_UNCONFIGURED';
    const technicianSpecialty = resolveTechnicianSpecialty(role, specialty);
    return technicianSpecialty ? `TECHNICIAN_${technicianSpecialty}` : 'TECHNICIAN_UNCONFIGURED';
  }

  if (assignmentRole === 'TRIAGE_NURSE') return 'TRIAGE_NURSE';
  if (assignmentRole === 'CLINICAL_NURSE') return 'CLINICAL_NURSE_ER';
  if (assignmentRole === 'PHYSICIAN') return 'PHYSICIAN_ER';
  if (assignmentRole === 'PHARMACIST') return 'PHARMACY';
  if (assignmentRole === 'RADIOLOGIST') return 'RADIOLOGY';
  if (assignmentRole === 'PATHOLOGIST') return 'PATHOLOGY';

  if (assignmentRole === 'SPECIALIST') {
    if (normalizedRole === 'PHARMACIST') return 'PHARMACY';
    if (normalizedRole === 'RADIOLOGIST') return 'RADIOLOGY';
    if (normalizedRole === 'PATHOLOGIST') return 'PATHOLOGY';
  }
  if (roster) return 'NO_SHIFT';

  const team = shift?.teamAssignment || {};
  if (team.triageNurseId && String(team.triageNurseId._id || team.triageNurseId) === userKey) return 'TRIAGE_NURSE';
  if (team.clinicalNurseId && String(team.clinicalNurseId._id || team.clinicalNurseId) === userKey) return team.clinicalNurseContext || 'CLINICAL_NURSE_ER';
  if (team.physicianId && String(team.physicianId._id || team.physicianId) === userKey) return team.physicianContext || 'PHYSICIAN_ER';

  if (!roster && normalizedRole === 'PHARMACIST') return 'PHARMACY';
  if (!roster && normalizedRole === 'RADIOLOGIST') return 'RADIOLOGY';
  if (!roster && normalizedRole === 'PATHOLOGIST') return 'PATHOLOGY';
  return 'NO_SHIFT';
};

export const shouldShowClockInBoard = ({ context, hasShiftToday, isClockedIn }) => (
  context === 'NO_SHIFT' || !hasShiftToday || !isClockedIn
);
