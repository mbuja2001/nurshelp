import { decodeJwt } from './jwt';
import { resolveShiftDashboardContext } from './shiftDashboardContext';

export const getEffectiveDutyRole = (context) => {
  if (context === 'TRIAGE_NURSE') return 'TRIAGE_NURSE';
  if (String(context || '').startsWith('CLINICAL_NURSE')) return 'CLINICAL_NURSE';
  if (String(context || '').startsWith('PHYSICIAN')) return 'PHYSICIAN';
  if (context === 'PHARMACY') return 'PHARMACIST';
  if (context === 'RADIOLOGY') return 'RADIOLOGIST';
  if (context === 'PATHOLOGY') return 'PATHOLOGIST';
  return null;
};

export const canAssignSelfAsDownstreamNurse = (actorId, assigneeId, effectiveDutyRole) => (
  !actorId
  || !assigneeId
  || String(actorId) !== String(assigneeId)
  || String(effectiveDutyRole || '').toUpperCase() === 'CLINICAL_NURSE'
);

export const fetchCurrentAssignmentIdentity = async (backendUrl, token) => {
  const userId = decodeJwt(token)?.id;
  if (!userId) throw new Error('Unable to identify the authenticated staff member.');

  const response = await fetch(`${backendUrl}/api/shifts/assigned/today`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: 'no-store'
  });
  if (!response.ok) throw new Error(`Unable to verify shift duty (${response.status}).`);

  const data = await response.json();
  return {
    userId: String(userId),
    dutyRole: getEffectiveDutyRole(resolveShiftDashboardContext({
      userId,
      role: decodeJwt(token)?.role,
      shift: data?.shift
    }))
  };
};
