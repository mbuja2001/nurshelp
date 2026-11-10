import mongoose from 'mongoose';
import { getAllowedShiftDuties } from '../config/hospitalStaffRoleCatalog.js';

const ROLE_CONTEXTS = {
  TRIAGE_NURSE: { role: 'NURSE', context: 'CLINICAL_NURSE_ER' },
  CLINICAL_NURSE: { role: 'NURSE', context: 'CLINICAL_NURSE_ER' },
  PHYSICIAN: { role: 'PHYSICIAN', context: 'PHYSICIAN_ER' },
};

export const normalizeShiftAssignmentRole = (role, systemRole, positionCode) => {
  const normalizedRole = String(role || '').toUpperCase().trim().replace(/[ -]+/g, '_');
  const allowedDuties = getAllowedShiftDuties(positionCode, systemRole);
  const normalizedSystemRole = String(systemRole || '').toUpperCase().trim().replace(/[ -]+/g, '_');
  if (!normalizedRole || !allowedDuties.includes(normalizedRole)) {
    if (!normalizedRole) throw new Error('An explicit valid shift assignment role is required');
    throw new Error(`Assignment role ${normalizedRole} is incompatible with staff role ${normalizedSystemRole || 'UNKNOWN'}`);
  }
  return normalizedRole;
};

export const getClinicalTeamRoleField = (assignmentRole) => ({
  TRIAGE_NURSE: 'triageNurseId',
  CLINICAL_NURSE: 'clinicalNurseId',
  PHYSICIAN: 'physicianId',
})[assignmentRole] || null;

export const normalizeRoleAssignment = (value) => {
  if (typeof value !== 'string' || !ROLE_CONTEXTS[value]) {
    throw new Error('Unsupported clinical role');
  }

  return ROLE_CONTEXTS[value];
};

export const validateClinicalTeam = ({
  shiftId,
  triageNurseId,
  clinicalNurseId,
  physicianId,
  staffById,
  hospitalId,
}) => {
  if (!mongoose.isValidObjectId(shiftId)) {
    throw new Error('Invalid shiftId');
  }

  const assignments = [
    ['triageNurseId', triageNurseId, 'NURSE'],
    ['clinicalNurseId', clinicalNurseId, 'NURSE'],
    ['physicianId', physicianId, 'PHYSICIAN'],
  ];

  const normalized = {};
  for (const [field, id, expectedRole] of assignments) {
    if (!id) continue;
    if (!mongoose.isValidObjectId(id)) {
      throw new Error(`${field} is not a valid staff ID`);
    }

    const staff = staffById.get(id.toString());
    if (!staff) {
      throw new Error(`${field} references a staff member that does not exist`);
    }
    if (staff.hospitalId?.toString() !== hospitalId.toString()) {
      throw new Error(`${field} references a staff member outside this hospital`);
    }
    if (staff.status !== 'ACTIVE') {
      throw new Error(`${field} references an inactive staff member`);
    }
    if (staff.systemRole !== expectedRole) {
      throw new Error(`${field} references a staff member with an incompatible role`);
    }

    normalized[field] = new mongoose.Types.ObjectId(id);
  }

  return normalized;
};

export const buildAssignedStaff = (currentAssigned, staffById, assignments) => {
  const assignedById = new Map();

  for (const entry of currentAssigned || []) {
    const staffId = entry.staffID?.toString?.() || entry.staffID?._id?.toString?.() || entry.staffID?.toString();
    if (!staffId || assignedById.has(staffId)) continue;

    const staff = staffById.get(staffId);
    assignedById.set(staffId, staff ? {
      ...entry,
      staffID: staff._id,
      type: staff.systemRole,
      name: `${staff.firstName || staff.name || ''} ${staff.lastName || staff.surname || ''}`.trim(),
      surname: staff.lastName || staff.surname || '',
      specialty: staff.clinicalSpecialty || staff.specialty || entry.specialty || '',
      status: entry.status || 'ASSIGNED',
    } : { ...entry });
  }

  for (const [field, id] of Object.entries(assignments)) {
    if (!id || assignedById.has(id.toString())) continue;
    const staff = staffById.get(id.toString());
    assignedById.set(id.toString(), {
      staffID: id,
      type: staff.systemRole,
      name: `${staff.firstName || staff.name || ''} ${staff.lastName || staff.surname || ''}`.trim(),
      surname: staff.lastName || staff.surname || '',
      specialty: staff.clinicalSpecialty || staff.specialty || '',
      status: 'ASSIGNED',
    });
  }

  return Array.from(assignedById.values());
};
