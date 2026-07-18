/**
 * Relationship-Based Access Control (ReBAC) for Circle of Care
 * 
 * Implements professional hospital access patterns:
 * - Staff in same department see all patients (shared patient pool)
 * - Staff can request break-glass access with audit logging
 * - Field-level security: Nurses see care plans, Physicians see orders
 */

import PatientEHRRecord from '../models/patientEHRRecord.model.js';
import { AuditLog } from '../models/auditLog.model.js';

/**
 * Get patients accessible to a staff member
 * Circle of Care: All staff in a department can see all patients in that department
 * 
 * @param {ObjectId} staffId - The staff member's ID
 * @param {Object} staff - The staff document with assignment and systemRole
 * @returns {Array} Patients accessible to this staff
 */
export const getAccessiblePatients = async (staffId, staff) => {
  if (!staff) {
    console.warn('[ReBAC] No staff document provided');
    return [];
  }

  try {
    // Circle of Care Query: Department-based access
    const patients = await PatientEHRRecord.find({
      hospitalId: staff.hospitalId,
      currentDepartment: mapDepartmentToClinical(staff.assignment?.department),
      status: { $in: ['ADMITTED', 'IN_PROGRESS'] },
    });

    console.log(
      `[ReBAC] Staff ${staffId} in ${staff.assignment?.department} can access ${patients.length} patients`
    );

    return patients;
  } catch (err) {
    console.error('[ReBAC] Error fetching accessible patients:', err.message);
    return [];
  }
};

/**
 * Check if staff can access a specific patient
 * 
 * @param {ObjectId} staffId - Staff member ID
 * @param {Object} staff - Staff document
 * @param {ObjectId} patientId - Patient ID
 * @param {Object} patient - Patient document
 * @returns {Object} { canAccess, reason, circleOfCare }
 */
export const checkPatientAccess = async (staffId, staff, patientId, patient) => {
  if (!staff || !patient) {
    return { canAccess: false, reason: 'Missing staff or patient', circleOfCare: null };
  }

  // Same department = immediate access
  if (mapDepartmentToClinical(staff.assignment?.department) === patient.currentDepartment) {
    return {
      canAccess: true,
      reason: 'Same department (Circle of Care)',
      circleOfCare: 'SAME_DEPARTMENT',
    };
  }

  // Different department = check break-glass history
  const hasBreakGlass = staff.breakGlassAccess?.some(
    (bg) => bg.patientId?.toString() === patientId.toString()
  );

  if (hasBreakGlass) {
    return {
      canAccess: true,
      reason: 'Break-glass access approved',
      circleOfCare: 'BREAK_GLASS',
    };
  }

  // No access
  return {
    canAccess: false,
    reason: 'Patient in different department - break-glass access required',
    circleOfCare: null,
  };
};

/**
 * Request break-glass access (outside department access)
 * Creates audit log entry
 * 
 * @param {ObjectId} staffId - Staff member requesting access
 * @param {ObjectId} patientId - Patient being accessed
 * @param {string} reason - Reason for access (e.g., "Consultation", "Emergency")
 * @param {ObjectId} hospitalId - Hospital context
 * @returns {Object} { success, auditLogId, message }
 */
export const requestBreakGlassAccess = async (staffId, patientId, reason, hospitalId) => {
  try {
    // Create audit log entry
    const audit = await AuditLog.create({
      action: 'BREAK_GLASS_ACCESS',
      actorId: staffId,
      resourceId: patientId,
      hospitalId,
      reason,
      authorized: true, // Assuming auto-approval in professional context
      circleOfCare: 'BREAK_GLASS',
      description: `Staff ${staffId} accessed patient ${patientId} outside their department: ${reason}`,
    });

    console.log(`[ReBAC] Break-glass access granted: ${audit._id}`);

    return {
      success: true,
      auditLogId: audit._id,
      message: 'Break-glass access recorded and logged',
    };
  } catch (err) {
    console.error('[ReBAC] Error creating break-glass audit:', err.message);
    return {
      success: false,
      message: 'Failed to log break-glass access',
      error: err.message,
    };
  }
};

/**
 * Log a patient access event (for audit trail)
 * 
 * @param {ObjectId} staffId - Staff member
 * @param {ObjectId} patientId - Patient
 * @param {string} action - Action (VIEW, EDIT, etc.)
 * @param {ObjectId} hospitalId - Hospital context
 * @param {string} circleOfCare - Type of access
 */
export const auditPatientAccess = async (staffId, patientId, action, hospitalId, circleOfCare) => {
  try {
    await AuditLog.create({
      action,
      actorId: staffId,
      resourceId: patientId,
      hospitalId,
      circleOfCare,
      timestamp: new Date(),
    });
  } catch (err) {
    console.error('[ReBAC] Error logging access:', err.message);
  }
};

/**
 * Get role-based field projection
 * Physicians see diagnostic orders; Nurses see care plans
 * 
 * @param {string} systemRole - PHYSICIAN, NURSE, TECHNICIAN, etc.
 * @returns {Object} MongoDB projection
 */
export const getFieldProjection = (systemRole) => {
  const baseFields = {
    name: 1,
    surname: 1,
    age: 1,
    gender: 1,
    mrn: 1,
    currentBedId: 1,
    currentDepartment: 1,
    esiLevel: 1,
    acuityScore: 1,
    latestVitals: 1,
    admissionTime: 1,
  };

  if (systemRole === 'PHYSICIAN') {
    return {
      ...baseFields,
      // Physicians see orders and diagnostic results
      activeOrders: 1,
      consultingSpecialties: 1,
      status: 1,
    };
  }

  if (systemRole === 'NURSE') {
    return {
      ...baseFields,
      // Nurses see care plans and medications
      surveillanceFlags: 1,
      vitalThresholds: 1,
      // Don't expose diagnostic details
      activeOrders: 0,
      consultingSpecialties: 0,
    };
  }

  // Default: minimum fields
  return baseFields;
};

/**
 * Utility: Map department enum to clinical department
 * 
 * @param {string} dept - Department enum (EMERGENCY, ICU, etc.)
 * @returns {string} Clinical department for patient query
 */
function mapDepartmentToClinical(dept) {
  const map = {
    EMERGENCY: 'ER',
    CARDIOLOGY: 'CARDIOLOGY',
    ICU: 'ICU',
    PEDIATRICS: 'PEDIATRICS',
    RADIOLOGY: 'RADIOLOGY',
    GENERAL_WARD: 'WARD',
  };
  return map[dept] || dept;
}

export default {
  getAccessiblePatients,
  checkPatientAccess,
  requestBreakGlassAccess,
  auditPatientAccess,
  getFieldProjection,
};
