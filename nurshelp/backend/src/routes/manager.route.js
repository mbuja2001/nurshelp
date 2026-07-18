/**
 * Manager Routes
 * PHASE 2 & 3: Staff management, shift scheduling, and approval workflows
 */

import express from "express";
import {
  getStaffStatus,
  getPendingApprovals,
  bulkApproveApprovals
} from "../controllers/manager.controller.js";
import { protect } from "../middleware/authmiddleware.js";
import { authorize, restrictByDepartment, auditLog } from "../middleware/rbac.js";

const router = express.Router();

// ============================================
// PHASE 2: STAFF MANAGEMENT
// ============================================

/**
 * PHASE 2.1: Register new staff member (nurse or physician)
 * POST /api/manager/staff/register
 * Protected: MANAGER, ADMIN only
 * 
 * Body: {
 *   "name": "Sarah",
 *   "surname": "Johnson",
 *   "email": "sarah@hospital.com",
 *   "password": "nurse123",
 *   "role": "NURSE",
 *   "department": "ED",
 *   "specialty": "Emergency Medicine"
 * }
 */
router.post(
  "/staff/register",
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('STAFF_REGISTERED'),
  async (req, res) => {
    try {
      // Accept both old and new field names
      const {
        name,
        surname,
        firstName,
        lastName,
        email,
        password,
        role,
        department,
        specialty,
        clinicalSpecialtyId,
        title,
        licenseNumber,
        npiNumber,
        isOnCall
      } = req.body;

      console.log('\n[STAFF REGISTER] Request received');
      console.log('  Name:', name, surname);
      console.log('  Email:', email);
      console.log('  Role:', role);
      console.log('  Department:', department);
      console.log('  req.hospitalId:', req.hospitalId);
      console.log('  req.user.hospitalId:', req.user?.hospitalId);

      const fName = firstName || name;
      const lName = lastName || surname;

      if (!fName || !lName || !email || !password || !role) {
        return res.status(400).json({
          success: false,
          message: 'Required fields: firstName/lastName, email, password, role'
        });
      }

      // CRITICAL: Validate hospitalId exists and convert to ObjectId if needed
      if (!req.hospitalId) {
        console.error('[STAFF REGISTER] CRITICAL ERROR: req.hospitalId is missing!');
        console.error('  Token hospitalId:', req.user?.hospitalId);
        console.error('  Request user:', req.user?._id);
        return res.status(400).json({
          success: false,
          message: 'Hospital context not found. Please ensure you are logged in as a hospital manager.'
        });
      }

      // Convert hospitalId to ObjectId if it's a string
      let hospitalIdForQuery = req.hospitalId;
      if (typeof req.hospitalId === 'string') {
        try {
          const { ObjectId } = await import('mongodb');
          hospitalIdForQuery = new ObjectId(req.hospitalId);
          console.log('[STAFF REGISTER] Converted string hospitalId to ObjectId:', hospitalIdForQuery.toString());
        } catch (conversionError) {
          console.error('[STAFF REGISTER] Failed to convert hospitalId:', conversionError.message);
          return res.status(400).json({
            success: false,
            message: 'Invalid hospital context format.'
          });
        }
      }

      const { ClinicalStaff } = await import("../models/clinicalstaff.model.js");
      const bcrypt = (await import("bcrypt")).default;

      console.log('[STAFF REGISTER] Validation passed');
      console.log('  hospitalId:', hospitalIdForQuery.toString?.() || hospitalIdForQuery);
      console.log('  email:', email);

      // Check for existing email within THIS hospital only (multi-tenant: allow same email in different hospitals)
      const existing = await ClinicalStaff.findOne({ 
        email,
        hospitalId: hospitalIdForQuery
      });
      if (existing) {
        console.warn('[STAFF REGISTER] Email already exists for this hospital:', email);
        return res.status(400).json({
          success: false,
          message: 'Email already registered in this hospital'
        });
      }

      console.log('[STAFF REGISTER] Email is unique for this hospital');

      // Validate and normalize role and department to avoid incorrect registrations
      const allowedRoles = ['PHYSICIAN', 'NURSE', 'ADMINISTRATIVE', 'TECHNICIAN', 'SUPERUSER', 'SPECIALIST', 'PHARMACIST', 'RADIOLOGIST', 'PATHOLOGIST'];
      // Handle both uppercase and mixed-case role inputs (e.g., "Physician" or "Nurse")
      let normalizedRole = (role || '').toString().toUpperCase().trim();
      // Map legacy/variant roles and specialty roles to base system roles
      const roleMap = {
        'SPECIALIST': 'PHYSICIAN',
        'DOCTOR': 'PHYSICIAN',
        'CLINICIAN': 'PHYSICIAN',
        'PHYSICIAN': 'PHYSICIAN',
        'NURSE': 'NURSE',
        'PHARMACIST': 'PHARMACIST',
        'PHARMACY TECHNICIAN': 'PHARMACIST',
        'PHARMACY TECHNICIAN LEAD': 'PHARMACIST',
        'RADIOLOGIST': 'RADIOLOGIST',
        'RADIOLOGY TECHNICIAN': 'RADIOLOGIST',
        'RADIOLOGY TECHNICIAN LEAD': 'RADIOLOGIST',
        'PATHOLOGIST': 'PATHOLOGIST',
        'PATHOLOGY TECHNICIAN': 'PATHOLOGIST',
        'PATHOLOGY TECHNICIAN LEAD': 'PATHOLOGIST',
        'DEPARTMENT HEAD': 'ADMINISTRATIVE',
        'DEPARTMENT ASSISTANT': 'ADMINISTRATIVE',
        'OTHER': 'SUPERUSER'
      };
      if (roleMap[normalizedRole]) {
        normalizedRole = roleMap[normalizedRole];
      }
      if (!allowedRoles.includes(normalizedRole)) {
        return res.status(400).json({ success: false, message: `Invalid role "${role}". Allowed roles: ${allowedRoles.join(', ')}` });
      }

      // Map legacy department names to new assignment.department enums
      const deptMap = {
        'ED': 'EMERGENCY',
        'EMERGENCY': 'EMERGENCY',
        'CARDIOLOGY': 'CARDIOLOGY',
        'ICU': 'ICU',
        'OR': 'GENERAL_WARD',
        'OB': 'GENERAL_WARD',
        'MEDICAL-SURGICAL': 'GENERAL_WARD',
        'MEDICAL_SURGICAL': 'GENERAL_WARD',
        'PEDIATRICS': 'PEDIATRICS',
        'RADIOLOGY': 'RADIOLOGY',
        'GENERAL': 'GENERAL_WARD'
      };

      const rawDept = (department || req.departmentFilter || 'GENERAL').toString();
      const normalizedDept = (deptMap[rawDept.toUpperCase()] || 'GENERAL_WARD');

      // Hash password (pre-save hook was removed to prevent issues with dual model exports)
      const hashedPassword = await bcrypt.hash(password, 10);

      const { Specialty } = await import('../models/specialty.model.js');

      // Resolve specialty by id, name or code
      let specialtyDoc = null;
      if (clinicalSpecialtyId) {
        specialtyDoc = await Specialty.findById(clinicalSpecialtyId).lean();
      }
      if (!specialtyDoc && specialty) {
        specialtyDoc = await Specialty.findOne({ $or: [{ name: specialty }, { code: specialty }] }).lean();
      }

      // Map normalized roles to valid credentials.title enum values
      const roleCredentialsMap = {
        'PHYSICIAN': 'MD',
        'NURSE': 'RN',
        'TECHNICIAN': 'NP',        // General tech roles
        'PHARMACIST': 'PA',
        'RADIOLOGIST': 'MD',
        'PATHOLOGIST': 'MD',
        'ADMINISTRATIVE': 'PA',    // Department Head, Assistant
        'SUPERUSER': 'PhD'         // Super admin roles
      };
      const credTitle = title || roleCredentialsMap[normalizedRole] || 'NP';
      const credLicense = licenseNumber || `LIC-${Date.now()}`;

      const newStaffData = {
        hospitalId: hospitalIdForQuery,
        firstName: fName,
        lastName: lName,
        email,
        password: hashedPassword,
        systemRole: normalizedRole,
        clinicalSpecialtyId: specialtyDoc?._id || null,
        clinicalSpecialty: specialtyDoc?.name || (specialty || 'General Practice'),
        credentials: {
          title: credTitle,
          licenseNumber: credLicense,
          npiNumber: npiNumber || null
        },
        assignment: {
          department: normalizedDept,
          isOnCall: !!isOnCall
        },
        status: 'ACTIVE',
        metadata: { createdBy: req.user._id }
      };

      const newStaff = new ClinicalStaff(newStaffData);

      console.log('[STAFF REGISTER] Staff document prepared');
      console.log('  hospitalId (pre-save):', newStaff.hospitalId?.toString?.() || newStaff.hospitalId);
      console.log('  hospitalId (type):', typeof newStaff.hospitalId);
      console.log('  firstName:', newStaff.firstName);
      console.log('  lastName:', newStaff.lastName);
      console.log('  email:', newStaff.email);
      console.log('  systemRole:', newStaff.systemRole);

      await newStaff.save();

      console.log('[STAFF REGISTER] Staff saved successfully to database!');
      console.log('  Staff._id:', newStaff._id?.toString?.() || newStaff._id);
      console.log('  Staff.hospitalId (post-save): ', newStaff.hospitalId?.toString?.() || newStaff.hospitalId);
      console.log('  Staff.email (post-save):', newStaff.email);
      console.log('  Staff hospitalId in DB:', newStaff.hospitalId);
      console.log('  Staff status in DB:', newStaff.status);
      console.log('  Staff email in DB:', newStaff.email);

      // Log in audit
      req.auditInfo.resourceId = newStaff._id;
      req.auditInfo.details = { email, role: normalizedRole, department: normalizedDept };

      res.status(201).json({
        success: true,
        message: `${normalizedRole} registered successfully`,
        staff: {
          id: newStaff._id,
          firstName: newStaff.firstName,
          lastName: newStaff.lastName,
          email: newStaff.email,
          systemRole: newStaff.systemRole,
          clinicalSpecialty: newStaff.clinicalSpecialty,
          clinicalSpecialtyId: newStaff.clinicalSpecialtyId,
          assignment: newStaff.assignment,
          hospitalId: newStaff.hospitalId
        }
      });
    } catch (err) {
      console.error('Register staff error:', err);
      res.status(500).json({
        success: false,
        message: 'Failed to register staff',
        error: err.message
      });
    }
  }
);

/**
 * PHASE 2.2: Get all staff members
 * GET /api/manager/staff/list
 * Protected: MANAGER, ADMIN only
 * 
 * Query params:
 * - department: filter by department (managers see their dept only)
 * - role: filter by role (NURSE, PHYSICIAN)
 * - isActive: true/false
 * - search: search by name/email
 */
router.get(
  "/staff/list",
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  async (req, res) => {
    try {
      const { role, isActive, search } = req.query;
      const { ClinicalStaff } = await import("../models/clinicalstaff.model.js");

      // DIAGNOSTIC LOGGING
      console.log('\n[STAFF LIST] Request received');
      console.log('  req.hospitalId:', req.hospitalId);
      console.log('  req.user._id:', req.user._id);
      console.log('  req.user.role:', req.user.role);
      console.log('  req.departmentFilter:', req.departmentFilter);

      // Convert hospitalId to ObjectId if it's a string
      let hospitalIdForQuery = req.hospitalId;
      if (typeof req.hospitalId === 'string') {
        try {
          const { ObjectId } = await import('mongodb');
          hospitalIdForQuery = new ObjectId(req.hospitalId);
          console.log('[STAFF LIST] Converted string hospitalId to ObjectId:', hospitalIdForQuery.toString());
        } catch (conversionError) {
          console.error('[STAFF LIST] Failed to convert hospitalId:', conversionError.message);
          return res.status(400).json({
            success: false,
            message: 'Invalid hospital context format.'
          });
        }
      }

      const query = {
        hospitalId: hospitalIdForQuery // Filter by hospital
      };

      console.log('[STAFF LIST] Query execution');
      console.log('  hospitalId:', hospitalIdForQuery?.toString?.() || hospitalIdForQuery);
      console.log('  hospitalId (type):', typeof hospitalIdForQuery);
      console.log('  Initial query:', JSON.stringify(query));

      // Build $or conditions for department and search filters (don't overwrite)
      const orConditions = [];

      // Department filtering - handle both old (department) and new (assignment.department) schemas
      if (req.departmentFilter && req.departmentFilter !== 'All') {
        orConditions.push({ department: req.departmentFilter });
        orConditions.push({ 'assignment.department': req.departmentFilter });
      }

      // Search filtering - firstName, lastName, or email
      if (search) {
        orConditions.push({ firstName: { $regex: search, $options: 'i' } });
        orConditions.push({ lastName: { $regex: search, $options: 'i' } });
        orConditions.push({ email: { $regex: search, $options: 'i' } });
      }

      // Only add $or if we have conditions
      if (orConditions.length > 0) {
        query.$or = orConditions;
        console.log('[STAFF LIST] Added OR conditions:', JSON.stringify(orConditions));
      }

      if (role && role !== 'All') {
        query.systemRole = role.toUpperCase();
        console.log('[STAFF LIST] Added role filter:', role.toUpperCase());
      }

      if (isActive !== undefined) {
        if (isActive === 'true') query.status = 'ACTIVE';
        else query.status = { $ne: 'ACTIVE' };
        console.log('[STAFF LIST] Added status filter:', isActive);
      }

      console.log('[STAFF LIST] Final query:', JSON.stringify(query));

      const staff = await ClinicalStaff.find(query).select('-password').sort({ firstName: 1 });

      console.log('[STAFF LIST] Query Results:');
      console.log('  Total found:', staff.length);
      if (staff.length === 0) {
        console.warn('  ⚠ NO STAFF FOUND for this hospital!');
        console.warn('  This could mean:');
        console.warn('    1. No staff registered yet');
        console.warn('    2. Staff registered with different hospitalId');
        console.warn('    3. Staff registered with null/undefined hospitalId');
      } else {
        console.log('  First staff member:', {
          _id: staff[0]._id?.toString?.() || staff[0]._id,
          name: staff[0].firstName + ' ' + staff[0].lastName,
          email: staff[0].email,
          hospitalId: staff[0].hospitalId?.toString?.() || staff[0].hospitalId,
          systemRole: staff[0].systemRole,
          status: staff[0].status
        });
      }

      res.status(200).json({
        success: true,
        count: staff.length,
        staff
      });
    } catch (err) {
      console.error('Get staff error:', err);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch staff',
        error: err.message
      });
    }
  }
);

/**
 * PHASE 2.3: Update staff member
 * PUT /api/manager/staff/:staffID
 * Protected: MANAGER, ADMIN only
 */
router.put(
  "/staff/:staffID",
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('STAFF_UPDATED'),
  async (req, res) => {
    try {
      const { staffID } = req.params;
      const updateData = req.body;

      // Prevent password updates through this endpoint
      delete updateData.password;

      const { ClinicalStaff } = await import("../models/clinicalstaff.model.js");

      const staff = await ClinicalStaff.findByIdAndUpdate(staffID, updateData, { returnDocument: 'after' }).select('-password');

      if (!staff) {
        return res.status(404).json({
          success: false,
          message: 'Staff member not found'
        });
      }

      res.status(200).json({
        success: true,
        message: 'Staff member updated successfully',
        staff
      });
    } catch (err) {
      console.error('Update staff error:', err);
      res.status(500).json({
        success: false,
        message: 'Failed to update staff',
        error: err.message
      });
    }
  }
);

/**
 * PHASE 2.4: Deactivate staff member
 * DELETE /api/manager/staff/:staffID
 * Protected: MANAGER, ADMIN only
 */
router.delete(
  "/staff/:staffID",
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('STAFF_DEACTIVATED'),
  async (req, res) => {
    try {
      const { staffID } = req.params;
      const { ClinicalStaff } = await import("../models/clinicalstaff.model.js");

      const staff = await ClinicalStaff.findByIdAndUpdate(
        staffID,
        { status: 'TERMINATED' },
        { returnDocument: 'after' }
      ).select('-password');

      if (!staff) {
        return res.status(404).json({
          success: false,
          message: 'Staff member not found'
        });
      }

      res.status(200).json({
        success: true,
        message: 'Staff member deactivated',
        staff
      });
    } catch (err) {
      console.error('Deactivate staff error:', err);
      res.status(500).json({
        success: false,
        message: 'Failed to deactivate staff',
        error: err.message
      });
    }
  }
);

// ============================================
// PHASE 3: SHIFT MANAGEMENT & APPROVALS
// ============================================

/**
 * Get staff status (dashboard overview)
 * GET /api/manager/staff-status
 * Protected: MANAGER, ADMIN only
 */
router.get(
  "/staff-status",
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  getStaffStatus
);

/**
 * Get pending approvals (break requests, late arrivals)
 * GET /api/manager/pending-approvals
 * Protected: MANAGER, ADMIN only
 */
router.get(
  "/pending-approvals",
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  getPendingApprovals
);

/**
 * Bulk approve/reject requests
 * POST /api/manager/bulk-approvals
 * Protected: MANAGER, ADMIN only
 * 
 * Body: {
 *   "approvals": [
 *     { "recordId": "...", "type": "LATE_ARRIVAL", "approved": true, "reason": "..." },
 *     { "recordId": "...", "type": "BREAK_REQUEST", "approved": false, "reason": "..." }
 *   ]
 * }
 */
router.post(
  "/bulk-approvals",
  protect,
  authorize('MANAGER', 'ADMIN'),
  restrictByDepartment(),
  auditLog('APPROVALS_PROCESSED'),
  bulkApproveApprovals
);

export default router;
