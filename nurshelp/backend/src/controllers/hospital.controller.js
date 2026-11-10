/**
 * Hospital Controller
 * Handles hospital registration and initialization
 * Multi-tenant architecture - each hospital is independent
 */

import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { Hospital } from '../models/hospital.model.js';
import { Manager } from '../models/manager.model.js';
import { AuditLog } from '../models/auditLog.model.js';
import { HOSPITAL_DEPARTMENTS, HOSPITAL_SPECIALTIES } from '../config/hospitalStructureCatalog.js';

const JWT_SECRET = process.env.JWT_SECRET || 'supersecretkey';

/**
 * HOSPITAL REGISTRATION (Phase 1)
 * Create new hospital instance with admin/manager
 */
export const registerHospital = async (req, res) => {
  try {
    const { hospitalName, managerName, managerSurname, managerEmail, managerPassword, department, departments = [] } = req.body;

    // Validate required fields
    if (!hospitalName || !managerName || !managerSurname || !managerEmail || !managerPassword) {
      return res.status(400).json({
        success: false,
        message: 'Missing required fields: hospitalName, managerName, managerSurname, managerEmail, managerPassword'
      });
    }

    // Check if email already exists (globally, for security)
    const existingManager = await Manager.findOne({ email: managerEmail });
    if (existingManager) {
      return res.status(400).json({
        success: false,
        message: 'Email already registered'
      });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(managerPassword, 10);

    // Step 1: Create Hospital
    const hospitalID = `HOSP-${Date.now()}`;
    const supportedDepartments = HOSPITAL_DEPARTMENTS.map(({ name }) => name);
    const requestedDepartments = Array.isArray(departments) ? departments : [];
    const configuredDepartments = [...new Set(requestedDepartments.map((value) => String(value).trim()).filter((value) => supportedDepartments.includes(value)))];
    const hospitalDepartments = configuredDepartments.length ? configuredDepartments : supportedDepartments.slice(0, 8);

    const newHospital = new Hospital({
      hospitalID,
      name: hospitalName,
      country: 'USA',
      adminManagerName: managerName,
      adminManagerEmail: managerEmail,
      isActive: true,
      departments: hospitalDepartments.map((name) => ({ name, isActive: true })),
    });

    // Step 1.5: Save hospital FIRST (critical for foreign keys)
    console.log('[HOSPITAL REGISTER] Saving hospital first...');
    await newHospital.save();
    console.log('[HOSPITAL REGISTER] Hospital saved with ID:', newHospital._id);

    // Step 2: Create Admin Manager for the hospital
    const managerID = `MGR-${Date.now()}`;
    const newManager = new Manager({
      hospitalId: newHospital._id,
      managedHospitalIds: [newHospital._id],
      managerID,
      name: managerName,
      surname: managerSurname,
      email: managerEmail,
      password: hashedPassword,
      department: department || 'All',
      role: 'ADMIN',
      permissions: {
        canCreateShifts: true,
        canEditShifts: true,
        canDeleteShifts: true,
        canAssignStaff: true,
        canApproveLeaves: true,
        canManageRoles: true,
        canViewReports: true
      },
      isActive: true
    });

    // Step 3: Save manager
    console.log('[HOSPITAL REGISTER] Saving manager with hospitalId:', newHospital._id);
    await newManager.save();
    console.log('[HOSPITAL REGISTER] Manager saved with ID:', newManager._id);
    
    // Step 4: Update hospital with manager reference
    newHospital.adminManagerId = newManager._id;
    await newHospital.save();
    console.log('[HOSPITAL REGISTER] Hospital updated with adminManagerId');

    // Generate JWT token with hospital context
    const token = jwt.sign(
      {
        id: newManager._id,
        email: newManager.email,
        name: newManager.name,
        role: newManager.role,
        department: newManager.department,
        hospitalId: newHospital._id,
        hospitalName: newHospital.name,
      },
      JWT_SECRET,
      { expiresIn: '24h' }
    );

    console.log('[HOSPITAL REGISTER] TOKEN GENERATED');
    console.log('  Hospital ID:', newHospital._id);
    console.log('  Manager ID:', newManager._id);
    console.log('  Token includes hospitalId:', newHospital._id);

    res.status(201).json({
      success: true,
      message: 'Hospital registered successfully',
      hospital: {
        hospitalID: newHospital.hospitalID,
        name: newHospital.name,
        adminManager: {
          managerID: newManager.managerID,
          name: newManager.name,
          email: newManager.email,
          role: newManager.role
        }
      },
      token,
      user: {
        id: newManager._id,
        name: newManager.name,
        email: newManager.email,
        role: newManager.role,
        department: newManager.department,
        hospitalId: newHospital._id,
        hospitalName: newHospital.name,
      }
    });
  } catch (err) {
    console.error('Hospital registration error:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to register hospital',
      error: err.message
    });
  }
};

const managerHospitalIds = (manager) => [...new Set([
  ...(manager.managedHospitalIds || []).map(String),
  manager.hospitalId ? String(manager.hospitalId._id || manager.hospitalId) : null
].filter(Boolean))];

const managerToken = (manager, hospital) => jwt.sign({
  id: manager._id,
  email: manager.email,
  name: manager.name,
  role: manager.role,
  department: manager.department,
  hospitalId: hospital._id,
  hospitalName: hospital.name,
  managedHospitalIds: managerHospitalIds(manager)
}, JWT_SECRET, { expiresIn: '24h' });

export const listManagedHospitals = async (req, res) => {
  try {
    const manager = await Manager.findById(req.user._id).lean();
    if (!manager) return res.status(404).json({ success: false, message: 'Manager not found' });
    const hospitals = await Hospital.find({ _id: { $in: managerHospitalIds(manager) } }).select('hospitalID name country isActive departments').sort({ name: 1 }).lean();
    return res.json({ success: true, selectedHospitalId: String(req.user.hospitalId), hospitals });
  } catch (error) {
    console.error('[MANAGED HOSPITALS] List error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load managed hospitals' });
  }
};

export const createManagedHospital = async (req, res) => {
  try {
    const {
      hospitalName,
      country = 'USA',
      department = 'All',
      departments = [],
      adminName,
      adminSurname,
      adminEmail,
      adminPassword
    } = req.body || {};
    if (!hospitalName || !String(hospitalName).trim()) return res.status(400).json({ success: false, message: 'Hospital name is required' });
    const manager = await Manager.findById(req.user._id);
    if (!manager) return res.status(404).json({ success: false, message: 'Manager not found' });
    if (!adminName || !adminSurname || !adminEmail || !adminPassword) return res.status(400).json({ success: false, message: 'Additional administrator name, email, and password are required' });
    if (String(adminPassword).trim().length < 8) return res.status(400).json({ success: false, message: 'Additional administrator password must be at least 8 characters' });

    const normalizedAdminEmail = String(adminEmail).trim().toLowerCase();
    if (normalizedAdminEmail === String(manager.email).toLowerCase()) return res.status(400).json({ success: false, message: 'Use a different email for the additional hospital administrator' });

    const supportedDepartments = HOSPITAL_DEPARTMENTS.map(({ name }) => name);
    const configuredDepartments = [...new Set((Array.isArray(departments) ? departments : []).map((value) => String(value).trim()).filter((value) => supportedDepartments.includes(value)))];
    const hospitalDepartments = configuredDepartments.length ? configuredDepartments : supportedDepartments;
    const specialtyByDepartment = new Map();
    for (const configuration of Array.isArray(req.body?.departmentSpecialties) ? req.body.departmentSpecialties : []) {
      const name = String(configuration?.department || '').trim();
      const specialties = Array.isArray(configuration?.specialties) ? configuration.specialties : [];
      if (!hospitalDepartments.includes(name)) {
        return res.status(400).json({ success: false, message: `Specialties reference an unconfigured department: ${name || 'unknown'}` });
      }
      const normalizedSpecialties = [...new Set(specialties.map((value) => String(value).trim()).filter(Boolean))];
      if (normalizedSpecialties.some((specialty) => !HOSPITAL_SPECIALTIES.some((option) => option.name === specialty))) {
        return res.status(400).json({ success: false, message: `Department ${name} contains an unsupported specialty` });
      }
      specialtyByDepartment.set(name, normalizedSpecialties.map((specialty) => ({ name: specialty })));
    }
    const hospital = await Hospital.create({
      hospitalID: `HOSP-${Date.now()}`,
      name: String(hospitalName).trim(),
      country: String(country).trim() || 'USA',
      adminManagerId: manager._id,
      adminManagerName: manager.name,
      adminManagerEmail: manager.email,
      isActive: true,
      departments: hospitalDepartments.map((name) => ({
        name,
        isActive: true,
        specialties: specialtyByDepartment.get(name) || []
      }))
    });

    const additionalAdmin = await Manager.create({
      hospitalId: hospital._id,
      managedHospitalIds: [hospital._id],
      managerID: `MGR-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      name: String(adminName).trim(),
      surname: String(adminSurname).trim(),
      email: normalizedAdminEmail,
      password: await bcrypt.hash(String(adminPassword).trim(), 10),
      department: String(department).trim() || 'All',
      role: 'ADMIN',
      permissions: { canCreateShifts: true, canEditShifts: true, canDeleteShifts: true, canAssignStaff: true, canApproveLeaves: true, canManageRoles: true, canViewReports: true },
      isActive: true
    });

    const ids = managerHospitalIds(manager);
    manager.managedHospitalIds = [...new Set([...ids, String(hospital._id)])];
    await manager.save();
    await AuditLog.create({ action: 'HOSPITAL_CREATED', actorId: manager._id, hospitalId: hospital._id, resourceType: 'HOSPITAL', authorized: true, changeDetails: { additionalAdminId: additionalAdmin._id, additionalAdminEmail: additionalAdmin.email, departments: hospitalDepartments }, description: 'Administrator created a managed hospital and additional administrator account' });
    return res.status(201).json({
      success: true,
      hospital: { ...hospital.toObject(), additionalAdmin: { id: additionalAdmin._id, name: additionalAdmin.name, surname: additionalAdmin.surname, email: additionalAdmin.email } },
      selectedHospitalId: String(hospital._id),
      token: managerToken(manager, hospital),
      message: 'Hospital created under your administrator account; additional administrator credentials were created'
    });
  } catch (error) {
    console.error('[MANAGED HOSPITALS] Create error:', error);
    return res.status(500).json({ success: false, message: 'Failed to create managed hospital' });
  }
};

export const selectManagedHospital = async (req, res) => {
  try {
    const manager = await Manager.findById(req.user._id).populate('hospitalId');
    if (!manager) return res.status(404).json({ success: false, message: 'Manager not found' });
    const allowed = managerHospitalIds(manager);
    if (!allowed.includes(String(req.params.hospitalId))) return res.status(403).json({ success: false, message: 'You do not manage this hospital' });
    const hospital = await Hospital.findById(req.params.hospitalId);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    return res.json({ success: true, token: managerToken(manager, hospital), hospital: { _id: hospital._id, hospitalID: hospital.hospitalID, name: hospital.name } });
  } catch (error) {
    console.error('[MANAGED HOSPITALS] Select error:', error);
    return res.status(500).json({ success: false, message: 'Failed to select managed hospital' });
  }
};

export const updateManagedHospitalCredentials = async (req, res) => {
  try {
    const manager = await Manager.findById(req.user._id).populate('hospitalId');
    if (!manager) return res.status(404).json({ success: false, message: 'Manager not found' });

    const allowed = managerHospitalIds(manager);
    if (!allowed.includes(String(req.params.hospitalId))) {
      return res.status(403).json({ success: false, message: 'You do not manage this hospital' });
    }

    const hospital = await Hospital.findById(req.params.hospitalId);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });

    const { name, surname, email, password, phone } = req.body || {};
    if (!name && !surname && !email && !phone && !password) {
      return res.status(400).json({ success: false, message: 'At least one credential field is required' });
    }

    if (email && email !== manager.email) {
      const existingManager = await Manager.findOne({ email: email.toLowerCase().trim(), _id: { $ne: manager._id } }).lean();
      if (existingManager) {
        return res.status(409).json({ success: false, message: 'An account is already using that email address' });
      }
    }

    if (name) manager.name = String(name).trim();
    if (surname) manager.surname = String(surname).trim();
    if (email) manager.email = String(email).trim().toLowerCase();
    if (phone !== undefined) manager.phone = String(phone).trim() || '';
    if (password && String(password).trim()) {
      manager.password = await bcrypt.hash(String(password).trim(), 10);
    }

    manager.hospitalId = hospital._id;
    await manager.save();

    if (hospital.adminManagerEmail !== manager.email || hospital.adminManagerName !== manager.name) {
      hospital.adminManagerName = manager.name;
      hospital.adminManagerEmail = manager.email;
      await hospital.save();
    }

    await AuditLog.create({ action: 'HOSPITAL_CREDENTIALS_UPDATED', actorId: manager._id, hospitalId: hospital._id, resourceType: 'HOSPITAL_ADMIN', authorized: true, changeDetails: { email: manager.email, name: manager.name, surname: manager.surname, passwordChanged: Boolean(password) }, description: 'Administrator updated hospital administrator credentials' });

    const token = managerToken(manager, hospital);

    return res.json({
      success: true,
      message: 'Hospital credentials updated successfully',
      token,
      hospital: { _id: hospital._id, hospitalID: hospital.hospitalID, name: hospital.name },
      user: {
        id: manager._id,
        name: manager.name,
        email: manager.email,
        role: manager.role,
        department: manager.department,
        hospitalId: hospital._id,
        hospitalName: hospital.name,
      }
    });
  } catch (error) {
    console.error('[MANAGED HOSPITALS] Update credentials error:', error);
    return res.status(500).json({ success: false, message: 'Failed to update managed hospital credentials' });
  }
};

/**
 * LOGIN MANAGER/ADMIN (Phase 1)
 * Managers and admins log in to access the dashboard
 */
export const loginManager = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Email and password are required'
      });
    }

    // Find manager by email
    const manager = await Manager.findOne({ email }).populate('hospitalId');

    if (!manager) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    const isPasswordValid = await bcrypt.compare(password, manager.password);

    if (!isPasswordValid) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    // Update last login
    manager.lastLogin = new Date();
    await manager.save();

    // Generate JWT token with hospital context
    const token = managerToken(manager, manager.hospitalId);

    console.log('[MANAGER LOGIN] TOKEN GENERATED');
    console.log('  Manager ID:', manager._id);
    console.log('  Hospital ID:', manager.hospitalId._id);
    console.log('  Email:', manager.email);
    console.log('  Role:', manager.role);
    console.log('  Token includes hospitalId:', manager.hospitalId._id);

    res.status(200).json({
      success: true,
      message: 'Login successful',
      token,
      user: {
        id: manager._id,
        name: manager.name,
        email: manager.email,
        role: manager.role,
        department: manager.department,
        hospitalId: manager.hospitalId._id,
        hospitalName: manager.hospitalId.name,
        permissions: manager.permissions
      }
    });
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({
      success: false,
      message: 'Login failed',
      error: err.message
    });
  }
};

export const getHospitalTrialMode = async (req, res) => {
  try {
    const hospital = await Hospital.findById(req.hospitalId)
      .select('name environment clinicalTrialMode clinicalTrialModeUpdatedAt clinicalTrialModeUpdatedByName clinicalTrialModeReason')
      .lean();
    if (!hospital) {
      return res.status(404).json({ success: false, code: 'HOSPITAL_NOT_FOUND', message: 'Hospital not found' });
    }

    return res.json({
      success: true,
      settings: {
        hospitalName: hospital.name,
        environment: hospital.environment || 'SANDBOX',
        clinicalTrialMode: Boolean(hospital.clinicalTrialMode),
        updatedAt: hospital.clinicalTrialModeUpdatedAt || null,
        updatedBy: hospital.clinicalTrialModeUpdatedByName || null,
        reason: hospital.clinicalTrialModeReason || null,
      },
    });
  } catch (error) {
    console.error('[HOSPITAL TRIAL MODE] Read failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to load hospital trial settings' });
  }
};
