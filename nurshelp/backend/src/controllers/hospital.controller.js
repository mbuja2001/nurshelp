/**
 * Hospital Controller
 * Handles hospital registration and initialization
 * Multi-tenant architecture - each hospital is independent
 */

import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { Hospital } from '../models/hospital.model.js';
import { Manager } from '../models/manager.model.js';

const JWT_SECRET = process.env.JWT_SECRET || 'supersecretkey';

/**
 * HOSPITAL REGISTRATION (Phase 1)
 * Create new hospital instance with admin/manager
 */
export const registerHospital = async (req, res) => {
  try {
    const { hospitalName, managerName, managerSurname, managerEmail, managerPassword, department } = req.body;

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
    const newHospital = new Hospital({
      hospitalID,
      name: hospitalName,
      country: 'USA',
      adminManagerName: managerName,
      adminManagerEmail: managerEmail,
      isActive: true,
    });

    // Step 1.5: Save hospital FIRST (critical for foreign keys)
    console.log('[HOSPITAL REGISTER] Saving hospital first...');
    await newHospital.save();
    console.log('[HOSPITAL REGISTER] Hospital saved with ID:', newHospital._id);

    // Step 2: Create Admin Manager for the hospital
    const managerID = `MGR-${Date.now()}`;
    const newManager = new Manager({
      hospitalId: newHospital._id,
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
    const token = jwt.sign(
      {
        id: manager._id,
        email: manager.email,
        name: manager.name,
        role: manager.role,
        department: manager.department,
        hospitalId: manager.hospitalId._id,
        hospitalName: manager.hospitalId.name,
      },
      JWT_SECRET,
      { expiresIn: '24h' }
    );

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
