import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { Manager } from '../models/manager.model.js';
import { ClinicalStaff, Practitioner } from '../models/clinicalstaff.model.js';
import { SandboxAccount } from '../models/sandboxAccount.model.js';

const JWT_SECRET = process.env.JWT_SECRET || 'supersecretkey';

export const deriveTokenRole = (role) => {
  const rawRole = (role || '').toString().toUpperCase().trim();

  if (rawRole === 'ADMINISTRATIVE' || rawRole === 'ADMIN' || rawRole === 'MANAGER' || rawRole === 'SHIFT_MANAGER' || rawRole === 'DEPARTMENT_MANAGER') {
    return 'ADMIN';
  }

  return rawRole;
};

export const registerSandboxAccount = async (req, res) => {
  try {
    const name = String(req.body?.name || '').trim();
    const email = String(req.body?.email || '').trim().toLowerCase();
    const password = String(req.body?.password || '');

    if (!name || !email || !password) {
      return res.status(400).json({ success: false, message: 'Name, email, and password are required' });
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return res.status(400).json({ success: false, message: 'Enter a valid email address' });
    }
    if (password.length < 8) {
      return res.status(400).json({ success: false, message: 'Password must be at least 8 characters' });
    }

    const emailRegex = new RegExp(`^${email.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i');
    const [existingAccount, existingManager, existingStaff] = await Promise.all([
      SandboxAccount.findOne({ email: emailRegex }).select('_id').lean(),
      Manager.findOne({ email: emailRegex }).select('_id').lean(),
      ClinicalStaff.findOne({ email: emailRegex }).select('_id').lean()
    ]);
    if (existingAccount || existingManager || existingStaff) {
      return res.status(409).json({ success: false, message: 'An account already uses this email address' });
    }

    const account = await SandboxAccount.create({
      name,
      email,
      password: await bcrypt.hash(password, 12)
    });
    const token = jwt.sign({
      id: account._id,
      email: account.email,
      name: account.name,
      role: 'ACCOUNT_OWNER',
      identityType: 'ACCOUNT'
    }, JWT_SECRET, { expiresIn: '24h' });

    return res.status(201).json({
      success: true,
      message: 'Sandbox account created',
      token,
      user: { id: account._id, name: account.name, email: account.email, role: 'ACCOUNT_OWNER', identityType: 'ACCOUNT' }
    });
  } catch (error) {
    console.error('[ACCOUNT REGISTER] Error:', error);
    return res.status(500).json({ success: false, message: 'Unable to create account' });
  }
};

// ----------------------
// UNIFIED LOGIN
// ----------------------
export const login = async (req, res) => {
  try {
    console.log("[AUTH LOGIN] Attempting login with:", req.body.email);

    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Email and password are required'
      });
    }

    // Helper: case-insensitive exact match for email
    const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const emailRegex = new RegExp(`^${escapeRegex(email)}$`, 'i');

    const account = await SandboxAccount.findOne({ email: emailRegex }).select('+password');
    if (account) {
      const passwordValid = await bcrypt.compare(password, account.password);
      if (!passwordValid || !account.isActive) {
        return res.status(401).json({ success: false, message: 'Invalid credentials' });
      }

      account.lastLogin = new Date();
      await account.save();
      const token = jwt.sign({
        id: account._id,
        email: account.email,
        name: account.name,
        role: 'ACCOUNT_OWNER',
        identityType: 'ACCOUNT'
      }, JWT_SECRET, { expiresIn: '24h' });

      return res.status(200).json({
        success: true,
        message: 'Login successful',
        token,
        user: { id: account._id, name: account.name, email: account.email, role: 'ACCOUNT_OWNER', identityType: 'ACCOUNT' }
      });
    }

    // Try Manager collection first (case-insensitive)
    let user = await Manager.findOne({ email: emailRegex }).populate('hospitalId');
    let userType = 'manager';

    // If not found, try ClinicalStaff collection
    if (!user) {
      console.log("[AUTH LOGIN] Manager not found, searching ClinicalStaff collection...");
      user = await ClinicalStaff.findOne({ email: emailRegex }).select('+password').populate('hospitalId');
      userType = 'staff';
      if (user) {
        console.log("[AUTH LOGIN] STAFF FOUND in ClinicalStaff:", {
          email: user.email,
          hospitalId: user.hospitalId?._id?.toString?.() || user.hospitalId,
          firstName: user.firstName,
          lastName: user.lastName,
          systemRole: user.systemRole
        });
      }
    }

    // Fallbacks to handle schema/model changes: try Practitioner and lowercase/email regex searches
    if (!user) {
      try {
        console.log("[AUTH LOGIN] Staff not found, trying Practitioner collection...");
        user = await Practitioner.findOne({ email: emailRegex }).select('+password').populate('hospitalId');
        if (user) {
          userType = 'staff';
          console.log("[AUTH LOGIN] STAFF FOUND in Practitioner:", {
            email: user.email,
            hospitalId: user.hospitalId?._id?.toString?.() || user.hospitalId,
            firstName: user.firstName,
            lastName: user.lastName
          });
        }
      } catch (e) {
        console.error('[AUTH LOGIN] Practitioner lookup error:', e.message);
      }
    }

    if (!user) {
      try {
        console.log("[AUTH LOGIN] Final fallback: searching by partial email...");
        const partial = email.split('@')[0];
        const alt = await ClinicalStaff.findOne({ email: { $regex: new RegExp(partial, 'i') } }).select('+password');
        if (alt) {
          console.warn('[AUTH LOGIN] Found partial match in ClinicalStaff by regex:', {
            email: alt.email,
            hospitalId: alt.hospitalId,
            firstName: alt.firstName,
            lastName: alt.lastName
          });
        }
      } catch (e) {
        console.error('[AUTH LOGIN] Partial regex lookup error:', e.message);
      }
    }

    if (!user) {
      console.error("[AUTH LOGIN] USER NOT FOUND for email:", email);
      console.error("[AUTH LOGIN] This means:");
      console.error("   1. No staff/manager with this email exists");
      console.error("   2. Staff may not have been successfully registered");
      console.error("   3. Staff registered without email");
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    const displayName = user.name || (user.firstName ? `${user.firstName} ${user.lastName}` : 'Staff');
    const displayRole = user.role || user.systemRole;
    const displayHospitalId = user.hospitalId?._id?.toString?.() || user.hospitalId?.toString?.() || user.hospitalId;
    console.log("[AUTH LOGIN] USER AUTHENTICATED:", `${displayName} (Role: ${displayRole}, Type: ${userType}, Hospital: ${displayHospitalId})`);

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    console.log("[AUTH LOGIN] Password valid:", isPasswordValid);

    if (!isPasswordValid) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    // Extract hospital info
    const hospitalName = user?.hospitalId?.name || 'Hospital';
    const hospitalId = user?.hospitalId?._id || user.hospitalId;

    // Update last login for managers and staff
    if (userType === 'manager') {
      user.lastLogin = new Date();
      await user.save();
    } else if (userType === 'staff') {
      if (!user.metadata) user.metadata = {};
      user.metadata.lastLogin = new Date();
      await user.save();
    }

    // Normalize fields for token
    const tokenName = user.name || (user.firstName ? `${user.firstName} ${user.lastName}` : undefined);
    const systemRole = user.role || user.systemRole || '';
    const tokenRole = deriveTokenRole(systemRole);
    const tokenDept = user.department || user.assignment?.department;
    const tokenSpecialty = user.specialty || user.clinicalSpecialty || 'General';
    const tokenNsgRole = user.nsgRole || null;

    // Generate JWT token with role and hospital context
    const token = jwt.sign(
      {
        id: user._id,
        email: user.email,
        name: tokenName,
        role: tokenRole,
        systemRole: String(systemRole).toUpperCase().trim(),
        department: tokenDept,
        specialty: tokenSpecialty,
        nsgRole: tokenNsgRole,
        hospitalId: hospitalId,
        hospitalName: hospitalName
      },
      JWT_SECRET,
      { expiresIn: '24h' }
    );

    console.log("[AUTH LOGIN] Token created with role:", tokenRole, "specialty:", tokenSpecialty);

    // Build user response object
    const userResponse = {
      id: user._id,
      name: tokenName,
      email: user.email,
      role: tokenRole,
      systemRole: String(systemRole).toUpperCase().trim(),
      department: tokenDept,
      specialty: tokenSpecialty,
      nsgRole: tokenNsgRole,
      hospitalId: hospitalId,
      hospitalName: hospitalName
    };

    // Add manager-specific fields
    if (userType === 'manager') {
      userResponse.permissions = user.permissions;
    }

    // Add staff-specific surname/first/last for backward compatibility
    if (userType === 'staff') {
      userResponse.firstName = user.firstName;
      userResponse.lastName = user.lastName;
    }

    res.status(200).json({
      success: true,
      message: 'Login successful',
      token,
      user: userResponse
    });

  } catch (err) {
    console.error("[AUTH LOGIN] ERROR:", err);
    res.status(500).json({
      success: false,
      message: 'Login failed',
      error: err.message
    });
  }
};

// ----------------------
// DEBUG TOKEN (TEMP)
// ----------------------
export const debugToken = async (req, res) => {
  try {
    let token = null;
    if (req.headers.authorization?.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }
    if (!token) return res.status(400).json({ success: false, message: 'No token provided' });

    let decoded;
    try {
      decoded = jwt.verify(token, JWT_SECRET);
    } catch (err) {
      return res.status(400).json({ success: false, message: 'Invalid token', error: err.message });
    }

    const result = { decoded };

    // Try lookups
    result.manager = await Manager.findById(decoded.id).select('-password').lean().catch(() => null);
    result.clinicalStaff = await ClinicalStaff.findById(decoded.id).select('-password').lean().catch(() => null);
    result.practitioner = await Practitioner.findById(decoded.id).select('-password').lean().catch(() => null);

    // Also try lookup by email
    if (!result.clinicalStaff && decoded.email) {
      result.byEmail = await ClinicalStaff.findOne({ email: decoded.email }).select('-password').lean().catch(() => null);
    }

    return res.status(200).json({ success: true, result });
  } catch (err) {
    console.error('[DEBUG TOKEN] ERROR:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ----------------------
// FIND USER BY EMAIL (TEMP)
// ----------------------
export const findUserByEmail = async (req, res) => {
  try {
    const { email } = req.body || req.query || {};
    if (!email) return res.status(400).json({ success: false, message: 'email required' });

    const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const emailRegex = new RegExp(`^${escapeRegex(email)}$`, 'i');

    const manager = await Manager.findOne({ email: emailRegex }).select('-password').lean().catch(() => null);
    const clinical = await ClinicalStaff.findOne({ email: emailRegex }).select('-password').lean().catch(() => null);
    const pract = await Practitioner.findOne({ email: emailRegex }).select('-password').lean().catch(() => null);

    return res.status(200).json({ success: true, found: { manager, clinical, practitioner: pract } });
  } catch (err) {
    console.error('[FIND USER] ERROR:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
};
