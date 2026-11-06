import bcrypt from 'bcrypt';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { ClinicalTrial } from '../models/clinicalTrial.model.js';
import { Hospital } from '../models/hospital.model.js';
import { Manager } from '../models/manager.model.js';
import { SandboxAccount } from '../models/sandboxAccount.model.js';
import { AuditLog } from '../models/auditLog.model.js';
import {
  buildClinicalTrialModeFilter,
  buildClinicalTrialModeUpdate,
  evaluateClinicalTrialModeChange,
} from '../utils/clinicalTrialModePolicy.js';
import { EHR_PROTOCOLS, normalizeEhrProtocols, resolveEhrIntegrationAccess } from '../utils/ehrIntegrationAccess.js';

const departments = ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'General'];
const productionOnboardingUnavailable = 'Production hospital onboarding is not available yet';

const writeSecondaryAudit = async (event) => {
  try {
    await AuditLog.create(event);
  } catch (error) {
    console.error('[AUDIT LOG] Primary hospital history was saved, but the secondary audit copy failed:', error);
  }
};

const normalizedDepartments = (values) => {
  const selected = [...new Set((Array.isArray(values) ? values : []).map((value) => String(value).trim()).filter((value) => departments.includes(value)))];
  return selected.length ? selected : departments;
};

const normalizeAdminInput = (body = {}) => ({
  name: String(body.adminName || body.name || '').trim(),
  surname: String(body.adminSurname || body.surname || '').trim(),
  email: String(body.adminEmail || body.email || '').trim().toLowerCase(),
  password: String(body.adminPassword || body.password || '')
});

const validateAdminInput = ({ name, surname, email, password }) => {
  if (!name || !surname || !email || !password) return 'Admin name, surname, email, and password are required';
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return 'Enter a valid hospital admin email address';
  if (password.length < 8) return 'Hospital admin password must be at least 8 characters';
  return null;
};

const checkEmailAvailable = async (email) => {
  const regex = new RegExp(`^${email.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i');
  const [manager, staff, account] = await Promise.all([
    Manager.exists({ email: regex }),
    ClinicalStaff.exists({ email: regex }),
    SandboxAccount.exists({ email: regex })
  ]);
  return !(manager || staff || account);
};

const createHospitalAdmin = async (hospital, credentials) => Manager.create({
  hospitalId: hospital._id,
  managedHospitalIds: [hospital._id],
  managerID: `MGR-${Date.now()}-${Math.floor(Math.random() * 100000)}`,
  name: credentials.name,
  surname: credentials.surname,
  email: credentials.email,
  password: await bcrypt.hash(credentials.password, 12),
  department: 'All',
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
  isActive: false
});

const serializeHospital = (hospital, admins = []) => ({
  id: String(hospital._id),
  hospitalID: hospital.hospitalID,
  name: hospital.name,
  country: hospital.country,
  isActive: hospital.isActive,
  environment: hospital.environment || 'SANDBOX',
  clinicalTrialMode: Boolean(hospital.clinicalTrialMode),
  clinicalTrialModeUpdatedAt: hospital.clinicalTrialModeUpdatedAt || null,
  clinicalTrialModeUpdatedByName: hospital.clinicalTrialModeUpdatedByName || null,
  clinicalTrialModeReason: hospital.clinicalTrialModeReason || null,
  ehrIntegrationAccess: resolveEhrIntegrationAccess(
    hospital.ehrConfig?.integrationAccess,
    hospital.environment || 'SANDBOX'
  ),
  departments: (hospital.departments || []).map((department) => department.name),
  admins: admins.map((admin) => ({
    id: String(admin._id),
    name: admin.name,
    surname: admin.surname,
    email: admin.email,
    role: admin.role,
    isActive: admin.isActive
  }))
});

export const listAccountHospitals = async (req, res) => {
  try {
    const hospitals = await Hospital.find({ ownerAccountId: req.sandboxAccount._id }).sort({ createdAt: -1 }).lean();
    const results = await Promise.all(hospitals.map(async (hospital) => {
      const admins = await Manager.find({ hospitalId: hospital._id, role: 'ADMIN' })
        .select('name surname email role isActive')
        .sort({ createdAt: 1 })
        .lean();
      return serializeHospital(hospital, admins);
    }));
    return res.json({ success: true, hospitals: results });
  } catch (error) {
    console.error('[ACCOUNT HOSPITALS] List failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to load your hospitals' });
  }
};

export const createAccountHospital = async (req, res) => {
  const hospitalName = String(req.body?.hospitalName || '').trim();
  const country = String(req.body?.country || 'USA').trim() || 'USA';
  const environment = String(req.body?.environment || 'SANDBOX').trim().toUpperCase();
  const clinicalTrialMode = req.body?.clinicalTrialMode === undefined ? false : req.body.clinicalTrialMode;
  const clinicalTrialModeReason = String(req.body?.clinicalTrialModeReason || '').trim();
  const credentials = normalizeAdminInput(req.body);
  const validationError = validateAdminInput(credentials);
  if (!hospitalName) return res.status(400).json({ success: false, message: 'Hospital name is required' });
  if (validationError) return res.status(400).json({ success: false, message: validationError });
  if (!['SANDBOX', 'PRODUCTION'].includes(environment)) {
    return res.status(400).json({ success: false, message: 'Environment must be SANDBOX or PRODUCTION' });
  }
  if (environment !== 'SANDBOX') {
    return res.status(403).json({ success: false, code: 'PRODUCTION_ONBOARDING_UNAVAILABLE', message: productionOnboardingUnavailable });
  }
  if (typeof clinicalTrialMode !== 'boolean') {
    return res.status(400).json({ success: false, message: 'Clinical trial mode must be true or false' });
  }
  if (clinicalTrialMode && !clinicalTrialModeReason) {
    return res.status(400).json({ success: false, message: 'A reason is required when enabling clinical trial mode' });
  }

  try {
    if (!(await checkEmailAvailable(credentials.email))) {
      return res.status(409).json({ success: false, message: 'That email is already used by an account, administrator, or staff member' });
    }

    const createdAt = new Date();
    const initialModeReason = clinicalTrialModeReason || 'Clinical trial mode disabled by default at hospital creation';
    const hospital = await Hospital.create({
      hospitalID: `HOSP-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      name: hospitalName,
      country,
      environment,
      clinicalTrialMode,
      clinicalTrialModeUpdatedAt: createdAt,
      clinicalTrialModeUpdatedBy: req.sandboxAccount._id,
      clinicalTrialModeUpdatedByName: req.sandboxAccount.name,
      clinicalTrialModeReason: initialModeReason,
      clinicalTrialModeHistory: [{
        eventType: 'MODE_CHANGED',
        from: null,
        to: clinicalTrialMode,
        reason: initialModeReason,
        by: req.sandboxAccount._id,
        byName: req.sandboxAccount.name,
        at: createdAt,
      }],
      ownerAccountId: req.sandboxAccount._id,
      adminManagerName: credentials.name,
      adminManagerEmail: credentials.email,
      isActive: true,
      departments: normalizedDepartments(req.body?.departments).map((name) => ({ name, isActive: true }))
    });

    let admin;
    try {
      admin = await createHospitalAdmin(hospital, credentials);
      hospital.adminManagerId = admin._id;
      await hospital.save();
      admin.isActive = true;
      await admin.save();
    } catch (error) {
      console.error('[ACCOUNT HOSPITALS] Hospital was created, but administrator provisioning failed:', error);
      const historyEvent = {
        eventType: 'ADMIN_CREATION_FAILED',
        from: Boolean(hospital.clinicalTrialMode),
        to: Boolean(hospital.clinicalTrialMode),
        reason: 'Hospital administrator provisioning failed during hospital creation',
        by: req.sandboxAccount._id,
        byName: req.sandboxAccount.name,
        at: new Date(),
        details: { error: error.message },
      };
      try {
        const updatedHospital = await Hospital.findByIdAndUpdate(
          hospital._id,
          { $push: { clinicalTrialModeHistory: historyEvent } },
          { new: true }
        );
        if (!updatedHospital) throw new Error('Hospital disappeared before the provisioning failure could be recorded');
        await writeSecondaryAudit({
          action: 'HOSPITAL_CREATED',
          actorId: req.sandboxAccount._id,
          actorIdentityType: 'ACCOUNT_OWNER',
          hospitalId: hospital._id,
          resourceType: 'HOSPITAL',
          authorized: true,
          reason: historyEvent.reason,
          changeDetails: { adminCreated: false, error: error.message },
          description: 'Hospital was created but administrator provisioning failed',
        });
        return res.status(201).json({
          success: true,
          partial: true,
          adminCreated: false,
          code: 'ADMIN_CREATION_FAILED',
          message: 'Hospital workspace was saved, but its administrator could not be created. Add an administrator to recover.',
          hospital: serializeHospital(updatedHospital),
        });
      } catch (historyError) {
        console.error('[ACCOUNT HOSPITALS] Failed to record administrator provisioning failure in hospital history:', historyError);
        return res.status(500).json({
          success: false,
          partial: true,
          adminCreated: false,
          auditRecorded: false,
          code: 'ADMIN_CREATION_FAILED_AUDIT_FAILED',
          message: 'Hospital workspace was saved, but administrator creation failed and the failure could not be recorded. Refresh and retry administrator setup.',
          hospital: serializeHospital(hospital),
        });
      }
    }

    await writeSecondaryAudit({
      action: 'HOSPITAL_CREATED',
      actorId: req.sandboxAccount._id,
      actorIdentityType: 'ACCOUNT_OWNER',
      hospitalId: hospital._id,
      resourceType: 'HOSPITAL',
      authorized: true,
      ipAddress: req.ip,
      userAgent: req.get('user-agent'),
      changeDetails: { environment, clinicalTrialMode, clinicalTrialModeReason: initialModeReason },
      description: `Account owner created a ${environment.toLowerCase()} hospital workspace`,
    });

    return res.status(201).json({
      success: true,
      adminCreated: true,
      hospital: serializeHospital(hospital, [admin]),
      initialAdminCredentials: { name: credentials.name, email: credentials.email, password: credentials.password },
      message: 'Hospital created. Save the admin password now; it cannot be retrieved later.'
    });
  } catch (error) {
    console.error('[ACCOUNT HOSPITALS] Create failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to create hospital and administrator credentials' });
  }
};

const findOwnedHospital = (accountId, hospitalId) => Hospital.findOne({ _id: hospitalId, ownerAccountId: accountId });

export const updateAccountHospitalTrialMode = async (req, res) => {
  const requestedMode = req.body?.clinicalTrialMode;
  const reason = String(req.body?.reason || '').trim();
  if (typeof requestedMode !== 'boolean') {
    return res.status(400).json({ success: false, code: 'INVALID_TRIAL_MODE', message: 'Clinical trial mode must be true or false' });
  }
  if (!reason) {
    return res.status(400).json({ success: false, message: 'A reason is required to change clinical trial mode' });
  }

  try {
    const hospital = await findOwnedHospital(req.sandboxAccount._id, req.params.hospitalId);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found in your account' });

    const activeTrialActualCount = requestedMode
      ? 0
      : await ClinicalTrial.countDocuments({ hospitalId: hospital._id, status: 'ACTIVE' });
    const hasActiveTrials = activeTrialActualCount > 0 || (hospital.activeTrialCount || 0) > 0;
    const decision = evaluateClinicalTrialModeChange({
      currentMode: hospital.clinicalTrialMode,
      requestedMode,
      hasActiveTrials,
    });

    if (!decision.allowed) {
      return res.status(decision.code === 'ACTIVE_TRIALS_EXIST' ? 409 : 400).json({
        success: false,
        code: decision.code,
        message: decision.code === 'ACTIVE_TRIALS_EXIST'
          ? 'Clinical trial mode cannot be disabled while this hospital has active trials'
          : 'Invalid clinical trial mode change',
      });
    }

    if (!decision.changed) {
      return res.json({ success: true, hospital: serializeHospital(hospital) });
    }

    const changedAt = new Date();
    const updatedHospital = await Hospital.findOneAndUpdate(
      buildClinicalTrialModeFilter({
        hospitalId: hospital._id,
        ownerAccountId: req.sandboxAccount._id,
        currentMode: Boolean(hospital.clinicalTrialMode),
        requestedMode,
      }),
      buildClinicalTrialModeUpdate({
        currentMode: hospital.clinicalTrialMode,
        requestedMode,
        reason,
        actorId: req.sandboxAccount._id,
        actorName: req.sandboxAccount.name,
        at: changedAt,
      }),
      { new: true, runValidators: true }
    );
    if (!updatedHospital) {
      if (!requestedMode) {
        const currentHospital = await Hospital.findById(hospital._id).select('activeTrialCount').lean();
        if (currentHospital?.activeTrialCount > 0) {
          return res.status(409).json({
            success: false,
            code: 'ACTIVE_TRIALS_EXIST',
            message: 'Clinical trial mode cannot be disabled while this hospital has active trials',
          });
        }
      }
      return res.status(409).json({ success: false, code: 'SETTINGS_CHANGED', message: 'Hospital trial settings changed; refresh and try again' });
    }

    await writeSecondaryAudit({
      action: 'HOSPITAL_TRIAL_MODE_CHANGED',
      actorId: req.sandboxAccount._id,
      actorIdentityType: 'ACCOUNT_OWNER',
      hospitalId: hospital._id,
      resourceType: 'HOSPITAL',
      authorized: true,
      ipAddress: req.ip,
      userAgent: req.get('user-agent'),
      reason,
      changeDetails: {
        before: { clinicalTrialMode: Boolean(hospital.clinicalTrialMode) },
        after: { clinicalTrialMode: requestedMode },
        actor: req.sandboxAccount.email,
      },
      description: `Account owner ${requestedMode ? 'enabled' : 'disabled'} clinical trial mode`,
    });

    return res.json({ success: true, hospital: serializeHospital(updatedHospital) });
  } catch (error) {
    console.error('[ACCOUNT HOSPITALS] Trial mode update failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to update clinical trial mode' });
  }
};

export const reconcileAccountHospitalTrialCount = async (req, res) => {
  try {
    const hospital = await findOwnedHospital(req.sandboxAccount._id, req.params.hospitalId);
    if (!hospital) return res.status(404).json({ success: false, code: 'HOSPITAL_NOT_FOUND', message: 'Hospital not found in your account' });

    const reconciliation = await ClinicalTrial.reconcileHospitalActiveTrialCount({
      hospitalId: hospital._id,
      actorId: req.sandboxAccount._id,
      actorName: req.sandboxAccount.name,
    });
    if (!reconciliation.success) {
      return res.status(404).json({ success: false, code: reconciliation.code, message: 'Hospital not found' });
    }
    return res.json({ success: true, reconciliation });
  } catch (error) {
    console.error('[ACCOUNT HOSPITALS] Trial count reconciliation failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to reconcile the active-trial count' });
  }
};

export const updateAccountHospitalEhrAccess = async (req, res) => {
  const reason = String(req.body?.reason || '').trim();
  const clearAccountConfiguration = req.body?.clearAccountConfiguration === true;
  const requestedProtocols = req.body?.enabledProtocols;
  const enabledProtocols = normalizeEhrProtocols(requestedProtocols);
  if (!reason) {
    return res.status(400).json({ success: false, code: 'REASON_REQUIRED', message: 'A reason is required to change integration access' });
  }
  if (!clearAccountConfiguration && (!Array.isArray(requestedProtocols) || requestedProtocols.some((protocol) => !EHR_PROTOCOLS.includes(protocol)))) {
    return res.status(400).json({ success: false, code: 'INVALID_EHR_PROTOCOL', message: `Supported protocols are ${EHR_PROTOCOLS.join(' and ')}` });
  }

  try {
    const hospital = await findOwnedHospital(req.sandboxAccount._id, req.params.hospitalId);
    if (!hospital) return res.status(404).json({ success: false, code: 'HOSPITAL_NOT_FOUND', message: 'Hospital not found in your account' });
    if (hospital.environment && hospital.environment !== 'SANDBOX') {
      return res.status(403).json({ success: false, code: 'PRODUCTION_ONBOARDING_UNAVAILABLE', message: productionOnboardingUnavailable });
    }

    const before = resolveEhrIntegrationAccess(hospital.ehrConfig?.integrationAccess, hospital.environment);
    const integrationAccess = clearAccountConfiguration
      ? null
      : {
        source: 'ACCOUNT_OWNER',
        enabledProtocols,
        attestedAt: new Date(),
        attestedBy: req.sandboxAccount._id,
        attestedByName: req.sandboxAccount.name,
        reason,
      };
    const updatedHospital = await Hospital.findOneAndUpdate(
      {
        _id: hospital._id,
        ownerAccountId: req.sandboxAccount._id,
        $or: [{ environment: 'SANDBOX' }, { environment: { $exists: false } }],
      },
      { $set: { 'ehrConfig.integrationAccess': integrationAccess } },
      { new: true, runValidators: true }
    );
    if (!updatedHospital) {
      return res.status(409).json({ success: false, code: 'HOSPITAL_ACCESS_CHANGED', message: 'Hospital ownership or environment changed; refresh and try again' });
    }

    await writeSecondaryAudit({
      action: 'EHR_INTEGRATION_ACCESS_CHANGED',
      actorId: req.sandboxAccount._id,
      actorIdentityType: 'ACCOUNT_OWNER',
      hospitalId: hospital._id,
      resourceType: 'EHR_INTEGRATION_ACCESS',
      authorized: true,
      reason,
      ipAddress: req.ip,
      userAgent: req.get('user-agent'),
      changeDetails: {
        before: { source: before.source, enabledProtocols: before.enabledProtocols },
        after: { source: integrationAccess?.source || null, enabledProtocols: integrationAccess?.enabledProtocols || [] },
      },
      description: 'Sandbox account owner updated hospital EHR integration access',
    });

    return res.json({
      success: true,
      ehrIntegrationAccess: resolveEhrIntegrationAccess(integrationAccess, hospital.environment),
    });
  } catch (error) {
    console.error('[ACCOUNT HOSPITALS] EHR integration access update failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to update EHR integration access' });
  }
};

export const addAccountHospitalAdmin = async (req, res) => {
  const credentials = normalizeAdminInput(req.body);
  const validationError = validateAdminInput(credentials);
  if (validationError) return res.status(400).json({ success: false, message: validationError });

  try {
    const hospital = await findOwnedHospital(req.sandboxAccount._id, req.params.hospitalId);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found in your account' });
    if (!(await checkEmailAvailable(credentials.email))) {
      return res.status(409).json({ success: false, message: 'That email is already used by an account, administrator, or staff member' });
    }

    const currentAdmin = hospital.adminManagerId
      ? await Manager.findOne({ _id: hospital.adminManagerId, hospitalId: hospital._id, role: 'ADMIN' })
      : null;
    const admin = await createHospitalAdmin(hospital, credentials);
    if (!currentAdmin || !currentAdmin.isActive) {
      const updatedHospital = await Hospital.findOneAndUpdate(
        { _id: hospital._id, ownerAccountId: req.sandboxAccount._id },
        { $set: { adminManagerId: admin._id, adminManagerName: admin.name, adminManagerEmail: admin.email } },
        { new: true }
      );
      if (!updatedHospital) {
        throw new Error('Unable to attach administrator to the hospital workspace');
      }
    }
    admin.isActive = true;
    await admin.save();
    return res.status(201).json({
      success: true,
      admin: { id: String(admin._id), name: admin.name, surname: admin.surname, email: admin.email, role: admin.role, isActive: admin.isActive },
      credentials: { name: credentials.name, email: credentials.email, password: credentials.password },
      message: 'Administrator created. Save the password now; it cannot be retrieved later.'
    });
  } catch (error) {
    console.error('[ACCOUNT HOSPITALS] Admin create failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to create hospital administrator' });
  }
};

export const resetAccountHospitalAdminPassword = async (req, res) => {
  const password = String(req.body?.password || '');
  if (password.length < 8) return res.status(400).json({ success: false, message: 'New password must be at least 8 characters' });

  try {
    const hospital = await findOwnedHospital(req.sandboxAccount._id, req.params.hospitalId);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found in your account' });
    const admin = await Manager.findOne({ _id: req.params.adminId, hospitalId: hospital._id, role: 'ADMIN' });
    if (!admin) return res.status(404).json({ success: false, message: 'Hospital administrator not found' });

    admin.password = await bcrypt.hash(password, 12);
    await admin.save();
    return res.json({
      success: true,
      credentials: { name: admin.name, email: admin.email, password },
      message: 'Password reset. Save the new password now; it cannot be retrieved later.'
    });
  } catch (error) {
    console.error('[ACCOUNT HOSPITALS] Admin password reset failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to reset administrator password' });
  }
};