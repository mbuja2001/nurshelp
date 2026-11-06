import mongoose from 'mongoose';
import { Notification } from '../models/notification.model.js';
import { Encounter } from '../models/encounter.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { Hospital } from '../models/hospital.model.js';
import { AuditLog } from '../models/auditLog.model.js';
import { canViewSurgery, sortOperationsEvents } from '../utils/operationsFeed.js';
import { buildCapacityAlerts, calculateClinicianWorkload, normalizeRole, validateAcuityWeightOrdering, workloadQuery } from '../utils/clinicianWorkload.js';
import { EHR_PROTOCOLS, normalizeEhrProtocols, resolveEhrIntegrationAccess } from '../utils/ehrIntegrationAccess.js';

export const resolveRequestHospitalId = (req, requestedHospitalId = null) => {
  const explicitHospitalId = requestedHospitalId ?? req?.body?.hospitalId ?? req?.query?.hospitalId ?? req?.params?.hospitalId;
  const authenticatedHospitalId = req?.user?.hospitalId ?? req?.user?.hospitalID ?? null;

  if (explicitHospitalId && authenticatedHospitalId && String(explicitHospitalId) !== String(authenticatedHospitalId)) {
    const error = new Error('Authenticated hospital does not match the requested hospital for this write.');
    error.statusCode = 403;
    throw error;
  }

  if (!authenticatedHospitalId && !explicitHospitalId) {
    const error = new Error('Authenticated hospital is required');
    error.statusCode = 401;
    throw error;
  }

  return explicitHospitalId || authenticatedHospitalId;
};

export const resolveAuthenticatedEhrHospitalId = (req) => {
  const hospitalId = req?.hospitalId ?? req?.user?.hospitalId ?? req?.user?.hospitalID ?? null;
  if (!hospitalId) {
    const error = new Error('Authenticated hospital is required');
    error.statusCode = 401;
    throw error;
  }
  return hospitalId;
};

export const buildCapacityAuditDetails = ({ actor, hospitalId, level, target, role, oldValue, newValue }) => ({
  actor: actor ?? 'unknown',
  hospitalId: hospitalId ? String(hospitalId) : null,
  level,
  target,
  role: role ? normalizeRole(role) : null,
  oldValue,
  newValue,
  timestamp: new Date().toISOString()
});

const DEFAULT_EHR_CONFIG = {
  canonicalFormat: 'FHIR R4',
  adapters: [
    {
      id: 'hl7v2',
      name: 'HL7v2',
      status: 'Not configured',
      connection: 'Not configured',
      host: '',
      port: '2575',
      protocol: 'MLLP'
    },
    {
      id: 'fhir-r4',
      name: 'FHIR R4',
      status: 'Not configured',
      connection: 'Not configured',
      endpoint: '',
      protocol: 'FHIR REST'
    }
  ],
  integrationAccess: null,
  deadLetter: {
    inboundFailures: 0,
    retryQueue: 0,
    failedMessages: [],
    retryItems: []
  },
  retention: [
    { id: 'primary-records', category: 'Primary clinical record', scope: 'Primary record', window: '4 years', minimum: '6 years', compliant: false },
    { id: 'audit-logs', category: 'Audit logs', scope: 'Primary record', window: '7 years', minimum: '6 years', compliant: true }
  ],
  terminologyRows: [
    { code: 'RED', status: 'Mapped', value: 'Priority = RED', color: '#ef4444' },
    { code: 'ORANGE', status: 'Mapped', value: 'Priority = ORANGE', color: '#f97316' },
    { code: 'YELLOW', status: 'Mapped', value: 'Priority = YELLOW', color: '#eab308' },
    { code: 'GREEN', status: 'Mapped', value: 'Priority = GREEN', color: '#22c55e' }
  ],
  accessMatrix: [
    { role: 'Physician', canView: true, canEdit: true, scope: 'Full encounter access' },
    { role: 'Nurse', canView: true, canEdit: false, scope: 'Ward-scoped records' }
  ],
  protocolPolicy: {
    satsVersion: 'SATS v3.1',
    tewsVersion: 'TEWS v2.4',
    reportRequirements: 'System default',
    retentionGovernance: 'POPIA-aligned'
  },
  alertRouting: [
    { alertType: 'QUEUE_CONGESTION', targets: ['Charge nurse', 'ED duty lead'], escalation: 'High' }
  ]
};

const normalizeEhrConfig = (config = {}, environment = 'SANDBOX') => {
  const merged = structuredClone ? structuredClone(DEFAULT_EHR_CONFIG) : JSON.parse(JSON.stringify(DEFAULT_EHR_CONFIG));
  const source = config && typeof config === 'object' ? config : {};
  const integrationAccess = resolveEhrIntegrationAccess(source.integrationAccess, environment);
  const enabledProtocols = integrationAccess.enabledProtocols;
  const sourceAdapters = Array.isArray(source.adapters) ? source.adapters : [];
  const adapters = merged.adapters.map((template) => {
    const saved = sourceAdapters.find((adapter) => adapter?.id === template.id) || {};
    const enabled = enabledProtocols.includes(template.id);
    const hasConfiguration = template.id === 'hl7v2'
      ? Boolean(saved.host && saved.port)
      : Boolean(saved.endpoint);
    return {
      ...template,
      ...(enabled ? (template.id === 'hl7v2'
        ? { host: saved.host || '', port: saved.port || template.port, sendingFacility: saved.sendingFacility || '', receivingFacility: saved.receivingFacility || '' }
        : { endpoint: saved.endpoint || '', authMode: saved.authMode || 'NONE' }) : {}),
      status: enabled ? (hasConfiguration ? 'Configured' : 'Access enabled') : 'Not configured',
      connection: enabled && hasConfiguration ? 'Not tested' : 'Not configured',
      lastTest: null,
    };
  });
  return {
    canonicalFormat: source.canonicalFormat || merged.canonicalFormat,
    integrationAccess,
    adapters,
    deadLetter: source.deadLetter && typeof source.deadLetter === 'object' ? {
      ...merged.deadLetter,
      ...source.deadLetter,
      failedMessages: Array.isArray(source.deadLetter.failedMessages) ? source.deadLetter.failedMessages : merged.deadLetter.failedMessages,
      retryItems: Array.isArray(source.deadLetter.retryItems) ? source.deadLetter.retryItems : merged.deadLetter.retryItems
    } : merged.deadLetter,
    retention: Array.isArray(source.retention) && source.retention.length ? source.retention : merged.retention,
    terminologyRows: Array.isArray(source.terminologyRows) && source.terminologyRows.length ? source.terminologyRows : merged.terminologyRows,
    accessMatrix: Array.isArray(source.accessMatrix) && source.accessMatrix.length ? source.accessMatrix : merged.accessMatrix,
    protocolPolicy: source.protocolPolicy && typeof source.protocolPolicy === 'object' ? { ...merged.protocolPolicy, ...source.protocolPolicy } : merged.protocolPolicy,
    alertRouting: Array.isArray(source.alertRouting) && source.alertRouting.length ? source.alertRouting : merged.alertRouting
  };
};

const validateEhrAdapters = (adapters) => {
  for (const adapter of adapters) {
    if (adapter.id === 'hl7v2') {
      const host = String(adapter.host || '').trim();
      const port = String(adapter.port || '').trim();
      if (host && (!/^[a-zA-Z0-9.-]+$/.test(host) || host.startsWith('.') || host.endsWith('.'))) {
        return 'Enter a valid HL7 host name or IP address';
      }
      if (host && (!/^\d+$/.test(port) || Number(port) < 1 || Number(port) > 65535)) {
        return 'HL7 port must be a number between 1 and 65535';
      }
    }
    if (adapter.id === 'fhir-r4' && adapter.endpoint) {
      try {
        const endpoint = new URL(adapter.endpoint);
        if (!['http:', 'https:'].includes(endpoint.protocol)
          || endpoint.username
          || endpoint.password
          || endpoint.search
          || endpoint.hash) {
          return 'FHIR base URL must use HTTP or HTTPS and must not contain credentials, query parameters, or fragments';
        }
      } catch {
        return 'Enter a valid FHIR endpoint URL';
      }
    }
  }
  return null;
};

const writeEhrAudit = async (event) => {
  try {
    await AuditLog.create(event);
  } catch (error) {
    console.error('[EHR CONFIG] Configuration was saved, but its secondary audit copy failed:', error);
  }
};

const workloadSelect = '_id status isWaiting specialistAttended assigned_clinicians assignedPhysicianId assignedPhysicianName assignedNurseId assignedNurseName nurse_id triage';

const buildWorkloadContext = async (hospitalId, encounters, includeClinicianId = null) => {
  const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId)
    ? { _id: hospitalId }
    : { hospitalID: hospitalId };
  const hospital = await Hospital.findOne(hospitalQuery)
    .select('workloadPolicy')
    .lean();
  const clinicianIds = [...new Set(encounters.flatMap((encounter) => [
    ...(encounter.assigned_clinicians || []).map((clinician) => clinician?.id || clinician?._id),
    encounter.assignedPhysicianId,
    encounter.assignedNurseId
  ]).concat(includeClinicianId || []).filter(Boolean).map(String))];
  const staff = clinicianIds.length && mongoose.Types.ObjectId.isValid(hospitalId)
    ? await ClinicalStaff.find({ _id: { $in: clinicianIds }, hospitalId })
      .select('_id firstName lastName name systemRole role assignment workloadCapacity clinicalSpecialty')
      .lean()
    : [];
  const staffById = new Map(staff.map((member) => [String(member._id), member]));
  const context = encounters.map((encounter) => ({
    ...encounter,
    assigned_clinicians: (() => {
      const assigned = encounter.assigned_clinicians?.length
        ? [...encounter.assigned_clinicians]
        : [
            encounter.assignedPhysicianId && { id: encounter.assignedPhysicianId, name: encounter.assignedPhysicianName, role: 'PHYSICIAN' },
            encounter.assignedNurseId && { id: encounter.assignedNurseId, name: encounter.assignedNurseName, role: 'CLINICAL_NURSE' }
          ].filter(Boolean);
      const triageNurseId = encounter.nurse_id || encounter.triage?.nurse_id || encounter.triage?.triageNurseId;
      if (triageNurseId && !assigned.some((clinician) => String(clinician.id || clinician._id) === String(triageNurseId))) {
        assigned.push({ id: triageNurseId, name: encounter.triage?.nurse_name || encounter.triage?.triageNurseName, role: 'TRIAGE_NURSE' });
      }
      return assigned;
    })().map((assigned) => {
      const member = staffById.get(String(assigned.id || assigned._id));
      const role = normalizeRole(assigned.role || member?.systemRole || member?.role);
      return {
        ...assigned,
        id: String(assigned.id || assigned._id || member?._id),
        name: assigned.name || member?.name || [member?.firstName, member?.lastName].filter(Boolean).join(' '),
        role,
        department: assigned.department || member?.assignment?.department,
        workloadCapacity: assigned.workloadCapacity ?? member?.workloadCapacity
      };
    })
  }));
  return {
    encounters: context,
    roster: staff.map((member) => ({
      id: String(member._id),
      name: member.name || [member.firstName, member.lastName].filter(Boolean).join(' '),
      role: normalizeRole(member.systemRole || member.role),
      department: member.assignment?.department,
      workloadCapacity: member.workloadCapacity
    })),
    acuityWeightByPriority: hospital?.workloadPolicy?.acuityWeightByPriority || {},
    defaultCapacityByRole: hospital?.workloadPolicy?.defaultCapacityByRole || {},
    departmentShiftConfig: hospital?.workloadPolicy?.departmentShiftConfig || {}
  };
};

const severityForPriority = (priority) => ({
  RED: 'CRITICAL',
  ORANGE: 'HIGH',
  YELLOW: 'MEDIUM',
  GREEN: 'LOW'
}[String(priority || '').toUpperCase()] || 'INFO');

const patientFromEncounter = (encounter) => ({
  encounterId: String(encounter._id),
  mrn: encounter.patient?.id_number || null,
  name: encounter.patient?.name || null,
  chiefComplaint: encounter.patient?.symptoms || null
});

export const queuePriority = (encounter) => {
  const report = [encounter.manual_triage_report, encounter.ai_triage_report, encounter.triage]
    .find((candidate) => candidate && Object.keys(candidate).length > 0) || {};
  const nestedReport = [report.manualTriageReport, report.aiTriageReport, report.clinicalAnalysis]
    .find((candidate) => candidate && Object.keys(candidate).length > 0) || {};
  const rawCategory = report.priority_code || report.priorityCode || report.triageCategory || report.satsCategory || report.satsLevel
    || nestedReport.priority_code || nestedReport.priorityCode || nestedReport.triageCategory || nestedReport.satsCategory || nestedReport.satsLevel
    || nestedReport.selected_priority || nestedReport.sats_color || encounter.triageCategory;
  const explicitEsiLevel = Number(report.esi_level || report.esiLevel || report.ESI || nestedReport.esi_level || nestedReport.esiLevel || nestedReport.ESI);
  const esiLevel = Number.isInteger(explicitEsiLevel) && explicitEsiLevel >= 1 && explicitEsiLevel <= 4
    ? explicitEsiLevel
    : Number(encounter.severity);
  const normalizedCategory = String(rawCategory ?? '').trim().toLowerCase();
  const category = rawCategory
    ? (() => {
        const map = { red: 'RED', orange: 'ORANGE', yellow: 'YELLOW', green: 'GREEN', blue: 'GREEN', critical: 'RED', urgent: 'RED', moderate: 'YELLOW', low: 'GREEN' };
        const digits = Number.parseInt(String(rawCategory).replace(/[^0-9]/g, ''), 10);
        if (digits >= 1 && digits <= 4) return ({ 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' })[digits];
        return map[normalizedCategory] || String(rawCategory).toUpperCase();
      })()
    : Number.isInteger(esiLevel) && esiLevel >= 1 && esiLevel <= 4
    ? ({ 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' }[esiLevel])
    : '';
  const validCategory = ['RED', 'ORANGE', 'YELLOW', 'GREEN'].includes(category) ? category : null;
  const createdAt = new Date(encounter.waitStartedAt || encounter.arrivalAt || encounter.createdAt || encounter.date || Date.now());
  const waitMs = Math.max(0, Date.now() - createdAt.getTime());
  const thresholdMs = { RED: 0, ORANGE: 10 * 60 * 1000, YELLOW: 60 * 60 * 1000, GREEN: 240 * 60 * 1000 }[validCategory] ?? 60 * 60 * 1000;
  const escalationTriggered = waitMs >= thresholdMs;
  return {
    code: validCategory,
    score: null,
    waitMs,
    label: validCategory ? `${validCategory}${escalationTriggered ? ' - breached' : ''}` : 'Needs review',
    escalationTriggered,
    rank: { RED: 4, ORANGE: 3, YELLOW: 2, GREEN: 1 }[validCategory] || 0
  };
};

const queueEntry = (encounter) => {
  const priority = queuePriority(encounter);
  const assignedClinician = encounter.assigned_clinicians?.find(Boolean) || null;
  return {
    id: String(encounter._id),
    encounterId: String(encounter._id),
    patient: patientFromEncounter(encounter),
    priority,
    queueState: encounter.status === 'IN_PROGRESS' ? 'IN_PROGRESS' : 'WAITING',
    assignedClinician: assignedClinician ? {
      id: assignedClinician.id || assignedClinician._id || null,
      name: assignedClinician.name || null,
      role: assignedClinician.role || null
    } : null,
    signals: priority.escalationTriggered ? ['Time-to-target breached'] : []
  };
};

export const getClinicianWorkload = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    console.info('[WORKLOAD API] GET /api/operations/clinician-workload', {
      clinicianId: String(req.user?._id || req.user?.id || 'unknown'),
      role: req.user?.role || 'unknown',
      hospitalId: String(hospitalId || 'missing')
    });
    if (!hospitalId) return res.status(401).json({ success: false, message: 'Authenticated hospital is required' });

    const encounters = await Encounter.find(workloadQuery(hospitalId)).select(workloadSelect).lean();
    const context = await buildWorkloadContext(hospitalId, encounters, req.user?._id || req.user?.id);
    const workload = calculateClinicianWorkload(context.encounters, context);
    console.info('[WORKLOAD API] response', {
      encounters: encounters.length,
      clinicians: workload.clinicians.length,
      configured: workload.clinicians.filter((entry) => entry.workloadStatus === 'Configured').length,
      unconfigured: workload.clinicians.filter((entry) => entry.workloadStatus === 'Unconfigured').length,
      acuityWeights: context.acuityWeightByPriority,
      triageLane: workload.lanes.triageNurses.map((entry) => ({
        clinicianId: entry.clinicianId,
        activeCases: entry.activeCases,
        activeWorkload: entry.activeWorkload,
        acuityByPriority: entry.acuityByPriority
      }))
    });
    return res.json({
      success: true,
      ...workload,
      acuityWeightByPriority: context.acuityWeightByPriority,
      acuityWeightSource: Object.keys(context.acuityWeightByPriority || {}).length > 0
        ? 'hospital workload policy'
        : 'fallback 1.0x; hospital acuity policy is not configured',
      capacityPolicyConfigured: workload.clinicians.some((entry) => entry.capacity !== null)
    });
  } catch (error) {
    console.error('[CLINICIAN WORKLOAD] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load clinician workload' });
  }
};

export const getCapacityConfig = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    console.info('[WORKLOAD API] GET /api/operations/capacity-config', {
      clinicianId: String(req.user?._id || req.user?.id || 'unknown'),
      role: req.user?.role || 'unknown',
      hospitalId: String(hospitalId || 'missing')
    });
    if (!hospitalId) return res.status(401).json({ success: false, message: 'Authenticated hospital is required' });
    const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId)
      ? { _id: hospitalId }
      : { hospitalID: hospitalId };
    const hospital = await Hospital.findOne(hospitalQuery).select('workloadPolicy departments ehrConfig environment').lean();
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });

    const role = String(req.user?.role || '').toUpperCase();
    const canEdit = {
      clinician: ['ADMIN', 'SUPERUSER'].includes(role),
      department: ['ADMIN', 'SUPERUSER', 'MANAGER'].includes(role),
      hospital: ['ADMIN', 'SUPERUSER'].includes(role)
    };
    const staffQuery = { hospitalId };
    if (role === 'MANAGER' && req.user?.department) staffQuery['assignment.department'] = req.user.department;
    const clinicians = await ClinicalStaff.find(staffQuery)
      .select('_id firstName lastName name systemRole role assignment workloadCapacity clinicalSpecialty')
      .sort({ lastName: 1, firstName: 1 })
      .lean();

    console.info('[WORKLOAD API] capacity config response', {
      clinicianCount: clinicians.length,
      defaultRoles: Object.keys(hospital.workloadPolicy?.defaultCapacityByRole || {}),
      departmentDefaults: Object.keys(hospital.workloadPolicy?.departmentShiftConfig || {})
    });

    return res.json({
      success: true,
      canEdit,
      actor: { role, department: req.user?.department || null },
      departments: hospital.departments || [],
      policies: {
        acuityWeightByPriority: hospital.workloadPolicy?.acuityWeightByPriority || {},
        defaultCapacityByRole: hospital.workloadPolicy?.defaultCapacityByRole || {},
        departmentShiftConfig: hospital.workloadPolicy?.departmentShiftConfig || {}
      },
      clinicians: clinicians.map((clinician) => ({
        id: String(clinician._id),
        name: clinician.name || [clinician.firstName, clinician.lastName].filter(Boolean).join(' '),
        role: normalizeRole(clinician.systemRole || clinician.role),
        department: clinician.assignment?.department || null,
        specialty: clinician.clinicalSpecialty || null,
        workloadCapacity: clinician.workloadCapacity ?? null
      })),
      ehrConfig: normalizeEhrConfig(hospital.ehrConfig || {}, hospital.environment || 'SANDBOX')
    });
  } catch (error) {
    console.error('[CAPACITY CONFIG] Load error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load capacity configuration' });
  }
};

export const getEhrConfig = async (req, res) => {
  try {
    const hospitalId = resolveAuthenticatedEhrHospitalId(req);
    const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId) ? { _id: hospitalId } : { hospitalID: hospitalId };
    const hospital = await Hospital.findOne(hospitalQuery).select('ehrConfig environment').lean();
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    const config = normalizeEhrConfig(hospital.ehrConfig || DEFAULT_EHR_CONFIG, hospital.environment || 'SANDBOX');
    config.integrationAccess = resolveEhrIntegrationAccess(hospital.ehrConfig?.integrationAccess, hospital.environment);
    return res.json({ success: true, config });
  } catch (error) {
    console.error('[EHR CONFIG] Load error:', error);
    const statusCode = error?.statusCode || 500;
    return res.status(statusCode).json({ success: false, message: error.message || 'Failed to load EHR config' });
  }
};

export const updateEhrConfig = async (req, res) => {
  try {
    const hospitalId = resolveAuthenticatedEhrHospitalId(req);
    const actorRole = String(req.user?.role || '').toUpperCase();
    if (!['ADMIN', 'SUPERUSER', 'MANAGER'].includes(actorRole)) {
      return res.status(403).json({ success: false, message: 'Only administrators may edit EHR configuration' });
    }
    const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId) ? { _id: hospitalId } : { hospitalID: hospitalId };
    const hospital = await Hospital.findOne(hospitalQuery);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    if (hospital.environment && hospital.environment !== 'SANDBOX') {
      return res.status(403).json({ success: false, code: 'PRODUCTION_ONBOARDING_UNAVAILABLE', message: 'Production EHR onboarding requires a separately approved license workflow' });
    }

    const incoming = req.body?.config || req.body || {};
    const currentAccess = resolveEhrIntegrationAccess(hospital.ehrConfig?.integrationAccess, hospital.environment);
    const requestedAdapters = Array.isArray(incoming.adapters) ? incoming.adapters : [];
    for (const adapter of requestedAdapters) {
      if (!adapter || typeof adapter !== 'object') continue;
      if (!EHR_PROTOCOLS.includes(adapter.id)) continue;
      const hasSettings = adapter.id === 'hl7v2'
        ? Boolean(String(adapter.host || '').trim() || (adapter.sendingFacility || adapter.receivingFacility))
        : Boolean(String(adapter.endpoint || '').trim());
      if (hasSettings && !currentAccess.enabledProtocols.includes(adapter.id)) {
        return res.status(403).json({
          success: false,
          code: 'EHR_ACCESS_REQUIRED',
          message: `The account owner or hospital administrator must enable ${adapter.id} integration access before configuration`,
        });
      }
    }

    const existing = normalizeEhrConfig(hospital.ehrConfig || DEFAULT_EHR_CONFIG, hospital.environment || 'SANDBOX');
    const incomingById = new Map(requestedAdapters.filter((adapter) => adapter && typeof adapter === 'object').map((adapter) => [adapter.id, adapter]));
    const mergedAdapters = existing.adapters.map((adapter) => ({
      ...adapter,
      ...(incomingById.get(adapter.id) || {}),
    }));
    const mergedInput = {
      ...existing,
      ...incoming,
      adapters: mergedAdapters,
      integrationAccess: hospital.ehrConfig?.integrationAccess?.toObject?.() || hospital.ehrConfig?.integrationAccess || null,
    };
    const next = normalizeEhrConfig(mergedInput, hospital.environment || 'SANDBOX');
    next.integrationAccess = hospital.ehrConfig?.integrationAccess?.toObject?.() || hospital.ehrConfig?.integrationAccess || null;
    const validationError = validateEhrAdapters(next.adapters);
    if (validationError) return res.status(400).json({ success: false, code: 'INVALID_EHR_CONFIGURATION', message: validationError });

    hospital.ehrConfig = next;
    await hospital.save();

    await writeEhrAudit({
      action: 'EHR_CONFIG_CHANGED',
      actorId: req.user._id,
      actorIdentityType: 'MANAGER',
      hospitalId: hospital._id,
      resourceType: 'EHR_CONFIGURATION',
      authorized: true,
      changeDetails: {
        actor: req.user?.email || req.user?.name || 'unknown',
        hospitalId: String(hospital._id),
        timestamp: new Date().toISOString(),
        summary: 'EHR connection configuration updated',
        enabledProtocols: currentAccess.enabledProtocols,
      },
      description: 'Hospital EHR configuration was updated'
    });

    return res.json({ success: true, config: next });
  } catch (error) {
    console.error('[EHR CONFIG] Update error:', error);
    const statusCode = error?.statusCode || 500;
    return res.status(statusCode).json({ success: false, message: error.message || 'Failed to update EHR config' });
  }
};

export const testEhrAdapter = async (req, res) => {
  try {
    const hospitalId = resolveAuthenticatedEhrHospitalId(req);
    const actorRole = String(req.user?.role || '').toUpperCase();
    if (!['ADMIN', 'SUPERUSER', 'MANAGER'].includes(actorRole)) {
      return res.status(403).json({ success: false, message: 'Only administrators may test adapter connectivity' });
    }
    const { adapterId } = req.body || {};
    const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId) ? { _id: hospitalId } : { hospitalID: hospitalId };
    const hospital = await Hospital.findOne(hospitalQuery);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    if (hospital.environment && hospital.environment !== 'SANDBOX') {
      return res.status(403).json({ success: false, code: 'PRODUCTION_ONBOARDING_UNAVAILABLE', message: 'Production EHR onboarding requires a separately approved license workflow' });
    }

    if (!EHR_PROTOCOLS.includes(adapterId)) {
      return res.status(400).json({ success: false, code: 'INVALID_EHR_PROTOCOL', message: `Supported protocols are ${EHR_PROTOCOLS.join(' and ')}` });
    }
    if (!resolveEhrIntegrationAccess(hospital.ehrConfig?.integrationAccess, hospital.environment).enabledProtocols.includes(adapterId)) {
      return res.status(403).json({ success: false, code: 'EHR_ACCESS_REQUIRED', message: 'Integration access has not been enabled for this protocol' });
    }

    return res.status(501).json({
      success: false,
      code: 'CONNECTION_TEST_UNAVAILABLE',
      message: 'Connection settings are saved for QA, but no licensed external connector is configured to perform a live test.',
      adapterId,
    });
  } catch (error) {
    console.error('[EHR CONFIG] Test error:', error);
    const statusCode = error?.statusCode || 500;
    return res.status(statusCode).json({ success: false, message: error.message || 'Failed to test EHR adapter' });
  }
};

export const enableSandboxEhrAccess = async (req, res) => {
  try {
    const hospitalId = resolveAuthenticatedEhrHospitalId(req);
    const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId) ? { _id: hospitalId } : { hospitalID: hospitalId };
    const reason = String(req.body?.reason || '').trim();
    if (!reason) {
      return res.status(400).json({ success: false, code: 'REASON_REQUIRED', message: 'A reason is required to enable sandbox integration configuration' });
    }
    const hospital = await Hospital.findOne(hospitalQuery);
    if (!hospital) return res.status(404).json({ success: false, code: 'HOSPITAL_NOT_FOUND', message: 'Hospital not found' });
    if (hospital.environment && hospital.environment !== 'SANDBOX') {
      return res.status(403).json({ success: false, code: 'PRODUCTION_ONBOARDING_UNAVAILABLE', message: 'Production EHR onboarding requires a separately approved license workflow' });
    }
    if (hospital.ehrConfig?.integrationAccess?.source) {
      return res.status(409).json({ success: false, code: 'ACCOUNT_INTEGRATION_POLICY_EXISTS', message: 'The sandbox account already manages integration access for this hospital' });
    }

    const at = new Date();
    const integrationAccess = {
      source: 'HOSPITAL_ADMIN_SANDBOX',
      enabledProtocols: EHR_PROTOCOLS,
      attestedAt: at,
      attestedBy: req.user._id,
      attestedByName: req.user.name || req.user.email || 'Hospital administrator',
      reason,
    };
    const updatedHospital = await Hospital.findOneAndUpdate(
      {
        ...hospitalQuery,
        $or: [
          { environment: 'SANDBOX' },
          { environment: { $exists: false } },
        ],
        $and: [{
          $or: [
            { 'ehrConfig.integrationAccess.source': null },
            { 'ehrConfig.integrationAccess.source': { $exists: false } },
          ],
        }],
      },
      { $set: { 'ehrConfig.integrationAccess': integrationAccess } },
      { new: true, runValidators: true }
    );
    if (!updatedHospital) {
      return res.status(409).json({ success: false, code: 'ACCOUNT_INTEGRATION_POLICY_EXISTS', message: 'The sandbox account already manages integration access for this hospital' });
    }

    await writeEhrAudit({
      action: 'EHR_INTEGRATION_ACCESS_CHANGED',
      actorId: req.user._id,
      actorIdentityType: 'MANAGER',
      hospitalId: hospital._id,
      resourceType: 'EHR_INTEGRATION_ACCESS',
      authorized: true,
      reason,
      ipAddress: req.ip,
      userAgent: req.get('user-agent'),
      changeDetails: { source: integrationAccess.source, enabledProtocols: integrationAccess.enabledProtocols },
      description: 'Hospital administrator enabled sandbox-only EHR configuration',
    });

    return res.json({
      success: true,
      integrationAccess: resolveEhrIntegrationAccess(integrationAccess, hospital.environment || 'SANDBOX'),
      message: 'Sandbox configuration is enabled. This does not establish an external connection or confirm production licensing.',
    });
  } catch (error) {
    console.error('[EHR CONFIG] Sandbox access update failed:', error);
    const statusCode = error?.statusCode || 500;
    return res.status(statusCode).json({ success: false, message: error.message || 'Failed to enable sandbox EHR configuration' });
  }
};

export const updateCapacity = async (req, res) => {
  try {
    const hospitalId = resolveRequestHospitalId(req, req.body?.hospitalId);
    const actorRole = String(req.user?.role || '').toUpperCase();
    const { level, clinicianId, department, role, capacity } = req.body || {};
    const numeric = capacity === null || capacity === '' ? null : Number(capacity);
    if (numeric !== null && (!Number.isFinite(numeric) || numeric <= 0)) {
      return res.status(400).json({ success: false, message: 'Capacity must be a positive number or null' });
    }
    if (!hospitalId || !['clinician', 'department', 'hospital'].includes(level)) {
      return res.status(400).json({ success: false, message: 'Valid hospital and capacity level are required' });
    }
    const canEdit = level === 'clinician'
      ? ['ADMIN', 'SUPERUSER'].includes(actorRole)
      : level === 'department'
        ? ['ADMIN', 'SUPERUSER', 'MANAGER'].includes(actorRole)
        : ['ADMIN', 'SUPERUSER'].includes(actorRole);
    if (!canEdit) return res.status(403).json({ success: false, message: 'Insufficient role to edit this capacity level' });

    const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId)
      ? { _id: hospitalId }
      : { hospitalID: hospitalId };
    const hospital = await Hospital.findOne(hospitalQuery);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    let oldValue = null;
    let target = level;
    if (level === 'clinician') {
      const staff = await ClinicalStaff.findOne({ _id: clinicianId, hospitalId });
      if (!staff) return res.status(404).json({ success: false, message: 'Clinician not found in hospital' });
      oldValue = staff.workloadCapacity ?? null;
      staff.workloadCapacity = numeric;
      await staff.save();
      target = String(staff._id);
    } else if (level === 'department') {
      if (!department || !role) return res.status(400).json({ success: false, message: 'Department and role are required' });
      if (actorRole === 'MANAGER' && req.user?.department && req.user.department !== department) {
        return res.status(403).json({ success: false, message: 'Managers may only edit their department' });
      }
      const config = hospital.workloadPolicy?.departmentShiftConfig?.toObject?.() || hospital.workloadPolicy?.departmentShiftConfig || {};
      const next = { ...config, [department]: { ...(config[department] || {}), capacityByRole: { ...(config[department]?.capacityByRole || {}), [normalizeRole(role)]: numeric } } };
      oldValue = config[department]?.capacityByRole?.[normalizeRole(role)] ?? null;
      hospital.workloadPolicy.departmentShiftConfig = next;
      await hospital.save();
      target = `${department}:${normalizeRole(role)}`;
    } else {
      const defaults = hospital.workloadPolicy?.defaultCapacityByRole?.toObject?.() || hospital.workloadPolicy?.defaultCapacityByRole || {};
      const normalizedRole = normalizeRole(role);
      if (!normalizedRole) return res.status(400).json({ success: false, message: 'Role is required' });
      oldValue = defaults[normalizedRole] ?? null;
      hospital.workloadPolicy.defaultCapacityByRole = { ...defaults, [normalizedRole]: numeric };
      await hospital.save();
      target = normalizedRole;
    }

    await AuditLog.create({
      action: 'CAPACITY_CHANGED',
      actorId: req.user._id,
      hospitalId: hospital._id,
      resourceType: 'WORKLOAD_CAPACITY',
      authorized: true,
      changeDetails: buildCapacityAuditDetails({
        actor: req.user?.email || req.user?.name || 'unknown',
        hospitalId: hospital._id,
        level,
        target,
        role,
        oldValue,
        newValue: numeric
      }),
      description: `Workload capacity changed at ${level} level for ${target}`
    });
    return res.json({ success: true, level, target, oldValue, capacity: numeric });
  } catch (error) {
    console.error('[CAPACITY UPDATE] Error:', error);
    const statusCode = error?.statusCode || 500;
    return res.status(statusCode).json({ success: false, message: error.message || 'Failed to update workload capacity' });
  }
};

export const updateAcuityWeights = async (req, res) => {
  try {
    const hospitalId = resolveRequestHospitalId(req, req.body?.hospitalId);
    const actorRole = String(req.user?.role || '').toUpperCase();
    if (!hospitalId || !['ADMIN', 'SUPERUSER'].includes(actorRole)) {
      return res.status(403).json({ success: false, message: 'Only hospital administrators may edit acuity weights' });
    }
    const weights = req.body?.weights || {};
    const validation = validateAcuityWeightOrdering(weights);
    if (!validation.valid) {
      return res.status(400).json({ success: false, message: validation.message });
    }
    const next = validation.normalized;
    const hospitalQuery = mongoose.Types.ObjectId.isValid(hospitalId) ? { _id: hospitalId } : { hospitalID: hospitalId };
    const hospital = await Hospital.findOne(hospitalQuery);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    const oldWeights = hospital.workloadPolicy?.acuityWeightByPriority?.toObject?.() || hospital.workloadPolicy?.acuityWeightByPriority || {};
    hospital.workloadPolicy.acuityWeightByPriority = next;
    await hospital.save();
    await AuditLog.create({
      action: 'CAPACITY_CHANGED', actorId: req.user._id, hospitalId: hospital._id,
      resourceType: 'ACUITY_WEIGHT_POLICY', authorized: true,
      changeDetails: buildCapacityAuditDetails({
        actor: req.user?.email || req.user?.name || 'unknown',
        hospitalId: hospital._id,
        level: 'hospital',
        target: 'ALL_ROLES',
        role: 'HOSPITAL_DEFAULT',
        oldValue: oldWeights,
        newValue: next
      }),
      description: 'Hospital acuity workload weights changed'
    });
    return res.json({ success: true, weights: next });
  } catch (error) {
    console.error('[ACUITY WEIGHT UPDATE] Error:', error);
    const statusCode = error?.statusCode || 500;
    return res.status(statusCode).json({ success: false, message: error.message || 'Failed to update acuity weights' });
  }
};

const normalizeNotification = (notification) => ({
  id: String(notification._id),
  type: notification.type || 'HANDOVER_PENDING',
  category: notification.category || 'ALERT',
  severity: severityForPriority(notification.priority_code),
  status: notification.status,
  title: notification.title || 'Clinical handover pending',
  message: notification.message || notification.chief_complaint || 'A clinical update requires attention.',
  source: notification.source || 'HANDOVER',
  patient: notification.encounter_id ? {
    encounterId: String(notification.encounter_id._id || notification.encounter_id),
    mrn: notification.patient_mrn || null,
    name: notification.patient_name || null,
    chiefComplaint: notification.chief_complaint || null
  } : null,
  priority: notification.priority_code ? {
    code: notification.priority_code,
    score: null,
    waitMs: null,
    label: notification.priority_code
  } : null,
  metadata: notification.metadata || {},
  action: notification.action_route ? { label: notification.action_label || 'Open', route: notification.action_route } : null,
  createdAt: new Date(notification.created_at || notification.createdAt).toISOString(),
  updatedAt: new Date(notification.updatedAt || notification.created_at || notification.createdAt).toISOString()
});

const normalizeSurgery = (encounter, surgery) => {
  const priority = String(surgery.priority || 'MEDIUM').toUpperCase();
  const type = surgery.surgery_type || 'ELECTIVE';
  const isUrgent = type === 'EMERGENCY' || type === 'URGENT' || priority === 'URGENT';
  const eventType = surgery.status === 'CANCELLED' ? 'SURGERY_CANCELLED' : surgery.status === 'SCHEDULED' ? 'SURGERY_SCHEDULED' : 'SURGERY_STATUS_CHANGED';
  const timestamp = surgery.updated_at || surgery.recorded_at || surgery.requested_at || new Date();

  return {
    id: `surgery:${encounter._id}:${surgery.surgery_id || surgery._id}`,
    type: isUrgent && surgery.status !== 'CANCELLED' ? 'URGENT_SURGERY' : eventType,
    category: 'ALERT',
    severity: isUrgent ? 'HIGH' : 'INFO',
    status: 'UNREAD',
    title: isUrgent ? 'Urgent surgery requires coordination' : `Surgery ${String(surgery.status || 'updated').toLowerCase()}`,
    message: `${surgery.procedure_name || 'Procedure'} for ${encounter.patient?.name || 'patient'}${surgery.scheduled_at ? ` at ${new Date(surgery.scheduled_at).toLocaleString()}` : ''}.`,
    source: 'SURGERY',
    patient: patientFromEncounter(encounter),
    priority: null,
    metadata: {
      surgeryId: String(surgery.surgery_id || surgery._id),
      surgeryType: type,
      surgicalPriority: priority,
      status: surgery.status || 'PENDING',
      surgeon: surgery.surgeon || null,
      theatre: surgery.theatre || null
    },
    action: { label: 'Open surgery', route: `/scheduled-surgeries?encounterId=${encounter._id}` },
    createdAt: new Date(timestamp).toISOString(),
    updatedAt: new Date(timestamp).toISOString()
  };
};

export const getOperationsFeed = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    const clinicianId = req.user?._id || req.user?.id;
    const limit = Math.min(Math.max(Number.parseInt(req.query.limit, 10) || 50, 1), 100);

    if (!hospitalId || !clinicianId) {
      return res.status(401).json({ success: false, message: 'Authenticated hospital and clinician are required' });
    }

    const [notifications, surgeryEncounters, workloadEncounters, waitingEncounters] = await Promise.all([
      Notification.find({ clinician_id: clinicianId, hospital_id: hospitalId, status: 'UNREAD' })
        .sort({ created_at: -1 })
        .limit(limit)
        .lean(),
      Encounter.find({ hospitalId, 'surgery_records.0': { $exists: true } })
        .select('patient surgery_records')
        .lean(),
      Encounter.find(workloadQuery(hospitalId))
        .select(workloadSelect + ' patient ai_triage_report manual_triage_report triage triageCategory waitStartedAt arrivalAt createdAt date')
        .lean(),
      Encounter.find({
        hospitalId,
        specialistAttended: { $ne: true },
        $or: [{ isWaiting: true }, { status: 'TRIAGE_IN_PROGRESS' }]
      })
        .select('patient ai_triage_report manual_triage_report triage triageCategory waitStartedAt arrivalAt createdAt date status isWaiting specialistAttended assigned_clinicians assignedPhysicianId assignedPhysicianName assignedNurseId assignedNurseName')
        .lean()
    ]);

    const normalizedNotifications = notifications.map(normalizeNotification);

    const orderedQueueLane = waitingEncounters
      .map(queueEntry)
      .sort((left, right) => right.priority.rank - left.priority.rank || Number(right.priority.escalationTriggered) - Number(left.priority.escalationTriggered) || right.priority.waitMs - left.priority.waitMs || left.id.localeCompare(right.id));

    const workloadContext = await buildWorkloadContext(hospitalId, workloadEncounters);
    const workload = calculateClinicianWorkload(workloadContext.encounters, workloadContext);
    const providers = workload.clinicians;

    const dataIntegrityAlerts = orderedQueueLane
      .filter((entry) => !entry.priority.code)
      .map((entry) => ({
        id: `data-integrity:${entry.encounterId}`,
        type: 'DATA_INTEGRITY_ALERT',
        category: 'ALERT',
        severity: 'CRITICAL',
        status: 'UNREAD',
        title: 'Triage data needs review',
        message: `Triage category could not be validated for ${entry.patient.name || 'this patient'}. Manual re-triage is required.`,
        source: 'PRIORITY_QUEUE',
        patient: { ...entry.patient, chiefComplaint: null },
        priority: { code: null, score: null, waitMs: entry.priority.waitMs, label: 'Needs review' },
        metadata: { reason: 'Missing or invalid triage category', detectedAt: new Date().toISOString() },
        action: { label: 'Re-triage now', route: `/triage/re-assess/${entry.encounterId}` },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }));

    const queueAlerts = orderedQueueLane
      .filter((entry) => entry.priority.code === 'RED' || (entry.priority.code && entry.priority.escalationTriggered))
      .map((entry) => ({
        id: `queue:${entry.encounterId}`,
        type: entry.priority.code === 'RED' ? 'CRITICAL_TRIAGE' : 'PRIORITY_ESCALATED',
        category: 'ALERT',
        severity: entry.priority.code === 'RED' ? 'CRITICAL' : 'HIGH',
        status: 'UNREAD',
        title: entry.priority.code === 'RED' ? 'Critical triage requires attention' : 'Priority queue escalation',
        message: entry.priority.escalationTriggered
          ? `${entry.patient.name || 'Patient'} has breached the ${entry.priority.code} wait threshold.`
          : `${entry.patient.name || 'Patient'} is ${entry.priority.code} priority and requires attention.`,
        source: 'PRIORITY_QUEUE',
        patient: entry.patient,
        priority: { code: entry.priority.code, score: null, waitMs: entry.priority.waitMs, label: entry.priority.label },
        metadata: { signals: entry.signals },
        action: { label: 'Open queue', route: '/dashboard' },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }));

    const capacityAlerts = buildCapacityAlerts(workload.clinicians);

    const surgeryEvents = surgeryEncounters
      .flatMap((encounter) => (encounter.surgery_records || [])
        .filter((surgery) => canViewSurgery(encounter, surgery, req.user))
        .map((surgery) => normalizeSurgery(encounter, surgery)));

    const alerts = sortOperationsEvents([...normalizedNotifications, ...surgeryEvents, ...queueAlerts, ...capacityAlerts, ...dataIntegrityAlerts]).slice(0, limit);

    return res.json({
      success: true,
      serverTime: new Date().toISOString(),
      cursor: null,
      unreadNotifications: normalizedNotifications,
      feed: {
        alerts,
        messages: normalizedNotifications.slice(0, Math.max(1, Math.min(limit, 8))),
        queue: { waiting: orderedQueueLane.length, patients: orderedQueueLane, providers, workload: workload.lanes, engineReady: true },
        labs: [],
        prescriptions: [],
        surgeries: surgeryEvents
      }
    });
  } catch (error) {
    console.error('[OPERATIONS FEED] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load operations feed' });
  }
};
