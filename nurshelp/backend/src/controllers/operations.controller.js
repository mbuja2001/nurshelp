import { Notification } from '../models/notification.model.js';
import { Encounter } from '../models/encounter.model.js';
import { canViewSurgery, sortOperationsEvents } from '../utils/operationsFeed.js';

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

const queuePriority = (encounter) => {
  const report = [encounter.manual_triage_report, encounter.ai_triage_report, encounter.triage]
    .find((candidate) => candidate && Object.keys(candidate).length > 0) || {};
  const rawCategory = report.triageCategory || report.satsCategory || report.satsLevel || encounter.triageCategory;
  const esiLevel = Number(report.esi_level || report.ESI || encounter.severity);
  const category = Number.isInteger(esiLevel) && esiLevel >= 1 && esiLevel <= 4
    ? ({ 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' }[esiLevel])
    : String(rawCategory || '').toUpperCase();
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

    const [notifications, surgeryEncounters, waitingEncounters] = await Promise.all([
      Notification.find({ clinician_id: clinicianId, hospital_id: hospitalId, status: 'UNREAD' })
        .sort({ created_at: -1 })
        .limit(limit)
        .lean(),
      Encounter.find({ hospitalId, 'surgery_records.0': { $exists: true } })
        .select('patient surgery_records')
        .lean(),
      Encounter.find({
        hospitalId,
        specialistAttended: { $ne: true },
        $or: [{ isWaiting: true }, { status: 'TRIAGE_IN_PROGRESS' }]
      })
        .select('patient ai_triage_report manual_triage_report triage triageCategory waitStartedAt arrivalAt createdAt date status assigned_clinicians')
        .lean()
    ]);

    const normalizedNotifications = notifications.map(normalizeNotification);

    const orderedQueueLane = waitingEncounters
      .map(queueEntry)
      .sort((left, right) => right.priority.rank - left.priority.rank || Number(right.priority.escalationTriggered) - Number(left.priority.escalationTriggered) || right.priority.waitMs - left.priority.waitMs || left.id.localeCompare(right.id));

    const providerCounts = new Map();
    orderedQueueLane.forEach((entry) => {
      const clinician = entry.assignedClinician;
      const key = clinician?.id || clinician?.name || 'unassigned';
      const current = providerCounts.get(key) || {
        id: clinician?.id || null,
        name: clinician?.name || 'Unassigned',
        role: clinician?.role || null,
        activeCases: 0
      };
      current.activeCases += 1;
      providerCounts.set(key, current);
    });
    const maxProviderCases = Math.max(1, ...[...providerCounts.values()].map((provider) => provider.activeCases));
    const providers = [...providerCounts.values()].map((provider) => ({
      ...provider,
      workloadPercent: Math.min(100, Math.round((provider.activeCases / maxProviderCases) * 100))
    })).sort((left, right) => right.activeCases - left.activeCases);

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

    const surgeryEvents = surgeryEncounters
      .flatMap((encounter) => (encounter.surgery_records || [])
        .filter((surgery) => canViewSurgery(encounter, surgery, req.user))
        .map((surgery) => normalizeSurgery(encounter, surgery)));

    const alerts = sortOperationsEvents([...normalizedNotifications, ...surgeryEvents, ...queueAlerts, ...dataIntegrityAlerts]).slice(0, limit);

    return res.json({
      success: true,
      serverTime: new Date().toISOString(),
      cursor: null,
      unreadNotifications: normalizedNotifications,
      feed: {
        alerts,
        messages: normalizedNotifications.slice(0, Math.max(1, Math.min(limit, 8))),
        queue: { waiting: orderedQueueLane.length, patients: orderedQueueLane, providers, engineReady: true },
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
