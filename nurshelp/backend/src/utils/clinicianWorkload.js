const ROLE_LANES = Object.freeze({
  TRIAGE_NURSE: 'triageNurses',
  CLINICAL_NURSE: 'clinicalNurses',
  PHYSICIAN: 'physicians'
});

const normalizeRole = (role) => {
  const value = String(role || '').toUpperCase().trim();
  if (['TRIAGE', 'TRIAGE_NURSE', 'TRIAGE_NURSE_ER'].includes(value)) return 'TRIAGE_NURSE';
  if (['NURSE', 'CLINICAL_NURSE', 'CLINICAL_NURSE_ER', 'CLINICAL_NURSE_WARD'].includes(value)) return 'CLINICAL_NURSE';
  if (['DOCTOR', 'SURGEON', 'SPECIALIST', 'PHYSICIAN', 'PHYSICIAN_ER'].includes(value)) return 'PHYSICIAN';
  return value;
};

const stableId = (clinician) => {
  const value = clinician?.id || clinician?._id || clinician?.clinicianId || clinician?.staffId || clinician?.physicianId;
  return value ? String(value) : null;
};

const numericCapacity = (value) => {
  const capacity = Number(value);
  return Number.isFinite(capacity) && capacity > 0 ? capacity : null;
};

const acuityCode = (encounter) => {
  const report = [encounter?.manual_triage_report, encounter?.ai_triage_report, encounter?.triage]
    .find((candidate) => candidate && Object.keys(candidate).length > 0) || {};
  const nestedReport = [report.manualTriageReport, report.aiTriageReport, report.clinicalAnalysis]
    .find((candidate) => candidate && Object.keys(candidate).length > 0) || {};
  const raw = report.priority_code || report.priorityCode || report.triageCategory || report.satsCategory || report.satsLevel
    || nestedReport.priority_code || nestedReport.priorityCode || nestedReport.triageCategory || nestedReport.satsCategory || nestedReport.satsLevel
    || nestedReport.selected_priority || nestedReport.sats_color || encounter?.triageCategory;
  const explicitEsi = Number(report.esi_level || report.esiLevel || report.ESI || nestedReport.esi_level || nestedReport.esiLevel || nestedReport.ESI);
  const esi = Number.isInteger(explicitEsi) && explicitEsi >= 1 && explicitEsi <= 4
    ? explicitEsi
    : Number(encounter?.severity);
  if (raw !== undefined && raw !== null && raw !== '') {
    const normalized = String(raw).trim().toLowerCase();
    const mapped = {
      red: 'RED',
      orange: 'ORANGE',
      yellow: 'YELLOW',
      green: 'GREEN',
      blue: 'GREEN',
      critical: 'RED',
      urgent: 'RED',
      moderate: 'YELLOW',
      low: 'GREEN'
    };
    const numeric = Number.parseInt(String(raw).replace(/[^0-9]/g, ''), 10);
    if (numeric >= 1 && numeric <= 4) return ({ 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' })[numeric];
    if (mapped[normalized]) return mapped[normalized];
  }
  if (esi >= 1 && esi <= 4) return ({ 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' })[esi];

  const normalized = String(raw ?? '').trim().toLowerCase();
  const mapped = {
    red: 'RED',
    orange: 'ORANGE',
    yellow: 'YELLOW',
    green: 'GREEN',
    blue: 'GREEN',
    critical: 'RED',
    urgent: 'RED',
    moderate: 'YELLOW',
    low: 'GREEN'
  };

  const numeric = Number.parseInt(String(raw ?? '').replace(/[^0-9]/g, ''), 10);
  if (numeric >= 1 && numeric <= 4) return ({ 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' })[numeric];
  return mapped[normalized] || String(raw || '').toUpperCase();
};

const workloadWeight = (encounter, options) => {
  const code = acuityCode(encounter);
  const configured = Number(options.acuityWeightByPriority?.[code]);
  console.info('[WORKLOAD ACUITY]', {
    encounterId: String(encounter?._id || encounter?.id || 'unknown'),
    rawPriority: encounter?.triage?.manualTriageReport?.satsLevel
      || encounter?.triage?.manualTriageReport?.selected_priority
      || encounter?.triage?.aiTriageReport?.priority_code
      || encounter?.manual_triage_report?.satsLevel
      || null,
    resolvedCode: code || 'UNKNOWN',
    configuredWeight: Number.isFinite(configured) && configured > 0 ? configured : null,
    effectiveWeight: Number.isFinite(configured) && configured > 0 ? configured : 1
  });
  return {
    code: ['RED', 'ORANGE', 'YELLOW', 'GREEN'].includes(code) ? code : null,
    weight: Number.isFinite(configured) && configured > 0 ? configured : 1
  };
};

export const validateAcuityWeightOrdering = (weights = {}) => {
  const ordered = ['RED', 'ORANGE', 'YELLOW', 'GREEN'];
  const normalized = {};
  for (const priority of ordered) {
    const value = Number(weights?.[priority]);
    if (!Number.isFinite(value) || value <= 0) {
      return {
        valid: false,
        message: `${priority} acuity weight must be a positive number.`
      };
    }
    normalized[priority] = value;
  }

  const descending = ordered.every((priority, index, list) => {
    if (index === list.length - 1) return true;
    return normalized[priority] >= normalized[list[index + 1]];
  });

  if (!descending) {
    return {
      valid: false,
      message: 'Acuity weights must follow RED >= ORANGE >= YELLOW >= GREEN.'
    };
  }

  return { valid: true, normalized };
};

export const resolveCapacity = (clinician, options = {}) => {
  const individual = numericCapacity(
    clinician?.workloadCapacity ?? clinician?.capacity ?? clinician?.maxPatients
  );
  if (individual !== null) return { value: individual, source: 'clinician override' };

  const department = clinician?.department || clinician?.assignment?.department;
  const role = normalizeRole(clinician?.role || clinician?.profession || clinician?.type);
  const departmentConfig = options.departmentShiftConfig?.[department];
  const departmentCapacity = numericCapacity(departmentConfig?.capacityByRole?.[role]);
  if (departmentCapacity !== null) return { value: departmentCapacity, source: 'department default' };

  const hospitalCapacity = numericCapacity(options.defaultCapacityByRole?.[role]);
  if (hospitalCapacity !== null) return { value: hospitalCapacity, source: 'hospital default' };

  return { value: null, source: 'Unconfigured' };
};

const statusForRole = (role, encounter) => {
  const status = String(encounter?.status || '').toUpperCase();
  const waiting = Boolean(encounter?.isWaiting);
  if (status === 'DISCHARGED' || encounter?.specialistAttended === true) return false;
  if (role === 'TRIAGE_NURSE') return waiting || status === 'TRIAGE_IN_PROGRESS';
  if (role === 'CLINICAL_NURSE') return ['ACKNOWLEDGED', 'IN_PROGRESS', 'DIAGNOSED', 'SEEN', 'HOSPITALIZED'].includes(status);
  if (role === 'PHYSICIAN') return ['AWAITING_PHYSICIAN', 'ACKNOWLEDGED', 'IN_PROGRESS', 'DIAGNOSED', 'SEEN'].includes(status);
  return false;
};

const assignedClinicians = (encounter) => {
  const explicit = Array.isArray(encounter?.assigned_clinicians) ? encounter.assigned_clinicians : [];
  const fallback = [
    encounter?.assignedPhysicianId && { id: encounter.assignedPhysicianId, name: encounter.assignedPhysicianName, role: 'PHYSICIAN' },
    encounter?.assignedNurseId && { id: encounter.assignedNurseId, name: encounter.assignedNurseName, role: 'CLINICAL_NURSE' }
  ].filter(Boolean);
  const assigned = explicit.length ? [...explicit] : fallback;
  const triageNurseId = encounter?.nurse_id || encounter?.triage?.nurse_id || encounter?.triage?.triageNurseId;
  if (triageNurseId && !assigned.some((clinician) => String(clinician.id || clinician._id) === String(triageNurseId))) {
    assigned.push({ id: triageNurseId, name: encounter?.triage?.nurse_name || encounter?.triage?.triageNurseName, role: 'TRIAGE_NURSE' });
  }
  return assigned;
};

const emptyLanes = () => ({ triageNurses: [], clinicalNurses: [], physicians: [] });

export const calculateClinicianWorkload = (encounters = [], options = {}) => {
  const byLane = new Map(Object.keys(ROLE_LANES).map((role) => [role, new Map()]));

  (options.roster || []).forEach((clinician) => {
    const id = stableId(clinician);
    const role = normalizeRole(clinician?.role || clinician?.profession || clinician?.type);
    if (!id || !ROLE_LANES[role]) return;
    const capacity = resolveCapacity(clinician, options);
    byLane.get(role).set(id, {
      clinicianId: id,
      name: clinician.name || clinician.displayName || 'Unknown clinician',
      role,
      activeCases: 0,
      activeWorkload: 0,
      acuityByPriority: {},
      capacity: capacity.value,
      capacitySource: capacity.source,
      encounterIds: []
    });
  });

  encounters.forEach((encounter) => {
    const seenInEncounter = new Set();
    assignedClinicians(encounter).forEach((clinician) => {
      const id = stableId(clinician);
      const role = normalizeRole(clinician?.role || clinician?.profession || clinician?.type);
      if (!id || !ROLE_LANES[role]) return;
      const uniqueKey = `${role}:${id}`;
      if (seenInEncounter.has(uniqueKey) || !statusForRole(role, encounter)) return;
      seenInEncounter.add(uniqueKey);
      const lane = byLane.get(role);
      const capacity = resolveCapacity(clinician, options);
      const acuity = workloadWeight(encounter, options);
      const current = lane.get(id) || {
        clinicianId: id,
        name: clinician.name || clinician.displayName || 'Unknown clinician',
        role,
        activeCases: 0,
        activeWorkload: 0,
        acuityByPriority: {},
        capacity: capacity.value,
        capacitySource: capacity.source,
        encounterIds: []
      };
      current.activeCases += 1;
      current.activeWorkload += acuity.weight;
      current.acuityByPriority[acuity.code || 'UNKNOWN'] = (current.acuityByPriority[acuity.code || 'UNKNOWN'] || 0) + 1;
      current.encounterIds.push(String(encounter._id || encounter.id));
      lane.set(id, current);
    });
  });

  const lanes = emptyLanes();
  Object.entries(ROLE_LANES).forEach(([role, laneName]) => {
    const entries = [...byLane.get(role).values()];
    const maxCases = Math.max(1, ...entries.map((entry) => entry.activeCases));
    lanes[laneName] = entries
      .map((entry) => ({
        ...entry,
        workloadScore: entry.activeWorkload,
        activeCaseShare: Math.round((entry.activeCases / maxCases) * 100),
        workloadPercent: entry.capacity === null
          ? null
          : Math.min(100, Math.round((entry.activeWorkload / entry.capacity) * 100)),
        workloadStatus: entry.capacity === null ? 'Unconfigured' : 'Configured'
      }))
      .sort((left, right) => right.activeCases - left.activeCases || left.name.localeCompare(right.name));
  });

  const clinicians = Object.values(lanes).flat();
  return {
    clinicians,
    lanes,
    totals: {
      activeCases: new Set(clinicians.flatMap((entry) => entry.encounterIds)).size,
      clinicians: clinicians.length
    }
  };
};

export const workloadQuery = (hospitalId) => ({
  hospitalId,
  specialistAttended: { $ne: true },
  status: { $ne: 'DISCHARGED' },
  $or: [{ isWaiting: true }, { status: { $in: ['TRIAGE_IN_PROGRESS', 'AWAITING_PHYSICIAN', 'ACKNOWLEDGED', 'IN_PROGRESS', 'DIAGNOSED', 'SEEN', 'HOSPITALIZED'] } }]
});

export const buildCapacityAlerts = (clinicians = []) => clinicians.flatMap((clinician) => {
  const base = {
    source: 'CLINICIAN_WORKLOAD',
    patient: null,
    priority: null,
    metadata: { clinicianId: clinician.clinicianId, role: clinician.role },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };
  if (clinician.workloadStatus === 'Unconfigured') {
    return [{
      ...base,
      id: `capacity-unconfigured:${clinician.clinicianId}`,
      type: 'CAPACITY_UNCONFIGURED',
      category: 'HOUSEKEEPING',
      severity: 'LOW',
      status: 'UNREAD',
      title: 'Clinician capacity is unconfigured',
      message: `${clinician.name} has ${clinician.activeCases} active case(s) but no configured capacity.`,
      metadata: { ...base.metadata, capacitySource: clinician.capacitySource },
      action: { label: 'Configure capacity', route: '/staff-status' }
    }];
  }
  if (clinician.workloadPercent >= 100) {
    return [{
      ...base,
      id: `queue-congestion:${clinician.clinicianId}`,
      type: 'QUEUE_CONGESTION',
      category: 'ALERT',
      severity: 'HIGH',
      status: 'UNREAD',
      title: 'Clinician workload capacity reached',
      message: `${clinician.name} is at ${clinician.workloadPercent}% of configured capacity.`,
      metadata: { ...base.metadata, capacity: clinician.capacity, capacitySource: clinician.capacitySource },
      action: { label: 'Open queue', route: '/dashboard' }
    }];
  }
  return [];
});

export { ROLE_LANES, normalizeRole, stableId, statusForRole };
