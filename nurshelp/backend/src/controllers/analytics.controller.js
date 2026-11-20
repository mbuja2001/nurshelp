import { Encounter } from '../models/encounter.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { StaffClockRecord } from '../models/staffClockRecord.model.js';
import { AuditLog } from '../models/auditLog.model.js';

const toTitleCase = (value) => {
  const text = String(value || '').trim();
  if (!text) return 'Unspecified';
  return text
    .replace(/[_-]+/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
};

const normalizePriority = (value) => {
  if (value === null || value === undefined || value === '') return null;

  const raw = String(value).trim();
  const upper = raw.toUpperCase();

  if (['RED', 'CRITICAL', '1', 'EMERGENT', 'URGENT'].includes(upper)) return 'RED';
  if (['ORANGE', 'HIGH', '2', 'SEVERE'].includes(upper)) return 'ORANGE';
  if (['YELLOW', 'MEDIUM', '3', 'MODERATE'].includes(upper)) return 'YELLOW';
  if (['GREEN', 'LOW', '4', 'MINOR', 'STABLE'].includes(upper)) return 'GREEN';

  if (/^1(\.|\s|$)/.test(raw)) return 'RED';
  if (/^2(\.|\s|$)/.test(raw)) return 'ORANGE';
  if (/^3(\.|\s|$)/.test(raw)) return 'YELLOW';
  if (/^4(\.|\s|$)/.test(raw)) return 'GREEN';

  if (upper.includes('RED')) return 'RED';
  if (upper.includes('ORANGE')) return 'ORANGE';
  if (upper.includes('YELLOW')) return 'YELLOW';
  if (upper.includes('GREEN')) return 'GREEN';

  return null;
};

const getPriorityLabel = (encounter) => {
  const candidates = [
    { priorityCode: encounter?.physicianFinalUrgency || encounter?.finalUrgency },
    encounter?.manual_triage_report,
    encounter?.ai_triage_report,
    encounter?.triage,
    encounter?.manualTriageReport,
    encounter?.aiTriageReport,
  ];

  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== 'object') continue;

    const nested = candidate?.manualTriageReport || candidate?.aiTriageReport || candidate?.clinicalAnalysis || {};
    const rawPriority =
      candidate?.priority_code ||
      candidate?.priorityCode ||
      candidate?.triageCategory ||
      candidate?.satsCategory ||
      candidate?.satsLevel ||
      candidate?.level ||
      nested?.priority_code ||
      nested?.priorityCode ||
      nested?.triageCategory ||
      nested?.satsCategory ||
      nested?.satsLevel ||
      nested?.level;

    if (rawPriority) {
      return normalizePriority(rawPriority);
    }
  }

  return null;
};

const getDiagnosisLabel = (encounter) => {
  const candidates = [
    encounter?.diagnosis,
    encounter?.manual_triage_report?.diagnosis,
    encounter?.ai_triage_report?.diagnosis,
    encounter?.triage?.diagnosis,
  ];

  for (const candidate of candidates) {
    const value = String(candidate || '').trim();
    if (value) return toTitleCase(value);
  }

  return null;
};

const getWaitMinutes = (encounter) => {
  const anchor = encounter?.waitStartedAt || encounter?.arrivalAt || encounter?.date || encounter?.createdAt;
  if (!anchor) return null;

  const created = new Date(anchor).getTime();
  const now = Date.now();
  if (!Number.isFinite(created)) return null;
  return Math.max(0, Math.round((now - created) / 60000));
};

const getRange = (range = 'today') => {
  const now = new Date();
  const normalized = String(range).toLowerCase();

  if (normalized === 'week' || normalized === '7d' || normalized === 'last7d') {
    const start = new Date(now);
    start.setDate(start.getDate() - 6);
    start.setHours(0, 0, 0, 0);
    return { start, end: now };
  }

  if (normalized === 'month' || normalized === '30d' || normalized === 'last30d') {
    const start = new Date(now);
    start.setDate(start.getDate() - 29);
    start.setHours(0, 0, 0, 0);
    return { start, end: now };
  }

  if (normalized === 'yesterday') {
    const start = new Date(now);
    const end = new Date(now);
    start.setDate(start.getDate() - 1);
    start.setHours(0, 0, 0, 0);
    end.setDate(end.getDate() - 1);
    end.setHours(23, 59, 59, 999);
    return { start, end };
  }

  if (normalized === 'lastweek' || normalized === 'previousweek') {
    const start = new Date(now);
    const end = new Date(now);
    start.setDate(start.getDate() - 13);
    start.setHours(0, 0, 0, 0);
    end.setDate(end.getDate() - 7);
    end.setHours(23, 59, 59, 999);
    return { start, end };
  }

  if (normalized === 'lastmonth' || normalized === 'previousmonth') {
    const start = new Date(now);
    const end = new Date(now);
    start.setDate(start.getDate() - 59);
    start.setHours(0, 0, 0, 0);
    end.setDate(end.getDate() - 30);
    end.setHours(23, 59, 59, 999);
    return { start, end };
  }

  if (normalized === 'all') {
    return { start: new Date('2020-01-01T00:00:00.000Z'), end: now };
  }

  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  return { start, end: now };
};

const buildCaseReview = (encounter) => {
  const aiUrgency = getPriorityLabel({
    ai_triage_report: encounter?.ai_triage_report,
    triage: encounter?.triage,
  });

  const nurseUrgency = getPriorityLabel({
    manual_triage_report: encounter?.manual_triage_report,
  });

  const physicianUrgency = normalizePriority(encounter?.physicianFinalUrgency || encounter?.finalUrgency);
  if (!aiUrgency || !nurseUrgency) return null;

  const aiDiagnosis = getDiagnosisLabel({
    ai_triage_report: encounter?.ai_triage_report,
    triage: encounter?.triage,
  });
  const nurseDiagnosis = getDiagnosisLabel({
    manual_triage_report: encounter?.manual_triage_report,
  });
  const physicianDiagnosis = getDiagnosisLabel(encounter);

  const mistriageFlag = aiUrgency !== nurseUrgency && aiUrgency !== physicianUrgency;
  const refinementOccurred = aiDiagnosis && physicianDiagnosis && aiDiagnosis.toLowerCase() !== physicianDiagnosis.toLowerCase();
  const reviewOutcome = !physicianUrgency
    ? 'PENDING_CLINICAL_REVIEW'
    : mistriageFlag
      ? 'MISTRIAGE_MAJOR'
      : 'CORRECT';

  return {
    encounterId: String(encounter?._id || encounter?.encounterId || 'unknown'),
    aiSuggestedUrgency: aiUrgency,
    nurseApprovedUrgency: nurseUrgency,
    physicianFinalUrgency: physicianUrgency,
    aiSuggestedDiagnosis: aiDiagnosis,
    nurseApprovedDiagnosis: nurseDiagnosis,
    physicianFinalDiagnosis: physicianDiagnosis,
    reviewOutcome,
    mistriageFlag,
    refinementOccurred,
    reviewerNotes: encounter?.physician_notes || encounter?.nurseNotes || '',
  };
};

const toSeverityScale = (priority) => {
  switch (priority) {
    case 'RED':
      return 'Critical';
    case 'ORANGE':
      return 'High';
    case 'YELLOW':
      return 'Medium';
    case 'GREEN':
      return 'Low';
    default:
      return null;
  }
};

export const getTriageAnalytics = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    if (!hospitalId) {
      return res.status(401).json({ success: false, message: 'Hospital context is required' });
    }

    const { start, end } = getRange(req.query?.range || 'today');
    const query = {
      hospitalId,
      syntheticData: { $ne: true },
      date: {
        $gte: start,
        $lte: end,
      },
    };

    const [encounters, staffClockRecords, auditLogs] = await Promise.all([
      Encounter.find(query).sort({ date: -1 }).lean(),
      StaffClockRecord.find({
        hospitalId,
        date: { $gte: start, $lte: end },
        clockInTime: { $type: 'date' },
        status: { $in: ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT'] },
      })
        .populate({ path: 'staffId', select: 'firstName lastName', model: ClinicalStaff })
        .lean(),
      AuditLog.find({ hospitalId, timestamp: { $gte: start, $lte: end } }).select('action timestamp hospitalId').lean(),
    ]);

    const totalPatients = encounters.length;
    const waitMinutes = encounters.map(getWaitMinutes).filter(Number.isFinite);
    const averageWait = waitMinutes.length ? waitMinutes.reduce((sum, value) => sum + value, 0) / waitMinutes.length : null;

    const priorityBuckets = {
      Critical: { severity: 'Critical', allocated: 15, actual: 0, count: 0 },
      High: { severity: 'High', allocated: 30, actual: 0, count: 0 },
      Medium: { severity: 'Medium', allocated: 60, actual: 0, count: 0 },
      Low: { severity: 'Low', allocated: 120, actual: 0, count: 0 },
    };

    encounters.forEach((encounter) => {
      const priority = getPriorityLabel(encounter);
      const severity = toSeverityScale(priority);
      const bucket = priorityBuckets[severity];
      const waitMinutes = getWaitMinutes(encounter);
      if (!bucket || !Number.isFinite(waitMinutes)) return;

      bucket.count += 1;
      bucket.actual += waitMinutes;
    });

    const waitTimeVsSeverity = Object.values(priorityBuckets).map((bucket) => ({
      ...bucket,
      actual: bucket.count ? Number((bucket.actual / bucket.count).toFixed(1)) : 0,
    })).filter((bucket) => bucket.count > 0);

    const diagnosisCounts = new Map();
    encounters.forEach((encounter) => {
      const label = getDiagnosisLabel(encounter);
      if (!label) return;
      diagnosisCounts.set(label, (diagnosisCounts.get(label) || 0) + 1);
    });

    const diagnoses = [...diagnosisCounts.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, 7)
      .map(([name, count]) => ({
        name,
        count,
        percentage: totalPatients ? Number(((count / totalPatients) * 100).toFixed(1)) : 0,
      }));

    const caseReviews = encounters.map(buildCaseReview).filter(Boolean);

    const reviewedCases = caseReviews.filter((item) => item.reviewOutcome !== 'PENDING_CLINICAL_REVIEW');
    const mistriageCount = reviewedCases.filter((item) => item.mistriageFlag).length;
    const correctCount = reviewedCases.filter((item) => item.reviewOutcome === 'CORRECT').length;
    const triageAccuracy = reviewedCases.length ? Number(((correctCount / reviewedCases.length) * 100).toFixed(1)) : null;
    const mistriageRate = reviewedCases.length ? Number(((mistriageCount / reviewedCases.length) * 100).toFixed(1)) : null;

    const assignedCasesByStaff = new Map();
    encounters.forEach((encounter) => {
      const assignedIds = new Set([
        encounter?.assignedPhysicianId,
        encounter?.assignedNurseId,
        ...(encounter?.assigned_clinicians || []).map((clinician) => clinician?.id || clinician?._id),
      ].filter(Boolean).map(String));

      assignedIds.forEach((staffId) => {
        assignedCasesByStaff.set(staffId, (assignedCasesByStaff.get(staffId) || 0) + 1);
      });
    });

    const now = Date.now();
    const staffHoursById = new Map();
    staffClockRecords.forEach((record) => {
      const staffId = record.staffId?._id || record.staffId;
      const staffKey = staffId?.toString?.();
      const clockIn = new Date(record.clockInTime).getTime();
      const clockOut = record.clockOutTime
        ? new Date(record.clockOutTime).getTime()
        : Math.min(now, end.getTime());
      if (!staffKey || !Number.isFinite(clockIn) || !Number.isFinite(clockOut) || clockOut < clockIn) return;

      const shiftMs = record.clockOutTime && Number.isFinite(record.totalShiftDuration)
        ? record.totalShiftDuration
        : clockOut - clockIn;
      const breakStart = record.breakStartTime ? new Date(record.breakStartTime).getTime() : NaN;
      const breakEnd = record.breakEndTime ? new Date(record.breakEndTime).getTime() : NaN;
      const recordedBreakMs = Number.isFinite(record.totalBreakTime)
        ? record.totalBreakTime
        : Number.isFinite(breakStart) && Number.isFinite(breakEnd) && breakEnd >= breakStart
          ? breakEnd - breakStart
          : 0;
      const ongoingBreakMs = record.status === 'ON_BREAK' && Number.isFinite(breakStart)
        ? Math.max(0, Math.min(now, end.getTime()) - breakStart)
        : 0;
      const breakMs = Math.max(0, recordedBreakMs + ongoingBreakMs);
      const workMs = record.clockOutTime && Number.isFinite(record.totalWorkTime)
        ? record.totalWorkTime
        : Math.max(0, shiftMs - breakMs);
      const staffName = [record.staffId?.firstName, record.staffId?.lastName].filter(Boolean).join(' ').trim();
      const current = staffHoursById.get(staffKey) || {
        name: staffName || 'Staff record unavailable',
        shiftMs: 0,
        breakMs: 0,
        workMs: 0,
      };

      current.shiftMs += Math.max(0, shiftMs);
      current.breakMs += Math.max(0, breakMs);
      current.workMs += Math.max(0, workMs);
      staffHoursById.set(staffKey, current);
    });

    const staffHours = [...staffHoursById.entries()].map(([staffId, totals]) => {
      const actualHours = totals.workMs / 3600000;
      const actual = Number(actualHours.toFixed(1));
      const assignedCases = assignedCasesByStaff.get(staffId) || 0;
      return {
        name: totals.name,
        shift: Number((totals.shiftMs / 3600000).toFixed(1)),
        break: Number((totals.breakMs / 3600000).toFixed(1)),
        actual,
        assignedCases,
        casesPerHour: actualHours > 0 ? Number((assignedCases / actualHours).toFixed(1)) : null,
      };
    });
    const totalWorkedMilliseconds = [...staffHoursById.values()].reduce((sum, item) => sum + item.workMs, 0);
    const totalStaffHours = staffHours.length
      ? Number((totalWorkedMilliseconds / 3600000).toFixed(1))
      : null;
    const assignedCasesPerHour = totalWorkedMilliseconds > 0
      ? Number((staffHours.reduce((sum, item) => sum + item.assignedCases, 0) / (totalWorkedMilliseconds / 3600000)).toFixed(1))
      : null;
    const assignedCasesPerHourSampleSize = staffHours.filter((item) => item.actual > 0).length;

    const response = {
      success: true,
      hospitalId: String(hospitalId),
      range: req.query?.range || 'today',
      waitTimeVsSeverity,
      diagnoses,
      caseReviews,
      staffHours,
      dailyMetrics: {
        totalPatients,
        triageAccuracy,
        triageAccuracySampleSize: reviewedCases.length,
        mistriageRate,
        mistriageSampleSize: reviewedCases.length,
        avgWaitTime: averageWait === null ? null : Number(averageWait.toFixed(1)),
        avgWaitTimeSampleSize: waitMinutes.length,
        assignedCasesPerHour,
        assignedCasesPerHourSampleSize,
        totalStaffHours,
        totalStaffHoursSampleSize: staffHours.length,
      },
      eventSummary: {
        totalHospitalEvents: auditLogs.length + totalPatients,
        trackedEventSource: 'Encounter + AuditLog',
      },
    };

    return res.json(response);
  } catch (error) {
    console.error('[ANALYTICS] Error fetching triage analytics:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to load triage analytics',
      details: error.message,
    });
  }
};
