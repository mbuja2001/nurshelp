// SATS-aware priority utilities
export const formatWaitDetailed = (ms) => {
  if (!ms || ms < 0) return "00:00:00.00";
  const cs = Math.floor((ms % 1000) / 10);
  const sec = Math.floor(ms / 1000) % 60;
  const min = Math.floor(ms / (60 * 1000)) % 60;
  const hrs = Math.floor(ms / (3600 * 1000));
  const pad = (n, l = 2) => String(n).padStart(l, '0');
  return `${pad(hrs)}:${pad(min)}:${pad(sec)}.${pad(cs)}`;
};

export const severityFromReport = (p) => {
  if (!p) return 1;
  const rpt = p.manual_triage_report ?? p.triage?.manualTriageReport ?? p.report ?? p.patientReport ?? p.triage?.aiTriageReport ?? p.ai_triage_report ?? null;
  const sev = rpt?.severity ?? rpt?.esi_level ?? rpt?.esiLevel ?? p.severity ?? p.triage?.severity ?? 1;
  return Math.max(1, Math.min(4, Number(sev) || 1));
};

const normalizePriorityCategory = (value) => {
  if (value === undefined || value === null || value === '') return null;
  const raw = String(value).trim();
  if (!raw) return null;

  const mapping = {
    red: 'Red',
    orange: 'Orange',
    yellow: 'Yellow',
    green: 'Green',
    blue: 'Green',
    critical: 'Red',
    urgent: 'Red',
    moderate: 'Yellow',
    low: 'Green'
  };

  const direct = mapping[raw.toLowerCase()];
  if (direct) return direct;

  const upper = raw.toUpperCase();
  if (['RED', 'ORANGE', 'YELLOW', 'GREEN', 'BLUE', 'CRITICAL', 'URGENT', 'MODERATE', 'LOW'].includes(upper)) {
    return normalizePriorityCategory(upper);
  }

  const digits = Number.parseInt(raw.replace(/[^0-9]/g, ''), 10);
  if ([1, 2, 3, 4].includes(digits)) {
    return ['Red', 'Orange', 'Yellow', 'Green'][digits - 1];
  }
  if (digits === 5) return 'Green';

  return null;
};

export const waitStartFor = (p) => {
  return p.waitStartedAt ?? p.wait_started_at ?? p.arrivalAt ?? p.arrival_at ?? p.createdAt ?? p.created_at ?? p.admittedAt ?? p.admitted_at ?? p.date ?? null;
};

export const normalizePatient = (p) => {
  // do not mutate original
  const copy = { ...p };
  const rpt = copy.report ?? copy.patientReport ?? {};
  // Prefer nurse-confirmed report if present (authoritative/auditable)
  const nurseRpt = rpt.nurseConfirmedReport ?? rpt.nurseConfirmed ?? null;

  // Determine if a specialist actually attended: prefer explicit specialist fields, then nurse-confirmed report, then report-level confirmations
  if (typeof copy.specialistAttended === 'undefined') {
    copy.specialistAttended = !!(
      copy.specialistAttended ||
      copy.attended ||
      copy.assigned ||
      copy.specialistAttendedAt ||
      (nurseRpt && (nurseRpt.specialistAttended || nurseRpt.attended || nurseRpt.nurseConfirmedAttended)) ||
      rpt.specialistAttended ||
      rpt.attended ||
      rpt.nurseConfirmedAttended
    );
  }

  // Normalize attended timestamp from patient-level, nurse-confirmed, or report-level fields
  if (!copy.specialistAttendedAt && copy.specialistAttended) {
    copy.specialistAttendedAt = copy.attendedAt || copy.assignedAt || copy.specialist_attended_at || (nurseRpt && (nurseRpt.attendedAt || nurseRpt.specialistAttendedAt || nurseRpt.attended_at)) || rpt.attendedAt || rpt.specialistAttendedAt || rpt.attended_at || null;
  }

  // normalize attending doctor name, prefer nurse-confirmed report values
  copy.specialistDoctor = copy.specialistDoctor || (nurseRpt && (nurseRpt.attendingDoctor || nurseRpt.attendingDoctorName || nurseRpt.attendedByName || nurseRpt.nurseConfirmedDoctor)) || copy.attendingDoctor || copy.attendingDoctorName || copy.attending_doctor || copy.attendedByName || copy.attendedBy || copy.specialistName || copy.doctor || rpt.attendingDoctor || rpt.attendingDoctorName || rpt.attendedByName || rpt.nurseConfirmedDoctor || null;

  // Capture AI suggestion(s) for audit/history without overwriting nurse-confirmed values
  copy._aiSuggestedReport = rpt.aiSuggestedReport || rpt.suggestedReport || rpt.aiSuggestion || null;
  return copy;
};

const firstValue = (...values) => values.find((value) => value !== undefined && value !== null && value !== '');

const asNumber = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const getVitals = (value = {}) => value?.vitals || value?.vitalSigns || value?.triage?.vitals_parsed || value?.report?.vitals || {};

const getVitalHistory = (value = {}) => {
  const history = firstValue(
    value.vitalsHistory,
    value.vitalHistory,
    value.vitalsTrend,
    value.recentVitals,
    value.triage?.vitalsHistory,
    value.triage?.vital_history,
    value.triage?.vitals_trend,
    value.report?.vitalsHistory
  );
  if (!Array.isArray(history)) return [];
  return history.map((entry) => entry?.vitals || entry?.vitalSigns || entry).filter((entry) => entry && typeof entry === 'object');
};

const vitalsValue = (vitals, names) => asNumber(firstValue(...names.map((name) => vitals?.[name])));

const historyText = (value = {}) => JSON.stringify([
  value.criticalHistory,
  value.critical_history,
  value.history,
  value.medicalHistory,
  value.patient?.criticalHistory,
  value.triage?.criticalHistory,
  value.triage?.critical_history
]).toLowerCase();

const getCriticalHistorySignals = (value = {}) => {
  const text = historyText(value);
  const activeCoagulopathy = /haemophilia|hemophilia|warfarin.*apixaban|apixaban.*warfarin|concurrent anticoagul/.test(text);
  const asplenicSepsisRisk = /splenectomy|asplenic|asplenia/.test(text);
  const temperature = vitalsValue(getVitals(value), ['temperature', 'temp', 'temperature_celsius']);
  return { activeCoagulopathy, asplenicSepsisRisk: asplenicSepsisRisk && temperature !== null && temperature > 38.5 };
};

const getPhysiologicalDrift = (value = {}) => {
  const current = getVitals(value);
  const readings = [...getVitalHistory(value), current].filter((reading) => Object.keys(reading).length > 0);
  const previous = readings.length > 1 ? readings[readings.length - 2] : null;
  const currentHr = vitalsValue(current, ['heartRate', 'heart_rate', 'hr']);
  const currentSbp = vitalsValue(current, ['systolicBP', 'systolic_bp', 'bp_systolic', 'sbp']);
  const currentO2 = vitalsValue(current, ['oxygenSaturation', 'oxygen_saturation', 'o2', 'spo2', 'o2_sat']);
  const previousHr = vitalsValue(previous, ['heartRate', 'heart_rate', 'hr']);
  const previousSbp = vitalsValue(previous, ['systolicBP', 'systolic_bp', 'bp_systolic', 'sbp']);
  const previousO2 = vitalsValue(previous, ['oxygenSaturation', 'oxygen_saturation', 'o2', 'spo2', 'o2_sat']);
  const shockIndex = currentHr !== null && currentSbp > 0 ? currentHr / currentSbp : null;
  const hypoxicDrop = previousO2 !== null && currentO2 !== null ? previousO2 - currentO2 : 0;
  const relativeBradycardia = previousHr !== null && previousSbp !== null && currentHr !== null && currentSbp !== null && currentHr < previousHr && currentSbp < previousSbp;
  return {
    shockIndex,
    shockIndexAlert: shockIndex !== null && shockIndex > 1,
    hypoxicDrop,
    hypoxicTrendAlert: hypoxicDrop >= 3,
    relativeBradycardia
  };
};

export const formatTimestamp = (ts) => {
  if (!ts) return null;
  try {
    const d = new Date(Number(ts));
    if (isNaN(d.getTime())) return String(ts);
    return d.toLocaleString();
  } catch (e) {
    return String(ts);
  }
};

export const computePriority = (p, nowMs) => {
  const rpt = p.manual_triage_report ?? p.triage?.manualTriageReport ?? p.report ?? p.patientReport ?? p.triage?.aiTriageReport ?? p.ai_triage_report ?? p.triage ?? {};
  const satsCat = firstValue(
    rpt.priority_code,
    rpt.priorityCode,
    rpt.triageCategory,
    rpt.satsCategory,
    rpt.satsLevel,
    rpt.selected_priority,
    rpt.sats_color,
    rpt.sats,
    p.priority_code,
    p.priorityCode,
    p.triageCategory,
    p.triage?.triageCategory,
    p.triage?.priority_code,
    p.triage?.priorityCode,
    p.triage?.satsLevel,
    p.triage?.sats_level,
    p.ai_triage_report?.priority_code,
    p.satsLevel
  );
  const sev = severityFromReport(p);

  const satsMap = {
    Red: { base: 100000, threshold: 0 },
    Orange: { base: 4000, threshold: 10 * 60 * 1000 },
    Yellow: { base: 2000, threshold: 60 * 60 * 1000 },
    Green: { base: 1000, threshold: 240 * 60 * 1000 }
  };

  let category = 'Green';
  const normalizedCategory = normalizePriorityCategory(satsCat);
  if (normalizedCategory) {
    category = normalizedCategory;
  } else {
    if (sev === 3) category = 'Red';
    else if (sev === 2) category = 'Orange';
    else if (sev === 4) category = 'Green';
    else category = 'Green';
  }

  const meta = satsMap[category] || satsMap.Green;

  const start = waitStartFor(p);
  const startMs = start ? new Date(start).getTime() : NaN;
  const waitMs = Number.isFinite(startMs) ? Math.max(0, nowMs - startMs) : 0;
  const threshold = meta.threshold || (60 * 60 * 1000);
  const waitRatio = threshold > 0 ? (waitMs / threshold) : Infinity;

  const drift = getPhysiologicalDrift(p);
  const criticalHistory = getCriticalHistorySignals(p);
  const timeDebt = waitRatio >= 1;
  const hardClinicalOverride = criticalHistory.asplenicSepsisRisk || drift.relativeBradycardia || drift.shockIndexAlert;
  if (hardClinicalOverride && category !== 'Red') category = 'Red';
  const effectiveMeta = satsMap[category] || satsMap.Green;
  const timeDecayScore = Math.min(Math.floor((Math.exp(Math.min(waitRatio, 4)) - 1) * 900), 5000);
  const driftScore = (drift.shockIndexAlert ? 700 : 0) + (drift.hypoxicTrendAlert ? 600 : 0) + (drift.relativeBradycardia ? 800 : 0);
  const crossRiskScore = (criticalHistory.activeCoagulopathy ? 500 : 0) + (criticalHistory.asplenicSepsisRisk ? 900 : 0);
  const waitScore = Math.min(timeDecayScore + driftScore + crossRiskScore, 7000);

  const score = effectiveMeta.base + waitScore;
  let label = 'Low';
  if (category === 'Red') label = 'Critical';
  else if (criticalHistory.asplenicSepsisRisk || drift.relativeBradycardia || drift.shockIndexAlert) label = 'Immediate review';
  else if (timeDebt) label = 'High';
  else if (waitRatio >= 0.5) label = 'Med';
  else label = 'Low';
  return {
    score,
    label,
    category,
    waitRatio,
    timeDebt,
    physiologicalDrift: drift,
    crossRisk: criticalHistory,
    queueSignals: [
      drift.shockIndexAlert ? 'Shock index > 1.0' : null,
      drift.hypoxicTrendAlert ? 'O2 drop >= 3%' : null,
      drift.relativeBradycardia ? 'Heart rate and blood pressure both falling' : null,
      criticalHistory.activeCoagulopathy ? 'Active coagulopathy overlap' : null,
      criticalHistory.asplenicSepsisRisk ? 'Asplenic fever risk' : null,
      timeDebt ? 'Time-to-target breached' : null
    ].filter(Boolean)
  };
};

export default {
  formatWaitDetailed,
  severityFromReport,
  waitStartFor,
  computePriority,
  normalizePatient
};
