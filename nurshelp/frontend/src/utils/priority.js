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
  const rpt = p.report ?? p.patientReport ?? null;
  const sev = rpt?.severity ?? p.severity ?? p.triage?.severity ?? 1;
  return Math.max(1, Math.min(3, Number(sev) || 1));
};

export const waitStartFor = (p) => {
  return p.waitStartedAt ?? p.arrivalAt ?? p.createdAt ?? p.admittedAt ?? null;
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
  const rpt = p.report ?? p.patientReport ?? {};
  const satsCat = (rpt.satsCategory || rpt.sats || null);
  const sev = severityFromReport(p);

  const satsMap = {
    Red: { base: 100000, threshold: 0 },
    Orange: { base: 4000, threshold: 10 * 60 * 1000 },
    Yellow: { base: 2000, threshold: 60 * 60 * 1000 },
    Green: { base: 1000, threshold: 240 * 60 * 1000 }
  };

  let category = 'Green';
  if (typeof satsCat === 'string') {
    if (['Red','Orange','Yellow','Green'].includes(satsCat)) category = satsCat;
  } else {
    if (sev === 3) category = 'Red';
    else if (sev === 2) category = 'Orange';
    else category = 'Green';
  }

  const meta = satsMap[category] || satsMap.Green;
  if (category === 'Red') return { score: meta.base, label: 'Critical', category };

  const start = waitStartFor(p);
  const waitMs = start ? Math.max(0, nowMs - start) : 0;
  const threshold = meta.threshold || (60 * 60 * 1000);
  const waitRatio = threshold > 0 ? (waitMs / threshold) : Infinity;

  const maxWaitContribution = 5000;
  const waitScore = Math.min(Math.floor(waitRatio * maxWaitContribution), maxWaitContribution);

  const score = meta.base + waitScore;
  let label = 'Low';
  if (waitRatio >= 1) label = 'High';
  else if (waitRatio >= 0.5) label = 'Med';
  else label = 'Low';
  return { score, label, category, waitRatio };
};

export default {
  formatWaitDetailed,
  severityFromReport,
  waitStartFor,
  computePriority,
  normalizePatient
};
