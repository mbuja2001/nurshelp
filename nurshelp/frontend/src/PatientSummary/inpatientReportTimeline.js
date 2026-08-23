const toTimestamp = (value) => {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const formatDayLabel = (date) => {
  if (!date) return 'Unknown date';
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
};

const formatTimeLabel = (date) => {
  if (!date) return 'Time not recorded';
  // Format with split-second precision: HH:MM:SS
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
};

const formatVitals = (vitals = {}) => {
  const values = [
    vitals.bp || (vitals.bp_systolic && vitals.bp_diastolic ? `BP ${vitals.bp_systolic}/${vitals.bp_diastolic}` : null),
    vitals.hr && `HR ${vitals.hr}`,
    vitals.temp && `Temp ${vitals.temp}`,
    vitals.o2 && `SpO2 ${vitals.o2}`,
    vitals.resp && `Resp ${vitals.resp}`,
    vitals.avpu && `AVPU ${vitals.avpu}`
  ].filter(Boolean);
  return values.length ? values.join(' | ') : 'Vitals recorded';
};

const summarizeReport = (report = {}) => (
  report.ai_summary || report.summary || report.clinical_summary ||
  report.primary_diagnosis || report.diagnosis || report.impression ||
  'Clinical report recorded'
);

export const buildInpatientTimeline = (encounter = {}) => {
  const entries = [];

  console.log('[TIMELINE BUILDER] Building timeline for encounter:', {
    encounterId: encounter._id,
    patientName: encounter.patient?.name || encounter.patientName,
    hasProgressionNotes: (encounter.progression_notes || []).length,
    hasStatusHistory: (encounter.status_history || []).length,
    hasTreatments: (encounter.treatments || []).length,
    hasRecommendations: (encounter.recommendations || []).length,
    hasHandovers: (encounter.clinical_handovers || []).length,
    hasLabs: (encounter.lab_requests || []).length,
    hasMedications: (encounter.medication_administrations || []).length,
    hasSurgeries: (encounter.surgery_records || []).length
  });

  const pushEntry = (kind, title, body, timestamp, meta = {}) => {
    const parsedTimestamp = toTimestamp(timestamp);
    if (!parsedTimestamp) {
      console.log('[TIMELINE BUILDER] ⚠️ Skipped entry - invalid timestamp:', { kind, title, timestamp });
      return;
    }

    entries.push({
      id: `${kind}-${parsedTimestamp.getTime()}-${entries.length}`,
      kind,
      title,
      body,
      timestamp: parsedTimestamp,
      meta
    });
  };

  const encounterTimestamp = encounter.createdAt || encounter.updatedAt;

  if (encounter.vitals && Object.keys(encounter.vitals).some((key) => encounter.vitals[key] != null)) {
    pushEntry('VITALS', 'Vitals recorded', formatVitals(encounter.vitals), encounter.vitals.recordedAt || encounterTimestamp, {
      source: 'encounter.vitals',
      vitals: encounter.vitals
    });
  }

  const aiReport = encounter.ai_triage_report && Object.keys(encounter.ai_triage_report).length
    ? encounter.ai_triage_report
    : encounter.triage;
  if (aiReport && Object.keys(aiReport).length) {
    pushEntry('AI_TRIAGE', 'AI triage report', summarizeReport(aiReport), encounter.submittedAt || encounterTimestamp, {
      source: encounter.ai_triage_report && Object.keys(encounter.ai_triage_report).length ? 'ai_triage_report' : 'triage',
      report: aiReport
    });
  }

  if (encounter.manual_triage_report && Object.keys(encounter.manual_triage_report).length) {
    pushEntry('MANUAL_TRIAGE', 'Manual triage report', summarizeReport(encounter.manual_triage_report), encounter.manual_triage_report.recorded_at || encounter.updatedAt || encounterTimestamp, {
      source: 'manual_triage_report',
      report: encounter.manual_triage_report
    });
  }

  if (encounter.diagnosis || encounter.treatment_plan || encounter.physician_notes || encounter.assigned_physician?.diagnosed_at) {
    pushEntry('DIAGNOSIS', encounter.diagnosis || 'Diagnosis documented', [encounter.diagnosis, encounter.treatment_plan, encounter.physician_notes].filter(Boolean).join(' | ') || 'Diagnosis updated', encounter.assigned_physician?.diagnosed_at || encounter.updatedAt || encounterTimestamp, {
      source: 'diagnosis',
      diagnosis: encounter.diagnosis,
      treatment_plan: encounter.treatment_plan,
      physician_notes: encounter.physician_notes,
      assigned_physician: encounter.assigned_physician
    });
  }

  // Handover acknowledgments (NEW)
  console.log('[TIMELINE BUILDER] Processing clinical_handovers:', (encounter.clinical_handovers || []).length);
  (encounter.clinical_handovers || []).forEach((handover, index) => {
    if (handover.handover_status === 'ACKNOWLEDGED' && handover.acknowledged_at) {
      const specialistType = handover.specialist_role || handover.handover_role || 'Specialist';
      const specialistName = handover.specialist_name || handover.acknowledged_by || 'Unknown';
      console.log(`[TIMELINE BUILDER] Adding handover ack ${index}:`, { specialistType, specialistName, acknowledged_at: handover.acknowledged_at });
      pushEntry(
        'HANDOVER_ACK',
        `${specialistType} Acknowledged`,
        `${specialistName} acknowledged patient (${specialistType})`,
        handover.acknowledged_at,
        { handover, index }
      );
    }
  });

  console.log('[TIMELINE BUILDER] Processing progression_notes:', (encounter.progression_notes || []).length);
  (encounter.progression_notes || []).forEach((note, index) => {
    pushEntry(
      'PROGRESSION',
      note.note_type || 'Progress note',
      note.content || 'Progress note recorded',
      note.recorded_at || note.createdAt || encounter.createdAt,
      { note, index }
    );
  });

  console.log('[TIMELINE BUILDER] Processing status_history:', (encounter.status_history || []).length);
  (encounter.status_history || []).forEach((historyItem, index) => {
    pushEntry(
      'STATUS',
      historyItem.status || 'Status update',
      historyItem.notes || 'Status updated',
      historyItem.changed_at || historyItem.changedAt || encounter.createdAt,
      { historyItem, index }
    );
  });

  console.log('[TIMELINE BUILDER] Processing treatments:', (encounter.treatments || []).length);
  (encounter.treatments || []).forEach((treatment, index) => {
    pushEntry(
      'TREATMENT',
      treatment.name || treatment.intervention_type || 'Treatment recorded',
      treatment.notes || `${treatment.intervention_type || 'Treatment'} recorded`,
      treatment.administered_at || treatment.completed_at || treatment.prescribed_at || encounter.createdAt,
      { treatment, index }
    );
  });

  console.log('[TIMELINE BUILDER] Processing recommendations:', (encounter.recommendations || []).length);
  (encounter.recommendations || []).forEach((recommendation, index) => {
    pushEntry(
      'RECOMMENDATION',
      recommendation.recommendation_type || 'Recommendation',
      recommendation.description || 'Recommendation recorded',
      recommendation.completed_at || recommendation.recommended_at || encounter.createdAt,
      { recommendation, index }
    );
  });

  (encounter.medication_administrations || []).forEach((medication, index) => {
    pushEntry(
      'MEDICATION',
      medication.medication_name || 'Medication administration',
      [medication.status, medication.dosage, medication.route, medication.patient_response].filter(Boolean).join(' | ') || 'Medication administration recorded',
      medication.administered_at || medication.recorded_at || encounterTimestamp,
      { medication, index, source: 'medication_administrations' }
    );
  });

  (encounter.lab_requests || []).forEach((labRequest, index) => {
    pushEntry(
      'LAB_RESULT',
      labRequest.lab_type || `${labRequest.specialty || 'Laboratory'} report`,
      [labRequest.status, labRequest.report_content, labRequest.clinical_impression].filter(Boolean).join(' | ') || 'Laboratory request recorded',
      labRequest.submitted_at || labRequest.updated_at || labRequest.requested_at || encounterTimestamp,
      { labRequest, index, source: 'lab_requests' }
    );
  });

  (encounter.surgery_records || []).forEach((surgery, index) => {
    pushEntry(
      'SURGERY',
      surgery.procedure_name || 'Surgery report',
      [surgery.status, surgery.outcome, surgery.findings, surgery.complications].filter(Boolean).join(' | ') || 'Surgery record recorded',
      surgery.performed_at || surgery.updated_at || surgery.scheduled_at || surgery.requested_at || encounterTimestamp,
      { surgery, index, source: 'surgery_records' }
    );
  });

  if (encounter.discharge_report || encounter.status === 'DISCHARGED' || encounter.discharge_scheduled_at) {
    pushEntry(
      'DISCHARGE',
      'Discharge report',
      summarizeReport(encounter.discharge_report || {}) || 'Patient discharge recorded',
      encounter.discharge_report?.discharged_at || encounter.discharge_report?.createdAt || encounter.updatedAt || encounterTimestamp,
      { source: 'discharge_report', report: encounter.discharge_report, status: encounter.status }
    );
  }

  const sorted = entries.sort((a, b) => b.timestamp - a.timestamp);
  console.log('[TIMELINE BUILDER] ✅ Timeline complete:', { totalEntries: sorted.length, distinctKinds: new Set(sorted.map(e => e.kind)).size });
  return sorted;
};

export const groupTimelineByDay = (timeline = []) => {
  console.log('[GROUP BY DAY] Starting grouping with:', timeline.length, 'entries');

  const grouped = new Map();

  timeline.forEach((entry) => {
    const key = entry.timestamp.toISOString().split('T')[0];
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key).push(entry);
  });

  console.log('[GROUP BY DAY] Created', grouped.size, 'day groups');

  const result = Array.from(grouped.entries())
    .sort(([a], [b]) => (a < b ? 1 : -1))
    .map(([key, entries]) => {
      const dayEntries = entries
        .slice()
        .sort((a, b) => b.timestamp - a.timestamp)
        .map((entry) => ({
          ...entry,
          displayTime: formatTimeLabel(entry.timestamp)
        }));

      console.log(`[GROUP BY DAY] Day ${key}: ${dayEntries.length} entries`);
      return {
        key,
        label: formatDayLabel(new Date(key)),
        entries: dayEntries
      };
    });

  console.log('[GROUP BY DAY] ✅ Grouping complete:', result.length, 'days');
  return result;
};
