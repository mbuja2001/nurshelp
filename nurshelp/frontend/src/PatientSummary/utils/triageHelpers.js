// Utility helpers to normalize manual triage data to AI-style structure
export const SATS_TEWS_RULES = Object.freeze([
  { level: 'red', min: 7, max: Infinity, label: 'RED', description: '7 or more' },
  { level: 'orange', min: 5, max: 6, label: 'ORANGE', description: '5-6' },
  { level: 'yellow', min: 3, max: 4, label: 'YELLOW', description: '3-4' },
  { level: 'green', min: 0, max: 2, label: 'GREEN', description: '0-2' }
]);

export const getSatsAcuityFromTews = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'green';
  if (numeric >= 7) return 'red';
  if (numeric >= 5) return 'orange';
  if (numeric >= 3) return 'yellow';
  return 'green';
};

export const normalizeSatsLevel = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const asString = String(value).trim();
  const lowered = asString.toLowerCase();
  if (['red', 'orange', 'yellow', 'green', 'blue'].includes(lowered)) return lowered;
  const numeric = Number(asString.replace(/[^\d]/g, ''));
  if (numeric === 1) return 'red';
  if (numeric === 2) return 'orange';
  if (numeric === 3) return 'yellow';
  if (numeric === 4) return 'green';
  if (numeric === 5) return 'blue';
  return null;
};

export const getSATSBadge = (priority) => {
  const prioNum = parseInt(String(priority).replace(/\D/g, '').substring(0, 1));
  const satsMap = {
    1: { color: '#ffffff', bgColor: '#dc2626', label: 'RED', border: '#991b1b' },
    2: { color: '#ffffff', bgColor: '#ea580c', label: 'ORANGE', border: '#9a3412' },
    3: { color: '#000000', bgColor: '#fef3c7', label: 'YELLOW', border: '#92400e' },
    4: { color: '#ffffff', bgColor: '#059669', label: 'GREEN', border: '#047857' },
    5: { color: '#000000', bgColor: '#0891b2', label: 'BLUE', border: '#164e63' }
  };
  return satsMap[prioNum] || { color: '#000000', bgColor: '#f1f5f9', label: priority || 'N/A', border: '#94a3b8' };
};

// Normalize a manual triage report's clinicalAnalysis into the AI-style keys
export const normalizeClinicalAnalysis = (manualReport = {}) => {
  const parseList = (value) => {
    if (Array.isArray(value)) return value.filter((item) => item !== null && item !== undefined && item !== '');
    if (typeof value === 'string') {
      return value
        .split(/\r?\n|;|\|/)
        .map((item) => item.trim())
        .filter(Boolean);
    }
    if (value === null || value === undefined || value === '') return [];
    return [String(value)];
  };

  const ca = manualReport.clinicalAnalysis || manualReport.analysis || manualReport.aiTriageReport || manualReport.clinical_analysis || manualReport.triage?.manualTriageReport?.clinicalAnalysis || manualReport.triage?.aiTriageReport || {};

  const asString = (value) => {
    if (typeof value === 'string') return value;
    if (typeof value === 'number') return String(value);
    return '';
  };

  const clinicalSummary = asString(ca.clinicalSummary || ca.clinical_summary || manualReport.clinicalSummary || manualReport.clinical_summary || manualReport.ai_summary || manualReport.summary);
  const pathophysiology = asString(ca.pathophysiology || ca.pathophysiology_text || manualReport.pathophysiology);
  const clinicalAnalysisText = asString(ca.clinicalAnalysis || ca.clinical_analysis || manualReport.clinicalAnalysis || manualReport.clinical_analysis);
  const severityAssessment = asString(ca.severityAssessment || ca.severity_assessment || manualReport.severityAssessment || manualReport.severity_assessment);
  const clinicalFindings = ca.clinicalFindings || ca.clinical_findings || manualReport.clinicalFindings || manualReport.clinical_findings || manualReport.findings || [];
  const clinicalDiscriminators = ca.clinicalDiscriminators || ca.clinical_discriminators || ca.discriminators || manualReport.clinicalDiscriminators || manualReport.clinical_discriminators || manualReport.discriminators || [];
  const differentialDiagnoses = ca.differentialDiagnoses || ca.differential_diagnoses || ca.differential_diagnosis || manualReport.differentialDiagnoses || manualReport.differential_diagnoses || manualReport.differential_diagnosis || [];
  const immediateActions = ca.immediateActions || ca.immediate_actions || ca.recommendations || manualReport.immediateActions || manualReport.immediate_actions || manualReport.recommendations || [];
  const diagnosticWorkup = ca.diagnosticWorkup || ca.diagnostic_workup || manualReport.diagnosticWorkup || manualReport.diagnostic_workup || [];
  const dispositionRationale = ca.dispositionRationale || ca.disposition_rationale || manualReport.dispositionRationale || manualReport.disposition_rationale || '';
  const clinicalReasoningSummary = ca.clinicalReasoningSummary || ca.clinical_reasoning_summary || ca.reasoning_summary || manualReport.clinicalReasoningSummary || manualReport.clinical_reasoning_summary || manualReport.reasoning_summary || '';
  const recommendations = ca.recommendations || ca.treatment_recommendations || manualReport.recommendations || manualReport.treatment_recommendations || [];
  const managementActions = ca.managementActions || ca.management_actions || manualReport.managementActions || manualReport.management_actions || [];
  const advancedInterventions = ca.advancedInterventions || ca.advanced_interventions || manualReport.advancedInterventions || manualReport.advanced_interventions || [];
  const riskFactors = ca.riskFactors || ca.risk_factors || manualReport.riskFactors || manualReport.risk_factors || [];
  const pathognomonicMarkers = ca.pathognomonicMarkers || ca.pathognomonic_markers || manualReport.pathognomonicMarkers || manualReport.pathognomonic_markers || [];
  const lifeThreatsExcluded = ca.lifeThreatsExcluded || ca.life_threats_excluded || manualReport.lifeThreatsExcluded || manualReport.life_threats_excluded || [];
  const recommendedSpecialists = ca.recommendedSpecialists || ca.recommended_specialists || manualReport.recommendedSpecialists || manualReport.recommended_specialists || [];

  const satsReasoning = ca.SATS_reasoning || ca.sats_reasoning || manualReport.SATS_reasoning || manualReport.sats_reasoning || {};
  const tews_total = ca.tews_total ?? ca.total ?? satsReasoning.tews_total ?? satsReasoning.total ?? manualReport.tews_total ?? manualReport.sats_points ?? manualReport.urgencyScore ?? null;
  const priority_raw = ca.priority_code ?? ca.selected_priority ?? ca.sats_level ?? ca.satsLevel ?? satsReasoning.priority ?? manualReport.selected_priority ?? manualReport.satsLevel ?? manualReport.sats_color ?? manualReport.priority ?? manualReport.priority_code ?? null;
  const priority_code = priority_raw ?? null;
  const sats_level = normalizeSatsLevel(priority_raw);
  const urgencyScore = ca.acuityScore ?? ca.urgencyScore ?? ca.urgency_score ?? manualReport.urgencyScore ?? manualReport.urgency_score ?? null;

  return {
    clinicalSummary,
    clinical_summary: clinicalSummary,
    pathophysiology,
    clinical_analysis: clinicalAnalysisText,
    severity_assessment: severityAssessment,
    clinical_findings: parseList(clinicalFindings),
    clinical_discriminators: parseList(clinicalDiscriminators),
    differential_diagnoses: parseList(differentialDiagnoses),
    immediate_actions: parseList(immediateActions),
    diagnostic_workup: parseList(diagnosticWorkup),
    disposition_rationale: dispositionRationale,
    clinical_reasoning_summary: clinicalReasoningSummary,
    recommendations: parseList(recommendations),
    management_actions: parseList(managementActions),
    advanced_interventions: parseList(advancedInterventions),
    risk_factors: parseList(riskFactors),
    pathognomonic_markers: parseList(pathognomonicMarkers),
    life_threats_excluded: parseList(lifeThreatsExcluded),
    recommended_specialists: parseList(recommendedSpecialists),
    tews_total: (typeof tews_total === 'number') ? tews_total : (tews_total ? Number(tews_total) : null),
    priority_code,
    sats_level,
    urgencyScore
  };
};

export default {
  SATS_TEWS_RULES,
  getSatsAcuityFromTews,
  normalizeSatsLevel,
  getSATSBadge,
  normalizeClinicalAnalysis
};
