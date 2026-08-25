import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiChevronLeft, FiCheck } from 'react-icons/fi';
import { normalizeClinicalAnalysis, normalizeSatsLevel, getSATSBadge } from './utils/triageHelpers';

const ConfirmingManualClinicalHandover = () => {
  const navigate = useNavigate();
  const { encounterId } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [confirming, setConfirming] = useState(false);
  const [triageNurse, setTriageNurse] = useState(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

  // Decode JWT token to extract triage nurse (logged-in user)
  const getUserFromToken = (jwtToken) => {
    if (!jwtToken) return null;
    try {
      const parts = jwtToken.split('.');
      if (parts.length !== 3) return null;
      const decoded = JSON.parse(atob(parts[1]));
      return {
        id: decoded.id,
        name: decoded.name,
        email: decoded.email,
        role: decoded.role,
        hospitalId: decoded.hospitalId
      };
    } catch (err) {
      console.error('[ConfirmingManual] Error decoding token:', err);
      return null;
    }
  };

  useEffect(() => {
    if (token) {
      const nurse = getUserFromToken(token);
      if (nurse) setTriageNurse(nurse);
    }
  }, [token]);

  // helpers are imported from triageHelpers: normalizeSatsLevel, getSATSBadge, normalizeClinicalAnalysis

  useEffect(() => {
    const fetchEncounter = async () => {
      try {
        setLoading(true);
        const headers = {
          Authorization: `Bearer ${token}`,
          'Cache-Control': 'no-cache'
        };

        const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}?t=${Date.now()}`, {
          headers,
          cache: 'no-store'
        });

        if (!res.ok) throw new Error('Failed to load encounter');
        const data = await res.json();
        console.log('[ConfirmingManual] Loaded encounter payload:', data.encounter);
        setEncounter(data.encounter);
      } catch (err) {
        console.error('Error fetching encounter:', err);
        alert('Failed to load encounter');
      } finally {
        setLoading(false);
      }
    };

    if (encounterId) fetchEncounter();
  }, [encounterId, token, BACKEND_URL]);

  const handleConfirmHandover = async () => {
    try {
      setConfirming(true);
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      };

      // First, save the encounter data with manual/manual_triage_report fields
      // Normalize clinical analysis shape so manual triage persists using AI-like keys
      const rawManual = encounter.manual_triage_report || encounter.triage?.manualTriageReport || {};
      const normalized = normalizeClinicalAnalysis(rawManual);

      const manual_triage_report = {
        ...rawManual,
        ...normalized,
        clinical_analysis: normalized.clinical_analysis,
        clinical_summary: normalized.clinical_summary,
        clinical_discriminators: normalized.clinical_discriminators,
        differential_diagnoses: normalized.differential_diagnoses,
        immediate_actions: normalized.immediate_actions,
        diagnostic_workup: normalized.diagnostic_workup,
        recommendations: normalized.recommendations,
        clinical_reasoning_summary: normalized.clinical_reasoning_summary,
        disposition_rationale: normalized.disposition_rationale,
        management_actions: normalized.management_actions,
        advanced_interventions: normalized.advanced_interventions,
        risk_factors: normalized.risk_factors,
        pathognomonic_markers: normalized.pathognomonic_markers,
        life_threats_excluded: normalized.life_threats_excluded,
        recommended_specialists: normalized.recommended_specialists,
        severity_assessment: normalized.severity_assessment,
        tews_total: normalized.tews_total ?? rawManual.tews_total ?? null,
        urgencyScore: normalized.urgencyScore ?? rawManual.urgencyScore ?? rawManual.urgency_score ?? null,
        selected_priority: normalized.priority_code ?? rawManual.selected_priority ?? rawManual.satsLevel ?? rawManual.sats_color ?? null,
        satsLevel: normalized.sats_level ?? rawManual.satsLevel ?? rawManual.sats_color ?? null,
        acuityLevel: satsAcuityLevel
      };

      console.log('[ConfirmingManual] 🎯 HANDOVER - Final acuity mapping before send:', {
        tews_total: manual_triage_report.tews_total,
        urgencyScore: manual_triage_report.urgencyScore,
        satsLevel: manual_triage_report.satsLevel,
        acuityLevel: manual_triage_report.acuityLevel,
        resolution: 'acuityLevel = satsLevel (SATS priority), NOT urgencyScore (numeric TEWS)'
      });

      const savePayload = {
        triage: encounter.triage,
        assigned_clinicians: encounter.assigned_clinicians || [],
        manually_selected_clinicians: encounter.manually_selected_clinicians || [],
        manual_triage_report,
        assigned_ward: encounter.assigned_ward || null,
        assigned_bed: encounter.assigned_bed || null,
        ward_display_info: encounter.ward_display_info || null,
        bed_display_info: encounter.bed_display_info || null
      };

      console.log('[ConfirmingManual] Saving encounter fields before handover:', savePayload);

      // Include triage nurse metadata similar to AI handover flow
      const saveBody = {
        ...savePayload,
        triage_nurse: triageNurse ? {
          nurse_id: triageNurse.id,
          nurse_name: triageNurse.name,
          nurse_email: triageNurse.email,
          nurse_role: triageNurse.role,
          triaged_at: new Date().toISOString()
        } : null
      };

      console.log('[ConfirmingManual] PUT /ai-fields payload:', saveBody);
      const saveRes = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/ai-fields`, {
        method: 'PUT',
        headers,
        body: JSON.stringify(saveBody)
      });

      if (!saveRes.ok) {
        console.warn('Warning: Save response not ok:', saveRes.status);
        // Continue anyway - handover is the critical operation
      }

      // Then mark encounter as handed over with manual report
      const handoverBody = {
        reportType: 'MANUAL_TRIAGE_REPORT',
        clinicians: encounter.assigned_clinicians || [],
        manuallySelected: encounter.manually_selected_clinicians || [],
        triageNurse: triageNurse || null
      };
      console.log('[ConfirmingManual] POST /handover payload:', handoverBody);
      const handoverRes = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover`, {
        method: 'POST',
        headers,
        body: JSON.stringify(handoverBody)
      });

      if (!handoverRes.ok) throw new Error('Failed to confirm handover');

      alert('Clinical handover confirmed! Assigned clinicians have been notified.');
      navigate(`/report/manual-triage/${encounterId}`);
    } catch (err) {
      console.error('Error confirming handover:', err);
      alert('Failed to confirm handover: ' + err.message);
    } finally {
      setConfirming(false);
    }
  };

  if (loading) return <div style={{ padding: 20, textAlign: 'center' }}>Loading...</div>;
  if (!encounter) return <div style={{ padding: 20, textAlign: 'center' }}>Encounter not found</div>;

  const triageBundle = encounter.triage?.manualTriageReport || encounter.manual_triage_report || {};
  const manualReport = encounter.manual_triage_report && Object.keys(encounter.manual_triage_report).length > 0
    ? encounter.manual_triage_report
    : (encounter.triage?.manualTriageReport || {});
  const triageData = encounter.triage || {};
  const manuallySelected = encounter.manually_selected_clinicians || [];
  const assignedClinicians = encounter.assigned_clinicians || encounter.triage?.assigned_clinicians || [];
  
  // Debug logging
  console.log('[ConfirmingManual] DEBUG - Data extraction:');
  console.log('  - assignedClinicians:', assignedClinicians);
  console.log('  - assignedClinicians[0]:', assignedClinicians[0]);
  console.log('  - physician clinician:', assignedClinicians.find(c => c.role === 'PHYSICIAN'));
  console.log('  - manualReport.assigned_specialty:', manualReport.assigned_specialty);
  console.log('  - triageData.assigned_specialty:', triageData.assigned_specialty);
  
  // Normalize clinical analysis so we consistently render individual fields
  const clinicalAnalysis = normalizeClinicalAnalysis(manualReport || triageBundle || {});
  const normalizedManualReport = {
    ...manualReport,
    ...clinicalAnalysis
  };
  const sourceReport = normalizedManualReport;

  const getVitalStatus = (value, min, max) => {
    const num = parseFloat(value);
    if (isNaN(num)) return { color: '#64748b', label: 'Unknown' };
    if (num < min || num > max) return { color: '#dc2626', label: 'ABNORMAL', status: 'alert' };
    return { color: '#059669', label: 'NORMAL', status: 'normal' };
  };

  const getPriorityLabel = (priority) => {
    const prioNum = parseInt(String(priority).replace(/\D/g, '').substring(0, 1));
    const satsPoints = {
      1: '1 - EMERGENT',
      2: '2 - URGENT',
      3: '3 - SEMI-URGENT',
      4: '4 - NON-URGENT',
      5: '5 - NOT ILL'
    };
    return satsPoints[prioNum] || priority;
  };

  const physicianSpecialty = assignedClinicians.find(c => c.role === 'PHYSICIAN')?.specialty;
  const specialtyValue = physicianSpecialty || assignedClinicians[0]?.specialty || manualReport.assigned_specialty || manualReport.assignedSpecialty || triageData.assigned_specialty || triageData.assignedSpecialty || 'N/A';
  console.log('[ConfirmingManual] DEBUG - Specialty resolution:', { physicianSpecialty, specialtyValue });
  const wardDisplay = encounter.ward_display_info?.name ||
    encounter.ward_display_info?.wardName ||
    encounter.ward_display_info?.department ||
    manualReport.assignedWardName ||
    triageData.assignedWardName ||
    encounter.assigned_ward?.name ||
    encounter.assigned_ward?.wardName ||
    encounter.assigned_ward?.department ||
    encounter.assigned_ward ||
    triageData.assigned_ward ||
    encounter.triage?.ward?.name ||
    encounter.triage?.ward?.wardName ||
    'Not Assigned';
  const bedDisplay = encounter.bed_display_info?.bedNumber ||
    encounter.bed_display_info?.bed_number ||
    encounter.bed_display_info?.bedId ||
    manualReport.assignedBedNumber ||
    triageData.assignedBedNumber ||
    encounter.assigned_bed?.bedNumber ||
    encounter.assigned_bed?.bed_number ||
    encounter.assigned_bed?.bedId ||
    encounter.assigned_bed ||
    triageData.assigned_bed ||
    encounter.triage?.bed?.bedNumber ||
    encounter.triage?.bed?.bed_number ||
    'Not Assigned';
  const priorityLevel = normalizeSatsLevel(
    (clinicalAnalysis?.priority_code ?? clinicalAnalysis?.sats_level) ??
    manualReport?.selected_priority ??
    manualReport?.satsLevel ??
    manualReport?.sats_color ??
    encounter?.triage?.manualTriageReport?.satsLevel ??
    encounter?.triage?.priority_code ??
    'yellow'
  );
  const priorityBadge = priorityLevel ? getSATSBadge(priorityLevel) : getSATSBadge('N/A');
  const tewsTotal = Number(
    (clinicalAnalysis?.tews_total !== null && clinicalAnalysis?.tews_total !== undefined) ? clinicalAnalysis.tews_total : (
      manualReport?.tews_total ??
      manualReport?.SATS_reasoning?.tews_total ??
      manualReport?.SATS_reasoning?.total ??
      triageBundle?.SATS_reasoning?.tews_total ??
      triageBundle?.SATS_reasoning?.total ??
      triageBundle?.tews_total ??
      triageData?.SATS_reasoning?.tews_total ??
      triageData?.SATS_reasoning?.total ??
      triageData?.tews_total ??
      null
    )
  );
  // CRITICAL FIX: acuityScore should be the SATS priority level (red/orange/yellow/green or 1-5),
  const mobility = manualReport?.SATS_reasoning?.mobility ?? triageData?.SATS_reasoning?.mobility ?? encounter.vitals?.mobility ?? 'Walks';
  const traumaPresent = Boolean(manualReport?.SATS_reasoning?.traumaPresent ?? triageData?.SATS_reasoning?.traumaPresent ?? encounter.vitals?.traumaPresent);
  // NOT the numeric tews_total or urgencyScore which are numeric values!
  const satsAcuityLevel = normalizeSatsLevel(
    (clinicalAnalysis?.priority_code ?? clinicalAnalysis?.sats_level) ??
    manualReport?.selected_priority ??
    manualReport?.satsLevel ??
    manualReport?.sats_color ??
    triageBundle?.satsLevel ??
    triageBundle?.sats_color ??
    triageData?.satsLevel ??
    triageData?.sats_color ??
    'yellow'
  );
  
  console.log('[ConfirmingManual] 🔍 ACUITY FIELD FIX - Post-normalization check:', {
    tewsTotal: tewsTotal,
    satsAcuityLevel: satsAcuityLevel,
    manualReport_satsLevel: manualReport?.satsLevel,
    manualReport_urgencyScore: manualReport?.urgencyScore,
    clinicalAnalysis_tews_total: clinicalAnalysis?.tews_total,
    issue: 'urgencyScore was incorrectly used as acuity - now using satsLevel instead',
    correctMapping: {
      tewsNumericTotal: tewsTotal,
      acuityLevel: satsAcuityLevel
    }
  });

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20 }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
              Confirm Manual Clinical Handover
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>
              Review and confirm the nurse-verified triage report before handing over to assigned clinicians
            </p>
          </div>
          <button
            onClick={() => navigate(`/summary/${encounterId}`)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 16px',
              background: '#e2e8f0',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600
            }}
          >
            <FiChevronLeft size={16} /> Back to Edit
          </button>
        </div>

        {/* Patient Info */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #e2e8f0' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#0f172a' }}>Patient Information</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, fontSize: 13 }}>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Name:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>
                {encounter.patient?.name || encounter.patient?.firstName} {encounter.patient?.surname}
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>ID:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>{encounter.patient?.id_number || 'N/A'}</div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Age/Gender:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>{encounter.patient?.age} / {encounter.patient?.gender}</div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Chief Complaint:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>{encounter.patient?.symptoms || 'N/A'}</div>
            </div>
          </div>
        </div>

        {/* Patient Vitals */}
        {encounter.vitals && (
          <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #e2e8f0' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#0f172a' }}>📊 Patient Vitals</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, fontSize: 13 }}>
              {encounter.vitals.temp !== undefined && (
                <div style={{ padding: 12, background: '#f1f5f9', borderRadius: 6 }}>
                  <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Temperature</span>
                  <div style={{ color: '#1e293b', marginTop: 4, fontWeight: 600 }}>{encounter.vitals.temp}°C</div>
                </div>
              )}
              {encounter.vitals.hr !== undefined && (
                <div style={{ padding: 12, background: '#f1f5f9', borderRadius: 6 }}>
                  <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Heart Rate</span>
                  <div style={{ color: '#1e293b', marginTop: 4, fontWeight: 600 }}>{encounter.vitals.hr} bpm</div>
                </div>
              )}
              {encounter.vitals.o2 !== undefined && (
                <div style={{ padding: 12, background: '#f1f5f9', borderRadius: 6 }}>
                  <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>O₂ Saturation</span>
                  <div style={{ color: '#1e293b', marginTop: 4, fontWeight: 600 }}>{encounter.vitals.o2}%</div>
                </div>
              )}
              {encounter.vitals.bp_systolic !== undefined && (
                <div style={{ padding: 12, background: '#f1f5f9', borderRadius: 6 }}>
                  <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Blood Pressure (Sys)</span>
                  <div style={{ color: '#1e293b', marginTop: 4, fontWeight: 600 }}>{encounter.vitals.bp_systolic} mmHg</div>
                </div>
              )}
              {encounter.vitals.bp_diastolic !== undefined && (
                <div style={{ padding: 12, background: '#f1f5f9', borderRadius: 6 }}>
                  <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Blood Pressure (Dia)</span>
                  <div style={{ color: '#1e293b', marginTop: 4, fontWeight: 600 }}>{encounter.vitals.bp_diastolic} mmHg</div>
                </div>
              )}
              {encounter.vitals.resp !== undefined && (
                <div style={{ padding: 12, background: '#f1f5f9', borderRadius: 6 }}>
                  <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Respiration Rate</span>
                  <div style={{ color: '#1e293b', marginTop: 4, fontWeight: 600 }}>{encounter.vitals.resp} breaths/min</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Assignment Info */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #fecdd3', borderLeft: '4px solid #f87171' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#be123c' }}>📋 Assignment Information</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, fontSize: 13 }}>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Assigned Specialty:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>
                {specialtyValue}
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Ward:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>{wardDisplay}</div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Bed:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>{bedDisplay}</div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>SATS Priority:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>
                {priorityLevel ? (
                  <span style={{
                    display: 'inline-block',
                    background: priorityBadge.bgColor,
                    color: priorityBadge.color,
                    padding: '6px 10px',
                    borderRadius: 4,
                    border: `1px solid ${priorityBadge.border}`,
                    fontWeight: 700,
                    textTransform: 'uppercase'
                  }}>
                    {priorityBadge.label}
                  </span>
                ) : 'N/A'}
              </div>
            </div>
          </div>
        </div>

        {/* Manual Triage Report */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #fecaca', borderLeft: '4px solid #ef4444' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#991b1b' }}>👨‍⚕️ Nurse-Verified Triage Report</h3>

          {(encounter.triage?.triage_created_at || encounter.triage?.triage_updated_at) && (
            <div style={{ marginBottom: 16, padding: 12, background: '#ecfdf5', borderRadius: 6, borderLeft: '4px solid #10b981' }}>
              <div style={{ fontSize: 12, color: '#047857', fontWeight: 600, marginBottom: 8 }}>⏰ Manual Triage Audit Trail</div>
              {encounter.triage?.triage_created_at && (
                <div style={{ fontSize: 12, color: '#059669', marginBottom: 4 }}>
                  <span style={{ fontWeight: 500 }}>Created:</span> {new Date(encounter.triage.triage_created_at).toLocaleString()}
                </div>
              )}
              {encounter.triage?.triage_updated_at && (
                <div style={{ fontSize: 12, color: '#059669' }}>
                  <span style={{ fontWeight: 500 }}>Updated:</span> {new Date(encounter.triage.triage_updated_at).toLocaleString()}
                </div>
              )}
            </div>
          )}

          <div style={{ marginBottom: 16 }}>
            <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Assigned Specialty</span>
            <div style={{ background: '#fef2f2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
              {specialtyValue}
            </div>
          </div>

          {/* Pathophysiology */}
          {sourceReport?.pathophysiology && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Pathophysiology</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
                {sourceReport.pathophysiology}
              </div>
            </div>
          )}

          {/* Clinical Summary */}
          {sourceReport?.clinical_summary && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Summary</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
                {sourceReport.clinical_summary}
              </div>
            </div>
          )}

          {/* Clinical Discriminators */}
          {sourceReport?.clinical_discriminators && sourceReport.clinical_discriminators.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Discriminators</span>
              <div style={{ background: '#fef3c7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.clinical_discriminators.map((d, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{d}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Differential Diagnoses */}
          {sourceReport?.differential_diagnoses && sourceReport.differential_diagnoses.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>🩺 Differential Diagnoses</span>
              <div style={{ background: '#e0e7ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.differential_diagnoses.map((d, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{typeof d === 'string' ? d : JSON.stringify(d)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Immediate Actions */}
          {sourceReport?.immediate_actions && sourceReport.immediate_actions.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Immediate Actions</span>
              <div style={{ background: '#fee2e2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#7c2d12' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.immediate_actions.map((a, i) => (
                    <li key={i} style={{ marginBottom: 6, fontWeight: 600 }}>{typeof a === 'string' ? a : JSON.stringify(a)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Diagnostic Workup */}
          {sourceReport?.diagnostic_workup && sourceReport.diagnostic_workup.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Diagnostic Workup</span>
              <div style={{ background: '#cffafe', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.diagnostic_workup.map((d, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{typeof d === 'string' ? d : JSON.stringify(d)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Recommendations */}
          {sourceReport?.recommendations && sourceReport.recommendations.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Treatment Recommendations</span>
              <div style={{ background: '#f5f3ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.recommendations.map((r, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{typeof r === 'string' ? r : JSON.stringify(r)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Disposition Rationale */}
          {sourceReport?.disposition_rationale && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Disposition Rationale</span>
              <div style={{ background: '#e0f2fe', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                {sourceReport.disposition_rationale}
              </div>
            </div>
          )}

          {/* Management Actions */}
          {sourceReport?.management_actions && sourceReport.management_actions.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Management Actions</span>
              <div style={{ background: '#fef3c7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.management_actions.map((a, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{typeof a === 'string' ? a : JSON.stringify(a)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Advanced Interventions */}
          {sourceReport?.advanced_interventions && sourceReport.advanced_interventions.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Advanced Interventions</span>
              <div style={{ background: '#dbeafe', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.advanced_interventions.map((item, idx) => (
                    <li key={idx} style={{ marginBottom: 6 }}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Clinical Reasoning Summary */}
          {sourceReport?.clinical_reasoning_summary && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Reasoning Summary</span>
              <div style={{ background: '#dcfce7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
                {sourceReport.clinical_reasoning_summary}
              </div>
            </div>
          )}

          {/* Risk Factors */}
          {sourceReport?.risk_factors && sourceReport.risk_factors.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Risk Factors</span>
              <div style={{ background: '#fef3c7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.risk_factors.map((item, idx) => (
                    <li key={idx} style={{ marginBottom: 6 }}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Pathognomonic Markers */}
          {sourceReport?.pathognomonic_markers && sourceReport.pathognomonic_markers.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Pathognomonic Markers</span>
              <div style={{ background: '#fef3c7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.pathognomonic_markers.map((item, idx) => (
                    <li key={idx} style={{ marginBottom: 6 }}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Life Threats Excluded */}
          {sourceReport?.life_threats_excluded && sourceReport.life_threats_excluded.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Life Threats Excluded</span>
              <div style={{ background: '#fef3c7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.life_threats_excluded.map((item, idx) => (
                    <li key={idx} style={{ marginBottom: 6 }}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Recommended Specialists */}
          {sourceReport?.recommended_specialists && sourceReport.recommended_specialists.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Recommended Specialists</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {sourceReport.recommended_specialists.map((item, idx) => (
                    <li key={idx} style={{ marginBottom: 6 }}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {(Number.isFinite(tewsTotal) || manualReport?.sats_points || manualReport?.sats_points === 0) && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>TEWS Score & Acuity</span>
              <div style={{ background: '#fef2f2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', fontWeight: 600 }}>
                {Number.isFinite(tewsTotal) ? `${tewsTotal} points` : (manualReport?.sats_points ?? 'N/A')}
                {satsAcuityLevel ? ` • Acuity: ${satsAcuityLevel.toUpperCase()}` : ''}
                <div style={{ marginTop: 6, fontSize: 12, fontWeight: 500 }}>Mobility: {mobility} • Trauma modifier: {traumaPresent ? '+1' : '+0'}</div>
              </div>
            </div>
          )}
        </div>

        {/* Assigned Clinicians */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #dcfce7', borderLeft: '4px solid #22c55e' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#166534' }}>👥 Assigned Clinicians</h3>
          
          {manuallySelected.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#059669', marginBottom: 8 }}>📋 Manually Selected from Scheduler:</div>
              <div>
                {manuallySelected.map((c) => (
                  <div key={c.id} style={{ fontSize: 12, padding: 8, background: '#f0fdf4', borderRadius: 4, marginBottom: 6, border: '1px solid #d1fae5' }}>
                    • {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty || 'General'}
                    <div style={{ fontSize: 11, color: '#047857', marginTop: 2 }}>✓ From Scheduler</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {assignedClinicians.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#0369a1', marginBottom: 8 }}>👥 Assigned Clinicians:</div>
              <div>
                {assignedClinicians.map((c) => (
                  <div key={c.id} style={{ fontSize: 12, padding: 8, background: '#eff6ff', borderRadius: 4, marginBottom: 6, border: '1px solid #bfdbfe' }}>
                    • {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty || 'General'}
                  </div>
                ))}
              </div>
            </div>
          )}

          {manuallySelected.length === 0 && assignedClinicians.length === 0 && (
            <div style={{ fontSize: 12, color: '#64748b', padding: 8, background: '#f1f5f9', borderRadius: 4 }}>
              No clinicians selected for this handover
            </div>
          )}
        </div>

        {/* Field Changes Audit Trail */}
        {manualReport.field_changes && Object.keys(manualReport.field_changes).length > 0 && (
          <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #e0e7ff' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#3730a3' }}>📋 Fields Modified</h3>
            <div style={{ fontSize: 12 }}>
              {Object.entries(manualReport.field_changes).map(([field, changes]) => (
                <div key={field} style={{ padding: 8, background: '#f0f9ff', borderRadius: 4, marginBottom: 6, border: '1px solid #bfdbfe' }}>
                  <div style={{ fontWeight: 600, color: '#0284c7' }}>{field}</div>
                  <div style={{ color: '#64748b', marginTop: 4, fontSize: 11 }}>
                    Original: {typeof changes.original === 'string' ? changes.original : JSON.stringify(changes.original).substring(0, 50)}
                  </div>
                  <div style={{ color: '#059669', marginTop: 2, fontSize: 11 }}>
                    Override: {typeof changes.override === 'string' ? changes.override : JSON.stringify(changes.override).substring(0, 50)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 24 }}>
          <button
            onClick={() => navigate('/dashboard')}
            style={{
              padding: '10px 24px',
              background: '#f3f4f6',
              color: '#6b7280',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600
            }}
            disabled={confirming}
            title="Go back to Triage Dashboard"
          >
            ← Back to Dashboard
          </button>
          <button
            onClick={() => navigate(`/summary/${encounterId}`)}
            style={{
              padding: '10px 24px',
              background: '#e2e8f0',
              color: '#1e293b',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600
            }}
            disabled={confirming}
          >
            Go Back & Edit
          </button>
          <button
            onClick={handleConfirmHandover}
            style={{
              padding: '10px 24px',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: confirming ? 'not-allowed' : 'pointer',
              fontSize: 14,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              opacity: confirming ? 0.7 : 1
            }}
            disabled={confirming}
          >
            <FiCheck size={16} /> {confirming ? 'Confirming...' : 'Confirm & Hand Over'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmingManualClinicalHandover;
