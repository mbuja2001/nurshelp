import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiChevronLeft, FiCheck } from 'react-icons/fi';

const ConfirmingAIClinicalHandover = () => {
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
      console.log('[ConfirmingAI] Decoded user from token:', {
        id: decoded.id,
        name: decoded.name,
        email: decoded.email,
        role: decoded.role,
        hospitalId: decoded.hospitalId
      });
      return {
        id: decoded.id,
        name: decoded.name,
        email: decoded.email,
        role: decoded.role,
        hospitalId: decoded.hospitalId
      };
    } catch (err) {
      console.error('[ConfirmingAI] Error decoding token:', err);
      return null;
    }
  };

  // On mount, decode token to identify triage nurse
  useEffect(() => {
    if (token) {
      const nurse = getUserFromToken(token);
      if (nurse) {
        setTriageNurse(nurse);
        console.log('[ConfirmingAI] Triage nurse set:', nurse.name, `<${nurse.email}>`);
      }
    }
  }, [token]);

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
        const enc = data.encounter;
        
        // DEBUG: Log what we received from backend for confirmation
        console.log('[ConfirmingAI] Encounter loaded:', enc._id);
        console.log('[ConfirmingAI] assigned_specialty:', enc.triage?.assigned_specialty);
        console.log('[ConfirmingAI] assigned_clinicians (top-level):', enc.assigned_clinicians); // Top-level, not in triage!
        console.log('[ConfirmingAI] manually_selected_clinicians:', enc.manually_selected_clinicians);
        console.log('[ConfirmingAI] assigned_ward:', enc.assigned_ward, '| ID:', enc.assigned_ward?._id || enc.assigned_ward?.id);
        console.log('[ConfirmingAI] assigned_bed:', enc.assigned_bed, '| ID:', enc.assigned_bed?._id || enc.assigned_bed?.id);
        console.log('[ConfirmingAI] ward_display_info:', enc.ward_display_info);
        console.log('[ConfirmingAI] bed_display_info:', enc.bed_display_info);
        
        setEncounter(enc);
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
      console.log('[CONFIRM_HANDOVER] ========== INITIATING HANDOVER ==========');
      console.log('[CONFIRM_HANDOVER] Encounter ID:', encounterId);
      console.log('[CONFIRM_HANDOVER] Triage Nurse:', triageNurse?.name, `<${triageNurse?.email}>`);
      
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      };

      // First, save the encounter data with AI fields and triage nurse info
      console.log('[CONFIRM_HANDOVER] Saving encounter AI fields with triage nurse...');
      const saveRes = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/ai-fields`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          triage: encounter.triage,
          assigned_clinicians: encounter.assigned_clinicians || [],
          manually_selected_clinicians: encounter.manually_selected_clinicians || [],
          nurseNotes: encounter.nurseNotes,
          triage_nurse: triageNurse ? {
            nurse_id: triageNurse.id,
            nurse_name: triageNurse.name,
            nurse_email: triageNurse.email,
            nurse_role: triageNurse.role,
            triaged_at: new Date().toISOString()
          } : null
        })
      });

      if (!saveRes.ok) {
        console.warn('[CONFIRM_HANDOVER] Save response not ok:', saveRes.status);
        // Continue anyway - handover is the critical operation
      } else {
        console.log('[CONFIRM_HANDOVER] AI fields and triage nurse saved');
      }

      // Then mark encounter as handed over with AI report
      console.log('[CONFIRM_HANDOVER] Calling handover endpoint...');
      console.log('[CONFIRM_HANDOVER] Clinicians:', encounter.assigned_clinicians?.length || 0);
      console.log('[CONFIRM_HANDOVER] Manually selected:', encounter.manually_selected_clinicians?.length || 0);
      console.log('[CONFIRM_HANDOVER] All clinicians:', JSON.stringify(encounter.assigned_clinicians, null, 2));
      
      const handoverRes = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          reportType: 'AI_TRIAGE_REPORT',
          clinicians: encounter.assigned_clinicians || [],
          manuallySelected: encounter.manually_selected_clinicians || [],
          triageNurse: triageNurse
        })
      });

      if (!handoverRes.ok) throw new Error('Failed to create notifications');
      const handoverData = await handoverRes.json();
      
      console.log('[CONFIRM_HANDOVER] Full response:', handoverData);
      const notifCount = handoverData.notificationsCreated || handoverData.clinicians?.length || 0;
      console.log('[CONFIRM_HANDOVER] ✅ Handover successful! Notifications created for', notifCount, 'clinicians');
      alert('Handover confirmed! Notifications sent to ' + notifCount + ' clinicians.');
      
      navigate('/dashboard');
    } catch (err) {
      console.error('[CONFIRM_HANDOVER] ❌ Error:', err);
      alert('Failed to confirm handover: ' + err.message);
    } finally {
      setConfirming(false);
    }
  };

  if (loading) return <div style={{ padding: 20, textAlign: 'center' }}>Loading...</div>;
  if (!encounter) return <div style={{ padding: 20, textAlign: 'center' }}>Encounter not found</div>;

  const triageData = encounter.triage || {};
  const aiReport = triageData.aiTriageReport || {};
  const manuallySelected = encounter.manually_selected_clinicians || [];
  const aiRecommended = encounter.assigned_clinicians || []; // FIXED: Read from top-level, not from triage!

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20 }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
              Confirm AI Clinical Handover
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>
              Review and confirm the AI-generated triage report before handing over to assigned clinicians
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
            <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#0f172a' }}>Patient Vitals</h3>
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
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #ddd6fe', borderLeft: '4px solid #6366f1' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#4f46e5' }}>Assignment Information</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, fontSize: 13 }}>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Assigned Specialty:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>
                {/* Derive specialty from PHYSICIAN only (matching PatientReport logic) */}
                {(() => {
                  const allClinicians = [...(encounter.manually_selected_clinicians || []), ...(encounter.assigned_clinicians || [])];
                  const physicianSpecialty = allClinicians.find(c => c.role === 'PHYSICIAN')?.specialty;
                  if (physicianSpecialty) {
                    console.log('[ConfirmingAI] Displaying physician specialty:', physicianSpecialty);
                    return physicianSpecialty;
                  }
                  const fallbackSpec = encounter.triage?.assigned_specialty || 'N/A';
                  console.log('[ConfirmingAI] No physician found, showing triage specialty:', fallbackSpec);
                  return fallbackSpec;
                })()}
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Ward:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>
                {(() => {
                  const wardDisplay = encounter.ward_display_info?.name || 
                    encounter.ward_display_info?.wardName || 
                    encounter.ward_display_info?.department;
                  if (wardDisplay) {
                    console.log('[ConfirmingAI] Ward from display_info:', wardDisplay);
                    return wardDisplay;
                  }
                  console.log('[ConfirmingAI] No ward_display_info, assigned_ward ID:', encounter.assigned_ward);
                  return 'Not Assigned';
                })()}
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Bed:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>
                {(() => {
                  const bedDisplay = encounter.bed_display_info?.bedNumber || 
                    encounter.bed_display_info?.bed_number ||
                    encounter.bed_display_info?.bedId;
                  if (bedDisplay) {
                    console.log('[ConfirmingAI] Bed from display_info:', bedDisplay);
                    return bedDisplay;
                  }
                  console.log('[ConfirmingAI] No bed_display_info, assigned_bed ID:', encounter.assigned_bed);
                  return 'Not Assigned';
                })()}
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Priority:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>{encounter.triage?.priority_code || 'N/A'}</div>
            </div>
          </div>
        </div>
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #dbeafe', borderLeft: '4px solid #3b82f6' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#1e40af' }}>AI Triage Report</h3>
          
          {/* Triage Timestamps - Audit Trail */}
          {(encounter.triage?.triage_created_at || encounter.triage?.inference_timestamp) && (
            <div style={{ marginBottom: 16, padding: 12, background: '#ecfdf5', borderRadius: 6, borderLeft: '4px solid #10b981' }}>
              <div style={{ fontSize: 12, color: '#047857', fontWeight: 600, marginBottom: 8 }}>⏰ Triage Audit Trail</div>
              {encounter.triage?.triage_created_at && (
                <div style={{ fontSize: 12, color: '#059669', marginBottom: 4 }}>
                  <span style={{ fontWeight: 500 }}>Created:</span> {new Date(encounter.triage.triage_created_at).toLocaleString()}
                </div>
              )}
              {encounter.triage?.inference_timestamp && (
                <div style={{ fontSize: 12, color: '#059669' }}>
                  <span style={{ fontWeight: 500 }}>Inference:</span> {new Date(encounter.triage.inference_timestamp).toLocaleString()}
                </div>
              )}
            </div>
          )}
          
          {/* Priority Code */}
          {triageData.priority_code && (
            <div style={{ marginBottom: 16, padding: 12, background: '#dbeafe', borderRadius: 6, borderLeft: '4px solid #3b82f6' }}>
              <span style={{ color: '#1e40af', fontWeight: 700, fontSize: 13 }}>Priority Code: {triageData.priority_code}</span>
            </div>
          )}

          {/* Pathophysiology */}
          {aiReport?.pathophysiology && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Pathophysiology</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
                {aiReport.pathophysiology}
              </div>
            </div>
          )}

          {/* Clinical Summary */}
          {aiReport?.clinical_summary && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Summary</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
                {aiReport.clinical_summary}
              </div>
            </div>
          )}

          {/* Clinical Discriminators */}
          {aiReport?.clinical_discriminators && aiReport.clinical_discriminators.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Discriminators</span>
              <div style={{ background: '#fef3c7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {aiReport.clinical_discriminators.map((d, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{d}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Differential Diagnoses */}
          {aiReport?.differential_diagnoses && aiReport.differential_diagnoses.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>🩺 Differential Diagnoses</span>
              <div style={{ background: '#e0e7ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {aiReport.differential_diagnoses.map((d, i) => {
                    const diagnosis = typeof d === 'string' ? d : (d?.dx || JSON.stringify(d));
                    return <li key={i} style={{ marginBottom: 6 }}>{diagnosis}</li>;
                  })}
                </ul>
              </div>
            </div>
          )}

          {/* Immediate Actions */}
          {aiReport?.immediate_actions && aiReport.immediate_actions.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Immediate Actions</span>
              <div style={{ background: '#fee2e2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#7c2d12' }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {aiReport.immediate_actions.map((a, i) => (
                    <li key={i} style={{ marginBottom: 6, fontWeight: 600 }}>{a}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Diagnostic Workup */}
          {aiReport?.diagnostic_workup && aiReport.diagnostic_workup.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Diagnostic Workup</span>
              <div style={{ background: '#cffafe', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {aiReport.diagnostic_workup.map((d, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{d}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Recommendations */}
          {aiReport?.recommendations && aiReport.recommendations.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Treatment Recommendations</span>
              <div style={{ background: '#f5f3ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {aiReport.recommendations.map((r, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Disposition Rationale */}
          {aiReport?.disposition_rationale && (
            <div>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Disposition Rationale</span>
              <div style={{ background: '#e0f2fe', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                {aiReport.disposition_rationale}
              </div>
            </div>
          )}
          {aiReport?.clinical_findings && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Findings</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
                {Array.isArray(aiReport.clinical_findings) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {aiReport.clinical_findings.map((f, i) => (
                      <li key={i}>{typeof f === 'string' ? f : JSON.stringify(f)}</li>
                    ))}
                  </ul>
                ) : (
                  aiReport.clinical_findings
                )}
              </div>
            </div>
          )}

          {aiReport?.clinical_analysis && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Analysis</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                {aiReport.clinical_analysis}
              </div>
            </div>
          )}

          {triageData.sats_classification && (
            <div style={{ marginBottom: 16, padding: 12, background: '#f0fdf4', borderRadius: 6, borderLeft: '4px solid #22c55e' }}>
              <span style={{ color: '#166534', fontWeight: 700, fontSize: 13 }}>SATS Classification & Points</span>
              <div style={{ color: '#1e293b', marginTop: 6, fontSize: 13 }}>
                {triageData.sats_classification}
              </div>
            </div>
          )}

          {aiReport?.clinical_discriminators && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Discriminators</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                {Array.isArray(aiReport.clinical_discriminators) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {aiReport.clinical_discriminators.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                ) : (
                  aiReport.clinical_discriminators
                )}
              </div>
            </div>
          )}

          {aiReport?.differential_diagnoses && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Differential Diagnoses</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                {Array.isArray(aiReport.differential_diagnoses) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {aiReport.differential_diagnoses.map((d, i) => {
                      const diagnosis = typeof d === 'string' ? d : (d?.dx || JSON.stringify(d));
                      return (
                        <li key={i}>{diagnosis}</li>
                      );
                    })}
                  </ul>
                ) : (
                  typeof aiReport.differential_diagnoses === 'string' ? aiReport.differential_diagnoses : JSON.stringify(aiReport.differential_diagnoses)
                )}
              </div>
            </div>
          )}

          {/* Management Actions */}
          {aiReport?.management_actions && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Management Actions</span>
              <div style={{ background: '#fef3c7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                {Array.isArray(aiReport.management_actions) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {aiReport.management_actions.map((a, i) => (
                      <li key={i}>{typeof a === 'string' ? a : JSON.stringify(a)}</li>
                    ))}
                  </ul>
                ) : (
                  typeof aiReport.management_actions === 'string' ? aiReport.management_actions : JSON.stringify(aiReport.management_actions)
                )}
              </div>
            </div>
          )}

          {/* Diagnostic Workup */}
          {aiReport?.diagnostic_workup && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Diagnostic Workup</span>
              <div style={{ background: '#f3e8ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                {Array.isArray(aiReport.diagnostic_workup) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {aiReport.diagnostic_workup.map((d, i) => (
                      <li key={i}>{typeof d === 'string' ? d : JSON.stringify(d)}</li>
                    ))}
                  </ul>
                ) : (
                  typeof aiReport.diagnostic_workup === 'string' ? aiReport.diagnostic_workup : JSON.stringify(aiReport.diagnostic_workup)
                )}
              </div>
            </div>
          )}

          {/* Advanced Interventions */}
          {aiReport?.advanced_interventions && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Advanced Interventions</span>
              <div style={{ background: '#dbeafe', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b' }}>
                {Array.isArray(aiReport.advanced_interventions) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {aiReport.advanced_interventions.map((i, idx) => (
                      <li key={idx}>{typeof i === 'string' ? i : JSON.stringify(i)}</li>
                    ))}
                  </ul>
                ) : (
                  typeof aiReport.advanced_interventions === 'string' ? aiReport.advanced_interventions : JSON.stringify(aiReport.advanced_interventions)
                )}
              </div>
            </div>
          )}

          {/* Clinical Reasoning Summary */}
          {aiReport?.clinical_reasoning_summary && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Reasoning Summary</span>
              <div style={{ background: '#dcfce7', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
                {aiReport.clinical_reasoning_summary}
              </div>
            </div>
          )}

          {aiReport?.recommendations && (
            <div>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Recommendations & Next Steps</span>
              <div style={{ background: '#eff6ff', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                {Array.isArray(aiReport.recommendations) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {aiReport.recommendations.map((r, i) => (
                      <li key={i}>{typeof r === 'string' ? r : JSON.stringify(r)}</li>
                    ))}
                  </ul>
                ) : (
                  typeof aiReport.recommendations === 'string' ? aiReport.recommendations : JSON.stringify(aiReport.recommendations)
                )}
              </div>
            </div>
          )}
        </div>

        {/* Nurse Notes */}
        {encounter.nurseNotes && (
          <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #fed7aa', borderLeft: '4px solid #f59e0b' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#92400e' }}>Nurse Notes</h3>
            <div style={{ background: '#fffbeb', padding: 12, borderRadius: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
              {encounter.nurseNotes}
            </div>
          </div>
        )}

        {/* Triage Nurse Information - Extracted from JWT Token */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #dbeafe', borderLeft: '4px solid #0284c7' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#0369a1' }}>Triage Nurse</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 13 }}>
            {triageNurse ? (
              <>
                <div>
                  <span style={{ color: '#64748b', fontWeight: 600 }}>Name:</span>
                  <div style={{ color: '#1e293b', marginTop: 4 }}>{triageNurse.name}</div>
                </div>
                <div>
                  <span style={{ color: '#64748b', fontWeight: 600 }}>Email:</span>
                  <div style={{ color: '#1e293b', marginTop: 4 }}>{triageNurse.email}</div>
                </div>
                <div>
                  <span style={{ color: '#64748b', fontWeight: 600 }}>Role:</span>
                  <div style={{ color: '#1e293b', marginTop: 4 }}>{triageNurse.role}</div>
                </div>
                <div>
                  <span style={{ color: '#64748b', fontWeight: 600 }}>Hospital ID:</span>
                  <div style={{ color: '#1e293b', marginTop: 4 }}>{triageNurse.hospitalId}</div>
                </div>
              </>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 12 }}>
                Unable to extract triage nurse from token. Make sure you're logged in.
              </div>
            )}
          </div>
        </div>

        {/* Assigned Clinicians */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #dcfce7', borderLeft: '4px solid #22c55e' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#166534' }}>Assigned Clinicians</h3>
          
          {/* Manually Selected */}
          {manuallySelected.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#059669', marginBottom: 8 }}>Manually Selected from Scheduler:</div>
              <div>
                {manuallySelected.map((c) => (
                  <div key={c.id} style={{ fontSize: 12, padding: 8, background: '#f0fdf4', borderRadius: 4, marginBottom: 6, border: '1px solid #d1fae5' }}>
                    • {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty}
                    <div style={{ fontSize: 11, color: '#047857', marginTop: 2 }}>From Scheduler</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI Recommended */}
          {aiRecommended.length > 0 && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#0369a1', marginBottom: 8 }}>AI Recommended:</div>
              <div>
                {aiRecommended.map((c) => (
                  <div key={c.id} style={{ fontSize: 12, padding: 8, background: '#f0f9ff', borderRadius: 4, marginBottom: 6, border: '1px solid #bfdbfe' }}>
                    • {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

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
              background: '#3b82f6',
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
            <FiCheck size={16} /> {confirming ? 'Sending Notifications...' : 'Confirm & Send Handover'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmingAIClinicalHandover;
