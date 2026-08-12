import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiChevronLeft, FiCheck } from 'react-icons/fi';

const ConfirmingManualClinicalHandover = () => {
  const navigate = useNavigate();
  const { encounterId } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [confirming, setConfirming] = useState(false);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

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
      const saveRes = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/ai-fields`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          triage: encounter.triage,
          assigned_clinicians: encounter.assigned_clinicians || [],
          manually_selected_clinicians: encounter.manually_selected_clinicians || [],
          manual_triage_report: encounter.manual_triage_report,
          nurseNotes: encounter.nurseNotes
        })
      });

      if (!saveRes.ok) {
        console.warn('Warning: Save response not ok:', saveRes.status);
        // Continue anyway - handover is the critical operation
      }

      // Then mark encounter as handed over with manual report
      const handoverRes = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          reportType: 'MANUAL_TRIAGE_REPORT',
          clinicians: encounter.assigned_clinicians || [],
          manuallySelected: encounter.manually_selected_clinicians || []
        })
      });

      if (!handoverRes.ok) throw new Error('Failed to confirm handover');

      alert('Clinical handover confirmed! Assigned clinicians have been notified.');
      navigate('/dashboard');
    } catch (err) {
      console.error('Error confirming handover:', err);
      alert('Failed to confirm handover: ' + err.message);
    } finally {
      setConfirming(false);
    }
  };

  if (loading) return <div style={{ padding: 20, textAlign: 'center' }}>Loading...</div>;
  if (!encounter) return <div style={{ padding: 20, textAlign: 'center' }}>Encounter not found</div>;

  const manualReport = encounter.manual_triage_report || encounter.triage?.manualTriageReport || {};
  const triageData = encounter.triage || {};
  const manuallySelected = encounter.manually_selected_clinicians || [];
  const assignedClinicians = encounter.assigned_clinicians || [];

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
                {/* Derive specialty from PHYSICIAN only */}
                {(() => {
                  const allClinicians = [...(encounter.manually_selected_clinicians || []), ...(encounter.assigned_clinicians || [])];
                  const physicianSpecialty = allClinicians.find(c => c.role === 'PHYSICIAN')?.specialty;
                  if (physicianSpecialty) {
                    console.log('[ConfirmingManual] Displaying physician specialty:', physicianSpecialty);
                    return physicianSpecialty;
                  }
                  const fallbackSpec = encounter.triage?.assigned_specialty || 'N/A';
                  console.log('[ConfirmingManual] No physician found, showing triage specialty:', fallbackSpec);
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
                    console.log('[ConfirmingManual] Ward from display_info:', wardDisplay);
                    return wardDisplay;
                  }
                  console.log('[ConfirmingManual] No ward_display_info, assigned_ward ID:', encounter.assigned_ward);
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
                    console.log('[ConfirmingManual] Bed from display_info:', bedDisplay);
                    return bedDisplay;
                  }
                  console.log('[ConfirmingManual] No bed_display_info, assigned_bed ID:', encounter.assigned_bed);
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

        {/* Manual Triage Report */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #fecaca', borderLeft: '4px solid #ef4444' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#991b1b' }}>👨‍⚕️ Nurse-Verified Triage Report</h3>
          
          {/* Manual Triage Timestamps - Audit Trail */}
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
              {manualReport.assigned_specialty || triageData.assigned_specialty || 'N/A'}
            </div>
          </div>

          <div style={{ marginBottom: 16 }}>
            <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>AI Summary (Base)</span>
            <div style={{ background: '#fef2f2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6 }}>
              {manualReport.ai_summary || 'No summary available'}
            </div>
          </div>

          {manualReport.discriminators && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Clinical Discriminators (Verified)</span>
              <div style={{ background: '#fef2f2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                {Array.isArray(manualReport.discriminators) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {manualReport.discriminators.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                ) : (
                  manualReport.discriminators
                )}
              </div>
            </div>
          )}

          {manualReport.recommendations && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Recommendations</span>
              <div style={{ background: '#fef2f2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13 }}>
                {Array.isArray(manualReport.recommendations) ? (
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {manualReport.recommendations.map((r, i) => {
                      const recommendation = typeof r === 'string' ? r : JSON.stringify(r);
                      return (
                        <li key={i}>{recommendation}</li>
                      );
                    })}
                  </ul>
                ) : (
                  typeof manualReport.recommendations === 'string' ? manualReport.recommendations : JSON.stringify(manualReport.recommendations)
                )}
              </div>
            </div>
          )}

          {manualReport.sats_points && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>SATS Points</span>
              <div style={{ background: '#fef2f2', padding: 12, borderRadius: 6, marginTop: 6, fontSize: 13, color: '#1e293b', fontWeight: 600 }}>
                {manualReport.sats_points} {manualReport.sats_color ? `- ${manualReport.sats_color}` : ''}
              </div>
            </div>
          )}
        </div>

        {/* Nurse Notes */}
        {encounter.nurseNotes && (
          <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #fed7aa', borderLeft: '4px solid #f59e0b' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#92400e' }}>📝 Nurse Notes</h3>
            <div style={{ background: '#fffbeb', padding: 12, borderRadius: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
              {encounter.nurseNotes}
            </div>
          </div>
        )}

        {/* Assigned Clinicians */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #dcfce7', borderLeft: '4px solid #22c55e' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#166534' }}>👥 Assigned Clinicians</h3>
          
          {manuallySelected.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#059669', marginBottom: 8 }}>📋 Manually Selected from Scheduler:</div>
              <div>
                {manuallySelected.map((c) => (
                  <div key={c.id} style={{ fontSize: 12, padding: 8, background: '#f0fdf4', borderRadius: 4, marginBottom: 6, border: '1px solid #d1fae5' }}>
                    • {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty}
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

        {/* Nurse Notes */}
        {manualReport.notes && (
          <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #fed7aa', borderLeft: '4px solid #f59e0b' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#92400e' }}>📝 Nurse Notes</h3>
            <div style={{ background: '#fffbeb', padding: 12, borderRadius: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
              {manualReport.notes}
            </div>
          </div>
        )}

        {!manualReport.notes && encounter.nurseNotes && (
          <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #fed7aa', borderLeft: '4px solid #f59e0b' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#92400e' }}>📝 Nurse Notes</h3>
            <div style={{ background: '#fffbeb', padding: 12, borderRadius: 6, fontSize: 13, color: '#1e293b', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
              {encounter.nurseNotes}
            </div>
          </div>
        )}

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
