import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiChevronLeft, FiCheckCircle, FiClock, FiAlertCircle } from 'react-icons/fi';

const ViewClinicalHandover = () => {
  const navigate = useNavigate();
  const { encounterId } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [acknowledging, setAcknowledging] = useState(false);
  const [userRole, setUserRole] = useState(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

  // Decode JWT to get user info
  const getUserFromToken = () => {
    try {
      if (!token) return null;
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      const decoded = JSON.parse(atob(parts[1]));
      return decoded;
    } catch (e) {
      console.error('Error decoding token:', e);
      return null;
    }
  };

  useEffect(() => {
    const user = getUserFromToken();
    if (user) {
      setUserRole(user.role);
      console.log('[ViewHandover] Current user role:', user.role);
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
        setEncounter(data.encounter);
        console.log('[ViewHandover] Encounter loaded:', data.encounter._id);
      } catch (err) {
        console.error('Error fetching encounter:', err);
        alert('Failed to load handover report');
      } finally {
        setLoading(false);
      }
    };

    if (encounterId && token) fetchEncounter();
  }, [encounterId, token, BACKEND_URL]);

  const handleAcknowledge = async () => {
    try {
      setAcknowledging(true);
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      };

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover/acknowledge`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          status: 'ACKNOWLEDGED',
          notes: `${userRole} acknowledged handover`
        })
      });

      if (!res.ok) {
        throw new Error('Failed to acknowledge handover');
      }

      console.log('[ViewHandover] Handover acknowledged successfully');
      alert('Handover acknowledged. Notification sent to triage nurse.');
      navigate('/dashboard');
    } catch (error) {
      console.error('[ViewHandover] Error:', error);
      alert('Failed to acknowledge handover');
    } finally {
      setAcknowledging(false);
    }
  };

  const handleMarkBusy = async () => {
    try {
      setAcknowledging(true);
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      };

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover/acknowledge`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          status: 'IN_PROGRESS',
          notes: `${userRole} started working on patient`
        })
      });

      if (!res.ok) {
        throw new Error('Failed to update status');
      }

      console.log('[ViewHandover] Status updated to IN_PROGRESS');
      alert('Status updated. Triage nurse has been notified.');
      navigate('/dashboard');
    } catch (error) {
      console.error('[ViewHandover] Error:', error);
      alert('Failed to update status');
    } finally {
      setAcknowledging(false);
    }
  };

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontSize: 16, color: '#64748b' }}>Loading handover report...</div>
      </div>
    );
  }

  if (!encounter) {
    return (
      <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20 }}>
        <div style={{ maxWidth: 1000, margin: '0 auto', textAlign: 'center' }}>
          <h2 style={{ color: '#0f172a', marginBottom: 12 }}>Handover not found</h2>
          <button
            onClick={() => navigate('/dashboard')}
            style={{
              padding: '10px 24px',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600
            }}
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const triageData = encounter.triage || {};
  const aiReport = triageData.aiTriageReport || {};

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20 }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
              Clinical Handover Report
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>
              Patient triage report from {encounter.original_triage_nurse?.nurse_name || 'Assigned Nurse'}
            </p>
          </div>
          <button
            onClick={() => navigate('/dashboard')}
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
            <FiChevronLeft size={16} /> Back to Dashboard
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

        {/* Triage Nurse Information */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #dbeafe', borderLeft: '4px solid #0284c7' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#0369a1' }}>Triage Nurse</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 13 }}>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Name:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>{encounter.original_triage_nurse?.nurse_name || 'Unknown'}</div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Triaged At:</span>
              <div style={{ color: '#1e293b', marginTop: 4 }}>
                {encounter.original_triage_nurse?.triaged_at 
                  ? new Date(encounter.original_triage_nurse.triaged_at).toLocaleString()
                  : 'N/A'}
              </div>
            </div>
          </div>
        </div>

        {/* AI Triage Report */}
        <div style={{ background: 'white', padding: 16, borderRadius: 8, marginBottom: 16, border: '1px solid #dbeafe', borderLeft: '4px solid #3b82f6' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#1e40af' }}>AI Triage Report</h3>
          
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
              <span style={{ color: '#64748b', fontWeight: 600, fontSize: 12 }}>Differential Diagnoses</span>
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
            disabled={acknowledging}
          >
            Back to Dashboard
          </button>
          <button
            onClick={handleAcknowledge}
            style={{
              padding: '10px 24px',
              background: '#f59e0b',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: acknowledging ? 'not-allowed' : 'pointer',
              fontSize: 14,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              opacity: acknowledging ? 0.7 : 1
            }}
            disabled={acknowledging}
          >
            <FiClock size={16} /> {acknowledging ? 'Acknowledging...' : 'Acknowledge'}
          </button>
          <button
            onClick={handleMarkBusy}
            style={{
              padding: '10px 24px',
              background: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: acknowledging ? 'not-allowed' : 'pointer',
              fontSize: 14,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              opacity: acknowledging ? 0.7 : 1
            }}
            disabled={acknowledging}
          >
            <FiCheckCircle size={16} /> {acknowledging ? 'Updating...' : 'Mark as Busy'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ViewClinicalHandover;
