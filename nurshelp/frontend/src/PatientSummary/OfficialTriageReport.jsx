import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiChevronLeft, FiDownload, FiEdit } from 'react-icons/fi';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const OfficialTriageReport = () => {
  const navigate = useNavigate();
  const { encounterId } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [triageNurse, setTriageNurse] = useState(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

  // Decode JWT token to extract triage nurse
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
      console.error('[OfficialReport] Error decoding token:', err);
      return null;
    }
  };

  useEffect(() => {
    if (token) {
      const nurse = getUserFromToken(token);
      if (nurse) {
        setTriageNurse(nurse);
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
        
        console.log('[OfficialReport] Encounter loaded:', enc._id);
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

  // Determine report type - matches backend logic
  const getReportType = () => {
    if (encounter?.manual_triage_report?.selected_priority) {
      return 'MANUAL';
    }
    if (encounter?.ai_triage_report?.calculated_priority || encounter?.triage?.calculated_priority) {
      return 'AI';
    }
    return 'AI'; // Default to AI
  };

  // Get destination specialty (prioritize physician specialist)
  const getDestinationSpecialty = () => {
    const allClinicians = [...(encounter?.manually_selected_clinicians || []), ...(encounter?.assigned_clinicians || [])];
    // First, look for physician specialist
    const physician = allClinicians.find(c => c.role === 'PHYSICIAN');
    if (physician && physician.specialty) {
      return physician.specialty;
    }
    // Fallback to first clinician specialty
    if (allClinicians.length > 0 && allClinicians[0].specialty) {
      return allClinicians[0].specialty;
    }
    // Final fallback to triage assigned specialty
    return encounter?.triage?.assigned_specialty || 'Unassigned';
  };

  // Get priority badge color and label
  const getPriorityBadge = (priority) => {
    const priMap = {
      'ESI-1': { color: '#dc2626', bgColor: '#fee2e2', label: 'EMERGENT', border: '#991b1b' },
      'ESI-2': { color: '#ea580c', bgColor: '#ffedd5', label: 'URGENT', border: '#9a3412' },
      'ESI-3': { color: '#b45309', bgColor: '#fef3c7', label: 'MODERATE', border: '#92400e' },
      'ESI-4': { color: '#059669', bgColor: '#d1fae5', label: 'STABLE', border: '#047857' },
      'ESI-5': { color: '#0891b2', bgColor: '#cffafe', label: 'MINIMAL', border: '#164e63' }
    };
    return priMap[priority] || { color: '#475569', bgColor: '#f1f5f9', label: priority || 'N/A', border: '#94a3b8' };
  };

  // Determine vital sign color (abnormal = red/orange, normal = green)
  const getVitalColor = (value, min, max) => {
    const num = parseFloat(value);
    if (num < min || num > max) return '#dc2626'; // Red for abnormal
    return '#059669'; // Green for normal
  };
  // Get SATS (South African Triage Scale) color coding - SIMPLIFIED
  const getSATSBadge = (priority) => {
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

  const normalizeSatsLevel = (value) => {
    if (value === null || value === undefined || value === '') return null;
    const asString = String(value).trim();
    const lowered = asString.toLowerCase();
    if (['red', 'orange', 'yellow', 'green'].includes(lowered)) return lowered;
    const numeric = Number(asString.replace(/[^\d]/g, ''));
    if (numeric === 1) return 'red';
    if (numeric === 2) return 'orange';
    if (numeric === 3) return 'yellow';
    if (numeric === 4) return 'green';
    return null;
  };

  // Get vitals with flexible property names
  const getVitalValue = (property) => {
    if (!encounter?.vitals) return null;
    const v = encounter.vitals;
    // Handle different property name variations
    if (property === 'temp') return v.temp || v.temperature;
    if (property === 'hr') return v.hr || v.heart_rate || v.heartRate;
    if (property === 'o2') return v.o2 || v.o2_sat || v.oxygen || v.o2Sat;
    if (property === 'sysBP') return v.sysBP || v.sys_bp || v.systolic || v.bp_systolic || (v.bp && v.bp.systolic);
    if (property === 'diaBP') return v.diaBP || v.dia_bp || v.diastolic || v.bp_diastolic || (v.bp && v.bp.diastolic);
    if (property === 'rr') return v.rr || v.resp || v.respiration_rate || v.respirationRate;
    return null;
  };
  // Download PDF optimized for A4 single page
  const handleDownloadPDF = async () => {
    try {
      setDownloading(true);
      const reportElement = document.getElementById('official-report-content');
      
      if (!reportElement) {
        alert('Report content not found');
        return;
      }

      const canvas = await html2canvas(reportElement, {
        scale: 2,
        backgroundColor: '#ffffff',
        useCORS: true,
        logging: false,
        allowTaint: true
      });

      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 10;
      const contentWidth = pageWidth - (2 * margin);
      
      const imgData = canvas.toDataURL('image/png');
      const canvasHeight = canvas.height;
      const canvasWidth = canvas.width;
      const aspectRatio = canvasHeight / canvasWidth;
      
      const maxHeight = pageHeight - (2 * margin);
      let imgHeight = contentWidth * aspectRatio;
      
      if (imgHeight > maxHeight) {
        imgHeight = maxHeight;
      }

      pdf.addImage(imgData, 'PNG', margin, margin, contentWidth, imgHeight);

      const patientName = encounter?.patient?.name || encounter?.patient?.firstName || 'Patient';
      const filename = `TriageReport_${patientName}_${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(filename);

      console.log('[OfficialReport] PDF downloaded:', filename);
      alert('PDF downloaded successfully!');
    } catch (err) {
      console.error('Error downloading PDF:', err);
      alert('Failed to download PDF: ' + err.message);
    } finally {
      setDownloading(false);
    }
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#f8fafc',
        fontSize: 16,
        color: '#64748b'
      }}>
        Loading report...
      </div>
    );
  }

  if (!encounter) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#f8fafc',
        fontSize: 16,
        color: '#ef4444'
      }}>
        Failed to load encounter
      </div>
    );
  }

  const reportType = getReportType();
  const destinationSpecialty = getDestinationSpecialty();
  const triageData = encounter?.triage || {};
  const aiReport = encounter?.ai_triage_report || triageData.aiTriageReport || {};
  const manualReport = encounter?.manual_triage_report || {};
  const manuallySelected = encounter?.manually_selected_clinicians || [];
  const aiRecommended = encounter?.assigned_clinicians || [];
  const satsPriority = normalizeSatsLevel(
    reportType === 'AI'
      ? (aiReport?.calculated_priority ?? triageData?.priority_code ?? triageData?.SATS_reasoning?.priority ?? 'yellow')
      : (manualReport?.selected_priority ?? manualReport?.satsLevel ?? triageData?.priority_code ?? triageData?.manualTriageReport?.satsLevel ?? 'yellow')
  );
  const satsBadge = satsPriority ? getSATSBadge(satsPriority) : getSATSBadge('N/A');
  const tewsTotal = reportType === 'AI'
    ? (triageData?.SATS_reasoning?.tews_total ?? triageData?.tews_total ?? aiReport?.tews_total ?? null)
    : (triageData?.SATS_reasoning?.tews_total ?? triageData?.tews_total ?? manualReport?.tews_total ?? null);
  const acuityScore = reportType === 'AI'
    ? (aiReport?.urgencyScore ?? triageData?.urgencyScore ?? null)
    : (manualReport?.urgencyScore ?? manualReport?.urgency_score ?? triageData?.urgencyScore ?? triageData?.urgency_score ?? null);

  return (
    <div style={{ background: '#f8fafc', minHeight: '100vh', padding: '20px' }}>
      <div style={{ maxWidth: '210mm', margin: '0 auto' }}>
        {/* Header with back button */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 24,
          background: 'white',
          padding: '16px 20px',
          borderRadius: 8,
          border: '1px solid #e2e8f0'
        }}>
          <div>
            <h1 style={{
              margin: '0 0 4px 0',
              fontSize: 24,
              fontWeight: 700,
              color: '#0f172a'
            }}>
              Official Triage Report
            </h1>
            <p style={{
              margin: 0,
              fontSize: 14,
              color: '#64748b'
            }}>
              {reportType === 'MANUAL' ? 'Manual Nurse Triage' : 'AI-Assisted'} triage report
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

        {/* Report Content - Optimized for A4 PDF - COMPACT SINGLE PAGE */}
        <div id="official-report-content" style={{ 
          background: 'white', 
          padding: '11px', 
          borderRadius: 0,
          fontSize: '11px',
          lineHeight: '1.35',
          maxWidth: '210mm',
          margin: '0 auto'
        }}>
          
          {/* HEADER SECTION - Title + Metadata + Patient Name */}
          <div style={{
            marginBottom: '9px',
            paddingBottom: '7px',
            borderBottom: '3px solid #1e293b'
          }}>
            <h2 style={{ margin: '0 0 3px 0', fontSize: '15px', fontWeight: 600, color: '#000000', letterSpacing: '0.5px' }}>
              OFFICIAL TRIAGE REPORT
            </h2>
            <div style={{ fontSize: '10px', color: '#1e293b', marginBottom: '5px', fontWeight: 500 }}>
              {reportType === 'MANUAL' ? 'MANUALLY CONFIRMED' : 'AI-ASSISTED'} • {new Date(encounter.createdAt).toLocaleString()}
            </div>

            {/* Compact 3-column metadata - Patient, Triage Nurse, Destination */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr 1fr',
              gap: '9px',
              fontSize: '10px',
              marginBottom: '5px'
            }}>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>PATIENT:</span>
                <div style={{ color: '#000000', fontWeight: 500, fontSize: '10px', marginTop: '2px' }}>
                  {encounter.patient?.name || encounter.patient?.firstName} {encounter.patient?.surname || ''}
                </div>
                <div style={{ color: '#1e293b', fontWeight: 600, fontSize: '9px', marginTop: '3px' }}>ID: {encounter.patient?.id_number || 'N/A'}</div>
              </div>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>TRIAGE NURSE:</span>
                <div style={{ color: '#000000', fontWeight: 500, fontSize: '10px', marginTop: '2px' }}>{triageNurse?.name || 'Unknown'}</div>
                <div style={{ color: '#1e293b', fontWeight: 600, fontSize: '9px', marginTop: '3px' }}>Email: {triageNurse?.email || 'N/A'}</div>
              </div>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>DESTINATION:</span>
                <div style={{ color: '#000000', fontWeight: 500, fontSize: '10px', marginTop: '2px' }}>{destinationSpecialty}</div>
              </div>
            </div>
          </div>

          {/* PATIENT INFO - Demographics and Chief Complaint */}
          <div style={{ marginBottom: '7px' }}>
            <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>PATIENT DEMOGRAPHICS</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '7px', fontSize: '10px' }}>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>AGE:</span><span style={{ color: '#000000', fontWeight: 500, marginLeft: '4px' }}>{encounter.patient?.age} years</span>
              </div>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>GENDER:</span><span style={{ color: '#000000', fontWeight: 500, marginLeft: '4px' }}>{encounter.patient?.gender}</span>
              </div>
              <div style={{ gridColumn: 'span 2' }}>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>CHIEF COMPLAINT:</span><span style={{ color: '#000000', fontWeight: 500, marginLeft: '4px' }}>{encounter.patient?.symptoms || 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* VITAL SIGNS - 6 columns with color coding */}
          {encounter.vitals && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>VITAL SIGNS</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr 1fr', gap: '5px', fontSize: '10px' }}>
                {getVitalValue('temp') && (
                  <div style={{ 
                    padding: '5px', 
                    background: '#f0fdf4', 
                    borderLeft: `3px solid ${getVitalColor(getVitalValue('temp'), 36.5, 37.5)}`,
                    fontWeight: 500,
                    color: '#000000'
                  }}>
                    <span style={{ fontWeight: 600, color: '#1e293b' }}>Temp:</span> {getVitalValue('temp')}°C
                  </div>
                )}
                {getVitalValue('hr') && (
                  <div style={{ 
                    padding: '5px', 
                    background: '#f0fdf4', 
                    borderLeft: `3px solid ${getVitalColor(getVitalValue('hr'), 60, 100)}`,
                    fontWeight: 500,
                    color: '#000000'
                  }}>
                    <span style={{ fontWeight: 600, color: '#1e293b' }}>HR:</span> {getVitalValue('hr')} bpm
                  </div>
                )}
                {getVitalValue('o2') && (
                  <div style={{ 
                    padding: '5px', 
                    background: '#f0fdf4', 
                    borderLeft: `3px solid ${getVitalColor(getVitalValue('o2'), 95, 100)}`,
                    fontWeight: 500,
                    color: '#000000'
                  }}>
                    <span style={{ fontWeight: 600, color: '#1e293b' }}>O₂:</span> {getVitalValue('o2')}%
                  </div>
                )}
                {getVitalValue('sysBP') && (
                  <div style={{ 
                    padding: '5px', 
                    background: '#f0fdf4', 
                    borderLeft: `3px solid ${getVitalColor(getVitalValue('sysBP'), 90, 140)}`,
                    fontWeight: 500,
                    color: '#000000'
                  }}>
                    <span style={{ fontWeight: 600, color: '#1e293b' }}>BP(S):</span> {getVitalValue('sysBP')}
                  </div>
                )}
                {getVitalValue('diaBP') && (
                  <div style={{ 
                    padding: '5px', 
                    background: '#f0fdf4', 
                    borderLeft: `3px solid ${getVitalColor(getVitalValue('diaBP'), 60, 90)}`,
                    fontWeight: 500,
                    color: '#000000'
                  }}>
                    <span style={{ fontWeight: 600, color: '#1e293b' }}>BP(D):</span> {getVitalValue('diaBP')}
                  </div>
                )}
                {getVitalValue('rr') && (
                  <div style={{ 
                    padding: '5px', 
                    background: '#f0fdf4', 
                    borderLeft: `3px solid ${getVitalColor(getVitalValue('rr'), 12, 20)}`,
                    fontWeight: 500,
                    color: '#000000'
                  }}>
                    <span style={{ fontWeight: 600, color: '#1e293b' }}>RR:</span> {getVitalValue('rr')}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* PRIORITY & TRIAGE ASSESSMENT - With Color Badge */}
          {(reportType === 'AI' ? aiReport : manualReport) && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 800, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>TRIAGE ASSESSMENT</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '7px', fontSize: '10px' }}>
                <div>
                  <span style={{ color: '#1e293b', fontWeight: 600 }}>SATS CLASSIFICATION:</span>
                  <div style={{ marginTop: '4px' }}>
                    {satsPriority ? (
                      <div style={{
                        display: 'inline-block',
                        background: satsBadge.bgColor,
                        color: satsBadge.color,
                        padding: '8px 12px',
                        borderRadius: '4px',
                        fontWeight: 700,
                        fontSize: '12px',
                        border: `2px solid ${satsBadge.border}`,
                        letterSpacing: '0.5px'
                      }}>
                        {satsBadge.label}
                      </div>
                    ) : (
                      <div style={{ color: '#64748b', fontSize: '10px' }}>Not Available</div>
                    )}
                  </div>
                </div>
                <div>
                  <span style={{ color: '#1e293b', fontWeight: 600 }}>ESI LEVEL:</span>
                  <div style={{ marginTop: '4px', fontWeight: 600, color: '#000000', fontSize: '11px' }}>
                    {reportType === 'AI' ? (aiReport?.esi_level || triageData?.esi_level || 'N/A') : (manualReport?.esi_level || 'N/A')}
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '7px', paddingTop: '7px', borderTop: '1px solid #e2e8f0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '7px', fontSize: '10px' }}>
                <div>
                  <span style={{ color: '#1e293b', fontWeight: 600 }}>TEWS SCORE:</span>
                  <div style={{ marginTop: '4px', fontWeight: 700, color: '#0f172a' }}>
                    {tewsTotal !== null ? `${tewsTotal} points` : 'N/A'}
                  </div>
                </div>
                <div>
                  <span style={{ color: '#1e293b', fontWeight: 600 }}>ACUITY / URGENCY SCORE:</span>
                  <div style={{ marginTop: '4px', fontWeight: 700, color: '#0f172a' }}>
                    {acuityScore !== null ? `${acuityScore}` : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* CLINICAL SUMMARY */}
          {aiReport?.clinical_summary && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>CLINICAL SUMMARY</h4>
              <div style={{ fontSize: '10px', color: '#000000', padding: '6px', background: '#ffffff', borderRadius: 2, lineHeight: '1.4', fontWeight: 500 }}>
                {aiReport.clinical_summary}
              </div>
            </div>
          )}

          {/* PATHOPHYSIOLOGY */}
          {aiReport?.pathophysiology && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>PATHOPHYSIOLOGY</h4>
              <div style={{ fontSize: '10px', color: '#000000', padding: '6px', background: '#ffffff', borderRadius: 2, lineHeight: '1.4', fontWeight: 500 }}>
                {aiReport.pathophysiology}
              </div>
            </div>
          )}

          {/* CLINICAL DISCRIMINATORS */}
          {aiReport?.clinical_discriminators && aiReport.clinical_discriminators.length > 0 && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>CLINICAL DISCRIMINATORS</h4>
              <div style={{ fontSize: '10px', color: '#000000', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px', fontWeight: 500 }}>
                {aiReport.clinical_discriminators.map((disc, idx) => (
                  <div key={idx} style={{ padding: '3px' }}>▪ {disc}</div>
                ))}
              </div>
            </div>
          )}

          {/* DIFFERENTIAL DIAGNOSES */}
          {aiReport?.differential_diagnoses && aiReport.differential_diagnoses.length > 0 && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>DIFFERENTIAL DIAGNOSES</h4>
              <div style={{ fontSize: '10px', color: '#000000', fontWeight: 500 }}>
                {aiReport.differential_diagnoses.map((d, idx) => {
                  const diagnosis = typeof d === 'string' ? d : (d?.dx || JSON.stringify(d));
                  return (
                    <div key={idx} style={{ padding: '2px' }}>
                      ▪ {diagnosis}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* IMMEDIATE ACTIONS */}
          {aiReport?.immediate_actions && aiReport.immediate_actions.length > 0 && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>IMMEDIATE ACTIONS</h4>
              <div style={{ fontSize: '10px', color: '#000000', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px', fontWeight: 500 }}>
                {aiReport.immediate_actions.map((action, idx) => (
                  <div key={idx} style={{ padding: '3px' }}>▪ {action}</div>
                ))}
              </div>
            </div>
          )}

          {/* DIAGNOSTIC WORKUP */}
          {aiReport?.diagnostic_workup && aiReport.diagnostic_workup.length > 0 && (
            <div style={{ marginBottom: '7px' }}>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>DIAGNOSTIC WORKUP</h4>
              <div style={{ fontSize: '10px', color: '#000000', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px', fontWeight: 500 }}>
                {aiReport.diagnostic_workup.map((test, idx) => (
                  <div key={idx} style={{ padding: '3px' }}>▪ {test}</div>
                ))}
              </div>
            </div>
          )}

          {/* ASSIGNED CLINICIANS - Bottom of page */}
          <div style={{ marginTop: '9px', paddingTop: '7px', borderTop: '2px solid #1e293b' }}>
            <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000' }}>ASSIGNED CLINICIANS</h4>
            <div style={{ fontSize: '10px', color: '#000000', fontWeight: 500 }}>
              {[...(manuallySelected || []), ...(aiRecommended || [])].map((clinician, idx) => (
                <div key={idx} style={{ padding: '3px' }}>
                  ▪ {clinician.name} ({clinician.role}) - {clinician.specialty}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{
          display: 'flex',
          gap: 12,
          marginTop: 24,
          justifyContent: 'center'
        }}>
          <button
            onClick={() => navigate('/dashboard')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '10px 20px',
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
          <button
            onClick={() => navigate(`/handover/confirm/${encounterId}`)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '10px 20px',
              background: '#f0f9ff',
              border: '1px solid #0284c7',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600,
              color: '#0284c7'
            }}
          >
            <FiEdit size={16} /> Edit Report
          </button>
          <button
            onClick={handleDownloadPDF}
            disabled={downloading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '10px 20px',
              background: downloading ? '#cbd5e1' : '#1e40af',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: downloading ? 'not-allowed' : 'pointer',
              fontSize: 14,
              fontWeight: 600,
              opacity: downloading ? 0.7 : 1
            }}
          >
            <FiDownload size={16} /> {downloading ? 'Downloading...' : 'Download PDF'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default OfficialTriageReport;
