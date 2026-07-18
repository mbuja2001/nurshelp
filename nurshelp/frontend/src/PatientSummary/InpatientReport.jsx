import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiChevronLeft, FiDownload } from 'react-icons/fi';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { buildInpatientTimeline, groupTimelineByDay } from './inpatientReportTimeline';

const InpatientReport = () => {
  const navigate = useNavigate();
  const { encounterId } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [bed, setBed] = useState(null);
  const [selectedDayKey, setSelectedDayKey] = useState(null);
  const [selectedEntryId, setSelectedEntryId] = useState(null);
  const [timelineDays, setTimelineDays] = useState([]);
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);

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
        const enc = data.encounter;
        
        console.log('[INPATIENT REPORT] Encounter loaded:', enc._id);
        setEncounter(enc);

        const timeline = buildInpatientTimeline(enc);
        const groupedDays = groupTimelineByDay(timeline);
        setTimelineDays(groupedDays);
        setSelectedDayKey(groupedDays[0]?.key || null);
        setSelectedEntryId(groupedDays[0]?.entries?.[0]?.id || null);
        setCalendarMonth(new Date(groupedDays[0]?.key || Date.now()));

        // Bed lookup is optional and can fail if the bed endpoint is unavailable.
        if (enc.assigned_bed) {
          try {
            const bedRes = await fetch(`${BACKEND_URL}/api/beds/${enc.assigned_bed}`, {
              headers
            });
            if (bedRes.ok) {
              const bedData = await bedRes.json();
              setBed(bedData.bed || bedData.data || null);
            }
          } catch (err) {
            console.log('[INPATIENT REPORT] Bed details unavailable:', err.message);
          }
        }
      } catch (err) {
        console.error('Error fetching encounter:', err);
        alert('Failed to load inpatient report');
      } finally {
        setLoading(false);
      }
    };

    if (encounterId) fetchEncounter();
  }, [encounterId, token, BACKEND_URL]);

  const getSeverityConfig = (severity) => {
    const severityMap = {
      1: { label: 'CRITICAL', color: '#dc2626', bg: '#fee2e2' },
      2: { label: 'HIGH', color: '#ea580c', bg: '#ffedd5' },
      3: { label: 'MODERATE', color: '#b45309', bg: '#fef3c7' },
      4: { label: 'STABLE', color: '#059669', bg: '#d1fae5' },
      5: { label: 'MINIMAL', color: '#0891b2', bg: '#cffafe' }
    };
    return severityMap[severity] || { label: 'UNKNOWN', color: '#475569', bg: '#f1f5f9' };
  };

  const handleDownloadPDF = async () => {
    try {
      setDownloading(true);
      const reportElement = document.getElementById('inpatient-report-content');
      
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

      const patientName = encounter?.patient?.name || encounter?.patient?.firstName || encounter?.patientName || 'Patient';
      const filename = `InpatientReport_${patientName}_${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(filename);

      console.log('[INPATIENT REPORT] PDF downloaded:', filename);
      alert('PDF downloaded successfully!');
    } catch (err) {
      console.error('Error downloading PDF:', err);
      alert('Failed to download PDF: ' + err.message);
    } finally {
      setDownloading(false);
    }
  };

  useEffect(() => {
    if (!timelineDays.length) return;
    const matchingDay = timelineDays.find((day) => day.key === selectedDayKey) || timelineDays[0];
    if (!matchingDay) return;
    const hasSelectedEntry = matchingDay.entries.some((entry) => entry.id === selectedEntryId);
    if (!hasSelectedEntry) {
      setSelectedEntryId(matchingDay.entries[0]?.id || null);
    }
  }, [timelineDays, selectedDayKey, selectedEntryId]);

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
        Loading inpatient report...
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

  const severityConfig = getSeverityConfig(encounter?.severity || encounter?.triage?.aiTriageReport?.esi_level || 4);
  const patientName = encounter.patient?.name || encounter.patient?.firstName || encounter.patientName || 'Patient';
  const patientAge = encounter.patient?.age || encounter.age || '—';
  const diagnosis = encounter.diagnosis || encounter.triage?.aiTriageReport?.primary_diagnosis || 'No diagnosis documented yet';
  const treatmentPlan = encounter.treatment_plan || encounter.triage?.aiTriageReport?.recommended_treatment || '';
  const physicianNotes = encounter.physician_notes || encounter.notes || '';
  const dispositionStatus = encounter.status || encounter.disposition || 'HOSPITALIZED';
  const admissionDate = encounter.createdAt ? new Date(encounter.createdAt) : null;
  const daysHospitalized = admissionDate
    ? Math.max(1, Math.floor((Date.now() - admissionDate.getTime()) / 86400000) + 1)
    : 1;
  const treatments = encounter.treatments || [];
  const recommendations = encounter.recommendations || [];
  const progressionNotes = encounter.progression_notes || [];
  const wardName = encounter.ward_display_info?.wardName || encounter.ward || encounter.assigned_ward?.name || 'Not assigned';
  const bedLabel = encounter.bed_display_info?.bedId || encounter.bed_display_info?.bedNumber || encounter.bed_display_info?.bed_number || bed?.bedId || 'Not assigned';
  const clinicianName = encounter.assigned_physician?.name || encounter.assignedPhysicianName || encounter.assigned_clinicians?.find((item) => item.role === 'PHYSICIAN')?.name || 'Not assigned';
  const vitals = encounter.vitals || {};
  const summaryText = encounter.triage?.aiTriageReport?.ai_summary || encounter.triage?.summary || encounter.summary || physicianNotes || 'No daily summary has been documented yet.';
  const planText = physicianNotes || treatmentPlan || 'No discharge or follow-up plan has been documented yet.';
  const activeDay = timelineDays.find((day) => day.key === selectedDayKey) || timelineDays[0] || null;
  const activeTimelineEntries = activeDay?.entries || [];
  const activeEntry = activeTimelineEntries.find((entry) => entry.id === selectedEntryId) || activeTimelineEntries[0] || null;
  const selectedDayDate = selectedDayKey ? new Date(`${selectedDayKey}T00:00:00`) : null;
  const selectedDaySummary = (() => {
    const progressionFromDay = activeTimelineEntries
      .filter((entry) => entry.kind === 'PROGRESSION')
      .map((entry) => entry.body)
      .filter(Boolean);
    const statusFromDay = activeTimelineEntries
      .filter((entry) => entry.kind === 'STATUS')
      .map((entry) => entry.body)
      .filter(Boolean);

    if (activeEntry?.kind === 'PROGRESSION' && activeEntry.body) {
      return activeEntry.body;
    }
    if (progressionFromDay.length) {
      return progressionFromDay[0];
    }
    if (statusFromDay.length) {
      return statusFromDay[0];
    }
    return summaryText;
  })();
  const selectedDayTreatments = activeTimelineEntries
    .filter((entry) => entry.kind === 'TREATMENT')
    .map((entry) => entry.meta?.treatment)
    .filter(Boolean);
  const selectedDayRecommendations = activeTimelineEntries
    .filter((entry) => entry.kind === 'RECOMMENDATION')
    .map((entry) => entry.meta?.recommendation)
    .filter(Boolean);
  const selectedDayProgression = activeTimelineEntries
    .filter((entry) => entry.kind === 'PROGRESSION')
    .map((entry) => ({
      note_type: entry.title,
      content: entry.body,
      recorded_at: entry.timestamp?.toISOString(),
      patient_response: ''
    }));
  const selectedDayStatus = activeTimelineEntries.filter((entry) => entry.kind === 'STATUS');
  const monthStart = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), 1);
  const firstDayOfWeek = (monthStart.getDay() + 6) % 7;
  const daysInMonth = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 0).getDate();
  const calendarCells = Array.from({ length: firstDayOfWeek + daysInMonth }, (_, index) => {
    const dayNumber = index - firstDayOfWeek + 1;
    if (dayNumber < 1 || dayNumber > daysInMonth) {
      return null;
    }

    const date = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), dayNumber);
    const key = date.toISOString().split('T')[0];
    const dayData = timelineDays.find((item) => item.key === key);
    return {
      key,
      date,
      hasEvents: Boolean(dayData),
      isSelected: selectedDayKey === key,
      eventCount: dayData?.entries?.length || 0
    };
  }).filter(Boolean);

  const renderList = (items, emptyMessage) => {
    if (!items || items.length === 0) {
      return <div style={{ color: '#64748b', fontSize: '10px', marginTop: '4px' }}>{emptyMessage}</div>;
    }

    return (
      <div style={{ marginTop: '6px', display: 'grid', gap: '6px' }}>
        {items.map((item, index) => {
          if (typeof item === 'string') {
            return <div key={index} style={{ color: '#0f172a', fontSize: '10px' }}>• {item}</div>;
          }

          return (
            <div key={index} style={{ padding: '6px 0', borderBottom: index === items.length - 1 ? 'none' : '1px solid #e2e8f0' }}>
              <div style={{ color: '#0f172a', fontSize: '10px', fontWeight: 600 }}>
                {item.name || item.description || item.note_type || item.recommendation_type || 'Entry'}
              </div>
              {item.dosage || item.frequency || item.route ? (
                <div style={{ color: '#475569', fontSize: '9px', marginTop: '2px' }}>
                  {item.dosage ? `Dose: ${item.dosage}` : ''}{item.frequency ? ` • ${item.frequency}` : ''}{item.route ? ` • ${item.route}` : ''}
                </div>
              ) : null}
              {item.indication || item.priority || item.status ? (
                <div style={{ color: '#64748b', fontSize: '9px', marginTop: '2px' }}>
                  {item.indication ? `Indication: ${item.indication}` : ''}
                  {item.priority ? `${item.indication ? ' • ' : ''}Priority: ${item.priority}` : ''}
                  {item.status ? `${item.indication || item.priority ? ' • ' : ''}Status: ${item.status}` : ''}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{ background: '#f8fafc', minHeight: '100vh', padding: '20px' }}>
      <div style={{ maxWidth: '210mm', margin: '0 auto' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 16,
          marginBottom: 24,
          background: 'white',
          padding: '16px 20px',
          borderRadius: 8,
          border: '1px solid #e2e8f0',
          flexWrap: 'wrap'
        }}>
          <div style={{ flex: '1 1 280px', minWidth: 0 }}>
            <h1 style={{ margin: '0 0 4px 0', fontSize: 24, fontWeight: 700, color: '#0f172a', lineHeight: 1.2 }}>
              Inpatient Clinical Report
            </h1>
            <p style={{ margin: 0, fontSize: 14, color: '#64748b', lineHeight: 1.4 }}>
              Official inpatient record for the selected date and recorded clinical entry.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end', flex: '0 0 auto' }}>
            <button
              onClick={() => setShowDatePicker(true)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '9px 14px',
                background: '#0f766e',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 13,
                fontWeight: 600,
                minHeight: 40,
                whiteSpace: 'nowrap'
              }}
            >
              Select Date & Time
            </button>
            <button
              onClick={handleDownloadPDF}
              disabled={downloading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '9px 14px',
                background: '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                cursor: downloading ? 'wait' : 'pointer',
                fontSize: 13,
                fontWeight: 600,
                minHeight: 40,
                whiteSpace: 'nowrap'
              }}
            >
              <FiDownload size={16} /> {downloading ? 'Downloading...' : 'Download PDF'}
            </button>
            <button
              onClick={() => navigate(-1)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '9px 14px',
                background: '#e2e8f0',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 13,
                fontWeight: 600,
                minHeight: 40,
                whiteSpace: 'nowrap'
              }}
            >
              <FiChevronLeft size={16} /> Back
            </button>
          </div>
        </div>

        <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'white', padding: '10px 14px', borderRadius: 8, border: '1px solid #e2e8f0' }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: '#0f172a' }}>Selected report day</div>
            <div style={{ fontSize: 12, color: '#475569' }}>{selectedDayDate ? selectedDayDate.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }) : 'No day selected'}</div>
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>{timelineDays.length > 0 ? `${timelineDays.length} day${timelineDays.length === 1 ? '' : 's'} available` : 'No timeline data'}</div>
        </div>

        {showDatePicker ? (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.55)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
            <div style={{ width: 'min(100%, 480px)', background: 'white', borderRadius: 12, boxShadow: '0 16px 40px rgba(15, 23, 42, 0.2)', padding: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>Select report date and time</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>Choose the inpatient day and the specific recorded entry to display and print</div>
                </div>
                <button onClick={() => setShowDatePicker(false)} style={{ border: 'none', background: 'transparent', fontSize: 18, cursor: 'pointer', color: '#475569' }}>×</button>
              </div>

              {timelineDays.length > 0 ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <button onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() - 1, 1))} style={{ border: '1px solid #e2e8f0', background: 'white', borderRadius: 6, padding: '6px 8px', cursor: 'pointer' }}>←</button>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#0f172a' }}>{calendarMonth.toLocaleDateString(undefined, { month: 'long', year: 'numeric' })}</div>
                    <button onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))} style={{ border: '1px solid #e2e8f0', background: 'white', borderRadius: 6, padding: '6px 8px', cursor: 'pointer' }}>→</button>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 6, fontSize: 10, color: '#64748b', marginBottom: 6 }}>
                    {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((dayName) => <div key={dayName} style={{ textAlign: 'center' }}>{dayName}</div>)}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 6 }}>
                    {calendarCells.map((cell) => (
                      <button
                        key={cell.key}
                        disabled={!cell.hasEvents}
                        onClick={() => {
                          setSelectedDayKey(cell.key);
                          const dayData = timelineDays.find((item) => item.key === cell.key);
                          setSelectedEntryId(dayData?.entries?.[0]?.id || null);
                        }}
                        style={{
                          minHeight: 36,
                          borderRadius: 6,
                          border: cell.isSelected ? '1px solid #2563eb' : '1px solid #e2e8f0',
                          background: cell.isSelected ? '#eff6ff' : (cell.hasEvents ? 'white' : '#f8fafc'),
                          color: cell.hasEvents ? '#0f172a' : '#cbd5e1',
                          cursor: cell.hasEvents ? 'pointer' : 'default',
                          fontSize: 10,
                          fontWeight: 700,
                          padding: 4
                        }}
                      >
                        {cell.date.getDate()}
                        {cell.hasEvents ? <div style={{ fontSize: 8, color: '#2563eb', fontWeight: 600 }}>{cell.eventCount}</div> : null}
                      </button>
                    ))}
                  </div>

                  {activeDay?.entries?.length ? (
                    <div style={{ marginTop: 12 }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#0f172a', marginBottom: 6 }}>Recorded entries for this day</div>
                      <div style={{ display: 'grid', gap: 6, maxHeight: 220, overflowY: 'auto' }}>
                        {activeDay.entries.map((entry) => (
                          <button
                            key={entry.id}
                            onClick={() => {
                              setSelectedEntryId(entry.id);
                              setShowDatePicker(false);
                            }}
                            style={{
                              textAlign: 'left',
                              border: entry.id === selectedEntryId ? '1px solid #2563eb' : '1px solid #e2e8f0',
                              borderRadius: 8,
                              padding: '8px 10px',
                              background: entry.id === selectedEntryId ? '#eff6ff' : 'white',
                              cursor: 'pointer'
                            }}
                          >
                            <div style={{ fontSize: 10, fontWeight: 700, color: '#2563eb' }}>{entry.displayTime}</div>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#0f172a', marginTop: 2 }}>{entry.title}</div>
                            <div style={{ fontSize: 10, color: '#475569', marginTop: 2, whiteSpace: 'pre-wrap' }}>{entry.body}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </>
              ) : (
                <div style={{ color: '#64748b', fontSize: 12 }}>No recorded inpatient entries are available for this encounter yet.</div>
              )}
            </div>
          </div>
        ) : null}

        <div id="inpatient-report-content" style={{
          background: 'white',
          padding: '11px',
          borderRadius: 0,
          fontSize: '11px',
          lineHeight: '1.35',
          maxWidth: '210mm',
          margin: '0 auto'
        }}>
          <div style={{ marginBottom: '9px', paddingBottom: '7px', borderBottom: '3px solid #1e293b' }}>
            <h2 style={{ margin: '0 0 3px 0', fontSize: '15px', fontWeight: 600, color: '#000000', letterSpacing: '0.5px' }}>
              INPATIENT CLINICAL REPORT
            </h2>
            <div style={{ fontSize: '10px', color: '#1e293b', marginBottom: '5px', fontWeight: 500 }}>
              RECORD DATE • {activeEntry ? `${selectedDayDate ? selectedDayDate.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : 'Not recorded'} • ${activeEntry.displayTime}` : 'No report entry selected'}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr 0.8fr', gap: '9px', fontSize: '10px', marginBottom: '5px' }}>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>PATIENT:</span>
                <div style={{ color: '#000000', fontWeight: 500, fontSize: '10px', marginTop: '2px' }}>{patientName}</div>
                <div style={{ color: '#1e293b', fontWeight: 600, fontSize: '9px', marginTop: '3px' }}>ID: {encounter.patient?.id_number || encounter._id?.substring(0, 8) || 'N/A'}</div>
              </div>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>SEVERITY:</span>
                <div style={{ color: '#000000', fontWeight: 500, fontSize: '10px', marginTop: '2px' }}>{severityConfig.label}</div>
              </div>
              <div>
                <span style={{ color: '#1e293b', fontWeight: 600 }}>STATUS:</span>
                <div style={{ color: '#000000', fontWeight: 500, fontSize: '10px', marginTop: '2px' }}>{dispositionStatus === 'HOSPITALIZED' ? 'ADMITTED' : dispositionStatus}</div>
              </div>
            </div>
          </div>

          <div style={{ marginBottom: '7px' }}>
            <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>
              CLINICAL SUMMARY
            </h4>
            {activeEntry ? (
              <div style={{ marginTop: '5px', marginBottom: '6px', padding: '6px 8px', borderRadius: '6px', background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '9px', fontWeight: 700, color: '#2563eb', marginBottom: '2px' }}>{activeEntry.displayTime} • {activeEntry.title}</div>
                <div style={{ fontSize: '10px', color: '#0f172a', whiteSpace: 'pre-wrap' }}>{activeEntry.body || 'No clinical note recorded for this report entry.'}</div>
              </div>
            ) : (
              <p style={{ margin: '5px 0 0 0', color: '#0f172a', fontSize: '10px', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
                No inpatient report was logged for the selected date.
              </p>
            )}
          </div>

          <div style={{ marginBottom: '7px', display: 'grid', gridTemplateColumns: '1.15fr 0.85fr', gap: '8px' }}>
            <div>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>
                ASSESSMENT & DIAGNOSIS
              </h4>
              <p style={{ margin: '5px 0 0 0', color: '#0f172a', fontSize: '10px', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
                {diagnosis}
              </p>
            </div>
            <div>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>
                ADMISSION DETAILS
              </h4>
              <div style={{ marginTop: '5px', display: 'grid', gap: '4px', fontSize: '10px', color: '#0f172a' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Admitted</span><strong>{admissionDate ? admissionDate.toLocaleDateString() : 'Not recorded'}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Days in care</span><strong>{daysHospitalized}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Ward</span><strong>{wardName}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Bed</span><strong>{bedLabel}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Attending physician</span><strong>{clinicianName}</strong></div>
              </div>
            </div>
          </div>

          <div style={{ marginBottom: '7px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <div>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>
                VITALS
              </h4>
              <div style={{ marginTop: '5px', display: 'grid', gap: '4px', fontSize: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Blood Pressure</span><strong>{vitals.bp || 'Not recorded'}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Heart Rate</span><strong>{vitals.hr ? `${vitals.hr} bpm` : 'Not recorded'}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Temperature</span><strong>{vitals.temp ? `${vitals.temp}°C` : 'Not recorded'}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>O₂ Saturation</span><strong>{vitals.o2 ? `${vitals.o2}%` : 'Not recorded'}</strong></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Respiratory Rate</span><strong>{vitals.resp ? `${vitals.resp} /min` : 'Not recorded'}</strong></div>
              </div>
            </div>
            <div>
              <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>
                CARE PLAN & INTERVENTIONS
              </h4>
              {activeEntry?.kind === 'TREATMENT' && activeEntry.meta?.treatment ? (
                renderList([activeEntry.meta.treatment], 'No intervention recorded for this report entry.')
              ) : (
                renderList(selectedDayTreatments, 'No intervention recorded for this report entry.')
              )}
              {activeEntry?.kind !== 'TREATMENT' && treatmentPlan ? <div style={{ color: '#475569', fontSize: '10px', marginTop: '6px' }}>{treatmentPlan}</div> : null}
            </div>
          </div>

          <div style={{ marginBottom: '7px' }}>
            <h4 style={{ margin: '0 0 4px 0', fontSize: '11px', fontWeight: 600, color: '#000000', borderBottom: '2px solid #1e293b', paddingBottom: '3px' }}>
              FOLLOW-UP & DISPOSITION
            </h4>
            {activeEntry?.kind === 'RECOMMENDATION' && activeEntry.meta?.recommendation ? (
              renderList([activeEntry.meta.recommendation], 'No follow-up instruction recorded for this report entry.')
            ) : (
              renderList(selectedDayRecommendations, 'No follow-up instruction recorded for this report entry.')
            )}
            {selectedDayStatus.length > 0 ? (
              <div style={{ marginTop: '6px', color: '#475569', fontSize: '10px', whiteSpace: 'pre-wrap' }}>
                {selectedDayStatus[0].body}
              </div>
            ) : null}
          </div>

          <div style={{ marginTop: '10px', paddingTop: '7px', borderTop: '2px solid #1e293b', fontSize: '9px', color: '#1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
              <span>Document prepared for official inpatient record</span>
              <span>Record reference: {encounter._id?.substring(0, 12)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InpatientReport;
