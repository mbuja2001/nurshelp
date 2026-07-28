import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiChevronLeft, FiDownload } from 'react-icons/fi';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import './PrescriptionReport.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const valueOrNA = (value) => value === null || value === undefined || value === '' ? 'N/A' : value;
const formatDate = (value, withTime = false) => {
  if (!value) return 'N/A';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'N/A' : date.toLocaleString(undefined, withTime ? undefined : { year: 'numeric', month: 'short', day: 'numeric' });
};

const splitNameParts = (fullName = '') => {
  const parts = fullName.trim().split(' ').filter(Boolean);
  return {
    first: parts.slice(0, -1).join(' ') || parts[0] || 'Doctor',
    last: parts.slice(-1).join(' ') || ''
  };
};

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
  return date.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  });
};

const determineReportTypeLabel = (prescription = {}) => {
  const status = String(prescription.status || 'ISSUED').toUpperCase();
  if (['APPROVED', 'ACCEPTED', 'DISPENSED', 'COMPLETED', 'CANCELLED'].includes(status)) {
    return 'Acknowledged ePrescription Report';
  }
  return 'ePrescription Request Report';
};

const buildPrescriptionTimeline = (prescription = {}) => {
  const entries = [];
  const pushEntry = (kind, title, body, timestamp, meta = {}) => {
    const parsedTimestamp = toTimestamp(timestamp);
    if (!parsedTimestamp) return;
    entries.push({
      id: `${kind}-${parsedTimestamp.getTime()}-${entries.length}`,
      kind,
      title,
      body,
      timestamp: parsedTimestamp,
      meta
    });
  };

  const issuedAt = prescription.date_issued || prescription.dateIssued;
  pushEntry(
    'ISSUED',
    'Prescription issued',
    `Prescription created and signed by ${prescription.prescriber?.name || 'Prescriber'}.`,
    issuedAt
  );

  if (prescription.submitted_at) {
    pushEntry(
      'SUBMITTED',
      'Submitted to pharmacy',
      'Prescription submitted to pharmacy.',
      prescription.submitted_at
    );
  }

  if (prescription.status === 'CANCELLED' && prescription.cancelled_at) {
    pushEntry(
      'CANCELLED',
      'Prescription cancelled',
      prescription.cancellation_reason || 'Prescription cancellation recorded.',
      prescription.cancelled_at
    );
  }

  (prescription.items || []).forEach((item, index) => {
    if (item.accepted_at) {
      pushEntry(
        'ACCEPTED',
        `Accepted: ${item.medication_name || 'Medication'}`,
        'Item accepted for dispensing.',
        item.accepted_at,
        { item, index }
      );
    }
    if (item.dispensed_at) {
      pushEntry(
        'DISPENSED',
        `Dispensed: ${item.medication_name || 'Medication'}`,
        'Item dispensed by pharmacy.',
        item.dispensed_at,
        { item, index }
      );
    }
    if (item.actual_received) {
      pushEntry(
        'RECEIVED',
        `Received: ${item.medication_name || 'Medication'}`,
        'Item received by patient.',
        item.actual_received,
        { item, index }
      );
    }
  });

  return entries
    .sort((a, b) => b.timestamp - a.timestamp)
    .map((entry) => ({
      ...entry,
      displayTime: formatTimeLabel(entry.timestamp)
    }));
};

const groupTimelineByDay = (timeline = []) => {
  const grouped = new Map();
  timeline.forEach((entry) => {
    const key = entry.timestamp.toISOString().split('T')[0];
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(entry);
  });

  return Array.from(grouped.entries())
    .sort(([a], [b]) => (a < b ? 1 : -1))
    .map(([key, entries]) => ({
      key,
      label: formatDayLabel(new Date(key)),
      entries: entries
        .slice()
        .sort((a, b) => b.timestamp - a.timestamp)
        .map((entry) => ({
          ...entry,
          displayTime: formatTimeLabel(entry.timestamp)
        }))
    }));
};

export default function PrescriptionReport() {
  const navigate = useNavigate();
  const { prescriptionId } = useParams();
  const [prescription, setPrescription] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [timelineDays, setTimelineDays] = useState([]);
  const [selectedDayKey, setSelectedDayKey] = useState(null);
  const [selectedEntryId, setSelectedEntryId] = useState(null);
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);

  useEffect(() => {
    const loadPrescription = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${BACKEND_URL}/api/prescriptions/${prescriptionId}`, {
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          cache: 'no-store'
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || data.message || 'Failed to fetch prescription');
        setPrescription(data.prescription);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    if (prescriptionId) loadPrescription();
  }, [prescriptionId]);

  useEffect(() => {
    if (!prescription) return;
    const timeline = buildPrescriptionTimeline(prescription);
    const groupedDays = groupTimelineByDay(timeline);
    setTimelineDays(groupedDays);
    setSelectedDayKey(groupedDays[0]?.key || null);
    setSelectedEntryId(groupedDays[0]?.entries?.[0]?.id || null);
    setCalendarMonth(new Date(groupedDays[0]?.key || Date.now()));
  }, [prescription]);

  useEffect(() => {
    if (!timelineDays.length) return;
    const matchingDay = timelineDays.find((day) => day.key === selectedDayKey) || timelineDays[0];
    if (!matchingDay) return;
    const hasSelectedEntry = matchingDay.entries.some((entry) => entry.id === selectedEntryId);
    if (!hasSelectedEntry) {
      setSelectedEntryId(matchingDay.entries[0]?.id || null);
    }
  }, [timelineDays, selectedDayKey, selectedEntryId]);

  const handleDownloadPDF = async () => {
    const element = document.getElementById('official-prescription-content');
    if (!element) return;
    try {
      setDownloading(true);
      const canvas = await html2canvas(element, { scale: 2, backgroundColor: '#fff', useCORS: true, logging: false });
      const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
      const margin = 10;
      const width = pdf.internal.pageSize.getWidth() - margin * 2;
      const height = Math.min((canvas.height * width) / canvas.width, pdf.internal.pageSize.getHeight() - margin * 2);
      pdf.addImage(canvas.toDataURL('image/png'), 'PNG', margin, margin, width, height);
      pdf.save(`ePrescription_${prescription?.prescription_number || prescriptionId}_${new Date().toISOString().slice(0, 10)}.pdf`);
    } catch (err) {
      alert(`Failed to download prescription: ${err.message}`);
    } finally {
      setDownloading(false);
    }
  };

  if (loading) return <div className="prescription-state">Loading prescription report...</div>;
  if (error) return <div className="prescription-state prescription-state-error">Failed to load prescription: {error}</div>;
  if (!prescription) return <div className="prescription-state prescription-state-error">Prescription not found</div>;

  const prescriber = prescription.prescriber || {};
  const prescriberFullName = valueOrNA(prescriber.name);
  const patient = prescription.patient_details || {};
  const items = prescription.items || [];
  const status = String(prescription.status || 'ISSUED').toUpperCase();
  const reportTypeLabel = determineReportTypeLabel(prescription);
  const reportSubtitle = reportTypeLabel === 'Acknowledged ePrescription Report'
    ? 'Pharmacy acknowledgement and review summary for the medication order.'
    : 'Physician request report for the medication order submitted to pharmacy.';
  const issuedDateTime = formatDate(prescription.date_issued || prescription.dateIssued, true);
  const activeDay = timelineDays.find((day) => day.key === selectedDayKey) || timelineDays[0] || null;
  const activeTimelineEntries = activeDay?.entries || [];
  const activeEntry = activeTimelineEntries.find((entry) => entry.id === selectedEntryId) || activeTimelineEntries[0] || null;
  const selectedDayDate = selectedDayKey ? new Date(`${selectedDayKey}T00:00:00`) : null;
  const monthStart = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), 1);
  const firstDayOfWeek = (monthStart.getDay() + 6) % 7;
  const daysInMonth = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 0).getDate();
  const calendarCells = Array.from({ length: firstDayOfWeek + daysInMonth }, (_, index) => {
    const dayNumber = index - firstDayOfWeek + 1;
    if (dayNumber < 1 || dayNumber > daysInMonth) return null;
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

  return (
    <div className="prescription-page">
      <div className="prescription-toolbar">
        <div>
          <h1>{reportTypeLabel}</h1>
          <p>{reportSubtitle}</p>
        </div>
        <div className="prescription-toolbar-actions">
          <button type="button" className="report-action report-action-secondary" onClick={() => setShowDatePicker(true)}>Select Date & Time</button>
          <button type="button" className="report-action report-action-muted" onClick={() => navigate('/dashboard')}><FiChevronLeft size={16} /> Back to Dashboard</button>
          <button type="button" className="report-action report-action-primary" onClick={handleDownloadPDF} disabled={downloading}><FiDownload size={16} /> {downloading ? 'Preparing PDF...' : 'Download PDF'}</button>
        </div>
      </div>

      {showDatePicker ? (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.55)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
          <div style={{ width: 'min(100%, 540px)', background: 'white', borderRadius: 12, boxShadow: '0 16px 40px rgba(15, 23, 42, 0.2)', padding: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>Select report date and time</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Choose the prescription day and the event entry to display and print</div>
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
              <div style={{ color: '#64748b', fontSize: 12 }}>No recorded prescription timeline entries are available for this prescription yet.</div>
            )}
          </div>
        </div>
      ) : null}

      <article id="official-prescription-content" className="prescription-report">
        <header className="report-header">
          <h2>AUTHORIZED ELECTRONIC PRESCRIPTION</h2>
          <div className="report-subtitle">PHYSICIAN MEDICATION ORDER / HOSPITAL RECORD</div>
          <div className="report-meta-grid">
            <div><span>RX NUMBER:</span><strong>{valueOrNA(prescription.prescription_number)}</strong></div>
            <div><span>DATE ISSUED:</span><strong>{formatDate(prescription.date_issued || prescription.dateIssued, true)}</strong></div>
            <div><span>VALID UNTIL:</span><strong>{formatDate(prescription.valid_until || prescription.expiryDate)}</strong></div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '9px', marginTop: 8 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span style={{ color: '#1e293b', fontSize: 9, fontWeight: 700 }}>SELECTED REPORT DAY</span>
              <strong style={{ fontSize: 10, fontWeight: 600 }}>{selectedDayDate ? selectedDayDate.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'short', year: 'numeric' }) : 'No day selected'}</strong>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span style={{ color: '#1e293b', fontSize: 9, fontWeight: 700 }}>SELECTED ENTRY</span>
              <strong style={{ fontSize: 10, fontWeight: 600 }}>{activeEntry ? `${activeEntry.displayTime} • ${activeEntry.title}` : 'No entry selected'}</strong>
            </div>
          </div>
          <div className="report-identity-grid">
            <div><span>PATIENT:</span><strong>{valueOrNA(patient.name)}</strong><small>ID: {valueOrNA(patient.id_number || prescription.patientId)}</small></div>
            <div><span>PRESCRIBER:</span><strong>{valueOrNA(prescriber.name)}</strong><small>{valueOrNA(prescriber.specialty)}</small></div>
            <div><span>ORDER STATUS:</span><strong className={`report-status report-status-${status.toLowerCase()}`}>{status}</strong><small>Encounter: {valueOrNA(prescription.encounterId)}</small></div>
          </div>
        </header>

        <section className="report-section"><h3>SELECTED REPORT ENTRY</h3>
          <div className="report-note"><span>EVENT TITLE:</span><strong>{activeEntry ? activeEntry.title : 'N/A'}</strong></div>
          <div className="report-note"><span>DETAILS:</span><strong>{valueOrNA(activeEntry?.body)}</strong></div>
          <div className="report-note"><span>ENTRY TIME:</span><strong>{activeEntry ? `${selectedDayDate ? selectedDayDate.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : ''} ${activeEntry.displayTime}` : 'N/A'}</strong></div>
        </section>
        <section className="report-section"><h3>PATIENT DEMOGRAPHICS</h3><div className="report-grid-2">
          <div><span>NAME:</span><strong>{valueOrNA(patient.name)}</strong></div><div><span>ID NUMBER:</span><strong>{valueOrNA(patient.id_number || prescription.patientId)}</strong></div>
          <div><span>AGE:</span><strong>{patient.age ? `${patient.age} years` : 'N/A'}</strong></div><div><span>DATE OF BIRTH:</span><strong>{formatDate(patient.date_of_birth)}</strong></div>
          <div><span>MEDICAL AID:</span><strong>{valueOrNA(patient.medical_aid)}</strong></div><div><span>MEDICAL AID NUMBER:</span><strong>{valueOrNA(patient.medical_aid_number)}</strong></div>
          <div className="report-span-2"><span>ADDRESS:</span><strong>{valueOrNA(patient.address)}</strong></div>
        </div></section>

        <section className="report-section"><h3>PRESCRIBER INFORMATION</h3><div className="report-grid-2">
          <div><span>NAME:</span><strong>{valueOrNA(prescriber.name)}</strong></div><div><span>ROLE:</span><strong>{valueOrNA(prescriber.role)}</strong></div>
          <div><span>SPECIALTY:</span><strong>{valueOrNA(prescriber.specialty)}</strong></div><div><span>HPCSA NUMBER:</span><strong>{valueOrNA(prescriber.hpcsa_number)}</strong></div>
          <div><span>EMAIL:</span><strong>{valueOrNA(prescriber.email)}</strong></div><div><span>PHONE:</span><strong>{valueOrNA(prescriber.phone)}</strong></div>
          <div className="report-span-2"><span>PRACTICE / FACILITY:</span><strong>{valueOrNA(prescriber.practice_address)}</strong></div>
        </div></section>

        <section className="report-section"><h3>MEDICATION ORDER</h3>
          {items.map((item, index) => <div className="medication-order" key={item._id || index}>
            <div className="medication-order-heading"><strong>{index + 1}. {valueOrNA(item.medication_name)}</strong><span>{valueOrNA(item.schedule)}</span></div>
            <div className="report-grid-2">
              <div><span>DOSAGE:</span><strong>{valueOrNA(item.dosage || item.strength)}</strong></div><div><span>FORM / UNIT:</span><strong>{valueOrNA(item.form || item.unit)}</strong></div>
              <div><span>FREQUENCY:</span><strong>{valueOrNA(item.frequency)}</strong></div><div><span>ROUTE:</span><strong>{valueOrNA(item.route)}</strong></div>
              <div><span>DURATION:</span><strong>{valueOrNA(item.duration)}</strong></div><div><span>QUANTITY:</span><strong>{valueOrNA(item.quantity)}</strong></div>
              <div><span>REFILLS:</span><strong>{valueOrNA(item.refills ?? 0)}</strong></div><div><span>ITEM STATUS:</span><strong>{valueOrNA(item.status)}</strong></div>
              <div className="report-span-2"><span>DIRECTIONS:</span><strong>{valueOrNA(item.directions || `Take ${item.dosage || 'as directed'} ${item.frequency || ''} by ${item.route || 'the prescribed route'}.`)}</strong></div>
              {item.indication && <div className="report-span-2"><span>INDICATION:</span><strong>{item.indication}</strong></div>}
            </div>
          </div>)}
          {!items.length && <p className="report-empty">No medication items recorded.</p>}
        </section>

        <section className="report-section"><h3>CLINICAL INSTRUCTIONS</h3>
          <div className="report-note"><span>SPECIAL INSTRUCTIONS:</span><strong>{valueOrNA(prescription.special_instructions)}</strong></div>
          <div className="report-note"><span>CONTRAINDICATIONS:</span><strong>{valueOrNA(prescription.contraindications)}</strong></div>
          <div className="report-note"><span>GENERIC SUBSTITUTION:</span><strong>{prescription.substitution_permitted ? 'Permitted' : 'Not permitted'}</strong></div>
        </section>

        <section className="report-section dispensing-section"><h3>PHARMACY DISPENSING STATUS</h3><div className="report-grid-2">
          <div><span>SUBMITTED TO PHARMACY:</span><strong>{prescription.submitted_to_pharmacy ? 'YES' : 'NO'}</strong></div><div><span>PRESCRIPTION STATUS:</span><strong>{status}</strong></div>
          <div className="report-span-2"><span>ITEMS:</span><strong>{items.map(item => `${item.medication_name || 'Medication'} - ${item.status || 'PENDING'}`).join(' | ') || 'N/A'}</strong></div>
        </div></section>

        <footer className="report-footer">
          <div className="signature-grid">
            <div className="signature-block">
              <span>DOCTOR SIGNATURE</span>
              <div className="signature-line">{prescriberFullName}</div>
              <small>Physician full name on signature line</small>
            </div>
            <div className="signature-block">
              <span>DATE / TIME</span>
              <div className="signature-line">{issuedDateTime}</div>
              <small>Date and time of issue</small>
            </div>
          </div>
          <div className="report-reference">Document prepared for the official hospital clinical record • Reference: {String(prescription._id || prescriptionId).slice(-8).toUpperCase()}</div>
        </footer>
      </article>
    </div>
  );
}
