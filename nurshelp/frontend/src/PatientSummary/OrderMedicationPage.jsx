import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const normalizeHistory = (value) => {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === 'object') return Object.values(value);
  return [];
};

const formatDate = (value) => {
  if (!value) return 'Unknown';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
};

const OrderMedicationPage = () => {
  const navigate = useNavigate();
  const { encounterId } = useParams();

  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState('');
  const [form, setForm] = useState({
    medication_name: '',
    dosage: '',
    frequency: '',
    route: 'Oral',
    schedule: 'Schedule 3',
    indication: '',
    prescriptionNotes: ''
  });

  useEffect(() => {
    const token = localStorage.getItem('token');
    const fetchEncounter = async () => {
      try {
        setLoading(true);
        const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}`, {
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {})
          }
        });
        if (!res.ok) throw new Error('Failed to load patient context');
        const data = await res.json();
        setEncounter(data.encounter || data);
      } catch (err) {
        setError(err.message || 'Unable to load patient context');
      } finally {
        setLoading(false);
      }
    };

    if (encounterId) fetchEncounter();
  }, [encounterId]);

  const hospitalContext = useMemo(() => {
    const patient = encounter?.patient || {};
    const history = {
      pastMeds: normalizeHistory(encounter?.medication_history || encounter?.past_medications || encounter?.medications || encounter?.medicationOrders),
      pastLabs: normalizeHistory(encounter?.lab_history || encounter?.lab_requests || encounter?.past_lab_requests || encounter?.labResults),
      pastSurgeries: normalizeHistory(encounter?.surgery_history || encounter?.surgeries || encounter?.past_surgeries),
      notes: normalizeHistory(encounter?.inpatient_history || encounter?.clinical_history || encounter?.history || encounter?.progress_notes)
    };

    return {
      patientName: encounter?.patientName || encounter?.firstName || patient?.name || 'Patient',
      mrn: encounter?.patient?.id_number || encounter?.patientId || encounter?.mrn || 'Unknown',
      diagnosis: encounter?.diagnosis || encounter?.triage?.aiTriageReport?.diagnosis || 'Pending assessment',
      history,
      encounterStatus: encounter?.status || 'ACTIVE',
      bed: encounter?.triage?.assigned_bed?.bed_number || encounter?.assigned_bed || 'Not assigned'
    };
  }, [encounter]);

  const openPatientHistory = () => {
    const identifier = encounter?.patient?.id_number || encounter?._id || encounterId;
    if (identifier) navigate(`/ehr?patient=${encodeURIComponent(identifier)}`);
  };

  const handleChange = (field, value) => setForm((prev) => ({ ...prev, [field]: value }));

  const handleSubmit = async () => {
    if (!encounterId || !form.medication_name.trim()) {
      setError('Please enter the medication name before submitting.');
      return;
    }

    setSaving(true);
    setError('');
    setSuccess('');

    try {
      const token = localStorage.getItem('token');
      const decoded = JSON.parse(atob((token || '.').split('.')[1] || '')) || {};
      const patientId = encounter?.patient?.id_number || encounter?.patient?.id || encounter?.patient?._id || encounterId;
      const physicianId = decoded.id;

      const response = await fetch(`${BACKEND_URL}/api/prescriptions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {})
        },
        body: JSON.stringify({
          encounterId,
          patientId,
          physicianId,
          medicationName: form.medication_name,
          prescriber: {
            name: decoded.name || 'Physician',
            role: decoded.role || 'PHYSICIAN',
            specialty: decoded.specialty || 'General'
          },
          patient_details: {
            name: hospitalContext.patientName,
            id_number: hospitalContext.mrn,
            age: encounter?.patient?.age || encounter?.age || null
          },
          items: [{
            medication_name: form.medication_name,
            dosage: form.dosage,
            frequency: form.frequency,
            route: form.route,
            schedule: form.schedule,
            indication: form.indication,
            quantity: 1,
            unit: 'Tablets'
          }],
          special_instructions: form.prescriptionNotes || form.indication,
          substitution_permitted: true
        })
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || data.message || 'Failed to create medication order');

      const prescriptionId = data.prescription?._id || data._id;
      setSuccess('Medication order created successfully.');
      if (prescriptionId) {
        setTimeout(() => navigate(`/prescription/${prescriptionId}`), 400);
      }
    } catch (err) {
      setError(err.message || 'Failed to create medication order');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <div style={{ padding: 32, color: '#475569' }}>Loading medication order workflow…</div>;
  }

  return (
    <div style={{ padding: 28, maxWidth: 1280, margin: '0 auto', fontFamily: 'Inter, system-ui, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ background: 'linear-gradient(135deg, #0f172a, #1e293b)', color: '#fff', borderRadius: 24, padding: 24, marginBottom: 20, boxShadow: '0 20px 40px rgba(15, 23, 42, 0.14)' }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontSize: 24 }}>💊</span>
          <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.14em', opacity: 0.8 }}>Medication order</div>
        </div>
        <h1 style={{ margin: 0, fontSize: 30, fontWeight: 800 }}>{hospitalContext.patientName}</h1>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, marginTop: 14, fontSize: 13, opacity: 0.9 }}>
          <span>MRN: {hospitalContext.mrn}</span>
          <span>Primary diagnosis: {hospitalContext.diagnosis}</span>
          <span>Status: {hospitalContext.encounterStatus}</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 24 }}>
        <section style={{ background: '#fff', borderRadius: 22, border: '1px solid #e2e8f0', padding: 22, boxShadow: '0 12px 24px rgba(15, 23, 42, 0.04)' }}>
          <div style={{ marginBottom: 18 }}>
            <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.14em', color: '#475569', fontWeight: 700 }}>Medication workflow</div>
            <h2 style={{ margin: '8px 0 0', fontSize: 24, color: '#0f172a' }}>New medication order</h2>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
            <div>
              <label style={labelStyle}>Medication name</label>
              <input value={form.medication_name} onChange={(e) => handleChange('medication_name', e.target.value)} style={inputStyle} placeholder="e.g. Amoxicillin 500mg" />
            </div>
            <div>
              <label style={labelStyle}>Dosage</label>
              <input value={form.dosage} onChange={(e) => handleChange('dosage', e.target.value)} style={inputStyle} placeholder="500mg" />
            </div>
            <div>
              <label style={labelStyle}>Frequency</label>
              <input value={form.frequency} onChange={(e) => handleChange('frequency', e.target.value)} style={inputStyle} placeholder="Twice daily" />
            </div>
            <div>
              <label style={labelStyle}>Route</label>
              <select value={form.route} onChange={(e) => handleChange('route', e.target.value)} style={inputStyle}>
                <option>Oral</option>
                <option>IV</option>
                <option>IM</option>
                <option>Subcutaneous</option>
                <option>Topical</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>Schedule</label>
              <input value={form.schedule} onChange={(e) => handleChange('schedule', e.target.value)} style={inputStyle} placeholder="Schedule 3" />
            </div>
            <div>
              <label style={labelStyle}>Priority</label>
              <select onChange={(e) => handleChange('priority', e.target.value)} defaultValue="MEDIUM" style={inputStyle}>
                <option>LOW</option>
                <option>MEDIUM</option>
                <option>HIGH</option>
                <option>URGENT</option>
              </select>
            </div>
          </div>

          <div style={{ marginTop: 16 }}>
            <label style={labelStyle}>Clinical indication</label>
            <textarea value={form.indication} onChange={(e) => handleChange('indication', e.target.value)} placeholder="Why is this medication being ordered?" style={{ ...inputStyle, minHeight: 90, resize: 'vertical' }} />
          </div>

          <div style={{ marginTop: 16 }}>
            <label style={labelStyle}>Prescription notes</label>
            <textarea value={form.prescriptionNotes} onChange={(e) => handleChange('prescriptionNotes', e.target.value)} placeholder="Additional administration / safety instructions" style={{ ...inputStyle, minHeight: 90, resize: 'vertical' }} />
          </div>

          {error && <div style={{ marginTop: 18, background: '#fee2e2', color: '#991b1b', borderRadius: 12, padding: 12 }}>{error}</div>}
          {success && <div style={{ marginTop: 18, background: '#dcfce7', color: '#166534', borderRadius: 12, padding: 12 }}>{success}</div>}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginTop: 22, flexWrap: 'wrap' }}>
            <button onClick={() => navigate(-1)} style={secondaryButton}>Back</button>
            <button onClick={handleSubmit} disabled={saving} style={primaryButton}>
              {saving ? 'Submitting...' : 'Submit medication order'}
            </button>
          </div>
        </section>

        <aside style={{ background: '#fff', borderRadius: 22, border: '1px solid #e2e8f0', padding: 22, boxShadow: '0 12px 24px rgba(15, 23, 42, 0.04)' }}>
          <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.14em', color: '#475569', fontWeight: 700 }}>Hospital context</div>
          <h3 style={{ margin: '10px 0 14px', fontSize: 22, color: '#0f172a' }}>Clinical history</h3>
          <button type="button" onClick={openPatientHistory} style={{ width: '100%', marginBottom: 18, border: '1px solid #0f766e', borderRadius: 10, background: '#ecfdf5', color: '#115e59', padding: '10px 12px', cursor: 'pointer', fontWeight: 700 }}>
            View Patient History
          </button>

          <HistoryCard title="Recent clinician actions" items={hospitalContext.history.notes} accent="#f59e0b" />
        </aside>
      </div>
    </div>
  );
};

const HistoryCard = ({ title, items, accent }) => (
  <div style={{ border: '1px solid #e2e8f0', borderRadius: 18, padding: 14, background: '#f8fafc' }}>
    <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#334155', fontWeight: 700 }}>{title}</div>
    <div style={{ marginTop: 10, display: 'grid', gap: 8, maxHeight: 180, overflowY: 'auto', paddingRight: 4 }}>
      {items.length === 0 ? (
        <div style={{ color: '#64748b', fontSize: 13 }}>No history found for this patient yet.</div>
      ) : (
        items.slice(0, 8).map((item, idx) => {
          const detail = typeof item === 'string' ? item : item?.name || item?.medication_name || item?.lab_type || item?.procedure_name || item?.note || item?.description || item?.action || 'Clinical record';
          const timestamp = typeof item === 'string' ? '' : item?.requestedAt || item?.createdAt || item?.scheduled_at || item?.date || item?.recorded_at || item?.submittedAt || '';
          return (
            <div key={idx} style={{ borderLeft: `3px solid ${accent}`, background: '#fff', borderRadius: 10, padding: '10px 12px', color: '#1e293b' }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{detail}</div>
              {timestamp && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{formatDate(timestamp)}</div>}
            </div>
          );
        })
      )}
    </div>
  </div>
);

const labelStyle = { display: 'block', marginBottom: 8, fontSize: 12, fontWeight: 700, color: '#475569' };
const inputStyle = {
  width: '100%',
  border: '1px solid #cbd5e1',
  borderRadius: 12,
  padding: '12px 14px',
  fontSize: 14,
  background: '#fff',
  color: '#0f172a',
  outline: 'none'
};
const primaryButton = {
  border: 'none',
  borderRadius: 12,
  padding: '12px 22px',
  background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
  color: '#fff',
  fontWeight: 700,
  cursor: 'pointer'
};
const secondaryButton = {
  border: '1px solid #cbd5e1',
  borderRadius: 12,
  padding: '12px 18px',
  background: '#fff',
  color: '#0f172a',
  fontWeight: 700,
  cursor: 'pointer'
};

export default OrderMedicationPage;
