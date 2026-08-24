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

const valueOrNA = (value) => (value === null || value === undefined || value === '' ? 'N/A' : value);

const RequestLabPage = () => {
  const navigate = useNavigate();
  const { encounterId, specialty: routeSpecialty } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState('');
  const [selectedSpecialty, setSelectedSpecialty] = useState(routeSpecialty ? routeSpecialty.toUpperCase() : 'RADIOLOGY');
  const [form, setForm] = useState({
    specialty: routeSpecialty ? routeSpecialty.toUpperCase() : 'RADIOLOGY',
    lab_type: '',
    description: '',
    priority: 'MEDIUM',
    patientPregnancyStatus: 'Negative',
    allergies: '',
    eGFR: '',
    serumCreatinine: '',
    dateOfLab: '',
    implants: 'None',
    modality: 'X-Ray',
    anatomicalTarget: 'N/A',
    bodyPart: '',
    specimenType: '',
    clinicianName: '',
    clinicianSignature: '',
    contactNumber: ''
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
        const nextEncounter = data.encounter || data;
        setEncounter(nextEncounter);

        const decoded = token ? JSON.parse(atob(token.split('.')[1])) : {};
        setForm((prev) => ({
          ...prev,
          clinicianName: decoded.name || prev.clinicianName || 'Clinician',
          clinicianSignature: decoded.name || prev.clinicianSignature || 'Clinician Signature'
        }));
      } catch (err) {
        setError(err.message || 'Unable to load patient context');
      } finally {
        setLoading(false);
      }
    };

    if (encounterId) fetchEncounter();
  }, [encounterId]);

  useEffect(() => {
    const nextSpecialty = routeSpecialty ? routeSpecialty.toUpperCase() : selectedSpecialty;
    setSelectedSpecialty(nextSpecialty);
    setForm((prev) => ({ ...prev, specialty: nextSpecialty }));
  }, [routeSpecialty, selectedSpecialty]);

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
      dob: encounter?.patient?.dob || encounter?.dob || 'Unknown',
      mrn: encounter?.patient?.id_number || encounter?.patientId || encounter?.mrn || 'Unknown',
      diagnosis: encounter?.diagnosis || encounter?.triage?.aiTriageReport?.diagnosis || 'Pending assessment',
      location: encounter?.assigned_ward || encounter?.location || encounter?.triage?.assigned_bed?.bed_number || 'Ward/Clinic/OPD',
      history,
      encounterStatus: encounter?.status || 'ACTIVE',
      bed: encounter?.triage?.assigned_bed?.bed_number || encounter?.assigned_bed || 'Not assigned'
    };
  }, [encounter]);

  const handleSpecialtySelect = (specialty) => {
    const normalized = String(specialty).toUpperCase();
    setSelectedSpecialty(normalized);
    setForm((prev) => ({ ...prev, specialty: normalized }));
    navigate(`/request-lab/${encounterId}/${normalized.toLowerCase()}`);
  };

  const handleSubmit = async () => {
    if (!encounterId || !form.lab_type || !form.description.trim()) {
      setError('Please complete the requested test and the core clinical question before submitting.');
      return;
    }

    setSaving(true);
    setError('');
    setSuccess('');

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${BACKEND_URL}/api/lab-requests/${encounterId}/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {})
        },
        body: JSON.stringify({
          specialty: selectedSpecialty,
          lab_type: form.lab_type,
          description: form.description,
          priority: form.priority,
          requested_by: form.clinicianName || 'Clinician',
          clinical_context: {
            patientPregnancyStatus: form.patientPregnancyStatus,
            allergies: form.allergies,
            eGFR: form.eGFR,
            serumCreatinine: form.serumCreatinine,
            dateOfLab: form.dateOfLab,
            implants: form.implants,
            modality: form.modality,
            anatomicalTarget: form.anatomicalTarget,
            bodyPart: form.bodyPart,
            priorImaging: form.priorImaging,
            priorImagingDetails: form.priorImagingDetails,
            clinicianName: form.clinicianName,
            clinicianSignature: form.clinicianSignature,
            contactNumber: form.contactNumber
          },
          diagnosis: hospitalContext.diagnosis,
          patientName: hospitalContext.patientName,
          patientMrn: hospitalContext.mrn,
          patientLocation: hospitalContext.location,
          official_request: true
        })
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Failed to create lab request');

      setSuccess('Official lab request submitted successfully. Opening the signed request...');
      navigate(`/report/lab/${encounterId}`, {
        state: {
          request: data.request || data.labRequest || data,
          form,
          specialty: selectedSpecialty,
          hospitalContext
        }
      });
    } catch (err) {
      setError(err.message || 'Failed to create lab request');
    } finally {
      setSaving(false);
    }
  };


  if (loading) {
    return <div style={{ padding: 32, color: '#475569' }}>Loading lab request workflow…</div>;
  }

  if (!routeSpecialty) {
    return (
      <div style={{ padding: 32, maxWidth: 980, margin: '0 auto', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif', background: '#f8fafc' }}>
        <div style={{ background: '#fff', borderRadius: 28, border: '1px solid #e2e8f0', boxShadow: '0 18px 40px rgba(15, 23, 42, 0.04)', padding: 28 }}>
          <div style={{ fontSize: 12, letterSpacing: '0.14em', color: '#475569', textTransform: 'uppercase', fontWeight: 700 }}>Lab ordering workflow</div>
          <h2 style={{ margin: '10px 0 8px', fontSize: 28, color: '#0f172a' }}>Choose specialist request</h2>
          <p style={{ margin: '0 0 24px', color: '#475569', fontSize: 16, lineHeight: 1.6 }}>
            Select the specialist team for the requested workup. Each pathway opens a focused clinical request form.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 18, alignItems: 'stretch', minWidth: 0 }}>
            {['RADIOLOGY', 'PATHOLOGY'].map((specialty) => {
              const isSelected = selectedSpecialty === specialty;
              return (
                <button
                  key={specialty}
                  type="button"
                  onClick={() => handleSpecialtySelect(specialty)}
                  style={{
                    border: isSelected ? '1px solid #0f172a' : '1px solid #cbd5e1',
                    borderRadius: 20,
                    background: isSelected ? 'linear-gradient(135deg, #0f172a, #1d4ed8)' : 'linear-gradient(135deg, #dfeaf5, #edf3ff)',
                    color: isSelected ? '#ffffff' : '#0f172a',
                    padding: '24px 18px',
                    textAlign: 'left',
                    cursor: 'pointer',
                    boxShadow: isSelected ? '0 16px 28px rgba(15, 23, 42, 0.18)' : '0 12px 24px rgba(15, 23, 42, 0.04)',
                    width: '100%',
                    minHeight: 200,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: 12,
                    wordBreak: 'break-word',
                    overflowWrap: 'anywhere',
                    lineHeight: 1.35,
                    whiteSpace: 'normal'
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 12, letterSpacing: '0.14em', textTransform: 'uppercase', color: isSelected ? 'rgba(255,255,255,0.8)' : '#475569', marginBottom: 8, fontWeight: 700 }}>
                      {specialty === 'RADIOLOGY' ? 'Imaging' : 'Laboratory'}
                    </div>
                    <div style={{ fontSize: 27, fontWeight: 800, marginBottom: 10, lineHeight: 1.15, wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                      {specialty === 'RADIOLOGY' ? 'Radiology' : 'Pathology'}
                    </div>
                  </div>
                  <div style={{ color: isSelected ? '#e2e8f0' : '#475569', lineHeight: 1.45, fontSize: 15, maxWidth: '100%', wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                    {specialty === 'RADIOLOGY'
                      ? 'Chest X-ray, CT, MRI, ultrasound, contrast screening.'
                      : 'CBC, blood culture, chemistry, histology, tissue and specimen work-up requests.'}
                  </div>
                </button>
              );
            })}
          </div>

          <div style={{ marginTop: 24, display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
            <button onClick={() => navigate(-1)} style={secondaryButton}>Back</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: 28, maxWidth: 1420, margin: '0 auto', fontFamily: 'Inter, system-ui, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ background: 'linear-gradient(135deg, #0f172a, #1e293b)', color: '#fff', borderRadius: 24, padding: 24, marginBottom: 20, boxShadow: '0 20px 40px rgba(15, 23, 42, 0.14)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
          <div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: 24 }}>🧪</span>
              <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.14em', opacity: 0.8 }}>Official lab request</div>
            </div>
            <h1 style={{ margin: 0, fontSize: 30, fontWeight: 800 }}>{hospitalContext.patientName}</h1>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <button onClick={() => navigate(-1)} style={secondaryButton}>Back</button>
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, marginTop: 14, fontSize: 13, opacity: 0.9 }}>
          <span>MRN: {hospitalContext.mrn}</span>
          <span>DOB: {hospitalContext.dob}</span>
          <span>Primary diagnosis: {hospitalContext.diagnosis}</span>
          <span>Status: {hospitalContext.encounterStatus}</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.25fr 0.75fr', gap: 24 }}>
        <section style={{ background: '#fff', borderRadius: 22, border: '1px solid #e2e8f0', padding: 22, boxShadow: '0 12px 24px rgba(15, 23, 42, 0.04)' }}>
          <div style={{ marginBottom: 18 }}>
            <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.14em', color: '#475569', fontWeight: 700 }}>
              {selectedSpecialty === 'RADIOLOGY' ? 'Radiology request' : 'Pathology request'}
            </div>
            <h2 style={{ margin: '8px 0 0', fontSize: 24, color: '#0f172a' }}>
              {selectedSpecialty === 'RADIOLOGY' ? 'Imaging request form' : 'Specimen request form'}
            </h2>
            <p style={{ margin: '8px 0 0', color: '#475569', fontSize: 14, lineHeight: 1.5 }}>
              {selectedSpecialty === 'RADIOLOGY'
                ? 'Document the imaging need, protocol details, and safety screening before dispatch.'
                : 'Document the specimen type, pathology request, and clinical context for the laboratory team.'}
            </p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 20, flexWrap: 'wrap' }}>
            <div style={{ fontSize: 12, letterSpacing: '0.14em', textTransform: 'uppercase', color: '#475569', fontWeight: 700 }}>
              {selectedSpecialty === 'RADIOLOGY' ? 'Radiology selected' : 'Pathology selected'}
            </div>
            <button
              type="button"
              onClick={() => navigate(`/request-lab/${encounterId}`)}
              style={{ ...secondaryButton, padding: '10px 14px', fontSize: 13 }}
            >
              Change specialty
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
            <div>
              <label style={labelStyle}>Requested modality / test</label>
              <input
                value={form.lab_type}
                onChange={(e) => setForm((prev) => ({ ...prev, lab_type: e.target.value }))}
                style={inputStyle}
                placeholder={selectedSpecialty === 'RADIOLOGY' ? 'Chest X-ray / CT Scan / MRI' : 'CBC / Blood Culture / Histology'}
              />
            </div>
            <div>
              <label style={labelStyle}>Priority</label>
              <select value={form.priority} onChange={(e) => setForm((prev) => ({ ...prev, priority: e.target.value }))} style={inputStyle}>
                <option value="MEDIUM">Medium</option>
                <option value="HIGH">High</option>
                <option value="URGENT">Urgent</option>
              </select>
            </div>
          </div>

          {selectedSpecialty === 'PATHOLOGY' && (
            <div style={{ marginTop: 16 }}> 
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Specimen / Sample Type</label>
                  <input
                    value={form.specimenType}
                    onChange={(e) => setForm((prev) => ({ ...prev, specimenType: e.target.value }))}
                    style={inputStyle}
                    placeholder="Blood, urine, tissue, swab, aspirate..."
                  />
                </div>
              </div>
            </div>
          )}

          {selectedSpecialty === 'RADIOLOGY' ? (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
                <div>
                  <label style={labelStyle}>Requested Modality</label>
                  <select value={form.modality} onChange={(e) => setForm((prev) => ({ ...prev, modality: e.target.value }))} style={inputStyle}>
                    <option>X-Ray</option>
                    <option>CT</option>
                    <option>MRI</option>
                    <option>Ultrasound</option>
                    <option>PET-CT</option>
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Anatomical Target & Laterality</label>
                  <select value={form.anatomicalTarget} onChange={(e) => setForm((prev) => ({ ...prev, anatomicalTarget: e.target.value }))} style={inputStyle}>
                    <option>Left</option>
                    <option>Right</option>
                    <option>Bilateral</option>
                    <option>N/A</option>
                  </select>
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Specific Body Part / Protocol</label>
                  <input value={form.bodyPart} onChange={(e) => setForm((prev) => ({ ...prev, bodyPart: e.target.value }))} style={inputStyle} placeholder="Chest, abdomen, pelvis, protocol details..." />
                </div>
              </div>
            </div>
          ) : (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Specimen / Sample Type</label>
                  <input value={form.specimenType} onChange={(e) => setForm((prev) => ({ ...prev, specimenType: e.target.value }))} style={inputStyle} placeholder="Blood, urine, tissue, swab, aspirate..." />
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Requested Pathology Test</label>
                  <input value={form.lab_type} onChange={(e) => setForm((prev) => ({ ...prev, lab_type: e.target.value }))} style={inputStyle} placeholder="CBC, blood cultures, chemistry panel, histology..." />
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: 18 }}>
            <label style={labelStyle}>Core clinical question</label>
            <textarea value={form.description} onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))} placeholder="Rule out acute appendicitis / assess sepsis / evaluate infiltrates / confirm suspected pathology" style={{ ...inputStyle, minHeight: 120, resize: 'vertical' }} />
          </div>

          {selectedSpecialty === 'RADIOLOGY' && (
            <div style={{ marginTop: 18 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
                <div>
                  <label style={labelStyle}>Pregnancy Status</label>
                  <select value={form.patientPregnancyStatus} onChange={(e) => setForm((prev) => ({ ...prev, patientPregnancyStatus: e.target.value }))} style={inputStyle}>
                    <option>Negative</option>
                    <option>Positive</option>
                    <option>N/A</option>
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Known Allergies / Contrast</label>
                  <input value={form.allergies} onChange={(e) => setForm((prev) => ({ ...prev, allergies: e.target.value }))} style={inputStyle} placeholder="Contrast / iodine / shellfish allergies" />
                </div>
                <div>
                  <label style={labelStyle}>eGFR</label>
                  <input value={form.eGFR} onChange={(e) => setForm((prev) => ({ ...prev, eGFR: e.target.value }))} style={inputStyle} placeholder="mL/min/1.73m²" />
                </div>
                <div>
                  <label style={labelStyle}>Serum Creatinine</label>
                  <input value={form.serumCreatinine} onChange={(e) => setForm((prev) => ({ ...prev, serumCreatinine: e.target.value }))} style={inputStyle} placeholder="mg/dL" />
                </div>
                <div>
                  <label style={labelStyle}>Date of Lab</label>
                  <input type="date" value={form.dateOfLab} onChange={(e) => setForm((prev) => ({ ...prev, dateOfLab: e.target.value }))} style={inputStyle} />
                </div>
                <div>
                  <label style={labelStyle}>Implants</label>
                  <select value={form.implants} onChange={(e) => setForm((prev) => ({ ...prev, implants: e.target.value }))} style={inputStyle}>
                    <option>None</option>
                    <option>Pacemaker</option>
                    <option>Metal Fragments</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: 18 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
              <div>
                <label style={labelStyle}>Requesting physician name</label>
                <input value={form.clinicianName} onChange={(e) => setForm((prev) => ({ ...prev, clinicianName: e.target.value }))} style={inputStyle} placeholder="Doctor name / signature" />
              </div>
              <div>
                <label style={labelStyle}>Contact / bleep number</label>
                <input value={form.contactNumber} onChange={(e) => setForm((prev) => ({ ...prev, contactNumber: e.target.value }))} style={inputStyle} placeholder="For critical findings" />
              </div>
            </div>
          </div>

          {error && <div style={{ marginTop: 18, background: '#fee2e2', color: '#991b1b', borderRadius: 12, padding: 12 }}>{error}</div>}
          {success && <div style={{ marginTop: 18, background: '#dcfce7', color: '#166534', borderRadius: 12, padding: 12 }}>{success}</div>}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginTop: 22, flexWrap: 'wrap' }}>
            <button onClick={() => navigate(-1)} style={secondaryButton}>Back</button>
            <button onClick={handleSubmit} disabled={saving} style={primaryButton}>
              {saving ? 'Submitting...' : 'Submit official request'}
            </button>
          </div>
        </section>

        <aside style={{ background: '#fff', borderRadius: 22, border: '1px solid #e2e8f0', padding: 22, boxShadow: '0 12px 24px rgba(15, 23, 42, 0.04)' }}>
          <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.14em', color: '#475569', fontWeight: 700 }}>EHR context</div>
          <h3 style={{ margin: '10px 0 14px', fontSize: 22, color: '#0f172a' }}>Patient history rail</h3>
          <p style={{ color: '#475569', fontSize: 13, lineHeight: 1.55, margin: '0 0 16px' }}>This request becomes a timestamped event in the longitudinal EHR after submission.</p>
          <div style={{ display: 'grid', gap: 16, marginTop: 16 }}>
            <HistoryCard title="Past medications" items={hospitalContext.history.pastMeds} accent="#10b981" compact />
            <HistoryCard title="Lab requests" items={hospitalContext.history.pastLabs} accent="#3b82f6" compact />
            <HistoryCard title="Past surgeries" items={hospitalContext.history.pastSurgeries} accent="#8b5cf6" compact />
          </div>
        </aside>
      </div>
    </div>
  );
};

const HistoryCard = ({ title, items, accent, compact = false }) => (
  <div style={{ border: '1px solid #e2e8f0', borderRadius: 18, padding: compact ? 10 : 14, background: '#f8fafc' }}>
    <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#334155', fontWeight: 700 }}>{title}</div>
    <div style={{ marginTop: 8, display: 'grid', gap: 8, maxHeight: compact ? 120 : 180, overflowY: 'auto', paddingRight: 4 }}>
      {items.length === 0 ? (
        <div style={{ color: '#64748b', fontSize: 13 }}>No history found for this patient yet.</div>
      ) : (
        items.slice(0, compact ? 4 : 8).map((item, idx) => {
          const detail = typeof item === 'string' ? item : item?.name || item?.medication_name || item?.lab_type || item?.procedure_name || item?.note || item?.description || item?.action || 'Clinical record';
          const timestamp = typeof item === 'string' ? '' : item?.requestedAt || item?.createdAt || item?.scheduled_at || item?.date || item?.recorded_at || item?.submittedAt || '';
          return (
            <div key={idx} style={{ borderLeft: `3px solid ${accent}`, background: '#fff', borderRadius: 10, padding: '8px 10px', color: '#1e293b' }}>
              <div style={{ fontSize: 12, fontWeight: 700 }}>{detail}</div>
              {timestamp && <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{formatDate(timestamp)}</div>}
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
const tabButton = {
  border: '1px solid #cbd5e1',
  borderRadius: 999,
  padding: '10px 18px',
  fontWeight: 700,
  cursor: 'pointer',
  fontSize: 13
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

export default RequestLabPage;
