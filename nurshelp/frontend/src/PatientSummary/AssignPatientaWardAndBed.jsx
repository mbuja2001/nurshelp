import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { getBedLabel, getEncounterIdentifier, getPatientDisplayName } from './assignWardBedUtils';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const AssignPatientaWardAndBed = () => {
  const { encounterId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const initialEncounter = location.state?.encounter || null;
  const [encounter, setEncounter] = useState(initialEncounter);
  const [wards, setWards] = useState([]);
  const [beds, setBeds] = useState([]);
  const [selectedWard, setSelectedWard] = useState('');
  const [selectedBed, setSelectedBed] = useState('');
  const [loading, setLoading] = useState(!initialEncounter);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  const patientName = useMemo(() => getPatientDisplayName(encounter), [encounter]);
  const encounterIdentifier = useMemo(() => getEncounterIdentifier(encounter), [encounter]);

  const fetchEncounter = async () => {
    if (!encounterId) return;
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!res.ok) throw new Error('Unable to load encounter');
      const data = await res.json();
      const loaded = data.encounter || data.case || data;
      setEncounter(loaded);
      if (loaded?.triage?.assigned_bed?.ward) {
        setSelectedWard(loaded.triage.assigned_bed.ward);
      }
    } catch (err) {
      setError(err.message || 'Unable to load encounter');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!encounter) {
      fetchEncounter();
    } else {
      setLoading(false);
    }
  }, [encounterId]);

  useEffect(() => {
    const fetchWards = async () => {
      try {
        const token = localStorage.getItem('token');
        const res = await fetch(`${BACKEND_URL}/api/beds/wards`, {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });
        if (!res.ok) throw new Error('Unable to load wards');
        const data = await res.json();
        setWards(data.data || []);
      } catch (err) {
        console.error('[ASSIGN WARD/BED] Unable to load wards', err);
      }
    };

    fetchWards();
  }, []);

  useEffect(() => {
    const fetchBedsForWard = async () => {
      if (!selectedWard) {
        setBeds([]);
        setSelectedBed('');
        return;
      }

      try {
        const token = localStorage.getItem('token');
        const res = await fetch(`${BACKEND_URL}/api/beds/ward/${selectedWard}`, {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });
        if (!res.ok) throw new Error('Unable to load beds');
        const data = await res.json();
        const wardBeds = (data.data?.beds || []).filter((bed) => bed.status === 'Available' || bed.status === 'AVAILABLE');
        setBeds(wardBeds);
        setSelectedBed('');
      } catch (err) {
        console.error('[ASSIGN WARD/BED] Unable to load beds', err);
      }
    };

    fetchBedsForWard();
  }, [selectedWard]);

  const handleAssign = async () => {
    // Validate diagnosis exists
    if (!encounter?.diagnosis?.trim()) {
      setError('Patient must be diagnosed before hospitalization. Please complete the diagnosis first.');
      return;
    }

    if (!encounterIdentifier || !selectedWard || !selectedBed) {
      setError('Select a ward and bed before admitting the patient.');
      return;
    }

    setSaving(true);
    setError(null);
    setMessage(null);

    try {
      const token = localStorage.getItem('token');

      const dispositionRes = await fetch(`${BACKEND_URL}/api/encounters/${encounterIdentifier}/disposition`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          disposition: 'HOSPITALIZED',
          disposition_notes: `Admitted to ${selectedWard} / ${selectedBed}`
        })
      });

      if (!dispositionRes.ok) {
        const errData = await dispositionRes.json().catch(() => ({}));
        throw new Error(errData.message || 'Unable to admit patient');
      }

      const assignRes = await fetch(`${BACKEND_URL}/api/clinical/encounters/${encounterIdentifier}/assign-bed`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          bedId: selectedBed,
          hospitalize: true
        })
      });

      if (!assignRes.ok) {
        const errData = await assignRes.json().catch(() => ({}));
        throw new Error(errData.message || 'Unable to assign bed');
      }

      const updated = await assignRes.json();
      setMessage(`Patient admitted and assigned to ${getBedLabel(beds.find((bed) => bed._id === selectedBed))} in ${selectedWard}.`);
      navigate(`/diagnose/${encounterIdentifier}`, { state: { encounter: { ...(encounter || {}), status: 'HOSPITALIZED', triage: { ...(encounter?.triage || {}), assigned_bed: { id: selectedBed, bed_number: getBedLabel(beds.find((bed) => bed._id === selectedBed)), ward: selectedWard } } } }, replace: true });
    } catch (err) {
      setError(err.message || 'Unable to complete admission');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 1040, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <div>
          <p style={{ margin: 0, fontSize: 12, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#64748b' }}>Inpatient placement</p>
          <h1 style={{ margin: '8px 0 0 0', fontSize: 32, color: '#111827' }}>Assign ward and bed</h1>
          <p style={{ margin: '10px 0 0 0', color: '#475569', maxWidth: 700 }}>Select the active ward and bed for the patient before finalizing the hospital admission.</p>
        </div>
        <button onClick={() => navigate(-1)} style={{ padding: '12px 18px', borderRadius: 14, border: '1px solid #cbd5e1', background: 'white', color: '#475569', fontWeight: 700, cursor: 'pointer' }}>
          ← Back
        </button>
      </div>

      {message && <div style={{ padding: 14, borderRadius: 14, background: '#ecfdf5', color: '#166534', marginBottom: 16 }}>{message}</div>}
      {error && <div style={{ padding: 14, borderRadius: 14, background: '#fef2f2', color: '#b91c1c', marginBottom: 16 }}>{error}</div>}

      <div style={{ display: 'grid', gap: 24, gridTemplateColumns: '1.15fr 0.85fr' }}>
        <div style={{ background: 'white', borderRadius: 24, padding: 24, boxShadow: '0 18px 45px rgba(15,23,42,0.08)' }}>
          <p style={{ margin: 0, fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Patient</p>
          <h2 style={{ margin: '10px 0 0 0', fontSize: 24, color: '#0f172a' }}>{patientName}</h2>
          <p style={{ margin: '8px 0 0 0', color: '#475569' }}>Encounter ID: {encounterIdentifier || 'Unknown'}</p>

          <div style={{ display: 'grid', gap: 16, marginTop: 24 }}>
            <label style={{ display: 'grid', gap: 8 }}>
              <span style={{ fontWeight: 700, color: '#111827' }}>Ward</span>
              <select value={selectedWard} onChange={(e) => setSelectedWard(e.target.value)} style={{ padding: 12, borderRadius: 12, border: '1px solid #cbd5e1' }}>
                <option value=''>Select a ward</option>
                {wards.map((ward) => (
                  <option key={ward._id} value={ward._id}>{ward.wardName || ward.name} ({ward.department || 'Ward'})</option>
                ))}
              </select>
            </label>

            <label style={{ display: 'grid', gap: 8 }}>
              <span style={{ fontWeight: 700, color: '#111827' }}>Available bed</span>
              <select value={selectedBed} onChange={(e) => setSelectedBed(e.target.value)} style={{ padding: 12, borderRadius: 12, border: '1px solid #cbd5e1' }}>
                <option value=''>Select a bed</option>
                {beds.map((bed) => (
                  <option key={bed._id} value={bed._id}>{getBedLabel(bed)}</option>
                ))}
              </select>
            </label>
          </div>

          <button onClick={handleAssign} disabled={saving || !selectedWard || !selectedBed || !encounter?.diagnosis?.trim()} style={{ marginTop: 24, padding: '14px 20px', borderRadius: 16, border: 'none', background: (saving || !selectedWard || !selectedBed || !encounter?.diagnosis?.trim()) ? '#94a3b8' : '#2563eb', color: 'white', fontWeight: 700, cursor: (saving || !selectedWard || !selectedBed || !encounter?.diagnosis?.trim()) ? 'not-allowed' : 'pointer' }}>
            {saving ? 'Admitting…' : 'Finalize admission'}
          </button>
          {!encounter?.diagnosis?.trim() && (
            <p style={{ marginTop: 12, fontSize: 13, color: '#dc2626', fontWeight: 600 }}>
              ⚠️ Diagnosis required before admission
            </p>
          )}
        </div>

        <div style={{ background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.96))', color: 'white', borderRadius: 24, padding: 24, boxShadow: '0 18px 45px rgba(15,23,42,0.14)' }}>
          {!encounter?.diagnosis?.trim() && (
            <div style={{ background: 'rgba(220, 38, 38, 0.15)', border: '1px solid rgba(220, 38, 38, 0.4)', borderRadius: 12, padding: 16, marginBottom: 20 }}>
              <p style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#fca5a5', display: 'flex', alignItems: 'center', gap: 8 }}>
                ⚠️ Diagnosis Required
              </p>
              <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#fecaca', lineHeight: 1.6 }}>
                The patient must have a documented diagnosis before being admitted to an inpatient ward. Complete the clinical diagnosis step first.
              </p>
            </div>
          )}
          <p style={{ margin: 0, textTransform: 'uppercase', letterSpacing: '0.16em', fontSize: 12, color: '#93c5fd' }}>Admission workflow</p>
          <h3 style={{ margin: '12px 0 0 0', fontSize: 22 }}>Why this step matters</h3>
          <ul style={{ margin: '16px 0 0 0', lineHeight: 1.8, color: '#cbd5e1' }}>
            <li>Patient must have a documented diagnosis.</li>
            <li>Confirms the bed is actually available.</li>
            <li>Links the patient to the inpatient ward and bed in the system.</li>
            <li>Lets the physician continue into the treatment workflow without losing the admission context.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AssignPatientaWardAndBed;
