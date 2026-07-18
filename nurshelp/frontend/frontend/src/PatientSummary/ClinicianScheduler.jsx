import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { FiChevronLeft, FiCheck, FiX, FiRefreshCw } from 'react-icons/fi';

const ClinicianScheduler = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const encounterId = searchParams.get('encounterId');
  const mode = searchParams.get('mode');

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

  // Helper to format availability hours
  const formatAvailabilityHours = (clinician) => {
    // Check for clock-in time
    const clockInTime = clinician.clock_in_time || clinician.clockInTime || clinician.shift_start || clinician.shiftStart;
    const clockOutTime = clinician.clock_out_time || clinician.clockOutTime || clinician.shift_end || clinician.shiftEnd;
    
    if (!clockInTime) return 'Hours unknown';
    
    try {
      const inTime = new Date(clockInTime);
      const outTime = clockOutTime ? new Date(clockOutTime) : null;
      
      const inStr = inTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
      const outStr = outTime 
        ? outTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
        : 'End time not set';
      
      return `${inStr} - ${outStr}`;
    } catch (e) {
      return 'Hours unknown';
    }
  };

  // State
  const [clinicians, setClinicians] = useState([]);
  const [selectedClinicians, setSelectedClinicians] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [roleFilter, setRoleFilter] = useState('all'); // 'all', 'physician', 'nurse'

  // Fetch clocked-in staff
  useEffect(() => {
    const fetchClockedInStaff = async () => {
      try {
        setLoading(true);
        setError(null);

        // Get ALL clocked-in staff (regardless of specialty)
        const res = await fetch(
          `${BACKEND_URL}/api/clinical/staff/clocked-in`,
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );

        if (!res.ok) throw new Error('Failed to fetch clinicians');

        const data = await res.json();
        const staffList = data.clinicians || data.data || [];

        // Separate by role
        const physicians = staffList.filter(s => s.role === 'PHYSICIAN') || [];
        const nurses = staffList.filter(s => s.role === 'NURSE') || [];

        setClinicians([
          ...physicians.map(s => ({ ...s, role: 'PHYSICIAN' })),
          ...nurses.map(s => ({ ...s, role: 'NURSE' }))
        ]);
      } catch (err) {
        console.error('Error fetching clinicians:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchClockedInStaff();
  }, [token, BACKEND_URL]);

  // Toggle clinician selection
  const toggleClinician = (clinician) => {
    setSelectedClinicians(prev => {
      const isSelected = prev.find(c => c.id === clinician.id);
      if (isSelected) {
        return prev.filter(c => c.id !== clinician.id);
      } else {
        return [...prev, clinician];
      }
    });
  };

  // Save selected clinicians and return to encounter
  const handleSave = async () => {
    if (selectedClinicians.length === 0) {
      alert('Please select at least one clinician');
      return;
    }

    try {
      setSaving(true);

      // Update encounter with manually selected clinicians
      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/ai-fields`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          changedFields: {
            manually_selected_clinicians: selectedClinicians.map(c => ({
              id: c.id,
              name: c.name,
              role: c.role,
              specialty: c.specialty,
              score: c.score || 0.85,
              clock_in_time: c.clock_in_time,
              clock_out_time: c.clock_out_time,
              shift_start: c.shift_start,
              shift_end: c.shift_end,
              selected_at: new Date()
            }))
          }
        })
      });

      if (!res.ok) throw new Error('Failed to save clinicians');

      console.log('[ClinicianScheduler] ✅ Clinicians saved, navigating back with refetchEncounter flag');
      // Redirect back to patient report with flag to refetch encounter data
      navigate(`/summary/${encounterId}`, { state: { refetchEncounter: true } });
    } catch (err) {
      console.error('Error saving clinicians:', err);
      alert(`Error: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  // Filter clinicians by role
  const filteredClinicians = clinicians.filter(c => {
    if (roleFilter === 'all') return true;
    if (roleFilter === 'physician') return c.role === 'PHYSICIAN';
    if (roleFilter === 'nurse') return c.role === 'NURSE';
    return true;
  });

  // Group by role for display
  const physicians = filteredClinicians.filter(c => c.role === 'PHYSICIAN');
  const nurses = filteredClinicians.filter(c => c.role === 'NURSE');

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20 }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
              Select Available Clinical Staff
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>
              Choose from all clocked-in physicians and nurses, regardless of specialty
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
            <FiX size={16} /> Cancel
          </button>
        </div>

        {/* Role Filter */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
          {['all', 'physician', 'nurse'].map(role => (
            <button
              key={role}
              onClick={() => setRoleFilter(role)}
              style={{
                padding: '8px 16px',
                background: roleFilter === role ? '#3b82f6' : '#e2e8f0',
                color: roleFilter === role ? 'white' : '#475569',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: 13,
                textTransform: 'capitalize'
              }}
            >
              {role === 'all' ? 'All Staff' : role === 'physician' ? 'Physicians' : 'Nurses'}
            </button>
          ))}
        </div>

        {/* Error Message */}
        {error && (
          <div style={{
            padding: 12,
            marginBottom: 16,
            background: '#fee2e2',
            border: '1px solid #fecaca',
            borderRadius: 6,
            color: '#991b1b',
            fontSize: 14
          }}>
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{
            padding: 40,
            textAlign: 'center',
            color: '#64748b',
            fontSize: 16
          }}>
            Loading clocked-in clinicians...
          </div>
        )}

        {/* Staff Lists */}
        {!loading && (
          <>
            {/* Physicians */}
            {(roleFilter === 'all' || roleFilter === 'physician') && (
              <section style={{ marginBottom: 24 }}>
                <h2 style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: '#2563eb',
                  marginBottom: 12,
                  paddingBottom: 8,
                  borderBottom: '2px solid #bfdbfe'
                }}>
                  👨‍⚕️ Physicians ({physicians.length})
                </h2>
                {physicians.length === 0 ? (
                  <div style={{ color: '#94a3b8', fontSize: 14 }}>No physicians clocked in</div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                    {physicians.map(clinician => (
                      <div
                        key={clinician.id}
                        onClick={() => toggleClinician(clinician)}
                        style={{
                          padding: 16,
                          background: selectedClinicians.find(c => c.id === clinician.id) ? '#dbeafe' : 'white',
                          border: selectedClinicians.find(c => c.id === clinician.id)
                            ? '2px solid #3b82f6'
                            : '1px solid #e2e8f0',
                          borderRadius: 8,
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          position: 'relative'
                        }}
                      >
                        {selectedClinicians.find(c => c.id === clinician.id) && (
                          <div style={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            background: '#3b82f6',
                            color: 'white',
                            borderRadius: '50%',
                            width: 24,
                            height: 24,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}>
                            <FiCheck size={14} />
                          </div>
                        )}
                        <div style={{ fontWeight: 700, fontSize: 14, color: '#0f172a', marginBottom: 4 }}>
                          {clinician.name}
                        </div>
                        <div style={{ fontSize: 12, color: '#475569', marginBottom: 2 }}>
                          MD • {clinician.specialty || 'General Medicine'}
                        </div>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                          🕒 {formatAvailabilityHours(clinician)}
                        </div>
                        <div style={{ fontSize: 11, color: '#64748b' }}>
                          Available
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* Nurses */}
            {(roleFilter === 'all' || roleFilter === 'nurse') && (
              <section style={{ marginBottom: 24 }}>
                <h2 style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: '#7c3aed',
                  marginBottom: 12,
                  paddingBottom: 8,
                  borderBottom: '2px solid #ede9fe'
                }}>
                  👩‍⚕️ Nurses ({nurses.length})
                </h2>
                {nurses.length === 0 ? (
                  <div style={{ color: '#94a3b8', fontSize: 14 }}>No nurses clocked in</div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                    {nurses.map(clinician => (
                      <div
                        key={clinician.id}
                        onClick={() => toggleClinician(clinician)}
                        style={{
                          padding: 16,
                          background: selectedClinicians.find(c => c.id === clinician.id) ? '#faf5ff' : 'white',
                          border: selectedClinicians.find(c => c.id === clinician.id)
                            ? '2px solid #7c3aed'
                            : '1px solid #e2e8f0',
                          borderRadius: 8,
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          position: 'relative'
                        }}
                      >
                        {selectedClinicians.find(c => c.id === clinician.id) && (
                          <div style={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            background: '#7c3aed',
                            color: 'white',
                            borderRadius: '50%',
                            width: 24,
                            height: 24,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}>
                            <FiCheck size={14} />
                          </div>
                        )}
                        <div style={{ fontWeight: 700, fontSize: 14, color: '#0f172a', marginBottom: 4 }}>
                          {clinician.name}
                        </div>
                        <div style={{ fontSize: 12, color: '#475569', marginBottom: 2 }}>
                          RN • {clinician.specialty || 'Nursing'}
                        </div>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                          🕒 {formatAvailabilityHours(clinician)}
                        </div>
                        <div style={{ fontSize: 11, color: '#64748b' }}>
                          Available
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}
          </>
        )}

        {/* Summary & Save */}
        <div style={{
          position: 'sticky',
          bottom: 0,
          padding: 16,
          background: 'white',
          borderTop: '1px solid #e2e8f0',
          borderRadius: 8,
          marginTop: 24,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 -4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#475569' }}>
            {selectedClinicians.length} clinician{selectedClinicians.length !== 1 ? 's' : ''} selected
          </div>
          <button
            onClick={handleSave}
            disabled={selectedClinicians.length === 0 || saving}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '10px 24px',
              background: selectedClinicians.length === 0 ? '#cbd5e1' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: selectedClinicians.length === 0 ? 'not-allowed' : 'pointer',
              fontWeight: 600,
              fontSize: 14,
              opacity: saving ? 0.7 : 1
            }}
          >
            {saving ? (
              <>
                <FiRefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />
                Saving...
              </>
            ) : (
              <>
                <FiCheck size={16} />
                Save & Continue
              </>
            )}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default ClinicianScheduler;
