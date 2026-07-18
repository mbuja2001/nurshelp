import React, { useEffect, useState } from 'react';
import './StaffScheduleDisplay.css';

/**
 * StaffScheduleDisplay - Shows available nurses, physicians, and beds
 * Used in both Manual and AI-Assisted triage for staff selection
 */
function StaffScheduleDisplay({ onStaffSelect = {} }) {
  const [staffData, setStaffData] = useState({
    nurses: [],
    physicians: [],
    beds: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedNurse, setSelectedNurse] = useState(null);
  const [selectedPhysician, setSelectedPhysician] = useState(null);
  const [selectedBed, setSelectedBed] = useState(null);

  const BACKEND_URL =
    import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

  // Fetch available staff schedule
  useEffect(() => {
    const fetchStaffSchedule = async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem('token');

        const response = await fetch(
          `${BACKEND_URL}/api/triage/staff?date=${new Date().toISOString().split('T')[0]}`,
          {
            headers: token ? { Authorization: `Bearer ${token}` } : {}
          }
        );

        if (!response.ok) {
          throw new Error(`Failed to fetch staff schedule: ${response.status}`);
        }

        const response_data = await response.json();
        const liveData = response_data.data || { nurses: [], physicians: [], beds: [] };
        
        // Normalize live data to match UI expectations
        const normalizedData = {
          nurses: (liveData.nurses || []).map(n => ({
            id: n._id || n.nurseId,
            name: n.name,
            specialty: n.specialty || `ESI Level ${n.esiLevel || 3}`,
            available: n.isAvailable !== false,
            currentBedCount: n.currentBedCount || 0,
            maxBeds: n.maxBeds || 5
          })),
          physicians: (liveData.physicians || []).map(p => ({
            id: p._id || p.physicianId,
            name: p.name,
            specialty: p.specialty || 'General',
            available: p.isAvailable !== false,
            currentPatientCount: p.currentPatientCount || 0,
            maxPatients: p.maxPatients || 20
          })),
          beds: (liveData.beds || []).map(b => ({
            id: b.bedNumber || b._id,
            department: b.department || 'General',
            status: b.status?.toLowerCase() || 'available',
            acuityLevel: b.acuityLevel || 'General',
            bedType: b.bedType || 'Standard'
          }))
        };
        
        console.log('✅ Live staff data loaded:', normalizedData);
        setStaffData(normalizedData);
        setError(null);
      } catch (err) {
        console.error('Error fetching staff schedule:', err);
        setError(err.message);
        console.log('⚠ Temporarily using mock data (fix scheduling)');
        // Mock data for testing
        setStaffData({
          nurses: [
            { id: '1', name: 'Sarah Johnson', specialty: 'ED', available: true },
            { id: '2', name: 'Mike Chen', specialty: 'ICU', available: true },
            { id: '3', name: 'Lisa Rodriguez', specialty: 'General', available: false }
          ],
          physicians: [
            { id: '1', name: 'Dr. Smith', specialty: 'Cardiology', available: true },
            { id: '2', name: 'Dr. Patel', specialty: 'Internal Medicine', available: true },
            { id: '3', name: 'Dr. Williams', specialty: 'Emergency', available: true }
          ],
          beds: [
            { id: 'ED-01', department: 'ED', status: 'available' },
            { id: 'ED-02', department: 'ED', status: 'available' },
            { id: 'ICU-01', department: 'ICU', status: 'occupied' },
            { id: 'General-05', department: 'General', status: 'available' }
          ]
        });
      } finally {
        setLoading(false);
      }
    };

    fetchStaffSchedule();
  }, [BACKEND_URL]);

  const handleNurseSelect = (nurse) => {
    setSelectedNurse(nurse);
    if (onStaffSelect.onNurseSelect) {
      onStaffSelect.onNurseSelect(nurse);
    }
  };

  const handlePhysicianSelect = (physician) => {
    setSelectedPhysician(physician);
    if (onStaffSelect.onPhysicianSelect) {
      onStaffSelect.onPhysicianSelect(physician);
    }
  };

  const handleBedSelect = (bed) => {
    setSelectedBed(bed);
    if (onStaffSelect.onBedSelect) {
      onStaffSelect.onBedSelect(bed);
    }
  };

  if (loading) {
    return <div style={containerStyle}><p>Loading staff schedule...</p></div>;
  }

  return (
    <div style={containerStyle}>
      {error && <div style={errorStyle}>⚠ {error}</div>}

      <h3 style={headingStyle}>Available Staff & Beds</h3>

      {/* Nurses Section */}
      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Assigned Nurse</h4>
        <div style={staffGridStyle}>
          {staffData.nurses && staffData.nurses.length > 0 ? (
            staffData.nurses.map((nurse) => (
              <div
                key={nurse.id}
                style={{
                  ...staffCardStyle,
                  ...(selectedNurse?.id === nurse.id ? selectedStyle : {}),
                  ...(nurse.available === false ? disabledStyle : {})
                }}
                onClick={() =>
                  nurse.available !== false && handleNurseSelect(nurse)
                }
                role="button"
                tabIndex={0}
              >
                <div style={staffNameStyle}>{nurse.name}</div>
                <div style={staffSpecialtyStyle}>{nurse.specialty}</div>
                <div style={availabilityBadgeStyle}>
                  {nurse.available ? '✓ Available' : '✗ Off Shift'}
                </div>
              </div>
            ))
          ) : (
            <p>No nurses available</p>
          )}
        </div>
      </div>

      {/* Physicians Section */}
      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Assigned Physician</h4>
        <div style={staffGridStyle}>
          {staffData.physicians && staffData.physicians.length > 0 ? (
            staffData.physicians.map((physician) => (
              <div
                key={physician.id}
                style={{
                  ...staffCardStyle,
                  ...(selectedPhysician?.id === physician.id ? selectedStyle : {}),
                  ...(physician.available === false ? disabledStyle : {})
                }}
                onClick={() =>
                  physician.available !== false && handlePhysicianSelect(physician)
                }
                role="button"
                tabIndex={0}
              >
                <div style={staffNameStyle}>{physician.name}</div>
                <div style={staffSpecialtyStyle}>{physician.specialty}</div>
                <div style={availabilityBadgeStyle}>
                  {physician.available ? '✓ Available' : '✗ Unavailable'}
                </div>
              </div>
            ))
          ) : (
            <p>No physicians available</p>
          )}
        </div>
      </div>

      {/* Beds Section */}
      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Bed Assignment</h4>
        <div style={staffGridStyle}>
          {staffData.beds && staffData.beds.length > 0 ? (
            staffData.beds.map((bed) => (
              <div
                key={bed.id}
                style={{
                  ...bedCardStyle,
                  ...(selectedBed?.id === bed.id ? selectedStyle : {}),
                  ...(bed.status !== 'available' ? disabledStyle : {})
                }}
                onClick={() =>
                  bed.status === 'available' && handleBedSelect(bed)
                }
                role="button"
                tabIndex={0}
              >
                <div style={bedIdStyle}>{bed.id}</div>
                <div style={bedDeptStyle}>{bed.department}</div>
                <div
                  style={{
                    ...bedStatusStyle,
                    ...(bed.status === 'available'
                      ? { backgroundColor: '#d1fae5', color: '#065f46' }
                      : { backgroundColor: '#fee2e2', color: '#7f1d1d' })
                  }}
                >
                  {bed.status === 'available' ? '✓ Available' : '⊘ Occupied'}
                </div>
              </div>
            ))
          ) : (
            <p>No beds available</p>
          )}
        </div>
      </div>

      {/* Selection Summary */}
      {(selectedNurse || selectedPhysician || selectedBed) && (
        <div style={summaryStyle}>
          <h4>Selection Summary</h4>
          {selectedNurse && (
            <p>
              <strong>Nurse:</strong> {selectedNurse.name} ({selectedNurse.specialty})
            </p>
          )}
          {selectedPhysician && (
            <p>
              <strong>Physician:</strong> {selectedPhysician.name} ({selectedPhysician.specialty})
            </p>
          )}
          {selectedBed && (
            <p>
              <strong>Bed:</strong> {selectedBed.id} ({selectedBed.department})
            </p>
          )}
        </div>
      )}
    </div>
  );
}

const containerStyle = {
  padding: '20px',
  backgroundColor: '#f8fafc',
  borderRadius: '8px',
  marginBottom: '20px'
};

const headingStyle = {
  marginTop: 0,
  marginBottom: '20px',
  fontSize: '18px',
  fontWeight: '600',
  color: '#1e293b'
};

const sectionStyle = {
  marginBottom: '25px'
};

const sectionTitleStyle = {
  marginTop: 0,
  marginBottom: '12px',
  fontSize: '14px',
  fontWeight: '600',
  color: '#475569',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const staffGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
  gap: '12px'
};

const staffCardStyle = {
  padding: '12px',
  backgroundColor: 'white',
  border: '2px solid #e2e8f0',
  borderRadius: '6px',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  position: 'relative'
};

const bedCardStyle = {
  padding: '12px',
  backgroundColor: 'white',
  border: '2px solid #e2e8f0',
  borderRadius: '6px',
  cursor: 'pointer',
  transition: 'all 0.2s ease'
};

const selectedStyle = {
  borderColor: '#2563eb',
  backgroundColor: '#eff6ff',
  boxShadow: '0 0 0 3px rgba(37, 99, 235, 0.1)'
};

const disabledStyle = {
  opacity: 0.5,
  cursor: 'not-allowed',
  backgroundColor: '#f5f5f5'
};

const staffNameStyle = {
  fontWeight: '600',
  color: '#1e293b',
  fontSize: '14px',
  marginBottom: '4px'
};

const staffSpecialtyStyle = {
  fontSize: '12px',
  color: '#64748b',
  marginBottom: '6px'
};

const availabilityBadgeStyle = {
  fontSize: '11px',
  padding: '4px 8px',
  borderRadius: '4px',
  backgroundColor: '#f1f5f9',
  color: '#475569',
  fontWeight: '500'
};

const bedIdStyle = {
  fontWeight: '700',
  color: '#1e293b',
  fontSize: '16px',
  marginBottom: '4px'
};

const bedDeptStyle = {
  fontSize: '12px',
  color: '#64748b',
  marginBottom: '6px'
};

const bedStatusStyle = {
  fontSize: '11px',
  padding: '4px 8px',
  borderRadius: '4px',
  fontWeight: '500'
};

const errorStyle = {
  padding: '12px',
  backgroundColor: '#fee2e2',
  color: '#991b1b',
  borderRadius: '4px',
  marginBottom: '12px',
  fontSize: '13px'
};

const summaryStyle = {
  marginTop: '20px',
  padding: '15px',
  backgroundColor: 'white',
  border: '1px solid #cbd5e1',
  borderRadius: '6px'
};

export default StaffScheduleDisplay;
