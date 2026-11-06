import React, { useEffect, useMemo, useState } from 'react';
import './StaffScheduleDisplay.css';
import { canAssignSelfAsDownstreamNurse, fetchCurrentAssignmentIdentity } from '../../utils/assignmentValidation';
import { decodeJwt } from '../../utils/jwt';

/**
 * StaffScheduleDisplay - Shows available nurses, physicians, specialists, and beds
 * Used in both Manual and AI-Assisted triage for staff selection
 */
function StaffScheduleDisplay({ onStaffSelect = {} }) {
  const [staffList, setStaffList] = useState([]);
  const [wards, setWards] = useState([]);
  const [wardBeds, setWardBeds] = useState({});
  const [expandedWards, setExpandedWards] = useState({});
  const [selectedNurse, setSelectedNurse] = useState(null);
  const [selectedPhysician, setSelectedPhysician] = useState(null);
  const [selectedSpecialist, setSelectedSpecialist] = useState(null);
  const [selectedBed, setSelectedBed] = useState(null);
  const [roleFilter, setRoleFilter] = useState('all');
  const [clinicianSearch, setClinicianSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [workloadById, setWorkloadById] = useState({});
  const [workloadError, setWorkloadError] = useState(null);
  const [assignmentIdentity, setAssignmentIdentity] = useState(null);
  const [assignmentIdentityError, setAssignmentIdentityError] = useState(null);
  const [assignmentWarning, setAssignmentWarning] = useState(null);

  const BACKEND_URL =
    import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

  const normalizeClinician = (item, fallbackRole = 'PHYSICIAN') => ({
    id: item._id || item.id || item.physicianId || item.clinicianId || item.staffId || `${fallbackRole}-${Math.random().toString(36).slice(2, 8)}`,
    name: item.name || item.displayName || item.fullName || 'Unknown',
    role: (item.role || item.profession || item.type || fallbackRole).toString().toUpperCase(),
    specialty: item.specialty || item.department || item.team || 'General',
    available: item.available === true || item.isAvailable === true || item.availabilityStatus === 'ON_SHIFT',
    availabilityLabel: item.availabilityStatus === 'ON_BREAK'
      ? 'On break · unavailable'
      : item.availabilityStatus === 'OFF_SHIFT' || item.available === false || item.isAvailable === false
        ? 'Off shift · unavailable'
        : item.available === true || item.isAvailable === true || item.availabilityStatus === 'ON_SHIFT'
          ? 'On shift'
          : 'Availability unverified',
    shiftStart: item.clock_in_time || item.clockInTime || item.shift_start || item.shiftStart || null,
    shiftEnd: item.clock_out_time || item.clockOutTime || item.shift_end || item.shiftEnd || null,
    score: typeof item.score === 'number' ? item.score : item.match ?? item.weight ?? 0
  });

  const normalizeWard = (ward) => {
    const apiId = ward._id || ward.id || ward.wardId || null;
    const id = apiId || ward.wardName || ward.name || `${ward.department || 'Ward'}-${Math.random().toString(36).slice(2, 6)}`;
    return {
      id,
      apiId,
      hospitalId: ward.hospitalId || ward.hospital_id || null,
      name: ward.wardName || ward.name || ward.department || 'General Ward',
      department: ward.department || ward.wardName || 'General',
      currentStatus: ward.currentStatus || {},
      totalBeds: typeof ward.totalBeds === 'number' ? ward.totalBeds : undefined,
      availableBeds: typeof ward.availableBeds === 'number' ? ward.availableBeds : ward.currentStatus?.available
    };
  };

  const normalizeBed = (bed, idx) => ({
    id: bed._id || bed.bedId || bed.bedNumber || `bed-${idx}`,
    bedNumber: bed.bedNumber || bed.bedId || bed.roomNumber || `Bed ${idx + 1}`,
    department: bed.department || bed.ward || 'General',
    wardName: bed.wardName || bed.ward || bed.department || 'General Ward',
    status: bed.status?.toLowerCase() || 'available',
    bedType: bed.bedType || 'Standard',
    equipment: [
      ...(Array.isArray(bed.equipment) ? bed.equipment : bed.equipment ? [bed.equipment] : []),
      ...(Array.isArray(bed.equipment_baseline) ? bed.equipment_baseline : []),
      bed.hasMonitor && 'Monitor',
      bed.hasVentilator && 'Ventilator',
      bed.hasDialysis && 'Dialysis Machine'
    ].filter(Boolean).filter((item, index, items) => items.indexOf(item) === index),
    equipment_baseline: bed.equipment_baseline || [],
    hasMonitor: Boolean(bed.hasMonitor),
    hasVentilator: Boolean(bed.hasVentilator),
    hasDialysis: Boolean(bed.hasDialysis),
    roomNumber: bed.roomNumber || bed.room || null,
    acuityLevel: bed.acuityLevel || 'General'
  });

  useEffect(() => {
    const fetchLiveData = async () => {
      try {
        setLoading(true);
        setError(null);
        setWorkloadError(null);
        setAssignmentIdentityError(null);
        const token = localStorage.getItem('token');
        const headers = token ? { Authorization: `Bearer ${token}` } : {};

        const identityRequest = fetchCurrentAssignmentIdentity(BACKEND_URL, token)
          .then(setAssignmentIdentity)
          .catch((identityError) => {
            setAssignmentIdentity({ userId: String(decodeJwt(token)?.id || ''), dutyRole: null });
            setAssignmentIdentityError(identityError.message || 'Shift duty could not be verified.');
          });
        const [staffRes, wardsRes, workloadRes] = await Promise.all([
          fetch(`${BACKEND_URL}/api/clinical/staff/clocked-in?includeRoster=true`, { headers }),
          fetch(`${BACKEND_URL}/api/beds/wards`, { headers }),
          fetch(`${BACKEND_URL}/api/operations/clinician-workload`, { headers }),
          identityRequest
        ]);

        if (!staffRes.ok) throw new Error(`Failed to fetch clinicians: ${staffRes.status}`);
        if (!wardsRes.ok) throw new Error(`Failed to fetch wards: ${wardsRes.status}`);

        const staffData = await staffRes.json();
        const wardData = await wardsRes.json();

        const clinicians = (staffData.clinicians || staffData.data?.clinicians || staffData.data || []).map((item) => normalizeClinician(item));
        const allWards = (wardData.wards || wardData.data?.wards || wardData.data || []).map(normalizeWard);
        let workloadData = null;
        if (workloadRes.ok) {
          try {
            workloadData = await workloadRes.json();
            if (!Array.isArray(workloadData?.clinicians)) throw new Error('Malformed clinician workload data.');
          } catch {
            workloadData = null;
            setWorkloadError('Workload policy response was invalid. Capacity is unavailable until it can be loaded.');
          }
        } else {
          setWorkloadError(`Workload policy is unavailable (request failed: ${workloadRes.status}).`);
        }
        const workloadMap = Object.fromEntries((workloadData?.clinicians || []).map((item) => [String(item.clinicianId || item.id), item]));

        setWorkloadById(workloadMap);
        setStaffList(clinicians.map((clinician) => ({
          ...clinician,
          activeCases: workloadMap[String(clinician.id)]?.activeCases ?? workloadMap[clinician.name]?.activeCases ?? 0,
          activeWorkload: workloadMap[String(clinician.id)]?.activeWorkload ?? workloadMap[clinician.name]?.activeWorkload ?? 0,
          workloadPercent: workloadMap[String(clinician.id)]?.workloadPercent ?? workloadMap[clinician.name]?.workloadPercent ?? null,
          workloadStatus: workloadMap[String(clinician.id)]?.workloadStatus || workloadMap[clinician.name]?.workloadStatus || (workloadData ? 'Unconfigured' : 'Unavailable'),
          capacitySource: workloadMap[String(clinician.id)]?.capacitySource || workloadMap[clinician.name]?.capacitySource || (workloadData ? 'Unconfigured' : 'Unavailable')
        })));
        setWards(allWards);
        setError(null);
      } catch (err) {
        console.error('Error fetching live staff/wards:', err);
        setError(err.message || 'Failed to fetch live staff or wards');
        setStaffList([]);
        setWards([]);
      } finally {
        setLoading(false);
      }
    };

    fetchLiveData();
  }, [BACKEND_URL]);

  const isValidObjectId = (value) => typeof value === 'string' && /^[0-9a-fA-F]{24}$/.test(value);

  const fetchBedsForWard = async (wardId, apiId) => {
    const targetWardId = apiId || (isValidObjectId(wardId) ? wardId : null);
    if (!targetWardId) {
      console.warn('Skipping bed fetch because no valid ward ID is available for', wardId, apiId);
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const res = await fetch(`${BACKEND_URL}/api/beds/ward/${encodeURIComponent(targetWardId)}`, { headers });
      if (!res.ok) {
        throw new Error(`Failed to fetch beds for ward ${targetWardId}: ${res.status}`);
      }
      const data = await res.json();
      const beds = (data.data?.beds || data.beds || data.data || []).map((bed, idx) => normalizeBed(bed, idx));
      setWardBeds((prev) => ({ ...prev, [wardId]: beds }));
    } catch (err) {
      console.error('Error fetching beds for ward:', err);
      setWardBeds((prev) => ({ ...prev, [wardId]: [] }));
    }
  };

  const handleWardToggle = (wardId, apiId, isOpen) => {
    setExpandedWards((prev) => {
      const nextExpanded = { ...prev, [wardId]: !!isOpen };
      if (isOpen && !wardBeds[wardId]) {
        fetchBedsForWard(wardId, apiId);
      }
      return nextExpanded;
    });
  };

  const handleNurseSelect = (nurse) => {
    if (!canAssignSelfAsDownstreamNurse(assignmentIdentity?.userId, nurse.id, assignmentIdentity?.dutyRole)) {
      setAssignmentWarning('Your shift assigns you as the triage nurse, so you cannot also be selected as the downstream clinical nurse.');
      return;
    }
    setAssignmentWarning(null);
    setSelectedNurse(nurse);
    if (onStaffSelect.onNurseSelect) onStaffSelect.onNurseSelect(nurse);
  };

  const handlePhysicianSelect = (physician) => {
    setSelectedPhysician(physician);
    if (onStaffSelect.onPhysicianSelect) onStaffSelect.onPhysicianSelect(physician);
  };

  const handleSpecialistSelect = (specialist) => {
    setSelectedSpecialist(specialist);
    if (onStaffSelect.onSpecialistSelect) onStaffSelect.onSpecialistSelect(specialist);
  };

  const handleBedSelect = (bed) => {
    setSelectedBed(bed);
    if (onStaffSelect.onBedSelect) onStaffSelect.onBedSelect(bed);
  };

  const filteredClinicians = useMemo(() => {
    const query = clinicianSearch.trim().toLowerCase();
    return staffList.filter((clinician) => {
      if (roleFilter === 'physician' && clinician.role !== 'PHYSICIAN') return false;
      if (roleFilter === 'nurse' && clinician.role !== 'NURSE') return false;
      if (roleFilter === 'specialist' && clinician.role !== 'SPECIALIST') return false;
      return !query || `${clinician.name} ${clinician.specialty}`.toLowerCase().includes(query);
    });
  }, [roleFilter, clinicianSearch, staffList]);

  const nurses = filteredClinicians.filter((c) => c.role === 'NURSE');
  const physicians = filteredClinicians.filter((c) => c.role === 'PHYSICIAN');
  const specialists = filteredClinicians.filter((c) => c.role === 'SPECIALIST');

  if (loading) {
    return <div style={containerStyle}><p>Loading staff schedule...</p></div>;
  }

  return (
    <div style={containerStyle}>
      {error && <div style={errorStyle}>⚠ {error}</div>}
      {workloadError && <div style={errorStyle} role="alert">⚠ {workloadError}</div>}
      {assignmentIdentityError && <div style={errorStyle} role="status">⚠ {assignmentIdentityError} Self-assignment is disabled unless your verified shift duty is Clinical Nurse.</div>}
      {assignmentWarning && <div style={errorStyle} role="alert">⚠ {assignmentWarning}</div>}

      <h3 style={headingStyle}>Available Clinicians & Beds</h3>

      <div className="staff-filter-bar">
        <select value={roleFilter} onChange={(e) => setRoleFilter(e.target.value)} aria-label="Filter clinicians by role">
          <option value="all">All clinicians</option>
          <option value="physician">Physicians</option>
          <option value="nurse">Nurses</option>
          <option value="specialist">Specialists</option>
        </select>
        <input value={clinicianSearch} onChange={(e) => setClinicianSearch(e.target.value)} placeholder="Search clinician or specialty" aria-label="Search clinicians" />
      </div>

      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Nurse Selection</h4>
        <div style={staffGridStyle}>
          {nurses.length > 0 ? nurses.map((nurse) => {
            const selfAssignmentBlocked = !canAssignSelfAsDownstreamNurse(
              assignmentIdentity?.userId,
              nurse.id,
              assignmentIdentity?.dutyRole
            );
            return (
            <div
              key={nurse.id}
              style={{
                ...staffCardStyle,
                ...(selectedNurse?.id === nurse.id ? selectedStyle : {}),
                ...(nurse.available === false || selfAssignmentBlocked ? disabledStyle : {})
              }}
              onClick={() => nurse.available !== false && handleNurseSelect(nurse)}
              role="button"
              tabIndex={0}
              aria-disabled={nurse.available === false || selfAssignmentBlocked}
              title={selfAssignmentBlocked ? 'Your current shift duty does not permit self-assignment as the downstream nurse.' : undefined}
            >
              <div style={staffNameStyle}>{nurse.name}</div>
              <div style={staffSpecialtyStyle}>{nurse.specialty}</div>
              <div title={workloadError || `Capacity source: ${nurse.capacitySource}`} style={workloadStyle}>
                {workloadError ? 'Clinician Workload: Unavailable' : `Clinician Workload: ${nurse.activeCases} cases · ${nurse.activeWorkload} load · ${nurse.workloadPercent === null ? 'Unconfigured' : `${nurse.workloadPercent}%`}`}
              </div>
              <div style={availabilityBadgeStyle}>{nurse.available ? '✓ On shift' : `✗ ${nurse.availabilityLabel}`}</div>
            </div>
          );}) : <p>No nurses in the active hospital roster.</p>}
        </div>
      </div>

      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Physician Selection</h4>
        <div style={staffGridStyle}>
          {physicians.length > 0 ? physicians.map((physician) => (
            <div
              key={physician.id}
              style={{
                ...staffCardStyle,
                ...(selectedPhysician?.id === physician.id ? selectedStyle : {}),
                ...(physician.available === false ? disabledStyle : {})
              }}
              onClick={() => physician.available !== false && handlePhysicianSelect(physician)}
              role="button"
              tabIndex={0}
              aria-disabled={physician.available === false}
            >
              <div style={staffNameStyle}>{physician.name}</div>
              <div style={staffSpecialtyStyle}>{physician.specialty}</div>
              <div title={workloadError || `Capacity source: ${physician.capacitySource}`} style={workloadStyle}>{workloadError ? 'Clinician Workload: Unavailable' : `Clinician Workload: ${physician.activeCases} cases · ${physician.activeWorkload} load · ${physician.workloadPercent === null ? 'Unconfigured' : `${physician.workloadPercent}%`}`}</div>
              <div style={availabilityBadgeStyle}>{physician.available ? '✓ On shift' : `✗ ${physician.availabilityLabel}`}</div>
            </div>
          )) : <p>No physicians in the active hospital roster.</p>}
        </div>
      </div>

      {specialists.length > 0 && (
        <div style={sectionStyle}>
          <h4 style={sectionTitleStyle}>Specialist Selection</h4>
          <div style={staffGridStyle}>
            {specialists.map((specialist) => (
              <div
                key={specialist.id}
                style={{
                  ...staffCardStyle,
                  ...(selectedSpecialist?.id === specialist.id ? selectedStyle : {}),
                  ...(specialist.available === false ? disabledStyle : {})
                }}
                onClick={() => specialist.available !== false && handleSpecialistSelect(specialist)}
                role="button"
                tabIndex={0}
                aria-disabled={specialist.available === false}
              >
                <div style={staffNameStyle}>{specialist.name}</div>
                <div style={staffSpecialtyStyle}>{specialist.specialty}</div>
                <div title={workloadError || `Capacity source: ${specialist.capacitySource}`} style={workloadStyle}>{workloadError ? 'Clinician Workload: Unavailable' : `Clinician Workload: ${specialist.activeCases} cases · ${specialist.activeWorkload} load · ${specialist.workloadPercent === null ? 'Unconfigured' : `${specialist.workloadPercent}%`}`}</div>
                <div style={availabilityBadgeStyle}>{specialist.available ? '✓ On shift' : `✗ ${specialist.availabilityLabel}`}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Bed Assignment</h4>
        <div className="ward-bed-groups">
          {wards.length > 0 ? wards.map((ward) => (
            <details
              key={ward.id}
              className="ward-bed-group"
              open={expandedWards[ward.id] || false}
              onToggle={(event) => handleWardToggle(ward.id, ward.apiId, event.currentTarget.open)}
            >
              <summary>
                {ward.name}
                <span>{(ward.availableBeds ?? wardBeds[ward.id]?.length ?? 0)} bed{(ward.availableBeds ?? wardBeds[ward.id]?.length ?? 0) === 1 ? '' : 's'}</span>
              </summary>
              {expandedWards[ward.id] && (
                <div style={staffGridStyle}>
                  {(wardBeds[ward.id] || []).length > 0 ? (
                    wardBeds[ward.id].map((bed) => (
                      <div
                        key={bed.id}
                        style={{
                          ...bedCardStyle,
                          ...(selectedBed?.id === bed.id ? selectedStyle : {}),
                          ...(bed.status !== 'available' ? disabledStyle : {})
                        }}
                        onClick={() => bed.status === 'available' && handleBedSelect(bed)}
                        role="button"
                        tabIndex={0}
                      >
                        <div style={bedIdStyle}>{bed.bedNumber}</div>
                        <div style={bedDeptStyle}>{bed.department}</div>
                        {bed.equipment.length > 0 && (
                          <div style={{ marginTop: 6, color: '#475569', fontSize: 11, lineHeight: 1.35 }}>
                            Equipment: {bed.equipment.join(', ')}
                          </div>
                        )}
                        <div style={{
                          ...bedStatusStyle,
                          ...(bed.status === 'available'
                            ? { backgroundColor: '#d1fae5', color: '#065f46' }
                            : { backgroundColor: '#fee2e2', color: '#7f1d1d' })
                        }}>
                          {bed.status === 'available' ? '✓ Available' : '⊘ Occupied'}
                        </div>
                      </div>
                    ))
                  ) : (
                    <p style={{ gridColumn: '1 / -1', margin: 0 }}>No beds loaded for this ward yet.</p>
                  )}
                </div>
              )}
            </details>
          )) : (
            <p>No wards available</p>
          )}
        </div>
      </div>

      {(selectedNurse || selectedPhysician || selectedSpecialist || selectedBed) && (
        <div style={summaryStyle}>
          <h4>Selection Summary</h4>
          {selectedNurse && <p><strong>Nurse:</strong> {selectedNurse.name} ({selectedNurse.specialty})</p>}
          {selectedPhysician && <p><strong>Physician:</strong> {selectedPhysician.name} ({selectedPhysician.specialty})</p>}
          {selectedSpecialist && <p><strong>Specialist:</strong> {selectedSpecialist.name} ({selectedSpecialist.specialty})</p>}
          {selectedBed && <p><strong>Bed:</strong> {selectedBed.bedNumber} ({selectedBed.department}){selectedBed.equipment?.length ? ` - ${selectedBed.equipment.join(', ')}` : ''}</p>}
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
  border: '2px solid #2563eb',
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

const workloadStyle = {
  fontSize: '11px',
  color: '#475569',
  marginBottom: '4px',
  fontWeight: '600'
};

const bedIdStyle = {
  fontWeight: '700',
  color: '#1e293b',
  marginBottom: '8px'
};

const bedDeptStyle = {
  fontSize: '12px',
  color: '#475569',
  marginBottom: '8px'
};

const bedStatusStyle = {
  padding: '6px 10px',
  borderRadius: '999px',
  fontSize: '11px',
  fontWeight: '700',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center'
};

const availabilityBadgeStyle = {
  marginTop: '8px',
  padding: '6px 10px',
  borderRadius: '999px',
  fontSize: '11px',
  fontWeight: '700',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: '#eff6ff',
  color: '#1d4ed8'
};

const summaryStyle = {
  padding: '16px',
  borderRadius: '12px',
  backgroundColor: '#ffffff',
  border: '1px solid #e2e8f0'
};

const errorStyle = {
  padding: '12px 14px',
  marginBottom: '18px',
  borderRadius: '10px',
  backgroundColor: '#fef2f2',
  color: '#991b1b',
  border: '1px solid #fecaca'
};

export default StaffScheduleDisplay;
