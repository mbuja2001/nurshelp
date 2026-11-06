import React, { useState, useEffect } from 'react';
import { FiX, FiEdit2, FiClock, FiCheckCircle, FiAlertCircle, FiBarChart2, FiUser } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { decodeJwt } from '../../utils/jwt';

const Profile = ({ nurseName: initialNurseName }) => {
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const decodedToken = decodeJwt(localStorage.getItem('token')) || {};
  const accountRole = String(decodedToken.role || '').toUpperCase();
  const isManagementAccount = ['ADMIN', 'ADMINISTRATIVE', 'SUPERUSER', 'MANAGER', 'SHIFT_MANAGER', 'DEPARTMENT_MANAGER'].includes(accountRole);
  const isStaffShiftRole = ['NURSE', 'PHYSICIAN'].includes(accountRole);

  const getAuthenticatedClinicianId = () => {
    const decoded = decodeJwt(localStorage.getItem('token')) || {};
    return String(decoded.id || decoded._id || decoded.userId || decoded.staffId || decoded.clinicianId || '');
  };

  // ============ STATE ============
  const [nurse, setNurse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showShiftModal, setShowShiftModal] = useState(false);
  const [editForm, setEditForm] = useState({
    firstName: '',
    lastName: '',
    idNumber: '',
    employeeId: '',
  });
  const [shiftForm, setShiftForm] = useState({
    expectedHours: 8,
    shiftSignIn: '',
    shiftSignOut: '',
  });
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [liveWorkload, setLiveWorkload] = useState(null);
  const [liveWorkloadLoading, setLiveWorkloadLoading] = useState(false);

  // ============ FETCH STAFF PROFILE ============
  const fetchNurseProfile = async () => {
    if (isManagementAccount) {
      setNurse({
        firstName: decodedToken.firstName || decodedToken.name || '',
        lastName: decodedToken.lastName || decodedToken.surname || '',
        email: decodedToken.email || '',
        role: accountRole,
        specialty: decodedToken.specialty || decodedToken.clinicalSpecialty || '',
        department: decodedToken.department || '',
      });
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/staff/profile`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` },
      });
      if (!response.ok) {
        const json = await response.json().catch(() => ({}));
        throw new Error(json.message || 'Failed to load staff profile');
      }

      const data = await response.json();
      setNurse(data);
      setEditForm({
        firstName: data.firstName || '',
        lastName: data.lastName || '',
        idNumber: data.idNumber || '',
        employeeId: data.employeeId || '',
      });
      setShiftForm({
        expectedHours: data.shift?.expectedHours || 8,
        shiftSignIn: data.shift?.signIn || '',
        shiftSignOut: data.shift?.signOut || '',
      });
    } catch (error) {
      console.error('Error fetching staff profile:', error);
      setErrorMessage(error.message || 'Unable to load profile.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNurseProfile();
    if (!isManagementAccount) fetchLiveWorkload();
  }, []);

  const fetchLiveWorkload = async () => {
    setLiveWorkloadLoading(true);
    console.info('[PROFILE WORKLOAD] GET', `${BACKEND_URL}/api/operations/clinician-workload`);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/clinician-workload`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      console.info('[PROFILE WORKLOAD] response', {
        status: response.status,
        success: data.success,
        clinicians: data.clinicians?.length || 0,
        acuityWeights: data.acuityWeightByPriority || {},
        current: data.clinicians?.find((entry) => String(entry.clinicianId || entry.id || entry._id) === getAuthenticatedClinicianId()) || null
      });
      if (!response.ok) throw new Error(data.message || 'Unable to load live workload');
      const clinicianId = getAuthenticatedClinicianId();
      const current = (data.clinicians || []).find((entry) => String(entry.clinicianId || entry.id || entry._id) === clinicianId);
      console.info('[PROFILE WORKLOAD] decoded clinician match', {
        clinicianId,
        matched: Boolean(current),
        matchedRole: current?.role || null,
        capacity: current?.capacity ?? null,
        capacitySource: current?.capacitySource || null
      });
      setLiveWorkload(current || { clinicianId, activeCases: 0, activeWorkload: 0, workloadPercent: 0, workloadStatus: 'No active cases', capacity: null, capacitySource: 'Not required', acuityByPriority: {}, acuityWeights: {}, acuityWeightSource: 'fallback 1.0x' });
      if (current) {
        setLiveWorkload({
          ...current,
          acuityWeights: data.acuityWeightByPriority || {},
          acuityWeightSource: data.acuityWeightSource || 'fallback 1.0x'
        });
      }
    } catch (error) {
      console.error('Error fetching live workload:', error);
      setLiveWorkload({ unavailable: true });
    } finally {
      setLiveWorkloadLoading(false);
    }
  };

  // ============ MODAL HANDLERS ============
  const openEditModal = () => {
    setEditForm({
      firstName: nurse.firstName || '',
      lastName: nurse.lastName || '',
      idNumber: nurse.idNumber || '',
      employeeId: nurse.employeeId || '',
    });
    setShowEditModal(true);
  };

  const closeEditModal = () => {
    setShowEditModal(false);
    setEditForm({
      firstName: '',
      lastName: '',
      idNumber: '',
      employeeId: '',
    });
  };

  const openShiftModal = () => {
    setShiftForm({
      expectedHours: nurse.shift?.expectedHours || 8,
      shiftSignIn: nurse.shift?.signIn || '',
      shiftSignOut: nurse.shift?.signOut || '',
    });
    setShowShiftModal(true);
  };

  const closeShiftModal = () => {
    setShowShiftModal(false);
    setShiftForm({
      expectedHours: 8,
      shiftSignIn: '',
      shiftSignOut: '',
    });
  };

  // ============ SAVE PROFILE ============
  const saveProfile = async () => {
    if (!editForm.firstName.trim() || !editForm.lastName.trim() || !editForm.idNumber.trim() || !editForm.employeeId.trim()) {
      setErrorMessage('All fields are required');
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/staff/profile`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(editForm),
      });

      if (!response.ok) {
        const json = await response.json().catch(() => ({}));
        throw new Error(json.message || 'Unable to update profile');
      }

      const updatedNurse = await response.json();
      setNurse(updatedNurse);
      setSuccessMessage('Profile updated successfully');
      setTimeout(() => setSuccessMessage(''), 3000);
      closeEditModal();
    } catch (error) {
      console.error('Error updating profile:', error);
      setErrorMessage(error.message || 'Unable to update profile');
    }
  };

  // ============ SAVE SHIFT ============
  const saveShift = async () => {
    if (!shiftForm.shiftSignIn) {
      setErrorMessage('Sign-in time is required');
      return;
    }
    if (shiftForm.shiftSignOut && new Date(shiftForm.shiftSignOut) < new Date(shiftForm.shiftSignIn)) {
      setErrorMessage('Sign-out time must be after sign-in time');
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/staff/shift`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({
          expectedHours: parseInt(shiftForm.expectedHours),
          signIn: shiftForm.shiftSignIn,
          signOut: shiftForm.shiftSignOut,
        }),
      });

      if (!response.ok) {
        const json = await response.json().catch(() => ({}));
        throw new Error(json.message || 'Unable to update shift');
      }

      const updatedShift = await response.json();
      setNurse({
        ...nurse,
        shift: updatedShift,
      });
      setSuccessMessage('Shift updated successfully');
      setTimeout(() => setSuccessMessage(''), 3000);
      closeShiftModal();
    } catch (error) {
      console.error('Error updating shift:', error);
      setErrorMessage(error.message || 'Unable to update shift');
    }
  };

  // ============ UTILITY FUNCTIONS ============
  const formatDate = (dateString) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const calculateWorkedHours = () => {
    if (!nurse.shift?.signIn) return 0;
    const signIn = new Date(nurse.shift.signIn);
    const signOut = nurse.shift.signOut ? new Date(nurse.shift.signOut) : new Date();
    const diffMs = signOut - signIn;
    const diffHours = diffMs / (1000 * 60 * 60);
    return Math.max(0, parseFloat(diffHours.toFixed(2)));
  };

  const workedHours = nurse ? calculateWorkedHours() : 0;
  const shiftProgress = nurse ? (workedHours / (nurse.shift?.expectedHours || 8)) * 100 : 0;
  const isShiftComplete = nurse?.shift?.signOut ? true : false;

  // ============ LOADING STATE ============
  if (loading) {
    return (
      <div style={{ padding: 24, textAlign: 'center' }}>
        <p style={{ color: '#6b7280' }}>Loading profile...</p>
      </div>
    );
  }

  if (!nurse) {
    return (
      <div style={{ padding: 24, textAlign: 'center' }}>
        <p style={{ color: '#ef4444' }}>Failed to load profile</p>
      </div>
    );
  }

  // ============ RENDER ============
  const username = nurse?.firstName && nurse?.lastName 
    ? `${nurse.firstName} ${nurse.lastName}`
    : initialNurseName || nurse?.firstName || 'Staff';
  const roleLabel = String(nurse.role || accountRole || 'Staff').replace(/_/g, ' ');
  const activeCases = Number(liveWorkload?.activeCases || 0);
  const hasActiveCases = activeCases > 0;
  const resolvedCapacityLabel = liveWorkload?.capacity ?? (hasActiveCases ? 'Unconfigured' : 'No active cases');
  const utilizationLabel = liveWorkload?.workloadPercent != null
    ? `${liveWorkload.workloadPercent}%`
    : hasActiveCases ? 'Unconfigured' : '0%';
  const capacitySourceLabel = hasActiveCases
    ? (liveWorkload?.capacitySource || 'Unconfigured')
    : 'Not required';
  const acuityMixLabel = Object.entries(liveWorkload?.acuityByPriority || {})
    .map(([key, count]) => `${count} ${key} (${Number(liveWorkload?.acuityWeights?.[key] ?? 1).toFixed(1)}x)`)
    .join(' · ') || 'No active cases';
  const acuityWeightSourceLabel = liveWorkload?.acuityWeightSource || 'fallback 1.0x';
  const workloadHasCapacity = liveWorkload?.capacity != null;

  return (
    <SidebarLayout username={username} handleLogout={() => {}}>
      <div style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 32 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
            <FiUser size={32} style={{ marginRight: 12, verticalAlign: 'middle', color: '#0369a1' }} />
            Profile
          </h1>
          <p style={{ margin: '8px 0 0 0', fontSize: 14, color: '#6b7280' }}>
            {[roleLabel, nurse.specialty, nurse.department].filter(Boolean).join(' · ')}
          </p>
        </div>
        {!isManagementAccount && <button
          onClick={openEditModal}
          style={{
            padding: '10px 16px',
            backgroundColor: '#0f172a',
            color: 'white',
            border: 'none',
            borderRadius: 8,
            cursor: 'pointer',
            fontSize: 14,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <FiEdit2 size={16} />
          Edit Profile
        </button>}
      </div>

      {/* Success/Error Messages */}
      {successMessage && (
        <div style={{
          padding: 12,
          marginBottom: 16,
          backgroundColor: '#dcfce7',
          color: '#166534',
          borderRadius: 8,
          fontSize: 14,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span><FiCheckCircle size={16} style={{ marginRight: 8, display: 'inline' }} />{successMessage}</span>
          <button onClick={() => setSuccessMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534' }}>
            <FiX size={18} />
          </button>
        </div>
      )}
      {errorMessage && (
        <div style={{
          padding: 12,
          marginBottom: 16,
          backgroundColor: '#fee2e2',
          color: '#991b1b',
          borderRadius: 8,
          fontSize: 14,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span><FiAlertCircle size={16} style={{ marginRight: 8, display: 'inline' }} />{errorMessage}</span>
          <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b' }}>
            <FiX size={18} />
          </button>
        </div>
      )}

      {!isManagementAccount && <div style={{ marginBottom: 24, padding: 24, background: 'linear-gradient(135deg, #ecfeff 0%, #f8fafc 100%)', borderRadius: 12, border: '1px solid #bae6fd', boxShadow: '0 1px 3px rgba(15,23,42,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, marginBottom: 18 }}>
          <div>
            <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8, fontSize: 17, color: '#0f172a' }}><FiBarChart2 color="#0369a1" /> My Live Workload</h3>
            <p style={{ margin: '6px 0 0', color: '#475569', fontSize: 13, maxWidth: 680 }}>Live assignment load for your current clinical lane. Workload units come from assigned cases and acuity; utilization is only calculated when a capacity policy exists.</p>
          </div>
        </div>
        {liveWorkloadLoading ? <div style={{ color: '#64748b', fontSize: 13 }}>Calculating live workload...</div> : liveWorkload?.unavailable ? (
          <div style={{ color: '#9a3412', fontSize: 13 }}>Live workload is temporarily unavailable.</div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(155px, 1fr))', gap: 10 }}>
              <div style={{ ...workloadMetricStyle, background: '#ffffff', border: '1px solid #dbeafe' }}><span>Assigned patients</span><strong style={{ fontSize: 25 }}>{liveWorkload?.activeCases ?? 0}</strong><small>Current active cases</small></div>
              <div style={{ ...workloadMetricStyle, background: '#ffffff', border: '1px solid #dbeafe' }}><span>Weighted workload</span><strong style={{ fontSize: 25 }}>{Number(liveWorkload?.activeWorkload ?? 0).toFixed(1)} <small style={{ fontSize: 12 }}>units</small></strong><small>Acute load after acuity weights</small></div>
              <div style={{ ...workloadMetricStyle, background: workloadHasCapacity ? '#ffffff' : '#fff7ed', border: `1px solid ${workloadHasCapacity ? '#dbeafe' : '#fed7aa'}` }}><span>Resolved capacity</span><strong style={{ fontSize: 20 }}>{resolvedCapacityLabel}</strong><small>{workloadHasCapacity ? 'Maximum workload units' : 'No policy baseline returned'}</small></div>
              <div style={{ ...workloadMetricStyle, background: workloadHasCapacity ? '#ffffff' : '#fff7ed', border: `1px solid ${workloadHasCapacity ? '#dbeafe' : '#fed7aa'}` }}><span>Utilization</span><strong style={{ fontSize: 25 }}>{utilizationLabel}</strong><small>{workloadHasCapacity ? 'Workload / capacity' : 'Cannot calculate without capacity'}</small></div>
            </div>
            <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 10 }}>
              <div style={{ padding: '12px 14px', background: '#ffffff', border: '1px solid #dbeafe', borderRadius: 10 }}>
                <div style={{ color: '#64748b', fontSize: 11, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Acuity mix</div>
                <div style={{ marginTop: 5, color: '#0f172a', fontWeight: 750, fontSize: 13 }}>{acuityMixLabel}</div>
              </div>
              <div style={{ padding: '12px 14px', background: '#ffffff', border: '1px solid #dbeafe', borderRadius: 10 }}>
                <div style={{ color: '#64748b', fontSize: 11, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Capacity source</div>
                <div style={{ marginTop: 5, color: workloadHasCapacity ? '#166534' : '#9a3412', fontWeight: 750, fontSize: 13 }}>{capacitySourceLabel}</div>
              </div>
              <div style={{ padding: '12px 14px', background: '#ffffff', border: '1px solid #dbeafe', borderRadius: 10 }}>
                <div style={{ color: '#64748b', fontSize: 11, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Acuity weight source</div>
                <div style={{ marginTop: 5, color: acuityWeightSourceLabel.startsWith('fallback') ? '#9a3412' : '#166534', fontWeight: 750, fontSize: 13 }}>{acuityWeightSourceLabel}</div>
              </div>
            </div>
          </>
        )}
      </div>}

      {/* Main Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 24 }}>
        {/* Personal Information Card */}
        <div style={{
          padding: 24,
          backgroundColor: 'white',
          borderRadius: 12,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e5e7eb',
        }}>
          <h3 style={{ margin: '0 0 20px 0', fontSize: 16, fontWeight: 700, color: '#0f172a' }}>Personal Information</h3>
          <div style={{ display: 'grid', gap: 16 }}>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>First Name</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{nurse.firstName}</p>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Last Name</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{nurse.lastName}</p>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>ID Number</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{nurse.idNumber}</p>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Employee ID</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{nurse.employeeId || '-'}</p>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Email</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{nurse.email || '-'}</p>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Role</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{roleLabel}</p>
            </div>
            {nurse.specialty && <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Specialty</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{nurse.specialty}</p>
            </div>}
          </div>
        </div>

        {/* Shift Information Card */}
        {!isManagementAccount && isStaffShiftRole && <div style={{
          padding: 24,
          backgroundColor: 'white',
          borderRadius: 12,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e5e7eb',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#0f172a' }}>Shift Details</h3>
            <button
              onClick={openShiftModal}
              style={{
                padding: '6px 12px',
                backgroundColor: '#0369a1',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: 600,
              }}
            >
              Update
            </button>
          </div>
          <div style={{ display: 'grid', gap: 16 }}>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Sign In</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>
                <FiClock size={14} style={{ marginRight: 8, display: 'inline' }} />
                {formatDate(nurse.shift?.signIn) || '-'}
              </p>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Sign Out</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>
                {nurse.shift?.signOut ? formatDate(nurse.shift.signOut) : <span style={{ color: '#f59e0b' }}>Shift In Progress</span>}
              </p>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Expected Hours</label>
              <p style={{ margin: 0, fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{nurse.shift?.expectedHours} hours</p>
            </div>
          </div>
        </div>}

        {/* Worked Hours Card (Immutable) */}
        {!isManagementAccount && isStaffShiftRole && <div style={{
          padding: 24,
          backgroundColor: 'white',
          borderRadius: 12,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e5e7eb',
        }}>
          <h3 style={{ margin: '0 0 20px 0', fontSize: 16, fontWeight: 700, color: '#0f172a' }}>
            <FiBarChart2 size={16} style={{ marginRight: 8, display: 'inline' }} />
            Worked Hours (Calculated)
          </h3>
          <div style={{ display: 'grid', gap: 16 }}>
            <div>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 8 }}>Hours Logged</label>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{workedHours.toFixed(2)}</div>
              <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#6b7280' }}>of {nurse.shift?.expectedHours} expected hours</p>
            </div>
            
            {/* Progress Bar */}
            <div>
              <div style={{
                width: '100%',
                height: 12,
                backgroundColor: '#e5e7eb',
                borderRadius: 6,
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${Math.min(shiftProgress, 100)}%`,
                  height: '100%',
                  backgroundColor: shiftProgress >= 100 ? '#10b981' : '#0369a1',
                  transition: 'width 0.3s ease',
                }} />
              </div>
              <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#6b7280' }}>
                {shiftProgress.toFixed(0)}% complete
              </p>
            </div>

            {/* Status Badge */}
            <div>
              <span style={{
                display: 'inline-block',
                padding: '6px 12px',
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 600,
                backgroundColor: isShiftComplete ? '#dcfce7' : '#fef3c7',
                color: isShiftComplete ? '#166534' : '#92400e',
              }}>
                {isShiftComplete ? 'Shift Complete' : 'Shift In Progress'}
              </span>
            </div>
        </div>
        </div>}
      </div>

      {/* ============ EDIT PROFILE MODAL ============ */}
      {showEditModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: 12,
            padding: 32,
            maxWidth: 500,
            width: '100%',
            boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
              <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>Edit Profile</h2>
              <button onClick={closeEditModal} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>
                <FiX size={24} color="#6b7280" />
              </button>
            </div>

            <div style={{ display: 'grid', gap: 16, marginBottom: 24 }}>
              <div>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#0f172a', marginBottom: 6 }}>First Name</label>
                <input
                  type="text"
                  value={editForm.firstName}
                  onChange={(e) => setEditForm({ ...editForm, firstName: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: 8,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#0f172a', marginBottom: 6 }}>Last Name</label>
                <input
                  type="text"
                  value={editForm.lastName}
                  onChange={(e) => setEditForm({ ...editForm, lastName: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: 8,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#0f172a', marginBottom: 6 }}>ID Number</label>
                <input
                  type="text"
                  value={editForm.idNumber}
                  onChange={(e) => setEditForm({ ...editForm, idNumber: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: 8,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#0f172a', marginBottom: 6 }}>Employee ID</label>
                <input
                  type="text"
                  value={editForm.employeeId}
                  onChange={(e) => setEditForm({ ...editForm, employeeId: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: 8,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <button
                onClick={closeEditModal}
                style={{
                  padding: '10px 16px',
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                  border: 'none',
                  borderRadius: 8,
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: 600,
                }}
              >
                Cancel
              </button>
              <button
                onClick={saveProfile}
                style={{
                  padding: '10px 16px',
                  backgroundColor: '#10b981',
                  color: 'white',
                  border: 'none',
                  borderRadius: 8,
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: 600,
                }}
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ============ UPDATE SHIFT MODAL ============ */}
      {showShiftModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: 12,
            padding: 32,
            maxWidth: 500,
            width: '100%',
            boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
              <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>Update Shift</h2>
              <button onClick={closeShiftModal} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>
                <FiX size={24} color="#6b7280" />
              </button>
            </div>

            <div style={{ display: 'grid', gap: 16, marginBottom: 24 }}>
              <div>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#0f172a', marginBottom: 6 }}>Sign In Time</label>
                <input
                  type="datetime-local"
                  value={shiftForm.shiftSignIn}
                  onChange={(e) => setShiftForm({ ...shiftForm, shiftSignIn: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: 8,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#0f172a', marginBottom: 6 }}>Sign Out Time (Optional)</label>
                <input
                  type="datetime-local"
                  value={shiftForm.shiftSignOut}
                  onChange={(e) => setShiftForm({ ...shiftForm, shiftSignOut: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: 8,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#0f172a', marginBottom: 6 }}>Expected Hours</label>
                <input
                  type="number"
                  min="1"
                  max="24"
                  value={shiftForm.expectedHours}
                  onChange={(e) => setShiftForm({ ...shiftForm, expectedHours: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: 8,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <p style={{ margin: '16px 0 0 0', padding: 12, backgroundColor: '#f0f9ff', borderRadius: 8, fontSize: 12, color: '#0c4a6e' }}>
                💡 <strong>Note:</strong> Worked hours are automatically calculated from sign-in and sign-out times. This is immutable and based on actual logged times.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <button
                onClick={closeShiftModal}
                style={{
                  padding: '10px 16px',
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                  border: 'none',
                  borderRadius: 8,
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: 600,
                }}
              >
                Cancel
              </button>
              <button
                onClick={saveShift}
                style={{
                  padding: '10px 16px',
                  backgroundColor: '#0369a1',
                  color: 'white',
                  border: 'none',
                  borderRadius: 8,
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: 600,
                }}
              >
                Save Shift
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
    </SidebarLayout>
  );
};

const workloadMetricStyle = {
  display: 'grid',
  gap: 6,
  minWidth: 0,
  padding: '12px 13px',
  border: '1px solid #bae6fd',
  borderRadius: 9,
  background: 'rgba(255,255,255,0.72)',
  color: '#475569',
  fontSize: 11,
  fontWeight: 700
};

export default Profile;
