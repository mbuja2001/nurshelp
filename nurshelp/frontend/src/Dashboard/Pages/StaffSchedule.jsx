import React, { useState, useEffect } from 'react';
import { FiX, FiPlus, FiDownload, FiUpload, FiEdit2, FiTrash2, FiChevronDown } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';

const StaffSchedule = () => {
  const { userName } = useUserFromToken();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');
  const userStr = localStorage.getItem('user');
  
  // Safely parse user data with error handling
  let user = {};
  try {
    user = userStr && userStr.trim() ? JSON.parse(userStr) : {};
  } catch (err) {
    console.error('Failed to parse user data from localStorage:', err);
    user = {};
  }

  // ============ MAIN TAB STATE ============
  const [mainTab, setMainTab] = useState('schedule'); // 'schedule', 'clock-in-out', 'history', 'statistics'
  
  // ============ SCHEDULE TAB STATE ============
  const [scheduleTab, setScheduleTab] = useState('physicians'); // 'my-schedule', 'physicians', 'nurses'
  const [targetDate, setTargetDate] = useState(new Date().toISOString().split('T')[0]); // YYYY-MM-DD
  
  const [physicians, setPhysicians] = useState([]);
  const [nurses, setNurses] = useState([]);
  const [mySchedule, setMySchedule] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState({ specialty: 'all', status: 'all' });
  
  const [showModal, setShowModal] = useState(false);
  const [modalMode, setModalMode] = useState('view'); // 'view', 'edit', 'add'
  const [modalType, setModalType] = useState('physician'); // 'physician', 'nurse'
  const [selectedStaff, setSelectedStaff] = useState(null);
  
  const [editForm, setEditForm] = useState({
    physicianId: '',
    nurseId: '',
    name: '',
    surname: '',
    specialty: '',
    status: 'ON_DUTY',
    shiftStart: '08:00',
    shiftEnd: '16:00',
    maxPatients: 20,
    maxBeds: 5,
    esiLevel: 3,
    certifications: '',
    notes: ''
  });
  
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [csvUploadMode, setCsvUploadMode] = useState(null); // 'physicians' or 'nurses'

  // ============ CLOCK IN/OUT STATE ============
  const [currentRecord, setCurrentRecord] = useState(null);
  const [liveTime, setLiveTime] = useState(new Date());
  const [onBreak, setOnBreak] = useState(false);
  const [breakStartTime, setBreakStartTime] = useState(null);
  const [clockLoading, setClockLoading] = useState(false);
  const [showShiftSelection, setShowShiftSelection] = useState(false);
  const [availableShifts, setAvailableShifts] = useState([]);
  const [selectedShift, setSelectedShift] = useState(null);
  const [shiftsLoading, setShiftsLoading] = useState(false);

  // ============ CLOCK HISTORY STATE ============
  const [clockHistory, setClockHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // ============ STATISTICS STATE ============
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);

  const SPECIALTIES = ['Cardiology', 'Pulmonology', 'Neurology', 'General Medicine', 'Emergency Medicine', 'Trauma Surgery', 'Orthopedics', 'Psychiatry', 'Internal Medicine'];
  const STATUSES = ['ON_DUTY', 'ON_CALL', 'OFF', 'VACATION'];
  const ESI_LEVELS = [1, 2, 3];
  const CERTIFICATIONS_LIST = ['ACLS', 'PALS', 'BLS', 'TNCC', 'ENPC'];

  // Initialize history dates
  useEffect(() => {
    if (token) {
      const end = new Date();
      const start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
      setStartDate(start.toISOString().split('T')[0]);
      setEndDate(end.toISOString().split('T')[0]);
    }
  }, [token]);

  // Live clock
  useEffect(() => {
    const interval = setInterval(() => setLiveTime(new Date()), 500);
    return () => clearInterval(interval);
  }, []);

  // Load data based on active tab
  useEffect(() => {
    if (!token) return;

    if (mainTab === 'clock-in-out') {
      fetchTodayRecord();
    } else if (mainTab === 'history') {
      fetchClockHistory(startDate, endDate);
    } else if (mainTab === 'statistics') {
      fetchStatistics();
    } else if (mainTab === 'schedule') {
      if (scheduleTab === 'my-schedule') {
        fetchMySchedule(targetDate);
      } else if (scheduleTab === 'physicians') {
        fetchPhysicians(targetDate);
      } else {
        fetchNurses(targetDate);
      }
    }
  }, [mainTab, scheduleTab, token]);

  // ============ CLOCK IN/OUT METHODS ============
  const fetchTodayRecord = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/today-record`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setCurrentRecord(data.record);
        setOnBreak(data.record?.onBreak || false);
      }
    } catch (err) {
      console.error('Error fetching today record:', err);
    }
  };

  const fetchAvailableShifts = async () => {
    setShiftsLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/available-shifts`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setAvailableShifts(data.shifts || []);
        if (data.shifts && data.shifts.length === 1) {
          setSelectedShift(data.shifts[0]);
        }
      } else {
        setErrorMessage('Failed to fetch shifts');
      }
    } catch (err) {
      setErrorMessage('Error loading shifts: ' + err.message);
    } finally {
      setShiftsLoading(false);
    }
  };

  const handleClockIn = async () => {
    // Show shift selection modal if available shifts exist
    if (!currentRecord?.clockInTime) {
      await fetchAvailableShifts();
      setShowShiftSelection(true);
      return;
    }
  };

  const handleConfirmClockIn = async () => {
    if (!selectedShift) {
      setErrorMessage('Please select a shift');
      return;
    }

    setClockLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/clock-in`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          location: user?.location || 'ED',
          shiftId: selectedShift.id,
          shiftStart: selectedShift.shiftStart,
          shiftEnd: selectedShift.shiftEnd
        })
      });
      const data = await res.json();
      if (data.success) {
        setCurrentRecord(data.record);
        setShowShiftSelection(false);
        setSelectedShift(null);
        setSuccessMessage('✓ Clocked in successfully');
        setTimeout(() => setSuccessMessage(''), 3000);
      } else {
        setErrorMessage(data.message || 'Clock-in failed');
      }
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setClockLoading(false);
    }
  };

  const handleClockOut = async () => {
    setClockLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/clock-out`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      const data = await res.json();
      if (data.success) {
        setCurrentRecord(data.record);
        setOnBreak(false);
        setSuccessMessage('✓ Clocked out successfully');
        setTimeout(() => setSuccessMessage(''), 3000);
      } else {
        setErrorMessage(data.message || 'Clock-out failed');
      }
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setClockLoading(false);
    }
  };

  const handleBreakStart = async () => {
    setClockLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/break-start`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      const data = await res.json();
      if (data.success) {
        setOnBreak(true);
        setBreakStartTime(new Date());
        setSuccessMessage('✓ Break started');
        setTimeout(() => setSuccessMessage(''), 3000);
      } else {
        setErrorMessage(data.message || 'Break start failed');
      }
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setClockLoading(false);
    }
  };

  const handleBreakEnd = async () => {
    setClockLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/break-end`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      const data = await res.json();
      if (data.success) {
        setOnBreak(false);
        setBreakStartTime(null);
        setSuccessMessage('✓ Break ended');
        setTimeout(() => setSuccessMessage(''), 3000);
      } else {
        setErrorMessage(data.message || 'Break end failed');
      }
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setClockLoading(false);
    }
  };

  // ============ CLOCK HISTORY METHODS ============
  const fetchClockHistory = async (start, end) => {
    setHistoryLoading(true);
    try {
      let url = `${BACKEND_URL}/api/nurses/clockhistory?`;
      if (start) url += `startDate=${start}&`;
      if (end) url += `endDate=${end}`;

      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setClockHistory(data.records || []);
      } else {
        setErrorMessage(data.message);
      }
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleApplyFilter = () => {
    if (!startDate || !endDate) {
      setErrorMessage('Please select both dates');
      return;
    }
    fetchClockHistory(startDate, endDate);
  };

  const handleDownloadCSV = () => {
    if (clockHistory.length === 0) {
      setErrorMessage('No records to download');
      return;
    }

    const headers = ['Date', 'Clock In', 'Clock Out', 'Break Duration (min)', 'Total Work Hours', 'Location', 'Late Arrival (min)', 'Late Approval', 'Break Approval', 'Notes'];
    const rows = clockHistory.map(record => [
      new Date(record.date).toLocaleDateString(),
      record.clockInTime ? new Date(record.clockInTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--',
      record.clockOutTime ? new Date(record.clockOutTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--',
      record.totalBreakTime ? Math.floor(record.totalBreakTime / 1000 / 60) : 0,
      record.totalWorkTime ? (record.totalWorkTime / 1000 / 60 / 60).toFixed(2) : '--',
      record.location || 'ED',
      record.lateArrivalMinutes || 0,
      record.lateArrivalApprovalStatus || 'N/A',
      record.breakApprovalStatus || 'N/A',
      record.notes || ''
    ]);

    const csv = [headers, ...rows].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clock-history-${startDate}-to-${endDate}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    setSuccessMessage('✓ CSV exported successfully');
    setTimeout(() => setSuccessMessage(''), 3000);
  };

  // ============ STATISTICS METHOD ============
  const fetchStatistics = async () => {
    setStatsLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/statistics?range=monthly`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setStats(data.stats);
      } else {
        setErrorMessage(data.message);
      }
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setStatsLoading(false);
    }
  };

  // ============ FETCH PHYSICIANS ============
  const fetchPhysicians = async (date = targetDate) => {
    setLoading(true);
    try {
      const url = new URL(`${BACKEND_URL}/api/schedules/physicians`);
      url.searchParams.append('date', date);
      
      const res = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!res.ok) throw new Error('Failed to fetch physicians');
      const data = await res.json();
      setPhysicians(Array.isArray(data.physicians) ? data.physicians : []);
      setErrorMessage('');
    } catch (err) {
      setErrorMessage('Error loading physicians: ' + err.message);
      setPhysicians([]);
    } finally {
      setLoading(false);
    }
  };

  // ============ FETCH NURSES ============
  const fetchNurses = async (date = targetDate) => {
    setLoading(true);
    try {
      const url = new URL(`${BACKEND_URL}/api/schedules/nurses`);
      url.searchParams.append('date', date);
      
      const res = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!res.ok) throw new Error('Failed to fetch nurses');
      const data = await res.json();
      setNurses(Array.isArray(data.nurses) ? data.nurses : []);
      setErrorMessage('');
    } catch (err) {
      setErrorMessage('Error loading nurses: ' + err.message);
      setNurses([]);
    } finally {
      setLoading(false);
    }
  };

  // ============ FETCH MY SCHEDULE ============
  const fetchMySchedule = async (date = targetDate) => {
    setLoading(true);
    try {
      // Try to get nurse schedule first
      let url = new URL(`${BACKEND_URL}/api/schedules/nurses/me`);
      url.searchParams.append('date', date);
      let res = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      
      if (res.ok) {
        const data = await res.json();
        setMySchedule({ ...data, type: 'nurse' });
      } else {
        // Try physician schedule
        url = new URL(`${BACKEND_URL}/api/schedules/physicians/me`);
        url.searchParams.append('date', date);
        res = await fetch(url, {
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        });
        if (res.ok) {
          const data = await res.json();
          setMySchedule({ ...data, type: 'physician' });
        } else {
          setMySchedule(null);
        }
      }
      setErrorMessage('');
    } catch (err) {
      setErrorMessage('Error loading your schedule: ' + err.message);
      setMySchedule(null);
    } finally {
      setLoading(false);
    }
  };

  // ============ MODAL HANDLERS ============
  const openAddPhysicianModal = () => {
    setEditForm({
      physicianId: 'PHY' + Date.now(),
      name: '',
      surname: '',
      specialty: SPECIALTIES[0],
      status: 'ON_DUTY',
      shiftStart: '08:00',
      shiftEnd: '16:00',
      maxPatients: 20,
      notes: ''
    });
    setModalType('physician');
    setModalMode('add');
    setShowModal(true);
  };

  const openAddNurseModal = () => {
    setEditForm({
      nurseId: 'NUR' + Date.now(),
      name: '',
      surname: '',
      status: 'ON_DUTY',
      shiftStart: '07:00',
      shiftEnd: '19:00',
      maxBeds: 5,
      esiLevel: 3,
      certifications: '',
      notes: ''
    });
    setModalType('nurse');
    setModalMode('add');
    setShowModal(true);
  };

  const openEditModal = (staff, type) => {
    setSelectedStaff(staff);
    if (type === 'physician') {
      setEditForm({
        physicianId: staff.physicianId,
        name: staff.name,
        surname: staff.surname,
        specialty: staff.specialty,
        status: staff.status,
        shiftStart: staff.shiftStart || '08:00',
        shiftEnd: staff.shiftEnd || '16:00',
        maxPatients: staff.maxPatients || 20,
        notes: staff.notes || ''
      });
    } else {
      setEditForm({
        nurseId: staff.nurseId,
        name: staff.name,
        surname: staff.surname,
        status: staff.status,
        shiftStart: staff.shiftStart || '07:00',
        shiftEnd: staff.shiftEnd || '19:00',
        maxBeds: staff.maxBeds || 5,
        esiLevel: staff.esiLevel || 3,
        certifications: (staff.certifications || []).join(', '),
        notes: staff.notes || ''
      });
    }
    setModalType(type);
    setModalMode('edit');
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setModalMode('view');
    setSelectedStaff(null);
  };

  // ============ SAVE STAFF ============
  const saveStaff = async () => {
    if (!editForm.name.trim()) {
      setErrorMessage('Name is required');
      return;
    }

    try {
      const headers = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      };

      const payload = {
        date: targetDate,
        ...editForm
      };

      const endpoint = modalType === 'physician' ? '/api/schedules/physicians' : '/api/schedules/nurses';
      
      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('Failed to save schedule');
      setSuccessMessage(`Schedule saved successfully`);
      closeModal();
      
      if (modalType === 'physician') {
        fetchPhysicians(targetDate);
      } else {
        fetchNurses(targetDate);
      }
      
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // ============ DELETE STAFF ============
  const deleteStaff = async () => {
    // Note: DELETE endpoint would need to be implemented in backend
    setErrorMessage('Delete functionality not yet implemented');
  };

  // ============ EXPORT CSV ============
  const exportCSV = async () => {
    try {
      const staff = modalType === 'physician' ? physicians : nurses;
      
      const headers = modalType === 'physician' 
        ? ['physicianId,name,surname,email,specialty,date,status,shiftStart,shiftEnd,maxPatients,notes']
        : ['nurseId,name,surname,email,licenseNumber,date,status,shiftStart,shiftEnd,maxBeds,esiLevel,notes'];
      
      const rows = staff.map(s => {
        if (modalType === 'physician') {
          return `${s.physicianId},${s.name},${s.surname},${s.email || ''},${s.specialty},${targetDate},${s.status},${s.shiftStart || '08:00'},${s.shiftEnd || '16:00'},${s.maxPatients || 20},"${s.notes || ''}"`;
        } else {
          return `${s.nurseId},${s.name},${s.surname},${s.email || ''},${s.licenseNumber || ''},${targetDate},${s.status},${s.shiftStart || '07:00'},${s.shiftEnd || '19:00'},${s.maxBeds || 5},${s.esiLevel || 3},"${s.notes || ''}"`;
        }
      });
      
      const csv = [...headers, ...rows].join('\n');

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `staff-schedule-${targetDate}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      setSuccessMessage('CSV exported successfully');
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // ============ IMPORT CSV ============
  const importCSV = async (file) => {
    if (!file) return;

    try {
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      if (lines.length < 2) throw new Error('CSV file must have header and at least one data row');

      setSuccessMessage('CSV import would be processed here');
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // ============ FILTERED DATA ============
  const filteredStaff = (scheduleTab === 'physicians' ? physicians : nurses).filter(s => {
    if (scheduleTab === 'physicians') {
      if (filter.specialty !== 'all' && s.specialty !== filter.specialty) return false;
      if (filter.status !== 'all' && s.status !== filter.status) return false;
    } else {
      if (filter.status !== 'all' && s.status !== filter.status) return false;
    }
    return true;
  });

  // ============ RENDER CLOCK TAB ============
  const renderClockTab = () => (
    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
      {/* Alerts */}
      {errorMessage && (
        <div style={{ marginBottom: 20, padding: '14px 18px', backgroundColor: '#fee2e2', border: '1px solid #fecaca', borderRadius: '10px', color: '#991b1b', fontSize: '13px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
          <span style={{ fontWeight: '600' }}>⚠️ {errorMessage}</span>
          <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b', fontSize: '18px' }}>✕</button>
        </div>
      )}
      {successMessage && (
        <div style={{ marginBottom: 20, padding: '14px 18px', backgroundColor: '#dcfce7', border: '1px solid #86efac', borderRadius: '10px', color: '#166534', fontSize: '13px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
          <span style={{ fontWeight: '600' }}>✓ {successMessage}</span>
          <button onClick={() => setSuccessMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534', fontSize: '18px' }}>✕</button>
        </div>
      )}

      {/* Main Clock Display Card */}
      <div style={{
        backgroundColor: '#fff',
        borderRadius: '16px',
        padding: '48px 40px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
        border: '1px solid #e2e8f0',
        marginBottom: 28,
        textAlign: 'center'
      }}>
        {/* Current Time - Large Display */}
        <div style={{ marginBottom: 32 }}>
          <p style={{ margin: '0 0 12px 0', fontSize: '13px', fontWeight: '600', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Current Time
          </p>
          <h2 style={{ margin: 0, fontSize: '72px', fontWeight: '700', color: '#0f172a', fontFamily: 'monospace', letterSpacing: '-2px' }}>
            {liveTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true })}
          </h2>
          <p style={{ margin: '8px 0 0 0', fontSize: '15px', color: '#64748b', fontWeight: '500' }}>
            {liveTime.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
          </p>
        </div>

        {/* Status Cards Row */}
        {currentRecord && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 36, padding: '20px 0', borderTop: '1px solid #e2e8f0', borderBottom: '1px solid #e2e8f0' }}>
            <div>
              <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '600', color: '#64748b', textTransform: 'uppercase' }}>Status</p>
              <p style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: currentRecord.clockOutTime ? '#ef4444' : '#10b981' }}>
                {currentRecord.clockOutTime ? '✓ Clocked Out' : '● On Duty'}
              </p>
            </div>
            <div>
              <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '600', color: '#64748b', textTransform: 'uppercase' }}>Work Time</p>
              <p style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                {currentRecord.totalWorkTime ? (currentRecord.totalWorkTime / 1000 / 60 / 60).toFixed(1) : '0'}h
              </p>
            </div>
            <div>
              <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '600', color: '#64748b', textTransform: 'uppercase' }}>Breaks</p>
              <p style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                {currentRecord.totalBreakTime ? Math.floor(currentRecord.totalBreakTime / 1000 / 60) : 0}m
              </p>
            </div>
          </div>
        )}

        {/* Main Action Buttons */}
        <div style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
          {!currentRecord?.clockInTime ? (
            <button
              onClick={handleClockIn}
              disabled={clockLoading}
              style={{
                padding: '18px 56px',
                backgroundColor: '#10b981',
                color: '#fff',
                border: 'none',
                borderRadius: '12px',
                cursor: clockLoading ? 'not-allowed' : 'pointer',
                fontWeight: '700',
                fontSize: '16px',
                opacity: clockLoading ? 0.6 : 1,
                boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)',
                transition: 'all 0.25s ease',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}
              onMouseEnter={(e) => !clockLoading && (e.target.style.boxShadow = '0 8px 20px rgba(16, 185, 129, 0.4)')}
              onMouseLeave={(e) => !clockLoading && (e.target.style.boxShadow = '0 4px 12px rgba(16, 185, 129, 0.3)')}
            >
              🔵 Clock In
            </button>
          ) : !currentRecord?.clockOutTime ? (
            <>
              <button
                onClick={onBreak ? handleBreakEnd : handleBreakStart}
                disabled={clockLoading}
                style={{
                  padding: '18px 40px',
                  backgroundColor: '#f59e0b',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: clockLoading ? 'not-allowed' : 'pointer',
                  fontWeight: '700',
                  fontSize: '16px',
                  opacity: clockLoading ? 0.6 : 1,
                  boxShadow: '0 4px 12px rgba(245, 158, 11, 0.3)',
                  transition: 'all 0.25s ease',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}
                onMouseEnter={(e) => !clockLoading && (e.target.style.boxShadow = '0 8px 20px rgba(245, 158, 11, 0.4)')}
                onMouseLeave={(e) => !clockLoading && (e.target.style.boxShadow = '0 4px 12px rgba(245, 158, 11, 0.3)')}
              >
                {onBreak ? '✓ End Break' : '☕ Start Break'}
              </button>
              <button
                onClick={handleClockOut}
                disabled={clockLoading}
                style={{
                  padding: '18px 56px',
                  backgroundColor: '#ef4444',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: clockLoading ? 'not-allowed' : 'pointer',
                  fontWeight: '700',
                  fontSize: '16px',
                  opacity: clockLoading ? 0.6 : 1,
                  boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)',
                  transition: 'all 0.25s ease',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}
                onMouseEnter={(e) => !clockLoading && (e.target.style.boxShadow = '0 8px 20px rgba(239, 68, 68, 0.4)')}
                onMouseLeave={(e) => !clockLoading && (e.target.style.boxShadow = '0 4px 12px rgba(239, 68, 68, 0.3)')}
              >
                🛑 Clock Out
              </button>
            </>
          ) : (
            <div style={{ padding: '20px 40px', backgroundColor: '#f1f5f9', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
              <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#475569' }}>
                ✓ You have completed your shift
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      {currentRecord && (
        <div style={{ backgroundColor: '#fff', borderRadius: '16px', padding: '24px 28px', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Today's Summary
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16 }}>
            <div>
              <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#64748b', fontWeight: '500', textTransform: 'uppercase' }}>Clock In</p>
              <p style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a', fontFamily: 'monospace' }}>
                {currentRecord.clockInTime ? new Date(currentRecord.clockInTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--'}
              </p>
            </div>
            <div>
              <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#64748b', fontWeight: '500', textTransform: 'uppercase' }}>Clock Out</p>
              <p style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a', fontFamily: 'monospace' }}>
                {currentRecord.clockOutTime ? new Date(currentRecord.clockOutTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--'}
              </p>
            </div>
            <div>
              <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#64748b', fontWeight: '500', textTransform: 'uppercase' }}>Break Taken</p>
              <p style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#f59e0b' }}>
                {currentRecord.totalBreakTime ? Math.floor(currentRecord.totalBreakTime / 1000 / 60) : 0} min
              </p>
            </div>
            <div>
              <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#64748b', fontWeight: '500', textTransform: 'uppercase' }}>Location</p>
              <p style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>
                {currentRecord.location || 'ED'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Shift Selection Modal */}
      {showShiftSelection && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ backgroundColor: '#fff', borderRadius: '16px', padding: '32px', maxWidth: '500px', width: '90%', boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)', border: '1px solid #e2e8f0' }}>
            <h2 style={{ margin: '0 0 8px 0', fontSize: '20px', fontWeight: '700', color: '#0f172a' }}>Select Your Shift</h2>
            <p style={{ margin: '0 0 24px 0', fontSize: '14px', color: '#64748b' }}>Choose an assigned shift to clock in</p>

            {shiftsLoading ? (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: '#64748b' }}>Loading shifts...</div>
            ) : availableShifts.length === 0 ? (
              <div style={{ padding: '40px 20px', backgroundColor: '#fef3c7', borderRadius: '12px', color: '#92400e', fontSize: '14px', fontWeight: '500', textAlign: 'center' }}>
                ⚠️ No shifts available for today
              </div>
            ) : (
              <>
                <div style={{ display: 'grid', gap: 12, marginBottom: 24 }}>
                  {availableShifts.map((shift, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedShift(shift)}
                      style={{
                        padding: '16px',
                        backgroundColor: selectedShift?.id === shift.id ? '#ecfdf5' : '#f8fafc',
                        border: selectedShift?.id === shift.id ? '2px solid #10b981' : '1px solid #cbd5e1',
                        borderRadius: '12px',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        textAlign: 'left'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = selectedShift?.id === shift.id ? '#ecfdf5' : '#f1f5f9';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = selectedShift?.id === shift.id ? '#ecfdf5' : '#f8fafc';
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div>
                          <p style={{ margin: '0 0 4px 0', fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                            {shift.shiftStart} - {shift.shiftEnd}
                          </p>
                          <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>
                            {shift.department || 'Department'} • {shift.location || 'ED'}
                          </p>
                        </div>
                        <div style={{
                          width: '20px',
                          height: '20px',
                          borderRadius: '50%',
                          border: '2px solid ' + (selectedShift?.id === shift.id ? '#10b981' : '#cbd5e1'),
                          backgroundColor: selectedShift?.id === shift.id ? '#10b981' : 'transparent'
                        }} />
                      </div>
                    </button>
                  ))}
                </div>

                {/* Action Buttons */}
                <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
                  <button
                    onClick={() => {
                      setShowShiftSelection(false);
                      setSelectedShift(null);
                      setAvailableShifts([]);
                    }}
                    style={{
                      padding: '12px 24px',
                      backgroundColor: '#f1f5f9',
                      color: '#0f172a',
                      border: '1px solid #cbd5e1',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: '600',
                      fontSize: '14px',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => e.target.style.backgroundColor = '#e2e8f0'}
                    onMouseLeave={(e) => e.target.style.backgroundColor = '#f1f5f9'}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleConfirmClockIn}
                    disabled={!selectedShift || clockLoading}
                    style={{
                      padding: '12px 24px',
                      backgroundColor: selectedShift && !clockLoading ? '#10b981' : '#cbd5e1',
                      color: '#fff',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: selectedShift && !clockLoading ? 'pointer' : 'not-allowed',
                      fontWeight: '700',
                      fontSize: '14px',
                      transition: 'all 0.2s',
                      opacity: clockLoading ? 0.6 : 1
                    }}
                    onMouseEnter={(e) => selectedShift && !clockLoading && (e.target.style.backgroundColor = '#059669')}
                    onMouseLeave={(e) => selectedShift && !clockLoading && (e.target.style.backgroundColor = '#10b981')}
                  >
                    {clockLoading ? 'Clocking In...' : 'Clock In'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );

  // ============ RENDER HISTORY TAB ============
  const renderHistoryTab = () => (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ backgroundColor: '#fff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '32px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
        <h2 style={{ margin: '0 0 24px 0', fontSize: '20px', fontWeight: '700', color: '#0f172a' }}>
          📋 Time & Attendance History
        </h2>

        {errorMessage && mainTab === 'history' && (
          <div style={{ marginBottom: 20, padding: '14px 18px', backgroundColor: '#fee2e2', border: '1px solid #fecaca', borderRadius: '10px', color: '#991b1b', fontSize: '13px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: '600' }}>⚠️ {errorMessage}</span>
            <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b', fontSize: '18px' }}>✕</button>
          </div>
        )}

        {/* Filter Controls */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 24, padding: '18px', backgroundColor: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: '700', color: '#0f172a', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>From Date</label>
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={{ padding: '10px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '14px', backgroundColor: '#fff', fontWeight: '500' }} />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: '700', color: '#0f172a', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>To Date</label>
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={{ padding: '10px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '14px', backgroundColor: '#fff', fontWeight: '500' }} />
          </div>
          <button onClick={handleApplyFilter} style={{ padding: '10px 20px', backgroundColor: '#10b981', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '600', fontSize: '13px', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.backgroundColor = '#059669'} onMouseLeave={(e) => e.target.style.backgroundColor = '#10b981'}>
            Apply Filter
          </button>
          <button onClick={handleDownloadCSV} style={{ padding: '10px 20px', backgroundColor: '#0f172a', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '600', fontSize: '13px', transition: 'all 0.2s' }} onMouseEnter={(e) => e.target.style.opacity = '0.9'} onMouseLeave={(e) => e.target.style.opacity = '1'}>
            ↓ Export CSV
          </button>
        </div>

        {/* Records Table */}
        {historyLoading ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '60px 20px', fontSize: '14px', fontWeight: '500' }}>Loading records...</div>
        ) : clockHistory.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '60px 20px', backgroundColor: '#f8fafc', borderRadius: '12px', fontSize: '14px', fontWeight: '500' }}>
            No time records found for the selected period
          </div>
        ) : (
          <div style={{ overflowX: 'auto', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ backgroundColor: '#f1f5f9', borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Date</th>
                  <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Clock In</th>
                  <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Clock Out</th>
                  <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Work Hours</th>
                  <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Location</th>
                  <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {clockHistory.map((record, idx) => (
                  <tr key={idx} style={{ backgroundColor: idx % 2 === 0 ? '#fff' : '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '14px 16px', fontWeight: '600', color: '#0f172a' }}>{new Date(record.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</td>
                    <td style={{ padding: '14px 16px', textAlign: 'center', color: '#0f172a', fontFamily: 'monospace', fontWeight: '500' }}>{record.clockInTime ? new Date(record.clockInTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '—'}</td>
                    <td style={{ padding: '14px 16px', textAlign: 'center', color: '#0f172a', fontFamily: 'monospace', fontWeight: '500' }}>{record.clockOutTime ? new Date(record.clockOutTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '—'}</td>
                    <td style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#10b981' }}>{record.totalWorkTime ? (record.totalWorkTime / 1000 / 60 / 60).toFixed(2) : '0'}h</td>
                    <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                      <span style={{ display: 'inline-block', padding: '4px 10px', backgroundColor: '#ecfdf5', color: '#059669', borderRadius: '6px', fontSize: '11px', fontWeight: '600' }}>
                        {record.location || 'ED'}
                      </span>
                    </td>
                    <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                      <span style={{
                        display: 'inline-block',
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '600',
                        backgroundColor: record.lateArrivalApprovalStatus === 'APPROVED' ? '#dcfce7' : record.lateArrivalApprovalStatus === 'PENDING' ? '#fef3c7' : record.lateArrivalApprovalStatus === 'REJECTED' ? '#fee2e2' : '#f1f5f9',
                        color: record.lateArrivalApprovalStatus === 'APPROVED' ? '#166534' : record.lateArrivalApprovalStatus === 'PENDING' ? '#92400e' : record.lateArrivalApprovalStatus === 'REJECTED' ? '#991b1b' : '#475569'
                      }}>
                        {record.lateArrivalApprovalStatus === 'APPROVED' ? '✓ Approved' : record.lateArrivalApprovalStatus === 'PENDING' ? '⏳ Pending' : record.lateArrivalApprovalStatus === 'REJECTED' ? '✗ Rejected' : 'Normal'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );

  // ============ RENDER STATISTICS TAB ============
  const renderStatisticsTab = () => (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ backgroundColor: '#fff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '32px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
        <h2 style={{ margin: '0 0 24px 0', fontSize: '20px', fontWeight: '700', color: '#0f172a' }}>
          📊 Performance Analytics
        </h2>

        {statsLoading ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '80px 20px', fontSize: '14px', fontWeight: '500' }}>
            Loading analytics data...
          </div>
        ) : stats ? (
          <>
            {/* KPI Cards Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16, marginBottom: 32 }}>
              <div style={{
                backgroundColor: '#ecfdf5',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid #d1fae5',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)'
              }}>
                <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '700', color: '#065f46', textTransform: 'uppercase', letterSpacing: '0.3px' }}>Total Work Hours</p>
                <p style={{ margin: 0, fontSize: '32px', fontWeight: '800', color: '#10b981' }}>{stats.totalWorkHours?.toFixed(1) || '0'}</p>
                <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#047857', fontWeight: '500' }}>This month</p>
              </div>

              <div style={{
                backgroundColor: '#f0f9ff',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid #bfdbfe',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)'
              }}>
                <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '700', color: '#0c4a6e', textTransform: 'uppercase', letterSpacing: '0.3px' }}>Shifts Completed</p>
                <p style={{ margin: 0, fontSize: '32px', fontWeight: '800', color: '#0284c7' }}>{stats.shiftsCompleted || '0'}</p>
                <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#0369a1', fontWeight: '500' }}>Completed successfully</p>
              </div>

              <div style={{
                backgroundColor: '#fef2f2',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid #fee2e2',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)'
              }}>
                <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '700', color: '#7f1d1d', textTransform: 'uppercase', letterSpacing: '0.3px' }}>Attendance Rate</p>
                <p style={{ margin: 0, fontSize: '32px', fontWeight: '800', color: '#dc2626' }}>{stats.attendanceRate?.toFixed(0) || '0'}%</p>
                <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#b91c1c', fontWeight: '500' }}>On-time arrivals</p>
              </div>

              <div style={{
                backgroundColor: '#fff7ed',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid #fed7aa',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)'
              }}>
                <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '700', color: '#92400e', textTransform: 'uppercase', letterSpacing: '0.3px' }}>Late Arrivals</p>
                <p style={{ margin: 0, fontSize: '32px', fontWeight: '800', color: '#f59e0b' }}>{stats.lateArrivals || '0'}</p>
                <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#d97706', fontWeight: '500' }}>Requires approval</p>
              </div>
            </div>

            {/* Summary Box */}
            <div style={{
              backgroundColor: '#f8fafc',
              borderRadius: '12px',
              padding: '20px 24px',
              border: '1px solid #e2e8f0'
            }}>
              <h3 style={{ margin: '0 0 12px 0', fontSize: '13px', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Summary
              </h3>
              <p style={{ margin: 0, color: '#475569', fontSize: '13px', lineHeight: '1.6', fontWeight: '500' }}>
                {stats.attendanceRate >= 95 ? '✓ Excellent attendance record. You are maintaining outstanding punctuality.' : stats.attendanceRate >= 90 ? '✓ Good attendance record. A few late arrivals noted.' : 'Please improve your punctuality to maintain consistent attendance standards.'}
              </p>
            </div>
          </>
        ) : (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '80px 20px', fontSize: '14px', fontWeight: '500' }}>
            No analytics data available yet
          </div>
        )}
      </div>
    </div>
  );

  // ============ RENDER SCHEDULE TAB ============
  const renderScheduleTab = () => (
    <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
      {/* Date Picker */}
      <div style={{ marginBottom: 24, display: 'flex', gap: 12, alignItems: 'center', backgroundColor: '#fff', borderRadius: '12px', padding: '16px 20px', border: '1px solid #e2e8f0' }}>
        <label style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.5px' }}>📅 Schedule Date:</label>
        <input 
          type="date" 
          value={targetDate} 
          onChange={(e) => setTargetDate(e.target.value)}
          style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '14px', backgroundColor: '#fff', fontWeight: '500' }}
        />
      </div>

      {/* Schedule Sub-Tabs - Professional Styling */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {[
          { key: 'my-schedule', label: 'My Schedule', icon: '👤' },
          { key: 'physicians', label: 'Physicians', icon: '👨‍⚕️' },
          { key: 'nurses', label: 'Nurses', icon: '👩‍⚕️' }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setScheduleTab(tab.key)}
            style={{
              padding: '14px 20px',
              backgroundColor: scheduleTab === tab.key ? '#ecfdf5' : 'transparent',
              color: scheduleTab === tab.key ? '#10b981' : '#64748b',
              border: 'none',
              borderBottom: scheduleTab === tab.key ? '3px solid #10b981' : 'none',
              cursor: 'pointer',
              fontWeight: scheduleTab === tab.key ? '700' : '600',
              fontSize: '13px',
              transition: 'all 0.25s ease',
              borderRadius: '8px 8px 0 0',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}
            onMouseEnter={(e) => scheduleTab !== tab.key && (e.target.style.backgroundColor = '#f1f5f9')}
            onMouseLeave={(e) => scheduleTab !== tab.key && (e.target.style.backgroundColor = 'transparent')}
          >
            <span style={{ marginRight: '6px' }}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* MY SCHEDULE */}
      {scheduleTab === 'my-schedule' && (
        <div style={{ backgroundColor: '#fff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '32px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
          {mySchedule ? (
            <>
              <h2 style={{ margin: '0 0 24px 0', fontSize: '20px', fontWeight: '700', color: '#0f172a' }}>
                👤 {mySchedule.name}
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 20 }}>
                <div style={{ backgroundColor: '#f8fafc', borderRadius: '12px', padding: '18px', border: '1px solid #e2e8f0' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: '11px', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Schedule Date</p>
                  <p style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: '#0f172a', fontFamily: 'monospace' }}>{targetDate}</p>
                </div>
                <div style={{ backgroundColor: '#ecfdf5', borderRadius: '12px', padding: '18px', border: '1px solid #d1fae5' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: '11px', fontWeight: '700', color: '#065f46', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Status</p>
                  <p style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: '#10b981', textTransform: 'uppercase' }}>{mySchedule.status}</p>
                </div>
                <div style={{ backgroundColor: '#f0f9ff', borderRadius: '12px', padding: '18px', border: '1px solid #bfdbfe' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: '11px', fontWeight: '700', color: '#0c4a6e', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Shift Hours</p>
                  <p style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: '#0284c7', fontFamily: 'monospace' }}>{mySchedule.shiftStart} - {mySchedule.shiftEnd}</p>
                </div>
              </div>
            </>
          ) : (
            <div style={{ padding: '60px', textAlign: 'center', color: '#64748b' }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '16px', fontWeight: '600' }}>No schedule found</p>
              <p style={{ margin: 0, fontSize: '13px' }}>for {targetDate}</p>
            </div>
          )}
        </div>
      )}

      {/* PHYSICIANS/NURSES */}
      {(scheduleTab === 'physicians' || scheduleTab === 'nurses') && (
        <div style={{ backgroundColor: '#fff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '32px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)' }}>
          <h2 style={{ margin: '0 0 24px 0', fontSize: '20px', fontWeight: '700', color: '#0f172a' }}>
            {scheduleTab === 'physicians' ? '👨‍⚕️ Physicians' : '👩‍⚕️ Nurses'} Schedule
          </h2>

          {/* Professional Toolbar */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap', alignItems: 'center' }}>
            <button 
              onClick={scheduleTab === 'physicians' ? openAddPhysicianModal : openAddNurseModal}
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 8, 
                padding: '10px 20px', 
                backgroundColor: '#10b981', 
                color: '#fff', 
                border: 'none', 
                borderRadius: '8px', 
                cursor: 'pointer', 
                fontWeight: '600', 
                fontSize: '13px',
                transition: 'all 0.2s',
                boxShadow: '0 2px 8px rgba(16, 185, 129, 0.2)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = '#059669';
                e.target.style.boxShadow = '0 4px 12px rgba(16, 185, 129, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = '#10b981';
                e.target.style.boxShadow = '0 2px 8px rgba(16, 185, 129, 0.2)';
              }}
            >
              <FiPlus size={16} /> Add {scheduleTab === 'physicians' ? 'Physician' : 'Nurse'}
            </button>

            <label style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 20px', backgroundColor: '#f8fafc', color: '#0f172a', border: '1px solid #cbd5e1', borderRadius: '8px', cursor: 'pointer', fontWeight: '600', fontSize: '13px', transition: 'all 0.2s', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <FiUpload size={16} /> Import CSV
              <input type="file" accept=".csv" onChange={(e) => {
                setCsvUploadMode(scheduleTab === 'physicians' ? 'physicians' : 'nurses');
                importCSV(e.target.files?.[0]);
              }} style={{ display: 'none' }} />
            </label>

            <button onClick={exportCSV} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 20px', backgroundColor: '#0f172a', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '600', fontSize: '13px', transition: 'all 0.2s', textTransform: 'uppercase', letterSpacing: '0.5px' }} 
              onMouseEnter={(e) => e.target.style.opacity = '0.9'}
              onMouseLeave={(e) => e.target.style.opacity = '1'}>
              <FiDownload size={16} /> Export CSV
            </button>

            <div style={{ flex: 1 }} />

            {/* Filters */}
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              {scheduleTab === 'physicians' && (
                <select value={filter.specialty} onChange={(e) => setFilter(prev => ({ ...prev, specialty: e.target.value }))} style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', backgroundColor: '#fff', fontSize: '13px', cursor: 'pointer', fontWeight: '500' }}>
                  <option value="all">All Specialties</option>
                  {SPECIALTIES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              )}

              <select value={filter.status} onChange={(e) => setFilter(prev => ({ ...prev, status: e.target.value }))} style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', backgroundColor: '#fff', fontSize: '13px', cursor: 'pointer', fontWeight: '500' }}>
                <option value="all">All Status</option>
                {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>

          {/* Professional Table */}
          {loading ? (
            <div style={{ textAlign: 'center', color: '#64748b', padding: '60px 20px', fontSize: '14px', fontWeight: '500' }}>Loading {scheduleTab}...</div>
          ) : filteredStaff.length === 0 ? (
            <div style={{ textAlign: 'center', color: '#64748b', padding: '60px 20px', backgroundColor: '#f8fafc', borderRadius: '12px', fontSize: '14px', fontWeight: '500' }}>No {scheduleTab} found</div>
          ) : (
            <div style={{ overflowX: 'auto', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f1f5f9', borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Name</th>
                    {scheduleTab === 'physicians' && <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Specialty</th>}
                    <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Status</th>
                    <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Shift</th>
                    {scheduleTab === 'physicians' && <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Patients</th>}
                    {scheduleTab === 'nurses' && <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Beds</th>}
                    <th style={{ padding: '14px 16px', textAlign: 'right', fontWeight: '700', color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.3px', fontSize: '11px' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredStaff.map((member, idx) => (
                    <tr key={scheduleTab === 'physicians' ? member.physicianId : member.nurseId} style={{ backgroundColor: idx % 2 === 0 ? '#fff' : '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '14px 16px', fontWeight: '600', color: '#0f172a' }}>{member.name} {member.surname || ''}</td>
                      {scheduleTab === 'physicians' && (
                        <td style={{ padding: '14px 16px', textAlign: 'left' }}>
                          <span style={{ display: 'inline-block', padding: '4px 10px', backgroundColor: '#f0f9ff', color: '#0369a1', borderRadius: '6px', fontSize: '11px', fontWeight: '600' }}>
                            {member.specialty}
                          </span>
                        </td>
                      )}
                      <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                        <span style={{ display: 'inline-block', padding: '4px 10px', backgroundColor: member.status === 'ON_DUTY' ? '#dcfce7' : '#fef3c7', color: member.status === 'ON_DUTY' ? '#166534' : '#92400e', borderRadius: '6px', fontSize: '11px', fontWeight: '600' }}>
                          {member.status}
                        </span>
                      </td>
                      <td style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '600', color: '#0f172a', fontFamily: 'monospace' }}>{member.shiftStart} - {member.shiftEnd}</td>
                      {scheduleTab === 'physicians' && (
                        <td style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#10b981' }}>{member.currentPatientCount}/{member.maxPatients}</td>
                      )}
                      {scheduleTab === 'nurses' && (
                        <td style={{ padding: '14px 16px', textAlign: 'center', fontWeight: '700', color: '#10b981' }}>{member.currentBedCount}/{member.maxBeds}</td>
                      )}
                      <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                        <button onClick={() => openEditModal(member, scheduleTab === 'physicians' ? 'physician' : 'nurse')} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '6px 12px', backgroundColor: '#f0f9ff', color: '#0369a1', border: '1px solid #bfdbfe', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600', transition: 'all 0.2s' }}
                          onMouseEnter={(e) => {
                            e.target.style.backgroundColor = '#bfdbfe';
                            e.target.style.color = '#0c4a6e';
                          }}
                          onMouseLeave={(e) => {
                            e.target.style.backgroundColor = '#f0f9ff';
                            e.target.style.color = '#0369a1';
                          }}>
                          <FiEdit2 size={14} /> Edit
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // ============ MAIN RENDER ============
  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div style={{ backgroundColor: '#f8fafc', minHeight: '100vh', padding: '24px 32px' }}>
        {/* Professional Header */}
        <div style={{ marginBottom: 40, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1 style={{ margin: '0 0 4px 0', fontSize: '32px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.5px' }}>
              Staff Time & Attendance
            </h1>
            <p style={{ margin: 0, color: '#64748b', fontSize: '14px', fontWeight: '500' }}>
              {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
            </p>
          </div>
          <div style={{ textAlign: 'right', padding: '12px 20px', backgroundColor: '#ecfdf5', borderRadius: '10px', border: '1px solid #d1fae5' }}>
            <p style={{ margin: 0, fontSize: '12px', color: '#059669', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              System Status
            </p>
            <p style={{ margin: '4px 0 0 0', fontSize: '18px', fontWeight: '700', color: '#10b981' }}>
              ✓ Active
            </p>
          </div>
        </div>

        {/* Professional Tab Navigation */}
        <div style={{
          display: 'flex',
          gap: 8,
          marginBottom: 32,
          borderBottom: '2px solid #e2e8f0',
          paddingBottom: 0
        }}>
          {[
            { id: 'clock-in-out', label: 'Clock In/Out', icon: '⏰' },
            { id: 'schedule', label: 'My Schedule', icon: '📅' },
            { id: 'history', label: 'Time History', icon: '📋' },
            { id: 'statistics', label: 'Analytics', icon: '📊' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setMainTab(tab.id)}
              style={{
                padding: '14px 20px',
                backgroundColor: mainTab === tab.id ? '#ecfdf5' : 'transparent',
                color: mainTab === tab.id ? '#10b981' : '#64748b',
                border: 'none',
                borderBottom: mainTab === tab.id ? '3px solid #10b981' : 'none',
                cursor: 'pointer',
                fontWeight: mainTab === tab.id ? '700' : '600',
                fontSize: '13px',
                transition: 'all 0.25s ease',
                borderRadius: '8px 8px 0 0'
              }}
              onMouseEnter={(e) => !mainTab === tab.id && (e.target.style.backgroundColor = '#f1f5f9')}
              onMouseLeave={(e) => !mainTab === tab.id && (e.target.style.backgroundColor = 'transparent')}
            >
              <span style={{ marginRight: '6px' }}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Alerts */}
        {successMessage && (
          <div style={{ marginBottom: 20, padding: '12px 16px', backgroundColor: '#dcfce7', border: '1px solid #86efac', borderRadius: '8px', color: '#166534', fontSize: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>{successMessage}</span>
            <button onClick={() => setSuccessMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534' }}>✕</button>
          </div>
        )}
        {errorMessage && mainTab !== 'clock-in-out' && mainTab !== 'history' && mainTab !== 'statistics' && (
          <div style={{ marginBottom: 20, padding: '12px 16px', backgroundColor: '#fee2e2', border: '1px solid #fecaca', borderRadius: '8px', color: '#991b1b', fontSize: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>{errorMessage}</span>
            <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b' }}>✕</button>
          </div>
        )}

        {/* SCHEDULE TAB */}
        {mainTab === 'schedule' && renderScheduleTab()}

        {/* CLOCK IN/OUT TAB */}
        {mainTab === 'clock-in-out' && renderClockTab()}

        {/* HISTORY TAB */}
        {mainTab === 'history' && renderHistoryTab()}

        {/* STATISTICS TAB */}
        {mainTab === 'statistics' && renderStatisticsTab()}

        {/* Modal */}
        {showModal && (
          <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }}>
            <div style={{ backgroundColor: '#fff', borderRadius: '12px', width: '90%', maxWidth: '600px', maxHeight: '90vh', overflow: 'auto', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)' }}>
              {/* Modal Header */}
              <div style={{ padding: '20px 24px', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#f9fafb' }}>
                <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#0f172a' }}>
                  {modalMode === 'add' ? `Add ${modalType === 'physician' ? 'Physician' : 'Nurse'}` : `Edit ${modalType === 'physician' ? 'Physician' : 'Nurse'}`}
                </h2>
                <button onClick={closeModal} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: '20px' }}>
                  <FiX />
                </button>
              </div>

              {/* Modal Body */}
              <div style={{ padding: '24px' }}>
                {/* Name & Surname */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>First Name *</label>
                    <input type="text" value={editForm.name} onChange={(e) => setEditForm(prev => ({ ...prev, name: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box' }} placeholder="John" />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Last Name</label>
                    <input type="text" value={editForm.surname} onChange={(e) => setEditForm(prev => ({ ...prev, surname: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box' }} placeholder="Smith" />
                  </div>
                </div>

                {/* Specialty (Physician only) */}
                {modalType === 'physician' && (
                  <div style={{ marginBottom: 16 }}>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Specialty *</label>
                    <select value={editForm.specialty} onChange={(e) => setEditForm(prev => ({ ...prev, specialty: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box', cursor: 'pointer' }}>
                      {SPECIALTIES.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                  </div>
                )}

                {/* Status */}
                <div style={{ marginBottom: 16 }}>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Status</label>
                  <select value={editForm.status} onChange={(e) => setEditForm(prev => ({ ...prev, status: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box', cursor: 'pointer' }}>
                    {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>

                {/* Shift Times */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Shift Start</label>
                    <input type="time" value={editForm.shiftStart} onChange={(e) => setEditForm(prev => ({ ...prev, shiftStart: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Shift End</label>
                    <input type="time" value={editForm.shiftEnd} onChange={(e) => setEditForm(prev => ({ ...prev, shiftEnd: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box' }} />
                  </div>
                </div>

                {/* Physician-specific */}
                {modalType === 'physician' && (
                  <div style={{ marginBottom: 16 }}>
                    <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Max Patients</label>
                    <input type="number" value={editForm.maxPatients} onChange={(e) => setEditForm(prev => ({ ...prev, maxPatients: parseInt(e.target.value) }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box' }} min="1" />
                  </div>
                )}

                {/* Nurse-specific */}
                {modalType === 'nurse' && (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                      <div>
                        <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Max Beds</label>
                        <input type="number" value={editForm.maxBeds} onChange={(e) => setEditForm(prev => ({ ...prev, maxBeds: parseInt(e.target.value) }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box' }} min="1" />
                      </div>
                      <div>
                        <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>ESI Training Level</label>
                        <select value={editForm.esiLevel} onChange={(e) => setEditForm(prev => ({ ...prev, esiLevel: parseInt(e.target.value) }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box', cursor: 'pointer' }}>
                          {ESI_LEVELS.map(l => <option key={l} value={l}>{l === 1 ? 'Expert' : l === 2 ? 'Intermediate' : 'Trainee'}</option>)}
                        </select>
                      </div>
                    </div>
                    <div style={{ marginBottom: 16 }}>
                      <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Certifications (comma-separated)</label>
                      <input type="text" value={editForm.certifications} onChange={(e) => setEditForm(prev => ({ ...prev, certifications: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box' }} placeholder="ACLS, BLS, PALS" />
                    </div>
                  </>
                )}

                {/* Notes */}
                <div style={{ marginBottom: 16 }}>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Notes</label>
                  <textarea value={editForm.notes} onChange={(e) => setEditForm(prev => ({ ...prev, notes: e.target.value }))} style={{ width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', boxSizing: 'border-box', minHeight: '80px', fontFamily: 'inherit' }} placeholder="Any special instructions or constraints..." />
                </div>
              </div>

              {/* Modal Footer */}
              <div style={{ padding: '16px 24px', borderTop: '1px solid #e5e7eb', display: 'flex', justifyContent: 'flex-end', gap: 12, backgroundColor: '#f9fafb' }}>
                <button onClick={closeModal} style={{ padding: '10px 16px', borderRadius: '8px', border: '1px solid #d1d5db', backgroundColor: '#fff', color: '#374151', cursor: 'pointer', fontWeight: 600, fontSize: '14px' }}>
                  Cancel
                </button>
                <button onClick={saveStaff} style={{ padding: '10px 16px', borderRadius: '8px', border: 'none', backgroundColor: '#0f172a', color: '#fff', cursor: 'pointer', fontWeight: 600, fontSize: '14px' }}>
                  Save
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

export default StaffSchedule;
