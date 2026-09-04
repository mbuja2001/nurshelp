import React, { useState, useEffect } from 'react';
import { FiX, FiPlus, FiDownload, FiUpload, FiEdit2, FiTrash2, FiChevronDown, FiFilter } from 'react-icons/fi';
import { FaClock, FaHistory, FaCalendarAlt, FaChartBar } from 'react-icons/fa';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';

const StaffSchedule = () => {
  const { userName } = useUserFromToken();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');
  const userStr = localStorage.getItem('user');
  const user = userStr ? JSON.parse(userStr) : {};

  // ============ TAB STATE ============
  const [mainTab, setMainTab] = useState('schedule'); // 'schedule', 'clock-in-out', 'history', 'statistics'
  const [scheduleTab, setScheduleTab] = useState('physicians'); // 'my-schedule', 'physicians', 'nurses'
  const [targetDate, setTargetDate] = useState(new Date().toISOString().split('T')[0]);
  
  // ============ STAFF SCHEDULE STATE ============
  const [physicians, setPhysicians] = useState([]);
  const [nurses, setNurses] = useState([]);
  const [mySchedule, setMySchedule] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState({ specialty: 'all', status: 'all' });
  
  const [showModal, setShowModal] = useState(false);
  const [modalMode, setModalMode] = useState('view');
  const [modalType, setModalType] = useState('physician');
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
  const [csvUploadMode, setCsvUploadMode] = useState(null);

  // ============ CLOCK STATE ============
  const [currentRecord, setCurrentRecord] = useState(null);
  const [liveTime, setLiveTime] = useState(new Date());
  const [onBreak, setOnBreak] = useState(false);
  const [breakStartTime, setBreakStartTime] = useState(null);
  const [clockLoading, setClockLoading] = useState(false);

  // ============ HISTORY STATE ============
  const [clockHistory, setClockHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // ============ STATISTICS STATE ============
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [selectedWeek, setSelectedWeek] = useState('current');
  const [schedule, setSchedule] = useState([]);

  const SPECIALTIES = ['Cardiology', 'Pulmonology', 'Neurology', 'General Medicine', 'Emergency Medicine', 'Trauma Surgery', 'Orthopedics', 'Psychiatry', 'Internal Medicine'];
  const STATUSES = ['ON_DUTY', 'ON_CALL', 'OFF', 'VACATION'];
  const ESI_LEVELS = [1, 2, 3];

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

  // Load data on tab change
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
      const res = await fetch(`${BACKEND_URL}/api/staff/today-record`, {
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

  const handleClockIn = async () => {
    setClockLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/clock-in`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ location: user?.location || 'ED' })
      });
      const data = await res.json();
      if (data.success) {
        setCurrentRecord(data.record);
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
      const res = await fetch(`${BACKEND_URL}/api/staff/clock-out`, {
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
      const res = await fetch(`${BACKEND_URL}/api/staff/break-start`, {
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
      const res = await fetch(`${BACKEND_URL}/api/staff/break-end`, {
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
      let url = `${BACKEND_URL}/api/staff/clock-history?`;
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

  // ============ SCHEDULE METHODS ============
  const fetchPhysicians = async (date = targetDate) => {
    setLoading(true);
    try {
      const url = new URL(`${BACKEND_URL}/api/schedules/physicians`);
      url.searchParams.append('date', date);
      
      const res = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
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

  const fetchNurses = async (date = targetDate) => {
    setLoading(true);
    try {
      const url = new URL(`${BACKEND_URL}/api/schedules/nurses`);
      url.searchParams.append('date', date);
      
      const res = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
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

  const fetchMySchedule = async (date = targetDate) => {
    setLoading(true);
    try {
      let url = new URL(`${BACKEND_URL}/api/schedules/nurses/me`);
      url.searchParams.append('date', date);
      let res = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      
      if (res.ok) {
        const data = await res.json();
        setMySchedule({ ...data, type: 'nurse' });
      } else {
        url = new URL(`${BACKEND_URL}/api/schedules/physicians/me`);
        url.searchParams.append('date', date);
        res = await fetch(url, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
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

  const fetchWeeklySchedule = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/schedule?week=${selectedWeek}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setSchedule(data.schedule || []);
      }
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getDayShifts = (dayName) => {
    return schedule.filter(s => s.dayOfWeek === dayName);
  };

  // ============ RENDERING ============
  const renderClockTab = () => (
    <div style={styles.card}>
      <h2 style={{ margin: '0 0 20px 0', fontSize: '22px', fontWeight: '700' }}>⏰ Clock In/Out</h2>

      {errorMessage && (
        <div style={styles.errorMessage}>
          {errorMessage}
          <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b' }}>✕</button>
        </div>
      )}
      {successMessage && (
        <div style={styles.successMessage}>
          {successMessage}
          <button onClick={() => setSuccessMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534' }}>✕</button>
        </div>
      )}

      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ margin: '0 0 10px 0', fontSize: '24px', fontWeight: '700' }}>
          ⏰ {liveTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </h3>
        <p style={{ margin: 0, color: '#6b7280' }}>
          {liveTime.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
        </p>
      </div>

      {currentRecord ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
          <div style={styles.card}>
            <h3 style={{ margin: '0 0 16px 0', color: '#374151' }}>Current Status</h3>
            <div style={{ display: 'grid', gap: '12px' }}>
              <div>
                <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Clock In Time</p>
                <p style={{ margin: 0, fontSize: '18px', fontWeight: '700' }}>
                  {currentRecord.clockInTime 
                    ? new Date(currentRecord.clockInTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
                    : '--:--'
                  }
                </p>
              </div>
              <div>
                <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Status</p>
                <p style={{ margin: 0 }}>
                  <span style={{
                    display: 'inline-block',
                    padding: '4px 12px',
                    borderRadius: '20px',
                    fontSize: '12px',
                    fontWeight: '600',
                    backgroundColor: currentRecord.clockOutTime ? '#dcfce7' : '#fef3c7',
                    color: currentRecord.clockOutTime ? '#166534' : '#92400e'
                  }}>
                    {currentRecord.clockOutTime ? '✓ Clocked Out' : '⏳ Working'}
                  </span>
                </p>
              </div>
            </div>
          </div>

          <div style={styles.card}>
            <h3 style={{ margin: '0 0 16px 0', color: '#374151' }}>Today's Summary</h3>
            <div style={{ display: 'grid', gap: '12px' }}>
              <div>
                <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Work Hours</p>
                <p style={{ margin: 0, fontSize: '18px', fontWeight: '700' }}>
                  {currentRecord.totalWorkTime 
                    ? (currentRecord.totalWorkTime / 1000 / 60 / 60).toFixed(1) + 'h'
                    : '--'
                  }
                </p>
              </div>
              <div>
                <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Break Time</p>
                <p style={{ margin: 0, fontSize: '18px', fontWeight: '700' }}>
                  {currentRecord.totalBreakTime 
                    ? Math.floor(currentRecord.totalBreakTime / 1000 / 60) + 'min'
                    : '0min'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        {!currentRecord?.clockInTime ? (
          <button
            onClick={handleClockIn}
            disabled={clockLoading}
            style={{
              padding: '10px 20px',
              backgroundColor: '#10b981',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            🔵 Clock In
          </button>
        ) : !currentRecord?.clockOutTime ? (
          <>
            <button
              onClick={onBreak ? handleBreakEnd : handleBreakStart}
              disabled={clockLoading}
              style={{
                padding: '10px 20px',
                backgroundColor: '#f59e0b',
                color: '#ffffff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px'
              }}
            >
              {onBreak ? '✓ Finish Break' : '☕ Start Break'}
            </button>
            <button
              onClick={handleClockOut}
              disabled={clockLoading}
              style={{
                padding: '10px 20px',
                backgroundColor: '#ef4444',
                color: '#ffffff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px'
              }}
            >
              🛑 Clock Out
            </button>
          </>
        ) : null}
      </div>
    </div>
  );

  const renderHistoryTab = () => (
    <div style={styles.card}>
      <h2 style={{ margin: '0 0 20px 0', fontSize: '22px', fontWeight: '700' }}>📋 Clock History</h2>

      {errorMessage && (
        <div style={styles.errorMessage}>
          {errorMessage}
          <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>✕</button>
        </div>
      )}

      <div style={{ ...styles.card, marginBottom: '20px', display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <div>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#374151', marginBottom: '6px' }}>Start Date</label>
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px' }} />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#374151', marginBottom: '6px' }}>End Date</label>
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px' }} />
        </div>
        <button onClick={handleApplyFilter} style={{ padding: '8px 16px', backgroundColor: '#3b82f6', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '600' }}>Apply</button>
        <button onClick={handleDownloadCSV} style={{ padding: '8px 16px', backgroundColor: '#10b981', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '600' }}>Export CSV</button>
      </div>

      {historyLoading ? (
        <div style={{ textAlign: 'center', color: '#6b7280', padding: '40px' }}>Loading...</div>
      ) : clockHistory.length === 0 ? (
        <div style={{ textAlign: 'center', color: '#6b7280', padding: '40px', backgroundColor: '#f9fafb', borderRadius: '8px' }}>No records found</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
            <thead>
              <tr style={{ backgroundColor: '#f3f4f6', borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Date</th>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Clock In</th>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Clock Out</th>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Work Hours</th>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Location</th>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Approval</th>
              </tr>
            </thead>
            <tbody>
              {clockHistory.map((record, idx) => (
                <tr key={idx} style={{ backgroundColor: idx % 2 === 0 ? '#fff' : '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '12px' }}>{new Date(record.date).toLocaleDateString()}</td>
                  <td style={{ padding: '12px' }}>{record.clockInTime ? new Date(record.clockInTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--'}</td>
                  <td style={{ padding: '12px' }}>{record.clockOutTime ? new Date(record.clockOutTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--'}</td>
                  <td style={{ padding: '12px' }}>{record.totalWorkTime ? (record.totalWorkTime / 1000 / 60 / 60).toFixed(2) : '--'}h</td>
                  <td style={{ padding: '12px' }}>{record.location || 'ED'}</td>
                  <td style={{ padding: '12px' }}>
                    <span style={{
                      display: 'inline-block',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '12px',
                      fontWeight: '600',
                      backgroundColor: record.lateArrivalApprovalStatus === 'APPROVED' ? '#dcfce7' : record.lateArrivalApprovalStatus === 'PENDING' ? '#fef3c7' : '#fee2e2',
                      color: record.lateArrivalApprovalStatus === 'APPROVED' ? '#166534' : record.lateArrivalApprovalStatus === 'PENDING' ? '#92400e' : '#991b1b'
                    }}>
                      {record.lateArrivalApprovalStatus || 'N/A'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  const renderStatisticsTab = () => (
    <div style={styles.card}>
      <h2 style={{ margin: '0 0 20px 0', fontSize: '22px', fontWeight: '700' }}>📊 Performance Statistics</h2>

      {statsLoading ? (
        <div style={{ textAlign: 'center', color: '#6b7280', padding: '40px' }}>Loading statistics...</div>
      ) : stats ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div style={{ ...styles.card, textAlign: 'center' }}>
            <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Total Work Hours</p>
            <p style={{ margin: 0, fontSize: '28px', fontWeight: '700' }}>{stats.totalWorkHours?.toFixed(1) || '0'}h</p>
          </div>
          <div style={{ ...styles.card, textAlign: 'center' }}>
            <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Shifts Completed</p>
            <p style={{ margin: 0, fontSize: '28px', fontWeight: '700' }}>{stats.shiftsCompleted || '0'}</p>
          </div>
          <div style={{ ...styles.card, textAlign: 'center' }}>
            <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Attendance Rate</p>
            <p style={{ margin: 0, fontSize: '28px', fontWeight: '700' }}>{stats.attendanceRate?.toFixed(0) || '0'}%</p>
          </div>
          <div style={{ ...styles.card, textAlign: 'center' }}>
            <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Late Arrivals</p>
            <p style={{ margin: 0, fontSize: '28px', fontWeight: '700' }}>{stats.lateArrivals || '0'}</p>
          </div>
        </div>
      ) : (
        <div style={{ textAlign: 'center', color: '#6b7280', padding: '40px' }}>No statistics available</div>
      )}
    </div>
  );

  const renderScheduleTab = () => (
    <div style={styles.card}>
      <h2 style={{ margin: '0 0 20px 0', fontSize: '22px', fontWeight: '700' }}>📅 Staff Schedule</h2>
      
      <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', borderBottom: '2px solid #e5e7eb', paddingBottom: '12px', flexWrap: 'wrap' }}>
        {[
          { key: 'my-schedule', label: '📍 My Schedule' },
          { key: 'physicians', label: '👨‍⚕️ Physicians' },
          { key: 'nurses', label: '👩‍⚕️ Nurses' }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setScheduleTab(tab.key)}
            style={{
              padding: '10px 16px',
              backgroundColor: scheduleTab === tab.key ? '#1f2937' : 'transparent',
              color: scheduleTab === tab.key ? '#fff' : '#6b7280',
              border: 'none',
              borderBottom: scheduleTab === tab.key ? '3px solid #1f2937' : 'none',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div style={{ marginBottom: '20px', display: 'flex', gap: '12px', alignItems: 'center' }}>
        <label style={{ fontSize: '14px', fontWeight: '600', color: '#374151' }}>Schedule Date:</label>
        <input 
          type="date" 
          value={targetDate} 
          onChange={(e) => setTargetDate(e.target.value)}
          style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px' }}
        />
      </div>

      {scheduleTab === 'my-schedule' && (
        <div>
          {mySchedule ? (
            <div style={{ ...styles.card, backgroundColor: '#f9fafb' }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: '700' }}>{mySchedule.name}</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Date</p>
                  <p style={{ margin: 0, fontSize: '14px', fontWeight: '600' }}>{targetDate}</p>
                </div>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Status</p>
                  <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: mySchedule.status === 'ON_DUTY' ? '#10b981' : '#6b7280' }}>{mySchedule.status}</p>
                </div>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>Shift</p>
                  <p style={{ margin: 0, fontSize: '14px', fontWeight: '600' }}>{mySchedule.shiftStart} - {mySchedule.shiftEnd}</p>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ padding: '40px', textAlign: 'center', color: '#6b7280', backgroundColor: '#f9fafb', borderRadius: '12px' }}>
              No schedule found for you on {targetDate}
            </div>
          )}
        </div>
      )}

      {(scheduleTab === 'physicians' || scheduleTab === 'nurses') && (
        <div>
          {loading ? (
            <div style={{ textAlign: 'center', color: '#6b7280', padding: '40px' }}>Loading {scheduleTab}...</div>
          ) : (scheduleTab === 'physicians' ? physicians : nurses).length === 0 ? (
            <div style={{ textAlign: 'center', color: '#6b7280', padding: '40px', backgroundColor: '#f9fafb', borderRadius: '8px' }}>No {scheduleTab} found</div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f3f4f6', borderBottom: '2px solid #e5e7eb' }}>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Name</th>
                    {scheduleTab === 'physicians' && <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Specialty</th>}
                    <th style={{ padding: '12px', textAlign: 'center', fontWeight: '600', color: '#374151' }}>Status</th>
                    <th style={{ padding: '12px', textAlign: 'center', fontWeight: '600', color: '#374151' }}>Shift</th>
                  </tr>
                </thead>
                <tbody>
                  {(scheduleTab === 'physicians' ? physicians : nurses).map((member, idx) => (
                    <tr key={idx} style={{ backgroundColor: idx % 2 === 0 ? '#fff' : '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                      <td style={{ padding: '12px', fontWeight: '600' }}>{member.name} {member.surname || ''}</td>
                      {scheduleTab === 'physicians' && (
                        <td style={{ padding: '12px' }}>
                          <span style={{ display: 'inline-block', padding: '4px 8px', backgroundColor: '#f0f9ff', color: '#0369a1', borderRadius: '6px', fontSize: '12px', fontWeight: '500' }}>
                            {member.specialty}
                          </span>
                        </td>
                      )}
                      <td style={{ padding: '12px', textAlign: 'center' }}>
                        <span style={{ display: 'inline-block', padding: '4px 8px', backgroundColor: member.status === 'ON_DUTY' ? '#dcfce7' : '#fee2e2', color: member.status === 'ON_DUTY' ? '#166534' : '#991b1b', borderRadius: '6px', fontSize: '12px', fontWeight: '500' }}>
                          {member.status}
                        </span>
                      </td>
                      <td style={{ padding: '12px', textAlign: 'center', fontWeight: '600' }}>{member.shiftStart} - {member.shiftEnd}</td>
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

  const styles = {
    card: {
      backgroundColor: '#ffffff',
      borderRadius: '12px',
      padding: '20px',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      border: '1px solid #e5e7eb'
    },
    errorMessage: {
      padding: '12px 16px',
      backgroundColor: '#fee2e2',
      border: '1px solid #fecaca',
      borderRadius: '8px',
      color: '#991b1b',
      marginBottom: '20px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    successMessage: {
      padding: '12px 16px',
      backgroundColor: '#dcfce7',
      border: '1px solid #86efac',
      borderRadius: '8px',
      color: '#166534',
      marginBottom: '20px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center'
    }
  };

  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div style={{ backgroundColor: '#f9fafb', padding: '32px', minHeight: '100vh' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: '700', color: '#0f172a' }}>Staff Schedule & Clock Management</h1>
          <p style={{ margin: 0, color: '#6b7280', fontSize: '14px' }}>Manage scheduling, clock in/out, view history and track performance</p>
        </div>

        {/* Main Tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', borderBottom: '2px solid #e5e7eb', flexWrap: 'wrap' }}>
          {[
            { id: 'clock-in-out', label: '⏰ Clock In/Out' },
            { id: 'history', label: '📋 Clock History' },
            { id: 'schedule', label: '📅 Schedule' },
            { id: 'statistics', label: '📊 Statistics' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setMainTab(tab.id)}
              style={{
                padding: '12px 20px',
                backgroundColor: mainTab === tab.id ? '#3b82f6' : 'transparent',
                color: mainTab === tab.id ? '#fff' : '#6b7280',
                border: 'none',
                borderBottom: mainTab === tab.id ? '3px solid #3b82f6' : 'none',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px',
                transition: 'all 0.2s'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {mainTab === 'clock-in-out' && renderClockTab()}
        {mainTab === 'history' && renderHistoryTab()}
        {mainTab === 'schedule' && renderScheduleTab()}
        {mainTab === 'statistics' && renderStatisticsTab()}
      </div>
    </SidebarLayout>
  );
};

export default StaffSchedule;
