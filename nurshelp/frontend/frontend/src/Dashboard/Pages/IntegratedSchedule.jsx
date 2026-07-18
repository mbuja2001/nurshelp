import React, { useState, useEffect } from 'react';
import {
  FaClock, FaHistory, FaCalendarAlt, FaChartBar, FaCheckCircle,
  FaPlay, FaStop, FaDownload, FaFilter, FaSignOutAlt
} from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const styles = {
  container: {
    width: '100%',
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '20px',
    backgroundColor: '#f3f4f6',
    minHeight: '100vh'
  },
  header: {
    marginBottom: '30px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  tabs: {
    display: 'flex',
    gap: '10px',
    marginBottom: '30px',
    borderBottom: '2px solid #e5e7eb',
    flexWrap: 'wrap'
  },
  tabBtn: {
    padding: '12px 20px',
    backgroundColor: 'transparent',
    border: 'none',
    borderBottom: '3px solid transparent',
    cursor: 'pointer',
    fontSize: '15px',
    fontWeight: '600',
    color: '#6b7280',
    transition: 'all 0.2s'
  },
  tabBtnActive: {
    color: '#1f2937',
    borderBottomColor: '#3b82f6'
  },
  mainCard: {
    backgroundColor: '#ffffff',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
    marginBottom: '24px'
  },
  twoColumnGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px',
    marginBottom: '24px'
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '12px',
    padding: '20px',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
    border: '1px solid #e5e7eb'
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#3b82f6',
    color: '#ffffff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    transition: 'all 0.2s'
  },
  buttonDanger: {
    backgroundColor: '#ef4444'
  },
  buttonSecondary: {
    backgroundColor: '#6b7280'
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
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '14px'
  },
  tableHeader: {
    backgroundColor: '#f3f4f6',
    fontWeight: '600',
    color: '#374151',
    padding: '12px',
    textAlign: 'left',
    borderBottom: '2px solid #e5e7eb'
  },
  tableCell: {
    padding: '12px',
    borderBottom: '1px solid #e5e7eb'
  },
  badge: {
    display: 'inline-block',
    padding: '4px 12px',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: '600'
  },
  badgeSuccess: {
    backgroundColor: '#dcfce7',
    color: '#166534'
  },
  badgeWarning: {
    backgroundColor: '#fef3c7',
    color: '#92400e'
  },
  badgeDanger: {
    backgroundColor: '#fee2e2',
    color: '#991b1b'
  },
  loadingText: {
    textAlign: 'center',
    color: '#6b7280',
    padding: '40px'
  },
  statCard: {
    textAlign: 'center',
    padding: '20px'
  },
  statValue: {
    fontSize: '32px',
    fontWeight: '700',
    color: '#1f2937',
    margin: '10px 0'
  },
  statLabel: {
    fontSize: '14px',
    color: '#6b7280',
    fontWeight: '500'
  }
};

export default function IntegratedSchedule({ token, user }) {
  const [activeTab, setActiveTab] = useState('clock-in-out');
  const [currentRecord, setCurrentRecord] = useState(null);
  const [liveTime, setLiveTime] = useState(new Date());
  const [onBreak, setOnBreak] = useState(false);
  const [breakStartTime, setBreakStartTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Clock history state
  const [clockHistory, setClockHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // Schedule state
  const [schedule, setSchedule] = useState([]);
  const [selectedWeek, setSelectedWeek] = useState('current');

  // Stats state
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);

  // Initialize dates
  useEffect(() => {
    if (token) {
      const end = new Date();
      const start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
      setStartDate(start.toISOString().split('T')[0]);
      setEndDate(end.toISOString().split('T')[0]);
    }
  }, [token]);

  // Live clock update
  useEffect(() => {
    const interval = setInterval(() => setLiveTime(new Date()), 500);
    return () => clearInterval(interval);
  }, []);

  // Load data when tab changes
  useEffect(() => {
    if (!token) return;

    if (activeTab === 'clock-in-out') {
      fetchTodayRecord();
    } else if (activeTab === 'history') {
      fetchClockHistory(startDate, endDate);
    } else if (activeTab === 'schedule') {
      fetchSchedule();
    } else if (activeTab === 'statistics') {
      fetchStatistics();
    }
  }, [activeTab, token]);

  // ===== CLOCK IN/OUT METHODS =====
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
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/clock-in`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          location: user?.location || 'ED'
        })
      });
      const data = await res.json();
      if (data.success) {
        setCurrentRecord(data.record);
        showSuccess('✓ Clocked in successfully');
      } else {
        setError(data.message || 'Clock-in failed');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClockOut = async () => {
    setLoading(true);
    setError(null);
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
        showSuccess('✓ Clocked out successfully');
      } else {
        setError(data.message || 'Clock-out failed');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBreakStart = async () => {
    setLoading(true);
    setError(null);
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
        showSuccess('✓ Break started');
      } else {
        setError(data.message || 'Break start failed');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBreakEnd = async () => {
    setLoading(true);
    setError(null);
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
        showSuccess('✓ Break ended');
      } else {
        setError(data.message || 'Break end failed');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ===== CLOCK HISTORY METHODS =====
  const fetchClockHistory = async (start, end) => {
    setHistoryLoading(true);
    setError(null);
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
        setError(data.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleApplyFilter = () => {
    if (!startDate || !endDate) {
      setError('Please select both dates');
      return;
    }
    if (new Date(startDate) > new Date(endDate)) {
      setError('Start date must be before end date');
      return;
    }
    fetchClockHistory(startDate, endDate);
  };

  const handleDownloadCSV = () => {
    if (clockHistory.length === 0) {
      setError('No records to download');
      return;
    }

    const headers = [
      'Date', 'Clock In', 'Clock Out', 'Break Duration (min)',
      'Total Work Hours', 'Location', 'Late Arrival (min)',
      'Late Approval', 'Break Approval', 'Notes'
    ];

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

    const csv = [headers, ...rows].map(row => {
      if (Array.isArray(row)) {
        return row.map(cell => `"${cell}"`).join(',');
      }
      return row;
    }).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clock-history-${startDate}-to-${endDate}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showSuccess('✓ CSV exported successfully');
  };

  // ===== SCHEDULE METHODS =====
  const fetchSchedule = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/nurses/schedule?week=${selectedWeek}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const data = await res.json();
      if (data.success) {
        setSchedule(data.schedule || []);
      } else {
        setError(data.message || 'Failed to fetch schedule');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ===== STATISTICS METHODS =====
  const fetchStatistics = async () => {
    setStatsLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/nurses/statistics?range=monthly`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const data = await res.json();
      if (data.success) {
        setStats(data.stats);
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setStatsLoading(false);
    }
  };

  // ===== UTILITY METHODS =====
  const showSuccess = (msg) => {
    setSuccessMessage(msg);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return '--:--';
    const date = new Date(`2000-01-01T${timeStr}`);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const getDayShifts = (dayName) => {
    return schedule.filter(s => s.dayOfWeek === dayName);
  };

  const getStatusColor = (status) => {
    if (status === 'COMPLETED') return '#10b981';
    if (status === 'IN_PROGRESS') return '#3b82f6';
    if (status === 'UPCOMING') return '#f59e0b';
    return '#d1d5db';
  };

  // ===== RENDER SECTIONS =====
  const renderClockInOut = () => (
    <div style={styles.mainCard}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '24px' }}>
        <div>
          <h2 style={{ margin: '0 0 10px 0', fontSize: '24px', fontWeight: '700' }}>
            ⏰ {liveTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </h2>
          <p style={{ margin: 0, color: '#6b7280' }}>{liveTime.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}</p>
        </div>
      </div>

      {error && <div style={styles.errorMessage}>{error}<button onClick={() => setError(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b' }}>✕</button></div>}
      {successMessage && <div style={styles.successMessage}>{successMessage}<button onClick={() => setSuccessMessage(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534' }}>✕</button></div>}

      {currentRecord ? (
        <div style={styles.twoColumnGrid}>
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
                    ...styles.badge,
                    ...(currentRecord.clockOutTime ? styles.badgeSuccess : styles.badgeWarning)
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
      ) : (
        <div style={{ ...styles.card, textAlign: 'center', color: '#6b7280' }}>
          No clock record found for today
        </div>
      )}

      <div style={{ ...styles.card, display: 'flex', gap: '12px', marginTop: '20px', flexWrap: 'wrap' }}>
        {!currentRecord?.clockInTime ? (
          <button
            onClick={handleClockIn}
            disabled={loading}
            style={{
              ...styles.button,
              backgroundColor: loading ? '#9ca3af' : '#10b981',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            <FaPlay size={16} /> Clock In
          </button>
        ) : !currentRecord?.clockOutTime ? (
          <>
            <button
              onClick={onBreak ? handleBreakEnd : handleBreakStart}
              disabled={loading}
              style={{
                ...styles.button,
                backgroundColor: loading ? '#9ca3af' : '#f59e0b'
              }}
            >
              {onBreak ? '✓ Finish Break' : '☕ Start Break'}
            </button>
            <button
              onClick={handleClockOut}
              disabled={loading}
              style={{
                ...styles.button,
                ...styles.buttonDanger,
                backgroundColor: loading ? '#9ca3af' : '#ef4444'
              }}
            >
              <FaStop size={16} /> Clock Out
            </button>
          </>
        ) : null}
      </div>
    </div>
  );

  const renderClockHistory = () => (
    <div style={styles.mainCard}>
      <h2 style={{ margin: '0 0 20px 0', fontSize: '22px', fontWeight: '700' }}>📋 Clock History</h2>

      {error && <div style={styles.errorMessage}>{error}<button onClick={() => setError(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b' }}>✕</button></div>}
      {successMessage && <div style={styles.successMessage}>{successMessage}<button onClick={() => setSuccessMessage(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534' }}>✕</button></div>}

      <div style={{ ...styles.card, marginBottom: '20px', display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <div>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#374151', marginBottom: '6px' }}>Start Date</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px' }}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#374151', marginBottom: '6px' }}>End Date</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px' }}
          />
        </div>
        <button
          onClick={handleApplyFilter}
          style={{ ...styles.button, backgroundColor: '#3b82f6' }}
        >
          <FaFilter size={14} /> Apply
        </button>
        <button
          onClick={handleDownloadCSV}
          disabled={clockHistory.length === 0}
          style={{ ...styles.button, backgroundColor: '#10b981', cursor: clockHistory.length === 0 ? 'not-allowed' : 'pointer' }}
        >
          <FaDownload size={14} /> Export CSV
        </button>
      </div>

      {historyLoading ? (
        <div style={styles.loadingText}>Loading clock history...</div>
      ) : clockHistory.length === 0 ? (
        <div style={{ ...styles.card, textAlign: 'center', color: '#6b7280' }}>
          No clock records found for the selected date range
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={styles.table}>
            <thead>
              <tr style={{ backgroundColor: '#f3f4f6', borderBottom: '2px solid #e5e7eb' }}>
                <th style={styles.tableHeader}>Date</th>
                <th style={styles.tableHeader}>Clock In</th>
                <th style={styles.tableHeader}>Clock Out</th>
                <th style={styles.tableHeader}>Work Hours</th>
                <th style={styles.tableHeader}>Location</th>
                <th style={styles.tableHeader}>Approval Status</th>
              </tr>
            </thead>
            <tbody>
              {clockHistory.map((record, idx) => (
                <tr key={idx} style={{ backgroundColor: idx % 2 === 0 ? '#fff' : '#f9fafb' }}>
                  <td style={styles.tableCell}>{new Date(record.date).toLocaleDateString()}</td>
                  <td style={styles.tableCell}>
                    {record.clockInTime ? new Date(record.clockInTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--'}
                  </td>
                  <td style={styles.tableCell}>
                    {record.clockOutTime ? new Date(record.clockOutTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : '--:--'}
                  </td>
                  <td style={styles.tableCell}>
                    {record.totalWorkTime ? (record.totalWorkTime / 1000 / 60 / 60).toFixed(2) : '--'}h
                  </td>
                  <td style={styles.tableCell}>{record.location || 'ED'}</td>
                  <td style={styles.tableCell}>
                    <span style={{
                      ...styles.badge,
                      ...(record.lateArrivalApprovalStatus === 'APPROVED' ? styles.badgeSuccess :
                          record.lateArrivalApprovalStatus === 'PENDING' ? styles.badgeWarning :
                          styles.badgeDanger)
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

  const renderSchedule = () => (
    <div style={styles.mainCard}>
      <h2 style={{ margin: '0 0 20px 0', fontSize: '22px', fontWeight: '700' }}>📅 My Schedule</h2>

      <div style={{ ...styles.card, marginBottom: '20px', display: 'flex', gap: '12px' }}>
        <button
          onClick={() => {
            setSelectedWeek('current');
            setLoading(true);
          }}
          style={{
            ...styles.button,
            backgroundColor: selectedWeek === 'current' ? '#3b82f6' : '#e5e7eb',
            color: selectedWeek === 'current' ? '#fff' : '#1f2937'
          }}
        >
          This Week
        </button>
        <button
          onClick={() => {
            setSelectedWeek('next');
            setLoading(true);
          }}
          style={{
            ...styles.button,
            backgroundColor: selectedWeek === 'next' ? '#3b82f6' : '#e5e7eb',
            color: selectedWeek === 'next' ? '#fff' : '#1f2937'
          }}
        >
          Next Week
        </button>
      </div>

      {loading ? (
        <div style={styles.loadingText}>Loading schedule...</div>
      ) : schedule.length === 0 ? (
        <div style={{ ...styles.card, textAlign: 'center', color: '#6b7280' }}>
          No shifts scheduled
        </div>
      ) : (
        <div style={styles.twoColumnGrid}>
          {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map(day => {
            const dayShifts = getDayShifts(day);
            return (
              <div key={day} style={styles.card}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '700' }}>{day}</h3>
                {dayShifts.length === 0 ? (
                  <p style={{ color: '#6b7280', margin: 0 }}>No shifts</p>
                ) : (
                  dayShifts.map((shift, idx) => (
                    <div key={idx} style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: idx === dayShifts.length - 1 ? 'none' : '1px solid #e5e7eb' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                        <div>
                          <p style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: '600' }}>
                            {shift.shiftStart} - {shift.shiftEnd}
                          </p>
                          <p style={{ margin: '0 0 4px 0', fontSize: '13px', color: '#6b7280' }}>
                            📍 {shift.location || 'ED'}
                          </p>
                        </div>
                        <span style={{
                          ...styles.badge,
                          backgroundColor: getStatusColor(shift.status),
                          color: '#fff'
                        }}>
                          {shift.status === 'COMPLETED' ? '✓' : shift.status === 'IN_PROGRESS' ? '⏳' : '📅'}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  const renderStatistics = () => (
    <div style={styles.mainCard}>
      <h2 style={{ margin: '0 0 20px 0', fontSize: '22px', fontWeight: '700' }}>📊 Performance Statistics</h2>

      {statsLoading ? (
        <div style={styles.loadingText}>Loading statistics...</div>
      ) : stats ? (
        <div>
          <div style={styles.twoColumnGrid}>
            <div style={{ ...styles.card, textAlign: 'center' }}>
              <FaClock size={32} style={{ color: '#3b82f6', margin: '10px 0' }} />
              <p style={styles.statLabel}>Total Work Hours</p>
              <p style={styles.statValue}>{stats.totalWorkHours?.toFixed(1) || '0'}h</p>
            </div>
            <div style={{ ...styles.card, textAlign: 'center' }}>
              <FaCheckCircle size={32} style={{ color: '#10b981', margin: '10px 0' }} />
              <p style={styles.statLabel}>Shifts Completed</p>
              <p style={styles.statValue}>{stats.shiftsCompleted || '0'}</p>
            </div>
            <div style={{ ...styles.card, textAlign: 'center' }}>
              <FaChartBar size={32} style={{ color: '#f59e0b', margin: '10px 0' }} />
              <p style={styles.statLabel}>Attendance Rate</p>
              <p style={styles.statValue}>{stats.attendanceRate?.toFixed(0) || '0'}%</p>
            </div>
            <div style={{ ...styles.card, textAlign: 'center' }}>
              <FaCalendarAlt size={32} style={{ color: '#ef4444', margin: '10px 0' }} />
              <p style={styles.statLabel}>Late Arrivals</p>
              <p style={styles.statValue}>{stats.lateArrivals || '0'}</p>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ ...styles.card, textAlign: 'center', color: '#6b7280' }}>
          No statistics available
        </div>
      )}
    </div>
  );

  return (
    <div style={styles.container}>
      < div style={styles.header}>
        <div>
          <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700' }}>⏱️ Integrated Schedule & Clock</h1>
          <p style={{ margin: '5px 0 0 0', color: '#6b7280' }}>Manage your schedule, clock, and performance</p>
        </div>
      </div>

      <div style={styles.tabs}>
        {[
          { id: 'clock-in-out', label: 'Clock In/Out', icon: FaClock },
          { id: 'history', label: 'Clock History', icon: FaHistory },
          { id: 'schedule', label: 'My Schedule', icon: FaCalendarAlt },
          { id: 'statistics', label: 'Statistics', icon: FaChartBar }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              ...styles.tabBtn,
              ...(activeTab === tab.id ? styles.tabBtnActive : {})
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'clock-in-out' && renderClockInOut()}
      {activeTab === 'history' && renderClockHistory()}
      {activeTab === 'schedule' && renderSchedule()}
      {activeTab === 'statistics' && renderStatistics()}
    </div>
  );
}
