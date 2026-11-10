// src/Dashboard/Pages/ClockInOutBoard.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaClock, FaPlay, FaStop, FaCoffee, FaSignOutAlt, FaHistory } from 'react-icons/fa';
import { clearAuthSession } from '../../utils/authSession';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function ClockInOutBoard({ token, user, onLogout, activeTab: propActiveTab, onClockStatusChange }) {
  const navigate = useNavigate();
  const [currentRecord, setCurrentRecord] = useState(null);
  // activeTab is controlled by parent (StaffScheduleManager). Fallback to 'clock'.
  const activeTab = propActiveTab || 'clock';
  const [assignedShifts, setAssignedShifts] = useState([]);
  const [shiftsLoading, setShiftsLoading] = useState(false);
  const [selectedShift, setSelectedShift] = useState(null);
  const [liveTime, setLiveTime] = useState(null);
  const [onBreak, setOnBreak] = useState(false);
  const [breakStartTime, setBreakStartTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [todayShift, setTodayShift] = useState(null);
  const [canClockIn, setCanClockIn] = useState(false);
  const [clockInBlockReason, setClockInBlockReason] = useState('');
  const [attendanceHistory, setAttendanceHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [profile, setProfile] = useState(user || null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState('');

  useEffect(() => {
    fetchTodayRecord();
    fetchTodayShift();
    fetchAssignedShifts(); // Load all shifts for selection
    fetchAttendanceHistory(); // Load history to check if shifts already covered
    
    // Immediate auto clock-out check
    setTimeout(() => checkAndRefreshAutoClockOut(), 500);
    
    const interval = setInterval(() => setLiveTime(new Date()), 500);
    
    // Add periodic check for auto clock-out (every 1 minute)
    const autoClockOutCheckInterval = setInterval(() => {
      checkAndRefreshAutoClockOut();
    }, 60000);
    
    return () => {
      clearInterval(interval);
      clearInterval(autoClockOutCheckInterval);
    };
  }, [token]);

  useEffect(() => {
    if (activeTab === 'myshifts') fetchAssignedShifts();
    if (activeTab === 'profile') fetchProfile();
    if (activeTab === 'history') {
      fetchAttendanceHistory();
      // Refresh attendance history every 30 seconds to catch new absences
      const historyInterval = setInterval(() => fetchAttendanceHistory(), 30000);
      return () => clearInterval(historyInterval);
    }
    // When on clock tab, also check for auto clock-out more frequently
    if (activeTab === 'clock') {
      const autoClockOutCheckInterval = setInterval(() => {
        checkAndRefreshAutoClockOut();
      }, 30000); // Check every 30 seconds when on clock tab
      return () => clearInterval(autoClockOutCheckInterval);
    }
  }, [activeTab]);

  const fetchProfile = async () => {
    if (!token) return;
    setProfileLoading(true);
    setProfileError('');
    try {
      const response = await fetch(`${BACKEND_URL}/api/staff/profile`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load profile (HTTP ${response.status})`);
      setProfile(data);
    } catch (fetchError) {
      console.error('Failed to fetch staff profile', fetchError);
      setProfileError(fetchError.message || 'Unable to load your profile.');
    } finally {
      setProfileLoading(false);
    }
  };

  // Check if current time is within 5 mins before shift start to end of shift
  useEffect(() => {
    if (todayShift && liveTime) {
      const now = new Date();
      const shiftStart = new Date(todayShift.startTime);
      const shiftEnd = new Date(todayShift.endTime);
      const fiveMinsBeforeStart = new Date(shiftStart.getTime() - 5 * 60000);

      if (now < fiveMinsBeforeStart) {
        setCanClockIn(false);
        const minsUntilShift = Math.ceil((fiveMinsBeforeStart - now) / 60000);
        setClockInBlockReason(`Clock-in available in ${minsUntilShift} min`);
      } else if (now > shiftEnd) {
        setCanClockIn(false);
        setClockInBlockReason('Shift has ended');
      } else {
        setCanClockIn(true);
        setClockInBlockReason('');
      }
    } else if (!todayShift) {
      setCanClockIn(false);
      setClockInBlockReason('No shift assigned for today');
    }
  }, [todayShift, liveTime]);

  const fetchAssignedShifts = async () => {
    if (!token) return;
    setShiftsLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/shifts/assigned`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setAssignedShifts(data.shifts || []);
      } else {
        setAssignedShifts([]);
      }
    } catch (err) {
      console.error('Failed to fetch assigned shifts', err);
      setAssignedShifts([]);
    } finally {
      setShiftsLoading(false);
    }
  };
  const fetchTodayShift = async () => {
    if (!token) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/shifts/assigned/today`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success && data.shift) {
        setTodayShift(data.shift);
      } else {
        setTodayShift(null);
      }
    } catch (err) {
      console.error('Failed to fetch today shift', err);
      setTodayShift(null);
    }
  };

  const fetchAttendanceHistory = async () => {
    if (!token) return;
    setHistoryLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/attendance-history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!res.headers.get('content-type')?.includes('application/json')) {
        throw new Error(`Attendance history returned an unexpected response (HTTP ${res.status})`);
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || `Unable to load attendance history (HTTP ${res.status})`);
      if (data.success) {
        setAttendanceHistory(data.records || []);
      } else {
        throw new Error(data.message || 'Unable to load attendance history');
      }
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch attendance history', err);
      setAttendanceHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  // Check if a shift has already been completed by the nurse today
  const hasAlreadyCovered = (shiftId) => {
    if (!shiftId) return false;
    return attendanceHistory.some(record => {
      const recordShiftId = typeof record.shiftId === 'object' ? record.shiftId?._id : record.shiftId;
      return recordShiftId === shiftId && record.status === 'CLOCKED_OUT';
    });
  };

  // Get shift details by ID from assigned shifts array
  const getShiftDetails = (shiftId) => {
    if (!shiftId) return null;
    // If already an object with shiftName, return it
    if (typeof shiftId === 'object' && shiftId.shiftName) {
      return shiftId;
    }
    // If it's a string (ID), look it up in assignedShifts
    const id = typeof shiftId === 'object' ? shiftId._id : shiftId;
    return assignedShifts.find(s => s._id === id);
  };

  const handleLogout = () => {
    clearAuthSession();
    if (onLogout) onLogout();
    navigate('/dashboard?login=1', { replace: true });
  };

  // Check if staff can clock in to a shift
  const getShiftEligibility = (shift) => {
    if (!liveTime) return { eligible: false, reason: 'Checking time...' };
    
    // Check if already covered
    if (hasAlreadyCovered(shift._id)) {
      return { eligible: false, reason: `✓ Already completed`, color: '#9ca3af' };
    }
    
    const now = new Date();
    let shiftStart = new Date(shift.startTime || shift.date);
    let shiftEnd = new Date(shift.endTime || new Date(shift.date).getTime() + 8 * 60 * 60 * 1000);
    
    // Parse shift start time if it's in HH:MM format
    if (shift.shiftStart) {
      const [hours, minutes] = shift.shiftStart.split(':').map(Number);
      shiftStart = new Date(shiftStart);
      shiftStart.setHours(hours, minutes, 0, 0);
    }
    
    // Parse shift end time
    if (shift.shiftEnd) {
      const [hours, minutes] = shift.shiftEnd.split(':').map(Number);
      shiftEnd = new Date(shiftEnd);
      shiftEnd.setHours(hours, minutes, 0, 0);
    }

    const fiveMinsBeforeStart = new Date(shiftStart.getTime() - 5 * 60000);
    const tenMinsAfterStart = new Date(shiftStart.getTime() + 10 * 60000);

    // Shift has ended - cannot clock in
    if (now > shiftEnd) {
      return { eligible: false, reason: `❌ Shift ended`, color: '#ef4444' };
    }

    // Shift not started yet
    if (now < fiveMinsBeforeStart) {
      const minsUntil = Math.ceil((fiveMinsBeforeStart - now) / 60000);
      return { eligible: false, reason: `🔒 In ${minsUntil}m`, color: '#6b7280' };
    }

    // More than 10 mins late
    if (now > tenMinsAfterStart) {
      return { eligible: true, reason: `⚠️ LATE - Flag will apply`, color: '#ef4444' };
    }

    // Normal window - ready to clock in
    return { eligible: true, reason: `✅ Ready`, color: '#10b981' };
  };

  const fetchTodayRecord = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/today-record`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        const record = data.record || null;
        setCurrentRecord(record);
        // Set onBreak based on status field, not onBreak property
        setOnBreak(record?.status === 'ON_BREAK');
        onClockStatusChange?.(record);
        console.log('[fetchTodayRecord] Record updated:', {
          status: record?.status,
          onBreak: record?.status === 'ON_BREAK',
          clockInTime: record?.clockInTime,
          breakStartTime: record?.breakStartTime
        });
        return record;
      }
    } catch (err) {
      console.error('Error fetching today record:', err);
    }
    return null;
  };

  const checkAndRefreshAutoClockOut = async () => {
    try {
      // Refresh current record to check if auto clock-out has happened
      const res = await fetch(`${BACKEND_URL}/api/staff/today-record`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      
      if (data.success && data.record) {
        const record = data.record;
        
        // Check if they were just auto-clocked out
        if (record.status === 'CLOCKED_OUT' && record.flags && record.flags.includes('AUTO_CLOCKOUT')) {
          setCurrentRecord(record);
          setOnBreak(false);
          
          // Show success message about auto clock-out
          setSuccessMessage('✅ You have been automatically clocked out as your shift has ended.');
          setTimeout(() => setSuccessMessage(null), 5000);
          
          console.log('[AUTO CLOCK OUT CHECK] User was auto-clocked out:', {
            clockOutTime: record.clockOutTime,
            flags: record.flags
          });
        }
      }
    } catch (err) {
      console.error('Error checking auto clock-out status:', err);
    }
  };

  const handleClockIn = async () => {
    setLoading(true);
    setError(null);
    
    console.log('[CLOCK IN] Starting clock-in process');
    console.log('  Selected Shift:', selectedShift);
    console.log('  User Location:', user?.location);
    
    if (!selectedShift) {
      setError('Please select a shift to clock in to');
      setLoading(false);
      return;
    }

    try {
      const clockInData = {
        shiftId: selectedShift._id,
        location: user?.location || 'ED'
      };
      
      console.log('  📤 Sending:', JSON.stringify(clockInData, null, 2));
      
      const res = await fetch(`${BACKEND_URL}/api/staff/clock-in`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(clockInData)
      });
      
      console.log('  Response Status:', res.status);
      console.log('  Response Headers:', res.headers);
      
      // Get response as text first
      const responseText = await res.text();
      console.log('  Raw Response Text:', responseText);
      
      // Try to parse as JSON
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseErr) {
        console.error('  ❌ JSON Parse Error:', parseErr);
        console.error('  Response was not valid JSON. Raw response:');
        console.error(responseText);
        setError(`Server error: ${responseText}`);
        setLoading(false);
        return;
      }
      
      console.log('  📥 Parsed Response:', data);
      
      if (data.success) {
        await fetchTodayRecord();
        showSuccess('✓ Clocked in successfully');
      } else {
        setError(data.message || 'Clock-in failed');
      }
    } catch (err) {
      console.error('  ❌ Fetch Error:', err);
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
        await fetchTodayRecord(); // Fetch full record from DB
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
        await fetchTodayRecord(); // Fetch full record from DB
        setOnBreak(true);
        setBreakStartTime(new Date());
        showSuccess('☕ Break started');
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
        },
        body: JSON.stringify({
          breakDuration: breakStartTime ? Math.floor((new Date() - breakStartTime) / 1000 / 60) : 0
        })
      });
      const data = await res.json();
      if (data.success) {        await fetchTodayRecord(); // Fetch full record from DB        setOnBreak(false);
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

  const showSuccess = (message) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const formatTime = (date) => {
    if (!date) return '--:--:--';
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatDuration = (milliseconds) => {
    if (!milliseconds) return '0h 0m';
    const totalMinutes = Math.floor(milliseconds / 1000 / 60);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return `${hours}h ${minutes}m`;
  };

  const isClockedIn = currentRecord && currentRecord.clockInTime && !currentRecord.clockOutTime;

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px' }}>
      {/* Success/Error Messages (Settings-style) */}
      {error && (
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
          border: '1px solid #fca5a5'
        }}>
          <span>{error}</span>
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b', fontSize: 18 }}>×</button>
        </div>
      )}
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
          border: '1px solid #a7f3d0'
        }}>
          <span>{successMessage}</span>
          <button onClick={() => setSuccessMessage(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534', fontSize: 18 }}>×</button>
        </div>
      )}

      {activeTab === 'profile' && (
        <section aria-labelledby="staff-profile-heading" style={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 24, boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', marginBottom: 20 }}>
            <div>
              <h2 id="staff-profile-heading" style={{ margin: '0 0 6px', color: '#0f172a', fontSize: 20 }}>Your Profile</h2>
              <p style={{ margin: 0, color: '#64748b', fontSize: 14 }}>Personal information for the signed-in staff account.</p>
            </div>
            <button type="button" onClick={handleLogout} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '9px 13px', border: '1px solid #fecaca', borderRadius: 8, background: '#fff7f7', color: '#b91c1c', fontWeight: 700, cursor: 'pointer' }}>
              <FaSignOutAlt aria-hidden="true" /> Sign out
            </button>
          </div>
          {profileError && <div role="alert" style={{ marginBottom: 16, padding: 12, border: '1px solid #fca5a5', borderRadius: 8, background: '#fef2f2', color: '#991b1b' }}>{profileError}</div>}
          {profileLoading ? (
            <div role="status" style={{ padding: '28px 0', color: '#64748b' }}>Loading your profile...</div>
          ) : profile ? (
            <dl style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 12, margin: 0 }}>
              {[
                ['Name', [profile.firstName, profile.lastName].filter(Boolean).join(' ') || user?.name || '—'],
                ['Email', profile.email || user?.email || '—'],
                ['Role', profile.positionTitle || profile.role || user?.role || '—'],
                ['Department', profile.department || '—'],
                ['Specialty', profile.specialty || '—'],
                ['Employee ID', profile.employeeId || '—'],
                ['Staff ID', profile.idNumber || '—'],
              ].map(([label, value]) => (
                <div key={label} style={{ padding: 14, border: '1px solid #e2e8f0', borderRadius: 8, background: '#f8fafc' }}>
                  <dt style={{ marginBottom: 5, color: '#64748b', fontSize: 12, fontWeight: 700 }}>{label}</dt>
                  <dd style={{ margin: 0, color: '#0f172a', fontSize: 14, fontWeight: 600, overflowWrap: 'anywhere' }}>{value}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <div style={{ padding: '28px 0', color: '#64748b' }}>Profile information is not available.</div>
          )}
        </section>
      )}

      {/* Clock View */}
      {activeTab === 'clock' && (
        <div>
          {/* Main Clock Card */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: 12,
            padding: 32,
            border: '1px solid #e5e7eb',
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
            marginBottom: 24,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 360px), 1fr))',
            gap: 40,
            alignItems: 'center'
          }}>
            {/* Live Time Display */}
            <div>
              <p style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Current Time</p>
              <div style={{ fontSize: 56, fontWeight: 800, fontFamily: '"Courier New", monospace', color: '#0f172a', letterSpacing: '2px', fontVariantNumeric: 'tabular-nums' }}>
                {liveTime && formatTime(liveTime)}
              </div>
              <p style={{ margin: '16px 0 0 0', fontSize: 13, color: '#6b7280' }}>Updates every second</p>
            </div>

            {/* Summary Metrics Grid */}
            <div>
              <p style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Today's Metrics</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div style={{ backgroundColor: '#f9fafb', padding: 16, borderRadius: 8, border: '1px solid #e5e7eb' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 600, color: '#6b7280' }}>Clock In</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a', fontFamily: '"Courier New", monospace', fontVariantNumeric: 'tabular-nums' }}>
                    {currentRecord?.clockInTime ? new Date(currentRecord.clockInTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : '--:--'}
                  </p>
                </div>

                <div style={{ backgroundColor: '#f9fafb', padding: 16, borderRadius: 8, border: '1px solid #e5e7eb' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 600, color: '#6b7280' }}>Clock Out</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a', fontFamily: '"Courier New", monospace', fontVariantNumeric: 'tabular-nums' }}>
                    {currentRecord?.clockOutTime ? new Date(currentRecord.clockOutTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : (onBreak ? '☕ On Break' : '---')}
                  </p>
                </div>

                <div style={{ backgroundColor: '#f9fafb', padding: 16, borderRadius: 8, border: '1px solid #e5e7eb' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 600, color: '#6b7280' }}>Work Time</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#10b981' }}>{formatDuration(currentRecord?.totalWorkTime)}</p>
                </div>

                <div style={{ backgroundColor: '#f9fafb', padding: 16, borderRadius: 8, border: '1px solid #e5e7eb' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 600, color: '#6b7280' }}>Break Time</p>
                  <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#f59e0b' }}>{formatDuration(currentRecord?.totalBreakTime)}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Shift End Time Warning - Auto Clock Out Info */}
          {isClockedIn && currentRecord?.shiftId && (
            <div style={{
              marginBottom: 24,
              backgroundColor: currentRecord.status === 'CLOCKED_OUT' && currentRecord.flags?.includes('AUTO_CLOCKOUT') ? '#dcfce7' : '#fef3c7',
              border: currentRecord.status === 'CLOCKED_OUT' && currentRecord.flags?.includes('AUTO_CLOCKOUT') ? '2px solid #16a34a' : '2px solid #d97706',
              borderRadius: 8,
              padding: 16
            }}>
              {currentRecord.status === 'CLOCKED_OUT' && currentRecord.flags?.includes('AUTO_CLOCKOUT') ? (
                <div>
                  <p style={{ margin: '0 0 8px 0', fontSize: 14, fontWeight: 700, color: '#166534' }}>✅ Auto Clock-Out Complete</p>
                  <p style={{ margin: 0, fontSize: 13, color: '#4d7c0f' }}>
                    You were automatically clocked out at {new Date(currentRecord.clockOutTime).toLocaleTimeString()} as your shift has ended.
                  </p>
                </div>
              ) : (
                <div>
                  <p style={{ margin: '0 0 8px 0', fontSize: 14, fontWeight: 700, color: '#92400e' }}>⏰ Shift End Time</p>
                  <p style={{ margin: 0, fontSize: 13, color: '#b45309' }}>
                    Your shift ends at {typeof currentRecord.shiftId === 'object' && currentRecord.shiftId.shiftEnd ? currentRecord.shiftId.shiftEnd : '---'}. 
                    You will be automatically clocked out when your shift time expires.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Main Action Buttons */}
          <div style={{ marginBottom: 24 }}>
            {!isClockedIn ? (
              <div>
                {/* SHIFT SELECTION */}
                <div style={{ marginBottom: 16, backgroundColor: '#f0fdf4', border: '2px solid #22c55e', borderRadius: 8, padding: 16 }}>
                  <p style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 700, color: '#15803d', textTransform: 'uppercase' }}>📍 Select Shift to Clock In</p>
                  
                  {assignedShifts.length === 0 ? (
                    <div style={{ color: '#6b7280', fontSize: 14, padding: 12 }}>No shifts assigned for today</div>
                  ) : (
                    <div style={{ display: 'grid', gap: 8 }}>
                      {assignedShifts.map((shift) => {
                        const isSelected = selectedShift?._id === shift._id;
                        const { eligible, reason, color } = getShiftEligibility(shift);
                        const alreadyCovered = hasAlreadyCovered(shift._id);
                        const canSelect = eligible && !alreadyCovered;
                        const shiftTime = `${shift.shiftStart || '?'} – ${shift.shiftEnd || '?'}`;
                        
                        return (
                          <button
                            key={shift._id}
                            onClick={() => canSelect && setSelectedShift(isSelected ? null : shift)}
                            disabled={!canSelect}
                            style={{
                              width: '100%',
                              padding: 12,
                              border: isSelected ? '2px solid #22c55e' : '2px solid #dbeafe',
                              backgroundColor: isSelected ? '#f0fdf4' : (canSelect ? '#eff6ff' : '#f3f4f6'),
                              borderRadius: 6,
                              cursor: canSelect ? 'pointer' : 'not-allowed',
                              textAlign: 'left',
                              transition: 'all 0.2s',
                              opacity: canSelect ? 1 : 0.6
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', gap: 12 }}>
                              <div>
                                <p style={{ margin: '0 0 4px 0', fontSize: 14, fontWeight: 700, color: '#0f172a' }}>
                                  {shift.shiftName || 'Shift'} • {shift.department || 'ED'}
                                </p>
                                <p style={{ margin: 0, fontSize: 12, color: '#6b7280' }}>{shiftTime}</p>
                              </div>
                              <div style={{ fontSize: 12, fontWeight: 600, color: color, whiteSpace: 'nowrap' }}>
                                {reason}
                              </div>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* CLOCK IN BUTTON */}
                <div style={{ position: 'relative' }}>
                  <button
                    onClick={handleClockIn}
                    disabled={loading || !selectedShift}
                    title={selectedShift ? 'Click to clock in' : 'Please select a shift first'}
                    style={{
                      width: '100%',
                      padding: '16px 24px',
                      backgroundColor: loading || !selectedShift ? '#cbd5e1' : '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: 8,
                      cursor: loading || !selectedShift ? 'not-allowed' : 'pointer',
                      fontWeight: 700,
                      fontSize: 16,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 12,
                      transition: 'all 0.2s',
                      opacity: loading || !selectedShift ? 0.7 : 1
                    }}
                  >
                    <FaPlay size={18} />
                    {selectedShift ? 'Clock In Now' : 'Select Shift to Clock In'}
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {!onBreak ? (
                  <>
                    <button
                      onClick={handleBreakStart}
                      disabled={loading}
                      style={{
                        padding: '16px 24px',
                        backgroundColor: loading ? '#d1d5db' : '#f59e0b',
                        color: 'white',
                        border: 'none',
                        borderRadius: 8,
                        cursor: loading ? 'default' : 'pointer',
                        fontWeight: 600,
                        fontSize: 15,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 8,
                        transition: 'all 0.2s'
                      }}
                    >
                      <FaCoffee size={16} />
                      Start Break
                    </button>
                    <button
                      onClick={handleClockOut}
                      disabled={loading}
                      style={{
                        padding: '16px 24px',
                        backgroundColor: loading ? '#d1d5db' : '#ef4444',
                        color: 'white',
                        border: 'none',
                        borderRadius: 8,
                        cursor: loading ? 'default' : 'pointer',
                        fontWeight: 700,
                        fontSize: 15,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 8,
                        transition: 'all 0.2s'
                      }}
                    >
                      <FaStop size={16} />
                      Clock Out
                    </button>
                  </>
                ) : (
                  <button
                    onClick={handleBreakEnd}
                    disabled={loading}
                    style={{
                      gridColumn: '1 / -1',
                      padding: '16px 24px',
                      backgroundColor: loading ? '#d1d5db' : '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: 8,
                      cursor: loading ? 'default' : 'pointer',
                      fontWeight: 700,
                      fontSize: 15,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 8,
                      transition: 'all 0.2s'
                    }}
                  >
                    <FaPlay size={16} />
                    End Break
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Break Status Alert */}
          {onBreak && breakStartTime && (
            <div style={{
              backgroundColor: '#fef3c7',
              border: '1px solid #fcd34d',
              borderRadius: 8,
              padding: 16,
              marginBottom: 24,
              display: 'flex',
              gap: 12,
              alignItems: 'flex-start'
            }}>
              <div style={{ fontSize: 24, marginTop: 2 }}>☕</div>
              <div style={{ flex: 1 }}>
                <p style={{ margin: '0 0 4px 0', fontWeight: 700, color: '#92400e', fontSize: 14 }}>You are currently on break</p>
                <p style={{ margin: 0, color: '#b45309', fontSize: 13 }}>Break duration: {formatDuration((new Date() - breakStartTime))}</p>
              </div>
            </div>
          )}

          {/* Department Info */}
          {isClockedIn && (
            <div style={{
              backgroundColor: '#f9fafb',
              borderRadius: 8,
              padding: 16,
              border: '1px solid #e5e7eb'
            }}>
              <p style={{ margin: '0 0 12px 0', fontSize: 14, fontWeight: 600, color: '#6b7280' }}>Current Assignment</p>
              {(() => {
                const shiftDetails = getShiftDetails(currentRecord?.shiftId);
                const shiftName = shiftDetails?.shiftName || (typeof currentRecord?.shiftId === 'object' ? currentRecord?.shiftId?.shiftName : 'Loading...');
                const department = shiftDetails?.department || (typeof currentRecord?.shiftId === 'object' ? currentRecord?.shiftId?.department : 'ED');
                
                return (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: 12, color: '#6b7280' }}>Shift Name</p>
                      <p style={{ margin: 0, fontSize: 15, fontWeight: 600, color: '#0f172a' }}>
                        {shiftName}
                      </p>
                    </div>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: 12, color: '#6b7280' }}>Location</p>
                      <p style={{ margin: 0, fontSize: 15, fontWeight: 600, color: '#0f172a' }}>{currentRecord?.location || 'ED'}</p>
                    </div>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: 12, color: '#6b7280' }}>Department</p>
                      <p style={{ margin: 0, fontSize: 15, fontWeight: 600, color: '#0f172a' }}>
                        {department}
                      </p>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* My Shifts View (Settings-style card) */}
      {activeTab === 'myshifts' && (
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: 12,
          padding: 24,
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)'
        }}>
          <h2 style={{ margin: '0 0 8px 0', fontSize: 20, fontWeight: 700, color: '#0f172a' }}>Your Assigned Shifts</h2>
          <p style={{ margin: '0 0 24px 0', fontSize: 14, color: '#6b7280' }}>Upcoming shifts scheduled for you</p>

          {shiftsLoading ? (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>Loading shifts...</div>
          ) : assignedShifts.length === 0 ? (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>No upcoming assigned shifts.</div>
          ) : (
            <div style={{ display: 'grid', gap: 12 }}>
              {assignedShifts.map(s => (
                <div
                  key={s._id}
                  style={{
                    padding: 16,
                    borderRadius: 8,
                    backgroundColor: '#f9fafb',
                    border: '1px solid #e5e7eb',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f0fdf4'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
                >
                  <div>
                    <p style={{ margin: '0 0 6px 0', fontSize: 15, fontWeight: 700, color: '#0f172a' }}>
                      {s.shiftName}
                    </p>
                    <p style={{ margin: 0, fontSize: 13, color: '#475569' }}>
                      {s.department} • {new Date(s.date).toLocaleDateString()} • {s.shiftStart}–{s.shiftEnd}
                    </p>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <p style={{
                      margin: '0 0 4px 0',
                      fontSize: 13,
                      fontWeight: 700,
                      color: s.status === 'PUBLISHED' ? '#065f46' : '#92400e',
                      backgroundColor: s.status === 'PUBLISHED' ? '#dcfce7' : '#fef3c7',
                      padding: '4px 12px',
                      borderRadius: 4,
                      display: 'inline-block'
                    }}>
                      {s.status}
                    </p>
                    <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#64748b' }}>Assigned</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Attendance History View */}
      {activeTab === 'history' && (
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: 12,
          padding: 24,
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)'
        }}>
          <h2 style={{ margin: '0 0 8px 0', fontSize: 20, fontWeight: 700, color: '#0f172a' }}>Attendance History</h2>
          <p style={{ margin: '0 0 24px 0', fontSize: 14, color: '#6b7280' }}>All your clocking actions and timestamps</p>

          {historyLoading ? (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>Loading history...</div>
          ) : attendanceHistory.length === 0 ? (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>No attendance records yet.</div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse'
              }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                    <th style={{ padding: '12px', textAlign: 'left', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Shift Name</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Date</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Clock In</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Clock Out</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Work Duration</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Status</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontSize: 12, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {attendanceHistory.map((record, idx) => (
                    <tr key={record._id || idx} style={{ borderBottom: '1px solid #e5e7eb' }}>
                      <td style={{ padding: '12px', fontSize: 14, color: '#0f172a', fontWeight: 600 }}>
                        <div>
                          <div>{record.shiftName || 'Unknown'}</div>
                          <div style={{ fontSize: 12, color: '#6b7280', fontWeight: 400 }}>{record.department || 'N/A'}</div>
                        </div>
                      </td>
                      <td style={{ padding: '12px', fontSize: 14, color: '#0f172a', fontWeight: 500 }}>{record.date}</td>
                      <td style={{ padding: '12px', fontSize: 14, color: '#0f172a', fontFamily: '"Courier New", monospace' }}>{record.clockInTime}</td>
                      <td style={{ padding: '12px', fontSize: 14, color: '#0f172a', fontFamily: '"Courier New", monospace' }}>{record.clockOutTime}</td>
                      <td style={{ padding: '12px', fontSize: 14, color: '#10b981', fontWeight: 600 }}>{record.workDuration}</td>
                      <td style={{ padding: '12px' }}>
                        <span style={{
                          display: 'inline-block',
                          padding: '4px 8px',
                          borderRadius: 4,
                          fontSize: 12,
                          fontWeight: 600,
                          backgroundColor: 
                            record.status === 'ABSENT' ? '#fee2e2' :
                            record.status === 'CLOCKED_OUT' ? '#dcfce7' : 
                            record.status === 'CLOCKED_IN' ? '#dbeafe' : '#fef3c7',
                          color: 
                            record.status === 'ABSENT' ? '#991b1b' :
                            record.status === 'CLOCKED_OUT' ? '#065f46' : 
                            record.status === 'CLOCKED_IN' ? '#0c4a6e' : '#92400e'
                        }}>
                          {record.status === 'ABSENT' ? '❌ ABSENT' : record.status}
                        </span>
                      </td>
                      <td style={{ padding: '12px' }}>
                        {record.flags && record.flags.length > 0 ? (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {record.flags.map((flag, i) => {
                              const isAbsent = flag === 'ABSENT';
                              const isDidntClockout = flag === 'DIDNT_CLOCKOUT';
                              
                              return (
                                <span key={i} style={{
                                  display: 'inline-block',
                                  padding: '2px 6px',
                                  borderRadius: 3,
                                  fontSize: 11,
                                  fontWeight: 600,
                                  backgroundColor: 
                                    isAbsent ? '#fee2e2' : 
                                    isDidntClockout ? '#fecaca' :
                                    '#fee2e2',
                                  color: 
                                    isAbsent ? '#991b1b' : 
                                    isDidntClockout ? '#7f1d1d' :
                                    '#991b1b'
                                }}>
                                  {isAbsent ? '❌ ABSENT' : 
                                   isDidntClockout ? '⚠️ DIDNT_CLOCKOUT' : 
                                   flag}
                                </span>
                              );
                            })}
                          </div>
                        ) : (
                          <span style={{ fontSize: 13, color: '#9ca3af' }}>—</span>
                        )}
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
};
