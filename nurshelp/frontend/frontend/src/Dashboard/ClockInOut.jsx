// src/Dashboard/ClockInOut.jsx
import React, { useState, useEffect } from 'react';

const BACKEND_URL = "http://localhost:5001";

export default function ClockInOut({ token, shiftId, onStatusUpdate }) {
  const [clockStatus, setClockStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  // Fetch clock status on mount
  useEffect(() => {
    if (token) {
      fetchClockStatus();
      const interval = setInterval(fetchClockStatus, 5000); // Update every 5 seconds
      return () => clearInterval(interval);
    }
  }, [token]);

  // Calculate elapsed time
  useEffect(() => {
    if (clockStatus?.clockRecord?.status === 'CLOCKED_IN' || clockStatus?.clockRecord?.status === 'ON_BREAK') {
      const clockInTime = new Date(clockStatus.clockRecord.clockInTime);
      const interval = setInterval(() => {
        const now = new Date();
        const elapsed = Math.floor((now - clockInTime) / 1000 / 60); // minutes
        setElapsedTime(elapsed);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [clockStatus]);

  const fetchClockStatus = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/clock-status`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      setClockStatus(data);
      setError(null);
    } catch (err) {
      setError(err.message);
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
          shiftId,
          location: 'ED'
        })
      });
      const data = await res.json();
      if (data.success) {
        await fetchClockStatus();
        onStatusUpdate?.('CLOCKED_IN');
      } else {
        setError(data.message);
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
        },
        body: JSON.stringify({})
      });
      const data = await res.json();
      if (data.success) {
        await fetchClockStatus();
        onStatusUpdate?.('CLOCKED_OUT');
      } else {
        setError(data.message);
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
        await fetchClockStatus();
        onStatusUpdate?.('ON_BREAK');
      } else {
        setError(data.message);
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
        await fetchClockStatus();
        onStatusUpdate?.('CLOCKED_IN');
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  const isClocked = clockStatus?.clockRecord?.status === 'CLOCKED_IN' || clockStatus?.clockRecord?.status === 'ON_BREAK';
  const onBreak = clockStatus?.clockRecord?.status === 'ON_BREAK';

  return (
    <div style={styles.container}>
      <h2>Shift Management</h2>

      {error && (
        <div style={styles.error}>
          ⚠️ {error}
        </div>
      )}

      {/* Main Clock Button */}
      <div style={styles.mainButton}>
        {!isClocked ? (
          <button
            onClick={handleClockIn}
            disabled={loading}
            style={{ ...styles.button, ...styles.clockInBtn }}
          >
            {loading ? '⏳ Clocking In...' : '🟢 CLOCK IN'}
          </button>
        ) : (
          <button
            onClick={handleClockOut}
            disabled={loading}
            style={{ ...styles.button, ...styles.clockOutBtn }}
          >
            {loading ? '⏳ Clocking Out...' : '🔴 CLOCK OUT'}
          </button>
        )}
      </div>

      {/* Status Display */}
      {clockStatus?.clockRecord && (
        <div style={styles.statusCard}>
          <div>
            <span style={styles.label}>Status:</span>
            <span style={styles.value}>
              {onBreak ? '☕ ON BREAK' : '⏱️ CLOCKED IN'}
            </span>
          </div>

          <div>
            <span style={styles.label}>Time Elapsed:</span>
            <span style={styles.value}>{formatTime(elapsedTime)}</span>
          </div>

          <div>
            <span style={styles.label}>Clock In:</span>
            <span style={styles.value}>
              {new Date(clockStatus.clockRecord.clockInTime).toLocaleTimeString()}
            </span>
          </div>

          {clockStatus.clockRecord.breakStartTime && (
            <div>
              <span style={styles.label}>Break Started:</span>
              <span style={styles.value}>
                {new Date(clockStatus.clockRecord.breakStartTime).toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Break Controls */}
      {isClocked && (
        <div style={styles.breakControls}>
          {!onBreak ? (
            <button
              onClick={handleBreakStart}
              disabled={loading}
              style={{ ...styles.button, ...styles.breakBtn }}
            >
              ☕ Start Break
            </button>
          ) : (
            <button
              onClick={handleBreakEnd}
              disabled={loading}
              style={{ ...styles.button, ...styles.resumeBtn }}
            >
              ▶️ Resume Work
            </button>
          )}
        </div>
      )}

      {/* Not Clocked In Message */}
      {!isClocked && clockStatus?.status === 'NOT_CLOCKED_IN' && (
        <div style={styles.noClockMessage}>
          You are not currently clocked in. Click the button above to start your shift.
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: '20px',
    maxWidth: '500px',
    margin: '20px auto',
    backgroundColor: '#f8f9fa',
    borderRadius: '12px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  },
  mainButton: {
    marginTop: '20px',
    marginBottom: '20px'
  },
  button: {
    width: '100%',
    padding: '16px',
    fontSize: '16px',
    fontWeight: 'bold',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.3s ease'
  },
  clockInBtn: {
    backgroundColor: '#28a745',
    color: 'white'
  },
  clockOutBtn: {
    backgroundColor: '#dc3545',
    color: 'white'
  },
  breakBtn: {
    backgroundColor: '#ff9800',
    color: 'white'
  },
  resumeBtn: {
    backgroundColor: '#2196F3',
    color: 'white'
  },
  statusCard: {
    backgroundColor: 'white',
    padding: '16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #e0e0e0'
  },
  error: {
    backgroundColor: '#f8d7da',
    color: '#721c24',
    padding: '12px',
    borderRadius: '4px',
    marginBottom: '16px',
    border: '1px solid #f5c6cb'
  },
  label: {
    display: 'block',
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#666',
    marginTop: '8px'
  },
  value: {
    display: 'block',
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#333',
    marginTop: '4px'
  },
  breakControls: {
    marginTop: '20px'
  },
  noClockMessage: {
    backgroundColor: '#e7f3ff',
    color: '#004085',
    padding: '12px',
    borderRadius: '4px',
    marginTop: '16px',
    border: '1px solid #b8daff'
  }
};
