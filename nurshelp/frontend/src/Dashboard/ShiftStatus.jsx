// src/Dashboard/ShiftStatus.jsx
import React, { useState, useEffect } from 'react';

const BACKEND_URL = "http://localhost:5001";

export default function ShiftStatus({ token }) {
  const [clockStatus, setClockStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    if (token) {
      fetchClockStatus();
      const interval = setInterval(fetchClockStatus, 5000);
      return () => clearInterval(interval);
    }
  }, [token]);

  useEffect(() => {
    if (clockStatus?.clockRecord?.status === 'CLOCKED_IN' || clockStatus?.clockRecord?.status === 'ON_BREAK') {
      const clockInTime = new Date(clockStatus.clockRecord.clockInTime);
      const interval = setInterval(() => {
        const now = new Date();
        const elapsed = Math.floor((now - clockInTime) / 1000 / 60);
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
    } catch (err) {
      console.error("Error fetching clock status:", err);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  const getStatusBadge = () => {
    const status = clockStatus?.clockRecord?.status;
    if (status === 'CLOCKED_IN') {
      return { text: '⏱️ Clocked In', color: '#28a745' };
    }
    if (status === 'ON_BREAK') {
      return { text: '☕ On Break', color: '#ff9800' };
    }
    if (status === 'CLOCKED_OUT') {
      return { text: '🔴 Clocked Out', color: '#6c757d' };
    }
    return { text: '⚪ Not Clocked', color: '#6c757d' };
  };

  const badge = getStatusBadge();

  if (loading) {
    return <div style={styles.container}>Loading...</div>;
  }

  return (
    <div style={styles.container}>
      <div style={{ ...styles.badge, backgroundColor: badge.color }}>
        {badge.text}
      </div>

      {clockStatus?.clockRecord && (
        <>
          <div style={styles.info}>
            <span style={styles.label}>Time Elapsed:</span>
            <span style={styles.value}>{formatTime(elapsedTime)}</span>
          </div>

          <div style={styles.info}>
            <span style={styles.label}>Clock In:</span>
            <span style={styles.value}>
              {new Date(clockStatus.clockRecord.clockInTime).toLocaleTimeString()}
            </span>
          </div>

          {clockStatus.clockRecord.breakStartTime && (
            <div style={styles.info}>
              <span style={styles.label}>Break Started:</span>
              <span style={styles.value}>
                {new Date(clockStatus.clockRecord.breakStartTime).toLocaleTimeString()}
              </span>
            </div>
          )}

          {clockStatus.clockRecord.clockOutTime && (
            <div style={styles.info}>
              <span style={styles.label}>Clock Out:</span>
              <span style={styles.value}>
                {new Date(clockStatus.clockRecord.clockOutTime).toLocaleTimeString()}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

const styles = {
  container: {
    backgroundColor: 'white',
    padding: '16px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    marginBottom: '16px'
  },
  badge: {
    display: 'inline-block',
    color: 'white',
    fontWeight: 'bold',
    padding: '8px 16px',
    borderRadius: '20px',
    marginBottom: '12px'
  },
  info: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 0',
    borderBottom: '1px solid #eee'
  },
  label: {
    fontSize: '12px',
    fontWeight: '600',
    color: '#666'
  },
  value: {
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#333'
  }
};
