// src/Dashboard/Pages/ScheduleBoard.jsx
import React, { useState, useEffect } from 'react';
import { FaCalendarAlt, FaClock, FaMapMarkerAlt } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function ScheduleBoard({ token }) {
  const [schedule, setSchedule] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedWeek, setSelectedWeek] = useState('current'); // 'current' or 'next'

  useEffect(() => {
    if (token) {
      fetchSchedule();
    }
  }, [token, selectedWeek]);

  const fetchSchedule = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/staff/available-shifts`,
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

  const getDayShifts = (dayName) => {
    return schedule.filter(s => s.dayOfWeek === dayName);
  };

  const getStatusColor = (status) => {
    if (status === 'COMPLETED') return '#10b981';
    if (status === 'IN_PROGRESS') return '#3b82f6';
    if (status === 'UPCOMING') return '#f59e0b';
    return '#d1d5db';
  };

  const getStatusLabel = (status) => {
    if (status === 'COMPLETED') return '✓ Completed';
    if (status === 'IN_PROGRESS') return '⏳ In Progress';
    if (status === 'UPCOMING') return '📅 Upcoming';
    return 'Scheduled';
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return '--:--';
    const date = new Date(`2000-01-01T${timeStr}`);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  };

  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>📅 My Schedule</h1>
        <p style={styles.subtitle}>View your upcoming shifts</p>
      </div>

      {error && <div style={styles.errorBanner}>{error}</div>}

      {/* Week Selector */}
      <div style={styles.weekSelector}>
        <button
          onClick={() => setSelectedWeek('current')}
          style={{
            ...styles.tabBtn,
            ...(selectedWeek === 'current' ? styles.tabBtnActive : {})
          }}
        >
          This Week
        </button>
        <button
          onClick={() => setSelectedWeek('next')}
          style={{
            ...styles.tabBtn,
            ...(selectedWeek === 'next' ? styles.tabBtnActive : {})
          }}
        >
          Next Week
        </button>
      </div>

      {loading ? (
        <div style={styles.loadingBox}>⏳ Loading schedule...</div>
      ) : schedule.length === 0 ? (
        <div style={styles.emptyBox}>📋 No shifts scheduled</div>
      ) : (
        <div style={styles.scheduleGrid}>
          {days.map(day => {
            const dayShifts = getDayShifts(day);
            const hasShift = dayShifts.length > 0;

            return (
              <div
                key={day}
                style={{
                  ...styles.dayCard,
                  ...(hasShift ? {} : styles.dayCardEmpty)
                }}
              >
                <h3 style={styles.dayName}>{day}</h3>
                
                {!hasShift ? (
                  <p style={styles.offText}>Day Off</p>
                ) : (
                  <div style={styles.shiftsContainer}>
                    {dayShifts.map((shift, idx) => (
                      <div key={idx} style={styles.shiftCard}>
                        <div
                          style={{
                            ...styles.statusIndicator,
                            backgroundColor: getStatusColor(shift.status)
                          }}
                        />

                        <div style={styles.shiftContent}>
                          <p style={styles.shiftTime}>
                            <FaClock style={{ marginRight: '6px' }} />
                            {formatTime(shift.startTime)} - {formatTime(shift.endTime)}
                          </p>

                          <p style={styles.shiftLocation}>
                            <FaMapMarkerAlt style={{ marginRight: '6px' }} />
                            {shift.location || 'ED'}
                          </p>

                          <p style={styles.shiftDate}>
                            <FaCalendarAlt style={{ marginRight: '6px' }} />
                            {formatDate(shift.date)}
                          </p>

                          <span
                            style={{
                              ...styles.statusBadge,
                              backgroundColor: getStatusColor(shift.status) + '20',
                              color: getStatusColor(shift.status)
                            }}
                          >
                            {getStatusLabel(shift.status)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Schedule Summary */}
      {schedule.length > 0 && (
        <div style={styles.summaryBox}>
          <h3 style={styles.summaryTitle}>Schedule Summary</h3>
          <div style={styles.summaryGrid}>
            <div style={styles.summaryItem}>
              <span style={styles.summaryValue}>{schedule.length}</span>
              <span style={styles.summaryLabel}>Shifts</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryValue}>
                {schedule.reduce((sum, s) => {
                  const start = new Date(`2000-01-01T${s.startTime}`);
                  const end = new Date(`2000-01-01T${s.endTime}`);
                  const hours = (end - start) / 1000 / 60 / 60;
                  return sum + hours;
                }, 0).toFixed(1)}
              </span>
              <span style={styles.summaryLabel}>Hours</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryValue}>
                {schedule.filter(s => s.status === 'COMPLETED').length}
              </span>
              <span style={styles.summaryLabel}>Completed</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryValue}>
                {schedule.filter(s => s.status === 'UPCOMING').length}
              </span>
              <span style={styles.summaryLabel}>Upcoming</span>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={styles.legend}>
        <h4 style={styles.legendTitle}>Status Legend</h4>
        <div style={styles.legendGrid}>
          <div style={styles.legendItem}>
            <div style={{ ...styles.legendColor, backgroundColor: '#10b981' }} />
            <span>Completed</span>
          </div>
          <div style={styles.legendItem}>
            <div style={{ ...styles.legendColor, backgroundColor: '#3b82f6' }} />
            <span>In Progress</span>
          </div>
          <div style={styles.legendItem}>
            <div style={{ ...styles.legendColor, backgroundColor: '#f59e0b' }} />
            <span>Upcoming</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    padding: '24px',
    maxWidth: '1200px',
    margin: '0 auto'
  },
  header: {
    marginBottom: '24px'
  },
  subtitle: {
    color: '#6b7280',
    marginTop: '6px',
    fontSize: '15px',
    margin: '6px 0 0 0'
  },
  errorBanner: {
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px'
  },
  weekSelector: {
    display: 'flex',
    gap: '12px',
    marginBottom: '24px',
    borderBottom: '2px solid #e5e7eb',
    paddingBottom: '12px'
  },
  tabBtn: {
    padding: '10px 16px',
    backgroundColor: 'transparent',
    border: 'none',
    cursor: 'pointer',
    fontWeight: '600',
    color: '#6b7280',
    fontSize: '15px'
  },
  tabBtnActive: {
    color: '#3b82f6',
    borderBottom: '2px solid #3b82f6',
    marginBottom: '-14px'
  },
  loadingBox: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#6b7280'
  },
  emptyBox: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#9ca3af'
  },
  scheduleGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '16px',
    marginBottom: '24px'
  },
  dayCard: {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb'
  },
  dayCardEmpty: {
    backgroundColor: '#f9fafb',
    opacity: 0.6
  },
  dayName: {
    margin: '0 0 16px 0',
    fontSize: '16px',
    fontWeight: '700',
    color: '#1f2937'
  },
  offText: {
    margin: 0,
    color: '#9ca3af',
    textAlign: 'center',
    padding: '40px 0',
    fontSize: '14px'
  },
  shiftsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px'
  },
  shiftCard: {
    backgroundColor: '#f9fafb',
    padding: '12px',
    borderRadius: '6px',
    display: 'flex',
    gap: '12px',
    alignItems: 'flex-start'
  },
  statusIndicator: {
    width: '4px',
    height: '60px',
    borderRadius: '2px',
    flexShrink: 0
  },
  shiftContent: {
    flex: 1
  },
  shiftTime: {
    margin: '0 0 6px 0',
    fontSize: '14px',
    fontWeight: '600',
    color: '#1f2937',
    display: 'flex',
    alignItems: 'center'
  },
  shiftLocation: {
    margin: '4px 0',
    fontSize: '13px',
    color: '#6b7280',
    display: 'flex',
    alignItems: 'center'
  },
  shiftDate: {
    margin: '4px 0 8px 0',
    fontSize: '12px',
    color: '#9ca3af',
    display: 'flex',
    alignItems: 'center'
  },
  statusBadge: {
    display: 'inline-block',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: '600'
  },
  summaryBox: {
    backgroundColor: 'white',
    padding: '24px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    marginBottom: '24px'
  },
  summaryTitle: {
    margin: '0 0 16px 0',
    fontSize: '16px',
    fontWeight: '700',
    color: '#1f2937'
  },
  summaryGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
    gap: '16px'
  },
  summaryItem: {
    textAlign: 'center',
    padding: '16px',
    backgroundColor: '#f9fafb',
    borderRadius: '6px'
  },
  summaryValue: {
    display: 'block',
    fontSize: '24px',
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: '4px'
  },
  summaryLabel: {
    display: 'block',
    fontSize: '12px',
    color: '#6b7280'
  },
  legend: {
    backgroundColor: 'white',
    padding: '16px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb'
  },
  legendTitle: {
    margin: '0 0 12px 0',
    fontSize: '14px',
    fontWeight: '700',
    color: '#1f2937'
  },
  legendGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
    gap: '16px'
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
    color: '#6b7280'
  },
  legendColor: {
    width: '12px',
    height: '12px',
    borderRadius: '2px'
  }
};
