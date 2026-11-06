import React, { useEffect, useState } from 'react';
import { FaCalendarAlt, FaClock } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
const RANGE_DAYS = { weekly: 7, monthly: 30, quarterly: 90 };

const getRangeStart = (range) => {
  const start = new Date();
  start.setDate(start.getDate() - (RANGE_DAYS[range] || RANGE_DAYS.weekly) + 1);
  start.setHours(0, 0, 0, 0);
  return start;
};

const parseRecordDate = (record) => {
  const value = record.shiftDate || record.date;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

export default function StatisticsPage({ token }) {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [timeRange, setTimeRange] = useState('weekly');

  useEffect(() => {
    if (!token) {
      setRecords([]);
      setError('Sign in to view attendance statistics.');
      setLoading(false);
      return;
    }

    let cancelled = false;
    const fetchAttendance = async () => {
      setLoading(true);
      setError('');
      try {
        const response = await fetch(`${BACKEND_URL}/api/staff/attendance-history`, {
          headers: { Authorization: `Bearer ${token}` },
          cache: 'no-store'
        });
        if (!response.headers.get('content-type')?.includes('application/json')) {
          throw new Error(`Attendance history returned an unexpected response (HTTP ${response.status})`);
        }
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || `Unable to load attendance (HTTP ${response.status})`);
        if (!data.success) throw new Error(data.message || 'Unable to load attendance');
        if (!cancelled) setRecords(Array.isArray(data.records) ? data.records : []);
      } catch (requestError) {
        if (!cancelled) setError(requestError.message || 'Unable to load attendance statistics.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchAttendance();
    return () => { cancelled = true; };
  }, [token]);

  const rangeStart = getRangeStart(timeRange);
  const periodRecords = records.filter((record) => {
    const date = parseRecordDate(record);
    return date && date >= rangeStart;
  });
  const scheduled = periodRecords.filter((record) => ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT', 'ABSENT'].includes(record.status));
  const completed = periodRecords.filter((record) => record.status === 'CLOCKED_OUT');
  const present = periodRecords.filter((record) => ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT'].includes(record.status));
  const late = periodRecords.filter((record) => (record.flags || []).includes('LATE_CLOCKIN'));
  const totalWorkMs = completed.reduce((sum, record) => sum + (Number(record.totalWorkTime) || 0), 0);
  const totalBreakMs = periodRecords.reduce((sum, record) => sum + (Number(record.totalBreakTime) || 0), 0);
  const attendanceRate = scheduled.length ? (present.length / scheduled.length) * 100 : null;
  const averageShiftHours = completed.length ? totalWorkMs / completed.length / 3600000 : 0;
  const formatHours = (milliseconds) => `${Math.floor(milliseconds / 3600000)}h ${Math.floor((milliseconds % 3600000) / 60000)}m`;

  if (loading) return <div style={styles.state}>Loading attendance statistics...</div>;
  if (error) return <div role="alert" style={styles.error}>{error}</div>;

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div>
          <h2 style={styles.title}>Attendance statistics</h2>
          <p style={styles.subtitle}>Shift attendance recorded for your staff account. This does not measure clinical or specialist performance.</p>
        </div>
        <label style={styles.rangeLabel}><FaCalendarAlt aria-hidden="true" /><span className="sr-only">Reporting period</span>
          <select aria-label="Reporting period" value={timeRange} onChange={(event) => setTimeRange(event.target.value)} style={styles.select}>
            <option value="weekly">This week</option>
            <option value="monthly">This month</option>
            <option value="quarterly">This quarter</option>
          </select>
        </label>
      </header>

      {scheduled.length === 0 ? (
        <div style={styles.state}>No shift attendance has been recorded for this period.</div>
      ) : (
        <>
          <div style={styles.grid}>
            <Metric label="Completed shifts" value={`${completed.length}`} note={`${scheduled.length} scheduled attendance records`} />
            <Metric label="Attendance rate" value={attendanceRate === null ? '—' : `${attendanceRate.toFixed(1)}%`} note="Present records / scheduled records" />
            <Metric label="Average shift" value={completed.length ? `${averageShiftHours.toFixed(1)} h` : '—'} note="Completed shifts only" />
            <Metric label="Late arrivals" value={`${late.length}`} note="Flagged in attendance records" />
            <Metric label="Recorded work time" value={formatHours(totalWorkMs)} note="Completed shifts only" />
            <Metric label="Recorded break time" value={formatHours(totalBreakMs)} note="Persisted break duration" />
          </div>
          <p style={styles.footnote}><FaClock aria-hidden="true" /> {scheduled.length} attendance records in this period. Metrics are calculated from this staff member’s persisted records.</p>
        </>
      )}
    </div>
  );
}

function Metric({ label, value, note }) {
  return <article style={styles.metric}><span style={styles.metricLabel}>{label}</span><strong style={styles.metricValue}>{value}</strong><small style={styles.metricNote}>{note}</small></article>;
}

const styles = {
  container: { maxWidth: 1200, margin: '0 auto', padding: 8 },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', marginBottom: 20 },
  title: { margin: 0, color: '#0f172a', fontSize: 22 },
  subtitle: { maxWidth: 720, margin: '6px 0 0', color: '#64748b', fontSize: 13, lineHeight: 1.5 },
  rangeLabel: { display: 'inline-flex', alignItems: 'center', gap: 8, color: '#475569' },
  select: { minHeight: 38, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 5, background: '#fff', color: '#0f172a' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 10 },
  metric: { display: 'grid', gap: 7, padding: 16, border: '1px solid #e2e8f0', borderRadius: 6, background: '#fff' },
  metricLabel: { color: '#64748b', fontSize: 12, fontWeight: 700, textTransform: 'uppercase' },
  metricValue: { color: '#0f172a', fontSize: 24 },
  metricNote: { color: '#64748b', fontSize: 11 },
  footnote: { display: 'flex', alignItems: 'center', gap: 8, marginTop: 14, color: '#64748b', fontSize: 12 },
  state: { padding: 24, color: '#64748b', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6, textAlign: 'center' },
  error: { padding: 14, color: '#991b1b', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6 }
};
