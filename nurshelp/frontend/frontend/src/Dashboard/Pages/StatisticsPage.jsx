// src/Dashboard/Pages/StatisticsPage.jsx
import React, { useState, useEffect } from 'react';
import { FaChartBar, FaCalendarAlt, FaClock, FaUser } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function StatisticsPage({ token }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('weekly'); // 'weekly', 'monthly', 'quarterly'

  useEffect(() => {
    if (token) {
      fetchStatistics();
    }
  }, [token, timeRange]);

  const fetchStatistics = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/nurses/statistics?range=${timeRange}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const data = await res.json();
      if (data.success) {
        setStats(data.stats);
      } else {
        setError(data.message || 'Failed to fetch statistics');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div style={styles.loadingBox}>⏳ Loading statistics...</div>;
  }

  if (error) {
    return <div style={styles.errorBox}>{error}</div>;
  }

  if (!stats) {
    return <div style={styles.emptyBox}>📊 No statistics available</div>;
  }

  const getPercentageColor = (percentage, threshold = 50) => {
    if (percentage >= threshold) return '#10b981';
    if (percentage >= threshold / 2) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>📊 Work Statistics</h1>
        <p style={styles.subtitle}>Your performance metrics and insights</p>
      </div>

      {/* Time Range Selector */}
      <div style={styles.rangeSelector}>
        <button
          onClick={() => setTimeRange('weekly')}
          style={{
            ...styles.rangeBtn,
            ...(timeRange === 'weekly' ? styles.rangeBtnActive : {})
          }}
        >
          This Week
        </button>
        <button
          onClick={() => setTimeRange('monthly')}
          style={{
            ...styles.rangeBtn,
            ...(timeRange === 'monthly' ? styles.rangeBtnActive : {})
          }}
        >
          This Month
        </button>
        <button
          onClick={() => setTimeRange('quarterly')}
          style={{
            ...styles.rangeBtn,
            ...(timeRange === 'quarterly' ? styles.rangeBtnActive : {})
          }}
        >
          This Quarter
        </button>
      </div>

      {/* Main Metrics Grid */}
      <div style={styles.metricsGrid}>
        {/* Total Hours */}
        <div style={styles.metricCard}>
          <div style={styles.metricIcon}>🕐</div>
          <div style={styles.metricContent}>
            <p style={styles.metricLabel}>Total Work Hours</p>
            <p style={styles.metricValue}>{stats.totalHours?.toFixed(1) || '0'} hrs</p>
            <p style={styles.metricTrend}>
              {stats.hoursTrend > 0 ? '📈' : stats.hoursTrend < 0 ? '📉' : '→'} 
              {' '}{Math.abs(stats.hoursTrend || 0).toFixed(1)} vs previous
            </p>
          </div>
        </div>

        {/* Shifts Completed */}
        <div style={styles.metricCard}>
          <div style={styles.metricIcon}>✓</div>
          <div style={styles.metricContent}>
            <p style={styles.metricLabel}>Shifts Completed</p>
            <p style={styles.metricValue}>{stats.shiftsCompleted || 0}</p>
            <p style={styles.metricTrend}>
              {stats.shiftsScheduled || 0} scheduled
            </p>
          </div>
        </div>

        {/* Attendance Rate */}
        <div style={styles.metricCard}>
          <div style={styles.metricIcon}>📍</div>
          <div style={styles.metricContent}>
            <p style={styles.metricLabel}>Attendance Rate</p>
            <p style={styles.metricValue}>{stats.attendanceRate?.toFixed(1) || '0'}%</p>
            <div style={styles.progressBar}>
              <div
                style={{
                  ...styles.progressFill,
                  width: `${Math.min(stats.attendanceRate || 0, 100)}%`,
                  backgroundColor: getPercentageColor(stats.attendanceRate || 0)
                }}
              />
            </div>
          </div>
        </div>

        {/* Average Shift Duration */}
        <div style={styles.metricCard}>
          <div style={styles.metricIcon}>⏱️</div>
          <div style={styles.metricContent}>
            <p style={styles.metricLabel}>Average Shift</p>
            <p style={styles.metricValue}>
              {Math.floor(stats.avgShiftDuration || 0)}h {Math.floor((stats.avgShiftDuration || 0) % 1 * 60)}m
            </p>
            <p style={styles.metricTrend}>per shift</p>
          </div>
        </div>
      </div>

      {/* Performance Section */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>⚡ Performance</h2>

        <div style={styles.performanceGrid}>
          {/* On-Time Arrivals */}
          <div style={styles.performanceCard}>
            <p style={styles.performanceLabel}>On-Time Arrivals</p>
            <p style={styles.performanceValue}>{stats.onTimeArrivals || 0}/{stats.shiftsCompleted || 0}</p>
            <div style={styles.performancePercentage}>
              <span style={{
                ...styles.percentage,
                color: getPercentageColor((stats.onTimeArrivals || 0) / (stats.shiftsCompleted || 1) * 100)
              }}>
                {((stats.onTimeArrivals || 0) / (stats.shiftsCompleted || 1) * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Late Arrivals */}
          <div style={styles.performanceCard}>
            <p style={styles.performanceLabel}>Late Arrivals</p>
            <p style={styles.performanceValue}>{stats.lateArrivals || 0}</p>
            <p style={styles.performanceNote}>
              {stats.lateArrivalsPending || 0} pending approval
            </p>
          </div>

          {/* Break Compliance */}
          <div style={styles.performanceCard}>
            <p style={styles.performanceLabel}>Break Compliance</p>
            <p style={styles.performanceValue}>{stats.breaksCompleted || 0}/{stats.shiftsCompleted || 0}</p>
            <div style={styles.performancePercentage}>
              <span style={{
                ...styles.percentage,
                color: getPercentageColor((stats.breaksCompleted || 0) / (stats.shiftsCompleted || 1) * 100)
              }}>
                {((stats.breaksCompleted || 0) / (stats.shiftsCompleted || 1) * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Total Break Time */}
          <div style={styles.performanceCard}>
            <p style={styles.performanceLabel}>Total Break Time</p>
            <p style={styles.performanceValue}>
              {Math.floor(stats.totalBreakTime || 0)}h {Math.floor((stats.totalBreakTime || 0) % 1 * 60)}m
            </p>
            <p style={styles.performanceNote}>across all shifts</p>
          </div>
        </div>
      </div>

      {/* Approval Status */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>✅ Approval Status</h2>

        <div style={styles.approvalGrid}>
          {/* Pending Approvals */}
          <div style={styles.approvalCard}>
            <div style={styles.approvalCardHeader}>
              <p style={styles.approvalLabel}>Pending Approvals</p>
              <span style={{
                ...styles.approvalBadge,
                backgroundColor: '#f59e0b'
              }}>
                {stats.pendingApprovals || 0}
              </span>
            </div>
            {stats.pendingApprovals > 0 && (
              <p style={{ margin: 0, fontSize: '13px', color: '#6b7280' }}>
                Awaiting manager review
              </p>
            )}
          </div>

          {/* Approved This Period */}
          <div style={styles.approvalCard}>
            <div style={styles.approvalCardHeader}>
              <p style={styles.approvalLabel}>Approved</p>
              <span style={{
                ...styles.approvalBadge,
                backgroundColor: '#10b981'
              }}>
                {stats.approvedThisPeriod || 0}
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '13px', color: '#6b7280' }}>
              this {timeRange}
            </p>
          </div>

          {/* Rejected This Period */}
          <div style={styles.approvalCard}>
            <div style={styles.approvalCardHeader}>
              <p style={styles.approvalLabel}>Rejected</p>
              <span style={{
                ...styles.approvalBadge,
                backgroundColor: '#ef4444'
              }}>
                {stats.rejectedThisPeriod || 0}
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '13px', color: '#6b7280' }}>
              this {timeRange}
            </p>
          </div>
        </div>
      </div>

      {/* Insights */}
      {stats.insights && stats.insights.length > 0 && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>💡 Insights</h2>

          <div style={styles.insightsList}>
            {stats.insights.map((insight, idx) => (
              <div key={idx} style={styles.insightCard}>
                <div style={styles.insightIcon}>
                  {insight.type === 'positive' && '👍'}
                  {insight.type === 'warning' && '⚠️'}
                  {insight.type === 'info' && 'ℹ️'}
                </div>
                <div>
                  <p style={styles.insightTitle}>{insight.title}</p>
                  <p style={styles.insightText}>{insight.message}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Comparison Section */}
      {stats.comparison && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>📈 Comparison</h2>

          <div style={styles.comparisonGrid}>
            <div style={styles.comparisonCard}>
              <p style={styles.comparisonLabel}>You</p>
              <p style={styles.comparisonValue}>{stats.comparison.yourValue?.toFixed(1)}</p>
              <p style={styles.comparisonUnit}>{stats.comparison.unit}</p>
            </div>

            <div style={styles.comparisonCard}>
              <p style={styles.comparisonLabel}>Department Average</p>
              <p style={styles.comparisonValue}>{stats.comparison.departmentAvg?.toFixed(1)}</p>
              <p style={styles.comparisonUnit}>{stats.comparison.unit}</p>
            </div>

            <div style={styles.comparisonCard}>
              <p style={styles.comparisonLabel}>Status</p>
              <p style={{
                ...styles.comparisonValue,
                color: (stats.comparison.yourValue || 0) >= (stats.comparison.departmentAvg || 0) ? '#10b981' : '#ef4444'
              }}>
                {(stats.comparison.yourValue || 0) >= (stats.comparison.departmentAvg || 0) ? 'Above' : 'Below'}
              </p>
              <p style={styles.comparisonUnit}>average</p>
            </div>
          </div>
        </div>
      )}
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
  loadingBox: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#6b7280'
  },
  errorBox: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: '#fee2e2',
    borderRadius: '8px',
    color: '#991b1b'
  },
  emptyBox: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#9ca3af'
  },
  rangeSelector: {
    display: 'flex',
    gap: '8px',
    marginBottom: '24px',
    borderBottom: '2px solid #e5e7eb',
    paddingBottom: '12px'
  },
  rangeBtn: {
    padding: '10px 16px',
    backgroundColor: 'transparent',
    border: 'none',
    cursor: 'pointer',
    fontWeight: '600',
    color: '#6b7280',
    fontSize: '14px'
  },
  rangeBtnActive: {
    color: '#3b82f6',
    borderBottom: '2px solid #3b82f6',
    marginBottom: '-14px'
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '16px',
    marginBottom: '32px'
  },
  metricCard: {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    display: 'flex',
    gap: '16px'
  },
  metricIcon: {
    fontSize: '32px',
    flexShrink: 0
  },
  metricContent: {
    flex: 1
  },
  metricLabel: {
    margin: '0 0 6px 0',
    fontSize: '13px',
    fontWeight: '600',
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  metricValue: {
    margin: '0 0 4px 0',
    fontSize: '24px',
    fontWeight: '700',
    color: '#1f2937'
  },
  metricTrend: {
    margin: 0,
    fontSize: '12px',
    color: '#9ca3af'
  },
  progressBar: {
    width: '100%',
    height: '6px',
    backgroundColor: '#e5e7eb',
    borderRadius: '3px',
    overflow: 'hidden',
    marginTop: '8px'
  },
  progressFill: {
    height: '100%',
    borderRadius: '3px',
    transition: 'width 0.3s'
  },
  section: {
    backgroundColor: 'white',
    padding: '24px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    marginBottom: '24px'
  },
  sectionTitle: {
    margin: '0 0 20px 0',
    fontSize: '18px',
    fontWeight: '700',
    color: '#1f2937'
  },
  performanceGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '16px'
  },
  performanceCard: {
    backgroundColor: '#f9fafb',
    padding: '16px',
    borderRadius: '6px',
    borderLeft: '4px solid #3b82f6'
  },
  performanceLabel: {
    margin: '0 0 8px 0',
    fontSize: '13px',
    fontWeight: '600',
    color: '#6b7280'
  },
  performanceValue: {
    margin: '0 0 6px 0',
    fontSize: '20px',
    fontWeight: '700',
    color: '#1f2937'
  },
  performancePercentage: {
    display: 'flex',
    alignItems: 'center'
  },
  percentage: {
    fontSize: '14px',
    fontWeight: '700'
  },
  performanceNote: {
    margin: '0',
    fontSize: '12px',
    color: '#9ca3af'
  },
  approvalGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: '16px'
  },
  approvalCard: {
    backgroundColor: '#f9fafb',
    padding: '16px',
    borderRadius: '6px',
    border: '1px solid #e5e7eb'
  },
  approvalCardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px'
  },
  approvalLabel: {
    margin: 0,
    fontSize: '14px',
    fontWeight: '600',
    color: '#1f2937'
  },
  approvalBadge: {
    padding: '4px 8px',
    borderRadius: '4px',
    color: 'white',
    fontWeight: '600',
    fontSize: '12px'
  },
  insightsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px'
  },
  insightCard: {
    backgroundColor: '#f9fafb',
    padding: '16px',
    borderRadius: '6px',
    display: 'flex',
    gap: '12px',
    alignItems: 'flex-start'
  },
  insightIcon: {
    fontSize: '20px',
    flexShrink: 0
  },
  insightTitle: {
    margin: '0 0 4px 0',
    fontSize: '14px',
    fontWeight: '600',
    color: '#1f2937'
  },
  insightText: {
    margin: 0,
    fontSize: '13px',
    color: '#6b7280'
  },
  comparisonGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
    gap: '16px'
  },
  comparisonCard: {
    backgroundColor: '#f9fafb',
    padding: '20px',
    borderRadius: '6px',
    borderTop: '3px solid #3b82f6',
    textAlign: 'center'
  },
  comparisonLabel: {
    margin: '0 0 10px 0',
    fontSize: '13px',
    fontWeight: '600',
    color: '#6b7280'
  },
  comparisonValue: {
    margin: '0 0 4px 0',
    fontSize: '28px',
    fontWeight: '700',
    color: '#1f2937'
  },
  comparisonUnit: {
    margin: 0,
    fontSize: '12px',
    color: '#9ca3af'
  }
};
