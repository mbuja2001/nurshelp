// src/Dashboard/Pages/MyClockHistoryPage.jsx
import React, { useState, useEffect } from 'react';
import { FaDownload, FaFilter, FaCalendarAlt } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function MyClockHistoryPage({ token }) {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    if (token) {
      // Set default date range (last 30 days)
      const end = new Date();
      const start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
      
      setStartDate(start.toISOString().split('T')[0]);
      setEndDate(end.toISOString().split('T')[0]);
      
      fetchHistory(start.toISOString().split('T')[0], end.toISOString().split('T')[0]);
    }
  }, [token]);

  const fetchHistory = async (start, end) => {
    setLoading(true);
    setError(null);
    try {
      let url = '/api/staff/clock-history?';
      if (start) url += `startDate=${start}&`;
      if (end) url += `endDate=${end}&`;

      const res = await fetch(`${BACKEND_URL}${url}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setRecords(data.records || []);
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleApply = () => {
    if (!startDate || !endDate) {
      setError('Please select both start and end dates');
      return;
    }
    if (new Date(startDate) > new Date(endDate)) {
      setError('Start date must be before end date');
      return;
    }
    fetchHistory(startDate, endDate);
  };

  const handleDownloadCSV = () => {
    if (records.length === 0) {
      alert('No records to download');
      return;
    }

    // Create CSV headers
    const headers = [
      'Date',
      'Clock In',
      'Clock Out',
      'Break Duration (min)',
      'Total Work Hours',
      'Location',
      'Late Arrival (min)',
      'Late Approval',
      'Break Approval',
      'Notes'
    ];

    // Create CSV rows
    const rows = records.map(record => [
      new Date(record.date).toLocaleDateString(),
      new Date(record.clockInTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      record.clockOutTime 
        ? new Date(record.clockOutTime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
        : '--:--',
      record.totalBreakTime ? Math.floor(record.totalBreakTime / 1000 / 60) : 0,
      record.totalWorkTime ? (record.totalWorkTime / 1000 / 60 / 60).toFixed(2) : '--',
      record.location || 'ED',
      record.lateArrivalMinutes || 0,
      record.lateArrivalApprovalStatus || 'N/A',
      record.breakApprovalStatus || 'N/A',
      record.notes || ''
    ]);

    // Create CSV content
    const csvContent = [
      headers.join(','),
      ...rows.map(row =>
        row.map(cell => `"${cell}"`).join(',')
      )
    ].join('\n');

    // Download
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `clock-history-${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  const formatTime = (dateStr) => {
    if (!dateStr) return '--:--';
    return new Date(dateStr).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDuration = (milliseconds) => {
    if (!milliseconds) return 0;
    return Math.floor(milliseconds / 1000 / 60);
  };

  const formatWorkHours = (milliseconds) => {
    if (!milliseconds) return '--';
    const totalMinutes = Math.floor(milliseconds / 1000 / 60);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return `${hours}h ${minutes}m`;
  };

  const getApprovalBadge = (status, color) => {
    if (!status || status === 'NOT_REQUESTED' || status === 'NOT_REQUIRED') return '--';
    
    const colors = {
      'APPROVED': '#10b981',
      'PENDING': '#f59e0b',
      'REJECTED': '#ef4444'
    };
    
    return (
      <span style={{
        ...styles.badge,
        backgroundColor: colors[status] || '#6b7280'
      }}>
        {status === 'APPROVED' && '✓'}
        {status === 'PENDING' && '⏳'}
        {status === 'REJECTED' && '✗'}
        {' ' + status}
      </span>
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>My Clock History</h1>
        <button
          onClick={handleDownloadCSV}
          style={styles.downloadBtn}
          disabled={records.length === 0}
        >
          <FaDownload style={{ marginRight: '8px' }} />
          Download CSV
        </button>
      </div>

      {error && <div style={styles.errorBanner}>{error}</div>}

      {/* Date Range Filter */}
      <div style={styles.filterCard}>
        <h3 style={styles.filterTitle}>
          <FaCalendarAlt style={{ marginRight: '8px' }} />
          Select Date Range
        </h3>
        
        <div style={styles.dateInputs}>
          <div style={styles.dateGroup}>
            <label style={styles.label}>From</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={styles.input}
            />
          </div>

          <div style={styles.dateGroup}>
            <label style={styles.label}>To</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={styles.input}
            />
          </div>

          <button
            onClick={handleApply}
            style={styles.applyBtn}
            disabled={loading}
          >
            <FaFilter style={{ marginRight: '8px' }} />
            Apply
          </button>
        </div>
      </div>

      {/* Results Summary */}
      {!loading && (
        <div style={styles.summaryBox}>
          <strong>{records.length}</strong> records found
          {records.length > 0 && (
            <>
              {' • '}
              Total hours: <strong>{calculateTotalHours(records)}</strong>
            </>
          )}
        </div>
      )}

      {/* Records Table */}
      {loading ? (
        <div style={styles.loadingBox}>⏳ Loading records...</div>
      ) : records.length === 0 ? (
        <div style={styles.emptyBox}>📋 No records found for this date range</div>
      ) : (
        <div style={styles.tableContainer}>
          <div style={styles.tableWrapper}>
            <table style={styles.table}>
              <thead>
                <tr style={styles.tableHeader}>
                  <th style={styles.th}>Date</th>
                  <th style={styles.th}>In</th>
                  <th style={styles.th}>Out</th>
                  <th style={styles.th}>Break (min)</th>
                  <th style={styles.th}>Work Hours</th>
                  <th style={styles.th}>Location</th>
                  <th style={styles.th}>Late</th>
                  <th style={styles.th}>Late Approval</th>
                  <th style={styles.th}>Break Approval</th>
                </tr>
              </thead>
              <tbody>
                {records.map((record, index) => (
                  <tr key={record._id || index} style={styles.tableRow}>
                    <td style={styles.td}>
                      {new Date(record.date).toLocaleDateString()}
                    </td>
                    <td style={styles.td}>
                      <strong>{formatTime(record.clockInTime)}</strong>
                    </td>
                    <td style={styles.td}>
                      {record.clockOutTime ? formatTime(record.clockOutTime) : '--:--'}
                    </td>
                    <td style={styles.td}>
                      {formatDuration(record.totalBreakTime)}
                    </td>
                    <td style={styles.td}>
                      <strong>{formatWorkHours(record.totalWorkTime)}</strong>
                    </td>
                    <td style={styles.td}>{record.location || 'ED'}</td>
                    <td style={styles.td}>
                      {record.lateArrivalMinutes > 0 ? (
                        <span style={styles.lateMinutes}>
                          {record.lateArrivalMinutes} min
                        </span>
                      ) : (
                        '--'
                      )}
                    </td>
                    <td style={styles.td}>
                      {getApprovalBadge(record.lateArrivalApprovalStatus)}
                    </td>
                    <td style={styles.td}>
                      {getApprovalBadge(record.breakApprovalStatus)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Legend */}
      {records.length > 0 && (
        <div style={styles.legend}>
          <div style={styles.legendItem}>
            <span style={{ ...styles.badge, backgroundColor: '#10b981' }}>✓ APPROVED</span>
            <span style={{ marginLeft: '8px', fontSize: '13px', color: '#6b7280' }}>Approval granted</span>
          </div>
          <div style={styles.legendItem}>
            <span style={{ ...styles.badge, backgroundColor: '#f59e0b' }}>⏳ PENDING</span>
            <span style={{ marginLeft: '8px', fontSize: '13px', color: '#6b7280' }}>Awaiting approval</span>
          </div>
          <div style={styles.legendItem}>
            <span style={{ ...styles.badge, backgroundColor: '#ef4444' }}>✗ REJECTED</span>
            <span style={{ marginLeft: '8px', fontSize: '13px', color: '#6b7280' }}>Approval denied</span>
          </div>
        </div>
      )}
    </div>
  );
}

function calculateTotalHours(records) {
  const totalMinutes = records.reduce((sum, record) => {
    if (record.totalWorkTime) {
      sum += Math.floor(record.totalWorkTime / 1000 / 60);
    }
    return sum;
  }, 0);

  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return `${hours}h ${minutes}m`;
}

const styles = {
  container: {
    padding: '20px',
    maxWidth: '1200px'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px'
  },
  downloadBtn: {
    padding: '10px 20px',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center'
  },
  errorBanner: {
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px'
  },
  filterCard: {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    marginBottom: '24px'
  },
  filterTitle: {
    margin: '0 0 16px 0',
    color: '#1f2937',
    fontSize: '16px',
    display: 'flex',
    alignItems: 'center'
  },
  dateInputs: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '16px'
  },
  dateGroup: {
    display: 'flex',
    flexDirection: 'column'
  },
  label: {
    fontWeight: '500',
    color: '#374151',
    marginBottom: '6px',
    fontSize: '14px'
  },
  input: {
    padding: '8px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    fontFamily: 'inherit'
  },
  applyBtn: {
    padding: '8px 16px',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'flex-end'
  },
  summaryBox: {
    backgroundColor: '#f0fdf4',
    border: '1px solid #bbf7d0',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px',
    color: '#166534'
  },
  loadingBox: {
    textAlign: 'center',
    padding: '40px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#6b7280'
  },
  emptyBox: {
    textAlign: 'center',
    padding: '40px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#9ca3af'
  },
  tableContainer: {
    backgroundColor: 'white',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    overflow: 'hidden'
  },
  tableWrapper: {
    overflowX: 'auto'
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '14px'
  },
  tableHeader: {
    backgroundColor: '#f9fafb',
    borderBottom: '2px solid #e5e7eb'
  },
  th: {
    padding: '12px 16px',
    textAlign: 'left',
    fontWeight: '600',
    color: '#374151',
    borderRight: '1px solid #e5e7eb'
  },
  tableRow: {
    borderBottom: '1px solid #e5e7eb'
  },
  td: {
    padding: '12px 16px',
    borderRight: '1px solid #e5e7eb',
    color: '#374151'
  },
  badge: {
    display: 'inline-block',
    padding: '4px 8px',
    borderRadius: '4px',
    color: 'white',
    fontSize: '12px',
    fontWeight: '600'
  },
  lateMinutes: {
    backgroundColor: '#fef3c7',
    color: '#92400e',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '12px',
    fontWeight: '600'
  },
  legend: {
    display: 'flex',
    gap: '24px',
    marginTop: '20px',
    padding: '16px',
    backgroundColor: '#f9fafb',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    flexWrap: 'wrap'
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center'
  }
};
