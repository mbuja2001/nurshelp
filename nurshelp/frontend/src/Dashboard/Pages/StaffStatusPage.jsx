// src/Dashboard/Pages/StaffStatusPage.jsx
import React, { useState, useEffect } from 'react';
import { FaSync, FaCheckCircle, FaTimesCircle, FaHourglass, FaFilter } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function StaffStatusPage({ token }) {
  const [staffStatus, setStaffStatus] = useState([]);
  const [summary, setSummary] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    location: '',
    status: 'ALL',
    showPendingOnly: false
  });
  const [selectedStaff, setSelectedStaff] = useState(null);
  const [approvalLoading, setApprovalLoading] = useState(false);
  const [approvalError, setApprovalError] = useState(null);
  const [sortBy, setSortBy] = useState('name');

  useEffect(() => {
    if (token) {
      fetchStaffStatus();
      const interval = setInterval(fetchStaffStatus, 10000); // Every 10 seconds
      return () => clearInterval(interval);
    }
  }, [token, filters]);

  const fetchStaffStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      let url = '/api/manager/staff-status?';
      if (filters.location) url += `location=${filters.location}&`;
      if (filters.status !== 'ALL') url += `status=${filters.status}&`;
      if (filters.showPendingOnly) url += 'showPendingApprovals=true&';

      const res = await fetch(`${BACKEND_URL}${url}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setStaffStatus(data.staff);
        setSummary(data.summary);
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleApproval = async (recordId, type, approved, reason = '') => {
    setApprovalLoading(true);
    setApprovalError(null);
    try {
      const url = type === 'LATE_ARRIVAL'
        ? `/api/staff/approve-late-arrival/${recordId}`
        : `/api/staff/approve-break/${recordId}`;

      const res = await fetch(`${BACKEND_URL}${url}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ approved, reason })
      });

      const data = await res.json();
      if (data.success) {
        await fetchStaffStatus();
        setSelectedStaff(null);
      } else {
        setApprovalError(data.message);
      }
    } catch (err) {
      setApprovalError(err.message);
    } finally {
      setApprovalLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'CLOCKED_IN': return '#10b981';
      case 'ON_BREAK': return '#f59e0b';
      case 'CLOCKED_OUT': return '#6c757d';
      default: return '#e5e7eb';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'CLOCKED_IN': return '✓';
      case 'ON_BREAK': return '☕';
      case 'CLOCKED_OUT': return '🔴';
      default: return '⚪';
    }
  };

  const formatDuration = (minutes) => {
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return `${h}h ${m}m`;
  };

  const filteredAndSortedStaff = [...staffStatus]
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          const nameA = `${a.nurse?.name} ${a.nurse?.surname}`;
          const nameB = `${b.nurse?.name} ${b.nurse?.surname}`;
          return nameA.localeCompare(nameB);
        case 'elapsed':
          return b.elapsedMinutes - a.elapsedMinutes;
        case 'clockin':
          return new Date(b.clockInTime) - new Date(a.clockInTime);
        default:
          return 0;
      }
    });

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>Staff Status Dashboard</h1>
        <button onClick={fetchStaffStatus} style={styles.refreshBtn} disabled={loading}>
          <FaSync style={{ marginRight: '8px' }} />
          Refresh
        </button>
      </div>

      {error && <div style={styles.errorBanner}>{error}</div>}

      {/* Summary Cards */}
      <div style={styles.summaryGrid}>
        <SummaryCard label="Total Staff" value={summary.total || 0} color="#3b82f6" />
        <SummaryCard label="Clocked In" value={summary.clockedIn || 0} color="#10b981" />
        <SummaryCard label="On Break" value={summary.onBreak || 0} color="#f59e0b" />
        <SummaryCard label="Clocked Out" value={summary.clockedOut || 0} color="#6c757d" />
        <SummaryCard
          label="Pending Approvals"
          value={summary.pendingApprovals || 0}
          color="#ef4444"
          highlight={summary.pendingApprovals > 0}
        />
      </div>

      {/* Filters */}
      <div style={styles.filterSection}>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>
            <FaFilter style={{ marginRight: '6px' }} />
            Filter:
          </label>
          <select
            value={filters.location}
            onChange={(e) => setFilters({ ...filters, location: e.target.value })}
            style={styles.select}
          >
            <option value="">All Locations</option>
            <option value="ED">ED</option>
            <option value="ICU">ICU</option>
            <option value="OR">OR</option>
            <option value="MATERNITY">Maternity</option>
            <option value="NICU">NICU</option>
          </select>

          <select
            value={filters.status}
            onChange={(e) => setFilters({ ...filters, status: e.target.value })}
            style={styles.select}
          >
            <option value="ALL">All Status</option>
            <option value="CLOCKED_IN">Clocked In</option>
            <option value="ON_BREAK">On Break</option>
            <option value="CLOCKED_OUT">Clocked Out</option>
          </select>

          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={filters.showPendingOnly}
              onChange={(e) => setFilters({ ...filters, showPendingOnly: e.target.checked })}
            />
            Pending Only
          </label>
        </div>

        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Sort:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            style={styles.select}
          >
            <option value="name">Name</option>
            <option value="elapsed">Time Elapsed</option>
            <option value="clockin">Clock-in Time</option>
          </select>
        </div>
      </div>

      {/* Staff Table */}
      {loading ? (
        <div style={styles.loadingBox}>⏳ Loading...</div>
      ) : filteredAndSortedStaff.length === 0 ? (
        <div style={styles.emptyBox}>No staff records found</div>
      ) : (
        <div style={styles.tableContainer}>
          <div style={styles.tableHeader}>
            <div style={styles.tableHeaderCell}>Name</div>
            <div style={styles.tableHeaderCell}>Status</div>
            <div style={styles.tableHeaderCell}>Location</div>
            <div style={styles.tableHeaderCell}>Clock In</div>
            <div style={styles.tableHeaderCell}>Elapsed</div>
            <div style={styles.tableHeaderCell}>Actions</div>
          </div>

          {filteredAndSortedStaff.map((staff) => (
            <div
              key={staff._id}
              style={{
                ...styles.tableRow,
                ...(staff.hasPendingApprovals ? styles.tableRowHighlight : {})
              }}
            >
              <div style={styles.tableCell}>
                <strong>{staff.nurse?.name} {staff.nurse?.surname}</strong>
                <div style={styles.cellSmall}>{staff.nurse?.email}</div>
              </div>

              <div style={styles.tableCell}>
                <div
                  style={{
                    ...styles.statusBadge,
                    backgroundColor: getStatusColor(staff.status)
                  }}
                >
                  {getStatusIcon(staff.status)} {staff.status}
                </div>
              </div>

              <div style={styles.tableCell}>{staff.location}</div>

              <div style={styles.tableCell}>
                {new Date(staff.clockInTime).toLocaleTimeString('en-US', {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </div>

              <div style={styles.tableCell}>
                <strong>{formatDuration(staff.elapsedMinutes)}</strong>
              </div>

              <div style={styles.tableCell}>
                {staff.hasPendingApprovals ? (
                  <button
                    onClick={() => setSelectedStaff(staff)}
                    style={styles.actionBtn}
                  >
                    ⚡ {staff.lateArrivalApprovalStatus === 'PENDING' && '1 '}
                    {staff.breakApprovalStatus === 'PENDING' && '1 '}
                    Approve
                  </button>
                ) : (
                  <span style={styles.nothingPending}>-</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Approval Modal */}
      {selectedStaff && (
        <div style={styles.modalOverlay}>
          <div style={styles.modalContent}>
            <h2>Staff Approvals</h2>
            <div style={styles.staffNameModal}>
              {selectedStaff.nurse?.name} {selectedStaff.nurse?.surname}
            </div>

            {approvalError && (
              <div style={styles.errorBanner}>{approvalError}</div>
            )}

            {selectedStaff.lateArrivalApprovalStatus === 'PENDING' && (
              <div style={styles.approvalCard}>
                <div style={styles.approvalType}>⏰ Late Arrival</div>
                <div style={styles.approvalDetail}>
                  <strong>{selectedStaff.lateArrivalMinutes} minutes late</strong>
                </div>
                <div style={styles.approvalTime}>
                  Clocked in: {new Date(selectedStaff.clockInTime).toLocaleTimeString()}
                </div>
                <div style={styles.approvalButtons}>
                  <button
                    onClick={() =>
                      handleApproval(selectedStaff._id, 'LATE_ARRIVAL', true)
                    }
                    style={styles.approveBtn}
                    disabled={approvalLoading}
                  >
                    ✓ Approve
                  </button>
                  <button
                    onClick={() =>
                      handleApproval(selectedStaff._id, 'LATE_ARRIVAL', false, 'No reason')
                    }
                    style={styles.rejectBtn}
                    disabled={approvalLoading}
                  >
                    ✗ Reject
                  </button>
                </div>
              </div>
            )}

            {selectedStaff.breakApprovalStatus === 'PENDING' && (
              <div style={styles.approvalCard}>
                <div style={styles.approvalType}>☕ Break Request</div>
                <div style={styles.approvalDetail}>
                  {selectedStaff.breakReason}
                </div>
                <div style={styles.approvalButtons}>
                  <button
                    onClick={() => handleApproval(selectedStaff._id, 'BREAK', true)}
                    style={styles.approveBtn}
                    disabled={approvalLoading}
                  >
                    ✓ Approve
                  </button>
                  <button
                    onClick={() =>
                      handleApproval(selectedStaff._id, 'BREAK', false, 'Coverage needed')
                    }
                    style={styles.rejectBtn}
                    disabled={approvalLoading}
                  >
                    ✗ Reject
                  </button>
                </div>
              </div>
            )}

            <button
              onClick={() => setSelectedStaff(null)}
              style={styles.closeModalBtn}
              disabled={approvalLoading}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryCard({ label, value, color, highlight }) {
  return (
    <div
      style={{
        ...styles.summaryCard,
        ...(highlight ? styles.summaryCardHighlight : {}),
        borderLeftColor: color
      }}
    >
      <div style={styles.summaryLabel}>{label}</div>
      <div style={{ ...styles.summaryValue, color }}>{value}</div>
    </div>
  );
}

const styles = {
  container: {
    padding: '20px'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px'
  },
  refreshBtn: {
    padding: '8px 16px',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
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
  summaryGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '12px',
    marginBottom: '24px'
  },
  summaryCard: {
    backgroundColor: 'white',
    padding: '16px',
    borderRadius: '8px',
    border: '2px solid #e5e7eb',
    borderLeft: '4px solid #3b82f6'
  },
  summaryCardHighlight: {
    backgroundColor: '#fef2f2',
    borderColor: '#fecaca'
  },
  summaryLabel: {
    fontSize: '12px',
    color: '#6b7280',
    marginBottom: '8px'
  },
  summaryValue: {
    fontSize: '28px',
    fontWeight: 'bold'
  },
  filterSection: {
    backgroundColor: 'white',
    padding: '16px',
    borderRadius: '8px',
    marginBottom: '24px',
    border: '1px solid #e5e7eb',
    display: 'flex',
    gap: '24px',
    flexWrap: 'wrap',
    alignItems: 'center'
  },
  filterGroup: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center'
  },
  filterLabel: {
    fontWeight: '500',
    color: '#374151',
    display: 'flex',
    alignItems: 'center'
  },
  select: {
    padding: '6px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    cursor: 'pointer'
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    cursor: 'pointer'
  },
  loadingBox: {
    textAlign: 'center',
    padding: '40px',
    color: '#6b7280'
  },
  emptyBox: {
    textAlign: 'center',
    padding: '40px',
    color: '#9ca3af',
    backgroundColor: 'white',
    borderRadius: '8px'
  },
  tableContainer: {
    backgroundColor: 'white',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    overflow: 'hidden'
  },
  tableHeader: {
    display: 'grid',
    gridTemplateColumns: 'repeat(6, 1fr)',
    backgroundColor: '#f9fafb',
    padding: '12px 16px',
    borderBottom: '2px solid #e5e7eb',
    fontWeight: '600',
    fontSize: '13px',
    color: '#6b7280'
  },
  tableHeaderCell: {
    padding: '8px'
  },
  tableRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(6, 1fr)',
    padding: '12px 16px',
    borderBottom: '1px solid #e5e7eb',
    alignItems: 'center',
    transition: 'background-color 0.2s'
  },
  tableRowHighlight: {
    backgroundColor: '#fef2f2'
  },
  tableCell: {
    padding: '8px'
  },
  cellSmall: {
    fontSize: '12px',
    color: '#9ca3af',
    marginTop: '4px'
  },
  statusBadge: {
    display: 'inline-block',
    color: 'white',
    padding: '4px 12px',
    borderRadius: '6px',
    fontSize: '12px',
    fontWeight: '600'
  },
  actionBtn: {
    padding: '6px 12px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: '600'
  },
  nothingPending: {
    color: '#d1d5db'
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '24px',
    maxWidth: '500px',
    width: '90%',
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
  },
  staffNameModal: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#3b82f6',
    marginBottom: '16px',
    paddingBottom: '16px',
    borderBottom: '2px solid #e5e7eb'
  },
  approvalCard: {
    backgroundColor: '#f9fafb',
    padding: '16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #e5e7eb'
  },
  approvalType: {
    fontWeight: '600',
    fontSize: '14px',
    color: '#1f2937',
    marginBottom: '8px'
  },
  approvalDetail: {
    fontSize: '14px',
    color: '#374151',
    marginBottom: '6px'
  },
  approvalTime: {
    fontSize: '12px',
    color: '#9ca3af',
    marginBottom: '12px'
  },
  approvalButtons: {
    display: 'flex',
    gap: '8px'
  },
  approveBtn: {
    flex: 1,
    padding: '8px 12px',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '13px'
  },
  rejectBtn: {
    flex: 1,
    padding: '8px 12px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '13px'
  },
  closeModalBtn: {
    width: '100%',
    padding: '10px 16px',
    backgroundColor: '#e5e7eb',
    color: '#374151',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600'
  }
};
