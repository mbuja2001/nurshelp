// src/Dashboard/Pages/ManagerApprovalPage.jsx
import React, { useState, useEffect } from 'react';
import { FaSearch, FaFilter, FaCheck, FaTimes, FaClock } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function ManagerApprovalPage({ token, userRole }) {
  const [records, setRecords] = useState([]);
  const [filteredRecords, setFilteredRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterStatus, setFilterStatus] = useState('PENDING');
  const [searchNurse, setSearchNurse] = useState('');
  const [actingUpon, setActingUpon] = useState(null);
  const [actionNotes, setActionNotes] = useState('');
  const [activeTab, setActiveTab] = useState('late-arrivals'); // 'late-arrivals' or 'breaks'

  useEffect(() => {
    if (token && userRole === 'MANAGER') {
      fetchPendingApprovals();
    }
  }, [token, userRole]);

  useEffect(() => {
    filterRecords();
  }, [records, filterStatus, searchNurse, activeTab]);

  const fetchPendingApprovals = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/managers/pending-approvals`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setRecords(data.records || []);
      } else {
        setError(data.message || 'Failed to fetch pending approvals');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const filterRecords = () => {
    let filtered = records;

    // Filter by tab
    if (activeTab === 'late-arrivals') {
      filtered = filtered.filter(r => r.approvalType === 'LATE_ARRIVAL');
    } else if (activeTab === 'breaks') {
      filtered = filtered.filter(r => r.approvalType === 'BREAK_OVERRIDE');
    }

    // Filter by status
    if (filterStatus !== 'ALL') {
      filtered = filtered.filter(r => r.status === filterStatus);
    }

    // Filter by nurse name/ID
    if (searchNurse.trim()) {
      const search = searchNurse.toLowerCase();
      filtered = filtered.filter(r =>
        (r.nurseName || '').toLowerCase().includes(search) ||
        (r.nurseId || '').toLowerCase().includes(search)
      );
    }

    setFilteredRecords(filtered);
  };

  const handleApprove = async (recordId) => {
    await submitAction(recordId, 'APPROVED');
  };

  const handleReject = async (recordId) => {
    await submitAction(recordId, 'REJECTED');
  };

  const submitAction = async (recordId, action) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/managers/approve-request`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          recordId,
          action,
          notes: actionNotes
        })
      });
      const data = await res.json();
      if (data.success) {
        // Remove from list and refresh
        setRecords(records.filter(r => r._id !== recordId));
        setActingUpon(null);
        setActionNotes('');
      } else {
        alert('Error: ' + data.message);
      }
    } catch (err) {
      alert('Error: ' + err.message);
    }
  };

  const getRecordColor = (approvalType) => {
    return approvalType === 'LATE_ARRIVAL' ? '#fef3c7' : '#dbeafe';
  };

  const formatTime = (dateStr) => {
    if (!dateStr) return '--:--';
    return new Date(dateStr).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    });
  };

  const getApprovalTypeLabel = (type) => {
    if (type === 'LATE_ARRIVAL') return '⏰ Late Arrival';
    if (type === 'BREAK_OVERRIDE') return '☕ Break Override';
    return type;
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>Manager Approval Center</h1>
        <button onClick={fetchPendingApprovals} style={styles.refreshBtn} disabled={loading}>
          🔄 Refresh
        </button>
      </div>

      {error && <div style={styles.errorBanner}>{error}</div>}

      {/* Tab Navigation */}
      <div style={styles.tabs}>
        <button
          onClick={() => setActiveTab('late-arrivals')}
          style={{
            ...styles.tabBtn,
            ...(activeTab === 'late-arrivals' ? styles.tabBtnActive : {})
          }}
        >
          ⏰ Late Arrivals ({records.filter(r => r.approvalType === 'LATE_ARRIVAL' && r.status === 'PENDING').length})
        </button>
        <button
          onClick={() => setActiveTab('breaks')}
          style={{
            ...styles.tabBtn,
            ...(activeTab === 'breaks' ? styles.tabBtnActive : {})
          }}
        >
          ☕ Break Overrides ({records.filter(r => r.approvalType === 'BREAK_OVERRIDE' && r.status === 'PENDING').length})
        </button>
      </div>

      {/* Filters */}
      <div style={styles.filterCard}>
        <div style={styles.filterRow}>
          <div style={styles.filterGroup}>
            <label style={styles.filterLabel}>
              <FaFilter style={{ marginRight: '8px' }} />
              Status
            </label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              style={styles.select}
            >
              <option value="ALL">All</option>
              <option value="PENDING">Pending</option>
              <option value="APPROVED">Approved</option>
              <option value="REJECTED">Rejected</option>
            </select>
          </div>

          <div style={styles.filterGroup}>
            <label style={styles.filterLabel}>
              <FaSearch style={{ marginRight: '8px' }} />
              Search Nurse
            </label>
            <input
              type="text"
              placeholder="Name or Employee ID"
              value={searchNurse}
              onChange={(e) => setSearchNurse(e.target.value)}
              style={styles.input}
            />
          </div>
        </div>

        <div style={styles.resultCount}>
          Showing <strong>{filteredRecords.length}</strong> record{filteredRecords.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Records */}
      {loading ? (
        <div style={styles.loadingBox}>⏳ Loading pending approvals...</div>
      ) : filteredRecords.length === 0 ? (
        <div style={styles.emptyBox}>
          ✅ No pending approvals at this time
        </div>
      ) : (
        <div style={styles.recordsList}>
          {filteredRecords.map((record) => (
            <div
              key={record._id}
              style={{
                ...styles.recordCard,
                backgroundColor: getRecordColor(record.approvalType)
              }}
            >
              {/* Record Header */}
              <div style={styles.recordHeader}>
                <div>
                  <h3 style={styles.recordTitle}>
                    {getApprovalTypeLabel(record.approvalType)}
                  </h3>
                  <p style={styles.recordInfo}>
                    {record.nurseName} • {record.nurseId}
                  </p>
                  <p style={styles.recordDate}>
                    {formatDate(record.date)}
                  </p>
                </div>
                <span style={{
                  ...styles.statusBadge,
                  backgroundColor: record.status === 'PENDING' ? '#f59e0b' : 
                                   record.status === 'APPROVED' ? '#10b981' : '#ef4444'
                }}>
                  {record.status}
                </span>
              </div>

              {/* Record Details */}
              <div style={styles.recordDetails}>
                {record.approvalType === 'LATE_ARRIVAL' && (
                  <>
                    <div style={styles.detailRow}>
                      <span style={styles.detailLabel}>Scheduled Clock In:</span>
                      <span style={styles.detailValue}>{formatTime(record.scheduledClockInTime)}</span>
                    </div>
                    <div style={styles.detailRow}>
                      <span style={styles.detailLabel}>Actual Clock In:</span>
                      <span style={styles.detailValue}>{formatTime(record.actualClockInTime)}</span>
                    </div>
                    <div style={styles.detailRow}>
                      <span style={styles.detailLabel}>Late By:</span>
                      <span style={{...styles.detailValue, color: '#991b1b', fontWeight: '600'}}>
                        {record.lateArrivalMinutes} minutes
                      </span>
                    </div>
                    {record.reason && (
                      <div style={styles.detailRow}>
                        <span style={styles.detailLabel}>Nurse's Reason:</span>
                        <p style={styles.reasonText}>{record.reason}</p>
                      </div>
                    )}
                  </>
                )}

                {record.approvalType === 'BREAK_OVERRIDE' && (
                  <>
                    <div style={styles.detailRow}>
                      <span style={styles.detailLabel}>Break Time:</span>
                      <span style={styles.detailValue}>
                        {formatTime(record.breakStartTime)} - {formatTime(record.breakEndTime)}
                      </span>
                    </div>
                    <div style={styles.detailRow}>
                      <span style={styles.detailLabel}>Duration:</span>
                      <span style={styles.detailValue}>{record.breakDurationMinutes} minutes</span>
                    </div>
                    <div style={styles.detailRow}>
                      <span style={styles.detailLabel}>Location:</span>
                      <span style={styles.detailValue}>{record.location || 'ED'}</span>
                    </div>
                    {record.reason && (
                      <div style={styles.detailRow}>
                        <span style={styles.detailLabel}>Reason:</span>
                        <p style={styles.reasonText}>{record.reason}</p>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Action Section */}
              {record.status === 'PENDING' && (
                <div style={styles.actionSection}>
                  {actingUpon === record._id ? (
                    <>
                      <textarea
                        placeholder="Add optional notes (visible to nurse)..."
                        value={actionNotes}
                        onChange={(e) => setActionNotes(e.target.value)}
                        style={styles.notesInput}
                      />
                      <div style={styles.actionButtons}>
                        <button
                          onClick={() => handleApprove(record._id)}
                          style={styles.approveBtn}
                        >
                          <FaCheck style={{ marginRight: '6px' }} />
                          Approve
                        </button>
                        <button
                          onClick={() => handleReject(record._id)}
                          style={styles.rejectBtn}
                        >
                          <FaTimes style={{ marginRight: '6px' }} />
                          Reject
                        </button>
                        <button
                          onClick={() => {
                            setActingUpon(null);
                            setActionNotes('');
                          }}
                          style={styles.cancelBtn}
                        >
                          Cancel
                        </button>
                      </div>
                    </>
                  ) : (
                    <button
                      onClick={() => setActingUpon(record._id)}
                      style={styles.reviewBtn}
                    >
                      <FaClock style={{ marginRight: '6px' }} />
                      Review & Decide
                    </button>
                  )}
                </div>
              )}

              {/* Approved/Rejected Info */}
              {record.status !== 'PENDING' && (
                <div style={styles.completedInfo}>
                  <span style={{ fontWeight: '600' }}>
                    {record.status === 'APPROVED' ? '✓ Approved' : '✗ Rejected'}
                  </span>
                  {record.managerNotes && (
                    <p style={styles.managerNotesLabel}>Notes: {record.managerNotes}</p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Summary Card */}
      {records.length > 0 && (
        <div style={styles.summaryCard}>
          <h3>Summary</h3>
          <div style={styles.summaryGrid}>
            <div style={styles.summaryItem}>
              <span style={styles.summaryValue}>
                {records.filter(r => r.status === 'PENDING').length}
              </span>
              <span style={styles.summaryLabel}>Pending Actions</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryValue}>
                {records.filter(r => r.status === 'APPROVED').length}
              </span>
              <span style={styles.summaryLabel}>Approved This Week</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryValue}>
                {records.filter(r => r.status === 'REJECTED').length}
              </span>
              <span style={styles.summaryLabel}>Rejected This Week</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: '20px',
    maxWidth: '1000px'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px'
  },
  refreshBtn: {
    padding: '10px 20px',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600'
  },
  errorBanner: {
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px'
  },
  tabs: {
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
  filterCard: {
    backgroundColor: 'white',
    padding: '16px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    marginBottom: '24px'
  },
  filterRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '16px',
    marginBottom: '12px'
  },
  filterGroup: {
    display: 'flex',
    flexDirection: 'column'
  },
  filterLabel: {
    fontWeight: '600',
    color: '#374151',
    marginBottom: '6px',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center'
  },
  select: {
    padding: '8px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    fontFamily: 'inherit'
  },
  input: {
    padding: '8px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    fontFamily: 'inherit'
  },
  resultCount: {
    fontSize: '13px',
    color: '#6b7280',
    marginTop: '12px'
  },
  loadingBox: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#6b7280',
    fontSize: '16px'
  },
  emptyBox: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    color: '#9ca3af',
    fontSize: '16px'
  },
  recordsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    marginBottom: '24px'
  },
  recordCard: {
    padding: '20px',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
    backgroundColor: '#fef3c7'
  },
  recordHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '16px',
    paddingBottom: '12px',
    borderBottom: '1px solid rgba(0,0,0,0.1)'
  },
  recordTitle: {
    margin: '0 0 4px 0',
    fontSize: '18px',
    fontWeight: '700',
    color: '#1f2937'
  },
  recordInfo: {
    margin: '4px 0',
    fontSize: '14px',
    color: '#6b7280'
  },
  recordDate: {
    margin: '4px 0 0 0',
    fontSize: '13px',
    color: '#9ca3af'
  },
  statusBadge: {
    padding: '6px 12px',
    borderRadius: '4px',
    color: 'white',
    fontWeight: '600',
    fontSize: '12px'
  },
  recordDetails: {
    marginBottom: '16px'
  },
  detailRow: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '12px',
    fontSize: '14px'
  },
  detailLabel: {
    fontWeight: '600',
    color: '#374151'
  },
  detailValue: {
    color: '#6b7280'
  },
  reasonText: {
    margin: '8px 0 0 0',
    padding: '8px 12px',
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderRadius: '4px',
    fontSize: '13px',
    color: '#4b5563'
  },
  actionSection: {
    paddingTop: '16px',
    borderTop: '1px solid rgba(0,0,0,0.1)'
  },
  notesInput: {
    width: '100%',
    padding: '10px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    fontFamily: 'inherit',
    marginBottom: '12px',
    minHeight: '80px'
  },
  actionButtons: {
    display: 'flex',
    gap: '8px'
  },
  approveBtn: {
    flex: 1,
    padding: '10px 16px',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  rejectBtn: {
    flex: 1,
    padding: '10px 16px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  cancelBtn: {
    padding: '10px 16px',
    backgroundColor: '#9ca3af',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600'
  },
  reviewBtn: {
    width: '100%',
    padding: '12px 16px',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  completedInfo: {
    paddingTop: '12px',
    borderTop: '1px solid rgba(0,0,0,0.1)',
    color: '#6b7280',
    fontSize: '14px'
  },
  managerNotesLabel: {
    margin: '6px 0 0 0',
    fontSize: '13px',
    fontStyle: 'italic'
  },
  summaryCard: {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb'
  },
  summaryGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '16px',
    marginTop: '16px'
  },
  summaryItem: {
    textAlign: 'center',
    padding: '16px',
    backgroundColor: '#f9fafb',
    borderRadius: '6px'
  },
  summaryValue: {
    display: 'block',
    fontSize: '28px',
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: '6px'
  },
  summaryLabel: {
    display: 'block',
    fontSize: '13px',
    color: '#6b7280'
  }
};
