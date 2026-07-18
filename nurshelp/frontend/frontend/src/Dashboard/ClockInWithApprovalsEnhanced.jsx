// src/Dashboard/ClockInWithApprovals.jsx
import React, { useState, useEffect } from 'react';
import { FaClock, FaPlay, FaPause, FaSignOutAlt, FaCheckCircle, FaExclamationTriangle, FaHourglass } from 'react-icons/fa';
import ShiftSelectionModal from './ShiftSelectionModal';
import BreakRequestModal from './BreakRequestModal';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function ClockInWithApprovals({ token }) {
  const [clockStatus, setClockStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [showShiftModal, setShowShiftModal] = useState(false);
  const [showBreakModal, setShowBreakModal] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [actionError, setActionError] = useState(null);
  const [notes, setNotes] = useState('');
  const [showNotes, setShowNotes] = useState(false);

  // Fetch clock status on mount and periodically
  useEffect(() => {
    if (token) {
      fetchClockStatus();
      const interval = setInterval(fetchClockStatus, 5000); // Every 5 seconds
      return () => clearInterval(interval);
    }
  }, [token]);

  // Update elapsed time every second
  useEffect(() => {
    if (clockStatus?.clockRecord?.status === 'CLOCKED_IN' || clockStatus?.clockRecord?.status === 'ON_BREAK') {
      const interval = setInterval(() => {
        const clockInTime = new Date(clockStatus.clockRecord.clockInTime);
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
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClockOut = async () => {
    setActionLoading(true);
    setActionError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/clock-out`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ notes: notes.trim() })
      });
      const data = await res.json();
      if (data.success) {
        setNotes('');
        setShowNotes(false);
        await fetchClockStatus();
      } else {
        setActionError(data.message);
      }
    } catch (err) {
      setActionError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleBreakStart = async () => {
    setActionLoading(true);
    setActionError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/break-start`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        await fetchClockStatus();
      } else {
        setActionError(data.message);
      }
    } catch (err) {
      setActionError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleBreakEnd = async () => {
    setActionLoading(true);
    setActionError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/break-end`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        await fetchClockStatus();
      } else {
        setActionError(data.message);
      }
    } catch (err) {
      setActionError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const formatTime = (date) => {
    if (!date) return '--:--';
    return new Date(date).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatDuration = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loadingBox}>⏳ Loading clock status...</div>
      </div>
    );
  }

  const status = clockStatus?.clockRecord?.status;
  const isClocked = status === 'CLOCKED_IN' || status === 'ON_BREAK';
  const lateArrivalStatus = clockStatus?.clockRecord?.lateArrivalApprovalStatus;
  const breakApprovalStatus = clockStatus?.clockRecord?.breakApprovalStatus;
  const lateArrivalMinutes = clockStatus?.clockRecord?.lateArrivalMinutes;

  return (
    <div style={styles.container}>
      <ShiftSelectionModal
        isOpen={showShiftModal}
        onClose={() => setShowShiftModal(false)}
        onSelectShift={async () => {
          await fetchClockStatus();
        }}
        token={token}
      />

      <BreakRequestModal
        isOpen={showBreakModal}
        onClose={() => {
          setShowBreakModal(false);
          fetchClockStatus();
        }}
        token={token}
        clockStatus={clockStatus}
      />

      {error && (
        <div style={styles.errorBanner}>
          ⚠️ {error}
        </div>
      )}

      {actionError && (
        <div style={styles.errorBanner}>
          ❌ {actionError}
        </div>
      )}

      {/* Pending Approvals Badges */}
      {isClocked && (
        <>
          {lateArrivalStatus === 'PENDING' && (
            <div style={styles.warningBadge}>
              <FaExclamationTriangle style={{ marginRight: '8px' }} />
              ⏳ {lateArrivalMinutes} min late - Pending manager approval
            </div>
          )}
          {lateArrivalStatus === 'REJECTED' && (
            <div style={styles.errorBadge}>
              <FaExclamationTriangle style={{ marginRight: '8px' }} />
              ❌ Late arrival not approved
            </div>
          )}
          {breakApprovalStatus === 'PENDING' && (
            <div style={styles.infoBadge}>
              <FaHourglass style={{ marginRight: '8px' }} />
              ⏳ Break request - Awaiting approval
            </div>
          )}
          {breakApprovalStatus === 'APPROVED' && (
            <div style={styles.successBadge}>
              <FaCheckCircle style={{ marginRight: '8px' }} />
              ✓ Break approved
            </div>
          )}
          {breakApprovalStatus === 'REJECTED' && (
            <div style={styles.errorBadge}>
              <FaExclamationTriangle style={{ marginRight: '8px' }} />
              ❌ Break request denied
            </div>
          )}
        </>
      )}

      <div style={styles.mainCard}>
        {/* Status Display */}
        <div style={styles.statusSection}>
          <div style={{
            ...styles.statusBadge,
            backgroundColor: status === 'CLOCKED_IN' ? '#10b981' :
              status === 'ON_BREAK' ? '#f59e0b' :
              status === 'CLOCKED_OUT' ? '#6c757d' : '#e5e7eb'
          }}>
            {status === 'CLOCKED_IN' && '✓ CLOCKED IN'}
            {status === 'ON_BREAK' && '☕ ON BREAK'}
            {status === 'CLOCKED_OUT' && '🔴 CLOCKED OUT'}
            {!status && '⚪ NOT CLOCKED IN'}
          </div>

          {isClocked && (
            <div style={styles.timeDisplay}>
              <div style={styles.timeItem}>
                <span style={styles.timeLabel}>Clocked In</span>
                <span style={styles.timeValue}>{formatTime(clockStatus?.clockRecord?.clockInTime)}</span>
              </div>
              <div style={styles.timeItem}>
                <span style={styles.timeLabel}>Elapsed</span>
                <span style={styles.timeValue}>{formatDuration(elapsedTime)}</span>
              </div>
              {clockStatus?.clockRecord?.breakStartTime && (
                <div style={styles.timeItem}>
                  <span style={styles.timeLabel}>Break Started</span>
                  <span style={styles.timeValue}>{formatTime(clockStatus?.clockRecord?.breakStartTime)}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div style={styles.buttonGrid}>
          {!isClocked ? (
            <button
              onClick={() => setShowShiftModal(true)}
              style={{...styles.button, ...styles.clockInButton}}
              disabled={actionLoading}
            >
              <FaClock size={32} style={styles.buttonIcon} />
              <span>CLOCK IN</span>
            </button>
          ) : (
            <>
              {status === 'CLOCKED_IN' && (
                <>
                  <button
                    onClick={handleBreakStart}
                    style={{...styles.button, ...styles.breakButton}}
                    disabled={actionLoading || breakApprovalStatus === 'PENDING'}
                    title={breakApprovalStatus === 'PENDING' ? 'Awaiting break approval' : 'Start break'}
                  >
                    <FaPlay size={28} style={styles.buttonIcon} />
                    <span>START BREAK</span>
                  </button>

                  <button
                    onClick={() => setShowBreakModal(true)}
                    style={{...styles.button, ...styles.requestBreakButton}}
                    disabled={actionLoading || breakApprovalStatus === 'PENDING'}
                  >
                    <FaPlay size={28} style={styles.buttonIcon} />
                    <span>REQUEST BREAK</span>
                  </button>
                </>
              )}

              {status === 'ON_BREAK' && (
                <button
                  onClick={handleBreakEnd}
                  style={{...styles.button, ...styles.endBreakButton}}
                  disabled={actionLoading}
                >
                  <FaPause size={32} style={styles.buttonIcon} />
                  <span>END BREAK</span>
                </button>
              )}

              <button
                onClick={() => setShowNotes(true)}
                style={{...styles.button, ...styles.clockOutButton}}
                disabled={actionLoading}
              >
                <FaSignOutAlt size={32} style={styles.buttonIcon} />
                <span>CLOCK OUT</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Clock Out Notes Modal */}
      {showNotes && (
        <div style={styles.modalOverlay}>
          <div style={styles.notesModal}>
            <h3 style={styles.notesTitle}>Clock Out Notes (Optional)</h3>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g., 'Extended due to emergency', 'Assisted with discharge'..."
              style={styles.notesTextarea}
              maxLength={200}
            />
            <div style={styles.charCount}>{notes.length}/200</div>
            <div style={styles.notesButtons}>
              <button
                onClick={() => setShowNotes(false)}
                style={styles.notesCancel}
                disabled={actionLoading}
              >
                Cancel
              </button>
              <button
                onClick={handleClockOut}
                style={{...styles.notesSubmit, ...(actionLoading ? {opacity: 0.6} : {})}}
                disabled={actionLoading}
              >
                {actionLoading ? '⏳ Clocking Out...' : '✓ Clock Out'}
              </button>
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
    maxWidth: '500px'
  },
  errorBanner: {
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #fecaca'
  },
  warningBadge: {
    backgroundColor: '#fef3c7',
    color: '#92400e',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #fcd34d',
    display: 'flex',
    alignItems: 'center'
  },
  infoBadge: {
    backgroundColor: '#dbeafe',
    color: '#0c4a6e',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #bfdbfe',
    display: 'flex',
    alignItems: 'center'
  },
  successBadge: {
    backgroundColor: '#dcfce7',
    color: '#166534',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #bbf7d0',
    display: 'flex',
    alignItems: 'center'
  },
  errorBadge: {
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #fecaca',
    display: 'flex',
    alignItems: 'center'
  },
  mainCard: {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    border: '2px solid #e5e7eb'
  },
  statusSection: {
    marginBottom: '24px'
  },
  statusBadge: {
    display: 'inline-block',
    color: 'white',
    padding: '12px 20px',
    borderRadius: '8px',
    fontWeight: '600',
    fontSize: '16px',
    marginBottom: '16px'
  },
  timeDisplay: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px'
  },
  timeItem: {
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: '#f9fafb',
    padding: '12px',
    borderRadius: '6px',
    border: '1px solid #e5e7eb'
  },
  timeLabel: {
    fontSize: '12px',
    color: '#6b7280',
    fontWeight: '500',
    marginBottom: '4px'
  },
  timeValue: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#1f2937'
  },
  buttonGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px'
  },
  button: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px 16px',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '14px',
    transition: 'all 0.2s',
    minHeight: '100px'
  },
  clockInButton: {
    gridColumn: '1 / -1',
    backgroundColor: '#10b981',
    color: 'white'
  },
  breakButton: {
    backgroundColor: '#f59e0b',
    color: 'white'
  },
  requestBreakButton: {
    backgroundColor: '#8b5cf6',
    color: 'white'
  },
  endBreakButton: {
    gridColumn: '1 / -1',
    backgroundColor: '#f59e0b',
    color: 'white'
  },
  clockOutButton: {
    gridColumn: '1 / -1',
    backgroundColor: '#ef4444',
    color: 'white'
  },
  buttonIcon: {
    marginBottom: '8px'
  },
  loadingBox: {
    textAlign: 'center',
    padding: '40px',
    color: '#6b7280'
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
  notesModal: {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '24px',
    maxWidth: '400px',
    width: '90%',
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
  },
  notesTitle: {
    marginTop: 0,
    marginBottom: '16px',
    color: '#1f2937'
  },
  notesTextarea: {
    width: '100%',
    padding: '12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontFamily: 'inherit',
    fontSize: '14px',
    minHeight: '100px',
    marginBottom: '8px',
    boxSizing: 'border-box'
  },
  charCount: {
    fontSize: '12px',
    color: '#9ca3af',
    textAlign: 'right',
    marginBottom: '16px'
  },
  notesButtons: {
    display: 'flex',
    gap: '12px',
    justifyContent: 'flex-end'
  },
  notesCancel: {
    padding: '10px 20px',
    backgroundColor: '#e5e7eb',
    color: '#374151',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '500'
  },
  notesSubmit: {
    padding: '10px 24px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600'
  }
};
