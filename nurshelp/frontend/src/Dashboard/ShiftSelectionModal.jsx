// src/Dashboard/ShiftSelectionModal.jsx
import React, { useState, useEffect } from 'react';
import { FaClock, FaMapMarkerAlt, FaTimes } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function ShiftSelectionModal({ isOpen, onClose, onSelectShift, token }) {
  const [shifts, setShifts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedShiftId, setSelectedShiftId] = useState(null);
  const [isClocking, setIsClocking] = useState(false);

  useEffect(() => {
    if (isOpen && token) {
      fetchAvailableShifts();
    }
  }, [isOpen, token]);

  const fetchAvailableShifts = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/available-shifts`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setShifts(data.shifts);
      } else {
        setError(data.message || 'Failed to fetch shifts');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClockIn = async () => {
    if (!selectedShiftId) {
      setError('Please select a shift');
      return;
    }

    setIsClocking(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/clock-in`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          shiftId: selectedShiftId,
          location: 'ED'
        })
      });
      const data = await res.json();
      if (data.success) {
        onSelectShift(data.clockRecord);
        onClose();
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsClocking(false);
    }
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return '--:--';
    return timeStr;
  };

  const isToday = (date) => {
    const today = new Date();
    const shiftDate = new Date(date);
    return today.toDateString() === shiftDate.toDateString();
  };

  const sortedShifts = [...shifts].sort((a, b) => {
    const aToday = isToday(a.date);
    const bToday = isToday(b.date);
    if (aToday !== bToday) return aToday ? -1 : 1;
    return new Date(a.date) - new Date(b.date);
  });

  if (!isOpen) return null;

  return (
    <div style={styles.modalOverlay}>
      <div style={styles.modalContent}>
        <div style={styles.modalHeader}>
          <h2>Select Your Shift</h2>
          <button
            onClick={onClose}
            style={styles.closeButton}
            disabled={isClocking}
          >
            <FaTimes size={20} />
          </button>
        </div>

        {error && (
          <div style={styles.errorBanner}>
            ❌ {error}
          </div>
        )}

        {loading ? (
          <div style={styles.loadingContainer}>⏳ Loading shifts...</div>
        ) : shifts.length === 0 ? (
          <div style={styles.emptyContainer}>
            📋 No scheduled shifts for the next 7 days
          </div>
        ) : (
          <>
            <div style={styles.shiftsGrid}>
              {sortedShifts.map((shift) => {
                const isToday_ = isToday(shift.date);
                return (
                  <div
                    key={shift._id}
                    style={{
                      ...styles.shiftCard,
                      ...(selectedShiftId === shift._id ? styles.shiftCardSelected : {}),
                      ...(isToday_ ? styles.shiftCardToday : {})
                    }}
                    onClick={() => setSelectedShiftId(shift._id)}
                  >
                    <div style={styles.shiftDate}>
                      {isToday_ ? '📌 TODAY' : new Date(shift.date).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric'
                      })}
                    </div>

                    <div style={styles.shiftTime}>
                      <FaClock style={styles.icon} />
                      {formatTime(shift.shiftStart)} - {formatTime(shift.shiftEnd)}
                    </div>

                    <div style={styles.shiftLocation}>
                      <FaMapMarkerAlt style={styles.icon} />
                      {shift.location || 'ED'}
                    </div>

                    {shift.breakStart && (
                      <div style={styles.breakInfo}>
                        ☕ Break: {formatTime(shift.breakStart)} - {formatTime(shift.breakEnd)}
                      </div>
                    )}

                    {shift.assignedBeds && shift.assignedBeds.length > 0 && (
                      <div style={styles.bedsInfo}>
                        🛏️ Beds: {shift.assignedBeds.join(', ')}
                      </div>
                    )}

                    {selectedShiftId === shift._id && (
                      <div style={styles.selectedBadge}>✓ SELECTED</div>
                    )}
                  </div>
                );
              })}
            </div>

            <div style={styles.actionButtons}>
              <button
                onClick={onClose}
                style={styles.cancelBtn}
                disabled={isClocking}
              >
                Cancel
              </button>
              <button
                onClick={handleClockIn}
                style={{
                  ...styles.clockInBtn,
                  ...(isClocking || !selectedShiftId ? styles.btnDisabled : {})
                }}
                disabled={isClocking || !selectedShiftId}
              >
                {isClocking ? '⏳ Clocking In...' : '✓ Clock In'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

const styles = {
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
    maxWidth: '600px',
    width: '90%',
    maxHeight: '80vh',
    overflowY: 'auto',
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
    padding: '20px'
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    borderBottom: '2px solid #e5e7eb',
    paddingBottom: '15px'
  },
  closeButton: {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    color: '#6b7280',
    fontSize: '24px',
    padding: '5px'
  },
  errorBanner: {
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '15px',
    borders: '1px solid #fecaca'
  },
  loadingContainer: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#6b7280',
    fontSize: '16px'
  },
  emptyContainer: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#9ca3af',
    fontSize: '16px'
  },
  shiftsGrid: {
    display: 'grid',
    gap: '12px',
    marginBottom: '20px'
  },
  shiftCard: {
    border: '2px solid #e5e7eb',
    borderRadius: '8px',
    padding: '16px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    backgroundColor: '#f9fafb'
  },
  shiftCardToday: {
    backgroundColor: '#f0fdf4',
    borderColor: '#86efac'
  },
  shiftCardSelected: {
    borderColor: '#10b981',
    backgroundColor: '#ecfdf5',
    boxShadow: '0 0 0 3px rgba(16,185,129,0.1)'
  },
  shiftDate: {
    fontWeight: '600',
    fontSize: '14px',
    color: '#059669',
    marginBottom: '8px'
  },
  shiftTime: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '16px',
    fontWeight: '500',
    color: '#1f2937',
    marginBottom: '8px'
  },
  shiftLocation: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '14px',
    color: '#6b7280',
    marginBottom: '8px'
  },
  breakInfo: {
    fontSize: '13px',
    color: '#8b5cf6',
    marginBottom: '6px',
    paddingLeft: '4px'
  },
  bedsInfo: {
    fontSize: '13px',
    color: '#0891b2',
    marginBottom: '6px',
    paddingLeft: '4px'
  },
  icon: {
    fontSize: '14px'
  },
  selectedBadge: {
    display: 'inline-block',
    backgroundColor: '#10b981',
    color: 'white',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '12px',
    fontWeight: '600',
    marginTop: '8px'
  },
  actionButtons: {
    display: 'flex',
    gap: '12px',
    justifyContent: 'flex-end'
  },
  cancelBtn: {
    padding: '10px 20px',
    backgroundColor: '#e5e7eb',
    color: '#374151',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '500',
    transition: 'all 0.2s'
  },
  clockInBtn: {
    padding: '10px 24px',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    transition: 'all 0.2s'
  },
  btnDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed'
  }
};
