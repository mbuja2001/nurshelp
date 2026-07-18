// src/Dashboard/BreakRequestModal.jsx
import React, { useState, useEffect } from 'react';
import { FaTimes, FaCheckCircle, FaHourglass, FaTimesCircle } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function BreakRequestModal({ isOpen, onClose, token, clockStatus }) {
  const [reason, setReason] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [submitted, setSubmitted] = useState(false);

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setReason('');
      setError(null);
      setSubmitted(false);
    }
  }, [isOpen]);

  const handleSubmit = async (e) => {
    e?.preventDefault?.();
    
    if (!reason.trim()) {
      setError('Please provide a reason for the break');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/break-request`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ reason: reason.trim() })
      });
      
      const data = await res.json();
      
      if (data.success) {
        setSubmitted(true);
        setReason('');
        setTimeout(() => {
          onClose?.();
        }, 2000);
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const breakStatus = clockStatus?.clockRecord?.breakApprovalStatus || 'NOT_REQUESTED';
  const breakReason = clockStatus?.clockRecord?.breakReason;
  const breakRejectionReason = clockStatus?.clockRecord?.breakRejectionReason;
  const breakApprovalTime = clockStatus?.clockRecord?.breakApprovalTime;

  const getStatusDisplay = () => {
    switch (breakStatus) {
      case 'APPROVED':
        return {
          icon: <FaCheckCircle />,
          color: '#10b981',
          title: '✓ Break Approved',
          message: 'Your break request has been approved.'
        };
      case 'PENDING':
        return {
          icon: <FaHourglass />,
          color: '#f59e0b',
          title: '⏳ Pending Approval',
          message: 'Your break request is awaiting manager approval.'
        };
      case 'REJECTED':
        return {
          icon: <FaTimesCircle />,
          color: '#ef4444',
          title: '✗ Break Denied',
          message: breakRejectionReason || 'Your break request was not approved.'
        };
      default:
        return null;
    }
  };

  const statusDisplay = getStatusDisplay();

  return (
    <div style={styles.modalOverlay}>
      <div style={styles.modalContent}>
        <div style={styles.modalHeader}>
          <h2>Break Request</h2>
          <button
            onClick={onClose}
            style={styles.closeButton}
            disabled={loading}
          >
            <FaTimes size={20} />
          </button>
        </div>

        {error && (
          <div style={styles.errorBanner}>
            ❌ {error}
          </div>
        )}

        {submitted && (
          <div style={styles.successBanner}>
            ✓ Break request submitted! Redirecting...
          </div>
        )}

        {statusDisplay && breakStatus !== 'NOT_REQUESTED' ? (
          <div style={{ ...styles.statusContainer, borderLeftColor: statusDisplay.color }}>
            <div style={{ fontSize: '32px', marginBottom: '12px', color: statusDisplay.color }}>
              {statusDisplay.icon}
            </div>
            <h3 style={{ color: statusDisplay.color, marginBottom: '8px' }}>
              {statusDisplay.title}
            </h3>
            <p style={{ color: '#6b7280', marginBottom: '15px' }}>
              {statusDisplay.message}
            </p>

            {breakReason && (
              <div style={styles.infoBox}>
                <strong>Reason:</strong> {breakReason}
              </div>
            )}

            {breakRejectionReason && (
              <div style={styles.infoBox}>
                <strong>Reason for denial:</strong> {breakRejectionReason}
              </div>
            )}

            {breakApprovalTime && (
              <div style={styles.infoBox}>
                <strong>As of:</strong> {new Date(breakApprovalTime).toLocaleTimeString()}
              </div>
            )}

            <div style={styles.actionButtons}>
              <button
                onClick={onClose}
                style={styles.closeBtn}
              >
                Close
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} style={styles.form}>
            <div style={styles.formGroup}>
              <label style={styles.label}>
                Reason for break (required)
              </label>
              <textarea
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="e.g., Lunch break, restroom break, personal matter..."
                style={styles.textarea}
                disabled={loading}
                maxLength={200}
              />
              <div style={styles.charCount}>
                {reason.length}/200
              </div>
            </div>

            <div style={styles.infoBox}>
              <p style={{ margin: '0 0 8px 0', fontWeight: '500' }}>
                💡 Tip: Be specific about your break reason
              </p>
              <p style={{ margin: '0', fontSize: '13px', color: '#6b7280' }}>
                This helps managers quickly approve break requests and maintain ward coverage.
              </p>
            </div>

            <div style={styles.actionButtons}>
              <button
                type="button"
                onClick={onClose}
                style={styles.cancelBtn}
                disabled={loading}
              >
                Cancel
              </button>
              <button
                type="submit"
                style={{
                  ...styles.submitBtn,
                  ...(loading ? styles.btnDisabled : {})
                }}
                disabled={loading || !reason.trim()}
              >
                {loading ? '⏳ Submitting...' : '📤 Request Break'}
              </button>
            </div>
          </form>
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
    zIndex: 1001
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: '12px',
    maxWidth: '500px',
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
    border: '1px solid #fecaca'
  },
  successBanner: {
    backgroundColor: '#dcfce7',
    color: '#166534',
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '15px',
    border: '1px solid #bbf7d0'
  },
  statusContainer: {
    backgroundColor: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderLeft: '4px solid #10b981',
    padding: '20px',
    borderRadius: '8px'
  },
  infoBox: {
    backgroundColor: '#f3f4f6',
    padding: '12px',
    borderRadius: '6px',
    fontSize: '13px',
    marginBottom: '12px',
    color: '#374151'
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px'
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  label: {
    fontWeight: '600',
    color: '#1f2937',
    fontSize: '14px'
  },
  textarea: {
    padding: '12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontFamily: 'inherit',
    fontSize: '14px',
    resize: 'vertical',
    minHeight: '100px',
    transition: 'border-color 0.2s'
  },
  charCount: {
    fontSize: '12px',
    color: '#9ca3af',
    textAlign: 'right'
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
  submitBtn: {
    padding: '10px 24px',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    transition: 'all 0.2s'
  },
  closeBtn: {
    padding: '10px 20px',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '500',
    transition: 'all 0.2s'
  },
  btnDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed'
  }
};
