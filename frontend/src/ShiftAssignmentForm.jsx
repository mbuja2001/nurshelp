import React, { useState, useEffect } from 'react';
import { FiUsers, FiAlertCircle, FiCheck } from 'react-icons/fi';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

/**
 * ShiftAssignmentForm Component
 * 
 * Admin UI to assign 3 critical roles to a shift:
 * 1. Triage Nurse (handles patient queue)
 * 2. Clinical Nurse (ER/ICU/SURGICAL - observes patients)
 * 3. Physician (ER/ICU/SURGICAL - medical decisions)
 * 
 * When submitted, calls POST /api/shift/assign-clinical-team
 * which determines which dashboard each staff member will see.
 */
export default function ShiftAssignmentForm({ shiftId, onSuccess, staffList = [] }) {
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState({ success: '', error: '' });
  const token = localStorage.getItem('token');

  // Assignment form state
  const [assignments, setAssignments] = useState({
    triageNurseId: '',
    clinicalNurseId: '',
    clinicalNurseContext: 'CLINICAL_NURSE_ER',
    physicianId: '',
    physicianContext: 'PHYSICIAN_ER'
  });

  const contextOptions = [
    { value: 'CLINICAL_NURSE_ER', label: 'ED (Emergency)' },
    { value: 'CLINICAL_NURSE_ICU', label: 'ICU' },
    { value: 'CLINICAL_NURSE_SURGICAL', label: 'Surgical' }
  ];

  const physicianContextOptions = [
    { value: 'PHYSICIAN_ER', label: 'ED (Emergency)' },
    { value: 'PHYSICIAN_ICU', label: 'ICU' },
    { value: 'PHYSICIAN_SURGICAL', label: 'Surgical' }
  ];

  // Filter staff by role
  const triageNurses = staffList.filter(s => s.role === 'NURSE' && s.specialty?.includes('Triage'));
  const clinicalNurses = staffList.filter(s => s.role === 'NURSE' && !s.specialty?.includes('Triage'));
  const physicians = staffList.filter(s => s.role === 'PHYSICIAN');

  const handleAssignmentChange = (field, value) => {
    setAssignments(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!shiftId) {
      setFeedback({ success: '', error: 'Shift ID is required' });
      return;
    }

    // Validate at least one role is assigned
    if (!assignments.triageNurseId && !assignments.clinicalNurseId && !assignments.physicianId) {
      setFeedback({ success: '', error: 'Please assign at least one role' });
      return;
    }

    setLoading(true);
    setFeedback({ success: '', error: '' });

    try {
      const payload = {
        shiftId,
        triageNurseId: assignments.triageNurseId || null,
        clinicalNurseId: assignments.clinicalNurseId || null,
        clinicalNurseContext: assignments.clinicalNurseContext,
        physicianId: assignments.physicianId || null,
        physicianContext: assignments.physicianContext
      };

      console.log('[SHIFT ASSIGNMENT]', 'Sending:', payload);

      const res = await fetch(`${BACKEND_URL}/api/shifts/assign-clinical-team`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (data.success) {
        setFeedback({
          success: '✓ Team assigned successfully! Dashboards will now be triggered for each staff member.',
          error: ''
        });

        // Reset form
        setAssignments({
          triageNurseId: '',
          clinicalNurseId: '',
          clinicalNurseContext: 'CLINICAL_NURSE_ER',
          physicianId: '',
          physicianContext: 'PHYSICIAN_ER'
        });

        // Callback to parent
        if (onSuccess) {
          setTimeout(() => onSuccess(), 1500);
        }
      } else {
        setFeedback({ success: '', error: data.message || 'Failed to assign team' });
      }
    } catch (err) {
      console.error('[SHIFT ASSIGNMENT ERROR]', err);
      setFeedback({ success: '', error: err.message || 'Network error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <FiUsers size={20} style={{ color: '#10b981' }} />
        <h3 style={styles.title}>Assign Team to Shift</h3>
      </div>

      {feedback.success && (
        <div style={styles.successMessage}>
          <FiCheck size={16} />
          {feedback.success}
        </div>
      )}

      {feedback.error && (
        <div style={styles.errorMessage}>
          <FiAlertCircle size={16} />
          {feedback.error}
        </div>
      )}

      <form onSubmit={handleSubmit} style={styles.form}>
        {/* Triage Nurse */}
        <div style={styles.fieldGroup}>
          <label style={styles.label}>
            🔵 Triage Nurse
            <span style={styles.description}> (Handles patient queue & chief complaints)</span>
          </label>
          <select
            value={assignments.triageNurseId}
            onChange={(e) => handleAssignmentChange('triageNurseId', e.target.value)}
            style={styles.select}
          >
            <option value="">-- Select Triage Nurse --</option>
            {triageNurses.map(staff => (
              <option key={staff._id} value={staff._id}>
                {staff.name} {staff.surname} • {staff.specialty || 'Triage Nurse'}
              </option>
            ))}
          </select>
        </div>

        {/* Clinical Nurse */}
        <div style={styles.fieldGroup}>
          <label style={styles.label}>
            🔴 Clinical Nurse
            <span style={styles.description}> (Observes patients & manages vital signs)</span>
          </label>
          <div style={styles.fieldRow}>
            <select
              value={assignments.clinicalNurseId}
              onChange={(e) => handleAssignmentChange('clinicalNurseId', e.target.value)}
              style={{ ...styles.select, flex: 1 }}
            >
              <option value="">-- Select Clinical Nurse --</option>
              {clinicalNurses.map(staff => (
                <option key={staff._id} value={staff._id}>
                  {staff.name} {staff.surname} • {staff.specialty || 'RN'}
                </option>
              ))}
            </select>
            <select
              value={assignments.clinicalNurseContext}
              onChange={(e) => handleAssignmentChange('clinicalNurseContext', e.target.value)}
              style={{ ...styles.select, marginLeft: '10px', flex: 0.6 }}
              disabled={!assignments.clinicalNurseId}
            >
              {contextOptions.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Physician */}
        <div style={styles.fieldGroup}>
          <label style={styles.label}>
            🟩 Physician
            <span style={styles.description}> (Makes medical decisions & disposition)</span>
          </label>
          <div style={styles.fieldRow}>
            <select
              value={assignments.physicianId}
              onChange={(e) => handleAssignmentChange('physicianId', e.target.value)}
              style={{ ...styles.select, flex: 1 }}
            >
              <option value="">-- Select Physician --</option>
              {physicians.map(staff => (
                <option key={staff._id} value={staff._id}>
                  Dr. {staff.name} {staff.surname} • {staff.specialty || 'MD'}
                </option>
              ))}
            </select>
            <select
              value={assignments.physicianContext}
              onChange={(e) => handleAssignmentChange('physicianContext', e.target.value)}
              style={{ ...styles.select, marginLeft: '10px', flex: 0.6 }}
              disabled={!assignments.physicianId}
            >
              {physicianContextOptions.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Summary */}
        {(assignments.triageNurseId || assignments.clinicalNurseId || assignments.physicianId) && (
          <div style={styles.summary}>
            <strong style={{ color: '#1e293b' }}>📋 Assignment Summary:</strong>
            <ul style={styles.summaryList}>
              {assignments.triageNurseId && (
                <li>
                  🔵 Triage Nurse: {triageNurses.find(s => s._id === assignments.triageNurseId)?.name}
                </li>
              )}
              {assignments.clinicalNurseId && (
                <li>
                  🔴 Clinical Nurse: {clinicalNurses.find(s => s._id === assignments.clinicalNurseId)?.name} ({assignments.clinicalNurseContext})
                </li>
              )}
              {assignments.physicianId && (
                <li>
                  🟩 Physician: {physicians.find(s => s._id === assignments.physicianId)?.name} ({assignments.physicianContext})
                </li>
              )}
            </ul>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || (!assignments.triageNurseId && !assignments.clinicalNurseId && !assignments.physicianId)}
          style={{
            ...styles.submitButton,
            opacity: loading || (!assignments.triageNurseId && !assignments.clinicalNurseId && !assignments.physicianId) ? 0.6 : 1,
            cursor: loading || (!assignments.triageNurseId && !assignments.clinicalNurseId && !assignments.physicianId) ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Assigning...' : '→ Assign Team & Trigger Dashboards'}
        </button>
      </form>

      {/* Info Box */}
      <div style={styles.infoBox}>
        <strong style={{ color: '#059669' }}>💡 What Happens Next:</strong>
        <ul style={styles.infoList}>
          <li>Each assigned staff member will see ONLY their role's dashboard</li>
          <li>Triage Nurse sees patient queue and vital signs form</li>
          <li>Clinical Nurse sees observation patient management</li>
          <li>Physician sees 3-lane case management view</li>
          <li>Unassigned staff see "Off-Duty" message</li>
        </ul>
      </div>
    </div>
  );
}

const styles = {
  container: {
    background: '#fff',
    border: '1px solid #e2e8f0',
    borderRadius: '8px',
    padding: '20px',
    marginTop: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '20px',
    borderBottom: '2px solid #e5e7eb',
    paddingBottom: '12px'
  },
  title: {
    margin: 0,
    color: '#1e293b',
    fontSize: '16px',
    fontWeight: '600'
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px'
  },
  fieldGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  label: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#1e293b',
    display: 'flex',
    alignItems: 'baseline',
    gap: '6px'
  },
  description: {
    fontSize: '12px',
    fontWeight: '400',
    color: '#64748b'
  },
  fieldRow: {
    display: 'flex',
    gap: '10px',
    alignItems: 'stretch'
  },
  select: {
    padding: '10px 12px',
    border: '1px solid #cbd5e1',
    borderRadius: '6px',
    fontSize: '14px',
    color: '#1e293b',
    backgroundColor: '#fff',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
    outline: 'none'
  },
  summary: {
    background: '#f0fdf4',
    border: '1px solid #a7f3d0',
    borderRadius: '6px',
    padding: '12px',
    marginTop: '10px'
  },
  summaryList: {
    margin: '8px 0 0 0',
    paddingLeft: '20px',
    fontSize: '13px',
    color: '#1e7745',
    lineHeight: '1.6'
  },
  submitButton: {
    padding: '12px 16px',
    backgroundColor: '#10b981',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: '600',
    marginTop: '8px',
    transition: 'background 0.2s'
  },
  successMessage: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '12px',
    background: '#dcfce7',
    color: '#166534',
    borderRadius: '6px',
    fontSize: '13px',
    marginBottom: '16px',
    border: '1px solid #a7f3d0'
  },
  errorMessage: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '12px',
    background: '#fee2e2',
    color: '#991b1b',
    borderRadius: '6px',
    fontSize: '13px',
    marginBottom: '16px',
    border: '1px solid #fca5a5'
  },
  infoBox: {
    background: '#eff6ff',
    border: '1px solid #bfdbfe',
    borderRadius: '6px',
    padding: '12px',
    marginTop: '16px',
    fontSize: '13px',
    color: '#1e40af'
  },
  infoList: {
    margin: '8px 0 0 0',
    paddingLeft: '20px',
    lineHeight: '1.6'
  }
};
