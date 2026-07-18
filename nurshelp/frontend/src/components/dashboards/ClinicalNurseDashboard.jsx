import React, { useState, useEffect } from 'react';
import { decodeJwt } from '../../utils/jwt';
import './ClinicalNurseDashboard.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

/**
 * ClinicalNurseDashboard Component
 * 
 * CLINICAL NURSE CONTEXT
 * Role: Manage assigned patients, acknowledge handovers, record vitals, escalate concerns
 * Controls:
 *   - View patients assigned to this nurse
 *   - Acknowledge patient handover
 *   - Mark as "Cared For" when starting care
 *   - Record and update vitals
 *   - Document medications / treatments
 *   - Escalate concerns to physician
 *   - Mark patients for disposition
 */

const ClinicalNurseDashboard = ({ config, context = 'ER' }) => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [acknowledging, setAcknowledging] = useState({});
  const [caringFor, setCaringFor] = useState({});

  const contextLabel = {
    'ER': 'Emergency Room',
    'ICU': 'Intensive Care Unit',
    'SURGICAL': 'Surgical Ward'
  }[context] || context;

  // Fetch assigned patients
  useEffect(() => {
    fetchAssignedPatients();
    const interval = setInterval(fetchAssignedPatients, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchAssignedPatients = async () => {
    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      const clinicianId = decoded?.id;

      if (!clinicianId) {
        setError('Could not identify clinician');
        setLoading(false);
        return;
      }

      const res = await fetch(`${BACKEND_URL}/api/encounters/clinician/${clinicianId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!res.ok) {
        throw new Error(`Failed to fetch assigned patients: ${res.status}`);
      }

      const data = await res.json();
      setPatients(data.cases || []);
      setError(null);
    } catch (err) {
      console.error('[CLINICAL NURSE] Fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAcknowledge = async (encounterId) => {
    try {
      setAcknowledging(prev => ({ ...prev, [encounterId]: true }));
      const token = localStorage.getItem('token');

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover/acknowledge`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          acknowledgedNotes: 'Patient acknowledged and ready for care'
        })
      });

      if (!res.ok) {
        throw new Error('Failed to acknowledge handover');
      }

      await fetchAssignedPatients();
    } catch (err) {
      console.error('[CLINICAL NURSE] Acknowledge error:', err);
      alert(`Error: ${err.message}`);
    } finally {
      setAcknowledging(prev => ({ ...prev, [encounterId]: false }));
    }
  };

  const handleCaredFor = async (encounterId) => {
    try {
      setCaringFor(prev => ({ ...prev, [encounterId]: true }));
      const token = localStorage.getItem('token');

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover/acknowledge`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          acknowledgedNotes: 'Care provided to patient',
          careedFor: true
        })
      });

      if (!res.ok) {
        throw new Error('Failed to mark patient as cared for');
      }

      await fetchAssignedPatients();
    } catch (err) {
      console.error('[CLINICAL NURSE] Care for error:', err);
      alert(`Error: ${err.message}`);
    } finally {
      setCaringFor(prev => ({ ...prev, [encounterId]: false }));
    }
  };

  return (
    <div className="clinical-nurse-dashboard">
      {/* Header */}
      <header className="dashboard-header clinical-header">
        <div className="header-left">
          <h1>⚕️ CLINICAL NURSE - {contextLabel}</h1>
          <p className="subtitle">{config?.staffName} • {config?.staffDepartment}</p>
        </div>
        <div className="header-right">
          <div className="shift-info">
            <span className="badge badge-active">Active Shift</span>
            <span className="time">{config?.shiftStartTime} - {config?.shiftEndTime}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-content">
        {/* Patient Assignments Table */}
        <section className="patient-assignments">
          <div className="section-header">
            <h2>📋 Your Assigned Patients</h2>
            <p className="count">Total: {patients.length}</p>
          </div>

          {loading ? (
            <div className="loading">Loading your patients...</div>
          ) : error ? (
            <div className="error">Error: {error}</div>
          ) : patients.length === 0 ? (
            <div className="no-patients">No patients assigned to you yet</div>
          ) : (
            <div className="table-wrapper">
              <table className="assignments-table">
                <thead>
                  <tr>
                    <th>Patient ID</th>
                    <th>Patient Name</th>
                    <th>Severity</th>
                    <th>Handover Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(pt => {
                    const handover = pt.handover || {};
                    const patientName = pt.patientName || pt.firstName || 'Unknown';
                    const severity = pt.triage?.aiTriageReport?.esi_level || pt.severity || 4;
                    const severityMap = {
                      1: { label: 'CRITICAL', color: '#991b1b' },
                      2: { label: 'HIGH', color: '#9a3412' },
                      3: { label: 'MODERATE', color: '#92400e' },
                      4: { label: 'LOW', color: '#166534' },
                      5: { label: 'LOW', color: '#166534' }
                    };
                    const severityConfig = severityMap[severity] || severityMap[4];

                    return (
                      <tr key={pt._id}>
                        <td>{pt._id?.substring(0, 8)}</td>
                        <td><strong>{patientName}</strong></td>
                        <td>
                          <span style={{ color: severityConfig.color, fontWeight: 600 }}>
                            {severityConfig.label}
                          </span>
                        </td>
                        <td>
                          <span className={`status ${handover.handover_status?.toLowerCase() || 'notified'}`}>
                            {handover.handover_status || 'NOTIFIED'}
                          </span>
                        </td>
                        <td>
                          {handover.handover_status !== 'ACKNOWLEDGED' ? (
                            <button
                              onClick={() => handleAcknowledge(pt._id)}
                              disabled={acknowledging[pt._id]}
                              className="btn btn-primary"
                            >
                              {acknowledging[pt._id] ? '⏳ Acknowledging...' : '✓ Acknowledge'}
                            </button>
                          ) : (
                            <button
                              onClick={() => handleCaredFor(pt._id)}
                              disabled={caringFor[pt._id]}
                              className="btn btn-success"
                            >
                              {caringFor[pt._id] ? '⏳ Saving...' : '💊 Cared For'}
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default ClinicalNurseDashboard;
