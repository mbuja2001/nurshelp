import React, { useState, useEffect } from 'react';
import './ClinicalNurseDashboard.css';

/**
 * ClinicalNurseDashboard Component
 * 
 * CLINICAL NURSE CONTEXT
 * Role: Manage observation patients, record vitals, escalate concerns
 * Controls:
 *   - View observation patients
 *   - Record and update vitals
 *   - Document medications / treatments
 *   - Escalate concerns to physician
 *   - Mark patients for disposition
 */

const ClinicalNurseDashboard = ({ config, context = 'ER' }) => {
  const [patients, setPatients] = useState([]);
  const [activeTab, setActiveTab] = useState('observation');
  const [selectedPatient, setSelectedPatient] = useState(null);

  useEffect(() => {
    console.log('[CLINICAL NURSE]', context, '- Dashboard loaded');
    console.log('Config:', config);
    // TODO: Fetch patients in observation
  }, [config, context]);

  const contextLabel = {
    'ER': 'Emergency Room',
    'ICU': 'Intensive Care Unit',
    'SURGICAL': 'Surgical Ward'
  }[context] || context;

  return (
    <div className="clinical-nurse-dashboard">
      {/* Header */}
      <header className="dashboard-header clinical-header">
        <div className="header-left">
          <h1>⚕️ CLINICAL NURSE - {contextLabel}</h1>
          <p className="subtitle">{config.staffName} • {config.staffDepartment}</p>
        </div>
        <div className="header-right">
          <div className="shift-info">
            <span className="badge badge-active">Active Shift</span>
            <span className="time">{config.shiftStartTime} - {config.shiftEndTime}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-content">
        {/* Left Sidebar: Assigned Physician */}
        <aside className="sidebar">
          <div className="card physician-card">
            <h3>🩺 Assigned Physician</h3>
            <div className="physician-info">
              <div className="name">--</div>
              <div className="status available">
                <span className="indicator">●</span> Available
              </div>
              <div className="contact">
                <button className="btn btn-sm btn-secondary">📞 Call</button>
                <button className="btn btn-sm btn-secondary">💬 Message</button>
              </div>
            </div>
          </div>

          <div className="card alerts-card">
            <h3>⚠️ Alerts</h3>
            <div className="alert-list">
              <p className="no-alerts">No active alerts</p>
            </div>
          </div>

          <div className="card context-info">
            <h3>ℹ️ Context Info</h3>
            <div className="info-item">
              <span className="label">Department:</span>
              <span className="value">{context}</span>
            </div>
            <div className="info-item">
              <span className="label">Shift:</span>
              <span className="value">{config.shiftName}</span>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <section className="main-content">
          {/* Tab Navigation */}
          <div className="tab-navigation">
            <button 
              className={`tab ${activeTab === 'observation' ? 'active' : ''}`}
              onClick={() => setActiveTab('observation')}
            >
              🏥 Observation Patients
            </button>
            <button 
              className={`tab ${activeTab === 'vitals' ? 'active' : ''}`}
              onClick={() => setActiveTab('vitals')}
            >
              📊 Vitals Monitoring
            </button>
            <button 
              className={`tab ${activeTab === 'medications' ? 'active' : ''}`}
              onClick={() => setActiveTab('medications')}
            >
              💊 Medications
            </button>
          </div>

          {/* Observation Patients Tab */}
          {activeTab === 'observation' && (
            <div className="tab-content observation-content">
              <div className="patient-list-header">
                <h2>Patients Awaiting Disposition</h2>
                <span className="count">0 patients</span>
              </div>
              <div className="no-patients">
                <div className="empty-state">
                  <p className="emoji">😌</p>
                  <h3>No Patients in Observation</h3>
                  <p>Observation patients will appear here once triaged</p>
                </div>
              </div>
            </div>
          )}

          {/* Vitals Monitoring Tab */}
          {activeTab === 'vitals' && (
            <div className="tab-content vitals-content">
              <div className="vitals-header">
                <h2>Vitals Monitoring Dashboard</h2>
              </div>
              <div className="no-patients">
                <div className="empty-state">
                  <p className="emoji">📈</p>
                  <h3>No Patients Being Monitored</h3>
                  <p>Vital signs for observation patients will appear here</p>
                </div>
              </div>
            </div>
          )}

          {/* Medications Tab */}
          {activeTab === 'medications' && (
            <div className="tab-content medications-content">
              <div className="medications-header">
                <h2>Medication Administration Record</h2>
              </div>
              <div className="no-patients">
                <div className="empty-state">
                  <p className="emoji">💊</p>
                  <h3>No Medications to Administer</h3>
                  <p>Prescribed medications will appear here</p>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Footer: Quick Actions */}
      <footer className="dashboard-footer">
        <button className="btn btn-primary">
          ✏️ Record Vitals
        </button>
        <button className="btn btn-primary">
          ⬆️ Escalate Concern
        </button>
        <button className="btn btn-secondary">
          🔔 Set Break Request
        </button>
      </footer>
    </div>
  );
};

export default ClinicalNurseDashboard;
