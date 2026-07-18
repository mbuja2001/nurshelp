import React, { useState, useEffect } from 'react';
import './TriageNurseDashboard.css';

/**
 * TriageNurseDashboard Component
 * 
 * TRIAGE NURSE CONTEXT
 * Role: First point of contact, assess patients, assign priority
 * Controls: 
 *   - Accept patient from waiting area
 *   - Record vitals
 *   - Perform triage assessment
 *   - Route to clinical nurse / physician
 */

const TriageNurseDashboard = ({ config }) => {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [activeTab, setActiveTab] = useState('queue');

  useEffect(() => {
    console.log('[TRIAGE NURSE] Dashboard loaded');
    console.log('Config:', config);
    // TODO: Fetch patients waiting for triage
  }, [config]);

  return (
    <div className="triage-nurse-dashboard">
      {/* Header */}
      <header className="dashboard-header triage-header">
        <div className="header-left">
          <h1>🏥 TRIAGE NURSE - Emergency Department</h1>
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
        {/* Left Sidebar: Clinical Team Info */}
        <aside className="sidebar">
          <div className="card team-card">
            <h3>👥 Clinical Team</h3>
            <div className="team-members">
              <div className="member clinical-nurse">
                <span className="label">Clinical Nurse:</span>
                <span className="name">Assigned</span>
                <span className="status available">●</span>
              </div>
              <div className="member physician">
                <span className="label">Physician:</span>
                <span className="name">Assigned</span>
                <span className="status available">●</span>
              </div>
            </div>
            <button className="btn btn-secondary btn-sm">Change Team</button>
          </div>

          <div className="card stats-card">
            <h3>📊 Session Stats</h3>
            <div className="stat">
              <span className="label">Patients Triaged:</span>
              <span className="value">0</span>
            </div>
            <div className="stat">
              <span className="label">Avg Triage Time:</span>
              <span className="value">--</span>
            </div>
            <div className="stat">
              <span className="label">Shift Duration:</span>
              <span className="value">8h</span>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <section className="main-content">
          {/* Tab Navigation */}
          <div className="tab-navigation">
            <button 
              className={`tab ${activeTab === 'queue' ? 'active' : ''}`}
              onClick={() => setActiveTab('queue')}
            >
              👥 Waiting Queue
            </button>
            <button 
              className={`tab ${activeTab === 'ongoing' ? 'active' : ''}`}
              onClick={() => setActiveTab('ongoing')}
            >
              ⏳ Ongoing Assessments
            </button>
            <button 
              className={`tab ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              📋 Today's History
            </button>
          </div>

          {/* Patient Queue Tab */}
          {activeTab === 'queue' && (
            <div className="tab-content queue-content">
              <div className="no-patients">
                <div className="empty-state">
                  <p className="emoji">😌</p>
                  <h3>No Patients Waiting</h3>
                  <p>All patients in the waiting area have been triaged</p>
                </div>
              </div>
              {/* When patients arrive, they'll appear here */}
            </div>
          )}

          {/* Ongoing Assessments Tab */}
          {activeTab === 'ongoing' && (
            <div className="tab-content ongoing-content">
              <div className="no-patients">
                <div className="empty-state">
                  <p className="emoji">·</p>
                  <h3>No Ongoing Assessments</h3>
                  <p>Ready to triage the next patient</p>
                </div>
              </div>
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="tab-content history-content">
              <div className="no-patients">
                <div className="empty-state">
                  <p className="emoji">📖</p>
                  <h3>No Triage History</h3>
                  <p>Triage assessments will appear here</p>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Footer: Quick Actions */}
      <footer className="dashboard-footer">
        <button className="btn btn-primary">
          🔔 Set Break Request
        </button>
        <button className="btn btn-secondary">
          💬 Send Message
        </button>
        <button className="btn btn-outline">
          ⚙️ Settings
        </button>
      </footer>
    </div>
  );
};

export default TriageNurseDashboard;
