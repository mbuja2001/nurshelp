import React, { useEffect } from 'react';
import './OffDutyDashboard.css';

/**
 * OffDutyDashboard Component
 * 
 * NO_SHIFT CONTEXT
 * Displayed when staff member has no shift assigned for today
 * Shows upcoming shifts and allows staff to view their schedule
 */

const OffDutyDashboard = ({ config }) => {
  useEffect(() => {
    console.log('[OFF DUTY] Dashboard loaded');
    console.log('Config:', config);
  }, [config]);

  return (
    <div className="off-duty-dashboard">
      {/* Header */}
      <header className="dashboard-header off-duty-header">
        <div className="header-left">
          <h1>📅 You're Off Duty Today</h1>
          <p className="subtitle">{config.staffName} • {config.staffDepartment}</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-content off-duty-content">
        <div className="off-duty-container">
          {/* Main Message */}
          <div className="off-duty-message">
            <div className="icon">🌴</div>
            <h2>Enjoy Your Day Off!</h2>
            <p>You don't have a shift assigned for today.</p>
          </div>

          {/* Quick Links */}
          <div className="quick-links">
            <h3>Quick Actions</h3>
            <div className="link-grid">
              <a href="#schedule" className="link-card">
                <span className="icon">📅</span>
                <span className="label">View Schedule</span>
              </a>
              <a href="#upcoming" className="link-card">
                <span className="icon">📌</span>
                <span className="label">Upcoming Shifts</span>
              </a>
              <a href="#requests" className="link-card">
                <span className="icon">✋</span>
                <span className="label">Shift Requests</span>
              </a>
              <a href="#profile" className="link-card">
                <span className="icon">👤</span>
                <span className="label">Profile</span>
              </a>
            </div>
          </div>

          {/* Information Cards */}
          <div className="info-section">
            <div className="info-card">
              <h4>📋 Today's Date</h4>
              <p className="date">{new Date().toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}</p>
            </div>

            <div className="info-card">
              <h4>👔 Role</h4>
              <p className="role">{config.staffGeneralRole}</p>
            </div>

            <div className="info-card">
              <h4>🏥 Department</h4>
              <p className="department">{config.staffDepartment}</p>
            </div>
          </div>

          {/* Upcoming Shifts */}
          <div className="upcoming-shifts">
            <h3>🔮 Upcoming Shifts</h3>
            <div className="shifts-list">
              <p className="no-shifts">No shifts scheduled yet</p>
            </div>
          </div>

          {/* Contact Section */}
          <div className="contact-section">
            <h3>Need anything?</h3>
            <p>If you need to request a shift swap or have questions about scheduling:</p>
            <button className="btn btn-primary">
              💬 Contact Manager
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="dashboard-footer off-duty-footer">
        <button className="btn btn-outline">
          ⚙️ Settings
        </button>
        <button className="btn btn-outline">
          🚪 Logout
        </button>
      </footer>
    </div>
  );
};

export default OffDutyDashboard;
