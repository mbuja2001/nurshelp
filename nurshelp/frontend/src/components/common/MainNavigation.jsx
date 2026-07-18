import React, { useState } from 'react';
import './MainNavigation.css';

/**
 * MainNavigation Component
 * Professional hospital-grade top navigation with role-based menu system
 * Features: Search, user profile, notifications, quick actions
 */
const MainNavigation = ({ userRole, userName, userDepartment, onLogout }) => {
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);

  const roleDisplayName = {
    TRIAGE_NURSE: 'Triage Nurse',
    CLINICAL_NURSE: 'Clinical Nurse',
    PHYSICIAN: 'Physician',
    MANAGER: 'Manager',
    ADMIN: 'Administrator'
  }[userRole] || userRole;

  return (
    <nav className="main-navigation">
      {/* Left: Logo + Hospital Name */}
      <div className="nav-left">
        <div className="hospital-logo">
          <span className="logo-icon">🏥</span>
          <span className="hospital-name">Nurse Helpdesk</span>
        </div>
      </div>

      {/* Center: Search Bar */}
      <div className="nav-center">
        <div className="search-container">
          <input
            type="text"
            placeholder="Search patient, staff, or resources..."
            className="search-input"
            aria-label="Global search"
          />
          <span className="search-icon">🔍</span>
        </div>
      </div>

      {/* Right: Actions + Profile */}
      <div className="nav-right">
        {/* Notifications */}
        <div className="nav-item notifications">
          <button
            className="icon-button"
            onClick={() => setShowNotifications(!showNotifications)}
            aria-label="Notifications"
            title="Alerts and notifications"
          >
            🔔
            <span className="notification-badge">3</span>
          </button>
          {showNotifications && (
            <div className="notification-dropdown">
              <div className="notification-item alert">
                <strong>Red Flag Alert:</strong> Patient in ICU-02 O2 saturation below 90%
              </div>
              <div className="notification-item info">
                <strong>Shift Change:</strong> Clinical team arrives in 15 minutes
              </div>
              <div className="notification-item success">
                <strong>Order Complete:</strong> Lab results ready for review
              </div>
              <a href="#" className="view-all">View all →</a>
            </div>
          )}
        </div>

        {/* Help */}
        <button
          className="icon-button"
          aria-label="Help"
          title="Get help"
        >
          ❓
        </button>

        {/* User Profile */}
        <div className="nav-item profile">
          <button
            className="profile-button"
            onClick={() => setShowProfileMenu(!showProfileMenu)}
            aria-label="User menu"
          >
            <span className="profile-avatar">👤</span>
            <div className="profile-info">
              <span className="profile-name">{userName || 'Staff'}</span>
              <span className="profile-role">{roleDisplayName}</span>
            </div>
            <span className="dropdown-arrow">▼</span>
          </button>

          {showProfileMenu && (
            <div className="profile-dropdown">
              <div className="profile-header">
                <span className="profile-avatar-large">👤</span>
                <div>
                  <strong>{userName || 'Staff Member'}</strong>
                  <p className="text-muted">{userDepartment || 'Department'}</p>
                </div>
              </div>
              <hr />
              <a href="#" className="dropdown-item">👤 My Profile</a>
              <a href="#" className="dropdown-item">⏰ Clock History</a>
              <a href="#" className="dropdown-item">⚙️ Settings</a>
              <a href="#" className="dropdown-item">📊 My Stats</a>
              <hr />
              <button
                className="dropdown-item logout"
                onClick={onLogout}
              >
                🚪 Logout
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default MainNavigation;
