// src/Dashboard/ClockDashboard.jsx
import React, { useState, useEffect } from 'react';
import { FaClock, FaHistory, FaCheckCircle, FaCog, FaChartBar, FaSignOutAlt, FaCalendarAlt } from 'react-icons/fa';

// Import all dashboard pages
import ClockInOutBoard from './Pages/ClockInOutBoard';
import MyClockHistoryPage from './Pages/MyClockHistoryPage';
import ManagerApprovalPage from './Pages/ManagerApprovalPage';
import AdminSettingsPage from './Pages/AdminSettingsPage';
import ScheduleBoard from './Pages/ScheduleBoard';
import StatisticsPage from './Pages/StatisticsPage';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function ClockDashboard({ user, token, onLogout }) {
  const [activePage, setActivePage] = useState('clock-in-out');
  const [userStats, setUserStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetchUserStats();
    }
  }, [token]);

  const fetchUserStats = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/staff/attendance-history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setUserStats(data.stats);
      }
    } catch (err) {
      console.error('Error fetching stats:', err);
    } finally {
      setLoading(false);
    }
  };

  // Navigation items based on user role
  const getNavItems = () => {
    const items = [
      {
        id: 'clock-in-out',
        label: 'Clock In/Out',
        icon: FaClock,
        show: true
      },
      {
        id: 'schedule',
        label: 'My Schedule',
        icon: FaCalendarAlt,
        show: true
      },
      {
        id: 'history',
        label: 'Clock History',
        icon: FaHistory,
        show: true
      },
      {
        id: 'statistics',
        label: 'Statistics',
        icon: FaChartBar,
        show: true
      }
    ];

    // Add manager/admin items
    if (user?.role === 'MANAGER' || user?.role === 'ADMIN') {
      items.push({
        id: 'approvals',
        label: 'Approvals',
        icon: FaCheckCircle,
        show: true
      });
    }

    if (user?.role === 'ADMIN') {
      items.push({
        id: 'settings',
        label: 'Settings',
        icon: FaCog,
        show: true
      });
    }

    return items.filter(item => item.show);
  };

  const handleLogout = () => {
    if (window.confirm('Are you sure you want to logout?')) {
      onLogout();
    }
  };

  const renderPage = () => {
    switch (activePage) {
      case 'clock-in-out':
        return <ClockInOutBoard token={token} user={user} />;
      case 'schedule':
        return <ScheduleBoard token={token} />;
      case 'history':
        return <MyClockHistoryPage token={token} />;
      case 'statistics':
        return <StatisticsPage token={token} />;
      case 'approvals':
        return <ManagerApprovalPage token={token} userRole={user?.role} />;
      case 'settings':
        return <AdminSettingsPage token={token} />;
      default:
        return <ClockInOutBoard token={token} user={user} />;
    }
  };

  return (
    <div style={styles.container}>
      {/* Sidebar Navigation */}
      <div style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <h2 style={styles.appTitle}>🏥 Nurse Helpdesk</h2>
          <p style={styles.version}>v1.0 - Clock Management</p>
        </div>

        {/* User Profile */}
        <div style={styles.userProfile}>
          <div style={styles.userAvatar}>
            {user?.name?.charAt(0).toUpperCase()}
          </div>
          <div style={styles.userInfo}>
            <p style={styles.userName}>{user?.name || 'User'}</p>
            <p style={styles.userRole}>
              {user?.role === 'NURSE' ? 'Staff' : user?.role || 'USER'}
            </p>
          </div>
        </div>

        {/* Navigation */}
        <nav style={styles.nav}>
          {getNavItems().map(item => (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              style={{
                ...styles.navItem,
                ...(activePage === item.id ? styles.navItemActive : {})
              }}
              title={item.label}
            >
              <item.icon style={{ marginRight: '8px', fontSize: '16px' }} />
              <span style={styles.navLabel}>{item.label}</span>
            </button>
          ))}
        </nav>

        {/* User Stats Widget */}
        {!loading && userStats && (
          <div style={styles.statsWidget}>
            <h4 style={styles.statsTitle}>This Week</h4>
            <div style={styles.statRow}>
              <span style={styles.statLabel}>Hours:</span>
              <span style={styles.statValue}>{userStats.weeklyHours?.toFixed(1) || '0'}h</span>
            </div>
            <div style={styles.statRow}>
              <span style={styles.statLabel}>Late:</span>
              <span style={styles.statValue}>{userStats.weeklyLateArrivals || 0}</span>
            </div>
            <div style={styles.statRow}>
              <span style={styles.statLabel}>Pending:</span>
              <span style={styles.statValue}>{userStats.pendingApprovals || 0}</span>
            </div>
          </div>
        )}

        {/* Logout Button */}
        <button onClick={handleLogout} style={styles.logoutBtn}>
          <FaSignOutAlt style={{ marginRight: '8px' }} />
          Logout
        </button>
      </div>

      {/* Main Content */}
      <div style={styles.mainContent}>
        <div style={styles.pageContent}>
          {renderPage()}
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    height: '100vh',
    backgroundColor: '#f3f4f6',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  sidebar: {
    width: '280px',
    backgroundColor: '#1f2937',
    color: 'white',
    padding: '0',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '2px 0 4px rgba(0,0,0,0.1)',
    overflowY: 'auto'
  },
  sidebarHeader: {
    padding: '20px 16px',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
    textAlign: 'center'
  },
  appTitle: {
    margin: '0 0 4px 0',
    fontSize: '16px',
    fontWeight: '700',
    letterSpacing: '0.5px'
  },
  version: {
    margin: '0',
    fontSize: '11px',
    color: '#9ca3af'
  },
  userProfile: {
    padding: '16px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    borderBottom: '1px solid rgba(255,255,255,0.1)'
  },
  userAvatar: {
    width: '44px',
    height: '44px',
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '700',
    fontSize: '18px',
    flexShrink: 0
  },
  userInfo: {
    flex: 1,
    minWidth: 0
  },
  userName: {
    margin: '0',
    fontWeight: '600',
    fontSize: '14px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis'
  },
  userRole: {
    margin: '2px 0 0 0',
    fontSize: '12px',
    color: '#d1d5db'
  },
  nav: {
    flex: 1,
    padding: '16px 0',
    display: 'flex',
    flexDirection: 'column'
  },
  navItem: {
    padding: '12px 16px',
    backgroundColor: 'transparent',
    border: 'none',
    color: '#d1d5db',
    textAlign: 'left',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    fontWeight: '500',
    fontSize: '14px',
    transition: 'all 0.2s ease',
    borderLeft: '3px solid transparent'
  },
  navItemActive: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    color: '#60a5fa',
    borderLeftColor: '#60a5fa'
  },
  navLabel: {
    flex: 1
  },
  statsWidget: {
    margin: '16px',
    padding: '16px',
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: '8px',
    borderTop: '1px solid rgba(255,255,255,0.1)'
  },
  statsTitle: {
    margin: '0 0 12px 0',
    fontSize: '12px',
    fontWeight: '700',
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  statRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '13px',
    marginBottom: '8px',
    padding: '4px 0'
  },
  statLabel: {
    color: '#9ca3af'
  },
  statValue: {
    fontWeight: '700',
    color: '#f3f4f6'
  },
  logoutBtn: {
    margin: '16px',
    padding: '10px 16px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.2s ease'
  },
  mainContent: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden'
  },
  pageContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '0'
  }
};
