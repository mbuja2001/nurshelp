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

  const navItems = getNavItems();

  return (
    <div style={styles.container}>
      <header style={styles.topBar}>
        <div style={styles.brandBlock}>
          <div style={styles.brandMark}>🏥</div>
          <div>
            <h1 style={styles.appTitle}>Nurse Helpdesk</h1>
            <p style={styles.version}>Clock management</p>
          </div>
        </div>
        <div style={styles.userBlock}>
          <span style={styles.userAvatar}>{user?.name?.charAt(0).toUpperCase() || 'U'}</span>
          <span style={styles.userName}>{user?.name || 'User'}</span>
          <span style={styles.userRole}>{user?.role === 'NURSE' ? 'Staff' : user?.role || 'USER'}</span>
          <button onClick={handleLogout} style={styles.logoutBtn} title="Log out">
            <FaSignOutAlt />
            <span>Logout</span>
          </button>
        </div>
      </header>

      <nav style={styles.topTabs} aria-label="Clock management sections">
        {navItems.map(item => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              style={{ ...styles.topTab, ...(activePage === item.id ? styles.topTabActive : {}) }}
              title={item.label}
            >
              <Icon />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {!loading && userStats && (
        <div style={styles.summaryStrip}>
          <span><strong>{userStats.weeklyHours?.toFixed(1) || '0'}h</strong> this week</span>
          <span><strong>{userStats.weeklyLateArrivals || 0}</strong> late arrivals</span>
          <span><strong>{userStats.pendingApprovals || 0}</strong> pending approvals</span>
        </div>
      )}

      <main style={styles.mainContent}>
        <div style={styles.pageContent}>{renderPage()}</div>
      </main>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f3f4f6',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: '#0f172a'
  },
  topBar: {
    backgroundColor: '#ffffff',
    borderBottom: '1px solid #e2e8f0',
    padding: '16px 24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 20,
    flexWrap: 'wrap'
  },
  brandBlock: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    minWidth: 0
  },
  appTitle: {
    margin: 0,
    fontSize: '20px',
    fontWeight: '700',
    letterSpacing: '0.2px',
    color: '#0f172a'
  },
  version: {
    margin: '3px 0 0',
    fontSize: '11px',
    color: '#64748b'
  },
  brandMark: {
    width: 38,
    height: 38,
    display: 'grid',
    placeItems: 'center',
    borderRadius: 10,
    background: '#ecfdf5',
    fontSize: 19
  },
  userBlock: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'wrap',
    justifyContent: 'flex-end'
  },
  userAvatar: {
    width: 34,
    height: 34,
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '700',
    fontSize: '15px',
    flexShrink: 0
  },
  userInfo: {
    flex: 1,
    minWidth: 0
  },
  userName: {
    margin: 0,
    fontWeight: '600',
    fontSize: '13px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis'
  },
  userRole: {
    margin: '2px 0 0 0',
    fontSize: '12px',
    color: '#64748b'
  },
  topTabs: {
    backgroundColor: '#ffffff',
    borderBottom: '2px solid #e2e8f0',
    padding: '0 24px',
    display: 'flex',
    gap: 24,
    overflowX: 'auto'
  },
  topTab: {
    padding: '15px 0 13px',
    backgroundColor: 'transparent',
    border: 'none',
    borderBottom: '3px solid transparent',
    color: '#64748b',
    textAlign: 'center',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    fontWeight: '500',
    fontSize: '13px',
    whiteSpace: 'nowrap',
    transition: 'all 0.2s ease'
  },
  topTabActive: {
    color: '#10b981',
    borderBottomColor: '#10b981',
    fontWeight: '700'
  },
  summaryStrip: {
    display: 'flex',
    alignItems: 'center',
    gap: 24,
    flexWrap: 'wrap',
    padding: '12px 32px',
    background: '#f8fafc',
    borderBottom: '1px solid #e2e8f0',
    color: '#64748b',
    fontSize: 12
  },
  logoutBtn: {
    padding: '8px 12px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    justifyContent: 'center',
    transition: 'background-color 0.2s ease'
  },
  mainContent: {
    width: '100%',
    minWidth: 0
  },
  pageContent: {
    width: '100%',
    maxWidth: 1440,
    margin: '0 auto',
    padding: '24px 32px 48px',
    minWidth: 0
  }
};
