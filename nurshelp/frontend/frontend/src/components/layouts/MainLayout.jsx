import React, { useState } from 'react';
import MainNavigation from './MainNavigation';
import SidebarNavigation from './SidebarNavigation';
import './MainLayout.css';

/**
 * MainLayout Component
 * 
 * Professional hospital dashboard layout combining:
 * - Top Navigation (global search, notifications, user profile)
 * - Left Sidebar (role-specific menu)
 * - Main Content Area (page content)
 * 
 * Props:
 *  - role: User role (triage-nurse, clinical-nurse, physician, manager, admin)
 *  - currentPage: Current active page ID
 *  - onNavigate: Callback for page navigation
 *  - userName: User full name for top navigation
 *  - userRole: User role display name
 *  - userAvatar: User avatar (emoji)
 *  - children: Page content to render
 * 
 * Usage:
 * <MainLayout role="physician" currentPage="dashboard">
 *   <PhysicianDashboard />
 * </MainLayout>
 */

const MainLayout = ({
  role = 'triage-nurse',
  currentPage = 'dashboard',
  onNavigate,
  userName = 'Dr. Sarah Johnson',
  userRole = 'Physician',
  userAvatar = '👨‍⚕️',
  children,
}) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notificationCount, setNotificationCount] = useState(3);

  const handleNavigate = (pageId) => {
    if (onNavigate) {
      onNavigate(pageId);
    }
  };

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className="main-layout">
      {/* Top Navigation */}
      <MainNavigation
        userName={userName}
        userRole={userRole}
        userAvatar={userAvatar}
        notificationCount={notificationCount}
        onNotificationClick={() => console.log('Notifications clicked')}
        onProfileClick={() => console.log('Profile clicked')}
      />

      {/* Main Content Container */}
      <div className="layout-container">
        {/* Sidebar Navigation */}
        <SidebarNavigation
          role={role}
          currentPage={currentPage}
          onNavigate={handleNavigate}
          collapsed={sidebarCollapsed}
        />

        {/* Main Content Area */}
        <main className={`main-content ${sidebarCollapsed ? 'sidebar-collapsed' : 'sidebar-expanded'}`}>
          {/* Header Bar with Breadcrumbs and Actions */}
          <div className="content-header">
            <div className="breadcrumb-section">
              <button className="sidebar-toggle" onClick={toggleSidebar} title="Toggle Sidebar">
                {sidebarCollapsed ? '☰' : '✕'}
              </button>
              <nav className="breadcrumbs" aria-label="Breadcrumb">
                <ol className="breadcrumb-list">
                  <li>
                    <a href="#/" className="breadcrumb-link">
                      Home
                    </a>
                    <span className="breadcrumb-separator">/</span>
                  </li>
                  <li className="breadcrumb-current">{currentPage.replace('-', ' ')}</li>
                </ol>
              </nav>
            </div>
            <div className="header-actions">
              {/* Quick Actions - Can be populated by child components */}
            </div>
          </div>

          {/* Page Content */}
          <div className="page-content">
            {children}
          </div>
        </main>
      </div>

      {/* Mobile Overlay */}
      {sidebarCollapsed && (
        <div className="sidebar-overlay" onClick={() => setSidebarCollapsed(true)} />
      )}
    </div>
  );
};

export default MainLayout;
