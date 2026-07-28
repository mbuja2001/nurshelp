import React, { useState } from 'react';
import './SidebarNavigation.css';

/**
 * SidebarNavigation Component
 * 
 * Role-specific sidebar menu system for hospital HMS
 * Displays relevant modules and pages based on user role and permissions
 * 
 * Props:
 *  - role: User role ('triage-nurse', 'clinical-nurse', 'physician', 'manager', 'admin')
 *  - currentPage: Current active page
 *  - onNavigate: Callback function when menu item clicked
 *  - collapsed: Boolean to show collapsed sidebar
 * 
 * Roles & Menu Structure:
 * 
 * Triage Nurse:
 *  - Dashboard (active patients)
 *  - Patient Queue (waiting room)
 *  - Vitals Monitoring
 *  - Quick Assessment
 * 
 * Clinical Nurse:
 *  - Dashboard (assigned patients)
 *  - Patient Care (detailed)
 *  - Medication Management
 *  - Wound Care & Procedures
 *  - Patient Education
 * 
 * Physician:
 *  - Dashboard (active cases)
 *  - Case Management
 *  - Departments/Wards
 *  - Lab Results
 *  - Orders Management
 * 
 * Manager:
 *  - Dashboard (KPIs)
 *  - Staff Management
 *  - Scheduling
 *  - Reports & Analytics
 *  - Facility Management
 * 
 * Administrator:
 *  - System Dashboard
 *  - Staff Management
 *  - Facility Management
 *  - Billing & Claims
 *  - System Configuration
 *  - Audit Logs
 */

const SidebarNavigation = ({ role = 'triage-nurse', currentPage = 'dashboard', onNavigate, collapsed = false }) => {
  const [expandedModules, setExpandedModules] = useState({});

  // Role-specific menu configurations
  const menuStructure = {
    'triage-nurse': [
      {
        module: 'Clinical',
        icon: '🏥',
        items: [
          { id: 'dashboard', label: 'Dashboard', icon: '📊' },
          { id: 'patient-queue', label: 'Patient Queue', icon: '⏳' },
          { id: 'vitals-monitoring', label: 'Vitals Monitoring', icon: '❤️' },
          { id: 'quick-assessment', label: 'Quick Assessment', icon: '⚡' },
        ],
      },
      {
        module: 'Documentation',
        icon: '📋',
        items: [
          { id: 'chief-complaint', label: 'Chief Complaint Entry', icon: '📝' },
          { id: 'vital-history', label: 'Vital Signs History', icon: '📈' },
          { id: 'notes', label: 'Triage Notes', icon: '🗂️' },
        ],
      },
    ],
    'clinical-nurse': [
      {
        module: 'Patient Care',
        icon: '👨‍⚕️',
        items: [
          { id: 'dashboard', label: 'My Patients', icon: '👥' },
          { id: 'patient-details', label: 'Patient Details', icon: '👤' },
          { id: 'care-plan', label: 'Care Plan', icon: '📋' },
          { id: 'vital-monitoring', label: 'Vitals', icon: '❤️' },
        ],
      },
      {
        module: 'Pharmacy & Labs',
        icon: '💊',
        items: [
          { id: 'pharmacy', label: 'Pharmacy', icon: '💊' },
          { id: 'radiology', label: 'Radiology', icon: '🩻' },
          { id: 'pathology', label: 'Pathology', icon: '🧪' },
        ],
      },
      {
        module: 'Treatment',
        icon: '💊',
        items: [
          { id: 'medications', label: 'Medication Management', icon: '💉' },
          { id: 'procedures', label: 'Procedures & Care', icon: '🩺' },
          { id: 'wound-care', label: 'Wound Care', icon: '🩹' },
          { id: 'patient-education', label: 'Patient Education', icon: '📚' },
        ],
      },
      {
        module: 'Documentation',
        icon: '📊',
        items: [
          { id: 'nursing-notes', label: 'Nursing Notes', icon: '📝' },
          { id: 'flow-sheets', label: 'Flow Sheets', icon: '📄' },
          { id: 'discharge-prep', label: 'Discharge Planning', icon: '🚪' },
        ],
      },
    ],
    physician: [
      {
        module: 'Cases',
        icon: '🔍',
        items: [
          { id: 'dashboard', label: 'Active Cases', icon: '📊' },
          { id: 'case-management', label: 'Case Details', icon: '📋' },
          { id: 'patient-history', label: 'Patient History', icon: '📑' },
          { id: 'escalations', label: 'Escalations', icon: '🚨' },
        ],
      },
      {
        module: 'Clinical Diagnostics',
        icon: '🛠️',
        items: [
          { id: 'lab-results', label: 'Lab Results', icon: '🧪' },
          { id: 'radiology', label: 'Radiology', icon: '🩻' },
          { id: 'pathology', label: 'Pathology', icon: '🧬' },
          { id: 'orders', label: 'Orders Management', icon: '📋' },
          { id: 'diagnostics', label: 'Diagnostics', icon: '🔬' },
          { id: 'pharmacy', label: 'Pharmacy', icon: '💊' },
        ],
      },
      {
        module: 'Department',
        icon: '🏢',
        items: [
          { id: 'wards', label: 'Wards & Beds', icon: '🛏️' },
          { id: 'teams', label: 'Care Teams', icon: '👥' },
          { id: 'handoff', label: 'Patient Handoff', icon: '🤝' },
        ],
      },
    ],
    manager: [
      {
        module: 'Operations',
        icon: '⚙️',
        items: [
          { id: 'dashboard', label: 'KPI Dashboard', icon: '📊' },
          { id: 'performance-metrics', label: 'Performance Metrics', icon: '📈' },
          { id: 'wait-times', label: 'Wait Time Analysis', icon: '⏱️' },
        ],
      },
      {
        module: 'Staff Management',
        icon: '👔',
        items: [
          { id: 'staff-list', label: 'Staff Directory', icon: '👥' },
          { id: 'scheduling', label: 'Shift Scheduling', icon: '📅' },
          { id: 'payroll', label: 'Payroll & HR', icon: '💰' },
          { id: 'training', label: 'Training & Development', icon: '🎓' },
        ],
      },
      {
        module: 'Facility',
        icon: '🏭',
        items: [
          { id: 'beds', label: 'Bed Management', icon: '🛏️' },
          { id: 'equipment', label: 'Equipment Tracking', icon: '🔧' },
          { id: 'maintenance', label: 'Maintenance Requests', icon: '🔨' },
          { id: 'supplies', label: 'Supply Chain', icon: '📦' },
        ],
      },
      {
        module: 'Analytics',
        icon: '📊',
        items: [
          { id: 'reports', label: 'Reports', icon: '📋' },
          { id: 'trends', label: 'Trend Analysis', icon: '📈' },
          { id: 'quality', label: 'Quality Metrics', icon: '⭐' },
        ],
      },
    ],
    admin: [
      {
        module: 'System',
        icon: '🖥️',
        items: [
          { id: 'dashboard', label: 'System Dashboard', icon: '📊' },
          { id: 'user-management', label: 'User Management', icon: '👤' },
          { id: 'roles-permissions', label: 'Roles & Permissions', icon: '🔐' },
          { id: 'audit-logs', label: 'Audit Logs', icon: '📋' },
        ],
      },
      {
        module: 'Configuration',
        icon: '⚙️',
        items: [
          { id: 'system-settings', label: 'System Settings', icon: '🔧' },
          { id: 'data-management', label: 'Data Management', icon: '💾' },
          { id: 'integrations', label: 'Integrations', icon: '🔗' },
          { id: 'backups', label: 'Backups & Recovery', icon: '💿' },
        ],
      },
      {
        module: 'Financial',
        icon: '💼',
        items: [
          { id: 'billing', label: 'Billing & Claims', icon: '💰' },
          { id: 'invoices', label: 'Invoices', icon: '🧾' },
          { id: 'insurance', label: 'Insurance Management', icon: '📄' },
          { id: 'payments', label: 'Payments', icon: '💳' },
        ],
      },
      {
        module: 'Facility',
        icon: '🏢',
        items: [
          { id: 'departments', label: 'Departments', icon: '🏥' },
          { id: 'beds', label: 'Bed Configuration', icon: '🛏️' },
          { id: 'equipment', label: 'Equipment Management', icon: '⚙️' },
          { id: 'supply-chain', label: 'Supply Chain Management', icon: '📦' },
        ],
      },
    ],
  };

  const toggleModule = (moduleName) => {
    setExpandedModules((prev) => ({
      ...prev,
      [moduleName]: !prev[moduleName],
    }));
  };

  const handleNavigate = (pageId) => {
    if (onNavigate) {
      onNavigate(pageId);
    }
  };

  const menu = menuStructure[role] || menuStructure['triage-nurse'];

  return (
    <aside className={`sidebar-navigation ${collapsed ? 'collapsed' : 'expanded'}`}>
      {/* Role Badge */}
      {!collapsed && (
        <div className="sidebar-role-badge">
          <span className="role-icon">👤</span>
          <span className="role-text">{role.replace('-', ' ').toUpperCase()}</span>
        </div>
      )}

      {/* Navigation Menu */}
      <nav className="sidebar-menu">
        {menu.map((module) => (
          <div key={module.module} className="menu-module">
            {/* Module Header (Expandable) */}
            <button
              className="module-header"
              onClick={() => toggleModule(module.module)}
              aria-expanded={expandedModules[module.module] || true}
            >
              <span className="module-icon">{module.icon}</span>
              {!collapsed && (
                <>
                  <span className="module-title">{module.module}</span>
                  <span
                    className={`expand-icon ${expandedModules[module.module] ? 'rotated' : ''}`}
                  >
                    ▼
                  </span>
                </>
              )}
            </button>

            {/* Module Items */}
            {(expandedModules[module.module] || !expandedModules.hasOwnProperty(module.module)) && (
              <div className="module-items">
                {module.items.map((item) => (
                  <button
                    key={item.id}
                    className={`menu-item ${currentPage === item.id ? 'active' : ''}`}
                    onClick={() => handleNavigate(item.id)}
                    title={!collapsed ? '' : item.label}
                  >
                    <span className="item-icon">{item.icon}</span>
                    {!collapsed && <span className="item-label">{item.label}</span>}
                    {currentPage === item.id && <span className="active-indicator">●</span>}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>

      {/* Sidebar Footer */}
      {!collapsed && (
        <div className="sidebar-footer">
          <button className="footer-item help" title="Help & Support">
            <span className="item-icon">❓</span>
            <span className="item-label">Help</span>
          </button>
          <button className="footer-item settings" title="Settings">
            <span className="item-icon">⚙️</span>
            <span className="item-label">Settings</span>
          </button>
          <button className="footer-item logout" title="Logout">
            <span className="item-icon">🚪</span>
            <span className="item-label">Logout</span>
          </button>
        </div>
      )}

      {/* Collapsed Footer (Icons Only) */}
      {collapsed && (
        <div className="sidebar-footer collapsed">
          <button title="Help">❓</button>
          <button title="Settings">⚙️</button>
          <button title="Logout">🚪</button>
        </div>
      )}
    </aside>
  );
};

export default SidebarNavigation;
