import React from 'react';
import './TriageTabHeader.css';

/**
 * TriageTabHeader - Header with tabs for Manual vs AI-Assisted Triage
 * Appears under "Patient Dashboard" title
 */
function TriageTabHeader({ activeTab, onTabChange, onBackClick, onLogout }) {
  return (
    <header style={headerStyle}>
      <button style={backButtonStyle} onClick={onLogout}>
        Log Out
      </button>

      <h2 style={titleStyle}>Patient Dashboard</h2>

      {/* Triage Tab Selection */}
      <div style={tabContainerStyle}>
        <button
          style={{
            ...tabStyle,
            ...(activeTab === 'manual' ? activeTabStyle : inactiveTabStyle)
          }}
          onClick={() => onTabChange('manual')}
        >
          Manual Triage
        </button>
        <button
          style={{
            ...tabStyle,
            ...(activeTab === 'ai' ? activeTabStyle : inactiveTabStyle)
          }}
          onClick={() => onTabChange('ai')}
        >
          Auto Triage
        </button>
      </div>

      <button style={intakeButtonStyle} onClick={onBackClick}>
        Intake Status
      </button>
    </header>
  );
}

const headerStyle = {
  display: 'flex',
  flexDirection: 'column',
  width: '100%',
  backgroundColor: '#ffffff',
  borderBottom: '1px solid #ddd',
  position: 'relative'
};

const topRowStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  width: '100%',
  height: '70px',
  padding: '0 40px',
  boxSizing: 'border-box',
};

const titleStyle = {
  margin: 0,
  position: 'absolute',
  top: '12px',
  left: '50%',
  transform: 'translateX(-50%)',
  fontSize: '20px',
  fontWeight: 'bold',
  color: '#333'
};

const backButtonStyle = {
  padding: '8px 16px',
  backgroundColor: '#c81616ff',
  color: 'white',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
  fontWeight: '600'
};

const intakeButtonStyle = {
  padding: '8px 16px',
  backgroundColor: '#646cff',
  color: 'white',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
  fontWeight: '600'
};

const tabContainerStyle = {
  display: 'flex',
  gap: '0',
  padding: '0 40px',
  marginTop: '0',
  borderTop: '1px solid #eee',
  backgroundColor: '#f9f9f9'
};

const tabStyle = {
  padding: '12px 24px',
  border: 'none',
  backgroundColor: 'transparent',
  cursor: 'pointer',
  fontSize: '14px',
  fontWeight: '500',
  transition: 'all 0.3s ease',
  borderBottom: '3px solid transparent',
  color: '#666'
};

const activeTabStyle = {
  color: '#2563eb',
  borderBottomColor: '#2563eb',
  backgroundColor: 'white'
};

const inactiveTabStyle = {
  color: '#999',
  borderBottomColor: 'transparent',
  ('&:hover': {
    color: '#666'
  })
};

export default TriageTabHeader;
