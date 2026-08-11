import React from 'react';
import './TriageTabContainer.css';

/**
 * TriageTabContainer - Tab header for switching between Manual and AI-Assisted triage
 * Only renders tabs; content is rendered by MiddleColumn based on activeTab prop
 */
function TriageTabContainer({ 
  activeTab = 'manual',
  onTabChange = () => {}
}) {
  return (
    <div style={containerStyle}>
      {/* Tab Navigation */}
      <div style={tabNavStyle}>
        <button
          style={{
            ...tabButtonStyle,
            ...(activeTab === 'manual' ? activeTabButtonStyle : inactiveTabButtonStyle)
          }}
          onClick={() => onTabChange('manual')}
        >
          Manual Triage
        </button>
        <button
          style={{
            ...tabButtonStyle,
            ...(activeTab === 'ai' ? activeTabButtonStyle : inactiveTabButtonStyle)
          }}
          onClick={() => onTabChange('ai')}
        >
          Auto-Triage
        </button>
      </div>
    </div>
  );
}

const containerStyle = {
  backgroundColor: 'white',
  borderRadius: '8px',
  overflow: 'hidden',
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
};

const tabNavStyle = {
  display: 'flex',
  borderBottom: '2px solid #e2e8f0',
  backgroundColor: '#f8fafc'
};

const tabButtonStyle = {
  flex: 1,
  padding: '16px 20px',
  border: 'none',
  backgroundColor: 'transparent',
  cursor: 'pointer',
  fontSize: '15px',
  fontWeight: '500',
  transition: 'all 0.2s ease',
  borderBottom: '3px solid transparent',
  color: '#64748b'
};

const activeTabButtonStyle = {
  color: '#2563eb',
  backgroundColor: 'white',
  borderBottomColor: '#2563eb',
  fontWeight: '600'
};

const inactiveTabButtonStyle = {
  color: '#94a3b8',
  cursor: 'pointer',
  transition: 'color 0.2s'
};

export default TriageTabContainer;
