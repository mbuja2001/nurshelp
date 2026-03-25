import React from 'react';

function Header({ onBackClick, onLogout }) {
  return (
    <header style={headerStyle}>
      <button style={backButtonStyle} onClick={onLogout}>
        Log Out
      </button>

      <h2 style={titleStyle}>Patient Dashboard</h2>

      <button style={intakeButtonStyle} onClick={onBackClick}>
        Intake Status
      </button>
    </header>
  );
}

const headerStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  width: '100%',
  height: '70px',
  padding: '0 40px',
  boxSizing: 'border-box',
  backgroundColor: '#ffffff',
  borderBottom: '1px solid #ddd',
  position: 'relative'
};

const titleStyle = {
  margin: 0,
  position: 'absolute',
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
  cursor: 'pointer'
  
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

export default Header;
