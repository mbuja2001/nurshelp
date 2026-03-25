import React, { useState } from "react";

function Footer() {
  return (
    <footer style={footerStyle}>
      <p style={textStyle}>2026 Living Compute Labs. All Intellectual Property Rights Reserved.</p>
      <p style={textStyle}>Nurse Help Desk v1.0</p>
    </footer>
  );
}

const footerStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '0 40px',
  height: '50px',
  backgroundColor: '#ffffff',
  borderTop: '1px solid #ddd',
  width: '100vw',
  boxSizing: 'border-box',
  marginTop: 'auto' 
};

const textStyle = {
  fontSize: '14px',
  color: '#666',
  margin: 0
};

export default Footer;