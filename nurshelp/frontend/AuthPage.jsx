// Dashboard/AuthPage.jsx

import React, { useState } from 'react';
import { styles } from './styles';

const AuthPage = ({
  username,
  setUsername,
  password,
  setPassword,
  handleLogin,
  handleRegister,
  setView
}) => {

  const [mode, setMode] = useState('login'); // login | register
  const [surname, setSurname] = useState('');
  const [email, setEmail] = useState('');
  const [focusedField, setFocusedField] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (mode === 'login') {
      handleLogin(e);
    } else {
      handleRegister({ username, surname, email, password });
    }
  };

  const getInputStyle = (fieldName) => {
    const baseStyle = styles.loginInput;
    return focusedField === fieldName 
      ? { ...baseStyle, ...styles.loginInputFocus }
      : baseStyle;
  };

  return (
    <div style={styles.patientBg}>
      <div style={styles.loginCard}>
        <h2 style={{ marginBottom: '30px', fontSize: '24px', fontWeight: 'bold', color: '#e2e8f0' }}>
          {mode === 'login' ? '🔐 Nurse Login' : '📝 Register Nurse'}
        </h2>

        <form onSubmit={handleSubmit}>
          {mode === 'register' && (
            <>
              <div style={styles.formGroup}>
                <label style={styles.formLabel}>
                  👤 Full Name <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="text"
                  placeholder="John"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onFocus={() => setFocusedField('name')}
                  onBlur={() => setFocusedField(null)}
                  style={getInputStyle('name')}
                  required
                />
              </div>

              <div style={styles.formGroup}>
                <label style={styles.formLabel}>
                  👨‍💼 Surname <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="text"
                  placeholder="Doe"
                  value={surname}
                  onChange={(e) => setSurname(e.target.value)}
                  onFocus={() => setFocusedField('surname')}
                  onBlur={() => setFocusedField(null)}
                  style={getInputStyle('surname')}
                  required
                />
              </div>

              <div style={styles.formGroup}>
                <label style={styles.formLabel}>
                  ✉️ Email Address <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="email"
                  placeholder="john.doe@hospital.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setFocusedField('email')}
                  onBlur={() => setFocusedField(null)}
                  style={getInputStyle('email')}
                  required
                />
              </div>
            </>
          )}

          {mode === 'login' && (
            <div style={styles.formGroup}>
              <label style={styles.formLabel}>
                ✉️ Email or Name <span style={{ color: '#ef4444' }}>*</span>
              </label>
              <input
                type="text"
                placeholder="john@hospital.com"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                onFocus={() => setFocusedField('username')}
                onBlur={() => setFocusedField(null)}
                style={getInputStyle('username')}
                required
              />
            </div>
          )}

          <div style={styles.formGroup}>
            <label style={styles.formLabel}>
              🔒 Password <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onFocus={() => setFocusedField('password')}
              onBlur={() => setFocusedField(null)}
              style={getInputStyle('password')}
              required
            />
          </div>

          <div style={{ display: 'flex', gap: '10px', marginTop: '24px' }}>
            <button type="submit" style={styles.btnWhite}>
              {mode === 'login' ? '🔓 Login' : '✓ Register'}
            </button>

            <button
              type="button"
              onClick={() => setView('home')}
              style={styles.btnOutline}
            >
              ← Back
            </button>
          </div>
        </form>

        <div style={{ marginTop: '24px', borderTop: '1px solid #334155', paddingTop: '20px' }}>
          <button
            onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
            style={{ ...styles.btnOutline, width: '100%', fontSize: '14px' }}
          >
            {mode === 'login'
              ? '➕ Create an account'
              : '🔄 Already have an account? Login'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
