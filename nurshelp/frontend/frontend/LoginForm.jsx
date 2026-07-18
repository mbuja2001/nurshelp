import React, { useState } from "react";
import { styles } from "./styles";

const LoginForm = ({ onLogin, onCancel }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("LoginForm submit:", { email, password }); // Debug payload
    setLoading(true);
    try {
      await onLogin({ email, password });
    } catch (err) {
      alert(err.message || "Login failed");
    } finally {
      setLoading(false);
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
          🔐 Authorization
        </h2>
        <form onSubmit={handleSubmit}>
          <div style={styles.formGroup}>
            <label style={styles.formLabel}>
              ✉️ Email Address <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="email"
              placeholder="your.email@hospital.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onFocus={() => setFocusedField('email')}
              onBlur={() => setFocusedField(null)}
              style={getInputStyle('email')}
              required
            />
          </div>

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

          <div style={{ display: "flex", gap: "10px", marginTop: "24px" }}>
            <button type="submit" style={styles.btnWhite} disabled={loading}>
              {loading ? "🔄 Logging in..." : "🔓 Enter"}
            </button>
            <button type="button" onClick={onCancel} style={styles.btnOutline}>
              ✕ Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default LoginForm;