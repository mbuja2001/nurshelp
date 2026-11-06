import React, { useState } from "react";
import { styles } from "./styles";

const LoginForm = ({ onLogin, onCancel, onCreateAccount }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await onLogin({ email, password });
    } catch (err) {
      alert(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.patientBg}>
      <div style={styles.loginCard}>
        <h2>Authorization</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={styles.loginInput}
            required
          />
          <div style={{ position: "relative", marginTop: "10px" }}>
            <input
              type={showPassword ? "text" : "password"}
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{ ...styles.loginInput, paddingRight: "110px" }}
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword((current) => !current)}
              style={{
                position: "absolute",
                right: "8px",
                top: "50%",
                transform: "translateY(-50%)",
                border: "1px solid #cbd5e1",
                borderRadius: 999,
                background: "#fff",
                padding: "5px 8px",
                fontSize: 11,
                fontWeight: 700,
                cursor: "pointer",
                color: "#0f172a"
              }}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          <div style={{ display: "flex", gap: "10px", marginTop: "20px", flexWrap: "wrap" }}>
            <button type="submit" style={styles.btnWhite} disabled={loading}>
              {loading ? "Logging in..." : "Enter"}
            </button>
            <button type="button" onClick={onCancel} style={styles.btnOutline}>
              Cancel
            </button>
          </div>
        </form>
        {onCreateAccount && <p style={{ margin: '18px 0 0', color: '#64748b', fontSize: 13 }}>New to Nurshelpdesk? <button type="button" onClick={onCreateAccount} style={{ padding: 0, border: 0, background: 'transparent', color: '#16745f', fontWeight: 800, cursor: 'pointer' }}>Create a sandbox account</button></p>}
      </div>
    </div>
  );
};

export default LoginForm;