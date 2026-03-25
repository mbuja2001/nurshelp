import React, { useState } from "react";
import { styles } from "./styles";

const LoginForm = ({ onLogin, onCancel }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

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
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ ...styles.loginInput, marginTop: "10px" }}
            required
          />
          <div style={{ display: "flex", gap: "10px", marginTop: "20px" }}>
            <button type="submit" style={styles.btnWhite} disabled={loading}>
              {loading ? "Logging in..." : "Enter"}
            </button>
            <button type="button" onClick={onCancel} style={styles.btnOutline}>
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default LoginForm;