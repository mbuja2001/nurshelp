import React, { useState } from "react";
import { styles } from "./styles";

const RegisterForm = ({ onRegister, onCancel }) => {
  const [form, setForm] = useState({ name: "", surname: "", email: "", password: "" });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.name || !form.surname || !form.email || !form.password) {
      alert("Please fill out all fields");
      return;
    }
    onRegister(form);
  };

  return (
    <div style={styles.patientBg}>
      <div style={styles.loginCard}>
        <h2 style={{ marginTop: 0 }}>Register Nurse</h2>
        <form onSubmit={handleSubmit}>
          <input name="name" placeholder="Name" value={form.name} onChange={handleChange} style={styles.loginInput} required />
          <input name="surname" placeholder="Surname" value={form.surname} onChange={handleChange} style={{ ...styles.loginInput, marginTop:"10px" }} required />
          <input name="email" type="email" placeholder="Email" value={form.email} onChange={handleChange} style={{ ...styles.loginInput, marginTop:"10px" }} required />
          <input name="password" type="password" placeholder="Password" value={form.password} onChange={handleChange} style={{ ...styles.loginInput, marginTop:"10px" }} required />
          <div style={{ display: "flex", gap: "10px", marginTop: "20px", justifyContent: "flex-end" }}>
            <button type="button" onClick={onCancel} style={styles.btnOutline}>Cancel</button>
            <button type="submit" style={styles.btnWhite}>Register</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RegisterForm;
