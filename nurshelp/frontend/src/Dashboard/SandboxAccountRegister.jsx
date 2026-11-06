import React, { useState } from 'react';
import { FiArrowLeft, FiUserPlus } from 'react-icons/fi';

const SandboxAccountRegister = ({ onRegister, onLogin, onCancel }) => {
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      await onRegister(form);
    } catch (submitError) {
      setError(submitError.message || 'Unable to create account.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="sandbox-register-page">
      <section className="sandbox-register-panel">
        <button className="sandbox-back-button" type="button" onClick={onCancel}><FiArrowLeft /> Back</button>
        <div className="sandbox-register-kicker">Nurshelpdesk workspace</div>
        <h1>Create your account</h1>
        <p className="sandbox-register-copy">Your account is separate from hospital staff access. You can use it to manage your sandbox workspace.</p>
        <form onSubmit={handleSubmit} className="sandbox-register-form">
          <label>Full name<input autoComplete="name" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} required /></label>
          <label>Email address<input type="email" autoComplete="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} required /></label>
          <label>Password<input type="password" autoComplete="new-password" minLength={8} value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} required /></label>
          {error && <div className="sandbox-form-error" role="alert">{error}</div>}
          <button className="sandbox-primary-button" type="submit" disabled={submitting}><FiUserPlus /> {submitting ? 'Creating account...' : 'Create account'}</button>
        </form>
        <div className="sandbox-login-prompt">Already registered? <button type="button" onClick={onLogin}>Log in</button></div>
      </section>
    </main>
  );
};

export default SandboxAccountRegister;