import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiAlertTriangle, FiCheckCircle, FiLogOut, FiRefreshCw, FiSend, FiShield } from 'react-icons/fi';
import { clearAuthSession } from '../../utils/authSession';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const statusLabel = (status) => status ? `HTTP ${status}` : 'No response';

export default function DashboardLoadError({ error, onRetry }) {
  const navigate = useNavigate();
  const [reportOpen, setReportOpen] = useState(false);
  const [title, setTitle] = useState('Dashboard unavailable');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('SYSTEM_ERROR');
  const [severity, setSeverity] = useState('HIGH');
  const [submitting, setSubmitting] = useState(false);
  const [reportError, setReportError] = useState('');
  const [ticketId, setTicketId] = useState('');

  const signOut = () => {
    clearAuthSession();
    navigate('/dashboard?login=1', { replace: true });
  };

  const submitReport = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setReportError('');
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Your sign-in has expired. Sign in again before reporting this issue.');
      const response = await fetch(`${BACKEND_URL}/api/support-tickets`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          title,
          description,
          category,
          severity,
          pagePath: window.location.pathname,
          failedRequest: error.failedRequest,
          errorCode: error.code,
          errorMessage: error.technicalMessage,
          primaryStatus: error.primaryStatus,
          fallbackStatus: error.fallbackStatus,
          occurredAt: error.occurredAt
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) {
        throw new Error(data.message || `Unable to submit support ticket (${response.status})`);
      }
      setTicketId(data.ticket?._id || 'submitted');
      setReportOpen(false);
      setDescription('');
    } catch (submitError) {
      setReportError(submitError.message || 'Unable to submit support ticket');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main style={{ minHeight: '100vh', boxSizing: 'border-box', padding: 'clamp(20px, 5vw, 64px) 20px', background: '#f1f5f8', color: '#102a43', fontFamily: 'inherit' }}>
      <div style={{ maxWidth: 920, margin: '0 auto' }}>
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, marginBottom: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#486581', fontSize: 13, fontWeight: 700 }}>
            <FiShield aria-hidden="true" color="#0f766e" size={18} />
            Clinical operations · Service status
          </div>
          <button type="button" onClick={signOut} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, minHeight: 40, padding: '0 14px', border: '1px solid #cbd5e1', borderRadius: 8, background: '#fff', color: '#334e68', fontWeight: 700, cursor: 'pointer' }}>
            <FiLogOut aria-hidden="true" /> Sign out
          </button>
        </header>

        <section role="alert" style={{ overflow: 'hidden', border: '1px solid #d5dee7', borderRadius: 16, background: '#fff', boxShadow: '0 14px 36px rgba(16, 42, 67, 0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, padding: '24px clamp(18px, 4vw, 34px)', borderBottom: '1px solid #e6edf2', background: '#f8fafc' }}>
            <span style={{ display: 'grid', flex: '0 0 42px', width: 42, height: 42, placeItems: 'center', borderRadius: 12, background: '#fff1f0', color: '#b42318' }}>
              <FiAlertTriangle aria-hidden="true" size={22} />
            </span>
            <div style={{ flex: 1 }}>
              <div style={{ color: '#b42318', fontSize: 11, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Workspace access interrupted</div>
              <h1 style={{ margin: '5px 0 0', color: '#102a43', fontSize: 'clamp(22px, 4vw, 30px)', lineHeight: 1.2 }}>Clinical dashboard unavailable</h1>
              <p style={{ margin: '9px 0 0', maxWidth: 650, color: '#486581', fontSize: 14, lineHeight: 1.6 }}>
                Your clinical workspace could not be verified. No patient chart content has been changed by this dashboard-loading attempt.
              </p>
            </div>
          </div>

          <div style={{ padding: '22px clamp(18px, 4vw, 34px) 28px' }}>
            <p style={{ margin: '0 0 16px', color: '#334e68', fontSize: 14, lineHeight: 1.6 }}>{error.message}</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10 }}>
              <Diagnostic label="Diagnostic code" value={error.code || 'DASHBOARD_LOAD_FAILED'} />
              {error.failedRequest && <Diagnostic label="Service request" value={error.failedRequest} />}
              {error.primaryStatus !== undefined && <Diagnostic label="Shift lookup" value={statusLabel(error.primaryStatus)} />}
              {error.fallbackStatus !== undefined && <Diagnostic label="Dashboard configuration" value={statusLabel(error.fallbackStatus)} />}
              <Diagnostic label="Occurred" value={error.occurredAt ? new Date(error.occurredAt).toLocaleString() : 'Just now'} />
            </div>

            {ticketId && (
              <div role="status" style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 16, padding: '11px 13px', border: '1px solid #a7f3d0', borderRadius: 9, background: '#ecfdf5', color: '#065f46', fontSize: 13, fontWeight: 700 }}>
                <FiCheckCircle aria-hidden="true" /> IT support ticket submitted{ticketId !== 'submitted' ? ` · ${ticketId}` : ''}.
              </div>
            )}

            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 22 }}>
              <button type="button" onClick={onRetry} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, minHeight: 42, padding: '0 16px', border: 0, borderRadius: 8, background: '#0f766e', color: '#fff', fontWeight: 800, cursor: 'pointer' }}>
                <FiRefreshCw aria-hidden="true" /> Retry dashboard
              </button>
              <button type="button" onClick={() => { setReportOpen((open) => !open); setReportError(''); }} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, minHeight: 42, padding: '0 16px', border: '1px solid #b8c7d1', borderRadius: 8, background: '#fff', color: '#243b53', fontWeight: 800, cursor: 'pointer' }}>
                <FiSend aria-hidden="true" /> {reportOpen ? 'Close report form' : 'Report to IT support'}
              </button>
            </div>

            {reportOpen && (
              <form onSubmit={submitReport} style={{ display: 'grid', gap: 14, marginTop: 22, padding: 18, border: '1px solid #d9e2ec', borderRadius: 12, background: '#f8fafc' }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: 18, color: '#102a43' }}>Report a dashboard issue</h2>
                  <p style={{ margin: '5px 0 0', color: '#627d98', fontSize: 12, lineHeight: 1.5 }}>This report includes the route and technical status codes shown above. Do not include patient names, identifiers, clinical details, passwords, or tokens.</p>
                </div>
                <label style={fieldStyle}>
                  Ticket title
                  <input required minLength={5} maxLength={120} value={title} onChange={(event) => setTitle(event.target.value)} style={inputStyle} />
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                  <label style={fieldStyle}>
                    Issue category
                    <select value={category} onChange={(event) => setCategory(event.target.value)} style={inputStyle}>
                      <option value="SYSTEM_ERROR">System error</option>
                      <option value="ACCESS">Access or sign-in</option>
                      <option value="PERFORMANCE">Performance</option>
                      <option value="OTHER">Other</option>
                    </select>
                  </label>
                  <label style={fieldStyle}>
                    Operational impact
                    <select value={severity} onChange={(event) => setSeverity(event.target.value)} style={inputStyle}>
                      <option value="LOW">Low</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="HIGH">High</option>
                      <option value="CRITICAL">Critical — clinical work blocked</option>
                    </select>
                  </label>
                </div>
                <label style={fieldStyle}>
                  What happened?
                  <textarea required minLength={10} maxLength={2000} rows={4} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Describe the issue without including patient details." style={{ ...inputStyle, resize: 'vertical' }} />
                </label>
                {reportError && <p role="alert" style={{ margin: 0, color: '#b42318', fontSize: 13 }}>{reportError}</p>}
                <div>
                  <button type="submit" disabled={submitting} style={{ minHeight: 40, padding: '0 16px', border: 0, borderRadius: 8, background: submitting ? '#9fb3c8' : '#0f766e', color: '#fff', fontWeight: 800, cursor: submitting ? 'wait' : 'pointer' }}>
                    {submitting ? 'Submitting ticket…' : 'Submit IT ticket'}
                  </button>
                </div>
              </form>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

const fieldStyle = { display: 'grid', gap: 6, color: '#334e68', fontSize: 12, fontWeight: 800 };
const inputStyle = { width: '100%', boxSizing: 'border-box', minHeight: 40, padding: '9px 10px', border: '1px solid #bcccdc', borderRadius: 7, background: '#fff', color: '#102a43', font: 'inherit', fontWeight: 500 };

function Diagnostic({ label, value }) {
  return (
    <div style={{ minWidth: 0, padding: '11px 12px', border: '1px solid #e6edf2', borderRadius: 9, background: '#fff' }}>
      <div style={{ color: '#627d98', fontSize: 10, fontWeight: 800, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ overflowWrap: 'anywhere', marginTop: 5, color: '#243b53', fontSize: 12, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
