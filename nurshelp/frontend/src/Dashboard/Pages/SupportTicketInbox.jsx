import React, { useCallback, useEffect, useState } from 'react';
import { FiAlertCircle, FiCheckCircle, FiRefreshCw, FiUserCheck } from 'react-icons/fi';
import { decodeJwt } from '../../utils/jwt';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
const STATUS_OPTIONS = [
  { value: 'NEW', label: 'New' },
  { value: 'IN_PROGRESS', label: 'In progress' },
  { value: 'RESOLVED', label: 'Resolved' }
];

export default function SupportTicketInbox() {
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [savingId, setSavingId] = useState('');
  const [error, setError] = useState('');
  const [statusFilter, setStatusFilter] = useState('OPEN');
  const [notes, setNotes] = useState({});
  const currentUserId = decodeJwt(localStorage.getItem('token'))?.id;

  const loadTickets = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const query = `?status=${encodeURIComponent(statusFilter)}`;
      const response = await fetch(`${BACKEND_URL}/api/support-tickets${query}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) throw new Error(data.message || `Unable to load tickets (${response.status})`);
      setTickets(Array.isArray(data.tickets) ? data.tickets : []);
    } catch (loadError) {
      setError(loadError.message || 'Unable to load IT support tickets');
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    loadTickets();
  }, [loadTickets]);

  const updateTicket = async (ticket, updates) => {
    setSavingId(ticket._id);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${BACKEND_URL}/api/support-tickets/${ticket._id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(updates)
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) throw new Error(data.message || `Unable to update ticket (${response.status})`);
      await loadTickets();
    } catch (updateError) {
      setError(updateError.message || 'Unable to update IT support ticket');
    } finally {
      setSavingId('');
    }
  };

  const openCount = tickets.filter((ticket) => ticket.status !== 'RESOLVED').length;

  return (
    <section style={{ display: 'grid', gap: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
        <div>
          <div style={{ color: '#0f766e', fontSize: 11, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase' }}>IT & maintenance</div>
          <h2 style={{ margin: '5px 0 0', color: '#102a43', fontSize: 23 }}>Support ticket queue</h2>
          <p style={{ margin: '6px 0 0', color: '#627d98', fontSize: 13 }}>Dashboard errors reported by staff in this hospital. Reports contain technical context only, not patient chart data.</p>
        </div>
        <button type="button" onClick={loadTickets} disabled={loading} style={buttonStyle}>
          <FiRefreshCw aria-hidden="true" /> {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </header>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ padding: '7px 10px', borderRadius: 999, background: '#ecfeff', color: '#155e75', fontSize: 12, fontWeight: 800 }}>
          {statusFilter === 'OPEN' ? `${openCount} open in current list` : `${tickets.length} tickets in current list`}
        </span>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: '#486581', fontSize: 12, fontWeight: 700 }}>
          Show
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} style={{ minHeight: 36, padding: '6px 9px', border: '1px solid #bcccdc', borderRadius: 7, background: '#fff', color: '#243b53' }}>
            <option value="OPEN">Open tickets</option>
            <option value="NEW">New</option>
            <option value="IN_PROGRESS">In progress</option>
            <option value="RESOLVED">Resolved</option>
          </select>
        </label>
      </div>

      {error && <div role="alert" style={{ padding: 12, border: '1px solid #fecaca', borderRadius: 8, background: '#fef2f2', color: '#991b1b', fontSize: 13 }}>{error}</div>}
      {loading && tickets.length === 0 && <p role="status" style={{ color: '#627d98' }}>Loading hospital support tickets…</p>}
      {!loading && tickets.length === 0 && !error && (
        <div style={{ display: 'grid', justifyItems: 'center', gap: 8, padding: 34, border: '1px dashed #bcccdc', borderRadius: 12, color: '#627d98', textAlign: 'center' }}>
          <FiCheckCircle size={24} color="#0f766e" aria-hidden="true" />
          <strong style={{ color: '#334e68' }}>No tickets in this view</strong>
          <span style={{ fontSize: 13 }}>New staff reports will appear here.</span>
        </div>
      )}

      {tickets.map((ticket) => (
        <article key={ticket._id} style={{ display: 'grid', gap: 13, padding: 17, border: '1px solid #d9e2ec', borderLeft: `4px solid ${ticket.severity === 'CRITICAL' ? '#b42318' : ticket.severity === 'HIGH' ? '#d97706' : '#0f766e'}`, borderRadius: 11, background: '#fff' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 14, flexWrap: 'wrap' }}>
            <div style={{ minWidth: 220, flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <h3 style={{ margin: 0, color: '#102a43', fontSize: 16 }}>{ticket.title}</h3>
                <Badge color={ticket.severity === 'CRITICAL' ? '#b42318' : ticket.severity === 'HIGH' ? '#9a3412' : '#155e75'}>{ticket.severity}</Badge>
                <Badge color={ticket.status === 'RESOLVED' ? '#166534' : '#334e68'}>{ticket.status.replace('_', ' ')}</Badge>
              </div>
              <p style={{ margin: '7px 0 0', color: '#486581', fontSize: 13, lineHeight: 1.5 }}>{ticket.description}</p>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 9, color: '#627d98', fontSize: 11 }}>
                <span>Reported by {ticket.reporterName} · {ticket.reporterRole}</span>
                <span>{ticket.createdAt ? new Date(ticket.createdAt).toLocaleString() : 'Time unavailable'}</span>
                <span>{ticket.category.replaceAll('_', ' ')}</span>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <select aria-label={`Status for ${ticket.title}`} value={ticket.status} onChange={(event) => updateTicket(ticket, { status: event.target.value })} disabled={savingId === ticket._id} style={{ minHeight: 36, padding: '6px 8px', border: '1px solid #bcccdc', borderRadius: 7, background: '#fff', color: '#243b53' }}>
                {STATUS_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
              <button type="button" onClick={() => updateTicket(ticket, { assignToMe: String(ticket.assignedToId || '') !== String(currentUserId || '') })} disabled={savingId === ticket._id} style={buttonStyle}>
                <FiUserCheck aria-hidden="true" /> {String(ticket.assignedToId || '') === String(currentUserId || '') ? 'Unassign from me' : ticket.assignedToId ? `Reassign from ${ticket.assignedToName || 'staff'}` : 'Assign to me'}
              </button>
            </div>
          </div>

          <details>
            <summary style={{ color: '#486581', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}>Technical report details</summary>
            <dl style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 9, margin: '12px 0 0' }}>
              <Detail label="Page" value={ticket.pagePath || 'Not recorded'} />
              <Detail label="Failed request" value={ticket.failedRequest || 'Not recorded'} />
              <Detail label="Diagnostic code" value={ticket.errorCode || 'Not recorded'} />
              <Detail label="Shift lookup" value={ticket.primaryStatus ? `HTTP ${ticket.primaryStatus}` : 'Not applicable'} />
              <Detail label="Config fallback" value={ticket.fallbackStatus ? `HTTP ${ticket.fallbackStatus}` : 'Not recorded'} />
              <Detail label="Technical message" value={ticket.errorMessage || 'Not recorded'} />
              <Detail label="Assigned to" value={ticket.assignedToName || 'Unassigned'} />
              {ticket.resolutionNote && <Detail label="Resolution note" value={ticket.resolutionNote} />}
            </dl>
          </details>

          {ticket.status !== 'RESOLVED' && (
            <label style={{ display: 'grid', gap: 6, color: '#486581', fontSize: 11, fontWeight: 800 }}>
              Resolution note (optional)
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <input maxLength={1000} value={notes[ticket._id] ?? ''} onChange={(event) => setNotes((current) => ({ ...current, [ticket._id]: event.target.value }))} placeholder="Record the maintenance action or next step" style={{ minWidth: 200, flex: '1 1 260px', minHeight: 36, boxSizing: 'border-box', padding: '7px 9px', border: '1px solid #bcccdc', borderRadius: 7 }} />
                <button type="button" onClick={() => updateTicket(ticket, { resolutionNote: notes[ticket._id] ?? '' })} disabled={savingId === ticket._id} style={buttonStyle}>
                  <FiAlertCircle aria-hidden="true" /> {savingId === ticket._id ? 'Saving…' : 'Save note'}
                </button>
              </div>
            </label>
          )}
        </article>
      ))}
    </section>
  );
}

const buttonStyle = { display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 7, minHeight: 36, padding: '0 11px', border: '1px solid #bcccdc', borderRadius: 7, background: '#fff', color: '#334e68', fontSize: 12, fontWeight: 800, cursor: 'pointer' };

function Badge({ color, children }) {
  return <span style={{ padding: '4px 7px', borderRadius: 999, background: `${color}12`, color, fontSize: 9, fontWeight: 900, letterSpacing: '0.04em' }}>{children}</span>;
}

function Detail({ label, value }) {
  return <div style={{ minWidth: 0, padding: 9, borderRadius: 7, background: '#f8fafc' }}><dt style={{ color: '#627d98', fontSize: 10, fontWeight: 800, textTransform: 'uppercase' }}>{label}</dt><dd style={{ margin: '4px 0 0', overflowWrap: 'anywhere', color: '#334e68', fontSize: 11 }}>{value}</dd></div>;
}
