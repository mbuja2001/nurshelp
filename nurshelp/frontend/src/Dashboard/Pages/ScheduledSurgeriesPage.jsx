import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaProcedures, FaCalendarAlt } from 'react-icons/fa';
import { FiFilter, FiChevronRight } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

export default function ScheduledSurgeriesPage() {
  const navigate = useNavigate();
  const [surgeries, setSurgeries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ dateRange: '30days', status: 'all', surgeon: 'all', showFilters: false });

  useEffect(() => { loadSurgeries(); }, []);
  useEffect(() => {
    let mounted = true;
    // Poll every 10s for live updates
    const iv = setInterval(() => {
      if (mounted) loadSurgeries();
    }, 10000);
    return () => { mounted = false; clearInterval(iv); };
  }, []);

  const loadSurgeries = async () => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      // Fetch encounters and extract surgery schedules
      // Use dedicated physician surgeries endpoint
      const res = await fetch(`${BACKEND_URL}/api/encounters/physician/surgeries?t=${Date.now()}`, { headers });
      if (!res.ok) throw new Error('Failed to fetch surgeries');
      const data = await res.json();
      if (!Array.isArray(data.surgeries)) {
        throw new Error('Invalid surgery schedule response');
      }
      const all = data.surgeries;
      all.sort((a, b) => new Date(a.scheduled_at || 0) - new Date(b.scheduled_at || 0));
      setSurgeries(all);
    } catch (err) {
      console.error('Error loading surgeries:', err);
      setSurgeries([]);
      setError(err.message || 'Unable to load surgeries');
    } finally {
      setLoading(false);
    }
  };

  const statusBadge = (status) => {
    const styles = {
      PENDING: { bg: '#fef3c7', color: '#b45309' },
      SCHEDULED: { bg: '#eff6ff', color: '#2563eb' },
      IN_PROGRESS: { bg: '#fde68a', color: '#92400e' },
      COMPLETED: { bg: '#dcfce7', color: '#166534' },
      CANCELLED: { bg: '#fee2e2', color: '#991b1b' }
    };
    return styles[status] || styles['PENDING'];
  };

  const filtered = surgeries.filter(s => {
    if (filters.status !== 'all' && s.status !== filters.status) return false;
    if (filters.surgeon !== 'all' && s.surgeon !== filters.surgeon) return false;
    return true;
  });

  return (
    <SidebarLayout>
    <div style={{ padding: 24, maxWidth: 1320, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 28 }}>Scheduled Surgeries</h1>
          <p style={{ margin: 0, color: '#64748b' }}>Manage upcoming operating theatre bookings and pre-op status.</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={() => navigate(-1)} style={{ padding: '10px 16px', borderRadius: 12, border: '1px solid #cbd5e1', background: '#fff' }}>Back</button>
          <button onClick={loadSurgeries} style={{ padding: '10px 16px', borderRadius: 12, border: '1px solid #6366f1', background: '#eef2ff', color: '#4338ca' }}>Refresh</button>
        </div>
      </div>

      <div role="status" style={{ marginBottom: 18, padding: 16, borderRadius: 12, background: '#fffbeb', color: '#92400e', border: '1px solid #fde68a' }}>
        Surgery creation and updates are temporarily disabled while the approved scheduling workflow is implemented. Existing verified assignments remain viewable.
      </div>

      {error && (
        <div role="alert" style={{ marginBottom: 18, padding: 16, borderRadius: 12, background: '#fef2f2', color: '#991b1b', border: '1px solid #fecaca' }}>
          <strong>Scheduled surgeries unavailable:</strong> {error}. No sample records are displayed.
        </div>
      )}

      {filters.showFilters && (
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 10, background: '#f8fafc' }}>
          <label style={{ display: 'block', marginBottom: 6 }}>Status</label>
          <select value={filters.status} onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}>
            <option value="all">All</option>
            <option value="PENDING">Pending</option>
            <option value="SCHEDULED">Scheduled</option>
            <option value="IN_PROGRESS">In Progress</option>
            <option value="COMPLETED">Completed</option>
            <option value="CANCELLED">Cancelled</option>
          </select>
        </div>
      )}

      {loading && <div style={{ padding: 20 }}>Loading...</div>}

      {!loading && !error && filtered.length === 0 && (
        <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No scheduled surgeries found.</div>
      )}

      {!loading && filtered.length > 0 && (
        <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ background: '#f8fafc' }}>
              <tr>
                <th style={{ padding: 12, textAlign: 'left' }}>Patient</th>
                <th style={{ padding: 12, textAlign: 'left' }}>Procedure</th>
                <th style={{ padding: 12, textAlign: 'left' }}>Surgeon</th>
                <th style={{ padding: 12, textAlign: 'center' }}>Scheduled</th>
                <th style={{ padding: 12, textAlign: 'center' }}>Status</th>
                <th style={{ padding: 12, textAlign: 'center' }}>Pre-op</th>
                <th style={{ padding: 12, textAlign: 'center' }}></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(s => {
                const badge = statusBadge(s.status);
                return (
                  <tr key={s.id} style={{ borderBottom: '1px solid #e5e7eb' }}>
                    <td style={{ padding: 12 }}>{s.patientName}</td>
                    <td style={{ padding: 12 }}>{s.procedure_name}</td>
                    <td style={{ padding: 12 }}>{s.surgeon}</td>
                    <td style={{ padding: 12, textAlign: 'center' }}>{s.scheduled_at ? new Date(s.scheduled_at).toLocaleString() : '-'}</td>
                    <td style={{ padding: 12, textAlign: 'center' }}><span style={{ padding: '6px 10px', borderRadius: 8, background: badge.bg, color: badge.color }}>{s.status}</span></td>
                    <td style={{ padding: 12, textAlign: 'center' }}>{s.preopChecklistComplete ? 'Yes' : 'No'}</td>
                    <td style={{ padding: 12, textAlign: 'center' }}>
                      <button onClick={() => navigate(`/summary/${s.encounterId}`)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8' }}><FiChevronRight /></button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

    </div>
    </SidebarLayout>
  );
}

