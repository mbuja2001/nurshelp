import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaProcedures, FaCalendarAlt } from 'react-icons/fa';
import { FiFilter, FiChevronRight } from 'react-icons/fi';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

export default function ScheduledSurgeriesPage() {
  const navigate = useNavigate();
  const [surgeries, setSurgeries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [scheduleForm, setScheduleForm] = useState({ encounterId: '', procedure_name: '', surgery_type: 'ELECTIVE', scheduled_at: '', surgeon: '', priority: 'MEDIUM' });
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
      const all = data.surgeries || data.data || [];
      all.sort((a, b) => new Date(a.scheduled_at || 0) - new Date(b.scheduled_at || 0));
      setSurgeries(all);
      setLoading(false);
    } catch (err) {
      console.error('Error loading surgeries:', err);
      setError(err.message);
      const now = new Date();
      const sample = [
        { id: 's1', patientName: 'John Doe', procedure_name: 'Appendectomy', scheduled_at: new Date(now.getTime() + 24*60*60*1000).toISOString(), surgeon: 'Dr. Smith', status: 'SCHEDULED', priority: 'HIGH', preopChecklistComplete: true, encounterId: 'e1' },
        { id: 's2', patientName: 'Jane Smith', procedure_name: 'Cholecystectomy', scheduled_at: new Date(now.getTime() + 48*60*60*1000).toISOString(), surgeon: 'Dr. Lee', status: 'PENDING', priority: 'MEDIUM', preopChecklistComplete: false, encounterId: 'e2' }
      ];
      setSurgeries(sample);
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

  const handleCreateSurgery = async () => {
    if (!scheduleForm.encounterId || !scheduleForm.procedure_name) {
      alert('Please provide an encounter ID and procedure name');
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const body = {
        procedure_name: scheduleForm.procedure_name,
        surgery_type: scheduleForm.surgery_type,
        scheduled_at: scheduleForm.scheduled_at || null,
        surgeon: scheduleForm.surgeon || undefined,
        priority: scheduleForm.priority
      };

      const res = await fetch(`${BACKEND_URL}/api/encounters/${scheduleForm.encounterId}/surgery`, {
        method: 'POST', headers, body: JSON.stringify(body)
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.message || 'Failed to create surgery record');
      }

      await res.json();
      setShowScheduleModal(false);
      setScheduleForm({ encounterId: '', procedure_name: '', surgery_type: 'ELECTIVE', scheduled_at: '', surgeon: '', priority: 'MEDIUM' });
      // Refresh local list and hint calendar to refresh
      await loadSurgeries();
      fetch(`${BACKEND_URL}/api/calendar/events`).catch(() => {});
    } catch (err) {
      console.error('Create surgery failed', err);
      alert(err.message || 'Failed to create surgery');
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 1320, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 28 }}>Scheduled Surgeries</h1>
          <p style={{ margin: 0, color: '#64748b' }}>Manage upcoming operating theatre bookings and pre-op status.</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={() => navigate(-1)} style={{ padding: '10px 16px', borderRadius: 12, border: '1px solid #cbd5e1', background: '#fff' }}>Back</button>
          <button onClick={loadSurgeries} style={{ padding: '10px 16px', borderRadius: 12, border: '1px solid #6366f1', background: '#eef2ff', color: '#4338ca' }}>Refresh</button>
          <button onClick={() => setShowScheduleModal(true)} style={{ padding: '10px 16px', borderRadius: 12, border: '1px solid #10b981', background: '#ecfdf5', color: '#059669' }}>New Surgery</button>
        </div>
      </div>

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

      {!loading && filtered.length === 0 && (
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

      {/* Schedule Modal */}
      {showScheduleModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 60 }} onClick={() => setShowScheduleModal(false)}>
          <div style={{ width: 720, background: '#fff', borderRadius: 12, padding: 20 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>Schedule Surgery</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <label style={{ fontSize: 12 }}>Encounter ID</label>
                <input value={scheduleForm.encounterId} onChange={(e) => setScheduleForm(prev => ({ ...prev, encounterId: e.target.value }))} style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #e5e7eb' }} />
              </div>
              <div>
                <label style={{ fontSize: 12 }}>Procedure</label>
                <input value={scheduleForm.procedure_name} onChange={(e) => setScheduleForm(prev => ({ ...prev, procedure_name: e.target.value }))} style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #e5e7eb' }} />
              </div>
              <div>
                <label style={{ fontSize: 12 }}>Surgeon</label>
                <input value={scheduleForm.surgeon} onChange={(e) => setScheduleForm(prev => ({ ...prev, surgeon: e.target.value }))} style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #e5e7eb' }} />
              </div>
              <div>
                <label style={{ fontSize: 12 }}>Surgery Type</label>
                <select value={scheduleForm.surgery_type} onChange={(e) => setScheduleForm(prev => ({ ...prev, surgery_type: e.target.value }))} style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #e5e7eb' }}>
                  <option value="ELECTIVE">ELECTIVE</option>
                  <option value="URGENT">URGENT</option>
                  <option value="EMERGENCY">EMERGENCY</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: 12 }}>Scheduled At</label>
                <input type="datetime-local" value={scheduleForm.scheduled_at} onChange={(e) => setScheduleForm(prev => ({ ...prev, scheduled_at: e.target.value }))} style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #e5e7eb' }} />
              </div>
              <div>
                <label style={{ fontSize: 12 }}>Priority</label>
                <select value={scheduleForm.priority} onChange={(e) => setScheduleForm(prev => ({ ...prev, priority: e.target.value }))} style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #e5e7eb' }}>
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                  <option value="URGENT">URGENT</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 16 }}>
              <button onClick={() => setShowScheduleModal(false)} style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', background: '#fff' }}>Cancel</button>
              <button onClick={handleCreateSurgery} style={{ padding: '8px 12px', borderRadius: 8, border: 'none', background: '#059669', color: '#fff' }}>Schedule</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

