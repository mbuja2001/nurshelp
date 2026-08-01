import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin from '@fullcalendar/interaction';
import { FiFilter } from 'react-icons/fi';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const typeLabels = {
  surgery: 'Scheduled Surgeries',
  lab: 'Lab Requests',
  med: 'Ordered Meds',
  meeting: 'Meetings',
  diagnosed: 'Diagnosed',
  discharged: 'Discharged',
  administered: 'Administered Meds'
};

export default function CalendarPage() {
  const navigate = useNavigate();
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({ surgeries: true, labRequests: true, orderedMeds: true, meetings: true, diagnosed: true, discharged: true, administered: true });
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    loadEvents();

    // Poll calendar events every 10 seconds for near-real-time updates
    const iv = setInterval(() => {
      if (mounted) loadEvents();
    }, 10000);

    return () => { mounted = false; clearInterval(iv); };
  }, []);

  const loadEvents = async () => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${BACKEND_URL}/api/calendar/events`, { headers });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.message || 'Failed to fetch calendar events');
      }

      const data = await res.json();
      if (Array.isArray(data.events)) {
        setEvents(data.events.map((e) => ({ ...e, title: e.title || e.type, start: e.start, end: e.end })));
        setLoading(false);
        return;
      }

      throw new Error('Invalid calendar payload received');
    } catch (err) {
      console.warn('[Calendar] Backend calendar endpoint failed, using sample events', err);
      setError(err.message);

      const now = new Date();
      const sample = [
        { id: 'surg-1', title: 'OR: Appendectomy', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 14, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 15, 30).toISOString(), type: 'surgery' },
        { id: 'lab-1', title: 'CT Chest - Bed 12', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 10, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 10, 30).toISOString(), type: 'lab' },
        { id: 'meet-1', title: 'Consultant Huddle', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 13, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 13, 30).toISOString(), type: 'meeting' },
        { id: 'med-1', title: 'Medication Order: IV Ceftriaxone', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 9, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 9, 15).toISOString(), type: 'med' },
        { id: 'diag-1', title: 'Diagnosed: Pneumonia', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 8, 30).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 9, 0).toISOString(), type: 'diagnosed' },
        { id: 'admin-1', title: 'Administered: IV Fluids', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 11, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 11, 15).toISOString(), type: 'administered' },
        { id: 'disc-1', title: 'Discharged: Mr. Kareem', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 16, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 16, 30).toISOString(), type: 'discharged' }
      ];
      setEvents(sample);
      setLoading(false);
    }
  };

  const filteredEvents = useMemo(() => events.filter((e) => {
    if (e.type === 'surgery') return filters.surgeries;
    if (e.type === 'lab') return filters.labRequests;
    if (e.type === 'med') return filters.orderedMeds;
    if (e.type === 'meeting') return filters.meetings;
    if (e.type === 'diagnosed') return filters.diagnosed;
    if (e.type === 'discharged') return filters.discharged;
    if (e.type === 'administered') return filters.administered;
    return true;
  }), [events, filters]);

  const toggleFilter = (key) => setFilters((prev) => ({ ...prev, [key]: !prev[key] }));

  const eventCounts = {
    surgery: events.filter((e) => e.type === 'surgery').length,
    lab: events.filter((e) => e.type === 'lab').length,
    med: events.filter((e) => e.type === 'med').length,
    meeting: events.filter((e) => e.type === 'meeting').length,
    diagnosed: events.filter((e) => e.type === 'diagnosed').length,
    discharged: events.filter((e) => e.type === 'discharged').length,
    administered: events.filter((e) => e.type === 'administered').length
  };

  const statusBadge = (type) => {
    switch (type) {
      case 'surgery': return { bg: '#eff6ff', color: '#2563eb' };
      case 'lab': return { bg: '#fef3c7', color: '#b45309' };
      case 'med': return { bg: '#d1fae5', color: '#0f766e' };
      case 'meeting': return { bg: '#ede9fe', color: '#5b21b6' };
      case 'diagnosed': return { bg: '#ffedd5', color: '#c2410c' };
      case 'discharged': return { bg: '#e2e8f0', color: '#1f2937' };
      case 'administered': return { bg: '#dcfce7', color: '#166534' };
      default: return { bg: '#f3f4f6', color: '#374151' };
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 1320, margin: '0 auto' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 20 }}>
        <div style={{ minWidth: 280 }}>
          <p style={{ margin: 0, fontSize: 12, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Physician Task Calendar</p>
          <h1 style={{ margin: '12px 0 6px', fontSize: 32, color: '#111827' }}>Scheduled surgeries, meds & labs</h1>
          <p style={{ margin: 0, color: '#475569', lineHeight: 1.7, maxWidth: 680 }}>
            Live surgery bookings are now reflected from the surgery workflow. Toggle task categories, refresh the view, and use this calendar as the physician’s command center for the next 30 days.
          </p>
        </div>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <button onClick={() => navigate(-1)} style={{ padding: '12px 18px', borderRadius: 14, border: '1px solid #cbd5e1', background: '#fff', color: '#334155', fontWeight: 700, cursor: 'pointer' }}>Back</button>
          <button onClick={() => setShowFilters((s) => !s)} title="Filters" style={{ padding: '12px 16px', borderRadius: 14, border: '1px solid #cbd5e1', background: '#fff', color: '#334155', display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
            <FiFilter /> Filters
          </button>
          <button onClick={loadEvents} style={{ padding: '12px 18px', borderRadius: 14, border: '1px solid #6366f1', background: '#eef2ff', color: '#4338ca', fontWeight: 700, cursor: 'pointer' }}>Refresh</button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(180px, 1fr))', gap: 14, marginBottom: 20 }}>
        {Object.entries({ surgery: 'Surgeries', lab: 'Labs', med: 'Med orders', meeting: 'Meetings', diagnosed: 'Diagnosed', discharged: 'Discharged', administered: 'Administered Meds' }).map(([key, label]) => {
          const count = eventCounts[key];
          return (
            <div key={key} style={{ padding: 18, borderRadius: 18, background: '#fff', border: '1px solid #e5e7eb', minHeight: 100, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#111827', marginBottom: 8 }}>{label}</div>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{count}</div>
              </div>
              <span style={{ fontSize: 12, color: '#64748b' }}>{typeLabels[key] || label}</span>
            </div>
          );
        })}
      </div>

      {showFilters && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 22, alignItems: 'center' }}>
          {[
            { key: 'surgeries', label: 'Scheduled Surgeries' },
            { key: 'labRequests', label: 'Lab Requests' },
            { key: 'orderedMeds', label: 'Ordered Meds' },
            { key: 'meetings', label: 'Meetings' },
            { key: 'diagnosed', label: 'Diagnosed' },
            { key: 'discharged', label: 'Discharged' },
            { key: 'administered', label: 'Administered Meds' }
          ].map((filter) => (
            <button
              key={filter.key}
              onClick={() => toggleFilter(filter.key)}
              style={{
                padding: '10px 16px',
                borderRadius: 999,
                border: filters[filter.key] ? '1px solid #4338ca' : '1px solid #d1d5db',
                background: filters[filter.key] ? '#eef2ff' : '#f8fafc',
                color: filters[filter.key] ? '#4338ca' : '#475569',
                cursor: 'pointer',
                fontWeight: 600
              }}
            >
              {filter.label}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div style={{ marginBottom: 18, padding: 16, borderRadius: 16, background: '#f8d7da', color: '#842029', border: '1px solid #f5c2c7' }}>
          <strong>Calendar warning:</strong> {error}
        </div>
      )}

      <div style={{ background: '#fff', borderRadius: 16, padding: 16, boxShadow: '0 6px 24px rgba(15, 23, 42, 0.06)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 18 }}>
          {Object.entries(typeLabels).map(([type, label]) => {
            const badge = statusBadge(type);
            return (
              <span key={type} style={{ padding: '8px 14px', borderRadius: 999, background: badge.bg, color: badge.color, fontSize: 13, fontWeight: 600 }}>{label}</span>
            );
          })}
        </div>

        <FullCalendar
          plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin]}
          initialView="timeGridWeek"
          headerToolbar={{ left: 'prev,next today', center: 'title', right: 'dayGridMonth,timeGridWeek,timeGridDay' }}
          events={filteredEvents}
          nowIndicator
          editable={false}
          droppable={false}
          height={720}
        />
      </div>

      {loading && (
        <div style={{ marginTop: 18, color: '#475569', fontSize: 14 }}>Loading events…</div>
      )}
    </div>
  );
}
