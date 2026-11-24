import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin from '@fullcalendar/interaction';
import { FiFilter } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const typeLabels = {
  surgery: 'Scheduled Surgeries',
  appointment: 'Appointments'
};

const disabledCategoryLabels = {
  meeting: 'Meetings',
  research: 'Research',
  lab: 'Lab requests',
  med: 'Medication orders',
  diagnosed: 'Diagnosis events',
  discharged: 'Discharge events',
  administered: 'Medication administrations',
  handoff: 'Handovers',
  radiology: 'Radiology',
  pathology: 'Pathology'
};

export default function CalendarPage() {
  const navigate = useNavigate();
  const [events, setEvents] = useState([]);
  const [disabledCategories, setDisabledCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({ surgery: true, appointment: true });
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
      if (!Array.isArray(data.events) || !Array.isArray(data.disabledCategories)) {
        throw new Error('Invalid calendar payload received');
      }
      if (data.events.some((event) => !['surgery', 'appointment'].includes(event.type))) {
        throw new Error('Calendar response included a disabled event category');
      }

      setEvents(data.events.map((event) => ({
        ...event,
        title: event.title || event.type,
        start: event.start,
        end: event.end
      })));
      setDisabledCategories(data.disabledCategories);
    } catch (err) {
      setEvents([]);
      setError(err.message || 'Unable to load calendar events');
    } finally {
      setLoading(false);
    }
  };

  const filteredEvents = useMemo(() => events.filter((event) => {
    return filters[event.type] === true;
  }), [events, filters]);

  const toggleFilter = (key) => setFilters((prev) => ({ ...prev, [key]: !prev[key] }));

  const eventCounts = {
    surgery: events.filter((event) => event.type === 'surgery').length,
    appointment: events.filter((event) => event.type === 'appointment').length
  };

  const statusBadge = (type) => {
    switch (type) {
      case 'surgery': return { bg: '#eff6ff', color: '#2563eb' };
      case 'appointment': return { bg: '#ede9fe', color: '#5b21b6' };
      default: return { bg: '#f3f4f6', color: '#374151' };
    }
  };

  return (
    <SidebarLayout>
    <div style={{ padding: 24, maxWidth: 1320, margin: '0 auto' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 20 }}>
        <div style={{ minWidth: 280 }}>
          <p style={{ margin: 0, fontSize: 12, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Authorized Calendar</p>
          <h1 style={{ margin: '12px 0 6px', fontSize: 32, color: '#111827' }}>Surgeries & appointments</h1>
          <p style={{ margin: 0, color: '#475569', lineHeight: 1.7, maxWidth: 680 }}>
            Only appointments assigned to you and surgeries with a verified surgeon assignment are shown.
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

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(180px, 1fr))', gap: 14, marginBottom: 20 }}>
        {Object.entries({ surgery: 'Surgeries', appointment: 'Appointments' }).map(([key, label]) => {
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
          {Object.entries(typeLabels).map(([key, label]) => (
            <button
              key={key}
              onClick={() => toggleFilter(key)}
              style={{
                padding: '10px 16px',
                borderRadius: 999,
                border: filters[key] ? '1px solid #4338ca' : '1px solid #d1d5db',
                background: filters[key] ? '#eef2ff' : '#f8fafc',
                color: filters[key] ? '#4338ca' : '#475569',
                cursor: 'pointer',
                fontWeight: 600
              }}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {disabledCategories.length > 0 && (
        <div role="status" style={{ marginBottom: 18, padding: 16, borderRadius: 16, background: '#fffbeb', color: '#92400e', border: '1px solid #fde68a' }}>
          Some calendar categories are temporarily unavailable: {disabledCategories.map((category) => disabledCategoryLabels[category] || category).join(', ')}. No records or counts for these categories are shown.
        </div>
      )}

      {error && (
        <div role="alert" style={{ marginBottom: 18, padding: 16, borderRadius: 16, background: '#f8d7da', color: '#842029', border: '1px solid #f5c2c7' }}>
          <strong>Calendar unavailable:</strong> {error}. No sample events are displayed.
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
        {!loading && !error && filteredEvents.length === 0 && (
          <p role="status" style={{ margin: '16px 0 0', color: '#64748b' }}>No assigned surgeries or appointments in this date range.</p>
        )}
      </div>

      {loading && (
        <div role="status" style={{ marginTop: 18, color: '#475569', fontSize: 14 }}>Loading events…</div>
      )}
    </div>
    </SidebarLayout>
  );
}
