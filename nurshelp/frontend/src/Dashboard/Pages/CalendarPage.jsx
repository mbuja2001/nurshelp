import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin from '@fullcalendar/interaction';
import { FiFilter } from 'react-icons/fi';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

export default function CalendarPage() {
  const navigate = useNavigate();
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({ surgeries: true, labRequests: true, orderedMeds: true, meetings: true });

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    setLoading(true);
    try {
      // Try backend aggregate endpoint if available
      const res = await fetch(`${BACKEND_URL}/api/calendar/events`, { headers: { 'Content-Type': 'application/json' } });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.events)) {
          setEvents(data.events.map((e) => ({ ...e, title: e.title || e.type, start: e.start, end: e.end })));
          setLoading(false);
          return;
        }
      }
    } catch (err) {
      // fallback to sample events
      console.warn('[Calendar] No backend calendar endpoint found, using sample events', err);
    }

    // Fallback sample events
    const now = new Date();
    const sample = [
      { id: 'surg-1', title: 'OR: Appendectomy', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 14, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 15, 30).toISOString(), type: 'surgery' },
      { id: 'lab-1', title: 'CT Chest - Bed 12', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 10, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 10, 30).toISOString(), type: 'lab' },
      { id: 'meet-1', title: 'Consultant Huddle', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 13, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 13, 30).toISOString(), type: 'meeting' },
      { id: 'med-1', title: 'Med review - Ward B', start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 9, 0).toISOString(), end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 9, 30).toISOString(), type: 'med' }
    ];
    setEvents(sample);
    setLoading(false);
  };

  const filteredEvents = useMemo(() => events.filter((e) => {
    if (e.type === 'surgery') return filters.surgeries;
    if (e.type === 'lab') return filters.labRequests;
    if (e.type === 'med') return filters.orderedMeds;
    if (e.type === 'meeting') return filters.meetings;
    return true;
  }), [events, filters]);

  const toggleFilter = (key) => setFilters(prev => ({ ...prev, [key]: !prev[key] }));

  return (
    <div style={{ padding: 20, maxWidth: 1300, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 28 }}>Calendar</h1>
          <p style={{ margin: 0, color: '#6b7280' }}>Google Calendar-like view for scheduled surgeries, lab requests, meds and meetings.</p>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button onClick={() => navigate(-1)} style={{ padding: '8px 12px', borderRadius: 8 }}>Back</button>
          <button title="Filters" onClick={() => setShowFilters(s => !s)} style={{ padding: 10, borderRadius: 10, border: '1px solid #e5e7eb', background: '#fff', cursor: 'pointer' }}>
            <FiFilter />
          </button>
        </div>
      </div>

      {showFilters && (
        <div style={{ marginBottom: 12, display: 'flex', gap: 12, alignItems: 'center' }}>
          <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
            <input type="checkbox" checked={filters.surgeries} onChange={() => toggleFilter('surgeries')} /> Scheduled Surgeries
          </label>
          <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
            <input type="checkbox" checked={filters.labRequests} onChange={() => toggleFilter('labRequests')} /> Lab Requests
          </label>
          <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
            <input type="checkbox" checked={filters.orderedMeds} onChange={() => toggleFilter('orderedMeds')} /> Ordered Meds
          </label>
          <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
            <input type="checkbox" checked={filters.meetings} onChange={() => toggleFilter('meetings')} /> Meetings
          </label>
          <button onClick={loadEvents} style={{ marginLeft: 12, padding: '8px 12px', borderRadius: 8 }}>Refresh</button>
        </div>
      )}

      <div style={{ background: '#fff', borderRadius: 12, padding: 12, boxShadow: '0 6px 24px rgba(0,0,0,0.06)' }}>
        <FullCalendar
          plugins={[ dayGridPlugin, timeGridPlugin, interactionPlugin ]}
          initialView="timeGridWeek"
          headerToolbar={{ left: 'prev,next today', center: 'title', right: 'dayGridMonth,timeGridWeek,timeGridDay' }}
          events={filteredEvents}
          nowIndicator
          editable={false}
          droppable={false}
          height={720}
        />
      </div>
    </div>
  );
}
