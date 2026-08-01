import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiFilter, FiChevronRight } from 'react-icons/fi';
import { FaHistory, FaChartLine } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

export default function HistoryActionsPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ 
    dateRange: '30days', 
    status: 'all',
    showFilters: false 
  });

  useEffect(() => {
    let mounted = true;
    loadHistory();

    // Poll for live updates every 10 seconds
    const iv = setInterval(() => {
      if (mounted) loadHistory();
    }, 10000);

    return () => { mounted = false; clearInterval(iv); };
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      // Fetch all encounters to show history
      // Use physician-specific aggregated endpoint first (returns lightweight encounters)
      const listRes = await fetch(`${BACKEND_URL}/api/encounters/physician/all?t=${Date.now()}`, { headers });
      if (!listRes.ok) throw new Error('Failed to fetch physician encounters');
      const listData = await listRes.json();
      const encounters = listData.encounters || [];

      // Fetch full encounter details for the most recent 20 encounters to get status_history and handovers
      const slice = encounters.slice(0, 20);
      const detailPromises = slice.map(e => fetch(`${BACKEND_URL}/api/encounters/${e._id}?t=${Date.now()}`, { headers }).then(r => r.ok ? r.json() : null).catch(() => null));
      const details = await Promise.all(detailPromises);

      const allEvents = [];
      details.forEach(detail => {
        if (!detail || !detail.encounter) return;
        const enc = detail.encounter;

        if (enc.status_history && Array.isArray(enc.status_history)) {
          enc.status_history.forEach(event => {
            allEvents.push({
              id: `${enc._id}-${event.status}-${event.changed_at || event.timestamp}`,
              encounterId: enc._id,
              patientName: enc.patient?.name || 'Unknown Patient',
              patientId: enc.patient?._id,
              status: event.status,
              timestamp: event.changed_at || event.timestamp,
              notes: event.notes || '',
              updatedBy: event.changed_by || event.updated_by || 'System'
            });
          });
        }

        if (enc.clinical_handovers && Array.isArray(enc.clinical_handovers)) {
          enc.clinical_handovers.forEach((handover, idx) => {
            allEvents.push({
              id: `${enc._id}-handover-${idx}`,
              encounterId: enc._id,
              patientName: enc.patient?.name || 'Unknown Patient',
              patientId: enc.patient?._id,
              status: 'HANDOVER',
              timestamp: handover.handover_at || handover.created_at,
              notes: handover.summary || '',
              updatedBy: handover.from_clinician?.name || handover.from || 'Unknown Clinician',
              handoverTo: handover.to_clinician?.name || handover.to || 'Unknown Clinician'
            });
          });
        }
      });

      allEvents.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      setHistory(allEvents);
      setLoading(false);
    } catch (err) {
      console.error('Error loading history:', err);
      setError(err.message);
      
      // Sample data for demonstration
      const now = new Date();
      const sampleHistory = [
        {
          id: '1',
          patientName: 'John Doe',
          patientId: 'p1',
          status: 'DISCHARGED',
          timestamp: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(),
          notes: 'Patient stable, discharged to home care',
          updatedBy: 'Dr. Smith'
        },
        {
          id: '2',
          patientName: 'Jane Smith',
          patientId: 'p2',
          status: 'HOSPITALIZED',
          timestamp: new Date(now.getTime() - 5 * 60 * 60 * 1000).toISOString(),
          notes: 'Admitted to ward B3',
          updatedBy: 'Dr. Johnson'
        },
        {
          id: '3',
          patientName: 'John Doe',
          patientId: 'p1',
          status: 'DIAGNOSED',
          timestamp: new Date(now.getTime() - 8 * 60 * 60 * 1000).toISOString(),
          notes: 'Diagnosis: Acute Respiratory Infection',
          updatedBy: 'Dr. Smith'
        },
        {
          id: '4',
          patientName: 'Mike Brown',
          patientId: 'p3',
          status: 'HANDOVER',
          timestamp: new Date(now.getTime() - 12 * 60 * 60 * 1000).toISOString(),
          notes: 'Shift handover: Patient stable, vitals normal',
          updatedBy: 'Nurse Alice',
          handoverTo: 'Nurse Bob'
        }
      ];
      setHistory(sampleHistory);
      setLoading(false);
    }
  };

  const statusBadge = (status) => {
    const statusStyles = {
      'TRIAGE': { bg: '#fef3c7', color: '#b45309', label: 'Triage' },
      'DIAGNOSED': { bg: '#ffedd5', color: '#c2410c', label: 'Diagnosed' },
      'HOSPITALIZED': { bg: '#eff6ff', color: '#2563eb', label: 'Hospitalized' },
      'DISCHARGED': { bg: '#e2e8f0', color: '#1f2937', label: 'Discharged' },
      'HANDOVER': { bg: '#ede9fe', color: '#5b21b6', label: 'Handover' },
      'CRITICAL': { bg: '#fee2e2', color: '#991b1b', label: 'Critical' }
    };
    return statusStyles[status] || statusStyles['TRIAGE'];
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffHours = Math.floor((now - date) / (1000 * 60 * 60));
    
    if (diffHours < 1) return 'Just now';
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString();
  };

  const filteredHistory = history.filter(item => {
    // Apply date range filter
    if (filters.dateRange !== 'all') {
      const now = new Date();
      const itemDate = new Date(item.timestamp);
      const diffDays = Math.floor((now - itemDate) / (1000 * 60 * 60 * 24));

      switch (filters.dateRange) {
        case '7days': if (diffDays > 7) return false; break;
        case '30days': if (diffDays > 30) return false; break;
        case '90days': if (diffDays > 90) return false; break;
        default: break;
      }
    }

    // Apply status filter
    if (filters.status !== 'all' && item.status !== filters.status) {
      return false;
    }

    return true;
  });

  return (
    <div style={{ padding: 24, maxWidth: 1320, margin: '0 auto', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      {/* Header */}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 28 }}>
        <div style={{ minWidth: 280 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <FaHistory style={{ fontSize: 24, color: '#5b21b6' }} />
            <p style={{ margin: 0, fontSize: 12, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Activity Log</p>
          </div>
          <h1 style={{ margin: '12px 0 6px', fontSize: 32, color: '#111827' }}>History & Actions</h1>
          <p style={{ margin: 0, color: '#475569', lineHeight: 1.7, maxWidth: 680 }}>
            Review recent physician and system actions, patient status transitions, and clinical handovers in chronological order.
          </p>
        </div>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <button 
            onClick={() => navigate(-1)} 
            style={{ 
              padding: '12px 18px', 
              borderRadius: 14, 
              border: '1px solid #cbd5e1', 
              background: '#fff', 
              color: '#334155', 
              fontWeight: 700, 
              cursor: 'pointer',
              fontSize: 14
            }}
          >
            Back
          </button>
          <button 
            onClick={() => setFilters(prev => ({ ...prev, showFilters: !prev.showFilters }))} 
            style={{ 
              padding: '12px 16px', 
              borderRadius: 14, 
              border: '1px solid #cbd5e1', 
              background: '#fff', 
              color: '#334155', 
              display: 'inline-flex', 
              alignItems: 'center', 
              gap: 8, 
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: 14
            }}
          >
            <FiFilter /> Filters
          </button>
          <button 
            onClick={loadHistory} 
            style={{ 
              padding: '12px 18px', 
              borderRadius: 14, 
              border: '1px solid #6366f1', 
              background: '#eef2ff', 
              color: '#4338ca', 
              fontWeight: 700, 
              cursor: 'pointer',
              fontSize: 14
            }}
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      {filters.showFilters && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 22, padding: 16, background: '#f8fafc', borderRadius: 12, border: '1px solid #e2e8f0' }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#334155', display: 'block', marginBottom: 6 }}>Date Range</label>
            <select 
              value={filters.dateRange}
              onChange={(e) => setFilters(prev => ({ ...prev, dateRange: e.target.value }))}
              style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 14, cursor: 'pointer' }}
            >
              <option value="all">All Time</option>
              <option value="7days">Last 7 Days</option>
              <option value="30days">Last 30 Days</option>
              <option value="90days">Last 90 Days</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#334155', display: 'block', marginBottom: 6 }}>Status</label>
            <select 
              value={filters.status}
              onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
              style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 14, cursor: 'pointer' }}
            >
              <option value="all">All Statuses</option>
              <option value="TRIAGE">Triage</option>
              <option value="DIAGNOSED">Diagnosed</option>
              <option value="HOSPITALIZED">Hospitalized</option>
              <option value="DISCHARGED">Discharged</option>
              <option value="HANDOVER">Handover</option>
              <option value="CRITICAL">Critical</option>
            </select>
          </div>
        </div>
      )}

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14, marginBottom: 24 }}>
        <div style={{ padding: 18, borderRadius: 12, background: '#fff', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Total Events</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{history.length}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#fff', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Last 24 Hours</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
            {history.filter(e => {
              const diffHours = Math.floor((new Date() - new Date(e.timestamp)) / (1000 * 60 * 60));
              return diffHours < 24;
            }).length}
          </div>
        </div>
      </div>

      {/* Loading/Error States */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
          <FaChartLine style={{ fontSize: 32, marginBottom: 12, opacity: 0.5 }} />
          <p>Loading history...</p>
        </div>
      )}

      {error && !loading && (
        <div style={{ padding: 16, background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 12, color: '#991b1b', marginBottom: 20 }}>
          <p style={{ margin: 0 }}>⚠️ {error} - Showing sample data</p>
        </div>
      )}

      {/* Timeline */}
      {!loading && filteredHistory.length > 0 ? (
        <div style={{ position: 'relative' }}>
          {/* Timeline Line */}
          <div style={{
            position: 'absolute',
            left: 31,
            top: 0,
            bottom: 0,
            width: 2,
            background: '#e5e7eb'
          }} />

          {filteredHistory.map((item, idx) => {
            const badgeStyle = statusBadge(item.status);
            return (
              <div 
                key={item.id}
                style={{
                  display: 'flex',
                  gap: 16,
                  marginBottom: 20,
                  position: 'relative'
                }}
              >
                {/* Timeline Dot */}
                <div style={{
                  width: 64,
                  display: 'flex',
                  justifyContent: 'center',
                  paddingTop: 4
                }}>
                  <div style={{
                    width: 14,
                    height: 14,
                    borderRadius: '50%',
                    background: badgeStyle.color,
                    border: '3px solid #fff',
                    boxShadow: '0 0 0 1px #e5e7eb'
                  }} />
                </div>

                {/* Event Card */}
                <div 
                  onClick={() => navigate(`/summary/${item.encounterId}`)}
                  style={{
                    flex: 1,
                    padding: 16,
                    background: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: 12,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.08)';
                    e.currentTarget.style.borderColor = '#cbd5e1';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.borderColor = '#e5e7eb';
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                      <span 
                        style={{
                          padding: '4px 10px',
                          borderRadius: 6,
                          fontSize: 12,
                          fontWeight: 600,
                          background: badgeStyle.bg,
                          color: badgeStyle.color
                        }}
                      >
                        {badgeStyle.label}
                      </span>
                      <span style={{ fontSize: 12, color: '#64748b' }}>
                        {formatTimestamp(item.timestamp)}
                      </span>
                    </div>

                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, color: '#111827' }}>
                        {item.patientName}
                      </div>
                      {item.notes && (
                        <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>
                          {item.notes}
                        </div>
                      )}
                    </div>

                    <div style={{ fontSize: 12, color: '#64748b' }}>
                      By <strong>{item.updatedBy}</strong>
                      {item.handoverTo && ` → ${item.handoverTo}`}
                    </div>
                  </div>

                  <FiChevronRight style={{ fontSize: 18, color: '#cbd5e1', marginTop: 4 }} />
                </div>
              </div>
            );
          })}
        </div>
      ) : !loading && (
        <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
          <FaHistory style={{ fontSize: 32, marginBottom: 12, opacity: 0.5 }} />
          <p>No history events found</p>
        </div>
      )}
    </div>
  );
}
