import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaHistory, FaChevronRight, FaFilter, FaChartLine } from 'react-icons/fa';

const TriagedPatients = () => {
  const navigate = useNavigate();
  const [allTriagedPatients, setAllTriagedPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ 
    statusFilter: 'all',
    showFilters: false,
    selectedTriageDate: null
  });
  const [showCalendarModal, setShowCalendarModal] = useState(false);
  const [calendarMonth, setCalendarMonth] = useState(new Date());

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
  const token = localStorage.getItem('token');

  // Fetch all triaged patients
  useEffect(() => {
    const fetchTriagedPatients = async () => {
      try {
        setLoading(true);
        const fetchUrl = `${BACKEND_URL}/api/encounters?status=AWAITING_PHYSICIAN,ACKNOWLEDGED,HOSPITALIZED,DISCHARGED&limit=500&bypassNurseFilter=true`;
        
        const res = await fetch(fetchUrl, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });

        if (!res.ok) throw new Error('Failed to fetch triaged patients');
        
        const data = await res.json();
        const sorted = (data.encounters || []).sort((a, b) => {
          const dateA = new Date(a.createdAt || 0).getTime();
          const dateB = new Date(b.createdAt || 0).getTime();
          return dateB - dateA;
        });
        
        setAllTriagedPatients(sorted);
      } catch (err) {
        console.error('[TRIAGED PATIENTS] Error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (token) fetchTriagedPatients();
  }, [token, BACKEND_URL]);

  // Status badge styling
  const getStatusStyle = (status) => {
    const statusMap = {
      'AWAITING_PHYSICIAN': { bg: '#fef3c7', color: '#b45309', label: '⏳ Awaiting Physician', icon: '📋' },
      'ACKNOWLEDGED': { bg: '#dcfce7', color: '#166534', label: '✓ Acknowledged', icon: '✓' },
      'HOSPITALIZED': { bg: '#eff6ff', color: '#0c4a6e', label: '🛏️ Hospitalized', icon: '🏥' },
      'DISCHARGED': { bg: '#e2e8f0', color: '#1e293b', label: '🚪 Discharged', icon: '👋' },
      'COMPLETED': { bg: '#d1fae5', color: '#065f46', label: '✅ Completed', icon: '✓' }
    };
    return statusMap[status] || statusMap['AWAITING_PHYSICIAN'];
  };

  // Format time relative to now
  const formatTimeAgo = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  // Format full date and time
  const formatFullDateTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Filter patients by status and optionally by date
  let filteredPatients = filters.statusFilter === 'all' 
    ? allTriagedPatients 
    : allTriagedPatients.filter(p => p.status === filters.statusFilter);

  // Filter by selected triage date if set
  if (filters.selectedTriageDate) {
    filteredPatients = filteredPatients.filter(p => {
      const triageDate = new Date(p.createdAt).toISOString().split('T')[0];
      return triageDate === filters.selectedTriageDate;
    });
  }

  // Generate calendar data
  const monthStart = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), 1);
  const firstDayOfWeek = (monthStart.getDay() + 6) % 7;
  const daysInMonth = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 0).getDate();
  
  const calendarCells = Array.from({ length: firstDayOfWeek + daysInMonth }, (_, index) => {
    const dayNumber = index - firstDayOfWeek + 1;
    if (dayNumber < 1 || dayNumber > daysInMonth) return null;

    const date = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), dayNumber);
    const key = date.toISOString().split('T')[0];
    
    // Count triages on this date
    const triagesOnDate = allTriagedPatients.filter(p => {
      const triageDate = new Date(p.createdAt).toISOString().split('T')[0];
      return triageDate === key;
    });
    
    return {
      key,
      date,
      hasEvents: triagesOnDate.length > 0,
      isSelected: filters.selectedTriageDate === key,
      eventCount: triagesOnDate.length
    };
  }).filter(Boolean);

  // Calculate stats
  const stats = {
    total: allTriagedPatients.length,
    awaiting: allTriagedPatients.filter(p => p.status === 'AWAITING_PHYSICIAN').length,
    hospitalized: allTriagedPatients.filter(p => p.status === 'HOSPITALIZED').length,
    discharged: allTriagedPatients.filter(p => p.status === 'DISCHARGED').length
  };

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto', fontFamily: 'system-ui, -apple-system, sans-serif', background: '#f9fafb', minHeight: '100vh' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 32 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <FaHistory style={{ fontSize: 24, color: '#7c3aed' }} />
            <p style={{ margin: 0, fontSize: 12, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.14em', fontWeight: 600 }}>Nurse Triage History</p>
          </div>
          <h1 style={{ margin: '12px 0 6px', fontSize: 32, fontWeight: 700, color: '#111827' }}>All Triaged Patients</h1>
          <p style={{ margin: 0, color: '#475569', lineHeight: 1.7, fontSize: 15 }}>
            Complete timeline of all patients triaged by nursing staff. Track progression from triage through hospitalization and discharge.
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
              fontWeight: 600, 
              cursor: 'pointer',
              fontSize: 14,
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = '#f1f5f9'}
            onMouseLeave={(e) => e.currentTarget.style.background = '#fff'}
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
              fontSize: 14,
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = '#f1f5f9'}
            onMouseLeave={(e) => e.currentTarget.style.background = '#fff'}
          >
            <FaFilter size={14} /> Status Filter
          </button>
          <button 
            onClick={() => setShowCalendarModal(true)} 
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
              fontSize: 14,
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = '#f1f5f9'}
            onMouseLeave={(e) => e.currentTarget.style.background = '#fff'}
          >
            📅 Browse by Date
          </button>
        </div>
      </div>

      {/* Filters */}
      {filters.showFilters && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 24, padding: 16, background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb' }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#334155', display: 'block', marginBottom: 6 }}>Filter by Status</label>
            <select 
              value={filters.statusFilter}
              onChange={(e) => setFilters(prev => ({ ...prev, statusFilter: e.target.value }))}
              style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 14, cursor: 'pointer', minWidth: 180 }}
            >
              <option value="all">All Statuses</option>
              <option value="AWAITING_PHYSICIAN">Awaiting Physician</option>
              <option value="ACKNOWLEDGED">Acknowledged</option>
              <option value="HOSPITALIZED">Hospitalized</option>
              <option value="DISCHARGED">Discharged</option>
              <option value="COMPLETED">Completed</option>
            </select>
          </div>
        </div>
      )}

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14, marginBottom: 28 }}>
        <div style={{ padding: 18, borderRadius: 12, background: '#fff', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Total Triaged</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: '#7c3aed' }}>{stats.total}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#fff', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Awaiting Physician</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: '#b45309' }}>{stats.awaiting}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#fff', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Hospitalized</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: '#0c4a6e' }}>{stats.hospitalized}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#fff', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Discharged</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: '#1e293b' }}>{stats.discharged}</div>
        </div>
      </div>

      {/* Loading/Error States */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 60, color: '#64748b' }}>
          <FaChartLine style={{ fontSize: 40, marginBottom: 16, opacity: 0.5 }} />
          <p style={{ fontSize: 16 }}>Loading triaged patients...</p>
        </div>
      )}

      {error && !loading && (
        <div style={{ padding: 16, background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 12, color: '#991b1b', marginBottom: 20 }}>
          <p style={{ margin: 0 }}>⚠️ {error}</p>
        </div>
      )}

      {/* Timeline */}
      {!loading && filteredPatients.length > 0 ? (
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

          {filteredPatients.map((patient, idx) => {
            const statusStyle = getStatusStyle(patient.status);
            const patientName = patient.patient?.name || patient.patientName || 'Unknown Patient';
            const triageTime = formatTimeAgo(patient.createdAt);
            const fullDateTime = formatFullDateTime(patient.createdAt);
            
            return (
              <div 
                key={patient._id}
                style={{
                  display: 'flex',
                  gap: 16,
                  marginBottom: 24,
                  position: 'relative'
                }}
              >
                {/* Timeline Dot */}
                <div style={{
                  width: 64,
                  display: 'flex',
                  justifyContent: 'center',
                  paddingTop: 4,
                  flexShrink: 0
                }}>
                  <div style={{
                    width: 16,
                    height: 16,
                    borderRadius: '50%',
                    background: statusStyle.color,
                    border: '3px solid #fff',
                    boxShadow: '0 0 0 2px #e5e7eb'
                  }} />
                </div>

                {/* Patient Card */}
                <div 
                  onClick={() => navigate(`/summary/${patient._id}`)}
                  style={{
                    flex: 1,
                    padding: 18,
                    background: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: 12,
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow = '0 8px 16px rgba(0,0,0,0.1)';
                    e.currentTarget.style.borderColor = '#cbd5e1';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.borderColor = '#e5e7eb';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  <div style={{ flex: 1 }}>
                    {/* Status Badge + Time */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                      <span 
                        style={{
                          padding: '5px 12px',
                          borderRadius: 6,
                          fontSize: 12,
                          fontWeight: 600,
                          background: statusStyle.bg,
                          color: statusStyle.color
                        }}
                      >
                        {statusStyle.label}
                      </span>
                      <span style={{ fontSize: 12, color: '#64748b', fontWeight: 500 }}>
                        {triageTime}
                      </span>
                    </div>

                    {/* Patient Name and ID */}
                    <div style={{ marginBottom: 10 }}>
                      <div style={{ fontSize: 15, fontWeight: 700, color: '#111827', marginBottom: 3 }}>
                        {patientName}
                      </div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>
                        ID: <strong>{patient.patientId || patient._id.slice(-8).toUpperCase()}</strong>
                      </div>
                    </div>

                    {/* Triage Details */}
                    <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.6 }}>
                      <div>📅 Triaged: <strong>{fullDateTime}</strong></div>
                      {patient.diagnosis && (
                        <div style={{ marginTop: 4 }}>🩺 Diagnosis: <strong>{patient.diagnosis}</strong></div>
                      )}
                      {patient.treatments && patient.treatments.length > 0 && (
                        <div style={{ marginTop: 4 }}>💊 Treatments: <strong>{patient.treatments.length}</strong></div>
                      )}
                    </div>
                  </div>

                  <FaChevronRight style={{ fontSize: 18, color: '#cbd5e1', marginTop: 4, flexShrink: 0 }} />
                </div>
              </div>
            );
          })}
        </div>
      ) : !loading && (
        <div style={{ textAlign: 'center', padding: 60, color: '#64748b' }}>
          <FaHistory style={{ fontSize: 40, marginBottom: 16, opacity: 0.5 }} />
          <p style={{ fontSize: 16 }}>No triaged patients found</p>
          <p style={{ fontSize: 13, marginTop: 8 }}>Patients triaged by nursing staff will appear here</p>
        </div>
      )}

      {/* Calendar Modal */}
      {showCalendarModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.55)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
          <div style={{ width: 'min(100%, 500px)', background: 'white', borderRadius: 12, boxShadow: '0 16px 40px rgba(15, 23, 42, 0.2)', padding: 20 }}>
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>Browse Triage History by Date</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Select a date to view all patients triaged on that day</div>
              </div>
              <button 
                onClick={() => setShowCalendarModal(false)} 
                style={{ border: 'none', background: 'transparent', fontSize: 24, cursor: 'pointer', color: '#475569', padding: 0 }}
              >
                ×
              </button>
            </div>

            {/* Selected Date Display */}
            {filters.selectedTriageDate && (
              <div style={{ padding: 10, background: '#f0f9ff', border: '1px solid #0ea5e9', borderRadius: 8, marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#0369a1' }}>
                  📅 Showing {filteredPatients.length} patient{filteredPatients.length !== 1 ? 's' : ''} triaged on {new Date(filters.selectedTriageDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', weekday: 'long' })}
                </div>
              </div>
            )}

            {/* Calendar */}
            <div>
              {/* Month Navigation */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <button 
                  onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() - 1, 1))} 
                  style={{ border: '1px solid #e2e8f0', background: 'white', borderRadius: 6, padding: '6px 10px', cursor: 'pointer', fontWeight: 600 }}
                >
                  ←
                </button>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>
                  {calendarMonth.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                </div>
                <button 
                  onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))} 
                  style={{ border: '1px solid #e2e8f0', background: 'white', borderRadius: 6, padding: '6px 10px', cursor: 'pointer', fontWeight: 600 }}
                >
                  →
                </button>
              </div>

              {/* Day Headers */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 6, fontSize: 11, color: '#64748b', marginBottom: 8, fontWeight: 600 }}>
                {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((dayName) => (
                  <div key={dayName} style={{ textAlign: 'center' }}>{dayName}</div>
                ))}
              </div>

              {/* Calendar Days */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 6, marginBottom: 16 }}>
                {calendarCells.map((cell) => (
                  <button
                    key={cell.key}
                    disabled={!cell.hasEvents}
                    onClick={() => {
                      setFilters(prev => ({ ...prev, selectedTriageDate: cell.key }));
                    }}
                    style={{
                      minHeight: 40,
                      borderRadius: 6,
                      border: cell.isSelected ? '2px solid #7c3aed' : '1px solid #e2e8f0',
                      background: cell.isSelected ? '#f3e8ff' : (cell.hasEvents ? 'white' : '#f8fafc'),
                      color: cell.hasEvents ? '#0f172a' : '#cbd5e1',
                      cursor: cell.hasEvents ? 'pointer' : 'default',
                      fontSize: 12,
                      fontWeight: 700,
                      padding: '4px 2px',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      if (cell.hasEvents) {
                        e.currentTarget.style.background = '#f3e8ff';
                        e.currentTarget.style.borderColor = '#7c3aed';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!cell.isSelected && cell.hasEvents) {
                        e.currentTarget.style.background = 'white';
                        e.currentTarget.style.borderColor = '#e2e8f0';
                      }
                    }}
                  >
                    <div>{cell.date.getDate()}</div>
                    {cell.hasEvents && <div style={{ fontSize: 10, color: '#7c3aed', fontWeight: 700 }}>{cell.eventCount}</div>}
                  </button>
                ))}
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <button
                  onClick={() => {
                    setFilters(prev => ({ ...prev, selectedTriageDate: null }));
                  }}
                  style={{
                    padding: '9px 14px',
                    borderRadius: 8,
                    border: '1px solid #cbd5e1',
                    background: '#fff',
                    color: '#334155',
                    cursor: 'pointer',
                    fontWeight: 600,
                    fontSize: 13
                  }}
                >
                  Clear Date
                </button>
                <button
                  onClick={() => setShowCalendarModal(false)}
                  style={{
                    padding: '9px 14px',
                    borderRadius: 8,
                    border: 'none',
                    background: '#7c3aed',
                    color: 'white',
                    cursor: 'pointer',
                    fontWeight: 600,
                    fontSize: 13
                  }}
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TriagedPatients;
