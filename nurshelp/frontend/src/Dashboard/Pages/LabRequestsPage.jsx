import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiFilter, FiPlus, FiChevronRight } from 'react-icons/fi';
import { FaFlask, FaChartLine } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

export default function LabRequestsPage() {
  const navigate = useNavigate();
  const [labRequests, setLabRequests] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ 
    specialty: 'all',
    status: 'all',
    priority: 'all',
    showFilters: false 
  });

  useEffect(() => {
    let mounted = true;
    loadLabRequests();

    // Poll for live updates every 10 seconds
    const iv = setInterval(() => {
      if (mounted) loadLabRequests();
    }, 10000);

    return () => { mounted = false; clearInterval(iv); };
  }, []);

  const loadLabRequests = async () => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      // Fetch all encounters and extract lab requests
      // Use physician lab requests aggregation endpoint for live data
      const res = await fetch(`${BACKEND_URL}/api/lab-requests/physician/all?t=${Date.now()}`, { headers });
      if (!res.ok) throw new Error('Failed to fetch lab requests');
      const data = await res.json();

      // Endpoint returns flattened lab_requests array
      const list = data.lab_requests || data.requests || data.encounters || [];
      setLabRequests(list);
      setLoading(false);
    } catch (err) {
      console.error('Error loading lab requests:', err);
      setError(err.message);
      
      // Sample data for demonstration
      const now = new Date();
      const sampleRequests = [
        {
          id: 'lab-1',
          patientName: 'John Doe',
          patientId: 'p1',
          specialty: 'RADIOLOGY',
          lab_type: 'Chest X-ray',
          description: 'Suspected pneumonia - routine chest imaging',
          priority: 'URGENT',
          status: 'IN_PROGRESS',
          requestedAt: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(),
          completedAt: null,
          requestedBy: 'Dr. Smith'
        },
        {
          id: 'lab-2',
          patientName: 'Jane Smith',
          patientId: 'p2',
          specialty: 'PATHOLOGY',
          lab_type: 'Complete Blood Count',
          description: 'Pre-operative labs required',
          priority: 'HIGH',
          status: 'ACKNOWLEDGED',
          requestedAt: new Date(now.getTime() - 5 * 60 * 60 * 1000).toISOString(),
          completedAt: null,
          requestedBy: 'Dr. Johnson'
        },
        {
          id: 'lab-3',
          patientName: 'Mike Brown',
          patientId: 'p3',
          specialty: 'RADIOLOGY',
          lab_type: 'CT Scan - Abdomen',
          description: 'Abdominal pain investigation',
          priority: 'MEDIUM',
          status: 'COMPLETED',
          requestedAt: new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString(),
          completedAt: new Date(now.getTime() - 12 * 60 * 60 * 1000).toISOString(),
          requestedBy: 'Dr. Williams'
        },
        {
          id: 'lab-4',
          patientName: 'Sarah Lee',
          patientId: 'p4',
          specialty: 'PATHOLOGY',
          lab_type: 'Blood Culture',
          description: 'Fever workup - possible sepsis',
          priority: 'URGENT',
          status: 'PENDING',
          requestedAt: new Date(now.getTime() - 30 * 60 * 1000).toISOString(),
          completedAt: null,
          requestedBy: 'Dr. Adams'
        }
      ];
      setLabRequests(sampleRequests);
      setLoading(false);
    }
  };

  const statusBadge = (status) => {
    const statusStyles = {
      'PENDING': { bg: '#fef3c7', color: '#b45309', label: 'Pending' },
      'ACKNOWLEDGED': { bg: '#dbeafe', color: '#0c4a6e', label: 'Acknowledged' },
      'IN_PROGRESS': { bg: '#eff6ff', color: '#2563eb', label: 'In Progress' },
      'COMPLETED': { bg: '#dcfce7', color: '#166534', label: 'Completed' },
      'REVIEWED': { bg: '#e2e8f0', color: '#1f2937', label: 'Reviewed' },
      'CANCELLED': { bg: '#fee2e2', color: '#991b1b', label: 'Cancelled' }
    };
    return statusStyles[status] || statusStyles['PENDING'];
  };

  const priorityBadge = (priority) => {
    const priorityStyles = {
      'LOW': { bg: '#e2e8f0', color: '#1f2937' },
      'MEDIUM': { bg: '#fef3c7', color: '#b45309' },
      'HIGH': { bg: '#fed7aa', color: '#92400e' },
      'URGENT': { bg: '#fee2e2', color: '#991b1b' }
    };
    return priorityStyles[priority] || priorityStyles['MEDIUM'];
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatTimeDiff = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffHours = Math.floor((now - date) / (1000 * 60 * 60));
    
    if (diffHours < 1) return 'Just now';
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  };

  const filteredRequests = labRequests.filter(req => {
    if (filters.specialty !== 'all' && req.specialty !== filters.specialty) return false;
    if (filters.status !== 'all' && req.status !== filters.status) return false;
    if (filters.priority !== 'all' && req.priority !== filters.priority) return false;
    return true;
  });

  const stats = {
    total: labRequests.length,
    pending: labRequests.filter(r => r.status === 'PENDING').length,
    inProgress: labRequests.filter(r => r.status === 'IN_PROGRESS').length,
    completed: labRequests.filter(r => r.status === 'COMPLETED').length,
    urgent: labRequests.filter(r => r.priority === 'URGENT').length
  };

  return (
    <div style={{ padding: 24, maxWidth: 1320, margin: '0 auto', fontFamily: 'system-ui, -apple-system, sans-serif', minHeight: '100vh', background: '#f8fafc' }}>
      {/* Header */}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 28, background: '#fff', padding: 24, borderRadius: 14, border: '1px solid #e5e7eb' }}>
        <div style={{ minWidth: 280 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <FaFlask style={{ fontSize: 24, color: '#b45309' }} />
            <p style={{ margin: 0, fontSize: 12, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Lab Management</p>
          </div>
          <h1 style={{ margin: '12px 0 6px', fontSize: 32, color: '#111827' }}>Lab Requests</h1>
          <p style={{ margin: 0, color: '#475569', lineHeight: 1.7, maxWidth: 680 }}>
            View and manage all radiology and pathology lab requests. Track status, priority, and completion.
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
            onClick={loadLabRequests} 
            style={{ 
              padding: '12px 18px', 
              borderRadius: 14, 
              border: '1px solid #10b981', 
              background: '#ecfdf5', 
              color: '#059669', 
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
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 22, padding: 16, background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb' }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#334155', display: 'block', marginBottom: 6 }}>Specialty</label>
            <select 
              value={filters.specialty}
              onChange={(e) => setFilters(prev => ({ ...prev, specialty: e.target.value }))}
              style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 14, cursor: 'pointer' }}
            >
              <option value="all">All Specialties</option>
              <option value="RADIOLOGY">Radiology</option>
              <option value="PATHOLOGY">Pathology</option>
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
              <option value="PENDING">Pending</option>
              <option value="ACKNOWLEDGED">Acknowledged</option>
              <option value="IN_PROGRESS">In Progress</option>
              <option value="COMPLETED">Completed</option>
              <option value="REVIEWED">Reviewed</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#334155', display: 'block', marginBottom: 6 }}>Priority</label>
            <select 
              value={filters.priority}
              onChange={(e) => setFilters(prev => ({ ...prev, priority: e.target.value }))}
              style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 14, cursor: 'pointer' }}
            >
              <option value="all">All Priorities</option>
              <option value="URGENT">Urgent</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
            </select>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 14, marginBottom: 24 }}>
        <div style={{ padding: 18, borderRadius: 12, background: '#fff', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Total Requests</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{stats.total}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#fef3c7', border: '1px solid #fcd34d' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#b45309', marginBottom: 8 }}>Pending</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#b45309' }}>{stats.pending}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#eff6ff', border: '1px solid #bfdbfe' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#0c4a6e', marginBottom: 8 }}>In Progress</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#0c4a6e' }}>{stats.inProgress}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#dcfce7', border: '1px solid #bbf7d0' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#166534', marginBottom: 8 }}>Completed</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#166534' }}>{stats.completed}</div>
        </div>
        <div style={{ padding: 18, borderRadius: 12, background: '#fee2e2', border: '1px solid #fca5a5' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#991b1b', marginBottom: 8 }}>Urgent</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#991b1b' }}>{stats.urgent}</div>
        </div>
      </div>

      {/* Loading/Error States */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 40, color: '#64748b', background: '#fff', borderRadius: 12 }}>
          <FaChartLine style={{ fontSize: 32, marginBottom: 12, opacity: 0.5 }} />
          <p>Loading lab requests...</p>
        </div>
      )}

      {error && !loading && (
        <div style={{ padding: 16, background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 12, color: '#991b1b', marginBottom: 20 }}>
          <p style={{ margin: 0 }}>⚠️ {error} - Showing sample data</p>
        </div>
      )}

      {/* Table */}
      {!loading && filteredRequests.length > 0 ? (
        <div style={{ overflowX: 'auto', background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
            <thead style={{ background: '#f8fafc', borderBottom: '1px solid #e5e7eb' }}>
              <tr>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>Patient</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>Test Type</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>Description</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 600, color: '#475569', fontSize: 12 }}>Status</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 600, color: '#475569', fontSize: 12 }}>Priority</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>Requested</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 600, color: '#475569', fontSize: 12 }}></th>
              </tr>
            </thead>
            <tbody>
              {filteredRequests.map((req, idx) => {
                const statusStyle = statusBadge(req.status);
                const priorityStyle = priorityBadge(req.priority);
                return (
                  <tr 
                    key={req.id}
                    style={{ 
                      borderBottom: '1px solid #e5e7eb',
                      ':hover': { background: '#f8fafc' },
                      transition: 'background 0.2s ease'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = '#f8fafc'}
                    onMouseLeave={(e) => e.currentTarget.style.background = '#fff'}
                  >
                    <td style={{ padding: '12px 16px', color: '#111827', fontWeight: 500 }}>
                      <div>{req.patientName}</div>
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{req.requestedBy}</div>
                    </td>
                    <td style={{ padding: '12px 16px', color: '#111827' }}>
                      <div style={{ fontWeight: 500 }}>{req.lab_type || 'N/A'}</div>
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{req.specialty}</div>
                    </td>
                    <td style={{ padding: '12px 16px', color: '#475569', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {req.description}
                    </td>
                    <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                      <span 
                        style={{
                          padding: '6px 10px',
                          borderRadius: 6,
                          fontSize: 12,
                          fontWeight: 600,
                          background: statusStyle.bg,
                          color: statusStyle.color,
                          display: 'inline-block'
                        }}
                      >
                        {statusStyle.label}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                      <span 
                        style={{
                          padding: '6px 10px',
                          borderRadius: 6,
                          fontSize: 12,
                          fontWeight: 600,
                          background: priorityStyle.bg,
                          color: priorityStyle.color,
                          display: 'inline-block'
                        }}
                      >
                        {req.priority}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', color: '#64748b', fontSize: 12 }}>
                      <div>{formatTime(req.requestedAt)}</div>
                      <div style={{ marginTop: 2, color: '#94a3b8' }}>{formatTimeDiff(req.requestedAt)}</div>
                    </td>
                    <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                      <button 
                        onClick={() => navigate(`/summary/${req.encounterId}`)}
                        style={{
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          color: '#94a3b8',
                          padding: 0,
                          fontSize: 16
                        }}
                      >
                        <FiChevronRight />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : !loading && (
        <div style={{ textAlign: 'center', padding: 40, color: '#64748b', background: '#fff', borderRadius: 12 }}>
          <FaFlask style={{ fontSize: 32, marginBottom: 12, opacity: 0.5 }} />
          <p>No lab requests found</p>
        </div>
      )}
    </div>
  );
}
