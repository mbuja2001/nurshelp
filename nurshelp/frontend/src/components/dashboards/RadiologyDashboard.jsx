import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaCalendarAlt, FaCheckCircle, FaFileAlt, FaFlask, FaHistory, FaClock } from 'react-icons/fa';
import { decodeJwt } from '../../utils/jwt';
import SidebarLayout from '../../Dashboard/SidebarLayout';
import EhrContextPanel from '../diagnostics/EhrContextPanel';
import './RadiologyDashboard.css';

/**
 * The dashboard reads and updates the hospital-scoped radiology request queue.
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const RadiologyDashboard = ({ config }) => {
  const navigate = useNavigate();
  const userInfo = decodeJwt(localStorage.getItem('token'));

  // ============ STATE MANAGEMENT ============
  const [activeView, setActiveView] = useState('hub');
  const [activeTab, setActiveTab] = useState('pending'); // pending, in-progress, completed, submitted
  const [labQueue, setLabQueue] = useState([]);
  const [selectedLabRequest, setSelectedLabRequest] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [actionError, setActionError] = useState('');
  const [actionSuccess, setActionSuccess] = useState('');
  const [savingAction, setSavingAction] = useState(false);
  const [filterPriority, setFilterPriority] = useState('ALL'); // ALL, URGENT, HIGH, MEDIUM, LOW
  const [hubFilter, setHubFilter] = useState('ALL');
  const [hubSearch, setHubSearch] = useState('');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportForm, setReportForm] = useState({
    report_content: '',
    clinical_impression: '',
    recommendations: [],
    report_images: []
  });
  const [newRecommendation, setNewRecommendation] = useState('');

  // ============ FETCH LAB REQUESTS ============
  const fetchLabRequests = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    setLoadError('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${BACKEND_URL}/api/lab-requests/queue/requests?specialty=RADIOLOGY`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.message || `Unable to load radiology requests (${response.status})`);
      }

      const data = await response.json();
      setLabQueue(data.lab_requests || []);
      console.log(`[RADIOLOGY] Fetched ${data.lab_requests?.length || 0} lab requests`);
    } catch (err) {
      console.error('[RADIOLOGY] Fetch error:', err);
      setLoadError(err.message || 'Unable to load radiology requests');
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  useEffect(() => {
    console.log('[RADIOLOGY] Dashboard loaded');
    fetchLabRequests();
    const interval = setInterval(() => fetchLabRequests(false), 15000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => setCurrentTime(new Date()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  // ============ TAB SWITCHING ============
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setSelectedLabRequest(null);
    setActiveView('worklist');
  };

  const scrollToWorklist = (tab) => {
    setActiveTab(tab);
    setSelectedLabRequest(null);
    setActiveView('worklist');
    window.setTimeout(() => {
      document.getElementById('specialist-worklist')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 0);
  };

  const openWorklistRequest = (labRequest, tab) => {
    setActiveTab(tab);
    setFilterPriority('ALL');
    setSelectedLabRequest(labRequest);
    setActiveView('worklist');
    window.setTimeout(() => {
      document.getElementById('specialist-worklist')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 0);
  };

  // ============ ACCEPT LAB REQUEST ============
  const handleAcceptLabRequest = async (labRequest) => {
    if (!labRequest || !labRequest.encounterId || !labRequest.request_id) {
      setActionError('This request is missing its encounter or request identifier.');
      return;
    }

    setSavingAction(true);
    setActionError('');
    setActionSuccess('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${BACKEND_URL}/api/lab-requests/${labRequest.encounterId}/${labRequest.request_id}/acknowledge`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            assigned_technician: userInfo?.name || 'Radiologist'
          })
        }
      );

      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to accept request (${response.status})`);

      setActiveTab('in-progress');
      setSelectedLabRequest({ ...labRequest, status: 'ACKNOWLEDGED', assigned_technician: userInfo?.name || 'Radiologist' });
      setActionSuccess('Request accepted and recorded.');
      await fetchLabRequests();
    } catch (err) {
      console.error('[RADIOLOGY] Accept error:', err);
      setActionError(err.message || 'Unable to accept request');
    } finally {
      setSavingAction(false);
    }
  };

  const handleStartRequest = async (labRequest) => {
    if (!labRequest?.encounterId || !labRequest?.request_id) {
      setActionError('This request is missing its encounter or request identifier.');
      return;
    }

    setSavingAction(true);
    setActionError('');
    setActionSuccess('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${BACKEND_URL}/api/lab-requests/${labRequest.encounterId}/${labRequest.request_id}/start`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to start request (${response.status})`);

      setSelectedLabRequest({
        ...labRequest,
        status: data.status,
        started_at: data.started_at,
        started_by: data.started_by
      });
      setActionSuccess('Radiologist work started and recorded.');
      await fetchLabRequests();
    } catch (err) {
      console.error('[RADIOLOGY] Start request error:', err);
      setActionError(err.message || 'Unable to start request');
    } finally {
      setSavingAction(false);
    }
  };

  // ============ SUBMIT LAB REPORT ============
  const handleSubmitReport = async () => {
    if (!selectedLabRequest || !reportForm.report_content.trim() || !reportForm.clinical_impression.trim()) {
      setActionError('Enter both findings and a clinical impression before submitting.');
      return;
    }

    setSavingAction(true);
    setActionError('');
    setActionSuccess('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `${BACKEND_URL}/api/lab-requests/${selectedLabRequest.encounterId}/${selectedLabRequest.request_id}/submit-report`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            report_content: reportForm.report_content,
            clinical_impression: reportForm.clinical_impression,
            recommendations: reportForm.recommendations
          })
        }
      );
      const data = await response.json().catch(() => ({}));

      if (!response.ok) throw new Error(data.message || `Unable to submit report (${response.status})`);

      setShowReportModal(false);
      setReportForm({ report_content: '', clinical_impression: '', recommendations: [], report_images: [] });
      setSelectedLabRequest(null);
      setActiveTab('completed');
      setActionSuccess('Report saved to the encounter and made available to the ordering physician.');
      await fetchLabRequests();
    } catch (err) {
      console.error('[RADIOLOGY] Submit report error:', err);
      setActionError(err.message || 'Unable to submit report');
    } finally {
      setSavingAction(false);
    }
  };

  // ============ ADD RECOMMENDATION ============
  const handleAddRecommendation = () => {
    if (newRecommendation.trim()) {
      setReportForm(prev => ({
        ...prev,
        recommendations: [...prev.recommendations, newRecommendation.trim()]
      }));
      setNewRecommendation('');
    }
  };

  // ============ REMOVE RECOMMENDATION ============
  const handleRemoveRecommendation = (idx) => {
    setReportForm(prev => ({
      ...prev,
      recommendations: prev.recommendations.filter((_, i) => i !== idx)
    }));
  };

  // ============ FILTER QUEUE BY PRIORITY ============
  const tabStatuses = {
    pending: ['PENDING'],
    'in-progress': ['ACKNOWLEDGED', 'IN_PROGRESS'],
    completed: ['COMPLETED'],
    submitted: ['REVIEWED']
  };
  const filteredQueue = labQueue.filter(req =>
    tabStatuses[activeTab].includes(req.status) &&
    (filterPriority === 'ALL' || req.priority === filterPriority)
  );
  const searchTerm = hubSearch.trim().toLowerCase();
  const filteredHubQueue = labQueue.filter((request) => {
    const type = (request.lab_type || '').toLowerCase();
    const matchesModality = hubFilter === 'ALL'
      || (hubFilter === 'CT' && (type.includes('ct') || type.includes('computed tomography')))
      || (hubFilter === 'MRI' && type.includes('mri'))
      || (hubFilter === 'X-ray' && (type.includes('x-ray') || type.includes('xray') || type.includes('radiograph')))
      || (hubFilter === 'Ultrasound' && (type.includes('ultrasound') || type.includes('sonograph')));
    const searchable = [
      request.patientName,
      request.patientId,
      request.request_id,
      request.requested_by,
      request.lab_type
    ].filter(Boolean).join(' ').toLowerCase();
    return matchesModality && (!searchTerm || searchable.includes(searchTerm));
  });
  const requestAge = (requestedAt) => {
    const timestamp = new Date(requestedAt).getTime();
    if (!requestedAt || Number.isNaN(timestamp)) return 'Time unavailable';
    const minutes = Math.max(0, Math.floor((currentTime.getTime() - timestamp) / 60000));
    if (minutes < 60) return `Waiting ${minutes} min`;
    return `Waiting ${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  };
  const stageRequests = {
    new: filteredHubQueue.filter((request) => request.status === 'PENDING'),
    vetted: filteredHubQueue.filter((request) => request.status === 'ACKNOWLEDGED'),
    reporting: filteredHubQueue.filter((request) => request.status === 'IN_PROGRESS'),
    released: filteredHubQueue.filter((request) => request.status === 'COMPLETED'),
    delivered: filteredHubQueue.filter((request) => request.status === 'REVIEWED')
  };
  const hubStages = [
    { key: 'new', title: 'New requests', detail: 'Awaiting radiologist review', requests: stageRequests.new, tab: 'pending' },
    { key: 'vetted', title: 'Vetted', detail: 'Accepted by radiology', requests: stageRequests.vetted, tab: 'in-progress' },
    { key: 'scheduled', title: 'Scheduled', detail: 'Requires PACS / RIS event', requests: [], integrationStage: true },
    { key: 'acquired', title: 'Acquired', detail: 'Requires PACS / RIS event', requests: [], integrationStage: true },
    { key: 'reporting', title: 'Reporting', detail: 'Radiologist work started', requests: stageRequests.reporting, tab: 'in-progress' },
    { key: 'released', title: 'Report released', detail: 'Saved to Nurshelp', requests: stageRequests.released, tab: 'completed' },
    { key: 'delivered', title: 'Delivered', detail: 'Acknowledged by physician', requests: stageRequests.delivered, tab: 'submitted' }
  ];
  const connections = [
    'Order handoff',
    'Report return',
    'Patient details sync',
    'Image viewer launch',
    'Modality worklist',
    'Patient portal'
  ];

  const renderHubRequest = (request, tab) => (
    <button
      className="radiology-hub-request"
      key={request.request_id}
      type="button"
      onClick={() => openWorklistRequest(request, tab)}
    >
      <span className="radiology-hub-request-top">
        <strong>{request.lab_type || 'Imaging request'}</strong>
        <span className={`radiology-priority-badge priority-${(request.priority || 'MEDIUM').toLowerCase()}`}>
          {request.priority || 'Priority unavailable'}
        </span>
      </span>
      <span className="radiology-hub-request-patient">{request.patientName || 'Patient name unavailable'}</span>
      <span className="radiology-hub-request-meta">
        {request.requested_by || 'Ordering clinician unavailable'} · {requestAge(request.requested_at)}
      </span>
    </button>
  );

  // ============ PRIORITY COLOR MAPPING ============
  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'URGENT': return '#ef4444';
      case 'HIGH': return '#f97316';
      case 'MEDIUM': return '#eab308';
      case 'LOW': return '#3b82f6';
      default: return '#64748b';
    }
  };

  // ============ RENDER QUEUE CARD ============
  const renderLabCard = (labRequest, showAcceptButton = true) => {
    return (
      <div
        key={labRequest.request_id}
        className="specialist-worklist-card"
        style={{
          padding: '18px',
          borderRadius: '16px',
          border: '1px solid rgba(255,255,255,0.12)',
          background: 'linear-gradient(135deg, rgba(6,182,212,0.08), rgba(14,165,233,0.08))',
          cursor: 'pointer',
          transition: 'all 0.2s',
          opacity: selectedLabRequest?.request_id === labRequest.request_id ? 1 : 0.8
        }}
        onMouseEnter={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'}
        onMouseLeave={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'}
        onClick={() => setSelectedLabRequest(labRequest)}
      >
        {/* Header: Patient + Priority */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div>
            <h4 style={{ margin: '0 0 4px 0', fontSize: '16px', fontWeight: 700, color: '#eff6ff' }}>
              {labRequest.patientName || 'Unknown Patient'}
            </h4>
            <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>
              Age: {labRequest.patientAge} • ID: {labRequest.patientId}
            </p>
          </div>
          <div
            style={{
              padding: '6px 12px',
              borderRadius: '20px',
              background: getPriorityColor(labRequest.priority),
              fontSize: '11px',
              fontWeight: 700,
              color: '#fff',
              textTransform: 'uppercase'
            }}
          >
            {labRequest.priority}
          </div>
        </div>

        {/* Lab Type */}
        <div style={{ marginBottom: '12px' }}>
          <p style={{ margin: '0 0 4px 0', fontSize: '13px', fontWeight: 600, color: '#06b6d4' }}>
            📋 {labRequest.lab_type}
          </p>
          <p style={{ margin: 0, fontSize: '12px', color: '#cbd5e1', lineHeight: '1.4' }}>
            {labRequest.description || 'No indication provided'}
          </p>
        </div>

        {/* Timestamps + Status */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
          <div>
            <p style={{ margin: '0 0 2px 0', fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
              Requested
            </p>
            <p style={{ margin: 0, fontSize: '12px', color: '#eff6ff' }}>
              {labRequest.requested_at ? new Date(labRequest.requested_at).toLocaleTimeString() : 'N/A'}
            </p>
          </div>
          <div>
            <p style={{ margin: '0 0 2px 0', fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
              Status
            </p>
            <p style={{ margin: 0, fontSize: '12px', color: '#06b6d4', fontWeight: 600 }}>
              {labRequest.status}
            </p>
          </div>
        </div>

        {/* Action Button */}
        {showAcceptButton && activeTab === 'pending' && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleAcceptLabRequest(labRequest);
            }}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '12px',
              border: 'none',
              background: 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
              color: '#fff',
              fontWeight: 700,
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => e.target.style.opacity = '0.9'}
            onMouseLeave={(e) => e.target.style.opacity = '1'}
          >
            ✓ Accept & vet request
          </button>
        )}
        {/* Open Encounter / View Report */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (!labRequest?.encounterId) {
              alert('Encounter not available');
              return;
            }
            // pass the lab request id so the summary page can open the report if implemented
            navigate(`/summary/${labRequest.encounterId}`, { state: { openLabRequestId: labRequest.request_id } });
          }}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '12px',
            border: 'none',
            background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(99,102,241,0.18))',
            color: '#e0e7ff',
            fontWeight: 700,
            fontSize: '13px',
            cursor: 'pointer',
            marginTop: '8px'
          }}
          onMouseEnter={(e) => e.target.style.opacity = '0.95'}
          onMouseLeave={(e) => e.target.style.opacity = '1'}
        >
          📑 Open Encounter / View Report
        </button>
      </div>
    );
  };

  return (
    <SidebarLayout username={userInfo?.name || 'Radiologist'}>
      <div className="specialist-dashboard specialist-dashboard-radiology" style={{
      display: 'flex',
      height: '100%',
      minHeight: 0,
      width: '100%',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      color: '#e2e8f0',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
    }}>
      {/* MAIN CONTENT */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflowY: 'auto', overflowX: 'hidden' }}>
        {/* HEADER */}
        <header className="radiology-dashboard-header" style={{
          padding: '24px',
          border: '1px solid #e5e7eb',
          borderRadius: '12px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
        }}>
          <div className="radiology-header-title">
              <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 800, background: 'linear-gradient(90deg, #06b6d4, #0ea5e9)', backgroundClip: 'text', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                🔬 RADIOLOGY DEPARTMENT
              </h1>
              <p style={{ margin: '8px 0 0 0', fontSize: '14px', color: '#94a3b8' }}>
                {userInfo?.name || 'Radiologist'} • {userInfo?.department || 'Imaging'}
              </p>
          </div>
          <div className="radiology-header-actions">
            <div className="radiology-header-meta">
              <span className="radiology-shift-time"><FaClock /> Shift: {config?.shiftStartTime || '—'} - {config?.shiftEndTime || '—'}</span>
              <span className="radiology-current-time">
                <FaClock />
                <span>{currentTime.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}</span>
                <span className="radiology-time-separator">•</span>
                <span>{currentTime.toLocaleDateString([], { month: 'short', day: 'numeric' })}</span>
              </span>
            </div>
            <nav className="radiology-header-shortcuts" aria-label="Radiology dashboard shortcuts">
              {[
                { icon: <FaCheckCircle />, label: 'Radiology hub', action: () => { setSelectedLabRequest(null); setActiveView('hub'); } },
                { icon: <FaCalendarAlt />, label: 'Calendar Events', action: () => navigate('/calendar') },
                { icon: <FaHistory />, label: 'History Actions', action: () => navigate('/history-actions') },
                { icon: <FaFlask />, label: 'Lab Requests', action: () => navigate('/lab-requests') },
                { icon: <FaFileAlt />, label: 'Reports Submitted', action: () => scrollToWorklist('completed') },
                { icon: <FaCheckCircle />, label: 'Physician Reviewed', action: () => scrollToWorklist('submitted') }
              ].map((item) => (
                <button key={item.label} type="button" onClick={item.action}>
                  <span aria-hidden="true">{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </div>
        </header>

        {actionError && <div className="radiology-action-feedback is-error" role="alert">{actionError}</div>}
        {actionSuccess && <div className="radiology-action-feedback is-success" role="status">{actionSuccess}</div>}

        {activeView === 'hub' && (
          <section className="radiology-hub" aria-label="Radiology hub">
            <div className="radiology-hub-toolbar">
              <div>
                <p className="radiology-hub-eyebrow">Imaging workflow</p>
                <h2>Radiology request pipeline</h2>
              </div>
              <label className="radiology-hub-search">
                <span className="sr-only">Search by patient, request, or requester</span>
                <input
                  type="search"
                  value={hubSearch}
                  onChange={(event) => setHubSearch(event.target.value)}
                  placeholder="Search patient, request, requester"
                />
              </label>
            </div>

            <div className="radiology-connection-strip" aria-label="PACS and RIS integration status">
              {connections.map((connection) => (
                <span className="radiology-connection-status" key={connection}>
                  <span aria-hidden="true" />
                  {connection}
                  <strong>Not configured</strong>
                </span>
              ))}
            </div>

            <div className="radiology-hub-filters" aria-label="Filter imaging requests">
              {['ALL', 'CT', 'MRI', 'X-ray', 'Ultrasound'].map((filter) => (
                <button
                  key={filter}
                  type="button"
                  aria-pressed={hubFilter === filter}
                  onClick={() => setHubFilter(filter)}
                >
                  {filter === 'ALL' ? 'All' : filter}
                </button>
              ))}
              <span className="radiology-live-count">
                {loading ? 'Loading live requests…' : loadError ? 'Live request queue unavailable' : `${filteredHubQueue.length} requests`}
              </span>
            </div>

            {loadError ? (
              <div className="radiology-hub-error" role="alert">
                Unable to load the hospital radiology queue: {loadError}
              </div>
            ) : (
              <div className="radiology-stage-board" aria-label="Imaging request stages">
                {hubStages.map((stage) => {
                  return (
                    <section className="radiology-stage-column" key={stage.key} aria-label={`${stage.title} stage`}>
                      <div className="radiology-stage-heading">
                        <div>
                          <h3>{stage.title}</h3>
                          <p>{stage.detail}</p>
                        </div>
                        <span className={stage.integrationStage ? 'radiology-stage-unavailable' : 'radiology-stage-count'}>
                          {stage.integrationStage ? 'PACS / RIS' : loading ? '…' : stage.requests.length}
                        </span>
                      </div>
                      <div className="radiology-stage-cards">
                        {stage.integrationStage ? (
                          <p className="radiology-stage-empty">This stage will populate from the configured imaging-system event feed.</p>
                        ) : loading ? (
                          <p className="radiology-stage-empty">Loading live requests…</p>
                        ) : stage.requests.length ? (
                          stage.requests.map((request) => renderHubRequest(request, stage.tab))
                        ) : (
                          <p className="radiology-stage-empty">No requests at this stage.</p>
                        )}
                      </div>
                    </section>
                  );
                })}
              </div>
            )}

            <section className="radiology-hub-panel radiology-queue-summary" aria-labelledby="radiology-action-title">
              <div className="radiology-hub-panel-heading">
                <div>
                  <p className="radiology-hub-eyebrow">Live Nurshelp requests</p>
                  <h2 id="radiology-action-title">Needs action now</h2>
                </div>
                <button type="button" onClick={() => scrollToWorklist('pending')}>Open worklist</button>
              </div>
              {loading ? (
                <p className="radiology-stage-empty">Loading the hospital request queue…</p>
              ) : loadError ? (
                <p className="radiology-stage-empty">The worklist is unavailable until the request queue loads.</p>
              ) : (
                <>
                  {[
                    { title: 'Urgent requests to vet', requests: stageRequests.new.filter((request) => ['URGENT', 'HIGH'].includes(request.priority)), tab: 'pending', action: 'Review request' },
                    { title: 'Accepted requests to start', requests: stageRequests.vetted, tab: 'in-progress', action: 'Open request' },
                    { title: 'Reports awaiting physician review', requests: stageRequests.released, tab: 'completed', action: 'View report' }
                  ].map((group) => (
                    <div className="radiology-action-group" key={group.title}>
                      <h3>{group.title}<span>{group.requests.length}</span></h3>
                      {group.requests.length ? group.requests.map((request) => (
                        <div className="radiology-action-row" key={request.request_id}>
                          <span className={`radiology-action-tag ${request.priority === 'URGENT' ? 'is-urgent' : ''}`}>
                            {request.priority || 'Priority unavailable'}
                          </span>
                          <div>
                            <strong>{request.lab_type || 'Imaging request'} · {request.patientName || 'Patient name unavailable'}</strong>
                            <p>{request.requested_by || 'Ordering clinician unavailable'} · {requestAge(request.requested_at)}</p>
                          </div>
                          <button type="button" onClick={() => openWorklistRequest(request, group.tab)}>{group.action}</button>
                        </div>
                      )) : <p className="radiology-stage-empty">No requests in this group.</p>}
                    </div>
                  ))}
                </>
              )}
            </section>
          </section>
        )}

        {activeView === 'worklist' && (
          <>
        {/* TABS */}
        <div id="specialist-worklist" className="specialist-worklist-tabs" style={{
          display: 'flex',
          gap: '8px',
          padding: '16px 32px',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          background: 'rgba(15,23,42,0.5)'
        }}>
          {[
            { key: 'pending', label: '📥 Pending Orders', count: labQueue.filter(r => r.status === 'PENDING').length },
            { key: 'in-progress', label: '⏳ In Progress', count: labQueue.filter(r => ['ACKNOWLEDGED', 'IN_PROGRESS'].includes(r.status)).length },
            { key: 'completed', label: '✓ Reports Submitted', count: labQueue.filter(r => r.status === 'COMPLETED').length },
            { key: 'submitted', label: '✓ Physician Reviewed', count: labQueue.filter(r => r.status === 'REVIEWED').length }
          ].map(tab => (
            <button
              key={tab.key}
            type="button"
            aria-pressed={activeTab === tab.key}
              onClick={() => handleTabChange(tab.key)}
              style={{
                padding: '10px 16px',
                borderRadius: '12px',
                border: 'none',
                background: activeTab === tab.key ? 'rgba(6,182,212,0.25)' : 'transparent',
                color: activeTab === tab.key ? '#06b6d4' : '#94a3b8',
                fontWeight: activeTab === tab.key ? 700 : 500,
                fontSize: '13px',
                cursor: 'pointer',
                transition: 'all 0.2s',
                whiteSpace: 'nowrap',
                borderBottom: activeTab === tab.key ? '2px solid #06b6d4' : '2px solid transparent'
              }}
              onMouseEnter={(e) => !activeTab !== tab.key && (e.target.style.color = '#cbd5e1')}
              onMouseLeave={(e) => !activeTab !== tab.key && (e.target.style.color = '#94a3b8')}
            >
              {tab.label} ({loading ? '...' : loadError ? '—' : tab.count})
            </button>
          ))}
        </div>

        {/* CONTENT AREA */}
        <div className={`specialist-worklist-content${selectedLabRequest ? ' has-selected-request' : ''}`} style={{ padding: '20px 32px' }}>
          {/* LEFT: LAB QUEUE LIST */}
          <div className="specialist-worklist-queue">
            <div className="specialist-worklist-priority-filters">
              {['ALL', 'URGENT', 'HIGH', 'MEDIUM', 'LOW'].map(priority => (
                <button
                  key={priority}
                  type="button"
                  aria-pressed={filterPriority === priority}
                  onClick={() => setFilterPriority(priority)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '8px',
                    border: filterPriority === priority ? '2px solid #06b6d4' : '1px solid rgba(255,255,255,0.15)',
                    background: filterPriority === priority ? 'rgba(6,182,212,0.2)' : 'transparent',
                    color: filterPriority === priority ? '#06b6d4' : '#94a3b8',
                    fontWeight: filterPriority === priority ? 600 : 500,
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  {priority}
                </button>
              ))}
              </div>

            {loading ? (
              <div style={{ textAlign: 'center', color: '#94a3b8', padding: '40px 20px' }}>
                Loading lab requests...
              </div>
            ) : loadError ? (
              <div role="alert" style={{ textAlign: 'center', color: '#fca5a5', padding: '40px 20px' }}>
                {loadError}
              </div>
            ) : filteredQueue.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#94a3b8', padding: '40px 20px' }}>
                No {filterPriority !== 'ALL' ? filterPriority.toLowerCase() : ''} lab requests at this time
              </div>
            ) : (
              filteredQueue.map(labRequest => renderLabCard(labRequest, activeTab === 'pending'))
            )}
          </div>

          {/* RIGHT: DETAIL PANEL */}
          {selectedLabRequest && (
            <div className="specialist-detail-panel" style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
              padding: '20px',
              borderRadius: '16px',
              border: '1px solid rgba(255,255,255,0.12)',
              background: 'linear-gradient(135deg, rgba(6,182,212,0.1), rgba(14,165,233,0.1))'
            }}>
              {/* PATIENT INFO */}
              <div style={{
                padding: '16px',
                borderRadius: '12px',
                background: 'rgba(15,23,42,0.6)',
                border: '1px solid rgba(255,255,255,0.08)'
              }}>
                <p style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#7c3aed', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
                  Patient Information
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
                      Name
                    </p>
                    <p style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: '#eff6ff' }}>
                      {selectedLabRequest.patientName || 'Unknown'}
                    </p>
                  </div>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
                      Age
                    </p>
                    <p style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: '#eff6ff' }}>
                      {selectedLabRequest.patientAge}
                    </p>
                  </div>
                </div>
              </div>

              {/* LAB REQUEST DETAILS */}
              <div style={{
                padding: '16px',
                borderRadius: '12px',
                background: 'rgba(15,23,42,0.6)',
                border: '1px solid rgba(255,255,255,0.08)'
              }}>
                <p style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#7c3aed', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
                  Imaging Request
                </p>
                <div style={{ display: 'grid', gap: '12px' }}>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
                      Type
                    </p>
                    <p style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: '#eff6ff' }}>
                      {selectedLabRequest.lab_type}
                    </p>
                  </div>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
                      Indication
                    </p>
                    <p style={{ margin: 0, fontSize: '13px', color: '#cbd5e1', lineHeight: '1.4' }}>
                      {selectedLabRequest.description || 'No indication'}
                    </p>
                  </div>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
                      Ordered By
                    </p>
                    <p style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: '#eff6ff' }}>
                      {selectedLabRequest.requested_by}
                    </p>
                  </div>
                </div>
              </div>

              <EhrContextPanel context={selectedLabRequest.ehrContext} />

              {/* ACTION BUTTONS */}
              {activeTab === 'pending' && selectedLabRequest.status === 'PENDING' && (
                <button
                  disabled={savingAction}
                  onClick={() => handleAcceptLabRequest(selectedLabRequest)}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    borderRadius: '12px',
                    border: 'none',
                    background: 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
                    color: '#fff',
                    fontWeight: 700,
                    fontSize: '14px',
                    cursor: 'pointer'
                  }}
                >
                  {savingAction ? 'Saving…' : 'Accept & vet request'}
                </button>
              )}

              {selectedLabRequest.status === 'ACKNOWLEDGED' && (
                <button
                  disabled={savingAction}
                  onClick={() => handleStartRequest(selectedLabRequest)}
                  className="radiology-primary-action"
                  type="button"
                >
                  {savingAction ? 'Saving…' : 'Start radiologist work'}
                </button>
              )}

              {selectedLabRequest.status === 'IN_PROGRESS' && (
                <button
                  disabled={savingAction}
                  onClick={() => {
                    setReportForm({ report_content: '', clinical_impression: '', recommendations: [], report_images: [] });
                    setActionError('');
                    setShowReportModal(true);
                  }}
                  className="radiology-primary-action"
                  type="button"
                >
                  Create & submit report
                </button>
              )}

              {['COMPLETED', 'REVIEWED'].includes(selectedLabRequest.status) && (
                <section className="radiology-saved-report" aria-label="Submitted radiology report">
                  <div className="radiology-report-heading">
                    <h3>Report returned to the EHR</h3>
                    <span>{selectedLabRequest.status === 'REVIEWED' ? 'Physician acknowledged' : 'Awaiting physician review'}</span>
                  </div>
                  <dl>
                    <dt>Findings</dt>
                    <dd>{selectedLabRequest.report_content || 'No findings stored on this report.'}</dd>
                    <dt>Clinical impression</dt>
                    <dd>{selectedLabRequest.clinical_impression || 'No impression stored on this report.'}</dd>
                    {selectedLabRequest.recommendations?.length > 0 && (
                      <>
                        <dt>Recommendations</dt>
                        <dd>{selectedLabRequest.recommendations.join('; ')}</dd>
                      </>
                    )}
                    <dt>Submitted by</dt>
                    <dd>{selectedLabRequest.submitted_by || 'Not recorded'}</dd>
                    <dt>Submitted at</dt>
                    <dd>{selectedLabRequest.submitted_at ? new Date(selectedLabRequest.submitted_at).toLocaleString() : 'Not recorded'}</dd>
                  </dl>
                  <button
                    className="radiology-secondary-action"
                    type="button"
                    onClick={() => navigate(`/summary/${selectedLabRequest.encounterId}`, { state: { openLabRequestId: selectedLabRequest.request_id } })}
                  >
                    Open encounter timeline
                  </button>
                </section>
              )}
            </div>
          )}
        </div>

          </>
        )}

      </main>

      {/* REPORT SUBMISSION MODAL */}
      {showReportModal && selectedLabRequest && (
        <div className="specialist-report-backdrop" style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 999
        }}
        onClick={() => setShowReportModal(false)}>
          <div className="specialist-report-modal" style={{
            background: 'linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95))',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: '16px',
            width: '90%',
            maxWidth: '600px',
            maxHeight: '80vh',
            overflow: 'auto',
            padding: '28px'
          }}
          onClick={(e) => e.stopPropagation()}>
            <h2 style={{ margin: '0 0 20px 0', fontSize: '20px', fontWeight: 700, color: '#eff6ff' }}>
              Radiology report: {selectedLabRequest.lab_type}
            </h2>
            <p style={{ margin: '0 0 20px 0', fontSize: '13px', color: '#94a3b8' }}>
              Patient: {selectedLabRequest.patientName}
            </p>

            {/* REPORT CONTENT */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', color: '#cbd5e1', fontWeight: 600 }}>
                📝 Findings & Observations
              </label>
              <textarea
                required
                value={reportForm.report_content}
                onChange={(e) => setReportForm({ ...reportForm, report_content: e.target.value })}
                placeholder="Describe imaging findings, measurements, abnormalities detected..."
                style={{
                  width: '100%',
                  minHeight: '120px',
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.15)',
                  background: 'rgba(15,23,42,0.6)',
                  color: '#eff6ff',
                  fontFamily: 'inherit',
                  fontSize: '13px',
                  resize: 'vertical'
                }}
              />
            </div>

            {/* CLINICAL IMPRESSION */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', color: '#cbd5e1', fontWeight: 600 }}>
                🔍 Clinical Impression
              </label>
              <textarea
                required
                value={reportForm.clinical_impression}
                onChange={(e) => setReportForm({ ...reportForm, clinical_impression: e.target.value })}
                placeholder="Radiologist's clinical interpretation and diagnosis..."
                style={{
                  width: '100%',
                  minHeight: '100px',
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.15)',
                  background: 'rgba(15,23,42,0.6)',
                  color: '#eff6ff',
                  fontFamily: 'inherit',
                  fontSize: '13px',
                  resize: 'vertical'
                }}
              />
            </div>

            {/* RECOMMENDATIONS */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', color: '#cbd5e1', fontWeight: 600 }}>
                💡 Recommendations
              </label>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
                <input
                  type="text"
                  value={newRecommendation}
                  onChange={(e) => setNewRecommendation(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddRecommendation()}
                  placeholder="Add recommendation and press Enter..."
                  style={{
                    flex: 1,
                    padding: '8px 12px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.15)',
                    background: 'rgba(15,23,42,0.6)',
                    color: '#eff6ff',
                    fontSize: '13px'
                  }}
                />
                <button
                  onClick={handleAddRecommendation}
                  style={{
                    padding: '8px 16px',
                    borderRadius: '8px',
                    border: 'none',
                    background: 'rgba(6,182,212,0.3)',
                    color: '#06b6d4',
                    fontWeight: 600,
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  Add
                </button>
              </div>
              {reportForm.recommendations.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {reportForm.recommendations.map((rec, idx) => (
                    <div
                      key={idx}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '8px 12px',
                        borderRadius: '8px',
                        background: 'rgba(6,182,212,0.15)',
                        border: '1px solid rgba(6,182,212,0.3)'
                      }}
                    >
                      <span style={{ fontSize: '12px', color: '#cbd5e1' }}>• {rec}</span>
                      <button
                        onClick={() => handleRemoveRecommendation(idx)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#f87171',
                          cursor: 'pointer',
                          fontSize: '16px'
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* BUTTONS */}
            <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
              <button
                onClick={() => setShowReportModal(false)}
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.15)',
                  background: 'transparent',
                  color: '#cbd5e1',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                disabled={savingAction}
                onClick={handleSubmitReport}
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'linear-gradient(135deg, #10b981, #059669)',
                  color: '#fff',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                {savingAction ? 'Saving report…' : 'Submit report to EHR'}
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </SidebarLayout>
  );
};

export default RadiologyDashboard;
