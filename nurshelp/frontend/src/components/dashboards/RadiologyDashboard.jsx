import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import decodeJwt from '../../utils/decodeJwt';
import './RadiologyDashboard.css';

/**
 * RadiologyDashboard Component
 * 
 * RADIOLOGY DEPARTMENT CONTEXT
 * Role: Diagnostic imaging specialists - process radiology requests
 * Controls:
 *   - View pending lab requests (CT, X-ray, ultrasound, MRI, etc.)
 *   - Accept & acknowledge lab orders
 *   - Document findings and clinical impressions
 *   - Submit reports back to ordering physician
 *   - Track report acknowledgment by physician
 */

const BACKEND_URL = 'http://localhost:5001';

const RadiologyDashboard = ({ config }) => {
  const navigate = useNavigate();
  const userInfo = decodeJwt(localStorage.getItem('token'));

  // ============ STATE MANAGEMENT ============
  const [activeTab, setActiveTab] = useState('pending'); // pending, in-progress, completed, submitted
  const [labQueue, setLabQueue] = useState([]);
  const [selectedLabRequest, setSelectedLabRequest] = useState(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [filterPriority, setFilterPriority] = useState('ALL'); // ALL, URGENT, HIGH, MEDIUM, LOW

  const [reportForm, setReportForm] = useState({
    report_content: '',
    clinical_impression: '',
    recommendations: [],
    report_images: []
  });
  const [newRecommendation, setNewRecommendation] = useState('');

  // ============ FETCH LAB REQUESTS ============
  const fetchLabRequests = async (status = 'PENDING') => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${BACKEND_URL}/api/lab-requests/queue/requests?specialty=RADIOLOGY&status=${status}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch lab requests');
      }

      const data = await response.json();
      setLabQueue(data.lab_requests || []);
      console.log(`[RADIOLOGY] Fetched ${data.lab_requests?.length || 0} ${status} lab requests`);
    } catch (err) {
      console.error('[RADIOLOGY] Fetch error:', err);
      alert(`Error fetching lab requests: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('[RADIOLOGY] Dashboard loaded');
    fetchLabRequests('PENDING');
  }, []);

  // ============ TAB SWITCHING ============
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setSelectedLabRequest(null);

    const statusMap = {
      'pending': 'PENDING',
      'in-progress': 'ACKNOWLEDGED',
      'completed': 'COMPLETED',
      'submitted': 'REVIEWED'
    };

    if (statusMap[tab]) {
      fetchLabRequests(statusMap[tab]);
    }
  };

  // ============ ACCEPT LAB REQUEST ============
  const handleAcceptLabRequest = async (labRequest) => {
    if (!labRequest || !labRequest.encounterId || !labRequest.request_id) {
      alert('Invalid lab request data');
      return;
    }

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

      if (!response.ok) {
        throw new Error('Failed to accept lab request');
      }

      alert(`✓ Lab request accepted: ${labRequest.lab_type}`);
      setSelectedLabRequest(null);
      fetchLabRequests('PENDING'); // Refresh queue
      fetchLabRequests('ACKNOWLEDGED'); // Show in in-progress
    } catch (err) {
      console.error('[RADIOLOGY] Accept error:', err);
      alert(`Error: ${err.message}`);
    }
  };

  // ============ SUBMIT LAB REPORT ============
  const handleSubmitReport = async () => {
    if (!selectedLabRequest || !reportForm.report_content || !reportForm.clinical_impression) {
      alert('Please fill in report content and clinical impression');
      return;
    }

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
            recommendations: reportForm.recommendations,
            report_images: reportForm.report_images,
            submitted_by: userInfo?.name || 'Radiologist'
          })
        }
      );

      if (!response.ok) {
        throw new Error('Failed to submit lab report');
      }

      alert(`✓ Report submitted for ${selectedLabRequest.lab_type}`);
      setShowReportModal(false);
      setReportForm({ report_content: '', clinical_impression: '', recommendations: [], report_images: [] });
      setSelectedLabRequest(null);
      fetchLabRequests('ACKNOWLEDGED'); // Refresh queue
    } catch (err) {
      console.error('[RADIOLOGY] Submit report error:', err);
      alert(`Error: ${err.message}`);
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
  const filteredQueue = labQueue.filter(req => {
    if (filterPriority === 'ALL') return true;
    return req.priority === filterPriority;
  });

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
            ✓ Accept & Start Imaging
          </button>
        )}
      </div>
    );
  };

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      color: '#e2e8f0',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
    }}>
      {/* MAIN CONTENT */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* HEADER */}
        <header style={{
          padding: '24px 32px',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          background: 'linear-gradient(180deg, rgba(15,23,42,0.95), rgba(30,41,59,0.88))'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 800, background: 'linear-gradient(90deg, #06b6d4, #0ea5e9)', backgroundClip: 'text', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                🔬 RADIOLOGY DEPARTMENT
              </h1>
              <p style={{ margin: '8px 0 0 0', fontSize: '14px', color: '#94a3b8' }}>
                {userInfo?.name || 'Radiologist'} • {userInfo?.department || 'Imaging'}
              </p>
            </div>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-end',
              gap: '8px'
            }}>
              <div style={{ padding: '8px 14px', borderRadius: '20px', background: 'rgba(6,182,212,0.15)', border: '1px solid rgba(6,182,212,0.3)' }}>
                <span style={{ fontSize: '12px', color: '#06b6d4', fontWeight: 600, textTransform: 'uppercase' }}>
                  Pending: {labQueue.filter(r => r.status === 'PENDING').length}
                </span>
              </div>
              <div style={{ padding: '8px 14px', borderRadius: '20px', background: 'rgba(10,185,129,0.15)', border: '1px solid rgba(10,185,129,0.3)' }}>
                <span style={{ fontSize: '12px', color: '#10b981', fontWeight: 600, textTransform: 'uppercase' }}>
                  In Progress: {labQueue.filter(r => r.status === 'ACKNOWLEDGED').length}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* TABS */}
        <div style={{
          display: 'flex',
          gap: '8px',
          padding: '16px 32px',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          background: 'rgba(15,23,42,0.5)',
          overflowX: 'auto'
        }}>
          {[
            { key: 'pending', label: '📥 Pending Orders', count: labQueue.filter(r => r.status === 'PENDING').length },
            { key: 'in-progress', label: '⏳ In Progress', count: labQueue.filter(r => r.status === 'ACKNOWLEDGED').length },
            { key: 'completed', label: '✓ Ready to Submit', count: labQueue.filter(r => r.status === 'COMPLETED' || r.status === 'IN_PROGRESS').length },
            { key: 'submitted', label: '📤 Submitted', count: labQueue.filter(r => r.status === 'REVIEWED' || r.status === 'COMPLETED').length }
          ].map(tab => (
            <button
              key={tab.key}
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
              {tab.label} ({tab.count})
            </button>
          ))}
        </div>

        {/* CONTENT AREA */}
        <div style={{ flex: 1, display: 'flex', gap: '20px', padding: '20px 32px', overflow: 'hidden' }}>
          {/* LEFT: LAB QUEUE LIST */}
          <div style={{
            flex: activeTab === 'pending' || activeTab === 'in-progress' ? '0 0 45%' : '1',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            overflowY: 'auto',
            paddingRight: '12px'
          }}>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              {['ALL', 'URGENT', 'HIGH', 'MEDIUM', 'LOW'].map(priority => (
                <button
                  key={priority}
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
            <div style={{
              flex: '0 0 55%',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
              padding: '20px',
              borderRadius: '16px',
              border: '1px solid rgba(255,255,255,0.12)',
              background: 'linear-gradient(135deg, rgba(6,182,212,0.1), rgba(14,165,233,0.1))',
              overflowY: 'auto'
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

              {/* ACTION BUTTONS */}
              {activeTab === 'pending' && selectedLabRequest.status === 'PENDING' && (
                <button
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
                  ✓ Accept Imaging Request
                </button>
              )}

              {activeTab === 'in-progress' && selectedLabRequest.status === 'ACKNOWLEDGED' && (
                <button
                  onClick={() => setShowReportModal(true)}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    borderRadius: '12px',
                    border: 'none',
                    background: 'linear-gradient(135deg, #10b981, #059669)',
                    color: '#fff',
                    fontWeight: 700,
                    fontSize: '14px',
                    cursor: 'pointer'
                  }}
                >
                  📋 Write & Submit Report
                </button>
              )}
            </div>
          )}
        </div>
      </main>

      {/* REPORT SUBMISSION MODAL */}
      {showReportModal && selectedLabRequest && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 999
        }}
        onClick={() => setShowReportModal(false)}>
          <div style={{
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
              📋 Imaging Report: {selectedLabRequest.lab_type}
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
                ✓ Submit Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RadiologyDashboard;
