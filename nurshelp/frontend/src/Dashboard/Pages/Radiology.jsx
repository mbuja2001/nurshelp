import React, { useState, useEffect } from 'react';
import { FiSearch, FiPlus, FiMinus, FiAlertCircle, FiRefreshCw, FiFilter, FiDownload } from 'react-icons/fi';
import { FaRadiation } from 'react-icons/fa';
import SidebarLayout from '../SidebarLayout';

const Radiology = () => {
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  
  const [studies, setStudies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [selectedStudy, setSelectedStudy] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportText, setReportText] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  // Mock radiology studies data
  const mockStudies = [
    {
      id: 'RAD-001',
      patientId: 'P-12345',
      patientName: 'John Smith',
      studyType: 'Chest X-Ray',
      modality: 'X-Ray',
      orderDate: '2026-05-22',
      status: 'completed',
      technician: 'Dr. Peter Mitchell',
      radiologist: 'Dr. Robert Brown',
      findings: 'No acute cardiopulmonary process',
      impression: 'Normal chest radiograph',
      priority: 'routine',
      images: 2
    },
    {
      id: 'RAD-002',
      patientId: 'P-12346',
      patientName: 'Sarah Johnson',
      studyType: 'CT Head',
      modality: 'CT',
      orderDate: '2026-05-21',
      status: 'completed',
      technician: 'Dr. Andrew Morris',
      radiologist: 'Dr. Robert Brown',
      findings: 'No intracranial hemorrhage or mass effect',
      impression: 'CT head without acute findings',
      priority: 'stat',
      images: 45
    },
    {
      id: 'RAD-003',
      patientId: 'P-12347',
      patientName: 'Michael Chen',
      studyType: 'Abdominal Ultrasound',
      modality: 'Ultrasound',
      orderDate: '2026-05-20',
      status: 'pending-report',
      technician: 'Dr. Marcus White',
      radiologist: 'Dr. Robert Brown',
      findings: 'Pending radiologist review',
      impression: 'Report pending',
      priority: 'routine',
      images: 18
    },
    {
      id: 'RAD-004',
      patientId: 'P-12348',
      patientName: 'Emma Wilson',
      studyType: 'MRI Knee',
      modality: 'MRI',
      orderDate: '2026-05-19',
      status: 'completed',
      technician: 'Dr. Jennifer Garcia',
      radiologist: 'Dr. Robert Brown',
      findings: 'Anterior cruciate ligament tear with meniscal involvement',
      impression: 'ACL tear with medial meniscus pathology',
      priority: 'routine',
      images: 52
    },
    {
      id: 'RAD-005',
      patientId: 'P-12349',
      patientName: 'David Brown',
      studyType: 'Spine X-Ray',
      modality: 'X-Ray',
      orderDate: '2026-05-22',
      status: 'archived',
      technician: 'Dr. Peter Mitchell',
      radiologist: 'Dr. Robert Brown',
      findings: 'Degenerative disc disease at L4-L5',
      impression: 'Chronic degenerative changes',
      priority: 'routine',
      images: 4
    }
  ];

  useEffect(() => {
    fetchStudies();
  }, []);

  const fetchStudies = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/radiology/studies`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setStudies(data);
      } else {
        setStudies(mockStudies);
      }
    } catch (err) {
      console.log('Using mock radiology studies data');
      setStudies(mockStudies);
    }
    setLoading(false);
  };

  const getStatusColor = (status) => {
    switch(status) {
      case 'completed': return { bg: '#dcfce7', color: '#166534', label: 'Completed' };
      case 'pending-report': return { bg: '#fef3c7', color: '#92400e', label: 'Pending Report' };
      case 'archived': return { bg: '#f3f4f6', color: '#4b5563', label: 'Archived' };
      case 'in-progress': return { bg: '#dbeafe', color: '#0c4a6e', label: 'In Progress' };
      default: return { bg: '#f3f4f6', color: '#4b5563', label: status };
    }
  };

  const getPriorityColor = (priority) => {
    switch(priority) {
      case 'stat': return { bg: '#fee2e2', color: '#991b1b' };
      case 'urgent': return { bg: '#fef3c7', color: '#92400e' };
      case 'routine': return { bg: '#ecfdf5', color: '#065f46' };
      default: return { bg: '#f3f4f6', color: '#4b5563' };
    }
  };

  const filteredStudies = studies.filter(study => {
    const matchesSearch = study.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         study.studyType.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         study.id.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (filterStatus === 'all') return matchesSearch;
    return matchesSearch && study.status === filterStatus;
  });

  const handleSubmitReport = async () => {
    if (!selectedStudy || !reportText.trim()) {
      setError('Please enter a report');
      setTimeout(() => setError(''), 3000);
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/radiology/report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          studyId: selectedStudy.id,
          report: reportText,
          timestamp: new Date()
        })
      });

      if (response.ok) {
        setSuccessMessage(`Report submitted for ${selectedStudy.studyType}`);
        setTimeout(() => setSuccessMessage(''), 3000);
        setShowReportModal(false);
        setReportText('');
        
        // Update study status
        setStudies(prev => prev.map(s => 
          s.id === selectedStudy.id 
            ? { ...s, status: 'completed' }
            : s
        ));
      }
    } catch (err) {
      setError('Failed to submit report: ' + err.message);
      setTimeout(() => setError(''), 3000);
    }
  };

  return (
    <SidebarLayout>
      <div style={{ padding: '24px', backgroundColor: '#f8fafc', minHeight: '100vh' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <FaRadiation size={28} style={{ color: '#10b981' }} />
            <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#0f172a' }}>
              Radiology Studies
            </h1>
          </div>
          <p style={{ margin: 0, color: '#64748b', fontSize: '14px' }}>
            Access medical imaging studies, review reports, and manage radiological findings
          </p>
        </div>

        {/* Messages */}
        {error && (
          <div style={{
            backgroundColor: '#fee2e2',
            color: '#991b1b',
            padding: '12px 16px',
            borderRadius: '8px',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <FiAlertCircle size={18} />
            {error}
          </div>
        )}

        {successMessage && (
          <div style={{
            backgroundColor: '#dcfce7',
            color: '#166534',
            padding: '12px 16px',
            borderRadius: '8px',
            marginBottom: '16px'
          }}>
            ✓ {successMessage}
          </div>
        )}

        {/* Controls */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto auto auto',
          gap: '12px',
          marginBottom: '24px',
          alignItems: 'center'
        }}>
          {/* Search */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            backgroundColor: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '8px 12px'
          }}>
            <FiSearch style={{ color: '#94a3b8', marginRight: '8px' }} />
            <input
              type="text"
              placeholder="Search studies..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                flex: 1,
                border: 'none',
                outline: 'none',
                fontSize: '14px'
              }}
            />
          </div>

          {/* Filter */}
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            style={{
              padding: '8px 12px',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              fontSize: '14px',
              backgroundColor: '#fff',
              cursor: 'pointer'
            }}
          >
            <option value="all">All Studies</option>
            <option value="completed">Completed</option>
            <option value="pending-report">Pending Report</option>
            <option value="in-progress">In Progress</option>
            <option value="archived">Archived</option>
          </select>

          {/* Refresh */}
          <button
            onClick={fetchStudies}
            style={{
              padding: '8px 16px',
              backgroundColor: '#f1f5f9',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '14px',
              fontWeight: '500',
              color: '#64748b',
              transition: 'all 200ms ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#e2e8f0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#f1f5f9';
            }}
          >
            <FiRefreshCw size={16} />
            Refresh
          </button>

          {/* Stats */}
          <div style={{
            display: 'flex',
            gap: '12px'
          }}>
            <div style={{
              backgroundColor: '#fff',
              padding: '8px 12px',
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '12px', color: '#64748b' }}>Total</div>
              <div style={{ fontSize: '18px', fontWeight: '700', color: '#10b981' }}>{studies.length}</div>
            </div>
            <div style={{
              backgroundColor: '#fff',
              padding: '8px 12px',
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '12px', color: '#64748b' }}>Pending</div>
              <div style={{ fontSize: '18px', fontWeight: '700', color: '#f59e0b' }}>
                {studies.filter(s => s.status === 'pending-report').length}
              </div>
            </div>
          </div>
        </div>

        {/* Studies Table */}
        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px', color: '#94a3b8' }}>
            Loading studies...
          </div>
        ) : (
          <div style={{ backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Patient</th>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Study Type</th>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Modality</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Images</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Priority</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Status</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredStudies.length === 0 ? (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '32px', color: '#94a3b8' }}>
                      No studies found
                    </td>
                  </tr>
                ) : (
                  filteredStudies.map(study => {
                    const statusInfo = getStatusColor(study.status);
                    const priorityInfo = getPriorityColor(study.priority);

                    return (
                      <tr key={study.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '12px 16px' }}>
                          <div style={{ fontWeight: '600', color: '#0f172a' }}>{study.patientName}</div>
                          <div style={{ fontSize: '12px', color: '#94a3b8' }}>{study.patientId}</div>
                        </td>
                        <td style={{ padding: '12px 16px', color: '#0f172a', fontWeight: '500' }}>{study.studyType}</td>
                        <td style={{ padding: '12px 16px', color: '#64748b' }}>{study.modality}</td>
                        <td style={{ padding: '12px 16px', textAlign: 'center', color: '#0f172a', fontWeight: '600' }}>
                          {study.images}
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <span style={{
                            backgroundColor: priorityInfo.bg,
                            color: priorityInfo.color,
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '600',
                            textTransform: 'uppercase'
                          }}>
                            {study.priority}
                          </span>
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <span style={{
                            backgroundColor: statusInfo.bg,
                            color: statusInfo.color,
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '600'
                          }}>
                            {statusInfo.label}
                          </span>
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
                            <button
                              onClick={() => {
                                setSelectedStudy(study);
                                setShowDetailModal(true);
                              }}
                              style={{
                                padding: '6px 12px',
                                backgroundColor: '#ecfdf5',
                                color: '#10b981',
                                border: '1px solid #d1fae5',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '12px',
                                fontWeight: '600',
                                transition: 'all 150ms ease'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#d1fae5';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#ecfdf5';
                              }}
                            >
                              View
                            </button>
                            {study.status === 'pending-report' && (
                              <button
                                onClick={() => {
                                  setSelectedStudy(study);
                                  setShowReportModal(true);
                                }}
                                style={{
                                  padding: '6px 12px',
                                  backgroundColor: '#dbeafe',
                                  color: '#0284c7',
                                  border: '1px solid #bfdbfe',
                                  borderRadius: '4px',
                                  cursor: 'pointer',
                                  fontSize: '12px',
                                  fontWeight: '600',
                                  transition: 'all 150ms ease'
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.backgroundColor = '#93c5fd';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.backgroundColor = '#dbeafe';
                                }}
                              >
                                Report
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Detail Modal */}
        {showDetailModal && selectedStudy && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }} onClick={() => setShowDetailModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '600px',
              width: '90%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                  {selectedStudy.studyType}
                </h2>
                <button
                  onClick={() => setShowDetailModal(false)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '24px', color: '#94a3b8' }}
                >
                  ✕
                </button>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '16px',
                marginBottom: '20px'
              }}>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Patient Name</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedStudy.patientName}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Patient ID</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedStudy.patientId}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Modality</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedStudy.modality}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Number of Images</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedStudy.images}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Order Date</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{new Date(selectedStudy.orderDate).toLocaleDateString()}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Technician</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedStudy.technician}</div>
                </div>
              </div>

              <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px', marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Radiologist</div>
                <div style={{ fontWeight: '600', color: '#0f172a', marginBottom: '12px' }}>{selectedStudy.radiologist}</div>
              </div>

              <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px', marginBottom: '20px' }}>
                <div style={{ fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>Findings</div>
                <div style={{ fontSize: '14px', color: '#475569' }}>{selectedStudy.findings}</div>
              </div>

              <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px', marginBottom: '20px' }}>
                <div style={{ fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>Impression</div>
                <div style={{ fontSize: '14px', color: '#475569' }}>{selectedStudy.impression}</div>
              </div>

              <button
                onClick={() => setShowDetailModal(false)}
                style={{
                  width: '100%',
                  padding: '12px 16px',
                  backgroundColor: '#10b981',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 200ms ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#059669';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#10b981';
                }}
              >
                Close
              </button>
            </div>
          </div>
        )}

        {/* Report Modal */}
        {showReportModal && selectedStudy && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001
          }} onClick={() => setShowReportModal(false)}>
            <div style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '600px',
              width: '90%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                Submit Radiology Report
              </h2>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Study
                </label>
                <div style={{
                  backgroundColor: '#f8fafc',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0'
                }}>
                  {selectedStudy.studyType} - {selectedStudy.patientName}
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>
                  Radiology Report
                </label>
                <textarea
                  value={reportText}
                  onChange={(e) => setReportText(e.target.value)}
                  placeholder="Enter findings, impression, and recommendations..."
                  style={{
                    width: '100%',
                    minHeight: '200px',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontFamily: 'monospace',
                    boxSizing: 'border-box',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#10b981';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = '#e2e8f0';
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => setShowReportModal(false)}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#f1f5f9',
                    color: '#64748b',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#e2e8f0';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#f1f5f9';
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmitReport}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    backgroundColor: '#10b981',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#059669';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#10b981';
                  }}
                >
                  Submit Report
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

export default Radiology;
