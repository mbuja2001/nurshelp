import React, { useState, useEffect } from 'react';
import { FiSearch, FiAlertCircle, FiRefreshCw, FiFilter, FiDownload } from 'react-icons/fi';
import { GiTestTubes } from 'react-icons/gi';
import SidebarLayout from '../SidebarLayout';

const Pathology = () => {
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [selectedResult, setSelectedResult] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  // Mock pathology results data
  const mockResults = [
    {
      id: 'PATH-001',
      patientId: 'P-12345',
      patientName: 'John Smith',
      testType: 'Complete Blood Count',
      specimen: 'Whole Blood',
      collectionDate: '2026-05-22',
      status: 'completed',
      technician: 'Dr. Leonard Roberts',
      pathologist: 'Dr. Victoria King',
      results: {
        'WBC': { value: '7.2', unit: 'K/uL', reference: '4.5-11.0', flag: 'normal' },
        'RBC': { value: '4.8', unit: 'M/uL', reference: '4.5-5.5', flag: 'normal' },
        'Hemoglobin': { value: '14.2', unit: 'g/dL', reference: '13.5-17.5', flag: 'normal' },
        'Hematocrit': { value: '42%', unit: '%', reference: '40-52', flag: 'normal' },
        'Platelets': { value: '245', unit: 'K/uL', reference: '150-400', flag: 'normal' }
      },
      priority: 'routine',
      interpretation: 'No abnormalities detected'
    },
    {
      id: 'PATH-002',
      patientId: 'P-12346',
      patientName: 'Sarah Johnson',
      testType: 'Blood Culture',
      specimen: 'Whole Blood',
      collectionDate: '2026-05-21',
      status: 'pending',
      technician: 'Dr. Leonard Roberts',
      pathologist: 'Dr. Victoria King',
      results: {},
      priority: 'stat',
      interpretation: 'Pending results - growth monitoring'
    },
    {
      id: 'PATH-003',
      patientId: 'P-12347',
      patientName: 'Michael Chen',
      testType: 'Urinalysis',
      specimen: 'Urine',
      collectionDate: '2026-05-20',
      status: 'completed',
      technician: 'Dr. Leonard Roberts',
      pathologist: 'Dr. Victoria King',
      results: {
        'Color': { value: 'Pale Yellow', unit: '', reference: 'Colorless to pale yellow', flag: 'normal' },
        'Clarity': { value: 'Clear', unit: '', reference: 'Clear', flag: 'normal' },
        'pH': { value: '6.5', unit: '', reference: '4.8-7.5', flag: 'normal' },
        'Specific Gravity': { value: '1.015', unit: '', reference: '1.005-1.030', flag: 'normal' },
        'Leukocytes': { value: 'Negative', unit: '', reference: 'Negative', flag: 'normal' }
      },
      priority: 'routine',
      interpretation: 'Normal urinalysis'
    },
    {
      id: 'PATH-004',
      patientId: 'P-12348',
      patientName: 'Emma Wilson',
      testType: 'Glucose Test',
      specimen: 'Serum',
      collectionDate: '2026-05-19',
      status: 'completed',
      technician: 'Dr. Leonard Roberts',
      pathologist: 'Dr. Victoria King',
      results: {
        'Fasting Glucose': { value: '118', unit: 'mg/dL', reference: '70-100', flag: 'high' },
        'Glucose Random': { value: '145', unit: 'mg/dL', reference: '<140', flag: 'normal' }
      },
      priority: 'routine',
      interpretation: 'Slightly elevated fasting glucose; further monitoring recommended'
    },
    {
      id: 'PATH-005',
      patientId: 'P-12349',
      patientName: 'David Brown',
      testType: 'Lipid Panel',
      specimen: 'Serum',
      collectionDate: '2026-05-18',
      status: 'archived',
      technician: 'Dr. Leonard Roberts',
      pathologist: 'Dr. Victoria King',
      results: {
        'Total Cholesterol': { value: '220', unit: 'mg/dL', reference: '<200', flag: 'high' },
        'HDL': { value: '35', unit: 'mg/dL', reference: '>40', flag: 'low' },
        'LDL': { value: '155', unit: 'mg/dL', reference: '<100', flag: 'high' },
        'Triglycerides': { value: '180', unit: 'mg/dL', reference: '<150', flag: 'high' }
      },
      priority: 'routine',
      interpretation: 'Abnormal lipid panel; consider dietary modification and follow-up testing'
    }
  ];

  useEffect(() => {
    fetchResults();
  }, []);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/pathology/results`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        setResults(mockResults);
      }
    } catch (err) {
      console.log('Using mock pathology results data');
      setResults(mockResults);
    }
    setLoading(false);
  };

  const getStatusColor = (status) => {
    switch(status) {
      case 'completed': return { bg: '#dcfce7', color: '#166534', label: 'Completed' };
      case 'pending': return { bg: '#fef3c7', color: '#92400e', label: 'Pending' };
      case 'archived': return { bg: '#f3f4f6', color: '#4b5563', label: 'Archived' };
      case 'processing': return { bg: '#dbeafe', color: '#0c4a6e', label: 'Processing' };
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

  const getFlagColor = (flag) => {
    switch(flag) {
      case 'high': return '#ef4444';
      case 'low': return '#f59e0b';
      case 'normal': return '#10b981';
      default: return '#64748b';
    }
  };

  const filteredResults = results.filter(result => {
    const matchesSearch = result.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         result.testType.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         result.id.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (filterStatus === 'all') return matchesSearch;
    return matchesSearch && result.status === filterStatus;
  });

  return (
    <SidebarLayout>
      <div style={{ padding: '24px', backgroundColor: '#f8fafc', minHeight: '100vh' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <GiTestTubes size={28} style={{ color: '#10b981' }} />
            <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#0f172a' }}>
              Pathology Results
            </h1>
          </div>
          <p style={{ margin: 0, color: '#64748b', fontSize: '14px' }}>
            Review laboratory test results, hematology reports, and clinical pathology findings
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
              placeholder="Search results..."
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
            <option value="all">All Results</option>
            <option value="completed">Completed</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="archived">Archived</option>
          </select>

          {/* Refresh */}
          <button
            onClick={fetchResults}
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
              <div style={{ fontSize: '18px', fontWeight: '700', color: '#10b981' }}>{results.length}</div>
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
                {results.filter(r => r.status === 'pending').length}
              </div>
            </div>
          </div>
        </div>

        {/* Results Table */}
        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px', color: '#94a3b8' }}>
            Loading results...
          </div>
        ) : (
          <div style={{ backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Patient</th>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Test Type</th>
                  <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Specimen</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Collection Date</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Priority</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Status</th>
                  <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.length === 0 ? (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '32px', color: '#94a3b8' }}>
                      No results found
                    </td>
                  </tr>
                ) : (
                  filteredResults.map(result => {
                    const statusInfo = getStatusColor(result.status);
                    const priorityInfo = getPriorityColor(result.priority);

                    return (
                      <tr key={result.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '12px 16px' }}>
                          <div style={{ fontWeight: '600', color: '#0f172a' }}>{result.patientName}</div>
                          <div style={{ fontSize: '12px', color: '#94a3b8' }}>{result.patientId}</div>
                        </td>
                        <td style={{ padding: '12px 16px', color: '#0f172a', fontWeight: '500' }}>{result.testType}</td>
                        <td style={{ padding: '12px 16px', color: '#64748b' }}>{result.specimen}</td>
                        <td style={{ padding: '12px 16px', textAlign: 'center', color: '#0f172a' }}>
                          {new Date(result.collectionDate).toLocaleDateString()}
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
                            {result.priority}
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
                          <button
                            onClick={() => {
                              setSelectedResult(result);
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
        {showDetailModal && selectedResult && (
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
              maxWidth: '700px',
              width: '90%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: '0 20px 25px rgba(0,0,0,0.15)'
            }} onClick={(e) => e.stopPropagation()}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>
                  {selectedResult.testType}
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
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedResult.patientName}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Patient ID</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedResult.patientId}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Specimen Type</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedResult.specimen}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Collection Date</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{new Date(selectedResult.collectionDate).toLocaleDateString()}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Technician</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedResult.technician}</div>
                </div>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Pathologist</div>
                  <div style={{ fontWeight: '600', color: '#0f172a' }}>{selectedResult.pathologist}</div>
                </div>
              </div>

              {Object.keys(selectedResult.results).length > 0 && (
                <div style={{ marginBottom: '20px' }}>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', fontWeight: '600', color: '#0f172a' }}>Test Results</h3>
                  <div style={{ backgroundColor: '#f8fafc', borderRadius: '8px', overflow: 'hidden' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ backgroundColor: '#ecfdf5', borderBottom: '1px solid #d1fae5' }}>
                          <th style={{ padding: '12px', textAlign: 'left', fontSize: '12px', fontWeight: '600', color: '#065f46' }}>Parameter</th>
                          <th style={{ padding: '12px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#065f46' }}>Value</th>
                          <th style={{ padding: '12px', textAlign: 'center', fontSize: '12px', fontWeight: '600', color: '#065f46' }}>Reference</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(selectedResult.results).map(([key, val]) => (
                          <tr key={key} style={{ borderBottom: '1px solid #d1fae5' }}>
                            <td style={{ padding: '12px', fontSize: '14px', fontWeight: '500', color: '#0f172a' }}>{key}</td>
                            <td style={{ padding: '12px', textAlign: 'center' }}>
                              <span style={{
                                color: getFlagColor(val.flag),
                                fontWeight: '600',
                                fontSize: '14px'
                              }}>
                                {val.value} {val.unit}
                              </span>
                            </td>
                            <td style={{ padding: '12px', textAlign: 'center', fontSize: '12px', color: '#64748b' }}>
                              {val.reference}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div style={{ backgroundColor: '#ecfdf5', padding: '12px', borderRadius: '8px', marginBottom: '20px' }}>
                <div style={{ fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '8px' }}>Interpretation</div>
                <div style={{ fontSize: '14px', color: '#065f46' }}>{selectedResult.interpretation}</div>
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
      </div>
    </SidebarLayout>
  );
};

export default Pathology;
