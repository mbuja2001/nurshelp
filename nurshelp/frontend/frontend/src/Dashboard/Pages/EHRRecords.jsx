import React, { useState, useEffect } from 'react';
import { FiX, FiSearch, FiFilter, FiClock, FiDownload, FiChevronDown } from 'react-icons/fi';
import { FaFlask, FaImage, FaToolbox, FaBed } from 'react-icons/fa';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';

const EHRRecords = () => {
  const { userName } = useUserFromToken();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

  // ============ STATE ============
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('ai-suggested'); // 'ai-suggested', 'nurse-approved', 'physician-attended'
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    triageFrom: '',
    triageTo: '',
    bedDurationMin: '',
    bedDurationMax: '',
    dischargeStatus: 'all', // 'all', 'pending', 'discharged'
  });
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const REPORT_STATUSES = {
    'ai-suggested': { label: 'AI Suggested', color: '#f59e0b', bgColor: '#fef3c7', textColor: '#92400e' },
    'nurse-approved': { label: 'Nurse Approved (Awaiting Physician)', color: '#0369a1', bgColor: '#f0f9ff', textColor: '#0c4a6e' },
    'physician-attended': { label: 'Physician Attended', color: '#10b981', bgColor: '#dcfce7', textColor: '#166534' }
  };

  // ============ FETCH PATIENTS ============
  const fetchPatients = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/encounters`, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!res.ok) throw new Error('Failed to fetch patients');
      const data = await res.json();
      setPatients(Array.isArray(data) ? data : data.encounters || []);
      setErrorMessage('');
    } catch (err) {
      setErrorMessage('Error loading EHR records: ' + err.message);
      // Fallback: mock data
      setPatients(generateMockPatients());
    } finally {
      setLoading(false);
    }
  };

  const generateMockPatients = () => {
    const statuses = ['ai-suggested', 'nurse-approved', 'physician-attended'];
    return Array.from({ length: 15 }, (_, i) => ({
      _id: `pat_${i + 1}`,
      patient: {
        name: `Patient ${String.fromCharCode(65 + (i % 26))} ${i + 1}`,
        age: 30 + Math.floor(Math.random() * 40),
        gender: i % 2 === 0 ? 'M' : 'F',
      },
      triage: {
        triageTime: new Date(Date.now() - Math.random() * 86400000 * 3).toISOString(),
        severity: Math.ceil(Math.random() * 5),
        assigned_specialty: ['Cardiology', 'Pulmonology', 'Neurology', 'General Medicine'][i % 4],
        chief_complaint: ['Chest Pain', 'Shortness of Breath', 'Headache', 'Abdominal Pain'][i % 4],
      },
      beds: {
        assigned: i % 3 !== 0 ? `Bed ${10 + (i % 12)}` : null,
        type: ['Standard', 'Monitored', 'ICU'][i % 3],
        occupiedSince: i % 3 !== 0 ? new Date(Date.now() - Math.random() * 86400000 * 2).toISOString() : null,
        estimatedDischarge: i % 3 !== 0 ? new Date(Date.now() + Math.random() * 86400000).toISOString() : null,
      },
      reportStatus: statuses[i % 3],
      aiSuggestedReport: {
        diagnosis: ['MI', 'Pneumonia', 'Migraine', 'Appendicitis'][i % 4],
        recommendations: ['Admit to CCU', 'Chest X-ray', 'CT Scan', 'Ultrasound'],
        confidence: 0.75 + Math.random() * 0.2,
      },
      nurseApprovedReport: i % 3 >= 1 ? {
        approvedAt: new Date(Date.now() - Math.random() * 3600000).toISOString(),
        nurseNotes: 'Patient stable, awaiting physician evaluation',
      } : null,
      physicianReport: i % 3 === 2 ? {
        attendedAt: new Date(Date.now() - Math.random() * 7200000).toISOString(),
        physicianName: `Dr. ${['Smith', 'Johnson', 'Williams', 'Brown'][i % 4]}`,
        labResults: [
          { test: 'Troponin', value: '0.02', unit: 'ng/mL', status: 'normal' },
          { test: 'BNP', value: '450', unit: 'pg/mL', status: 'elevated' },
        ],
        scans: [
          { type: 'Chest X-ray', findings: 'No acute findings', date: new Date(Date.now() - 3600000).toISOString() },
        ],
        resources: ['IV Line', 'Cardiac Monitor', 'Oxygen Therapy'],
        finalDiagnosis: ['Stable Angina', 'Acute Bronchitis'][i % 2],
        discharged: Math.random() > 0.5,
        dischargeDate: Math.random() > 0.5 ? new Date(Date.now() - Math.random() * 3600000).toISOString() : null,
      } : null,
    }));
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  // ============ FILTER & SEARCH LOGIC ============
  const formatDate = (isoString) => {
    if (!isoString) return 'N/A';
    return new Date(isoString).toLocaleDateString() + ' ' + new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getDaysDuration = (startDate, endDate) => {
    if (!startDate) return 0;
    const end = endDate || new Date();
    return Math.floor((new Date(end) - new Date(startDate)) / (1000 * 60 * 60 * 24));
  };

  const getHoursDuration = (startDate, endDate) => {
    if (!startDate) return 0;
    const end = endDate || new Date();
    return Math.floor((new Date(end) - new Date(startDate)) / (1000 * 60 * 60));
  };

  const filteredPatients = patients
    .filter(p => p.reportStatus === activeTab)
    .filter(p => {
      const query = searchQuery.toLowerCase();
      return p.patient.name.toLowerCase().includes(query) || 
             (p._id && p._id.toLowerCase().includes(query));
    })
    .filter(p => {
      if (filters.triageFrom && new Date(p.triage.triageTime) < new Date(filters.triageFrom)) return false;
      if (filters.triageTo && new Date(p.triage.triageTime) > new Date(filters.triageTo)) return false;
      
      const hours = getHoursDuration(p.beds.occupiedSince, p.beds.estimatedDischarge);
      if (filters.bedDurationMin && hours < parseInt(filters.bedDurationMin)) return false;
      if (filters.bedDurationMax && hours > parseInt(filters.bedDurationMax)) return false;

      if (filters.dischargeStatus !== 'all') {
        const isDischarged = p.physicianReport?.discharged || false;
        if (filters.dischargeStatus === 'discharged' && !isDischarged) return false;
        if (filters.dischargeStatus === 'pending' && isDischarged) return false;
      }
      return true;
    });

  const stats = {
    'ai-suggested': patients.filter(p => p.reportStatus === 'ai-suggested').length,
    'nurse-approved': patients.filter(p => p.reportStatus === 'nurse-approved').length,
    'physician-attended': patients.filter(p => p.reportStatus === 'physician-attended').length,
  };

  const getTabs = () => [
    { id: 'ai-suggested', label: 'AI Suggested Reports', count: stats['ai-suggested'] },
    { id: 'nurse-approved', label: 'Awaiting Physician', count: stats['nurse-approved'] },
    { id: 'physician-attended', label: 'Physician Attended', count: stats['physician-attended'] },
  ];

  // ============ RENDER ============
  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
    <div style={{ backgroundColor: '#f9fafb' }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: 700, color: '#0f172a' }}>EHR Records</h1>
        <p style={{ margin: 0, color: '#6b7280', fontSize: '14px' }}>Manage patient reports and clinical records across all stages</p>
      </div>

      {/* Alerts */}
      {successMessage && (
        <div style={{ marginBottom: 20, padding: '12px 16px', backgroundColor: '#dcfce7', border: '1px solid #86efac', borderRadius: '8px', color: '#166534', fontSize: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>{successMessage}</span>
          <button onClick={() => setSuccessMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534' }}>✕</button>
        </div>
      )}
      {errorMessage && (
        <div style={{ marginBottom: 20, padding: '12px 16px', backgroundColor: '#fee2e2', border: '1px solid #fecaca', borderRadius: '8px', color: '#991b1b', fontSize: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>{errorMessage}</span>
          <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b' }}>✕</button>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 24, borderBottom: '2px solid #e5e7eb' }}>
        {getTabs().map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 20px',
              backgroundColor: activeTab === tab.id ? '#fff' : 'transparent',
              borderBottom: activeTab === tab.id ? '3px solid #0f172a' : 'none',
              border: 'none',
              cursor: 'pointer',
              fontWeight: activeTab === tab.id ? 600 : 500,
              fontSize: '14px',
              color: activeTab === tab.id ? '#0f172a' : '#6b7280',
              position: 'relative',
              top: '2px',
            }}
          >
            {tab.label}
            <span style={{ marginLeft: 8, backgroundColor: REPORT_STATUSES[tab.id].bgColor, color: REPORT_STATUSES[tab.id].textColor, padding: '2px 6px', borderRadius: '4px', fontSize: '12px', fontWeight: 600 }}>
              {tab.count}
            </span>
          </button>
        ))}
      </div>

      {/* Search & Filter Toolbar */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ flex: 1, minWidth: '250px', display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px', backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
          <FiSearch size={18} style={{ color: '#6b7280' }} />
          <input
            type="text"
            placeholder="Search by patient name or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ flex: 1, border: 'none', outline: 'none', fontSize: '14px', backgroundColor: 'transparent' }}
          />
        </div>

        <button
          onClick={() => setShowFilters(!showFilters)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '10px 16px',
            backgroundColor: showFilters ? '#f0f9ff' : '#fff',
            color: showFilters ? '#0369a1' : '#6b7280',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 600,
            fontSize: '14px'
          }}
        >
          <FiFilter size={18} /> Filters {showFilters ? '▲' : '▼'}
        </button>
      </div>

      {/* Filters Dropdown */}
      {showFilters && (
        <div style={{ marginBottom: 24, padding: '16px', backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Triage From</label>
            <input
              type="datetime-local"
              value={filters.triageFrom}
              onChange={(e) => setFilters(prev => ({ ...prev, triageFrom: e.target.value }))}
              style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '13px', boxSizing: 'border-box' }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Triage To</label>
            <input
              type="datetime-local"
              value={filters.triageTo}
              onChange={(e) => setFilters(prev => ({ ...prev, triageTo: e.target.value }))}
              style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '13px', boxSizing: 'border-box' }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Bed Duration (hours) Min</label>
            <input
              type="number"
              value={filters.bedDurationMin}
              onChange={(e) => setFilters(prev => ({ ...prev, bedDurationMin: e.target.value }))}
              style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '13px', boxSizing: 'border-box' }}
              placeholder="0"
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Bed Duration (hours) Max</label>
            <input
              type="number"
              value={filters.bedDurationMax}
              onChange={(e) => setFilters(prev => ({ ...prev, bedDurationMax: e.target.value }))}
              style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '13px', boxSizing: 'border-box' }}
              placeholder="168"
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#374151', marginBottom: 6 }}>Discharge Status</label>
            <select
              value={filters.dischargeStatus}
              onChange={(e) => setFilters(prev => ({ ...prev, dischargeStatus: e.target.value }))}
              style={{ width: '100%', padding: '8px 10px', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '13px', boxSizing: 'border-box', cursor: 'pointer' }}
            >
              <option value="all">All Status</option>
              <option value="pending">Pending Discharge</option>
              <option value="discharged">Discharged</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <button
              onClick={() => setFilters({ triageFrom: '', triageTo: '', bedDurationMin: '', bedDurationMax: '', dischargeStatus: 'all' })}
              style={{ flex: 1, padding: '8px 12px', borderRadius: '6px', border: '1px solid #d1d5db', backgroundColor: '#fff', color: '#374151', cursor: 'pointer', fontWeight: 500, fontSize: '13px' }}
            >
              Clear Filters
            </button>
          </div>
        </div>
      )}

      {/* Patients List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#6b7280', backgroundColor: '#fff', borderRadius: '8px' }}>Loading EHR records...</div>
        ) : filteredPatients.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#6b7280', backgroundColor: '#fff', borderRadius: '8px' }}>No patients found</div>
        ) : (
          filteredPatients.map(patient => {
            const statusBadge = REPORT_STATUSES[patient.reportStatus];
            const hoursInBed = patient.beds.occupiedSince ? getHoursDuration(patient.beds.occupiedSince, patient.beds.estimatedDischarge) : 0;
            const severity = ['Low', 'Med', 'High', 'Critical', 'Resuscitation'][patient.triage.severity - 1] || 'Unknown';
            const severityColor = patient.triage.severity <= 2 ? '#10b981' : patient.triage.severity === 3 ? '#f59e0b' : '#ef4444';

            return (
              <div
                key={patient._id}
                onClick={() => { setSelectedPatient(patient); setShowDetailModal(true); }}
                style={{
                  backgroundColor: '#fff',
                  borderRadius: '12px',
                  border: '1px solid #e5e7eb',
                  padding: '16px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
                }}
                onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 10px 15px rgba(0, 0, 0, 0.1)'}
                onMouseLeave={(e) => e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)'}
              >
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr 2fr 1fr', gap: 16, alignItems: 'center' }}>
                  {/* Patient Info */}
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>
                      {patient.patient.name}
                    </p>
                    <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280' }}>
                      {patient.patient.age}y {patient.patient.gender} • {patient._id}
                    </p>
                    <span style={{ display: 'inline-block', padding: '2px 6px', borderRadius: '4px', fontSize: '11px', fontWeight: 600, color: statusBadge.textColor, backgroundColor: statusBadge.bgColor }}>
                      {statusBadge.label}
                    </span>
                  </div>

                  {/* Chief Complaint & Timeline */}
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                      {patient.triage.chief_complaint}
                    </p>
                    <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280' }}>
                      {patient.triage.assigned_specialty}
                    </p>
                    <p style={{ margin: 0, fontSize: '11px', color: '#6b7280' }}>
                      <FiClock size={12} style={{ display: 'inline', marginRight: 4 }} />
                      Triaged: {formatDate(patient.triage.triageTime)}
                    </p>
                  </div>

                  {/* Bed & Duration */}
                  <div>
                    {patient.beds.assigned ? (
                      <>
                        <p style={{ margin: '0 0 4px 0', fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                          <FaBed size={13} style={{ display: 'inline', marginRight: 4 }} />
                          {patient.beds.assigned} ({patient.beds.type})
                        </p>
                        <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#6b7280' }}>
                          Duration: {hoursInBed} hours
                        </p>
                        <p style={{ margin: 0, fontSize: '11px', color: '#6b7280' }}>
                          Est. Discharge: {formatDate(patient.beds.estimatedDischarge)}
                        </p>
                      </>
                    ) : (
                      <p style={{ margin: 0, fontSize: '12px', color: '#6b7280', fontStyle: 'italic' }}>No bed assigned</p>
                    )}
                  </div>

                  {/* Severity & Status */}
                  <div style={{ textAlign: 'right' }}>
                    <span style={{
                      display: 'inline-block',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      fontSize: '12px',
                      fontWeight: 700,
                      color: '#fff',
                      backgroundColor: severityColor
                    }}>
                      {severity}
                    </span>
                    {patient.physicianReport?.discharged && (
                      <p style={{ margin: '8px 0 0 0', fontSize: '11px', fontWeight: 600, color: '#10b981' }}>✓ Discharged</p>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Detail Modal */}
      {showDetailModal && selectedPatient && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50, overflowY: 'auto', padding: '20px' }}>
          <div style={{ backgroundColor: '#fff', borderRadius: '12px', width: '90%', maxWidth: '900px', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)', maxHeight: '90vh', overflowY: 'auto' }}>
            {/* Modal Header */}
            <div style={{ padding: '20px 24px', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#f9fafb', position: 'sticky', top: 0 }}>
              <div>
                <h2 style={{ margin: '0 0 4px 0', fontSize: '20px', fontWeight: 700, color: '#0f172a' }}>
                  {selectedPatient.patient.name}
                </h2>
                <p style={{ margin: 0, fontSize: '13px', color: '#6b7280' }}>
                  {selectedPatient.patient.age}y {selectedPatient.patient.gender} • ID: {selectedPatient._id}
                </p>
              </div>
              <button onClick={() => setShowDetailModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: '24px' }}>
                <FiX />
              </button>
            </div>

            {/* Modal Body */}
            <div style={{ padding: '24px' }}>
              {/* Triage Info */}
              <div style={{ marginBottom: 24, padding: '16px', backgroundColor: '#f9fafb', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
                <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>Triage Information</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#6b7280' }}>Chief Complaint</p>
                    <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>{selectedPatient.triage.chief_complaint}</p>
                  </div>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#6b7280' }}>Specialty</p>
                    <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>{selectedPatient.triage.assigned_specialty}</p>
                  </div>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#6b7280' }}>Severity</p>
                    <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#ef4444' }}>
                      {['Low', 'Med', 'High', 'Critical', 'Resuscitation'][selectedPatient.triage.severity - 1]}
                    </p>
                  </div>
                  <div>
                    <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#6b7280' }}>Triage Time</p>
                    <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>{formatDate(selectedPatient.triage.triageTime)}</p>
                  </div>
                </div>
              </div>

              {/* AI Suggested Report */}
              {selectedPatient.aiSuggestedReport && (
                <div style={{ marginBottom: 24, padding: '16px', backgroundColor: '#fef3c7', borderRadius: '8px', border: '1px solid #fcd34d' }}>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', fontWeight: 700, color: '#92400e' }}>AI Suggested Report</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#b45309' }}>Diagnosis</p>
                      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>{selectedPatient.aiSuggestedReport.diagnosis}</p>
                    </div>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#b45309' }}>Confidence</p>
                      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>
                        {Math.round(selectedPatient.aiSuggestedReport.confidence * 100)}%
                      </p>
                    </div>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#b45309' }}>Recommendations</p>
                      <p style={{ margin: 0, fontSize: '13px', color: '#0f172a' }}>
                        {selectedPatient.aiSuggestedReport.recommendations.join(', ')}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Bed & Resources */}
              {selectedPatient.beds.assigned && (
                <div style={{ marginBottom: 24, padding: '16px', backgroundColor: '#f0f9ff', borderRadius: '8px', border: '1px solid #bfdbfe' }}>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', fontWeight: 700, color: '#0c4a6e' }}>Bed Allocation</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#0c4a6e' }}>Bed</p>
                      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>{selectedPatient.beds.assigned} ({selectedPatient.beds.type})</p>
                    </div>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#0c4a6e' }}>Duration</p>
                      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>
                        {getHoursDuration(selectedPatient.beds.occupiedSince, selectedPatient.beds.estimatedDischarge)} hours
                      </p>
                    </div>
                    <div>
                      <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#0c4a6e' }}>Est. Discharge</p>
                      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>
                        {formatDate(selectedPatient.beds.estimatedDischarge)}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Physician Report */}
              {selectedPatient.physicianReport && (
                <div style={{ marginBottom: 24, padding: '16px', backgroundColor: '#dcfce7', borderRadius: '8px', border: '1px solid #86efac' }}>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', fontWeight: 700, color: '#166534' }}>Physician Report</h3>
                  
                  <div style={{ marginBottom: 16 }}>
                    <p style={{ margin: '0 0 4px 0', fontSize: '12px', fontWeight: 600, color: '#166534' }}>Physician</p>
                    <p style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>{selectedPatient.physicianReport.physicianName}</p>
                    <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>Attended: {formatDate(selectedPatient.physicianReport.attendedAt)}</p>
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: 600, color: '#166534' }}>Final Diagnosis</p>
                    <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>{selectedPatient.physicianReport.finalDiagnosis.join(', ')}</p>
                  </div>

                  {/* Lab Results */}
                  {selectedPatient.physicianReport.labResults && selectedPatient.physicianReport.labResults.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                      <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: 600, color: '#166534', display: 'flex', alignItems: 'center', gap: 4 }}>
                        <FaFlask size={12} /> Lab Results
                      </p>
                      {selectedPatient.physicianReport.labResults.map((result, idx) => (
                        <div key={idx} style={{ padding: '8px 0', borderBottom: idx < selectedPatient.physicianReport.labResults.length - 1 ? '1px solid #c7f0d8' : 'none', fontSize: '13px', color: '#0f172a' }}>
                          <strong>{result.test}</strong>: {result.value} {result.unit}
                          <span style={{ marginLeft: 8, fontSize: '11px', padding: '2px 4px', borderRadius: '3px', backgroundColor: result.status === 'normal' ? '#dcfce7' : '#fee2e2', color: result.status === 'normal' ? '#166534' : '#991b1b', fontWeight: 600 }}>
                            {result.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Scans */}
                  {selectedPatient.physicianReport.scans && selectedPatient.physicianReport.scans.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                      <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: 600, color: '#166534', display: 'flex', alignItems: 'center', gap: 4 }}>
                        <FaImage size={12} /> Scans
                      </p>
                      {selectedPatient.physicianReport.scans.map((scan, idx) => (
                        <div key={idx} style={{ padding: '8px 0', borderBottom: idx < selectedPatient.physicianReport.scans.length - 1 ? '1px solid #c7f0d8' : 'none', fontSize: '13px', color: '#0f172a' }}>
                          <strong>{scan.type}</strong>: {scan.findings}
                          <br />
                          <span style={{ fontSize: '11px', color: '#6b7280' }}>Date: {formatDate(scan.date)}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Resources */}
                  {selectedPatient.physicianReport.resources && selectedPatient.physicianReport.resources.length > 0 && (
                    <div>
                      <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: 600, color: '#166534', display: 'flex', alignItems: 'center', gap: 4 }}>
                        <FaToolbox size={12} /> Resources
                      </p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {selectedPatient.physicianReport.resources.map((resource, idx) => (
                          <span key={idx} style={{ padding: '4px 8px', borderRadius: '4px', backgroundColor: '#c7f0d8', color: '#166534', fontSize: '12px', fontWeight: 500 }}>
                            {resource}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Discharge Status */}
                  {selectedPatient.physicianReport.discharged && (
                    <div style={{ marginTop: 16, padding: '12px', backgroundColor: '#10b981', borderRadius: '6px', color: '#fff', fontSize: '13px', fontWeight: 600 }}>
                      ✓ Patient Discharged on {formatDate(selectedPatient.physicianReport.dischargeDate)}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div style={{ padding: '16px 24px', borderTop: '1px solid #e5e7eb', display: 'flex', justifyContent: 'flex-end', gap: 12, backgroundColor: '#f9fafb' }}>
              <button onClick={() => setShowDetailModal(false)} style={{ padding: '10px 16px', borderRadius: '8px', border: '1px solid #d1d5db', backgroundColor: '#fff', color: '#374151', cursor: 'pointer', fontWeight: 600, fontSize: '14px' }}>
                Close
              </button>
              <button style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '10px 16px', borderRadius: '8px', border: 'none', backgroundColor: '#0f172a', color: '#fff', cursor: 'pointer', fontWeight: 600, fontSize: '14px' }}>
                <FiDownload size={14} /> Download Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
    </SidebarLayout>
  );
};

export default EHRRecords;
