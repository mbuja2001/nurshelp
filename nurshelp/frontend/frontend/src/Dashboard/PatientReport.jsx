import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiChevronRight, FiPlus, FiFileText, FiUsers, FiBox, FiCheckCircle } from 'react-icons/fi';

const PatientReport = () => {
  const navigate = useNavigate();
  const [userRole, setUserRole] = useState(null);
  const [hospitalId, setHospitalId] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

  // JWT Decode to extract role and hospital
  const decodeJWT = (jwtToken) => {
    try {
      if (!jwtToken) return null;
      const parts = jwtToken.split('.');
      if (parts.length !== 3) return null;
      
      const decoded = JSON.parse(atob(parts[1]));
      return {
        role: decoded.role,
        hospitalId: decoded.hospitalId,
        id: decoded.id,
        name: decoded.name
      };
    } catch (err) {
      console.error('[PatientReport] JWT Decode Error:', err);
      return null;
    }
  };

  // Initialize user context from JWT
  useEffect(() => {
    const user = decodeJWT(token);
    if (user) {
      setUserRole(user.role);
      setHospitalId(user.hospitalId);
      console.log('[PatientReport] User initialized:', {
        role: user.role,
        hospitalId: user.hospitalId,
        name: user.name
      });
    }
  }, [token]);

  // Fetch patients from this hospital
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        if (!hospitalId || !token) return;
        
        setLoading(true);
        const headers = {
          Authorization: `Bearer ${token}`,
          'Cache-Control': 'no-cache'
        };

        const res = await fetch(
          `${BACKEND_URL}/api/patients?hospital_id=${hospitalId}&limit=20`,
          { headers, cache: 'no-store' }
        );

        if (!res.ok) throw new Error('Failed to fetch patients');
        const data = await res.json();
        setPatients(data.patients || []);
        console.log('[PatientReport] Fetched patients:', data.patients?.length || 0);
      } catch (err) {
        console.error('[PatientReport] Error fetching patients:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchPatients();
  }, [hospitalId, token, BACKEND_URL]);

  // Filter patients by search
  const filteredPatients = patients.filter(p =>
    searchTerm === '' ? true : (
      p.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.firstName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.surname?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.id_number?.includes(searchTerm)
    )
  );

  // Check if user can access a category
  const canAccessTriageReports = ['NURSE', 'PHYSICIAN'].includes(userRole);
  const canAccessOutcome = userRole === 'PHYSICIAN';
  const canAccessResourceUtils = ['PHYSICIAN', 'MANAGER'].includes(userRole);
  const canAccessAudit = userRole === 'MANAGER';

  // Category cards
  const categories = [
    {
      id: 'triage',
      title: 'Triage Reports',
      icon: FiFileText,
      description: 'AI, Manual, or Combined Triage Reports',
      color: 'bg-blue-50',
      borderColor: 'border-blue-200',
      accessible: canAccessTriageReports,
      roles: 'Nurse, Physician',
      actions: [
        { label: 'AI Triage', path: '/pending-triage', icon: '🤖' },
        { label: 'Manual Triage', path: '/manual-triage', icon: '👨‍⚕️' },
        { label: 'Combined (Both)', path: '/combined-triage', icon: '⚙️' }
      ]
    },
    {
      id: 'outcome',
      title: 'Encounter Outcome',
      icon: FiCheckCircle,
      description: 'Physician Diagnosis & Disposition',
      color: 'bg-green-50',
      borderColor: 'border-green-200',
      accessible: canAccessOutcome,
      roles: 'Physician',
      actions: [
        { label: 'View Outcomes', path: '/physician-outcomes', icon: '📋' },
        { label: 'Create New', path: '/new-outcome', icon: '➕' }
      ]
    },
    {
      id: 'resources',
      title: 'Resource Allocation',
      icon: FiBox,
      description: 'Ward/Room/Bed Assignments & Equipment',
      color: 'bg-amber-50',
      borderColor: 'border-amber-200',
      accessible: canAccessResourceUtils,
      roles: 'Physician, Manager',
      actions: [
        { label: 'View Resources', path: '/resource-allocation', icon: '🛏️' }
      ]
    },
    {
      id: 'audit',
      title: 'Closing Audit',
      icon: FiUsers,
      description: 'System Variance & Clinical Sign-off',
      color: 'bg-purple-50',
      borderColor: 'border-purple-200',
      accessible: canAccessAudit,
      roles: 'Manager',
      actions: [
        { label: 'Review Audits', path: '/audit-reports', icon: '✓' }
      ]
    }
  ];

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 24 }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
            EHR Patient Reports
          </h1>
          <p style={{ margin: 0, fontSize: 14, color: '#64748b' }}>
            Manage clinical handover, triage reports, and complete EHR records - Hospital: {hospitalId || 'Loading...'}
          </p>
        </div>

        {/* Patient Search Section */}
        {!selectedPatient && (
          <div style={{ background: 'white', padding: 20, borderRadius: 8, marginBottom: 24, border: '1px solid #e2e8f0' }}>
            <h2 style={{ margin: '0 0 16px 0', fontSize: 16, fontWeight: 600, color: '#0f172a' }}>
              Select Patient
            </h2>
            <input
              type="text"
              placeholder="Search by name or ID number..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: '100%',
                padding: '12px 16px',
                border: '1px solid #e2e8f0',
                borderRadius: 6,
                fontSize: 14,
                marginBottom: 16
              }}
            />

            {loading ? (
              <div style={{ textAlign: 'center', padding: 20, color: '#64748b' }}>Loading patients...</div>
            ) : filteredPatients.length === 0 ? (
              <div style={{ textAlign: 'center', padding: 20, color: '#94a3b8' }}>
                No patients found{searchTerm && ` matching "${searchTerm}"`}
              </div>
            ) : (
              <div style={{ display: 'grid', gap: 8 }}>
                {filteredPatients.slice(0, 10).map(patient => (
                  <div
                    key={patient._id}
                    onClick={() => setSelectedPatient(patient)}
                    style={{
                      padding: 12,
                      background: '#f8fafc',
                      borderRadius: 6,
                      border: '1px solid #e2e8f0',
                      cursor: 'pointer',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      transition: 'all 0.2s'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.background = '#f1f5f9';
                      e.currentTarget.style.borderColor = '#cbd5e1';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.background = '#f8fafc';
                      e.currentTarget.style.borderColor = '#e2e8f0';
                    }}
                  >
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>
                        {patient.firstName} {patient.surname}
                      </div>
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                        ID: {patient.id_number} | Age: {patient.age} | {patient.gender}
                      </div>
                    </div>
                    <FiChevronRight size={18} color="#94a3b8" />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Selected Patient Context */}
        {selectedPatient && (
          <div style={{ background: '#eff6ff', padding: 16, borderRadius: 8, marginBottom: 24, border: '1px solid #bfdbfe', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#1e40af' }}>
                Selected: {selectedPatient.firstName} {selectedPatient.surname}
              </div>
              <div style={{ fontSize: 12, color: '#0369a1', marginTop: 4 }}>
                MRN: {selectedPatient.id_number}
              </div>
            </div>
            <button
              onClick={() => {
                setSelectedPatient(null);
                setSearchTerm('');
              }}
              style={{
                padding: '8px 16px',
                background: '#dbeafe',
                color: '#0284c7',
                border: '1px solid #bfdbfe',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: 600
              }}
            >
              Change Patient
            </button>
          </div>
        )}

        {/* Categories Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: selectedPatient ? '1fr 1fr' : '1fr', gap: 24 }}>
          <div style={{ display: 'grid', gap: 16 }}>
            {categories.map((category) => {
              const IconComponent = category.icon;
              const isAccessible = selectedPatient && category.accessible;

              return (
                <div
                  key={category.id}
                  style={{
                    background: 'white',
                    padding: 20,
                    borderRadius: 8,
                    border: `2px solid ${category.borderColor.replace('border-', '')}`,
                    opacity: category.accessible ? 1 : 0.6
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                    <div style={{ fontSize: 24, opacity: 0.7 }}>
                      <IconComponent size={24} />
                    </div>
                    <div>
                      <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: '#0f172a' }}>
                        {category.title}
                      </h3>
                      <p style={{ margin: '4px 0 0 0', fontSize: 12, color: '#64748b' }}>
                        {category.description}
                      </p>
                    </div>
                  </div>

                  {!category.accessible && (
                    <div style={{ fontSize: 12, color: '#ce7777', marginBottom: 12, padding: 8, background: '#fee2e2', borderRadius: 4 }}>
                      Requires role: {category.roles}
                    </div>
                  )}

                  {category.accessible && !selectedPatient && (
                    <div style={{ fontSize: 12, color: '#94a3b8' }}>
                      Select a patient to access this category
                    </div>
                  )}

                  {isAccessible && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {category.actions.map((action, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            // Pass patient info via params or context
                            navigate(action.path, {
                              state: {
                                patientId: selectedPatient._id,
                                patientName: `${selectedPatient.firstName} ${selectedPatient.surname}`,
                                mrn: selectedPatient.id_number,
                                hospitalId: hospitalId
                              }
                            });
                          }}
                          style={{
                            padding: '10px 12px',
                            background: '#f8fafc',
                            border: '1px solid #e2e8f0',
                            borderRadius: 4,
                            cursor: 'pointer',
                            fontSize: 13,
                            fontWeight: 500,
                            color: '#0f172a',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 8,
                            transition: 'all 0.2s'
                          }}
                          onMouseOver={(e) => {
                            e.currentTarget.style.background = '#e2e8f0';
                            e.currentTarget.style.borderColor = '#cbd5e1';
                          }}
                          onMouseOut={(e) => {
                            e.currentTarget.style.background = '#f8fafc';
                            e.currentTarget.style.borderColor = '#e2e8f0';
                          }}
                        >
                          <span>{action.icon}</span>
                          {action.label}
                          <FiChevronRight size={14} style={{ marginLeft: 'auto' }} />
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Info Panel */}
          {selectedPatient && (
            <div style={{ display: 'grid', gap: 16 }}>
              {/* Patient Summary */}
              <div style={{ background: 'white', padding: 20, borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 600, color: '#0f172a' }}>Patient Summary</h3>
                <div style={{ display: 'grid', gap: 12, fontSize: 13 }}>
                  <div>
                    <span style={{ color: '#64748b', fontWeight: 600 }}>Full Name:</span>
                    <div style={{ color: '#1e293b', marginTop: 2 }}>{selectedPatient.firstName} {selectedPatient.surname}</div>
                  </div>
                  <div>
                    <span style={{ color: '#64748b', fontWeight: 600 }}>MRN:</span>
                    <div style={{ color: '#1e293b', marginTop: 2 }}>{selectedPatient.id_number}</div>
                  </div>
                  <div>
                    <span style={{ color: '#64748b', fontWeight: 600 }}>Age/Gender:</span>
                    <div style={{ color: '#1e293b', marginTop: 2 }}>{selectedPatient.age} / {selectedPatient.gender}</div>
                  </div>
                  <div>
                    <span style={{ color: '#64748b', fontWeight: 600 }}>Chief Complaint:</span>
                    <div style={{ color: '#1e293b', marginTop: 2 }}>{selectedPatient.symptoms || 'N/A'}</div>
                  </div>
                </div>
              </div>

              {/* Your Role & Access */}
              <div style={{ background: 'white', padding: 20, borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 600, color: '#0f172a' }}>Your Access</h3>
                <div style={{ display: 'grid', gap: 8, fontSize: 13 }}>
                  <div style={{ padding: 8, background: '#eff6ff', borderRadius: 4, borderLeft: '4px solid #3b82f6' }}>
                    <div style={{ fontWeight: 600, color: '#1e40af' }}>Role: {userRole || 'Unknown'}</div>
                  </div>
                  {canAccessTriageReports && (
                    <div style={{ padding: 8, background: '#f0fdf4', borderRadius: 4, borderLeft: '4px solid #22c55e' }}>
                      ✓ Can create/view Triage Reports
                    </div>
                  )}
                  {canAccessOutcome && (
                    <div style={{ padding: 8, background: '#f0fdf4', borderRadius: 4, borderLeft: '4px solid #22c55e' }}>
                      ✓ Can create Encounter Outcome
                    </div>
                  )}
                  {canAccessResourceUtils && (
                    <div style={{ padding: 8, background: '#f0fdf4', borderRadius: 4, borderLeft: '4px solid #22c55e' }}>
                      ✓ Can view Resource Allocation
                    </div>
                  )}
                  {canAccessAudit && (
                    <div style={{ padding: 8, background: '#f0fdf4', borderRadius: 4, borderLeft: '4px solid #22c55e' }}>
                      ✓ Can review Audit Reports
                    </div>
                  )}
                  {!canAccessTriageReports && !canAccessOutcome && !canAccessResourceUtils && !canAccessAudit && (
                    <div style={{ padding: 8, background: '#fee2e2', borderRadius: 4, borderLeft: '4px solid #ef4444' }}>
                      ⚠ No access to patient reports
                    </div>
                  )}
                </div>
              </div>

              {/* EHR Info */}
              <div style={{ background: 'white', padding: 20, borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <h3 style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 600, color: '#0f172a' }}>EHR System Info</h3>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>
                  <p>The EHR is divided into 6 categories:</p>
                  <ol style={{ margin: '8px 0', paddingLeft: 20 }}>
                    <li>Clinical Intake</li>
                    <li>Triage Decision Reports</li>
                    <li>Encounter Outcome</li>
                    <li>Resource Utilisation</li>
                    <li>Medication Administration</li>
                    <li>Closing Audit</li>
                  </ol>
                  <p style={{ marginTop: 8 }}>Each category is maintained for audit and continuity of care.</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Back Button */}
        <div style={{ marginTop: 32 }}>
          <button
            onClick={() => navigate('/dashboard')}
            style={{
              padding: '10px 24px',
              background: '#f3f4f6',
              color: '#6b7280',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600
            }}
          >
            ← Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};

export default PatientReport;
