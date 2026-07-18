import React, { useState, useEffect } from 'react';
import { FiSearch, FiFileText, FiChevronDown, FiChevronUp } from 'react-icons/fi';

const PatientLookup = () => {
  const [searchMRN, setSearchMRN] = useState('');
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [expandedEncounter, setExpandedEncounter] = useState(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

  // Search for patients in hospital
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchMRN.trim()) {
      alert('Please enter a patient MRN or name');
      return;
    }

    try {
      setLoading(true);
      const res = await fetch(
        `${BACKEND_URL}/api/patients/search?search=${encodeURIComponent(searchMRN)}`,
        {
          headers: { 
            Authorization: `Bearer ${token}`,
            'Cache-Control': 'no-cache'
          }
        }
      );

      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      setPatients(data.encounters || []);
      
      if (data.encounters.length === 0) {
        alert('No patients found matching your search');
      }
    } catch (err) {
      console.error('Search error:', err);
      alert('Error searching patients: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Get patient EHR (all encounters)
  const handleViewEHR = async (mrn) => {
    try {
      setLoading(true);
      const res = await fetch(
        `${BACKEND_URL}/api/patients/ehr/${mrn}`,
        {
          headers: { 
            Authorization: `Bearer ${token}`,
            'Cache-Control': 'no-cache'
          }
        }
      );

      if (!res.ok) throw new Error('EHR fetch failed');
      const data = await res.json();
      setSelectedPatient(data);
    } catch (err) {
      console.error('EHR error:', err);
      alert('Error retrieving EHR: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Toggle encounter details
  const toggleEncounter = (encId) => {
    setExpandedEncounter(expandedEncounter === encId ? null : encId);
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20 }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
            Patient EHR Lookup
          </h1>
          <p style={{ margin: 0, fontSize: 14, color: '#64748b' }}>
            Search and view complete electronic health records for patients in your hospital
          </p>
        </div>

        {/* Search Form */}
        <form onSubmit={handleSearch} style={{ marginBottom: 32 }}>
          <div style={{ display: 'flex', gap: 12, background: 'white', padding: 16, borderRadius: 8, border: '1px solid #e2e8f0' }}>
            <input
              type="text"
              placeholder="Enter Patient MRN (e.g., MRN-99124) or Name"
              value={searchMRN}
              onChange={(e) => setSearchMRN(e.target.value)}
              style={{
                flex: 1,
                padding: '12px 16px',
                border: '1px solid #cbd5e1',
                borderRadius: 6,
                fontSize: 14,
                outline: 'none'
              }}
            />
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: '12px 28px',
                background: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: 600,
                fontSize: 14,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                opacity: loading ? 0.7 : 1
              }}
            >
              <FiSearch size={18} /> {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
        </form>

        {/* Search Results - Initial Patient List */}
        {patients.length > 0 && !selectedPatient && (
          <div style={{ background: 'white', padding: 20, borderRadius: 8, marginBottom: 24, border: '1px solid #e2e8f0' }}>
            <h2 style={{ margin: '0 0 16px 0', fontSize: 18, fontWeight: 600, color: '#0f172a' }}>
              Search Results: {patients.length} Encounter(s) Found
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
              {patients.map((enc) => (
                <div
                  key={enc._id}
                  style={{
                    padding: 16,
                    background: '#f8fafc',
                    borderRadius: 6,
                    border: '1px solid #e2e8f0',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                  }}
                >
                  <div style={{ marginBottom: 8 }}>
                    <strong style={{ fontSize: 14, color: '#0f172a' }}>
                      {enc.patient?.name} {enc.patient?.surname}
                    </strong>
                  </div>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 12, lineHeight: 1.6 }}>
                    <div><span style={{ fontWeight: 600 }}>MRN:</span> {enc.patient?.id_number}</div>
                    <div><span style={{ fontWeight: 600 }}>Status:</span> {enc.status}</div>
                    <div><span style={{ fontWeight: 600 }}>Priority:</span> {enc.validated_effective_priority || 'N/A'}</div>
                    <div><span style={{ fontWeight: 600 }}>Date:</span> {new Date(enc.createdAt).toLocaleDateString()}</div>
                  </div>
                  <button
                    onClick={() => handleViewEHR(enc.patient?.id_number)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: 4,
                      cursor: 'pointer',
                      fontSize: 12,
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 6
                    }}
                  >
                    <FiFileText size={14} /> View Complete EHR
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Selected Patient EHR Display */}
        {selectedPatient && (
          <div style={{ background: 'white', padding: 24, borderRadius: 8, border: '1px solid #e2e8f0' }}>
            {/* EHR Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24, paddingBottom: 16, borderBottom: '1px solid #e2e8f0' }}>
              <div>
                <h2 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#0f172a' }}>
                  Complete EHR - {selectedPatient.mrn}
                </h2>
                <p style={{ margin: '8px 0 0 0', fontSize: 13, color: '#64748b' }}>
                  Hospital: {selectedPatient.hospital?.name} | Total Encounters: {selectedPatient.totalEncounters}
                </p>
              </div>
              <button
                onClick={() => setSelectedPatient(null)}
                style={{
                  padding: '8px 16px',
                  background: '#e2e8f0',
                  color: '#1e293b',
                  border: 'none',
                  borderRadius: 6,
                  cursor: 'pointer',
                  fontSize: 13,
                  fontWeight: 600
                }}
              >
                ← Back to Search
              </button>
            </div>

            {/* Encounters Timeline */}
            <div style={{ marginTop: 24 }}>
              {selectedPatient.encounters && selectedPatient.encounters.map((enc, idx) => (
                <div
                  key={enc._id}
                  style={{
                    marginBottom: 16,
                    padding: 16,
                    background: '#f1f5f9',
                    borderRadius: 6,
                    borderLeft: '4px solid #3b82f6',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onClick={() => toggleEncounter(enc._id)}
                >
                  {/* Encounter Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <strong style={{ fontSize: 14, color: '#0f172a' }}>
                        Encounter {idx + 1} - {enc.status}
                      </strong>
                      <span style={{ marginLeft: 12, fontSize: 12, color: '#64748b' }}>
                        {new Date(enc.createdAt).toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span
                        style={{
                          display: 'inline-block',
                          padding: '4px 12px',
                          background: enc.validated_effective_priority === 'RED' ? '#fee2e2' :
                                     enc.validated_effective_priority === 'ORANGE' ? '#fed7aa' :
                                     enc.validated_effective_priority === 'YELLOW' ? '#fef3c7' :
                                     '#dcfce7',
                          color: enc.validated_effective_priority === 'RED' ? '#991b1b' :
                                enc.validated_effective_priority === 'ORANGE' ? '#92400e' :
                                enc.validated_effective_priority === 'YELLOW' ? '#78350f' :
                                '#166534',
                          borderRadius: 4,
                          fontSize: 12,
                          fontWeight: 600,
                          marginRight: 12
                        }}
                      >
                        Priority: {enc.validated_effective_priority || 'N/A'}
                      </span>
                      {expandedEncounter === enc._id ? <FiChevronUp size={18} /> : <FiChevronDown size={18} />}
                    </div>
                  </div>

                  {/* Expanded Encounter Details */}
                  {expandedEncounter === enc._id && (
                    <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #e2e8f0' }}>
                      {/* Triage Nurse Info */}
                      {enc.original_triage_nurse && (
                        <div style={{ marginBottom: 16, padding: 12, background: 'white', borderRadius: 4, borderLeft: '3px solid #0284c7' }}>
                          <strong style={{ fontSize: 12, color: '#0284c7' }}>Triage Nurse</strong>
                          <div style={{ fontSize: 12, color: '#1e293b', marginTop: 6 }}>
                            {enc.original_triage_nurse.name || enc.original_triage_nurse.nurse_name}<br />
                            <span style={{ color: '#64748b', fontSize: 11 }}>
                              {enc.original_triage_nurse.email || enc.original_triage_nurse.nurse_email}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Vitals */}
                      {enc.vitals && (
                        <div style={{ marginBottom: 16, padding: 12, background: 'white', borderRadius: 4 }}>
                          <strong style={{ fontSize: 12, color: '#0f172a' }}>Vital Signs</strong>
                          <div style={{ fontSize: 12, color: '#1e293b', marginTop: 6, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                            {enc.vitals.temp && <div>Temp: {enc.vitals.temp}°C</div>}
                            {enc.vitals.hr && <div>HR: {enc.vitals.hr} bpm</div>}
                            {enc.vitals.o2 && <div>O₂: {enc.vitals.o2}%</div>}
                            {enc.vitals.bp_systolic && <div>BP: {enc.vitals.bp_systolic}/{enc.vitals.bp_diastolic} mmHg</div>}
                          </div>
                        </div>
                      )}

                      {/* AI Triage Report Summary */}
                      {enc.ai_triage_report && (
                        <div style={{ marginBottom: 16, padding: 12, background: '#dbeafe', borderRadius: 4, borderLeft: '3px solid #3b82f6' }}>
                          <strong style={{ fontSize: 12, color: '#1e40af' }}>AI Triage Report</strong>
                          <div style={{ fontSize: 12, color: '#1e293b', marginTop: 6, lineHeight: 1.5 }}>
                            {enc.ai_triage_report.clinical_summary || 'Report generated'}
                          </div>
                        </div>
                      )}

                      {/* Clinical Handovers */}
                      {enc.clinical_handovers && enc.clinical_handovers.length > 0 && (
                        <div style={{ marginBottom: 16, padding: 12, background: 'white', borderRadius: 4, borderLeft: '3px solid #10b981' }}>
                          <strong style={{ fontSize: 12, color: '#166534' }}>Clinical Handovers ({enc.clinical_handovers.length})</strong>
                          <ul style={{ fontSize: 12, color: '#1e293b', marginTop: 8, marginLeft: 16, marginBottom: 0 }}>
                            {enc.clinical_handovers.map((ho, i) => (
                              <li key={i} style={{ marginBottom: 6 }}>
                                <strong>{ho.clinician_name}</strong> ({ho.clinician_role})<br />
                                <span style={{ color: '#64748b', fontSize: 11 }}>
                                  Status: {ho.handover_status} • Sent: {new Date(ho.handover_at).toLocaleString()}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Chief Complaint & Patient Info */}
                      {enc.patient && (
                        <div style={{ marginBottom: 16, padding: 12, background: 'white', borderRadius: 4 }}>
                          <div style={{ fontSize: 12, color: '#1e293b' }}>
                            <div><strong>Chief Complaint:</strong> {enc.patient.symptoms || 'N/A'}</div>
                            <div style={{ marginTop: 6 }}><strong>Age:</strong> {enc.patient.age} • <strong>Gender:</strong> {enc.patient.gender}</div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {patients.length === 0 && !selectedPatient && !loading && (
          <div style={{ textAlign: 'center', padding: 60, color: '#64748b' }}>
            <FiSearch size={48} style={{ marginBottom: 16, opacity: 0.3 }} />
            <p style={{ fontSize: 16 }}>Search for a patient to view their EHR</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientLookup;
