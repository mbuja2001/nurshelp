import React, { useState } from 'react';
import { FiSearch, FiFilter, FiX, FiChevronRight, FiHeart, FiTrendingUp } from 'react-icons/fi';
import { MdLibraryBooks } from 'react-icons/md';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';

const ReferenceCases = () => {
  const { userName } = useUserFromToken();
  const [searchMode, setSearchMode] = useState('symptoms'); // 'symptoms', 'vitals', 'combined'
  const [symptoms, setSymptoms] = useState('');
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [vitals, setVitals] = useState({
    bloodPressure: '',
    heartRate: '',
    temperature: '',
    respiratoryRate: '',
    oxygenSaturation: ''
  });
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState([]);

  const commonSymptoms = [
    'Chest Pain', 'Shortness of Breath', 'Dizziness', 'Headache',
    'Abdominal Pain', 'Fever', 'Cough', 'Nausea', 'Vomiting',
    'Weakness', 'Fatigue', 'Confusion'
  ];

  const toggleSymptom = (symptom) => {
    if (selectedSymptoms.includes(symptom)) {
      setSelectedSymptoms(selectedSymptoms.filter(s => s !== symptom));
    } else {
      setSelectedSymptoms([...selectedSymptoms, symptom]);
    }
  };

  const handleSearch = () => {
    // Mock search - will integrate with RAG system
    const mockResults = [
      {
        id: 'case_001',
        patientId: 'P-2024-0891',
        age: 54,
        gender: 'M',
        primaryDiagnosis: 'Acute Coronary Syndrome',
        symptoms: ['Chest Pain', 'Shortness of Breath', 'Diaphoresis'],
        vitals: { bp: '156/95', hr: 112, temp: 37.2, rr: 22, o2: 94 },
        source: 'PMC-Case-12453',
        similarity: 94,
        outcomeStatus: 'Admitted - Cardiology',
        timeToTriage: '8 minutes'
      },
      {
        id: 'case_002',
        patientId: 'P-2024-0892',
        age: 67,
        gender: 'F',
        primaryDiagnosis: 'Unstable Angina',
        symptoms: ['Chest Pain', 'Anxiety', 'Shortness of Breath'],
        vitals: { bp: '142/88', hr: 98, temp: 37.1, rr: 20, o2: 96 },
        source: 'PMC-Case-12454',
        similarity: 87,
        outcomeStatus: 'Admitted - Cardiology',
        timeToTriage: '12 minutes'
      },
      {
        id: 'case_003',
        patientId: 'P-2024-0893',
        age: 48,
        gender: 'M',
        primaryDiagnosis: 'Musculoskeletal Chest Pain',
        symptoms: ['Chest Pain', 'Muscle Tension', 'Anxiety'],
        vitals: { bp: '128/82', hr: 88, temp: 36.9, rr: 18, o2: 98 },
        source: 'PMC-Case-12455',
        similarity: 78,
        outcomeStatus: 'Discharged - Home',
        timeToTriage: '15 minutes'
      },
    ];

    setResults(mockResults);
    setShowResults(true);
  };

  const CaseResultCard = ({ caseData }) => {
    const getSimilarityColor = (similarity) => {
      if (similarity >= 90) return '#10b981'; // green
      if (similarity >= 80) return '#f59e0b'; // amber
      return '#6b7280'; // gray
    };

    const getOutcomeColor = (status) => {
      if (status.includes('Admitted')) return '#0369a1'; // blue
      if (status.includes('Discharged')) return '#10b981'; // green
      return '#6b7280'; // gray
    };

    return (
      <div style={{
        padding: 24,
        backgroundColor: 'white',
        border: '1px solid #e5e7eb',
        borderRadius: 12,
        marginBottom: 16,
        transition: 'all 200ms ease',
        cursor: 'pointer',
        boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#10b981';
        e.currentTarget.style.boxShadow = '0 4px 12px rgba(16, 185, 129, 0.1)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#e5e7eb';
        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.05)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <h4 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#0f172a' }}>
                {caseData.primaryDiagnosis}
              </h4>
              <span style={{
                padding: '4px 8px',
                backgroundColor: '#f0fdf4',
                color: '#166534',
                borderRadius: 4,
                fontSize: 12,
                fontWeight: 600
              }}>
                Similar
              </span>
            </div>
            <p style={{ margin: '0 0 8px 0', fontSize: 13, color: '#6b7280' }}>
              Patient: {caseData.age}M, {caseData.gender} • Source: {caseData.source}
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              fontSize: 24,
              fontWeight: 700,
              color: getSimilarityColor(caseData.similarity),
              marginBottom: 4
            }}>
              {caseData.similarity}%
            </div>
            <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600 }}>Match Score</div>
          </div>
        </div>

        {/* Symptoms */}
        <div style={{ marginBottom: 16 }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>Symptoms</h5>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {caseData.symptoms.map((symptom, idx) => (
              <span key={idx} style={{
                padding: '6px 12px',
                backgroundColor: '#f1f5f9',
                color: '#0f172a',
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 500
              }}>
                {symptom}
              </span>
            ))}
          </div>
        </div>

        {/* Vitals */}
        <div style={{ marginBottom: 16 }}>
          <h5 style={{ margin: '0 0 12px 0', fontSize: 12, fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>Vitals</h5>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
            <div style={{ padding: 12, backgroundColor: '#f9fafb', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>BP</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{caseData.vitals.bp}</div>
            </div>
            <div style={{ padding: 12, backgroundColor: '#f9fafb', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>HR</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{caseData.vitals.hr} bpm</div>
            </div>
            <div style={{ padding: 12, backgroundColor: '#f9fafb', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>Temp</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{caseData.vitals.temp}°C</div>
            </div>
            <div style={{ padding: 12, backgroundColor: '#f9fafb', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>RR</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{caseData.vitals.rr} /min</div>
            </div>
            <div style={{ padding: 12, backgroundColor: '#f9fafb', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>O₂</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{caseData.vitals.o2}%</div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 16, borderTop: '1px solid #f1f5f9' }}>
          <div style={{ display: 'flex', gap: 16, fontSize: 13 }}>
            <div>
              <span style={{ color: '#6b7280' }}>Outcome: </span>
              <span style={{ fontWeight: 600, color: getOutcomeColor(caseData.outcomeStatus) }}>
                {caseData.outcomeStatus}
              </span>
            </div>
            <div>
              <span style={{ color: '#6b7280' }}>Triage Time: </span>
              <span style={{ fontWeight: 600, color: '#0f172a' }}>{caseData.timeToTriage}</span>
            </div>
          </div>
          <FiChevronRight size={18} color="#10b981" />
        </div>
      </div>
    );
  };

  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div style={{ maxWidth: 1200 }}>
        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <div style={{
              width: 48,
              height: 48,
              borderRadius: 10,
              backgroundColor: '#ecfdf5',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#10b981'
            }}>
              <MdLibraryBooks size={28} />
            </div>
            <div>
              <h1 style={{ margin: '0 0 4px 0', fontSize: 32, fontWeight: 800, color: '#0f172a' }}>
                Reference Cases Library
              </h1>
              <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
                Search similar clinical cases to support diagnosis and treatment decisions
              </p>
            </div>
          </div>
        </div>

        {/* Search Panel */}
        <div style={{
          padding: 32,
          backgroundColor: 'white',
          borderRadius: 14,
          border: '1px solid #e5e7eb',
          marginBottom: 32,
          boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
        }}>
          {/* Mode Selector */}
          <div style={{ marginBottom: 28 }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 700, color: '#0f172a', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Search Mode
            </h3>
            <div style={{ display: 'flex', gap: 12 }}>
              {[
                { id: 'symptoms', label: 'By Symptoms', icon: '🔍' },
                { id: 'vitals', label: 'By Vitals', icon: '❤️' },
                { id: 'combined', label: 'Combined Search', icon: '⚡' }
              ].map(mode => (
                <button
                  key={mode.id}
                  onClick={() => setSearchMode(mode.id)}
                  style={{
                    padding: '12px 20px',
                    borderRadius: 8,
                    border: '2px solid',
                    background: searchMode === mode.id ? '#10b981' : 'transparent',
                    color: searchMode === mode.id ? '#fff' : '#0f172a',
                    borderColor: searchMode === mode.id ? '#10b981' : '#e5e7eb',
                    cursor: 'pointer',
                    fontSize: 13,
                    fontWeight: 600,
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => {
                    if (searchMode !== mode.id) {
                      e.currentTarget.style.borderColor = '#10b981';
                      e.currentTarget.style.backgroundColor = '#f0fdf4';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (searchMode !== mode.id) {
                      e.currentTarget.style.borderColor = '#e5e7eb';
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  {mode.icon} {mode.label}
                </button>
              ))}
            </div>
          </div>

          {/* Symptoms Section */}
          {(searchMode === 'symptoms' || searchMode === 'combined') && (
            <div style={{ marginBottom: 28 }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 700, color: '#0f172a', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                Chief Complaints & Symptoms
              </h3>
              
              {/* Quick Select Buttons */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 10, marginBottom: 16 }}>
                {commonSymptoms.map(symptom => (
                  <button
                    key={symptom}
                    onClick={() => toggleSymptom(symptom)}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 8,
                      border: '2px solid',
                      background: selectedSymptoms.includes(symptom) ? '#ecfdf5' : 'white',
                      color: selectedSymptoms.includes(symptom) ? '#10b981' : '#0f172a',
                      borderColor: selectedSymptoms.includes(symptom) ? '#10b981' : '#e5e7eb',
                      cursor: 'pointer',
                      fontSize: 12,
                      fontWeight: 500,
                      transition: 'all 200ms ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = '#10b981';
                    }}
                    onMouseLeave={(e) => {
                      if (!selectedSymptoms.includes(symptom)) {
                        e.currentTarget.style.borderColor = '#e5e7eb';
                      }
                    }}
                  >
                    {symptom}
                  </button>
                ))}
              </div>

              {/* Custom Input */}
              <input
                type="text"
                placeholder="Or enter additional symptoms..."
                value={symptoms}
                onChange={(e) => setSymptoms(e.target.value)}
                style={{
                  width: '100%',
                  padding: '12px 16px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  fontSize: 14,
                  boxSizing: 'border-box',
                  transition: 'all 200ms ease'
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = '#10b981';
                  e.currentTarget.style.boxShadow = '0 0 0 3px rgba(16, 185, 129, 0.1)';
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = '#e5e7eb';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              />
            </div>
          )}

          {/* Vitals Section */}
          {(searchMode === 'vitals' || searchMode === 'combined') && (
            <div style={{ marginBottom: 28 }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: 14, fontWeight: 700, color: '#0f172a', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                Patient Vitals
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
                {[
                  { key: 'bloodPressure', label: 'Blood Pressure', placeholder: 'e.g., 120/80' },
                  { key: 'heartRate', label: 'Heart Rate (bpm)', placeholder: 'e.g., 75' },
                  { key: 'temperature', label: 'Temperature (°C)', placeholder: 'e.g., 37.0' },
                  { key: 'respiratoryRate', label: 'Respiratory Rate (/min)', placeholder: 'e.g., 16' },
                  { key: 'oxygenSaturation', label: 'O₂ Saturation (%)', placeholder: 'e.g., 98' }
                ].map(field => (
                  <div key={field.key}>
                    <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#64748b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.3 }}>
                      {field.label}
                    </label>
                    <input
                      type="text"
                      placeholder={field.placeholder}
                      value={vitals[field.key]}
                      onChange={(e) => setVitals({ ...vitals, [field.key]: e.target.value })}
                      style={{
                        width: '100%',
                        padding: '10px 12px',
                        borderRadius: 8,
                        border: '1px solid #e5e7eb',
                        fontSize: 14,
                        boxSizing: 'border-box',
                        transition: 'all 200ms ease'
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = '#10b981';
                        e.currentTarget.style.boxShadow = '0 0 0 3px rgba(16, 185, 129, 0.1)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = '#e5e7eb';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Search Button */}
          <button
            onClick={handleSearch}
            style={{
              width: '100%',
              padding: '14px 24px',
              backgroundColor: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              fontSize: 15,
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              transition: 'all 200ms ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#059669';
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 10px 20px rgba(16, 185, 129, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#10b981';
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <FiSearch size={18} />
            Search Similar Cases
          </button>
        </div>

        {/* Results Section */}
        {showResults && (
          <div>
            <div style={{ marginBottom: 24 }}>
              <h2 style={{ margin: '0 0 8px 0', fontSize: 22, fontWeight: 700, color: '#0f172a' }}>
                Similar Cases Found
              </h2>
              <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
                {results.length} clinically similar cases from the reference library
              </p>
            </div>

            {/* Results List */}
            {results.map(caseData => (
              <CaseResultCard key={caseData.id} caseData={caseData} />
            ))}

            {/* Empty State */}
            {results.length === 0 && (
              <div style={{
                padding: 48,
                textAlign: 'center',
                backgroundColor: '#f9fafb',
                borderRadius: 12,
                border: '1px solid #e5e7eb'
              }}>
                <FiSearch size={48} style={{ color: '#cbd5e1', margin: '0 0 16px 0' }} />
                <h3 style={{ margin: '0 0 8px 0', fontSize: 18, fontWeight: 700, color: '#64748b' }}>
                  No matching cases found
                </h3>
                <p style={{ margin: 0, fontSize: 14, color: '#94a3b8' }}>
                  Try adjusting your search criteria to find similar clinical cases
                </p>
              </div>
            )}
          </div>
        )}

        {/* Info Banner */}
        {!showResults && (
          <div style={{
            padding: 24,
            backgroundColor: '#f0fdf4',
            border: '1px solid #bbf7d0',
            borderRadius: 12
          }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: 14, fontWeight: 700, color: '#166534' }}>
              💡 How to use the Reference Cases Library
            </h3>
            <p style={{ margin: 0, fontSize: 13, color: '#15803d', lineHeight: 1.6 }}>
              Select a search mode, enter the patient's symptoms and/or vital signs, then click "Search Similar Cases". 
              The system will query our RAG database and display clinically similar cases ranked by similarity score. 
              Use these cases to support clinical decision-making and triage decisions.
            </p>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

export default ReferenceCases;
