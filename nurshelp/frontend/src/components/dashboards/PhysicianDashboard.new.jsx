import React, { useState, useEffect, useRef, useCallback } from 'react';
import './PhysicianDashboard.new.css';
import '../../Dashboard/DashboardTheme.css';

/**
 * PhysicianDashboard Component - PROFESSIONAL CLINICAL WORKSTATION
 * 
 * Critical Features:
 *   ✓ 3-Lane Clinical View (Active | Observation | Escalation)
 *   ✓ Real-time Bed Management with visual occupancy
 *   ✓ Quick Disposition Decision Making
 *   ✓ Performance Metrics at a glance
 *   ✓ Smooth animations and professional layout
 *   ✓ Dark-mode clinical aesthetic
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const PhysicianDashboard = ({ config = {}, context = 'ER' }) => {
  // ============ STATE MANAGEMENT ============
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Patient data by lane
  const [activeCases, setActiveCases] = useState([]);
  const [observationCases, setObservationCases] = useState([]);
  const [escalations, setEscalations] = useState([]);
  
  // Ward bed management
  const [beds, setBeds] = useState([]);
  const [bedOccupancy, setBedOccupancy] = useState({ total: 0, occupied: 0, available: 0 });
  
  // Session metrics
  const [metrics, setMetrics] = useState({
    casesReviewed: 0,
    dispositionsMade: 0,
    avgResponseTime: '--',
    complianceRate: '--'
  });
  
  // UI states
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [showDispositionModal, setShowDispositionModal] = useState(false);
  const [dispositionLoading, setDispositionLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  
  // Polling reference
  const pollingRef = useRef(null);

  // ============ FETCH PHYSICIAN CASES ============
  const fetchPhysicianCases = useCallback(async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      console.log('[PHYSICIAN DASHBOARD] Fetching cases with context:', context);
      const response = await fetch(`${BACKEND_URL}/api/physician/cases?context=${context}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch cases');
      
      const data = await response.json();
      console.log('[PHYSICIAN DASHBOARD] Cases received:', { active: data.activeCases?.length || 0, obs: data.observationCases?.length || 0, escal: data.escalations?.length || 0 });
      
      setActiveCases(data.activeCases || []);
      setObservationCases(data.observationCases || []);
      setEscalations(data.escalations || []);
      setMetrics(data.metrics || {});
      
      setError(null);
    } catch (err) {
      console.error('[PHYSICIAN] Fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [context]);

  // ============ FETCH WARD BEDS ============
  const fetchWardBeds = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      
      console.log('[PHYSICIAN DASHBOARD] Fetching beds');
      const response = await fetch(`${BACKEND_URL}/api/physician/beds?context=${context}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch bed data');
      
      const data = await response.json();
      console.log('[PHYSICIAN DASHBOARD] Beds received:', data.occupancy);
      setBeds(data.beds || []);
      setBedOccupancy(data.occupancy || { total: 0, occupied: 0, available: 0 });
    } catch (err) {
      console.error('[PHYSICIAN] Bed fetch error:', err);
    }
  }, [context]);

  // ============ SUBMIT DISPOSITION ============
  const submitDisposition = useCallback(async (disposition) => {
    if (!selectedPatient) return;
    
    try {
      setDispositionLoading(true);
      const token = localStorage.getItem('token');
      
      console.log('[PHYSICIAN DASHBOARD] Submitting disposition:', disposition);
      const response = await fetch(`${BACKEND_URL}/api/disposition`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          patientId: selectedPatient._id || selectedPatient.id,
          decision: disposition.decision,
          wardIfAdmit: disposition.ward,
          bedNumber: disposition.bed,
          notes: disposition.notes
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Disposition submission failed');
      }
      
      setSuccessMessage(`✓ Disposition recorded: ${disposition.decision}`);
      setShowDispositionModal(false);
      setSelectedPatient(null);
      
      setTimeout(() => setSuccessMessage(''), 3000);
      
      // Refresh data
      await fetchPhysicianCases();
      await fetchWardBeds();
    } catch (err) {
      console.error('[PHYSICIAN] Disposition error:', err);
      setError(err.message);
    } finally {
      setDispositionLoading(false);
    }
  }, [selectedPatient, fetchPhysicianCases, fetchWardBeds]);

  // ============ POLLING SETUP ============
  useEffect(() => {
    fetchPhysicianCases();
    fetchWardBeds();
    
    pollingRef.current = setInterval(() => {
      fetchPhysicianCases();
      fetchWardBeds();
    }, 30000);
    
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [fetchPhysicianCases, fetchWardBeds]);

  // ============ RENDER PATIENT CARD ============
  const PatientCard = ({ patient, lane }) => {
    const severityColor = {
      'RED': '#ef4444',
      'ORANGE': '#f97316',
      'YELLOW': '#eab308',
      'GREEN': '#22c55e',
      'BLUE': '#3b82f6'
    }[patient.priority] || '#64748b';
    
    const waitTime = Math.floor((Date.now() - new Date(patient.arrivedAt)) / 60000);
    
    return (
      <div 
        className="patient-card"
        style={{ borderLeftColor: severityColor }}
        onClick={() => {
          setSelectedPatient(patient);
          setShowDispositionModal(true);
        }}
      >
        <div className="patient-priority-badge" style={{ backgroundColor: severityColor }}>
          {patient.priority}
        </div>
        
        <div className="patient-header">
          <h4>{patient.firstName} {patient.lastName}</h4>
          <span className="patient-age">{patient.age}yo • {patient.gender}</span>
        </div>
        
        <div className="patient-complaint">
          {patient.chiefComplaint || 'N/A'}
        </div>
        
        <div className="patient-vitals">
          <div className="vital-item">
            <span className="vital-label">HR</span>
            <span className="vital-value">{patient.vitals?.hr || '--'}</span>
          </div>
          <div className="vital-item">
            <span className="vital-label">O₂</span>
            <span className="vital-value">{patient.vitals?.o2 || '--'}%</span>
          </div>
          <div className="vital-item">
            <span className="vital-label">BP</span>
            <span className="vital-value">{patient.vitals?.systolic || '--'}/{patient.vitals?.diastolic || '--'}</span>
          </div>
        </div>
        
        <div className="patient-footer">
          <span className="wait-time">⏱️ {waitTime}m</span>
          <button className="btn-quick-review">Review</button>
        </div>
      </div>
    );
  };

  // ============ RENDER DISPOSITION MODAL ============
  const DispositionModal = ({ patient }) => {
    const [choice, setChoice] = useState(null);
    const [selectedBed, setSelectedBed] = useState(null);
    const [notes, setNotes] = useState('');
    
    if (!patient) return null;
    
    const availableBeds = beds.filter(bed => bed.status === 'AVAILABLE');
    
    const handleSubmit = () => {
      if (!choice) {
        alert('Please select a disposition');
        return;
      }
      if (choice === 'ADMIT' && !selectedBed) {
        alert('Please select a bed for admission');
        return;
      }
      
      submitDisposition({
        decision: choice,
        ward: choice === 'ADMIT' ? 'ED' : null,
        bed: selectedBed,
        notes
      });
    };
    
    return (
      <div className="modal-overlay" onClick={() => setShowDispositionModal(false)}>
        <div className="modal-container" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h2>Clinical Decision</h2>
            <button 
              className="btn-close" 
              onClick={() => setShowDispositionModal(false)}
            >
              ✕
            </button>
          </div>
          
          <div className="modal-patient-info">
            <h3>{patient.firstName} {patient.lastName}</h3>
            <p className="complaint">{patient.chiefComplaint}</p>
            <p className="details">{patient.age} yo • {patient.gender} • {patient.priority} Priority</p>
          </div>
          
          <div className="modal-disposition-options">
            <label className="disposition-option discharge">
              <input 
                type="radio" 
                name="disposition"
                value="DISCHARGE"
                checked={choice === 'DISCHARGE'}
                onChange={(e) => setChoice(e.target.value)}
              />
              <div className="option-content">
                <span className="icon">🚪</span>
                <div className="option-text">
                  <h4>DISCHARGE</h4>
                  <p>Stable for home care</p>
                </div>
              </div>
            </label>
            
            <label className="disposition-option admit">
              <input 
                type="radio" 
                name="disposition"
                value="ADMIT"
                checked={choice === 'ADMIT'}
                onChange={(e) => setChoice(e.target.value)}
              />
              <div className="option-content">
                <span className="icon">🏥</span>
                <div className="option-text">
                  <h4>ADMIT</h4>
                  <p>Requires hospitalization</p>
                </div>
              </div>
            </label>
            
            <label className="disposition-option transfer">
              <input 
                type="radio" 
                name="disposition"
                value="TRANSFER"
                checked={choice === 'TRANSFER'}
                onChange={(e) => setChoice(e.target.value)}
              />
              <div className="option-content">
                <span className="icon">🚑</span>
                <div className="option-text">
                  <h4>TRANSFER</h4>
                  <p>To another facility</p>
                </div>
              </div>
            </label>
          </div>
          
          {choice === 'ADMIT' && (
            <div className="bed-selector-section">
              <label htmlFor="bed-select">Assign Bed:</label>
              <select 
                id="bed-select"
                value={selectedBed || ''} 
                onChange={(e) => setSelectedBed(e.target.value)}
              >
                <option value="">-- Select Available Bed --</option>
                {availableBeds.map(bed => (
                  <option key={bed.bedId} value={bed.bedId}>
                    {bed.bedId} ({bed.bedType})
                  </option>
                ))}
              </select>
              {availableBeds.length === 0 && (
                <p className="warning">⚠️ No beds available</p>
              )}
            </div>
          )}
          
          <div className="notes-section">
            <label htmlFor="notes">Clinical Notes:</label>
            <textarea 
              id="notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Assessment and reasoning..."
              rows={3}
            />
          </div>
          
          <div className="modal-footer">
            <button 
              className="btn btn-secondary"
              onClick={() => setShowDispositionModal(false)}
            >
              Cancel
            </button>
            <button 
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={dispositionLoading}
            >
              {dispositionLoading ? '⟳ Processing...' : 'Submit Decision'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  // ============ MAIN RENDER ============
  const totalCases = activeCases.length + observationCases.length + escalations.length;
  
  return (
    <div className="physician-dashboard">
      {/* HEADER */}
      <div className="dashboard-header">
        <div className="header-left">
          <h1>🏥 Physician Workstation</h1>
          <p className="subtitle">Clinical Decision Management • Real-time Ward Overview</p>
        </div>
        
        <div className="header-metrics">
          <div className="metric-box">
            <span className="metric-label">Cases</span>
            <span className="metric-value">{totalCases}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Decisions</span>
            <span className="metric-value">{metrics.dispositionsMade || 0}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Beds</span>
            <span className="metric-value">{bedOccupancy.available}/{bedOccupancy.total}</span>
          </div>
        </div>
      </div>

      {/* ALERTS & MESSAGES */}
      {error && (
        <div className="alert alert-error">
          ⚠️ {error}
        </div>
      )}
      {successMessage && (
        <div className="alert alert-success">
          {successMessage}
        </div>
      )}

      {/* MAIN CONTENT */}
      <div className="dashboard-body">
        {/* QUICK STATS */}
        <div className="quick-stats">
          <div className="stat-card critical">
            <span className="stat-icon">🔴</span>
            <div className="stat-info">
              <span className="stat-label">Active Cases</span>
              <span className="stat-count">{activeCases.length}</span>
            </div>
          </div>
          
          <div className="stat-card warning">
            <span className="stat-icon">🟡</span>
            <div className="stat-info">
              <span className="stat-label">Observation</span>
              <span className="stat-count">{observationCases.length}</span>
            </div>
          </div>
          
          <div className="stat-card alert">
            <span className="stat-icon">⚡</span>
            <div className="stat-info">
              <span className="stat-label">Escalations</span>
              <span className="stat-count">{escalations.length}</span>
            </div>
          </div>
          
          <div className="stat-card bed-occupancy-card">
            <span className="stat-icon">🛏️</span>
            <div className="stat-info">
              <span className="stat-label">Bed Availability</span>
              <span className="stat-count">{bedOccupancy.available} available</span>
            </div>
          </div>
        </div>

        {/* 3-LANE VIEW */}
        <div className="lanes-container">
          {/* LANE 1: ACTIVE CASES */}
          <div className="lane lane-active">
            <div className="lane-header">
              <h3>🔴 Active Cases</h3>
              <span className="lane-count">{activeCases.length}</span>
            </div>
            <div className="lane-content">
              {activeCases.length === 0 ? (
                <div className="empty-state">No active cases</div>
              ) : (
                activeCases.map(patient => (
                  <PatientCard key={patient._id || patient.id} patient={patient} lane="active" />
                ))
              )}
            </div>
          </div>

          {/* LANE 2: OBSERVATION */}
          <div className="lane lane-observation">
            <div className="lane-header">
              <h3>🟡 Observation</h3>
              <span className="lane-count">{observationCases.length}</span>
            </div>
            <div className="lane-content">
              {observationCases.length === 0 ? (
                <div className="empty-state">No patients in observation</div>
              ) : (
                observationCases.map(patient => (
                  <PatientCard key={patient._id || patient.id} patient={patient} lane="observation" />
                ))
              )}
            </div>
          </div>

          {/* LANE 3: ESCALATIONS */}
          <div className="lane lane-escalation">
            <div className="lane-header">
              <h3>⚡ Escalations</h3>
              <span className="lane-count">{escalations.length}</span>
            </div>
            <div className="lane-content">
              {escalations.length === 0 ? (
                <div className="empty-state">No escalations</div>
              ) : (
                escalations.map(patient => (
                  <PatientCard key={patient._id || patient.id} patient={patient} lane="escalation" />
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* DISPOSITION MODAL */}
      {showDispositionModal && <DispositionModal patient={selectedPatient} />}
    </div>
  );
};

export default PhysicianDashboard;
