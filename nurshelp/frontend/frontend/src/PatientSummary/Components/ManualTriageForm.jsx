import React, { useState, useEffect } from 'react';
import StaffScheduleDisplay from './StaffScheduleDisplay';
import './ManualTriageForm.css';

/**
 * ManualTriageForm - Complete manual triage workflow (SATS - South African Triage Scale)
 * 
 * Intelligent Hardware Routing:
 * - On mount: Queries /api/capabilities to detect GPU availability
 * - If GPU + RAG: Routes to /api/triage (hybrid RAG + LLM synthesis)
 * - If CPU only: Routes to /api/llm-triage (LLM only, no RAG)
 * 
 * User Journey:
 * 1. Symptoms/vitals already captured (from Vitals component)
 * 2. Nurse manually decides SATS level (Red/Orange/Yellow/Green)
 * 3. Enter clinical reasoning
 * 4. Select available nurse, physician, bed (optional)
 * 5. Submit (routes to appropriate endpoint based on hardware)
 */
function ManualTriageForm({ patientData = {}, vitalsData = {}, onSubmit = () => {} }) {
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

  // Hardware detection & routing state
  const [triageEndpoint, setTriageEndpoint] = useState('/api/triage');  // Default to hybrid
  const [hardwareInfo, setHardwareInfo] = useState(null);
  const [capabilitiesLoading, setCapabilitiesLoading] = useState(true);
  
  // Manual triage form state (SATS: Red/Orange/Yellow/Green)
  const [satsLevel, setSatsLevel] = useState('yellow');
  const [urgencyScore, setUrgencyScore] = useState('50');
  const [reasoning, setReasoning] = useState('');
  const [department, setDepartment] = useState('ED');
  const [interventions, setInterventions] = useState('');

  // Staff selection
  const [selectedNurse, setSelectedNurse] = useState(null);
  const [selectedPhysician, setSelectedPhysician] = useState(null);
  const [selectedBed, setSelectedBed] = useState(null);

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [showStaffSelection, setShowStaffSelection] = useState(false);

  // Fetch hardware capabilities on mount to determine which endpoint to use
  useEffect(() => {
    const fetchCapabilities = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/capabilities`);
        if (response.ok) {
          const data = await response.json();
          setHardwareInfo(data);
          
          // Route based on recommended endpoint
          const endpoint = data.routing?.recommended_endpoint || '/api/triage';
          setTriageEndpoint(endpoint);
          
          console.log(`📡 Triage routing initialized:`);
          console.log(`   Hardware: ${data.hardware.gpu_available ? '✅ GPU' : '❌ CPU-only'}`);
          console.log(`   Endpoint: ${endpoint === '/api/triage' ? '🚀 Hybrid (RAG + LLM)' : '⚡ LLM-only'}`);
        } else {
          console.error('Failed to fetch capabilities, using default hybrid mode');
          setTriageEndpoint('/api/triage');
        }
      } catch (err) {
        console.error('Error fetching capabilities:', err);
        setTriageEndpoint('/api/triage');  // Fallback to hybrid
      } finally {
        setCapabilitiesLoading(false);
      }
    };
    
    fetchCapabilities();
  }, [BACKEND_URL]);

  // Form validation
  // Bed is OPTIONAL - not every patient gets hospitalized
  const isFormValid = () => {
    return (
      satsLevel &&
      ['red', 'orange', 'yellow', 'green'].includes(satsLevel) &&
      reasoning.trim().length > 0 &&
      selectedNurse &&
      selectedPhysician
      // NOTE: selectedBed is OPTIONAL
    );
  };

  const handleStaffSelect = {
    onNurseSelect: (nurse) => {
      setSelectedNurse(nurse);
    },
    onPhysicianSelect: (physician) => {
      setSelectedPhysician(physician);
    },
    onBedSelect: (bed) => {
      setSelectedBed(bed);
    }
  };

  const handleSubmitTriage = async () => {
    if (!isFormValid()) {
      setSubmitError('Please complete all required fields: SATS level (Red/Orange/Yellow/Green), clinical reasoning, and assign a nurse & physician. Bed assignment is optional.');
      return;
    }

    setSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(false);

    try {
      const token = localStorage.getItem('token');

      // Build manual triage request (NO computeMode)
      const triagePayload = {
        triageNurseId: localStorage.getItem('currentStaffId') || 'nurse-001',
        // NO computeMode - signals MANUAL workflow
        
        // Clinical data from patient intake
        // Convert symptoms to string (never send as array)
        symptoms: typeof patientData.symptoms === 'string' 
          ? patientData.symptoms 
          : (Array.isArray(patientData.symptoms) ? patientData.symptoms.join(", ") : ""),
        vitalSigns: {
          temperature: parseFloat(vitalsData.temp) || null,
          heartRate: parseInt(vitalsData.hr) || null,
          respiratoryRate: parseInt(vitalsData.resp) || null,
          systolicBP: vitalsData.bp_systolic || null,
          diastolicBP: vitalsData.bp_diastolic || null,
          oxygenSaturation: parseFloat(vitalsData.o2) || null
        },
        medicalHistory: patientData.history ? [patientData.history] : [],
        medications: patientData.medications || [],
        allergies: patientData.allergies || [],
        chiefComplaint: patientData.symptoms || 'Patient presenting with reported symptoms',
        triageNotes: reasoning,

        // Manual SATS from nurse
        manualSatsLevel: satsLevel,  // 'red', 'yellow', or 'green'
        urgencyScore: parseInt(urgencyScore),
        triageReasoningNotes: reasoning,
        recommendedDepartment: department,
        immediateInterventions: interventions
          .split('\n')
          .filter((i) => i.trim().length > 0),

        // Selected staff
        assignedNurseId: selectedNurse?.id,
        assignedNurseName: selectedNurse?.name || 'Assigned Nurse',
        assignedPhysicianId: selectedPhysician?.id,
        assignedPhysicianName: selectedPhysician?.name || 'Not Assigned',
        bedAssigned: selectedBed?.id,
        
        // Patient data (REQUIRED for display in dashboard)
        patientId: patientData.id || patientData._id,
        encounterId: patientData.encounterId || null,
        patient: {
          id_number: patientData.id || patientData._id || '',
          name: patientData.name || '',
          surname: patientData.surname || '',
          // Support both new (ageYears/ageMonths) and legacy (age) formats
          age: patientData.ageYears ?? patientData.age ?? null,
          ageMonths: patientData.ageMonths ?? null,
          height_cm: patientData.height_cm ?? null,
          gender: patientData.gender || '',
          symptoms: typeof patientData.symptoms === 'string' 
            ? patientData.symptoms 
            : (Array.isArray(patientData.symptoms) ? patientData.symptoms.join(", ") : ""),
          history: patientData.history || '',
          duration: patientData.duration || '',
          painLevel: patientData.painLevel || ''
        },
        vitals: {
          temp: parseFloat(vitalsData.temp) || null,
          hr: parseInt(vitalsData.hr) || null,
          resp: parseInt(vitalsData.resp) || null,
          bp_systolic: vitalsData.bp_systolic || null,
          bp_diastolic: vitalsData.bp_diastolic || null,
          o2: parseFloat(vitalsData.o2) || null
        }
      };

      const response = await fetch(`${BACKEND_URL}${triageEndpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { Authorization: `Bearer ${token}` })
        },
        body: JSON.stringify(triagePayload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || `Failed to submit triage: ${response.status}`
        );
      }

      const result = await response.json();
      setSubmitSuccess(true);

      // Enhance result with staff details for confirmation page
      // Extract the actual data from the API response structure
      const apiData = result.data || result;
      const enrichedResult = {
        ...apiData,
        triageRequestId: apiData.triageRequestId,
        computeMode: apiData.computeMode,
        manualTriageReport: {
          ...apiData.manualTriageReport,
          assignedNurse: selectedNurse?.name || selectedNurse?.id || 'N/A',
          assignedPhysician: selectedPhysician?.name || selectedPhysician?.id || 'N/A',
          bedAssigned: selectedBed?.id || 'Not Assigned'
        }
      };

      // Call parent callback with enriched triage result
      if (onSubmit) {
        onSubmit(enrichedResult);
      }

      // Show success for 2 seconds, then clear form
      setTimeout(() => {
        setSatsLevel('yellow');
        setUrgencyScore('50');
        setReasoning('');
        setInterventions('');
        setSelectedNurse(null);
        setSelectedPhysician(null);
        setSelectedBed(null);
        setSubmitSuccess(false);
      }, 2000);
    } catch (err) {
      console.error('Manual triage submission error:', err);
      setSubmitError(err.message || 'Failed to submit manual triage');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={containerStyle}>
      <div style={formHeaderStyle}>
        <h2>Manual Triage Entry (SATS)</h2>
        <p style={formSubtitleStyle}>
          South African Triage Scale - Pure nurse judgment, no AI involvement. 
          Select SATS level (Red/Orange/Yellow/Green), assign nurse & physician.
          Bed assignment is optional (not all patients will be hospitalized).
        </p>
        
        {/* Hardware & Routing Status */}
        {!capabilitiesLoading && hardwareInfo && (
          <div style={{
            marginTop: '12px',
            padding: '8px 12px',
            backgroundColor: hardwareInfo.hardware?.gpu_available ? '#dcfce7' : '#fef3c7',
            border: `1px solid ${hardwareInfo.hardware?.gpu_available ? '#22c55e' : '#f59e0b'}`,
            borderRadius: '4px',
            fontSize: '12px',
            color: hardwareInfo.hardware?.gpu_available ? '#166534' : '#92400e',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span style={{ fontSize: '14px' }}>
              {hardwareInfo.hardware?.gpu_available ? '🚀' : '⚡'}
            </span>
            <span>
              {hardwareInfo.hardware?.gpu_available 
                ? `GPU Mode: Hybrid (RAG + LLM synthesis)` 
                : `CPU Mode: LLM-only (fast inference)`}
            </span>
            {hardwareInfo.hardware?.gpu_available && hardwareInfo.hardware?.gpu_vram_gb > 0 && (
              <span style={{ marginLeft: 'auto', fontSize: '11px', opacity: 0.8 }}>
                {hardwareInfo.hardware.gpu_type} • {hardwareInfo.hardware.gpu_vram_gb}GB VRAM
              </span>
            )}
          </div>
        )}
      </div>

      {/* Patient Summary */}
      <div style={patientSummaryStyle}>
        <h4 style={sectionHeadingStyle}>Patient Info</h4>
        <div style={patientInfoGridStyle}>
          <div>
            <span style={labelStyle}>Name:</span>
            <span style={valueStyle}>
              {patientData.name} {patientData.surname}
            </span>
          </div>
          <div>
            <span style={labelStyle}>Age:</span>
            <span style={valueStyle}>{patientData.age || 'N/A'} years</span>
          </div>
          <div>
            <span style={labelStyle}>Chief Complaint:</span>
            <span style={valueStyle}>{patientData.symptoms || 'N/A'}</span>
          </div>
        </div>
      </div>

      {/* Vitals Display */}
      <div style={vitalsBoxStyle}>
        <h4 style={sectionHeadingStyle}>Current Vitals</h4>
        <div style={vitalsGridStyle}>
          <div style={vitalItemStyle}>
            <div style={vitalLabelStyle}>Temperature</div>
            <div style={vitalValueStyle}>{vitalsData.temp || '--'} °C</div>
          </div>
          <div style={vitalItemStyle}>
            <div style={vitalLabelStyle}>Heart Rate</div>
            <div style={vitalValueStyle}>{vitalsData.hr || '--'} bpm</div>
          </div>
          <div style={vitalItemStyle}>
            <div style={vitalLabelStyle}>BP</div>
            <div style={vitalValueStyle}>{vitalsData.bp || '--'} mmHg</div>
          </div>
          <div style={vitalItemStyle}>
            <div style={vitalLabelStyle}>O₂ Sat</div>
            <div style={vitalValueStyle}>{vitalsData.o2 || '--'} %</div>
          </div>
          <div style={vitalItemStyle}>
            <div style={vitalLabelStyle}>Resp Rate</div>
            <div style={vitalValueStyle}>{vitalsData.resp || '--'} breaths/min</div>
          </div>
          <div style={vitalItemStyle}>
            <div style={vitalLabelStyle}>AVPU</div>
            <div style={vitalValueStyle}>{vitalsData.avpu || 'Alert'}</div>
          </div>
        </div>
      </div>

      {/* SATS Level Selection */}
      <div style={formGroupStyle}>
        <label style={labelStyle}>
          <strong>SATS Level (Your Decision)</strong>
          <span style={requiredStyle}>*</span>
        </label>
        <div style={esiGridStyle}>
          {['red', 'yellow', 'green'].map((level) => (
            <button
              key={level}
              style={{
                ...esiButtonStyle,
                ...(satsLevel === level ? esiSelectedStyle : {}),
                backgroundColor: getSATSColor(level)
              }}
              onClick={() => setSatsLevel(level)}
              className="sats-button"
            >
              <div style={esiLevelNumberStyle}>{level.toUpperCase()}</div>
              <div style={esiLevelNameStyle}>{getSATSName(level)}</div>
            </button>
          ))}
        </div>
        <div style={esiDescriptionStyle}>
          {satsLevel ? `${satsLevel.toUpperCase()} - ${getSATSDescription(satsLevel)}` : 'Select a SATS level'}
        </div>
      </div>

      {/* Acuity Score */}
      <div style={formGroupStyle}>
        <label style={labelStyle}>
          <strong>Acuity Score</strong> (0-100)
        </label>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={urgencyScore}
          onChange={(e) => setUrgencyScore(e.target.value)}
          style={sliderStyle}
        />
        <div style={acuityDisplayStyle}>{urgencyScore}</div>
      </div>

      {/* Clinical Reasoning */}
      <div style={formGroupStyle}>
        <label style={labelStyle}>
          <strong>Your Clinical Reasoning</strong>
          <span style={requiredStyle}>*</span>
        </label>
        <textarea
          value={reasoning}
          onChange={(e) => setReasoning(e.target.value)}
          placeholder="Describe the key clinical findings that led to your ESI decision..."
          style={textareaStyle}
          rows="4"
        />
        <div style={charCountStyle}>{reasoning.length} characters</div>
      </div>

      {/* Department */}
      <div style={formGroupStyle}>
        <label style={labelStyle}>
          <strong>Recommended Department</strong>
        </label>
        <select
          value={department}
          onChange={(e) => setDepartment(e.target.value)}
          style={selectStyle}
        >
          <option value="ED">Emergency Department</option>
          <option value="ICU">Intensive Care Unit</option>
          <option value="General">General Ward</option>
          <option value="Pediatric">Pediatric</option>
          <option value="OB">Obstetrics</option>
        </select>
      </div>

      {/* Immediate Interventions */}
      <div style={formGroupStyle}>
        <label style={labelStyle}>
          <strong>Immediate Interventions</strong>
        </label>
        <textarea
          value={interventions}
          onChange={(e) => setInterventions(e.target.value)}
          placeholder="List interventions (one per line):&#10;- IV access&#10;- Cardiac monitoring&#10;- Blood work"
          style={textareaStyle}
          rows="3"
        />
      </div>

      {/* Staff Selection */}
      <div style={staffSectionStyle}>
        <div style={staffHeaderStyle}>
          <h3>Staff & Bed Assignment</h3>
          <button
            style={{
              ...toggleButtonStyle,
              ...(showStaffSelection ? toggleButtonActiveStyle : {})
            }}
            onClick={() => setShowStaffSelection(!showStaffSelection)}
          >
            {showStaffSelection ? '▼ Hide' : '▶ Show'} Staff Selection
          </button>
        </div>

        {showStaffSelection && (
          <StaffScheduleDisplay onStaffSelect={handleStaffSelect} />
        )}

        {/* Selected Staff Display */}
        <div style={selectedStaffStyle}>
          <h4>Selected Assignment</h4>
          <div style={assignmentRowStyle}>
            <div style={assignmentItemStyle}>
              <span style={assignmentLabelStyle}>Assigned Nurse:</span>
              <span style={{...assignmentValueStyle, ...(selectedNurse ? {} : {color: '#cbd5e1'})}}>
                {selectedNurse ? selectedNurse.name : 'Not selected'}
              </span>
              {selectedNurse && <span style={checkmarkStyle}>✓</span>}
            </div>
            <div style={assignmentItemStyle}>
              <span style={assignmentLabelStyle}>Assigned Physician:</span>
              <span style={{...assignmentValueStyle, ...(selectedPhysician ? {} : {color: '#cbd5e1'})}}>
                {selectedPhysician ? selectedPhysician.name : 'Not selected'}
              </span>
              {selectedPhysician && <span style={checkmarkStyle}>✓</span>}
            </div>
            <div style={assignmentItemStyle}>
              <span style={assignmentLabelStyle}>Assigned Bed:</span>
              <span style={{...assignmentValueStyle, ...(selectedBed ? {} : {color: '#cbd5e1'})}}>
                {selectedBed ? selectedBed.id : 'Not selected'}
              </span>
              {selectedBed && <span style={checkmarkStyle}>✓</span>}
            </div>
          </div>
        </div>
      </div>

      {/* Status Messages */}
      {submitError && <div style={errorStyle}>{submitError}</div>}
      {submitSuccess && (
        <div style={successStyle}>
          ✓ Manual triage submitted successfully! Patient added to EHR with
          unattended status in nurse dashboard.
        </div>
      )}

      {/* Submit Button */}
      <button
        style={{
          ...submitButtonStyle,
          ...(isFormValid() ? {} : disabledButtonStyle),
          ...(submitting ? { opacity: 0.7, cursor: 'not-allowed' } : {})
        }}
        onClick={handleSubmitTriage}
        disabled={!isFormValid() || submitting}
      >
        {submitting ? 'Submitting...' : 'Submit Manual Triage'}
      </button>

      <div style={helpTextStyle}>
        <strong>Note:</strong> This submission creates a manual triage report
        with NO AI involvement. The patient will appear in the EHR and in the
        Nurse Dashboard as unattended (assigned staff shown). Specialist
        feedback can be added later for accuracy tracking.
      </div>
    </div>
  );
}

// Helper functions for ESI color coding
function getSATSColor(level) {
  const colors = {
    'red': '#ef4444',      // Red - Emergency/Immediate
    'orange': '#f97316',   // Orange - Very Urgent
    'yellow': '#eab308',   // Yellow - Urgent
    'green': '#22c55e'     // Green - Routine/Non-urgent
  };
  return colors[level] || '#cbd5e1';
}

function getSATSName(level) {
  const names = {
    'red': 'Emergency',
    'yellow': 'Urgent',
    'green': 'Routine'
  };
  return names[level];
}

function getSATSDescription(level) {
  const descriptions = {
    'red': 'Life-threatening condition requiring immediate intervention',
    'yellow': 'Urgent condition requiring prompt care',
    'green': 'Non-urgent condition, stable patient'
  };
  return descriptions[level];
}

// Styles
const containerStyle = {
  padding: '24px',
  backgroundColor: '#ffffff',
  borderRadius: '8px',
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
};

const formHeaderStyle = {
  marginBottom: '24px',
  borderBottom: '2px solid #e2e8f0',
  paddingBottom: '16px'
};

const formSubtitleStyle = {
  margin: '8px 0 0 0',
  fontSize: '14px',
  color: '#64748b'
};

const sectionHeadingStyle = {
  marginBottom: '12px',
  fontSize: '14px',
  fontWeight: '600',
  color: '#1e293b',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const patientSummaryStyle = {
  padding: '16px',
  backgroundColor: '#f0f9ff',
  borderRadius: '6px',
  marginBottom: '20px',
  borderLeft: '4px solid #0284c7'
};

const patientInfoGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
  gap: '16px'
};

const vitalsBoxStyle = {
  padding: '16px',
  backgroundColor: '#fef3c7',
  borderRadius: '6px',
  marginBottom: '20px',
  borderLeft: '4px solid #f59e0b'
};

const vitalsGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
  gap: '12px'
};

const vitalItemStyle = {
  padding: '12px',
  backgroundColor: 'white',
  borderRadius: '4px',
  textAlign: 'center'
};

const vitalLabelStyle = {
  fontSize: '12px',
  color: '#64748b',
  fontWeight: '500',
  marginBottom: '4px'
};

const vitalValueStyle = {
  fontSize: '18px',
  fontWeight: '700',
  color: '#1e293b'
};

const valueStyle = {
  fontSize: '14px',
  fontWeight: '600',
  color: '#1e293b',
  marginLeft: '8px'
};

const formGroupStyle = {
  marginBottom: '20px'
};

const labelStyle = {
  display: 'block',
  marginBottom: '8px',
  fontSize: '14px',
  color: '#1e293b',
  fontWeight: '500'
};

const requiredStyle = {
  color: '#dc2626',
  marginLeft: '4px'
};

const esiGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(5, 1fr)',
  gap: '12px',
  marginBottom: '12px'
};

const esiButtonStyle = {
  padding: '12px 8px',
  border: '2px solid #e2e8f0',
  borderRadius: '6px',
  cursor: 'pointer',
  color: 'white',
  fontWeight: '600',
  transition: 'all 0.2s ease',
  minHeight: '80px',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '4px'
};

const esiSelectedStyle = {
  borderColor: '#1e293b',
  borderWidth: '3px',
  boxShadow: '0 0 0 3px rgba(0,0,0,0.1)'
};

const esiLevelNumberStyle = {
  fontSize: '24px',
  fontWeight: '700'
};

const esiLevelNameStyle = {
  fontSize: '11px',
  fontWeight: '500'
};

const esiDescriptionStyle = {
  fontSize: '13px',
  color: '#64748b',
  padding: '8px 12px',
  backgroundColor: '#f1f5f9',
  borderRadius: '4px'
};

const sliderStyle = {
  width: '100%',
  height: '6px',
  marginBottom: '8px',
  cursor: 'pointer'
};

const acuityDisplayStyle = {
  fontSize: '24px',
  fontWeight: '700',
  color: '#2563eb',
  textAlign: 'center'
};

const textareaStyle = {
  width: '100%',
  padding: '12px',
  fontSize: '14px',
  border: '1px solid #cbd5e1',
  borderRadius: '4px',
  fontFamily: 'inherit',
  resize: 'vertical',
  boxSizing: 'border-box'
};

const charCountStyle = {
  fontSize: '12px',
  color: '#94a3b8',
  marginTop: '4px',
  textAlign: 'right'
};

const selectStyle = {
  width: '100%',
  padding: '10px 12px',
  fontSize: '14px',
  border: '1px solid #cbd5e1',
  borderRadius: '4px',
  backgroundColor: 'white',
  cursor: 'pointer'
};

const staffSectionStyle = {
  marginBottom: '20px',
  padding: '16px',
  backgroundColor: '#f8fafc',
  borderRadius: '6px',
  border: '2px solid #e2e8f0'
};

const staffHeaderStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: '16px'
};

const toggleButtonStyle = {
  padding: '8px 12px',
  backgroundColor: '#e2e8f0',
  color: '#475569',
  border: '1px solid #cbd5e1',
  borderRadius: '4px',
  cursor: 'pointer',
  fontSize: '13px',
  fontWeight: '500',
  transition: 'all 0.2s ease'
};

const toggleButtonActiveStyle = {
  backgroundColor: '#2563eb',
  color: 'white',
  borderColor: '#2563eb'
};

const selectedStaffStyle = {
  padding: '16px',
  backgroundColor: 'white',
  borderRadius: '6px',
  marginTop: '16px'
};

const assignmentRowStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
  gap: '16px'
};

const assignmentItemStyle = {
  padding: '12px',
  backgroundColor: '#f1f5f9',
  borderRadius: '4px',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center'
};

const assignmentLabelStyle = {
  fontSize: '13px',
  fontWeight: '600',
  color: '#475569'
};

const assignmentValueStyle = {
  fontSize: '14px',
  fontWeight: '600',
  color: '#1e293b'
};

const checkmarkStyle = {
  color: '#10b981',
  fontSize: '18px',
  fontWeight: '700'
};

const submitButtonStyle = {
  width: '100%',
  padding: '14px 20px',
  fontSize: '16px',
  fontWeight: '600',
  color: 'white',
  backgroundColor: '#2563eb',
  border: 'none',
  borderRadius: '6px',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  marginTop: '12px'
};

const disabledButtonStyle = {
  backgroundColor: '#cbd5e1',
  cursor: 'not-allowed',
  opacity: 0.6
};

const errorStyle = {
  padding: '12px 16px',
  backgroundColor: '#fee2e2',
  color: '#991b1b',
  borderRadius: '6px',
  marginBottom: '12px',
  fontSize: '14px',
  border: '1px solid #fecaca'
};

const successStyle = {
  padding: '12px 16px',
  backgroundColor: '#dcfce7',
  color: '#166534',
  borderRadius: '6px',
  marginBottom: '12px',
  fontSize: '14px',
  border: '1px solid #bbf7d0',
  fontWeight: '500'
};

const helpTextStyle = {
  marginTop: '16px',
  padding: '12px',
  backgroundColor: '#f0f9ff',
  borderRadius: '4px',
  fontSize: '13px',
  color: '#0c4a6e',
  borderLeft: '3px solid #0284c7'
};

export default ManualTriageForm;
