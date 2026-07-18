import React, { useState } from 'react';

/**
 * TriageConfirmationPage - Final confirmation + edit step for audit trail
 * 
 * MANUAL TRIAGE (no AI):
 *   → Simple confirmation
 *   → Stores: manualTriageReport only
 *   → EHR: ["Manual Report"]
 * 
 * AI-ASSISTED TRIAGE (with computeMode):
 *   → Side-by-side comparison (Manual vs AI)
 *   → Two paths:
 *     1. Accept AI → Stores: manualTriageReport + aiTriageReport
 *        EHR: ["Manual Report", "AI Report"]
 *     2. Override → Stores: manualTriageReport + aiTriageReport + nurseEditsReport (NEW)
 *        EHR: ["Manual Report", "AI Report", "Nurse Edits"]
 * 
 * SPECIALIST FEEDBACK (Later):
 *   → PATCH /api/triage/:id/specialist-feedback
 *   → Stores: specialistFeedbackReport
 *   → EHR can compare all 3 + specialist decision
 * 
 * TRACEABILITY:
 *   nurseEditsReport tracks WHAT the nurse changed and WHY
 *   Original AI suggestions preserved separately from final decision
 *   Allows auditing: if nurse wrong, we see what AI said
 *                    if AI wrong, we can audit and improve
 */
function TriageConfirmationPage({ 
  triageResult = {},
  patientData = {},
  onConfirm = () => {},
  onCancel = () => {}
}) {
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  
  // Determine if manual or AI-assisted
  const isManualOnly = !triageResult.computeMode;
  const isAIAssisted = triageResult.computeMode && ['RAG_FULL', 'HYBRID'].includes(triageResult.computeMode);
  
  // State for AI edits
  const [editMode, setEditMode] = useState(false);
  const [editedData, setEditedData] = useState({
    satsLevel: triageResult?.aiTriageReport?.satsLevel,
    recommendedPhysician: triageResult?.aiTriageReport?.recommendedPhysician,
    recommendedSpecialty: triageResult?.aiTriageReport?.specialty
  });
  const [editReasoning, setEditReasoning] = useState('');
  const [confirming, setConfirming] = useState(false);

  // Track which fields were actually edited
  const getEditedFields = () => {
    const edited = [];
    if (triageResult.aiTriageReport?.satsLevel !== editedData.satsLevel) {
      edited.push('satsLevel');
    }
    if (triageResult.aiTriageReport?.recommendedPhysician !== editedData.recommendedPhysician) {
      edited.push('recommendedPhysician');
    }
    return edited;
  };

  // ============ MANUAL TRIAGE CONFIRMATION ============
  const handleConfirmManual = async () => {
    setConfirming(true);
    try {
      console.log('📋 Confirming manual triage:', {
        triageRequestId: triageResult.triageRequestId,
        manualTriageReport: triageResult.manualTriageReport
      });

      const payload = {
        confirmed: true,
        confirmedAt: new Date().toISOString(),
        triageType: 'MANUAL_ONLY',
        manualTriageReport: triageResult.manualTriageReport,
        nurseEditsReport: null,
        aiTriageReport: null
      };

      const response = await fetch(
        `${BACKEND_URL}/api/triage/${triageResult.triageRequestId}/confirm`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || 'Failed to confirm triage');
      }
      
      const result = await response.json();
      console.log('✅ Triage confirmed:', result);
      onConfirm(result);
    } catch (error) {
      console.error('❌ Confirmation error:', error);
      alert('Error confirming triage: ' + (error.message || 'Unknown error'));
    } finally {
      setConfirming(false);
    }
  };

  // ============ AI-ASSISTED: ACCEPT WITHOUT EDITS ============
  const handleAcceptAI = async () => {
    setConfirming(true);
    try {
      const payload = {
        confirmed: true,
        confirmedAt: new Date().toISOString(),
        triageType: 'AI_ACCEPTED',
        manualTriageReport: triageResult.manualTriageReport,
        aiTriageReport: triageResult.aiTriageReport,
        nurseEditsReport: null // No edits made
      };

      const response = await fetch(
        `${BACKEND_URL}/api/triage/${triageResult.triageRequestId}/confirm`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }
      );

      if (!response.ok) throw new Error('Failed to confirm');
      const result = await response.json();
      onConfirm(result);
    } catch (error) {
      console.error('Confirmation error:', error);
      alert('Error confirming triage. Please try again.');
    } finally {
      setConfirming(false);
    }
  };

  // ============ AI-ASSISTED: SAVE WITH EDITS ============
  const handleSaveEdits = async () => {
    if (!editReasoning.trim()) {
      alert('Please explain why you are changing the AI recommendations');
      return;
    }

    setConfirming(true);
    try {
      const editedFields = getEditedFields();
      
      // Create nurseEditsReport as SEPARATE object for audit trail
      const nurseEditsReport = {
        timestamp: new Date().toISOString(),
        
        // What AI said (original, unchanged)
        originalAIRecommendations: {
          satsLevel: triageResult.aiTriageReport?.satsLevel,
          recommendedPhysician: triageResult.aiTriageReport?.recommendedPhysician,
          recommendedSpecialty: triageResult.aiTriageReport?.specialty,
          confidence: triageResult.aiTriageReport?.confidence,
          aiReasoning: triageResult.aiTriageReport?.reasoning
        },
        
        // What nurse decided instead
        finalNurseDecision: {
          satsLevel: editedData.satsLevel,
          recommendedPhysician: editedData.recommendedPhysician,
          recommendedSpecialty: editedData.recommendedSpecialty
        },
        
        // Track changes for audit
        fieldsEdited: editedFields,
        changesSummary: {
          satsLevelChanged: triageResult.aiTriageReport?.satsLevel !== editedData.satsLevel
            ? `${triageResult.aiTriageReport?.satsLevel} → ${editedData.satsLevel}`
            : null,
          physicianChanged: triageResult.aiTriageReport?.recommendedPhysician !== editedData.recommendedPhysician
            ? `${triageResult.aiTriageReport?.recommendedPhysician} → ${editedData.recommendedPhysician}`
            : null
        },
        
        // Why the nurse changed it (for learning)
        nurseReasoning: editReasoning
      };

      const payload = {
        confirmed: true,
        confirmedAt: new Date().toISOString(),
        triageType: 'AI_WITH_NURSE_EDITS',
        manualTriageReport: triageResult.manualTriageReport,
        aiTriageReport: triageResult.aiTriageReport,
        nurseEditsReport // NEW: Separate object tracking edits
      };

      const response = await fetch(
        `${BACKEND_URL}/api/triage/${triageResult.triageRequestId}/confirm`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }
      );

      if (!response.ok) throw new Error('Failed to save edits');
      const result = await response.json();
      onConfirm(result);
    } catch (error) {
      console.error('Error saving edits:', error);
      alert('Error saving edits. Please try again.');
    } finally {
      setConfirming(false);
    }
  };

  // ============ RENDER: MANUAL TRIAGE ============
  if (isManualOnly) {
    return (
      <div style={containerStyle}>
        <div style={contentStyle}>
          <div style={headerStyle}>
            <h2 style={titleStyle}>✓ Confirm Manual Triage</h2>
            <p style={subtitleStyle}>100% Human Decision - No AI Involved</p>
          </div>

          <div style={reportSectionStyle}>
            <h3 style={sectionTitleStyle}>Manual Triage Report</h3>
            
            <div style={gridStyle}>
              <div style={fieldStyle}>
                <label style={labelStyle}>Patient</label>
                <div style={valueStyle}>{patientData.name} {patientData.surname}</div>
              </div>

              <div style={fieldStyle}>
                <label style={labelStyle}>SATS Level</label>
                <div style={{ ...valueStyle, ...getSATSLevelStyle(triageResult.manualTriageReport?.satsLevel) }}>
                  {triageResult.manualTriageReport?.satsLevel?.toUpperCase()}
                </div>
              </div>

              <div style={fieldStyle}>
                <label style={labelStyle}>Urgency Score</label>
                <div style={valueStyle}>{triageResult.manualTriageReport?.urgencyScore || 'N/A'}</div>
              </div>

              <div style={{ ...fieldStyle, gridColumn: '1 / -1' }}>
                <label style={labelStyle}>Your Clinical Reasoning</label>
                <div style={reasoningStyle}>{triageResult.manualTriageReport?.reasoning}</div>
              </div>

              <div style={fieldStyle}>
                <label style={labelStyle}>Assigned Nurse</label>
                <div style={valueStyle}>{triageResult.manualTriageReport?.assignedNurse || 'N/A'}</div>
              </div>

              <div style={fieldStyle}>
                <label style={labelStyle}>Assigned Physician</label>
                <div style={valueStyle}>{triageResult.manualTriageReport?.assignedPhysician || 'N/A'}</div>
              </div>

              <div style={fieldStyle}>
                <label style={labelStyle}>Bed Assignment</label>
                <div style={valueStyle}>{triageResult.manualTriageReport?.bedAssigned || 'Not Assigned'}</div>
              </div>
            </div>
          </div>

          <div style={auditTrailStyle}>
            <h4 style={{ margin: '0 0 8px 0' }}>EHR Documentation:</h4>
            <ul style={{ margin: '0', paddingLeft: '20px' }}>
              <li>✓ Manual Triage Report</li>
              <li>○ AI Report (N/A - Manual path)</li>
              <li>○ Nurse Edits (N/A - No changes)</li>
            </ul>
          </div>

          <div style={actionsStyle}>
            <button
              style={confirmButtonStyle}
              onClick={handleConfirmManual}
              disabled={confirming}
            >
              {confirming ? 'Confirming...' : '✓ Confirm & Lock'}
            </button>
            <button
              style={cancelButtonStyle}
              onClick={onCancel}
              disabled={confirming}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============ RENDER: AI-ASSISTED TRIAGE ============
  if (isAIAssisted) {
    return (
      <div style={containerStyle}>
        <div style={contentStyle}>
          <div style={headerStyle}>
            <h2 style={titleStyle}>Review AI-Assisted Triage</h2>
            <p style={subtitleStyle}>Compare Manual vs AI ({triageResult.computeMode})</p>
          </div>

          {/* Side-by-side comparison */}
          <div style={comparisonGridStyle}>
            {/* LEFT: Manual Assessment */}
            <div style={comparisonColumnStyle}>
              <h3 style={columnTitleStyle}>Your Manual Assessment</h3>
              <div style={reportBoxStyle}>
                <div style={fieldStyle}>
                  <label style={labelStyle}>SATS Level</label>
                  <div style={{ ...valueStyle, ...getSATSLevelStyle(triageResult.manualTriageReport?.satsLevel) }}>
                    {triageResult.manualTriageReport?.satsLevel?.toUpperCase()}
                  </div>
                </div>

                <div style={fieldStyle}>
                  <label style={labelStyle}>Urgency Score</label>
                  <div style={valueStyle}>{triageResult.manualTriageReport?.urgencyScore || 'N/A'}</div>
                </div>

                <div style={fieldStyle}>
                  <label style={labelStyle}>Your Reasoning</label>
                  <div style={reasoningStyle}>{triageResult.manualTriageReport?.reasoning}</div>
                </div>

                <div style={fieldStyle}>
                  <label style={labelStyle}>Recommended Physician</label>
                  <div style={valueStyle}>{triageResult.manualTriageReport?.recommendedPhysician || 'Not specified'}</div>
                </div>
              </div>
            </div>

            {/* RIGHT: AI Recommendations */}
            <div style={comparisonColumnStyle}>
              <h3 style={columnTitleStyle}>AI Recommendations</h3>
              <div style={reportBoxStyle}>
                <div style={fieldStyle}>
                  <label style={labelStyle}>SATS Level</label>
                  <div style={aiValueStyle}>
                    <div style={{ ...valueStyle, ...getSATSLevelStyle(triageResult.aiTriageReport?.satsLevel) }}>
                      {triageResult.aiTriageReport?.satsLevel?.toUpperCase()}
                    </div>
                    <span style={confidenceStyle}>{triageResult.aiTriageReport?.confidence || '85'}% confidence</span>
                  </div>
                </div>

                <div style={fieldStyle}>
                  <label style={labelStyle}>AI Reasoning</label>
                  <div style={reasoningStyle}>{triageResult.aiTriageReport?.reasoning}</div>
                </div>

                <div style={fieldStyle}>
                  <label style={labelStyle}>Recommended Physician</label>
                  <div style={valueStyle}>
                    {triageResult.aiTriageReport?.recommendedPhysician} 
                    <br />
                    <span style={specialtyStyle}>({triageResult.aiTriageReport?.specialty})</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Edit Mode */}
          {!editMode ? (
            <div style={actionsStyle}>
              <button
                style={acceptButtonStyle}
                onClick={handleAcceptAI}
                disabled={confirming}
              >
                ✓ Accept AI Recommendations
              </button>
              <button
                style={editButtonStyle}
                onClick={() => setEditMode(true)}
              >
                ✎ Override & Edit
              </button>
              <button
                style={cancelButtonStyle}
                onClick={onCancel}
              >
                Cancel
              </button>
            </div>
          ) : (
            <div style={editSectionStyle}>
              <h3 style={sectionTitleStyle}>Override AI Recommendations</h3>
              <p style={{ color: '#64748b', fontSize: '13px', margin: '0 0 16px 0' }}>
                Your changes will be tracked separately for audit trail
              </p>

              <div style={gridStyle}>
                <div style={fieldStyle}>
                  <label style={labelStyle}>Override SATS Level?</label>
                  <select
                    value={editedData.satsLevel}
                    onChange={(e) => setEditedData({ ...editedData, satsLevel: e.target.value })}
                    style={selectStyle}
                  >
                    <option value="red">RED - Emergency</option>
                    <option value="yellow">YELLOW - Urgent</option>
                    <option value="green">GREEN - Routine</option>
                  </select>
                </div>

                <div style={fieldStyle}>
                  <label style={labelStyle}>Override Physician?</label>
                  <input
                    type="text"
                    value={editedData.recommendedPhysician}
                    onChange={(e) => setEditedData({ ...editedData, recommendedPhysician: e.target.value })}
                    placeholder="Enter physician name"
                    style={inputStyle}
                  />
                </div>
              </div>

              <div style={{ ...fieldStyle, gridColumn: '1 / -1', marginTop: '16px' }}>
                <label style={labelStyle}>Why are you overriding the AI?</label>
                <textarea
                  value={editReasoning}
                  onChange={(e) => setEditReasoning(e.target.value)}
                  placeholder="Document your clinical reasoning for the changes. This is tracked for audit trail..."
                  rows={5}
                  style={textareaStyle}
                  maxLength={500}
                />
                <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                  {editReasoning.length}/500 characters
                </div>
              </div>

              {getEditedFields().length > 0 && (
                <div style={changesStyle}>
                  <h4 style={{ margin: '0 0 8px 0' }}>Changes to be tracked:</h4>
                  {triageResult.aiTriageReport?.satsLevel !== editedData.satsLevel && (
                    <div style={changeItemStyle}>
                      SATS: {triageResult.aiTriageReport?.satsLevel?.toUpperCase()} → {editedData.satsLevel.toUpperCase()}
                    </div>
                  )}
                  {triageResult.aiTriageReport?.recommendedPhysician !== editedData.recommendedPhysician && (
                    <div style={changeItemStyle}>
                      Physician: {triageResult.aiTriageReport?.recommendedPhysician} → {editedData.recommendedPhysician}
                    </div>
                  )}
                </div>
              )}

              <div style={auditTrailStyle}>
                <h4 style={{ margin: '0 0 8px 0' }}>EHR Documentation:</h4>
                <ul style={{ margin: '0', paddingLeft: '20px' }}>
                  <li>✓ Manual Triage Report</li>
                  <li>✓ AI Triage Report (original, unchanged)</li>
                  <li>✓ Nurse Edits Report (NEW - tracks your changes)</li>
                </ul>
              </div>

              <div style={actionsStyle}>
                <button
                  style={confirmButtonStyle}
                  onClick={handleSaveEdits}
                  disabled={confirming || !editReasoning.trim()}
                >
                  {confirming ? 'Saving...' : '✓ Save Edits & Confirm'}
                </button>
                <button
                  style={cancelButtonStyle}
                  onClick={() => {
                    setEditMode(false);
                    setEditReasoning('');
                  }}
                  disabled={confirming}
                >
                  Cancel Changes
                </button>
              </div>
            </div>
          )}

          {!editMode && (
            <div style={infoStyle}>
              <p style={{ margin: '0 0 8px 0' }}>
                <strong>If you Accept AI:</strong> Both manual and AI reports stored. Specialist can compare later.
              </p>
              <p style={{ margin: '0' }}>
                <strong>If you Override:</strong> Your changes tracked separately. We can audit why AI differed and improve the model.
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return null;
}

// ============ STYLES ============

const containerStyle = {
  padding: '24px',
  backgroundColor: '#f8fafc',
  minHeight: '100vh'
};

const contentStyle = {
  maxWidth: '1200px',
  margin: '0 auto',
  backgroundColor: 'white',
  borderRadius: '8px',
  padding: '32px',
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
};

const headerStyle = {
  marginBottom: '32px',
  borderBottom: '2px solid #e2e8f0',
  paddingBottom: '16px'
};

const titleStyle = {
  fontSize: '28px',
  fontWeight: '700',
  color: '#1f2937',
  margin: '0 0 8px 0'
};

const subtitleStyle = {
  fontSize: '14px',
  color: '#64748b',
  margin: '0'
};

const reportSectionStyle = {
  marginBottom: '32px'
};

const sectionTitleStyle = {
  fontSize: '18px',
  fontWeight: '600',
  color: '#1f2937',
  marginBottom: '16px'
};

const gridStyle = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '16px'
};

const fieldStyle = {
  display: 'flex',
  flexDirection: 'column',
  gap: '6px'
};

const labelStyle = {
  fontSize: '13px',
  fontWeight: '600',
  color: '#64748b',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const valueStyle = {
  fontSize: '15px',
  color: '#1f2937',
  padding: '12px',
  backgroundColor: '#f1f5f9',
  borderRadius: '6px',
  fontWeight: '500'
};

const reasoningStyle = {
  fontSize: '14px',
  color: '#1f2937',
  padding: '12px',
  backgroundColor: '#f1f5f9',
  borderRadius: '6px',
  lineHeight: '1.5',
  whiteSpace: 'pre-wrap',
  maxHeight: '200px',
  overflow: 'auto'
};

const comparisonGridStyle = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '24px',
  marginBottom: '32px'
};

const comparisonColumnStyle = {
  display: 'flex',
  flexDirection: 'column',
  gap: '12px'
};

const columnTitleStyle = {
  fontSize: '16px',
  fontWeight: '600',
  color: '#1f2937'
};

const reportBoxStyle = {
  display: 'grid',
  gridTemplateColumns: '1fr',
  gap: '12px',
  padding: '16px',
  backgroundColor: '#f8fafc',
  borderRadius: '8px',
  border: '1px solid #e2e8f0'
};

const aiValueStyle = {
  display: 'flex',
  flexDirection: 'column',
  gap: '6px'
};

const confidenceStyle = {
  fontSize: '12px',
  color: '#64748b',
  fontStyle: 'italic'
};

const specialtyStyle = {
  fontSize: '12px',
  color: '#64748b'
};

const editSectionStyle = {
  padding: '24px',
  backgroundColor: '#fef3c7',
  borderRadius: '8px',
  border: '2px solid #fcd34d',
  marginBottom: '24px'
};

const selectStyle = {
  padding: '10px 12px',
  fontSize: '14px',
  border: '1px solid #cbd5e1',
  borderRadius: '6px',
  backgroundColor: 'white',
  cursor: 'pointer'
};

const inputStyle = {
  padding: '10px 12px',
  fontSize: '14px',
  border: '1px solid #cbd5e1',
  borderRadius: '6px',
  backgroundColor: 'white'
};

const textareaStyle = {
  padding: '10px 12px',
  fontSize: '14px',
  border: '1px solid #cbd5e1',
  borderRadius: '6px',
  backgroundColor: 'white',
  fontFamily: 'inherit',
  resize: 'vertical'
};

const changesStyle = {
  padding: '12px',
  backgroundColor: 'rgba(255, 255, 255, 0.9)',
  borderRadius: '6px',
  marginBottom: '16px',
  fontSize: '13px',
  border: '1px solid #fcd34d'
};

const changeItemStyle = {
  padding: '6px 0',
  color: '#92400e',
  fontWeight: '500'
};

const auditTrailStyle = {
  padding: '16px',
  backgroundColor: '#dbeafe',
  borderRadius: '6px',
  border: '1px solid #93c5fd',
  marginBottom: '24px',
  fontSize: '13px',
  color: '#1e40af'
};

const actionsStyle = {
  display: 'flex',
  gap: '12px',
  marginTop: '24px',
  justifyContent: 'center',
  flexWrap: 'wrap'
};

const confirmButtonStyle = {
  padding: '12px 24px',
  backgroundColor: '#10b981',
  color: 'white',
  border: 'none',
  borderRadius: '6px',
  fontSize: '14px',
  fontWeight: '600',
  cursor: 'pointer',
  transition: 'background-color 0.2s',
  minWidth: '150px'
};

const acceptButtonStyle = {
  padding: '12px 24px',
  backgroundColor: '#3b82f6',
  color: 'white',
  border: 'none',
  borderRadius: '6px',
  fontSize: '14px',
  fontWeight: '600',
  cursor: 'pointer',
  transition: 'background-color 0.2s',
  minWidth: '150px'
};

const editButtonStyle = {
  padding: '12px 24px',
  backgroundColor: '#f59e0b',
  color: 'white',
  border: 'none',
  borderRadius: '6px',
  fontSize: '14px',
  fontWeight: '600',
  cursor: 'pointer',
  transition: 'background-color 0.2s',
  minWidth: '150px'
};

const cancelButtonStyle = {
  padding: '12px 24px',
  backgroundColor: '#e5e7eb',
  color: '#374151',
  border: 'none',
  borderRadius: '6px',
  fontSize: '14px',
  fontWeight: '600',
  cursor: 'pointer',
  transition: 'background-color 0.2s'
};

const infoStyle = {
  marginTop: '24px',
  padding: '16px',
  backgroundColor: '#dbeafe',
  borderRadius: '6px',
  border: '1px solid #93c5fd',
  fontSize: '13px',
  color: '#1e40af',
  lineHeight: '1.6'
};

const getSATSLevelStyle = (level) => {
  const styles = {
    'red': { backgroundColor: '#fee2e2', color: '#991b1b', fontWeight: '700' },
    'yellow': { backgroundColor: '#fef3c7', color: '#92400e', fontWeight: '700' },
    'green': { backgroundColor: '#dcfce7', color: '#15803d', fontWeight: '700' }
  };
  return styles[level?.toLowerCase()] || {};
};

export default TriageConfirmationPage;
