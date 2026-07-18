// src/PatientSummary/Components/MiddleColumn.jsx
import React, { useState } from "react";
import ManualTriageForm from "./ManualTriageForm";
import TriageConfirmationPage from "./TriageConfirmationPage";

export default function MiddleColumn({ 
  data = [], 
  setData,
  activeTab = 'manual',
  patientData = {},
  vitalsData = {},
  onTriageSubmit = () => {}
}) {
  // State to track if we're showing confirmation page
  const [triageResultForConfirmation, setTriageResultForConfirmation] = useState(null);
  const [showingConfirmation, setShowingConfirmation] = useState(false);
  // Show triage form when manual tab is active
  if (activeTab === 'manual') {
    // If showing confirmation page, show that instead
    if (showingConfirmation && triageResultForConfirmation) {
      return (
        <div className="middle-column-layout">
          <TriageConfirmationPage
            triageResult={triageResultForConfirmation}
            patientData={patientData}
            onConfirm={(result) => {
              console.log('✅ Triage confirmed successfully:', result);
              // After confirmation is finalized, call parent callback
              onTriageSubmit(result);
              setShowingConfirmation(false);
              setTriageResultForConfirmation(null);
              alert('✓ Manual triage confirmed and locked in EHR');
            }}
            onCancel={() => {
              console.log('❌ Confirmation cancelled');
              // Cancel confirmation, go back to form
              setShowingConfirmation(false);
              setTriageResultForConfirmation(null);
            }}
          />
        </div>
      );
    }

    // Show form
    return (
      <div className="middle-column-layout">
        <ManualTriageForm
          patientData={patientData}
          vitalsData={vitalsData}
          onSubmit={(result) => {
            // Form submitted - show confirmation page instead of submitting immediately
            setTriageResultForConfirmation(result);
            setShowingConfirmation(true);
          }}
        />
      </div>
    );
  }

  // Show AI form when AI tab is active (currently placeholder)
  if (activeTab === 'ai') {
    return (
      <div className="middle-column-layout">
        <div style={aiPlaceholderStyle}>
          <h3>AI-Assisted Triage</h3>
          <p>Coming soon: Get AI suggestions along with your manual assessment.</p>
          <p>The AI will help with:</p>
          <ul>
            <li>SATS level suggestions with confidence scores</li>
            <li>Recommended physician based on specialty match</li>
            <li>Similar case retrieval for reference</li>
            <li>Acuity scoring assistance</li>
          </ul>
        </div>
      </div>
    );
  }

  // Fallback (shouldn't reach here with normal usage)
  return (
    <div className="middle-column-layout">
      <div style={aiPlaceholderStyle}>
        <p>Select a triage tab to begin</p>
      </div>
    </div>
  );
}

const aiPlaceholderStyle = {
  padding: '40px 24px',
  textAlign: 'center',
  color: '#64748b',
  minHeight: '400px',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  alignItems: 'center'
};