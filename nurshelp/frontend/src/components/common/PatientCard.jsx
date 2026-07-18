import React from 'react';
import './PatientCard.css';

/**
 * PatientCard Component
 * 
 * Reusable patient information display card for all dashboards
 * Shows key clinical info: vitals, acuity, ESI level, bed assignment
 * 
 * Props:
 *  - patient (required): Patient object with clinical data
 *  - onSelect: Callback when card is clicked
 *  - compact: Boolean to show compact vs full view
 *  - showVitals: Boolean to display vitals monitor
 * 
 * Patient object structure:
 * {
 *   _id: string
 *   firstName: string
 *   lastName: string
 *   age: number
 *   gender: string
 *   mrn: string
 *   dateOfBirth: string
 *   
 *   // Triage/Clinical Info
 *   arrivalTime: string (ISO timestamp)
 *   esiLevel: 1-5 (Emergency Severity Index)
 *   acuityScore: number (0-100)
 *   chiefComplaint: string
 *   triageNotes: string
 *   
 *   // Current Vitals
 *   vitals: {
 *     temperature: number (°F)
 *     heartRate: number (bpm)
 *     bloodPressure: string (systolic/diastolic)
 *     respiratoryRate: number (breaths/min)
 *     oxygenSaturation: number (%)
 *     lastVitalTime: string (ISO timestamp)
 *   }
 *   
 *   // Bed Assignment
 *   currentBedId: string
 *   currentWard: string
 *   bedStatus: 'OCCUPIED' | 'DISCHARGED' | 'TRANSFERRED'
 *   
 *   // Clinical History
 *   allergies: string[]
 *   medications: string[]
 *   comorbidities: string[]
 * }
 */

const PatientCard = ({ patient, onSelect, compact = false, showVitals = true }) => {
  if (!patient) return null;

  // Color coding for acuity levels
  const getAcuityColor = (score) => {
    if (score >= 80) return '#DC2626'; // Red - Critical
    if (score >= 60) return '#F59E0B'; // Orange - High
    if (score >= 40) return '#FBBF24'; // Yellow - Moderate
    return '#059669'; // Green - Low
  };

  const getESIColor = (level) => {
    const colors = {
      1: '#DC2626', // Red - Resuscitation
      2: '#F59E0B', // Orange - High risk
      3: '#3B82F6', // Blue - Stable but needs resources
      4: '#10B981', // Green - Minor
      5: '#6B7280', // Gray - Minimal
    };
    return colors[level] || '#9CA3AF';
  };

  const getESILabel = (level) => {
    const labels = {
      1: 'Resuscitation',
      2: 'High Risk',
      3: 'Resources',
      4: 'Minor',
      5: 'Minimal',
    };
    return labels[level] || 'Unknown';
  };

  // Format time since arrival
  const getTimeSinceArrival = () => {
    if (!patient.arrivalTime) return '—';
    const arrival = new Date(patient.arrivalTime);
    const now = new Date();
    const minutes = Math.floor((now - arrival) / 60000);

    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    return `${days}d`;
  };

  // Check if vitals are within normal range
  const isVitalAbnormal = (vital, value, type) => {
    const abnormalRanges = {
      temperature: { min: 98.6 - 2, max: 98.6 + 2 },
      heartRate: { min: 60, max: 100 },
      respiratoryRate: { min: 12, max: 20 },
      oxygenSaturation: { min: 95, max: 100 },
    };

    if (type === 'bloodPressure') {
      // BP is string like "120/80"
      const [systolic, diastolic] = value.split('/').map(Number);
      return systolic > 140 || systolic < 90 || diastolic > 90 || diastolic < 60;
    }

    const range = abnormalRanges[type];
    return !range || value < range.min || value > range.max;
  };

  const fullName = `${patient.firstName || ''} ${patient.lastName || ''}`.trim();
  const acuityScore = patient.acuityScore || 0;
  const acuityColor = getAcuityColor(acuityScore);

  return (
    <div className={`patient-card ${compact ? 'compact' : 'full'}`} onClick={onSelect}>
      {/* Header */}
      <div className="patient-card-header">
        <div className="patient-header-left">
          <div className="patient-avatar">{fullName.charAt(0)}</div>
          <div className="patient-basic-info">
            <div className="patient-name">{fullName || 'Unknown Patient'}</div>
            <div className="patient-demographics">
              {patient.age && <span>{patient.age}y</span>}
              {patient.gender && <span>{patient.gender.charAt(0).toUpperCase()}</span>}
              {patient.mrn && <span>MRN: {patient.mrn}</span>}
            </div>
          </div>
        </div>

        {/* Quick Status Indicators */}
        <div className="patient-header-right">
          {patient.esiLevel && (
            <div
              className="esi-badge"
              style={{ borderColor: getESIColor(patient.esiLevel) }}
            >
              <div className="esi-number" style={{ color: getESIColor(patient.esiLevel) }}>
                {patient.esiLevel}
              </div>
              <div className="esi-label">ESI</div>
            </div>
          )}

          {patient.acuityScore !== null && patient.acuityScore !== undefined && (
            <div className="acuity-indicator" style={{ backgroundColor: acuityColor }}>
              <div className="acuity-score">{Math.round(acuityScore)}</div>
              <div className="acuity-label">Acuity</div>
            </div>
          )}
        </div>
      </div>

      {/* Chief Complaint & Arrival Time */}
      {!compact && patient.chiefComplaint && (
        <div className="patient-card-section">
          <div className="section-label">Chief Complaint</div>
          <div className="chief-complaint">{patient.chiefComplaint}</div>
          <div className="arrival-time">Arrived {getTimeSinceArrival()} ago</div>
        </div>
      )}

      {/* Vitals Section */}
      {showVitals && patient.vitals && (
        <div className="patient-card-section vitals-section">
          <div className="section-label">Current Vitals</div>
          <div className="vitals-grid">
            <div className="vital-item">
              <div className="vital-label">Temp</div>
              <div
                className="vital-value"
                style={{
                  color: isVitalAbnormal('temperature', patient.vitals.temperature, 'temperature')
                    ? '#DC2626'
                    : '#059669',
                }}
              >
                {patient.vitals.temperature || '—'}°F
              </div>
            </div>

            <div className="vital-item">
              <div className="vital-label">HR</div>
              <div
                className="vital-value"
                style={{
                  color: isVitalAbnormal('heartRate', patient.vitals.heartRate, 'heartRate')
                    ? '#DC2626'
                    : '#059669',
                }}
              >
                {patient.vitals.heartRate || '—'} <span className="unit">bpm</span>
              </div>
            </div>

            <div className="vital-item">
              <div className="vital-label">BP</div>
              <div
                className="vital-value"
                style={{
                  color: isVitalAbnormal('bloodPressure', patient.vitals.bloodPressure, 'bloodPressure')
                    ? '#DC2626'
                    : '#059669',
                }}
              >
                {patient.vitals.bloodPressure || '—'}
              </div>
            </div>

            <div className="vital-item">
              <div className="vital-label">RR</div>
              <div
                className="vital-value"
                style={{
                  color: isVitalAbnormal(
                    'respiratoryRate',
                    patient.vitals.respiratoryRate,
                    'respiratoryRate'
                  )
                    ? '#DC2626'
                    : '#059669',
                }}
              >
                {patient.vitals.respiratoryRate || '—'} <span className="unit">/min</span>
              </div>
            </div>

            <div className="vital-item">
              <div className="vital-label">O₂ Sat</div>
              <div
                className="vital-value"
                style={{
                  color: isVitalAbnormal(
                    'oxygenSaturation',
                    patient.vitals.oxygenSaturation,
                    'oxygenSaturation'
                  )
                    ? '#DC2626'
                    : '#059669',
                }}
              >
                {patient.vitals.oxygenSaturation || '—'}%
              </div>
            </div>
          </div>
          {patient.vitals.lastVitalTime && (
            <div className="vital-timestamp">
              Last updated: {new Date(patient.vitals.lastVitalTime).toLocaleTimeString()}
            </div>
          )}
        </div>
      )}

      {/* Bed & Ward Info */}
      {!compact && patient.currentWard && (
        <div className="patient-card-section bed-section">
          <div className="section-label">Current Location</div>
          <div className="bed-info">
            <span className="ward">{patient.currentWard}</span>
            {patient.currentBedId && <span className="bed">{patient.currentBedId}</span>}
          </div>
          {patient.bedStatus && (
            <div className="bed-status" style={{ color: patient.bedStatus === 'OCCUPIED' ? '#059669' : '#9CA3AF' }}>
              {patient.bedStatus}
            </div>
          )}
        </div>
      )}

      {/* Alerts & Flags */}
      {!compact && (patient.allergies?.length > 0 || patient.medications?.length > 0) && (
        <div className="patient-card-section flags-section">
          {patient.allergies?.length > 0 && (
            <div className="flag">
              <span className="flag-icon alert">⚠️</span>
              <span className="flag-text">Allergies: {patient.allergies.join(', ')}</span>
            </div>
          )}
          {patient.medications?.length > 0 && (
            <div className="flag">
              <span className="flag-icon">💊</span>
              <span className="flag-text">{patient.medications.length} medications</span>
            </div>
          )}
        </div>
      )}

      {/* Footer - Action Indicator */}
      <div className="patient-card-footer">
        <span className="click-hint">Click to view full details →</span>
      </div>
    </div>
  );
};

export default PatientCard;
