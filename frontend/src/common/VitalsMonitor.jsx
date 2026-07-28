import React, { useState, useEffect } from 'react';
import './VitalsMonitor.css';

/**
 * VitalsMonitor Component
 * 
 * Real-time vital signs display with threshold-based alerts
 * Shows temperature, heart rate, blood pressure, respiratory rate, oxygen saturation
 * Displays warnings when vitals are outside normal ranges
 * 
 * Props:
 *  - vitals (required): Vitals object
 *  - patientId: For real-time updates (future WebSocket integration)
 *  - showChart: Boolean to display trend chart (future feature)
 *  - compact: Boolean for minimal view
 * 
 * Vitals object:
 * {
 *   temperature: number (°F)
 *   heartRate: number (bpm)
 *   bloodPressure: string ('systolic/diastolic')
 *   respiratoryRate: number (breaths/min)
 *   oxygenSaturation: number (%)
 *   lastVitalTime: string (ISO timestamp)
 * }
 */

const VitalsMonitor = ({ vitals, patientId, showChart = false, compact = false }) => {
  const [alerts, setAlerts] = useState([]);

  // Normal ranges for vital signs
  const VITAL_RANGES = {
    temperature: { min: 98.6 - 1.5, max: 98.6 + 1.5, unit: '°F' },
    heartRate: { min: 60, max: 100, unit: 'bpm' },
    respiratoryRate: { min: 12, max: 20, unit: '/min' },
    oxygenSaturation: { min: 95, max: 100, unit: '%' },
    // BP ranges: systolic and diastolic separately
    bpSystolic: { min: 90, max: 140 },
    bpDiastolic: { min: 60, max: 90 },
  };

  // Check vitals against thresholds
  useEffect(() => {
    const newAlerts = [];

    if (vitals) {
      // Check temperature
      if (vitals.temperature) {
        const temp = vitals.temperature;
        if (temp < VITAL_RANGES.temperature.min - 2) {
          newAlerts.push({
            id: 'temp-critical',
            type: 'critical',
            vital: 'Temperature',
            value: `${temp}°F`,
            message: `Hypothermia: ${temp}°F (severely low)`,
          });
        } else if (temp > VITAL_RANGES.temperature.max + 2) {
          newAlerts.push({
            id: 'temp-critical',
            type: 'critical',
            vital: 'Temperature',
            value: `${temp}°F`,
            message: `Fever: ${temp}°F (critically high)`,
          });
        } else if (temp < VITAL_RANGES.temperature.min || temp > VITAL_RANGES.temperature.max) {
          newAlerts.push({
            id: 'temp-warning',
            type: 'warning',
            vital: 'Temperature',
            value: `${temp}°F`,
            message: `Abnormal: ${temp}°F`,
          });
        }
      }

      // Check heart rate
      if (vitals.heartRate) {
        const hr = vitals.heartRate;
        if (hr < VITAL_RANGES.heartRate.min - 20 || hr > VITAL_RANGES.heartRate.max + 20) {
          newAlerts.push({
            id: 'hr-critical',
            type: 'critical',
            vital: 'Heart Rate',
            value: `${hr} bpm`,
            message: `Critical: ${hr} bpm`,
          });
        } else if (hr < VITAL_RANGES.heartRate.min || hr > VITAL_RANGES.heartRate.max) {
          newAlerts.push({
            id: 'hr-warning',
            type: 'warning',
            vital: 'Heart Rate',
            value: `${hr} bpm`,
            message: `Abnormal: ${hr} bpm`,
          });
        }
      }

      // Check blood pressure
      if (vitals.bloodPressure) {
        const [systolic, diastolic] = vitals.bloodPressure.split('/').map(Number);
        if (systolic > 180 || systolic < 70 || diastolic > 120 || diastolic < 40) {
          newAlerts.push({
            id: 'bp-critical',
            type: 'critical',
            vital: 'Blood Pressure',
            value: `${vitals.bloodPressure}`,
            message: `Critical: ${vitals.bloodPressure}`,
          });
        } else if (systolic > VITAL_RANGES.bpSystolic.max || systolic < VITAL_RANGES.bpSystolic.min ||
                   diastolic > VITAL_RANGES.bpDiastolic.max || diastolic < VITAL_RANGES.bpDiastolic.min) {
          newAlerts.push({
            id: 'bp-warning',
            type: 'warning',
            vital: 'Blood Pressure',
            value: `${vitals.bloodPressure}`,
            message: `Abnormal: ${vitals.bloodPressure}`,
          });
        }
      }

      // Check respiratory rate
      if (vitals.respiratoryRate) {
        const rr = vitals.respiratoryRate;
        if (rr < VITAL_RANGES.respiratoryRate.min - 5 || rr > VITAL_RANGES.respiratoryRate.max + 10) {
          newAlerts.push({
            id: 'rr-critical',
            type: 'critical',
            vital: 'Respiratory Rate',
            value: `${rr}/min`,
            message: `Critical: ${rr}/min`,
          });
        } else if (rr < VITAL_RANGES.respiratoryRate.min || rr > VITAL_RANGES.respiratoryRate.max) {
          newAlerts.push({
            id: 'rr-warning',
            type: 'warning',
            vital: 'Respiratory Rate',
            value: `${rr}/min`,
            message: `Abnormal: ${rr}/min`,
          });
        }
      }

      // Check oxygen saturation
      if (vitals.oxygenSaturation) {
        const o2 = vitals.oxygenSaturation;
        if (o2 < 90) {
          newAlerts.push({
            id: 'o2-critical',
            type: 'critical',
            vital: 'Oxygen Saturation',
            value: `${o2}%`,
            message: `Critical hypoxia: ${o2}%`,
          });
        } else if (o2 < VITAL_RANGES.oxygenSaturation.min) {
          newAlerts.push({
            id: 'o2-warning',
            type: 'warning',
            vital: 'Oxygen Saturation',
            value: `${o2}%`,
            message: `Low: ${o2}%`,
          });
        }
      }
    }

    setAlerts(newAlerts);
  }, [vitals]);

  if (!vitals) {
    return (
      <div className="vitals-monitor">
        <div className="vitals-status">
          <div className="status-icon">😴</div>
          <div className="status-message">No vital signs data available</div>
        </div>
      </div>
    );
  }

  const getVitalStatus = (vital, value) => {
    // Return 'normal', 'warning', or 'critical' based on value
    const matchingAlert = alerts.find(a => a.vital === vital);
    if (!matchingAlert) return 'normal';
    return matchingAlert.type;
  };

  return (
    <div className={`vitals-monitor ${compact ? 'compact' : 'full'}`}>
      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div className="vitals-alerts">
          {alerts.map((alert) => (
            <div key={alert.id} className={`alert-item ${alert.type}`}>
              <span className="alert-icon">
                {alert.type === 'critical' ? '🚨' : '⚠️'}
              </span>
              <span className="alert-message">{alert.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* Vitals Grid */}
      <div className="vitals-grid">
        {/* Temperature */}
        {vitals.temperature && (
          <div
            className={`vital-display ${getVitalStatus('Temperature', vitals.temperature)}`}
          >
            <div className="vital-header">
              <span className="vital-icon">🌡️</span>
              <span className="vital-name">Temperature</span>
            </div>
            <div className="vital-content">
              <div className="vital-number">{vitals.temperature}°F</div>
              <div className="normal-range">Normal: 97–99°F</div>
              <div className="vital-bar">
                <div
                  className={`vital-indicator temp-indicator`}
                  style={{
                    left: `${Math.max(0, Math.min(100, ((vitals.temperature - 95) / 5) * 100))}%`,
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Heart Rate */}
        {vitals.heartRate && (
          <div
            className={`vital-display ${getVitalStatus('Heart Rate', vitals.heartRate)}`}
          >
            <div className="vital-header">
              <span className="vital-icon">💓</span>
              <span className="vital-name">Heart Rate</span>
            </div>
            <div className="vital-content">
              <div className="vital-number">{vitals.heartRate} <span className="unit">bpm</span></div>
              <div className="normal-range">Normal: 60–100 bpm</div>
              <div className="vital-bar">
                <div
                  className={`vital-indicator hr-indicator`}
                  style={{
                    left: `${Math.max(0, Math.min(100, (vitals.heartRate / 130) * 100))}%`,
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Blood Pressure */}
        {vitals.bloodPressure && (
          <div
            className={`vital-display ${getVitalStatus('Blood Pressure', vitals.bloodPressure)}`}
          >
            <div className="vital-header">
              <span className="vital-icon">💪</span>
              <span className="vital-name">Blood Pressure</span>
            </div>
            <div className="vital-content">
              <div className="vital-number">{vitals.bloodPressure}</div>
              <div className="normal-range">Normal: 90–140 / 60–90</div>
              <div className="vital-mini-values">
                <span className="label">Sys:</span>
                <span className="value">{vitals.bloodPressure.split('/')[0]}</span>
                <span className="label">Dia:</span>
                <span className="value">{vitals.bloodPressure.split('/')[1]}</span>
              </div>
            </div>
          </div>
        )}

        {/* Respiratory Rate */}
        {vitals.respiratoryRate && (
          <div
            className={`vital-display ${getVitalStatus('Respiratory Rate', vitals.respiratoryRate)}`}
          >
            <div className="vital-header">
              <span className="vital-icon">🫁</span>
              <span className="vital-name">Respiratory Rate</span>
            </div>
            <div className="vital-content">
              <div className="vital-number">{vitals.respiratoryRate} <span className="unit">/min</span></div>
              <div className="normal-range">Normal: 12–20 /min</div>
              <div className="vital-bar">
                <div
                  className={`vital-indicator rr-indicator`}
                  style={{
                    left: `${Math.max(0, Math.min(100, (vitals.respiratoryRate / 30) * 100))}%`,
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Oxygen Saturation */}
        {vitals.oxygenSaturation && (
          <div
            className={`vital-display ${getVitalStatus('Oxygen Saturation', vitals.oxygenSaturation)}`}
          >
            <div className="vital-header">
              <span className="vital-icon">🫧</span>
              <span className="vital-name">O₂ Saturation</span>
            </div>
            <div className="vital-content">
              <div className="vital-number">{vitals.oxygenSaturation}%</div>
              <div className="normal-range">Normal: 95–100%</div>
              <div className="vital-bar">
                <div
                  className={`vital-indicator o2-indicator`}
                  style={{
                    left: `${Math.max(0, Math.min(100, vitals.oxygenSaturation))}%`,
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Last Updated */}
      {vitals.lastVitalTime && (
        <div className="vitals-timestamp">
          Last updated: {new Date(vitals.lastVitalTime).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default VitalsMonitor;
