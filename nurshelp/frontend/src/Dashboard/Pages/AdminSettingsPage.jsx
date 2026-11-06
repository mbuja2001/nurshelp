// src/Dashboard/Pages/AdminSettingsPage.jsx
import React, { useState, useEffect } from 'react';
import { FaSave, FaExclamationTriangle, FaClipboardList } from 'react-icons/fa';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function AdminSettingsPage({ token }) {
  const [settings, setSettings] = useState({
    lateArrivalThresholdMinutes: 15,
    earlyDepartureThresholdMinutes: 15,
    mandatoryBreakDurationMinutes: 30,
    maxBreakDurationMinutes: 60,
    clockInWindowMinutes: 30,
    autoApprovalEnabled: false,
    dailyScheduleStartHour: 6,
    dailyScheduleEndHour: 22,
    requireBreakApproval: true,
    requireLateApproval: true,
    notificationEmailEnabled: true,
    maxConsecutiveHoursWithoutBreak: 6
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [errors, setErrors] = useState({});
  const [trialMode, setTrialMode] = useState(null);
  const [trialModeError, setTrialModeError] = useState('');

  useEffect(() => {
    fetchSettings();
    fetchTrialMode();
  }, [token]);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/admin/settings`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success && data.settings) {
        setSettings(data.settings);
      }
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to load settings' });
    } finally {
      setLoading(false);
    }
  };

  const fetchTrialMode = async () => {
    setTrialModeError('');
    try {
      const response = await fetch(`${BACKEND_URL}/api/hospital/clinical-trial-mode`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) {
        throw new Error(data.message || 'Unable to load hospital trial settings');
      }
      setTrialMode(data.settings);
    } catch (error) {
      setTrialModeError(error.message || 'Unable to load hospital trial settings');
    }
  };

  const handleChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
    // Clear error for this field
    if (errors[key]) {
      setErrors(prev => ({
        ...prev,
        [key]: null
      }));
    }
  };

  const validateSettings = () => {
    const newErrors = {};

    if (settings.lateArrivalThresholdMinutes < 1 || settings.lateArrivalThresholdMinutes > 120) {
      newErrors.lateArrivalThresholdMinutes = 'Must be between 1 and 120 minutes';
    }
    if (settings.mandatoryBreakDurationMinutes < 15 || settings.mandatoryBreakDurationMinutes > 120) {
      newErrors.mandatoryBreakDurationMinutes = 'Must be between 15 and 120 minutes';
    }
    if (settings.maxBreakDurationMinutes < settings.mandatoryBreakDurationMinutes) {
      newErrors.maxBreakDurationMinutes = 'Must be greater than or equal to mandatory break duration';
    }
    if (settings.clockInWindowMinutes < 5 || settings.clockInWindowMinutes > 120) {
      newErrors.clockInWindowMinutes = 'Must be between 5 and 120 minutes';
    }
    if (settings.dailyScheduleStartHour < 0 || settings.dailyScheduleStartHour > 23) {
      newErrors.dailyScheduleStartHour = 'Must be between 0 and 23';
    }
    if (settings.dailyScheduleEndHour < 0 || settings.dailyScheduleEndHour > 23) {
      newErrors.dailyScheduleEndHour = 'Must be between 0 and 23';
    }
    if (settings.dailyScheduleStartHour >= settings.dailyScheduleEndHour) {
      newErrors.dailyScheduleEndHour = 'End hour must be after start hour';
    }
    if (settings.maxConsecutiveHoursWithoutBreak < 2 || settings.maxConsecutiveHoursWithoutBreak > 12) {
      newErrors.maxConsecutiveHoursWithoutBreak = 'Must be between 2 and 12 hours';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = async () => {
    if (!validateSettings()) {
      setMessage({ type: 'error', text: 'Please fix validation errors above' });
      return;
    }

    setSaving(true);
    setMessage(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/admin/settings`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
      });
      const data = await res.json();
      if (data.success) {
        setMessage({ type: 'success', text: 'Settings saved successfully!' });
      } else {
        setMessage({ type: 'error', text: data.message || 'Failed to save settings' });
      }
    } catch (err) {
      setMessage({ type: 'error', text: err.message });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <div style={styles.loadingBox}>⏳ Loading settings...</div>;
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>⚙️ System Settings</h1>
        <p style={styles.subtitle}>Configure system-wide clock-in/out behavior</p>
      </div>

      {message && (
        <div style={{
          ...styles.messageBanner,
          backgroundColor: message.type === 'success' ? '#d1fae5' : '#fee2e2',
          color: message.type === 'success' ? '#065f46' : '#991b1b',
          borderColor: message.type === 'success' ? '#a7f3d0' : '#fca5a5'
        }}>
          {message.text}
        </div>
      )}

      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Clinical trial readiness</h2>
        {trialModeError ? (
          <div role="alert" style={styles.trialModeError}>{trialModeError}</div>
        ) : trialMode ? (
          <dl style={styles.trialModeDetails}>
            <div><dt>Hospital</dt><dd>{trialMode.hospitalName}</dd></div>
            <div><dt>Environment</dt><dd>{trialMode.environment}</dd></div>
            <div><dt>Clinical trial mode</dt><dd>{trialMode.clinicalTrialMode ? 'On' : 'Off'}</dd></div>
            <div><dt>Last changed by</dt><dd>{trialMode.updatedBy || 'Not changed'}</dd></div>
            <div><dt>Last changed</dt><dd>{trialMode.updatedAt ? new Date(trialMode.updatedAt).toLocaleString() : 'Not changed'}</dd></div>
            <div><dt>Reason</dt><dd>{trialMode.reason || 'Not provided'}</dd></div>
          </dl>
        ) : (
          <p style={styles.subtitle}>Loading hospital trial settings...</p>
        )}
        <p style={styles.trialModeNote}>This is read-only for hospital administrators. Trial mode does not create trials or authorize trial tasks.</p>
      </div>

      {/* Timing Settings Section */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>⏰ Timing & Threshold Settings</h2>
        <div style={styles.settingsGrid}>
          <SettingField
            label="Late Arrival Threshold"
            value={settings.lateArrivalThresholdMinutes}
            onChange={(v) => handleChange('lateArrivalThresholdMinutes', v)}
            unit="minutes"
            error={errors.lateArrivalThresholdMinutes}
            help="Minutes after scheduled time to trigger late arrival flag"
          />

          <SettingField
            label="Early Departure Threshold"
            value={settings.earlyDepartureThresholdMinutes}
            onChange={(v) => handleChange('earlyDepartureThresholdMinutes', v)}
            unit="minutes"
            help="Minutes before scheduled end time to trigger early departure flag"
          />

          <SettingField
            label="Clock-In Window"
            value={settings.clockInWindowMinutes}
            onChange={(v) => handleChange('clockInWindowMinutes', v)}
            unit="minutes"
            error={errors.clockInWindowMinutes}
            help="Time window before shift to allow clock-in"
          />
        </div>
      </div>

      {/* Break Settings Section */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>☕ Break Settings</h2>
        <div style={styles.settingsGrid}>
          <SettingField
            label="Mandatory Break Duration"
            value={settings.mandatoryBreakDurationMinutes}
            onChange={(v) => handleChange('mandatoryBreakDurationMinutes', v)}
            unit="minutes"
            error={errors.mandatoryBreakDurationMinutes}
            help="Minimum break duration required per shift"
          />

          <SettingField
            label="Maximum Break Duration"
            value={settings.maxBreakDurationMinutes}
            onChange={(v) => handleChange('maxBreakDurationMinutes', v)}
            unit="minutes"
            error={errors.maxBreakDurationMinutes}
            help="Maximum allowed break duration without approval"
          />

          <SettingField
            label="Max Consecutive Hours Without Break"
            value={settings.maxConsecutiveHoursWithoutBreak}
            onChange={(v) => handleChange('maxConsecutiveHoursWithoutBreak', v)}
            unit="hours"
            error={errors.maxConsecutiveHoursWithoutBreak}
            help="Maximum hours worked before mandatory break required"
          />

          <ToggleSetting
            label="Require Break Approval"
            value={settings.requireBreakApproval}
            onChange={(v) => handleChange('requireBreakApproval', v)}
            help="Breaks exceeding max duration require manager approval"
          />
        </div>
      </div>

      {/* Approval Settings Section */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>✅ Approval Settings</h2>
        <div style={styles.settingsGrid}>
          <ToggleSetting
            label="Require Late Arrival Approval"
            value={settings.requireLateApproval}
            onChange={(v) => handleChange('requireLateApproval', v)}
            help="Late arrivals require manager review and approval"
          />

          <ToggleSetting
            label="Enable Auto-Approval"
            value={settings.autoApprovalEnabled}
            onChange={(v) => handleChange('autoApprovalEnabled', v)}
            help="Automatically approve minor infractions after 24 hours"
          />

          <ToggleSetting
            label="Enable Notification Emails"
            value={settings.notificationEmailEnabled}
            onChange={(v) => handleChange('notificationEmailEnabled', v)}
            help="Send email notifications for pending approvals"
          />
        </div>
      </div>

      {/* Schedule Settings Section */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>📅 Schedule Settings</h2>
        <div style={styles.settingsGrid}>
          <SettingField
            label="Daily Schedule Start Hour"
            value={settings.dailyScheduleStartHour}
            onChange={(v) => handleChange('dailyScheduleStartHour', v)}
            unit="24-hr"
            error={errors.dailyScheduleStartHour}
            help="Earliest hour when shifts can start"
          />

          <SettingField
            label="Daily Schedule End Hour"
            value={settings.dailyScheduleEndHour}
            onChange={(v) => handleChange('dailyScheduleEndHour', v)}
            unit="24-hr"
            error={errors.dailyScheduleEndHour}
            help="Latest hour when shifts can end"
          />
        </div>
      </div>

      {/* Warning Section */}
      <div style={styles.warningSection}>
        <FaExclamationTriangle style={{ marginRight: '12px', color: '#d97706' }} />
        <div>
          <h3 style={styles.warningTitle}>Settings Impact</h3>
          <p style={styles.warningText}>
            Changes to these settings will apply immediately to all future clock-in/out events and ongoing shift approvals. 
            Existing records will not be retroactively affected.
          </p>
        </div>
      </div>

      {/* Save Button */}
      <div style={styles.actionBar}>
        <button
          onClick={handleSave}
          disabled={saving}
          style={{
            ...styles.saveBtn,
            opacity: saving ? 0.6 : 1,
            cursor: saving ? 'not-allowed' : 'pointer'
          }}
        >
          <FaSave style={{ marginRight: '8px' }} />
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}

function SettingField({ label, value, onChange, unit, error, help }) {
  return (
    <div style={styles.settingField}>
      <label style={styles.fieldLabel}>{label}</label>
      <div style={styles.inputGroup}>
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value))}
          style={{
            ...styles.input,
            borderColor: error ? '#dc2626' : '#d1d5db'
          }}
        />
        <span style={styles.unit}>{unit}</span>
      </div>
      {error && <p style={styles.errorText}>{error}</p>}
      {!error && help && <p style={styles.helpText}>{help}</p>}
    </div>
  );
}

function ToggleSetting({ label, value, onChange, help }) {
  return (
    <div style={styles.settingField}>
      <label style={styles.fieldLabel}>{label}</label>
      <div style={styles.toggleContainer}>
        <button
          onClick={() => onChange(!value)}
          style={{
            ...styles.toggleBtn,
            backgroundColor: value ? '#10b981' : '#d1d5db'
          }}
        >
          <div
            style={{
              ...styles.toggleCircle,
              transform: value ? 'translateX(22px)' : 'translateX(2px)'
            }}
          />
        </button>
        <span style={styles.toggleLabel}>{value ? 'Enabled' : 'Disabled'}</span>
      </div>
      {help && <p style={styles.helpText}>{help}</p>}
    </div>
  );
}

const styles = {
  container: {
    padding: '20px',
    maxWidth: '1000px',
    marginBottom: '40px'
  },
  header: {
    marginBottom: '24px'
  },
  subtitle: {
    color: '#6b7280',
    marginTop: '6px',
    fontSize: '15px'
  },
  loadingBox: {
    textAlign: 'center',
    padding: '60px 20px',
    color: '#6b7280',
    fontSize: '16px'
  },
  messageBanner: {
    padding: '12px 16px',
    borderRadius: '8px',
    border: '1px solid',
    marginBottom: '24px',
    fontSize: '14px'
  },
  section: {
    backgroundColor: 'white',
    padding: '24px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    marginBottom: '24px'
  },
  sectionTitle: {
    fontSize: '18px',
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: '20px',
    marginTop: 0
  },
  trialModeDetails: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '14px',
    margin: 0
  },
  trialModeError: {
    padding: '12px',
    border: '1px solid #fca5a5',
    borderRadius: '6px',
    background: '#fef2f2',
    color: '#991b1b',
    fontSize: '13px'
  },
  trialModeNote: {
    margin: '16px 0 0',
    color: '#6b7280',
    fontSize: '12px'
  },
  settingsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '24px'
  },
  settingField: {
    display: 'flex',
    flexDirection: 'column'
  },
  fieldLabel: {
    fontWeight: '600',
    color: '#374151',
    marginBottom: '8px',
    fontSize: '14px'
  },
  inputGroup: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center'
  },
  input: {
    flex: 1,
    padding: '8px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    fontFamily: 'inherit'
  },
  unit: {
    color: '#6b7280',
    fontSize: '13px',
    fontWeight: '500',
    minWidth: '60px'
  },
  helpText: {
    margin: '4px 0 0 0',
    fontSize: '12px',
    color: '#6b7280'
  },
  errorText: {
    margin: '6px 0 0 0',
    fontSize: '12px',
    color: '#dc2626'
  },
  toggleContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px'
  },
  toggleBtn: {
    width: '50px',
    height: '28px',
    borderRadius: '14px',
    border: 'none',
    cursor: 'pointer',
    position: 'relative',
    transition: 'background-color 0.2s'
  },
  toggleCircle: {
    width: '24px',
    height: '24px',
    backgroundColor: 'white',
    borderRadius: '12px',
    position: 'absolute',
    top: '2px',
    transition: 'transform 0.2s'
  },
  toggleLabel: {
    fontSize: '13px',
    color: '#6b7280',
    fontWeight: '500'
  },
  warningSection: {
    backgroundColor: '#fef3c7',
    border: '1px solid #fcd34d',
    padding: '16px',
    borderRadius: '8px',
    marginBottom: '24px',
    display: 'flex',
    alignItems: 'flex-start',
    color: '#78350f'
  },
  warningTitle: {
    margin: '0 0 6px 0',
    fontSize: '15px',
    fontWeight: '700'
  },
  warningText: {
    margin: 0,
    fontSize: '13px',
    lineHeight: '1.5'
  },
  actionBar: {
    display: 'flex',
    gap: '12px',
    paddingTop: '20px',
    borderTop: '1px solid #e5e7eb'
  },
  saveBtn: {
    padding: '12px 24px',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    fontSize: '15px'
  }
};
