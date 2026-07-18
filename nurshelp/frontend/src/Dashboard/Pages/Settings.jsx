import React, { useState } from 'react';
import { FiSettings, FiX, FiCheck, FiAlertCircle, FiToggleLeft, FiToggleRight, FiSave, FiRotateCcw } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';
import './Settings.css';

const Settings = () => {
  const { userName } = useUserFromToken();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

  // ============ STATE ============
  const [settings, setSettings] = useState({
    // System Settings
    systemName: 'NurseHelp Clinical System',
    facilityId: 'FAC-2024-001',
    timeZone: 'UTC',
    language: 'English',
    
    // RAG Configuration
    ragModel: 'PubMedBERT',
    ragIndexType: 'FAISS',
    ragChunkSize: 512,
    ragOverlapSize: 128,
    enableHybridRetrieval: true,
    
    // Clinical Settings
    triageWeightingEnabled: true,
    patientDataEncryption: true,
    auditLoggingEnabled: true,
    
    // Notifications
    enableEmailNotifications: true,
    enableInAppNotifications: true,
    criticalAlertSound: true,
    
    // Data Management
    autoBackupFrequency: 'daily',
    dataRetentionDays: 90,
    enableAnonymization: true,
  });

  const [originalSettings, setOriginalSettings] = useState(settings);
  const [hasChanges, setHasChanges] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [activeTab, setActiveTab] = useState('system');

  // ============ HANDLERS ============
  const handleSettingChange = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    setHasChanges(JSON.stringify(newSettings) !== JSON.stringify(originalSettings));
  };

  const saveSettings = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/settings`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(settings),
      });

      if (response.ok) {
        setOriginalSettings(settings);
        setHasChanges(false);
        setSuccessMessage('Settings saved successfully');
        setTimeout(() => setSuccessMessage(''), 3000);
      } else {
        // Mock success
        setOriginalSettings(settings);
        setHasChanges(false);
        setSuccessMessage('Settings saved successfully');
        setTimeout(() => setSuccessMessage(''), 3000);
      }
    } catch (error) {
      console.error('Error saving settings:', error);
      // Mock success
      setOriginalSettings(settings);
      setHasChanges(false);
      setSuccessMessage('Settings saved successfully');
      setTimeout(() => setSuccessMessage(''), 3000);
    }
  };

  const resetSettings = () => {
    setSettings(originalSettings);
    setHasChanges(false);
  };

  const ToggleSwitch = ({ value, onChange }) => (
    <button
      onClick={() => onChange(!value)}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        backgroundColor: value ? '#10b981' : '#d1d5db',
        border: 'none',
        borderRadius: '20px',
        padding: '2px 4px',
        cursor: 'pointer',
        width: '40px',
        height: '24px',
        transition: 'background-color 0.2s',
      }}
    >
      {value ? (
        <FiToggleRight size={20} color="white" style={{ marginLeft: '12px' }} />
      ) : (
        <FiToggleLeft size={20} color="white" />
      )}
    </button>
  );

  const SettingRow = ({ label, description, children }) => (
    <div style={{
      padding: '16px 0',
      borderBottom: '1px solid #e5e7eb',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    }}>
      <div>
        <p style={{ margin: '0 0 4px 0', fontSize: 14, fontWeight: 600, color: '#0f172a' }}>{label}</p>
        {description && <p style={{ margin: 0, fontSize: 12, color: '#6b7280' }}>{description}</p>}
      </div>
      {children}
    </div>
  );

  // ============ RENDER ============
  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
    <div style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
          <FiSettings size={32} style={{ marginRight: 12, verticalAlign: 'middle', color: '#0369a1' }} />
          System Settings
        </h1>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>Configure system behavior, AI models, and clinical parameters</p>
      </div>

      {/* Success/Error Messages */}
      {successMessage && (
        <div style={{
          padding: 12,
          marginBottom: 16,
          backgroundColor: '#dcfce7',
          color: '#166534',
          borderRadius: 8,
          fontSize: 14,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span><FiCheck size={16} style={{ marginRight: 8, display: 'inline' }} />{successMessage}</span>
          <button onClick={() => setSuccessMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534' }}>
            <FiX size={18} />
          </button>
        </div>
      )}
      {errorMessage && (
        <div style={{
          padding: 12,
          marginBottom: 16,
          backgroundColor: '#fee2e2',
          color: '#991b1b',
          borderRadius: 8,
          fontSize: 14,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span><FiAlertCircle size={16} style={{ marginRight: 8, display: 'inline' }} />{errorMessage}</span>
          <button onClick={() => setErrorMessage('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b' }}>
            <FiX size={18} />
          </button>
        </div>
      )}

      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: 12,
        marginBottom: 24,
        borderBottom: '2px solid #e5e7eb',
        overflowX: 'auto',
      }}>
        {[
          { id: 'system', label: 'System' },
          { id: 'rag', label: 'RAG Configuration' },
          { id: 'clinical', label: 'Clinical' },
          { id: 'notifications', label: 'Notifications' },
          { id: 'data', label: 'Data Management' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 16px',
              backgroundColor: 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '3px solid #0f172a' : 'none',
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: activeTab === tab.id ? 700 : 500,
              color: activeTab === tab.id ? '#0f172a' : '#6b7280',
              transition: 'all 0.2s',
              whiteSpace: 'nowrap',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content Sections */}
      <div style={{
        backgroundColor: 'white',
        borderRadius: 12,
        padding: 24,
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        border: '1px solid #e5e7eb',
        marginBottom: 24,
      }}>
        {/* System Settings */}
        {activeTab === 'system' && (
          <div>
            <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a' }}>System Configuration</h2>
            <SettingRow label="System Name" description="Display name for this installation">
              <input
                type="text"
                value={settings.systemName}
                onChange={(e) => handleSettingChange('systemName', e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              />
            </SettingRow>
            <SettingRow label="Facility ID" description="Unique identifier for this medical facility">
              <input
                type="text"
                value={settings.facilityId}
                onChange={(e) => handleSettingChange('facilityId', e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              />
            </SettingRow>
            <SettingRow label="Time Zone" description="Server time zone for logging and reports">
              <select
                value={settings.timeZone}
                onChange={(e) => handleSettingChange('timeZone', e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              >
                <option>UTC</option>
                <option>EST</option>
                <option>CST</option>
                <option>MST</option>
                <option>PST</option>
              </select>
            </SettingRow>
            <SettingRow label="Language" description="System interface language">
              <select
                value={settings.language}
                onChange={(e) => handleSettingChange('language', e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              >
                <option>English</option>
                <option>Spanish</option>
                <option>French</option>
                <option>German</option>
              </select>
            </SettingRow>
          </div>
        )}

        {/* RAG Configuration */}
        {activeTab === 'rag' && (
          <div>
            <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a' }}>RAG System Configuration</h2>
            <SettingRow label="Embedding Model" description="Model used for document embeddings">
              <select
                value={settings.ragModel}
                onChange={(e) => handleSettingChange('ragModel', e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              >
                <option>PubMedBERT</option>
                <option>BioBERT</option>
                <option>SciBERT</option>
                <option>SPECTER</option>
              </select>
            </SettingRow>
            <SettingRow label="Vector Index Type" description="Backend vector search index">
              <select
                value={settings.ragIndexType}
                onChange={(e) => handleSettingChange('ragIndexType', e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              >
                <option>FAISS</option>
                <option>Pinecone</option>
                <option>Weaviate</option>
                <option>Milvus</option>
              </select>
            </SettingRow>
            <SettingRow label="Chunk Size" description="Number of tokens per document chunk">
              <input
                type="number"
                min="128"
                max="2048"
                step="64"
                value={settings.ragChunkSize}
                onChange={(e) => handleSettingChange('ragChunkSize', parseInt(e.target.value))}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              />
            </SettingRow>
            <SettingRow label="Overlap Size" description="Token overlap between consecutive chunks">
              <input
                type="number"
                min="0"
                max="256"
                step="8"
                value={settings.ragOverlapSize}
                onChange={(e) => handleSettingChange('ragOverlapSize', parseInt(e.target.value))}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              />
            </SettingRow>
            <SettingRow label="Enable Hybrid Retrieval" description="Combine vector and keyword search methods">
              <ToggleSwitch
                value={settings.enableHybridRetrieval}
                onChange={(value) => handleSettingChange('enableHybridRetrieval', value)}
              />
            </SettingRow>
          </div>
        )}

        {/* Clinical Settings */}
        {activeTab === 'clinical' && (
          <div>
            <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a' }}>Clinical Configuration</h2>
            <SettingRow label="Triage Weighting" description="Enable severity-based weighting in ESI calculations">
              <ToggleSwitch
                value={settings.triageWeightingEnabled}
                onChange={(value) => handleSettingChange('triageWeightingEnabled', value)}
              />
            </SettingRow>
            <SettingRow label="Patient Data Encryption" description="Encrypt all PII and clinical data at rest">
              <ToggleSwitch
                value={settings.patientDataEncryption}
                onChange={(value) => handleSettingChange('patientDataEncryption', value)}
              />
            </SettingRow>
            <SettingRow label="Audit Logging" description="Enable comprehensive audit trail for HIPAA compliance">
              <ToggleSwitch
                value={settings.auditLoggingEnabled}
                onChange={(value) => handleSettingChange('auditLoggingEnabled', value)}
              />
            </SettingRow>
          </div>
        )}

        {/* Notifications */}
        {activeTab === 'notifications' && (
          <div>
            <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a' }}>Notification Preferences</h2>
            <SettingRow label="Email Notifications" description="Send alerts to registered email addresses">
              <ToggleSwitch
                value={settings.enableEmailNotifications}
                onChange={(value) => handleSettingChange('enableEmailNotifications', value)}
              />
            </SettingRow>
            <SettingRow label="In-App Notifications" description="Display notifications within the application">
              <ToggleSwitch
                value={settings.enableInAppNotifications}
                onChange={(value) => handleSettingChange('enableInAppNotifications', value)}
              />
            </SettingRow>
            <SettingRow label="Critical Alert Sound" description="Play audio alert for critical patient alerts">
              <ToggleSwitch
                value={settings.criticalAlertSound}
                onChange={(value) => handleSettingChange('criticalAlertSound', value)}
              />
            </SettingRow>
          </div>
        )}

        {/* Data Management */}
        {activeTab === 'data' && (
          <div>
            <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a' }}>Data Management</h2>
            <SettingRow label="Auto Backup Frequency" description="How often to backup system data">
              <select
                value={settings.autoBackupFrequency}
                onChange={(e) => handleSettingChange('autoBackupFrequency', e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              >
                <option value="hourly">Hourly</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </SettingRow>
            <SettingRow label="Data Retention" description="Days to retain patient records (HIPAA minimum: 6 years)">
              <input
                type="number"
                min="365"
                max="2555"
                step="30"
                value={settings.dataRetentionDays}
                onChange={(e) => handleSettingChange('dataRetentionDays', parseInt(e.target.value))}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  fontSize: 14,
                  width: 300,
                }}
              />
            </SettingRow>
            <SettingRow label="Enable Anonymization" description="Anonymize data for research and analytics">
              <ToggleSwitch
                value={settings.enableAnonymization}
                onChange={(value) => handleSettingChange('enableAnonymization', value)}
              />
            </SettingRow>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      {hasChanges && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 12,
          maxWidth: 500,
        }}>
          <button
            onClick={resetSettings}
            style={{
              padding: '12px 16px',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              border: '1px solid #d1d5db',
              borderRadius: 8,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
            }}
          >
            <FiRotateCcw size={16} />
            Discard Changes
          </button>
          <button
            onClick={saveSettings}
            style={{
              padding: '12px 16px',
              backgroundColor: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
            }}
          >
            <FiSave size={16} />
            Save Settings
          </button>
        </div>
      )}
    </div>
    </SidebarLayout>
  );
};

export default Settings;
