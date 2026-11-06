import React, { useEffect, useState } from 'react';
import { FiActivity, FiCheck, FiCopy, FiGrid, FiHome, FiLogOut, FiMenu, FiPlus, FiRefreshCw, FiUser } from 'react-icons/fi';
import './AccountDashboard.css';

const AccountDashboard = ({ account, onLogout }) => {
  const [activeView, setActiveView] = useState('overview');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hospitals, setHospitals] = useState([]);
  const [hospitalsLoading, setHospitalsLoading] = useState(true);
  const [hospitalError, setHospitalError] = useState('');
  const [hospitalFormOpen, setHospitalFormOpen] = useState(false);
  const [hospitalSaving, setHospitalSaving] = useState(false);
  const [hospitalForm, setHospitalForm] = useState({
    hospitalName: '',
    country: '',
    environment: 'SANDBOX',
    clinicalTrialMode: false,
    clinicalTrialModeReason: '',
    adminName: account.name || '',
    adminSurname: '',
    adminEmail: '',
    adminPassword: ''
  });
  const [credentialsToShow, setCredentialsToShow] = useState(null);
  const [credentialAction, setCredentialAction] = useState(null);
  const [credentialPassword, setCredentialPassword] = useState('');
  const [credentialMessage, setCredentialMessage] = useState('');
  const [copiedCredentials, setCopiedCredentials] = useState(false);
  const [trialModeAction, setTrialModeAction] = useState(null);
  const [trialModeReason, setTrialModeReason] = useState('');
  const [trialCountResults, setTrialCountResults] = useState({});
  const [ehrAccessAction, setEhrAccessAction] = useState(null);

  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

  const loadHospitals = async () => {
    setHospitalsLoading(true);
    setHospitalError('');
    try {
      const response = await fetch(`${backendUrl}/api/account/hospitals`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to load hospital workspaces');
      setHospitals(Array.isArray(data.hospitals) ? data.hospitals : []);
    } catch (error) {
      setHospitalError(error.message || 'Unable to load hospital workspaces');
    } finally {
      setHospitalsLoading(false);
    }
  };

  useEffect(() => {
    loadHospitals();
  }, []);

  const createHospital = async (event) => {
    event.preventDefault();
    setHospitalSaving(true);
    setHospitalError('');
    try {
      const response = await fetch(`${backendUrl}/api/account/hospitals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${localStorage.getItem('token')}` },
        body: JSON.stringify(hospitalForm)
      });
      const data = await response.json().catch(() => ({}));
      if (data.partial && data.hospital) {
        setHospitals((current) => [data.hospital, ...current]);
        setHospitalError(data.message || 'Hospital was created without an administrator. Add an administrator to recover.');
        setHospitalFormOpen(false);
        setHospitalForm({ hospitalName: '', country: '', environment: 'SANDBOX', clinicalTrialMode: false, clinicalTrialModeReason: '', adminName: account.name || '', adminSurname: '', adminEmail: '', adminPassword: '' });
        return;
      }
      if (!response.ok) throw new Error(data.message || 'Unable to create hospital');
      setHospitals((current) => [data.hospital, ...current]);
      setCredentialsToShow({ hospitalName: data.hospital.name, ...data.initialAdminCredentials });
      setHospitalForm({ hospitalName: '', country: '', environment: 'SANDBOX', clinicalTrialMode: false, clinicalTrialModeReason: '', adminName: account.name || '', adminSurname: '', adminEmail: '', adminPassword: '' });
      setHospitalFormOpen(false);
    } catch (error) {
      setHospitalError(error.message || 'Unable to create hospital');
    } finally {
      setHospitalSaving(false);
    }
  };

  const startAdminAction = (hospital, admin, action) => {
    setCredentialAction({ hospital, admin, action });
    setCredentialPassword('');
    setCredentialMessage('');
    setHospitalError('');
  };

  const saveAdminCredential = async (event) => {
    event.preventDefault();
    if (!credentialAction) return;
    const isNewAdmin = credentialAction.action === 'add';
    const url = isNewAdmin
      ? `${backendUrl}/api/account/hospitals/${credentialAction.hospital.id}/admins`
      : `${backendUrl}/api/account/hospitals/${credentialAction.hospital.id}/admins/${credentialAction.admin.id}/reset-password`;
    const body = isNewAdmin
      ? { ...credentialAction.admin, password: credentialPassword }
      : { password: credentialPassword };

    setHospitalSaving(true);
    setHospitalError('');
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${localStorage.getItem('token')}` },
        body: JSON.stringify(body)
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to update administrator credentials');

      if (isNewAdmin) {
        setHospitals((current) => current.map((hospital) => hospital.id === credentialAction.hospital.id
          ? { ...hospital, admins: [...(hospital.admins || []), data.admin] }
          : hospital));
      }
      setCredentialsToShow({ hospitalName: credentialAction.hospital.name, ...(data.credentials || data.initialAdminCredentials) });
      setCredentialAction(null);
      setCredentialPassword('');
    } catch (error) {
      setHospitalError(error.message || 'Unable to update administrator credentials');
    } finally {
      setHospitalSaving(false);
    }
  };

  const copyCredentials = async () => {
    if (!credentialsToShow) return;
    const text = `Hospital: ${credentialsToShow.hospitalName}\nName: ${credentialsToShow.name}\nEmail: ${credentialsToShow.email}\nPassword: ${credentialsToShow.password}`;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedCredentials(true);
      window.setTimeout(() => setCopiedCredentials(false), 1800);
    } catch {
      setCredentialMessage('Clipboard access is unavailable. Select and copy the credentials below.');
    }
  };

  const updateHospitalField = (field, value) => setHospitalForm((current) => ({ ...current, [field]: value }));

  const startTrialModeChange = (hospital, enabled) => {
    setTrialModeAction({ hospital, enabled });
    setTrialModeReason('');
    setHospitalError('');
  };

  const saveTrialModeChange = async (event) => {
    event.preventDefault();
    if (!trialModeAction) return;
    setHospitalSaving(true);
    setHospitalError('');
    try {
      const response = await fetch(`${backendUrl}/api/account/hospitals/${trialModeAction.hospital.id}/clinical-trial-mode`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Authorization: `******'token')}` },
        body: JSON.stringify({ clinicalTrialMode: trialModeAction.enabled, reason: trialModeReason.trim() })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to update clinical trial mode');
      setHospitals((current) => current.map((hospital) => hospital.id === trialModeAction.hospital.id
        ? { ...hospital, ...data.hospital }
        : hospital));
      setTrialModeAction(null);
      setTrialModeReason('');
    } catch (error) {
      setHospitalError(error.message || 'Unable to update clinical trial mode');
    } finally {
      setHospitalSaving(false);
    }
  };

  const reconcileTrialCount = async (hospital) => {
    setHospitalError('');
    try {
      const response = await fetch(`${backendUrl}/api/account/hospitals/${hospital.id}/clinical-trial-mode/reconcile`, {
        method: 'POST',
        headers: { Authorization: `******'token')}` }
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to check the active-trial count');
      const result = data.reconciliation;
      setTrialCountResults((current) => ({
        ...current,
        [hospital.id]: result.matches
          ? `Count verified: ${result.activeTrialActualCount} active trial(s).`
          : `Mismatch detected and logged: counter ${result.activeTrialCount}, actual ${result.activeTrialActualCount}.`
      }));
    } catch (error) {
      setHospitalError(error.message || 'Unable to check the active-trial count');
    }
  };

  const startEhrAccessAction = (hospital, clearAccountConfiguration = false) => {
    setEhrAccessAction({
      hospital,
      clearAccountConfiguration,
      enabledProtocols: hospital.ehrIntegrationAccess?.enabledProtocols || [],
      reason: '',
    });
    setHospitalError('');
  };

  const saveEhrAccess = async (event) => {
    event.preventDefault();
    if (!ehrAccessAction) return;
    setHospitalSaving(true);
    setHospitalError('');
    try {
      const response = await fetch(`${backendUrl}/api/account/hospitals/${ehrAccessAction.hospital.id}/ehr-integration-access`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Authorization: `******'token')}` },
        body: JSON.stringify({
          enabledProtocols: ehrAccessAction.enabledProtocols,
          clearAccountConfiguration: ehrAccessAction.clearAccountConfiguration,
          reason: ehrAccessAction.reason.trim(),
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to update EHR integration access');
      setHospitals((current) => current.map((hospital) => hospital.id === ehrAccessAction.hospital.id
        ? { ...hospital, ehrIntegrationAccess: data.ehrIntegrationAccess }
        : hospital));
      setEhrAccessAction(null);
    } catch (error) {
      setHospitalError(error.message || 'Unable to update EHR integration access');
    } finally {
      setHospitalSaving(false);
    }
  };

  const navigation = [
    { id: 'overview', label: 'Overview', icon: FiGrid },
    { id: 'hospitals', label: 'Hospitals', icon: FiHome },
    { id: 'account', label: 'Account', icon: FiUser }
  ];

  const activeLabel = navigation.find((item) => item.id === activeView)?.label || 'Overview';

  return (
    <div className="account-shell">
      <aside className={`account-sidebar${sidebarOpen ? ' account-sidebar-open' : ''}`}>
        <div className="account-brand"><span className="account-brand-mark">N</span><span>Nurshelpdesk<small>Sandbox workspace</small></span></div>
        <div className="account-nav-label">WORKSPACE</div>
        <nav aria-label="Account navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" className={`account-nav-item${activeView === id ? ' account-nav-active' : ''}`} onClick={() => { setActiveView(id); setSidebarOpen(false); }}>
              <Icon aria-hidden="true" /> <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="account-sidebar-spacer" />
        <div className="account-identity"><span className="account-avatar">{(account.name || 'A').slice(0, 1).toUpperCase()}</span><span className="account-identity-copy"><strong>{account.name}</strong><small>Account owner</small></span></div>
        <button type="button" className="account-logout" onClick={onLogout}><FiLogOut aria-hidden="true" /> Sign out</button>
      </aside>

      {sidebarOpen && <button className="account-sidebar-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
      <main className="account-main">
        <header className="account-topbar">
          <button className="account-menu-toggle" type="button" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><FiMenu /></button>
          <div><span className="account-eyebrow">ACCOUNT WORKSPACE</span><h1>{activeLabel}</h1></div>
          <span className="account-status"><span /> Sandbox account</span>
        </header>

        {activeView === 'overview' && (
          <section className="account-content" aria-labelledby="account-welcome-title">
            <div className="account-welcome">
              <div className="account-welcome-icon"><FiActivity /></div>
              <div><div className="account-eyebrow">YOUR NURSHELPDESK ACCOUNT</div><h2 id="account-welcome-title">Welcome, {account.name.split(' ')[0]}</h2><p>This is your account workspace. Hospital workspaces you create will be listed here and kept separate from hospital staff sign-in.</p></div>
            </div>
            <div className="account-metrics">
              <article className="account-metric"><span>Hospitals</span><strong>{hospitalsLoading ? '...' : hospitals.length}</strong><small>Owned by this account</small></article>
              <article className="account-metric"><span>Live integrations</span><strong>{hospitalsLoading ? '...' : hospitals.filter((hospital) => hospital.integrationStatus === 'LIVE').length}</strong><small>Verified hospital connections</small></article>
              <article className="account-metric"><span>Access requests</span><strong>{hospitalsLoading ? '...' : hospitals.reduce((total, hospital) => total + (hospital.pendingAccessRequests || 0), 0)}</strong><small>Awaiting company review</small></article>
            </div>
            <section className="account-section"><div className="account-section-heading"><div><div className="account-eyebrow">HOSPITAL WORKSPACES</div><h2>Your hospitals</h2></div><button type="button" className="account-text-button" onClick={() => setActiveView('hospitals')}>View hospitals</button></div>{hospitals.slice(0, 3).map((hospital) => <div key={hospital.id} className="account-hospital-row"><div><strong>{hospital.name}</strong><span>{hospital.hospitalID} · {hospital.admins?.length || 0} admins</span></div><span className="account-offline-status"><i /> Offline</span></div>)}{!hospitalsLoading && hospitals.length === 0 && <div className="account-empty-state"><FiHome /><strong>No hospital workspaces yet</strong><span>Create a hospital workspace to get started.</span></div>}</section>
          </section>
        )}

        {activeView === 'hospitals' && (
          <section className="account-content account-hospitals-view">
            <div className="account-section-heading"><div><div className="account-eyebrow">ACCOUNT OWNERSHIP</div><h2>Hospital workspaces</h2></div><div className="account-heading-actions"><button type="button" className="account-icon-button" aria-label="Refresh hospitals" title="Refresh hospitals" onClick={loadHospitals} disabled={hospitalsLoading}><FiRefreshCw /></button><button type="button" className="account-create-button" onClick={() => { setHospitalFormOpen((open) => !open); setHospitalError(''); }}><FiPlus /> Create hospital</button></div></div>
            {hospitalError && <div className="account-alert" role="alert">{hospitalError}</div>}
            {credentialsToShow && <section className="account-credential-reveal" aria-label="New administrator credentials"><div className="account-credential-heading"><div><div className="account-eyebrow">SAVE THESE CREDENTIALS</div><h3>Hospital admin login</h3></div><button type="button" className="account-icon-button" aria-label="Dismiss credentials" title="Dismiss credentials" onClick={() => { setCredentialsToShow(null); setCredentialMessage(''); }}><FiCheck /></button></div><p>Password is shown only this time. Save it securely; it cannot be retrieved from the system later.</p><dl><div><dt>Hospital</dt><dd>{credentialsToShow.hospitalName}</dd></div><div><dt>Admin</dt><dd>{credentialsToShow.name}</dd></div><div><dt>Email</dt><dd>{credentialsToShow.email}</dd></div><div><dt>Password</dt><dd><code>{credentialsToShow.password}</code></dd></div></dl><div className="account-credential-actions"><button type="button" className="account-create-button" onClick={copyCredentials}><FiCopy /> {copiedCredentials ? 'Copied' : 'Copy credentials'}</button><span>Log in at <strong>/dashboard</strong> using these credentials.</span></div>{credentialMessage && <div className="account-inline-message" role="status">{credentialMessage}</div>}</section>}
            {hospitalFormOpen && (
              <form className="account-form-panel" onSubmit={createHospital}>
                <div className="account-form-heading">
                  <div><div className="account-eyebrow">NEW WORKSPACE</div><h3>Create a hospital</h3></div>
                </div>
                <div className="account-form-grid">
                  <label>Hospital name<input value={hospitalForm.hospitalName} onChange={(event) => updateHospitalField('hospitalName', event.target.value)} required /></label>
                  <label>Country<input value={hospitalForm.country} onChange={(event) => updateHospitalField('country', event.target.value)} placeholder="United States" /></label>
                  <label>
                    Environment
                    <select value={hospitalForm.environment} onChange={(event) => updateHospitalField('environment', event.target.value)}>
                      <option value="SANDBOX">Sandbox — mocked data only</option>
                    </select>
                  </label>
                  <label>Admin first name<input value={hospitalForm.adminName} onChange={(event) => updateHospitalField('adminName', event.target.value)} required /></label>
                  <label>Admin surname<input value={hospitalForm.adminSurname} onChange={(event) => updateHospitalField('adminSurname', event.target.value)} required /></label>
                  <label>Admin login email<input type="email" value={hospitalForm.adminEmail} onChange={(event) => updateHospitalField('adminEmail', event.target.value)} required /></label>
                  <label>Admin password<input type="password" autoComplete="new-password" minLength={8} value={hospitalForm.adminPassword} onChange={(event) => updateHospitalField('adminPassword', event.target.value)} required /></label>
                  <label className="account-trial-mode-choice">
                    <input
                      type="checkbox"
                      checked={hospitalForm.clinicalTrialMode}
                      onChange={(event) => setHospitalForm((current) => ({
                        ...current,
                        clinicalTrialMode: event.target.checked,
                        clinicalTrialModeReason: event.target.checked ? current.clinicalTrialModeReason : ''
                      }))}
                    />
                    Enable clinical trial mode (starts with no trials configured)
                  </label>
                  {hospitalForm.clinicalTrialMode && (
                    <label className="account-wide-field">
                      Reason for enabling trial mode
                      <textarea
                        value={hospitalForm.clinicalTrialModeReason}
                        onChange={(event) => updateHospitalField('clinicalTrialModeReason', event.target.value)}
                        required
                      />
                    </label>
                  )}
                </div>
                <div className="account-form-footer">
                  <span>The admin account is hospital-specific. Sandbox workspaces must use mocked data only.</span>
                  <button type="submit" className="account-create-button" disabled={hospitalSaving}>{hospitalSaving ? 'Creating...' : 'Create hospital'}</button>
                </div>
              </form>
            )}
            {credentialAction && <form className="account-admin-form" onSubmit={saveAdminCredential}><div className="account-form-heading"><div><div className="account-eyebrow">{credentialAction.action === 'add' ? 'HOSPITAL ACCESS' : 'CREDENTIAL RESET'}</div><h3>{credentialAction.action === 'add' ? `Add admin to ${credentialAction.hospital.name}` : `Reset ${credentialAction.admin.name}'s password`}</h3></div><button type="button" className="account-text-button" onClick={() => setCredentialAction(null)}>Cancel</button></div>{credentialAction.action === 'add' && <div className="account-form-grid"><label>First name<input value={credentialAction.admin.name || ''} onChange={(event) => setCredentialAction((current) => ({ ...current, admin: { ...current.admin, name: event.target.value } }))} required /></label><label>Surname<input value={credentialAction.admin.surname || ''} onChange={(event) => setCredentialAction((current) => ({ ...current, admin: { ...current.admin, surname: event.target.value } }))} required /></label><label>Admin login email<input type="email" value={credentialAction.admin.email || ''} onChange={(event) => setCredentialAction((current) => ({ ...current, admin: { ...current.admin, email: event.target.value } }))} required /></label></div>}<div className="account-admin-form-footer"><label>{credentialAction.action === 'add' ? 'Temporary password' : 'New password'}<input type="password" autoComplete="new-password" minLength={8} value={credentialPassword} onChange={(event) => setCredentialPassword(event.target.value)} required /></label><button className="account-create-button" type="submit" disabled={hospitalSaving}>{hospitalSaving ? 'Saving...' : credentialAction.action === 'add' ? 'Create admin' : 'Reset password'}</button></div></form>}
            {hospitalsLoading ? (
              <div className="account-empty-state account-loading-state">Loading hospital workspaces...</div>
            ) : hospitals.length ? (
              <div className="account-hospital-list">
                {hospitals.map((hospital) => (
                  <article className="account-hospital-card" key={hospital.id}>
                    <div className="account-hospital-card-heading">
                      <div>
                        <div className="account-eyebrow">{hospital.hospitalID}</div>
                        <h3>{hospital.name}</h3>
                        <span>{hospital.country || 'Country not set'} · {hospital.departments?.length || 0} departments</span>
                      </div>
                      <span className="account-offline-status"><i /> Offline</span>
                    </div>
                    <div className="account-trial-summary">
                      <div><span>Environment</span><strong>{hospital.environment || 'SANDBOX'}</strong></div>
                      <div><span>Clinical trial mode</span><strong>{hospital.clinicalTrialMode ? 'On' : 'Off'}</strong></div>
                      {hospital.clinicalTrialModeUpdatedAt && (
                        <p>
                          Last changed {new Date(hospital.clinicalTrialModeUpdatedAt).toLocaleString()}
                          {hospital.clinicalTrialModeUpdatedByName ? ` by ${hospital.clinicalTrialModeUpdatedByName}` : ''}
                          {hospital.clinicalTrialModeReason ? ` — ${hospital.clinicalTrialModeReason}` : ''}
                        </p>
                      )}
                    </div>
                    <div className="account-trial-actions">
                      <button
                        type="button"
                        className="account-text-button"
                        disabled={hospitalSaving}
                        onClick={() => startTrialModeChange(hospital, !hospital.clinicalTrialMode)}
                      >
                        {hospital.clinicalTrialMode ? 'Disable trial mode' : 'Enable trial mode'}
                      </button>
                      <button
                        type="button"
                        className="account-text-button"
                        onClick={() => reconcileTrialCount(hospital)}
                      >
                        Check active-trial count
                      </button>
                    </div>
                    {trialCountResults[hospital.id] && <div className="account-inline-message" role="status">{trialCountResults[hospital.id]}</div>}
                    {trialModeAction?.hospital.id === hospital.id && (
                      <form className="account-trial-form" onSubmit={saveTrialModeChange}>
                        <p>{trialModeAction.enabled ? 'Enabling this setting does not create trials.' : 'Trial mode cannot be disabled while trials are active.'}</p>
                        <label>
                          Reason
                          <textarea value={trialModeReason} onChange={(event) => setTrialModeReason(event.target.value)} required />
                        </label>
                        <div>
                          <button type="button" className="account-text-button" onClick={() => setTrialModeAction(null)}>Cancel</button>
                          <button type="submit" className="account-create-button" disabled={hospitalSaving}>
                            {hospitalSaving ? 'Saving...' : 'Confirm change'}
                          </button>
                        </div>
                      </form>
                    )}
                    <section className="account-trial-summary" aria-label="EHR integration access">
                      <div>
                        <span>EHR integration access</span>
                        <strong>
                          {hospital.ehrIntegrationAccess?.source === 'ACCOUNT_OWNER'
                            ? `${(hospital.ehrIntegrationAccess.enabledProtocols || []).map((protocol) => protocol.toUpperCase()).join(', ') || 'No protocols enabled'} — account policy`
                            : 'Not set at account level'}
                        </strong>
                      </div>
                      {hospital.environment === 'SANDBOX' && (
                        <p>
                          This is a sandbox access attestation only. It does not grant a third-party license or confirm a live connection.
                          {!hospital.ehrIntegrationAccess?.source && ' Hospital admins can configure sandbox access from EHR Configurations.'}
                        </p>
                      )}
                      <div className="account-trial-actions">
                        <button type="button" className="account-text-button" onClick={() => startEhrAccessAction(hospital)} disabled={hospitalSaving}>
                          {hospital.ehrIntegrationAccess?.source === 'ACCOUNT_OWNER' ? 'Edit EHR access' : 'Set EHR access'}
                        </button>
                        {hospital.ehrIntegrationAccess?.source === 'ACCOUNT_OWNER' && (
                          <button type="button" className="account-text-button" onClick={() => startEhrAccessAction(hospital, true)} disabled={hospitalSaving}>
                            Remove account policy
                          </button>
                        )}
                      </div>
                    </section>
                    {ehrAccessAction?.hospital.id === hospital.id && (
                      <form className="account-trial-form" onSubmit={saveEhrAccess}>
                        <p>
                          Configure which protocols this sandbox may prepare. This records an attestation; production licensing and connection testing are separate.
                        </p>
                        {!ehrAccessAction.clearAccountConfiguration && (
                          <div className="account-form-grid">
                            {[
                              ['hl7v2', 'HL7 v2'],
                              ['fhir-r4', 'FHIR R4'],
                            ].map(([protocol, label]) => (
                              <label className="account-trial-mode-choice" key={protocol}>
                                <input
                                  type="checkbox"
                                  checked={ehrAccessAction.enabledProtocols.includes(protocol)}
                                  onChange={(event) => setEhrAccessAction((current) => ({
                                    ...current,
                                    enabledProtocols: event.target.checked
                                      ? [...new Set([...current.enabledProtocols, protocol])]
                                      : current.enabledProtocols.filter((enabledProtocol) => enabledProtocol !== protocol),
                                  }))}
                                />
                                Allow {label} configuration
                              </label>
                            ))}
                          </div>
                        )}
                        <label>
                          Reason for this access change
                          <textarea
                            value={ehrAccessAction.reason}
                            onChange={(event) => setEhrAccessAction((current) => ({ ...current, reason: event.target.value }))}
                            required
                          />
                        </label>
                        <div>
                          <button type="button" className="account-text-button" onClick={() => setEhrAccessAction(null)}>Cancel</button>
                          <button type="submit" className="account-create-button" disabled={hospitalSaving}>
                            {hospitalSaving ? 'Saving...' : ehrAccessAction.clearAccountConfiguration ? 'Remove account policy' : 'Save EHR access'}
                          </button>
                        </div>
                      </form>
                    )}
                    <div className="account-admins-heading">
                      <div><div className="account-eyebrow">HOSPITAL ADMIN CREDENTIALS</div><strong>Admin sign-in accounts</strong></div>
                      <button type="button" className="account-text-button" onClick={() => startAdminAction(hospital, { name: '', surname: '', email: '' }, 'add')}><FiPlus /> Add admin</button>
                    </div>
                    {hospital.admins?.length ? (
                      <div className="account-admin-list">
                        {hospital.admins.map((admin) => (
                          <div className="account-admin-row" key={admin.id}>
                            <span className={`account-admin-indicator${admin.isActive ? ' account-admin-active' : ''}`} />
                            <div className="account-admin-info"><strong>{admin.name} {admin.surname}</strong><span>{admin.email}</span></div>
                            <span className="account-admin-role">{admin.isActive ? 'Admin' : 'Inactive'}</span>
                            <button type="button" className="account-reset-button" onClick={() => startAdminAction(hospital, admin, 'reset')}>Reset password</button>
                          </div>
                        ))}
                      </div>
                    ) : <div className="account-empty-inline">No hospital admin credentials are configured.</div>}
                  </article>
                ))}
              </div>
            ) : !hospitalFormOpen && (
              <div className="account-empty-state">
                <FiHome /><strong>No hospital workspaces yet</strong>
                <span>Create a hospital workspace to establish its first separate admin login.</span>
                <button type="button" className="account-create-button" onClick={() => setHospitalFormOpen(true)}><FiPlus /> Create hospital</button>
              </div>
            )}
          </section>
        )}

        {activeView === 'account' && (
          <section className="account-content"><div className="account-section-heading"><div><div className="account-eyebrow">SIGN-IN IDENTITY</div><h2>Account details</h2></div></div><div className="account-details"><div><span>Name</span><strong>{account.name}</strong></div><div><span>Email</span><strong>{account.email}</strong></div><div><span>Account type</span><strong>Sandbox account owner</strong></div></div></section>
        )}
      </main>
    </div>
  );
};

export default AccountDashboard;