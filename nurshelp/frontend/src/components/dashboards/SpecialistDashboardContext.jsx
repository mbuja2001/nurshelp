import React, { useEffect, useState } from 'react';
import {
  FiActivity,
  FiAlertCircle,
  FiBookOpen,
  FiCheckCircle,
  FiClock,
  FiFileText,
  FiImage,
  FiLayers,
  FiShield,
  FiUserCheck
} from 'react-icons/fi';
import './SpecialistDashboardContext.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const workspaceContent = {
  RADIOLOGY: {
    title: 'Radiology',
    sections: [
      {
        id: 'specialist-viewer',
        icon: FiImage,
        title: 'Imaging workbench',
        code: 'RW / RV · IMG-VIEW',
        description: 'Study references, QC and technician flags, priors, vendor viewer links, and report context.',
        items: ['PACS/RIS integration and per-study IMG-VIEW authorization', 'Short-lived, per-user viewer links with an access log', 'QC-gated reporting, priors, de-identification and trial read type'],
        blockedBy: 'A hospital administrator must configure and test the PACS/RIS and EHR launch adapter.'
      },
      {
        id: 'specialist-reporting',
        icon: FiFileText,
        title: 'Reporting and release',
        code: 'RR · IMG-REPORT',
        description: 'Vendor RIS reports are received as released versions; trial safety fields and amendment history remain visible.',
        items: ['Safety-finding decision and investigator notification', 'Critical-finding call, read-back and acknowledgement', 'A second radiologist must sign before every report release'],
        blockedBy: 'The SIM-RIS adapter and server-side report/reviewer authorization are not connected yet.'
      },
      {
        id: 'specialist-handoffs',
        icon: FiActivity,
        title: 'Findings and handoffs',
        code: 'RF · IMG-INCID',
        description: 'Technician flags, repeat-scan decisions, physician requests and protocol ownership.',
        items: ['Incidental-finding decisions with a signed rationale', 'QC repeat-scan and radiation-justification decisions', 'Handoffs close only with a recorded decision'],
        blockedBy: 'The task catalogue, audit history and e-signature service are not connected yet.'
      },
      {
        id: 'specialist-governance',
        icon: FiShield,
        title: 'Protocol and quality',
        code: 'RF-08–10 · IMG-PROTOCOL-APPROVE',
        description: 'Versioned imaging protocol sets and technician task-catalogue approval.',
        items: ['Draft, approved and superseded protocol versions', 'Effective-date reminders for amendments', 'Signed catalogue history and role-specific restrictions'],
        blockedBy: 'Protocol versioning and specialist sign-off records are not configured.'
      }
    ]
  },
  PATHOLOGY: {
    title: 'Pathology',
    sections: [
      {
        id: 'specialist-viewer',
        icon: FiLayers,
        title: 'Cases, samples and custody',
        code: 'PW / PV · SAMPLE-VIEW / SLIDE-VIEW',
        description: 'Case priority, sample references, processing, stability, aliquots and tissue inventory.',
        items: ['Per-sample SAMPLE-VIEW authorization', 'Collection-to-shipment custody and stability history', 'Slide/block metadata; simulated viewer contains no images'],
        blockedBy: 'The LIS and digital-pathology metadata adapters are not connected yet.'
      },
      {
        id: 'specialist-reporting',
        icon: FiFileText,
        title: 'Results, diagnosis and release',
        code: 'PW-06–11 / PV-07–10 · PATH-REPORT',
        description: 'Released LIS results and diagnoses with their source version, validation and amendment history.',
        items: ['Histology/cytology diagnosis and interpretive comments', 'QC-gated result release and critical-value read-back', 'A second pathologist must sign before every report release'],
        blockedBy: 'The SIM-LIS adapter and server-side report/reviewer authorization are not connected yet.'
      },
      {
        id: 'specialist-handoffs',
        icon: FiActivity,
        title: 'Escalations and tissue requests',
        code: 'PG-01–10 · PATH-ESCALATION / LAB-QC-APPROVE / TISSUE-RELEASE',
        description: 'Scientist escalations, QC corrective action, sponsor requests, and lab deviations.',
        items: ['Signed decisions for delta-checks and QC failures', 'Consent and remaining-tissue checks before sponsor release', 'Shipment and return recorded in custody history'],
        blockedBy: 'The escalation inbox, consent evidence and append-only action history are not configured.'
      },
      {
        id: 'specialist-governance',
        icon: FiShield,
        title: 'Laboratory governance',
        code: 'PG-11–16',
        description: 'Task catalogue, SOP impact review, critical values and accreditation.',
        items: ['Versioned pathology task catalogue and training requirements', 'Signed lab-manual impact assessment', 'Critical-value list and ISO 15189 / SANAS status'],
        blockedBy: 'Laboratory governance records and approval workflows are not configured.'
      }
    ]
  }
};

const accessHelp = [
  ['TRIAL_MODE_DISABLED', 'Clinical trial mode is off. Routine work remains available; trial work is blocked.', 'Hospital administrator'],
  ['SHIFT_NOT_ACTIVE', 'A verified, active shift and clock-in are required for trial sign-offs.', 'Shift coordinator'],
  ['TRIAL_ASSIGNMENT_MISSING', 'No active site-specific study-team assignment was found.', 'Study coordinator'],
  ['DELEGATION_MISSING', 'The exact task has not been PI-delegated to this user.', 'Principal investigator'],
  ['TRAINING_EXPIRED', 'Current GCP, protocol-version and task training are required.', 'Training coordinator'],
  ['SECOND_READER_REQUIRED', 'A different qualified specialist must countersign before release.', 'Second reviewer']
];

export const SpecialistDashboardContext = ({ specialty, config = {} }) => {
  const workspace = workspaceContent[specialty];
  const [trialMode, setTrialMode] = useState({ status: 'loading', settings: null, error: '' });

  useEffect(() => {
    const controller = new AbortController();
    const token = localStorage.getItem('token');

    const loadTrialMode = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/hospital/clinical-trial-mode`, {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
          cache: 'no-store'
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.success) {
          throw new Error(data.message || `Unable to load hospital trial settings (${response.status})`);
        }
        setTrialMode({ status: 'loaded', settings: data.settings, error: '' });
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('[SPECIALIST DASHBOARD] Failed to load hospital trial settings:', error);
          setTrialMode({ status: 'error', settings: null, error: error.message || 'Unable to load hospital trial settings' });
        }
      }
    };

    loadTrialMode();
    return () => controller.abort();
  }, []);

  if (!workspace) return null;

  const environment = trialMode.settings?.environment?.toUpperCase();
  const onDuty = Boolean(config.shiftId);

  return (
    <section className={`specialist-context specialist-context-${specialty.toLowerCase()}`} aria-label={`${workspace.title} clinical-trial context`}>
      <div className="specialist-context-topline">
        <div>
          <p className="specialist-eyebrow">SPECIALIST WORKSPACE</p>
          <h2>{workspace.title} clinical operations</h2>
        </div>
        <div className={`specialist-environment-badge ${environment === 'SANDBOX' ? 'is-sandbox' : environment === 'PRODUCTION' ? 'is-production' : 'is-unknown'}`}>
          <FiActivity aria-hidden="true" />
          {environment === 'SANDBOX' ? 'Sandbox · mock data only' : environment === 'PRODUCTION' ? 'Production' : 'Environment unavailable'}
        </div>
      </div>

      <div className="specialist-context-grid">
        <div className="specialist-context-item">
          <span className="specialist-context-icon"><FiShield aria-hidden="true" /></span>
          <div>
            <span className="specialist-context-label">Clinical trial mode</span>
            <strong>
              {trialMode.status === 'loading' ? 'Checking status…'
                : trialMode.status === 'error' ? 'Status unavailable'
                  : trialMode.settings.clinicalTrialMode ? 'Enabled · task authorization still required' : 'Off · trial actions blocked'}
            </strong>
            {trialMode.status === 'error' && <small role="alert">{trialMode.error}</small>}
          </div>
        </div>
        <div className="specialist-context-item">
          <span className="specialist-context-icon"><FiClock aria-hidden="true" /></span>
          <div>
            <span className="specialist-context-label">Shift / duty status</span>
            <strong>{onDuty ? 'Clocked in · verified shift' : 'Not verified'}</strong>
            <small>{config.shiftName || 'Trial sign-offs remain blocked unless on duty.'}</small>
          </div>
        </div>
        <div className="specialist-context-item">
          <span className="specialist-context-icon"><FiUserCheck aria-hidden="true" /></span>
          <div>
            <span className="specialist-context-label">Trial assignment</span>
            <strong>Not configured</strong>
            <small>Trial, protocol version, site, PI and study role are not assigned.</small>
          </div>
        </div>
        <div className="specialist-context-item">
          <span className="specialist-context-icon"><FiBookOpen aria-hidden="true" /></span>
          <div>
            <span className="specialist-context-label">Blinding & access</span>
            <strong>Not verified</strong>
            <small>Allocation and arm data are not displayed in this workspace.</small>
          </div>
        </div>
      </div>

      <nav className="specialist-module-nav" aria-label={`${workspace.title} workflow sections`}>
        <a href="#specialist-worklist">Worklist</a>
        {workspace.sections.map(({ id, title, icon: Icon }) => (
          <a href={`#${id}`} key={id}><Icon aria-hidden="true" />{title}</a>
        ))}
        <a href="#specialist-access"><FiBookOpen aria-hidden="true" />My access</a>
      </nav>

      <div className="specialist-checklist-grid">
        {workspace.sections.map((section) => {
          const Icon = section.icon;
          return (
            <article className="specialist-checklist-card" id={section.id} key={section.id}>
              <div className="specialist-card-heading">
                <span className="specialist-module-icon"><Icon aria-hidden="true" /></span>
                <div>
                  <p className="specialist-eyebrow">{section.code}</p>
                  <h3>{section.title}</h3>
                </div>
              </div>
              <p className="specialist-card-description">{section.description}</p>
              <ul>
                {section.items.map((item) => <li key={item}>{item}</li>)}
              </ul>
              <div className="specialist-not-connected">
                <FiAlertCircle aria-hidden="true" />
                <div>
                  <strong>Not connected yet</strong>
                  <p>{section.blockedBy}</p>
                  <small>Contact the hospital administrator or clinical coordinator.</small>
                </div>
              </div>
            </article>
          );
        })}
      </div>

      <section className="specialist-access-card" id="specialist-access" aria-labelledby="specialist-access-title">
        <div className="specialist-card-heading">
          <span className="specialist-module-icon"><FiCheckCircle aria-hidden="true" /></span>
          <div>
            <p className="specialist-eyebrow">SH-07 / SH-08 / SH-12</p>
            <h3 id="specialist-access-title">My access & why actions are blocked</h3>
          </div>
        </div>
        <p>GCP expiry, protocol/task training, HPCSA registration, delegated task codes and e-signature re-authentication are not yet available from the server.</p>
        <div className="specialist-deny-reasons">
          {accessHelp.map(([code, explanation, owner]) => (
            <div className="specialist-deny-reason" key={code}>
              <code>{code}</code>
              <span>{explanation}</span>
              <small>Fix with: {owner}</small>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
};

export default SpecialistDashboardContext;
