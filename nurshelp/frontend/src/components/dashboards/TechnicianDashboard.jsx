import React from 'react';
import {
  FiAlertCircle,
  FiClock,
  FiFileText,
  FiImage,
  FiPackage,
  FiShield,
  FiThermometer,
  FiTag,
  FiUploadCloud
} from 'react-icons/fi';
import SidebarLayout from '../../Dashboard/SidebarLayout';
import './TechnicianDashboard.css';

const SPECIALTY_LABELS = {
  PHARMACY: 'Pharmacy',
  RADIOLOGY: 'Radiology',
  PATHOLOGY: 'Pathology'
};

const TechnicianDashboard = ({ config = {}, specialty }) => {
  const specialtyLabel = SPECIALTY_LABELS[specialty];
  const isRadiology = specialty === 'RADIOLOGY';
  const isPathology = specialty === 'PATHOLOGY';
  const queueSummaries = [
    { label: isRadiology ? 'Protocol scans' : isPathology ? 'Samples in progress' : `${specialtyLabel} tasks`, detail: 'Today', icon: FiFileText },
    { label: isPathology ? 'Stability deadlines' : 'Visit windows', detail: isPathology ? 'Time-sensitive samples' : 'Closing today', icon: FiClock },
    { label: isRadiology ? 'Image QC' : isPathology ? 'Shipment readiness' : `${specialtyLabel} reviews`, detail: isRadiology ? 'Awaiting review' : 'Awaiting action', icon: isPathology ? FiPackage : FiImage },
    { label: isPathology ? 'Custody alerts' : 'Transfers', detail: 'Status', icon: isPathology ? FiTag : FiUploadCloud }
  ];
  const shiftHours = config.shiftStartTime && config.shiftEndTime
    ? `${config.shiftStartTime} – ${config.shiftEndTime}`
    : 'Not specified';

  return (
    <SidebarLayout>
      <div className="technician-dashboard">
        {!specialtyLabel ? (
          <section className="technician-access-card" role="alert">
            <FiAlertCircle aria-hidden="true" />
            <div>
              <p className="technician-eyebrow">Technician access</p>
              <h1>Specialty assignment required</h1>
              <p>No recognized specialty is configured for this technician account. Please contact your coordinator.</p>
            </div>
          </section>
        ) : (
          <>
            <header className="technician-page-heading">
              <div>
                <p className="technician-eyebrow">{specialtyLabel} · Technician workstation</p>
                <h1>{specialtyLabel} technician</h1>
                <p>Shift overview and authorized work queues.</p>
              </div>
              <span className="technician-shift-status"><span aria-hidden="true" /> Clocked in</span>
            </header>

            <section className="technician-shift-summary" aria-label="Current shift">
              <div className="technician-shift-summary-icon" aria-hidden="true"><FiClock /></div>
              <div className="technician-shift-summary-main">
                <span className="technician-section-label">Current shift</span>
                <strong>{config.shiftName || 'Assigned shift'}</strong>
              </div>
              <div className="technician-shift-summary-detail">
                <span className="technician-section-label">Department</span>
                <strong>{config.department || 'Not specified'}</strong>
              </div>
              <div className="technician-shift-summary-detail">
                <span className="technician-section-label">Shift hours</span>
                <strong>{shiftHours}</strong>
              </div>
            </section>

            {isPathology && (
              <section className="technician-lab-context" aria-label="Pathology study context">
                <div><span className="technician-section-label">Trial</span><strong>Not configured</strong></div>
                <div><span className="technician-section-label">Lab manual</span><strong>Not configured</strong></div>
                <div><span className="technician-section-label">Central laboratory</span><strong>Not configured</strong></div>
                <div><span className="technician-section-label">Blinding status</span><strong>Not verified</strong></div>
                <div className="technician-barcode-field">
                  <FiTag aria-hidden="true" />
                  <span>Barcode accession unavailable</span>
                </div>
              </section>
            )}

            <div className="technician-authorization-notice" role="status">
              <FiShield aria-hidden="true" />
              <div>
                <strong>{isPathology ? 'Sample worklists are not available yet' : 'Clinical worklists are not available yet'}</strong>
                <p>{isPathology
                  ? 'Study and laboratory access, PI-approved task delegation, current training, and protocol authorization must be confirmed before specimen details or custody actions can appear.'
                  : 'Study/site access, task delegation, current training, and protocol authorization must be confirmed before patient or study tasks can appear.'}</p>
              </div>
              <span>Contact your coordinator</span>
            </div>

            <div className="technician-dashboard-grid">
              <div className="technician-dashboard-primary">
                <section className="technician-queue-summary" aria-label="Work queue availability">
                  {queueSummaries.map(({ label, detail, icon: Icon }) => (
                    <article className="technician-queue-summary-card" key={label}>
                      <div className="technician-queue-summary-heading"><Icon aria-hidden="true" /><span>{label}</span></div>
                      <strong aria-label={`${label} unavailable`}>—</strong>
                      <span>{detail} · unavailable</span>
                    </article>
                  ))}
                </section>

                <section className="technician-panel technician-worklist-panel">
                  <div className="technician-panel-heading">
                    <div>
                      <h2>{isRadiology ? 'Protocol scan worklist' : isPathology ? 'Samples in progress' : `${specialtyLabel} work queue`}</h2>
                      <p>{isPathology ? 'The stability clock begins at collection; sample-specific timing is unavailable until an authorized worklist is connected.' : 'Tasks are shown only after coordinator authorization is configured.'}</p>
                    </div>
                    <span className="technician-unavailable-tag">Unavailable</span>
                  </div>
                  <div className="technician-table-wrap">
                    <table>
                      <thead>
                        <tr>
                          {isPathology ? (
                            <>
                              <th scope="col">Sample</th>
                              <th scope="col">Participant</th>
                              <th scope="col">Specimen</th>
                              <th scope="col">Collected</th>
                              <th scope="col">Next step</th>
                              <th scope="col">Stability</th>
                              <th scope="col">Access check</th>
                            </>
                          ) : (
                            <>
                              <th scope="col">Time</th>
                              <th scope="col">Participant</th>
                              <th scope="col">Visit</th>
                              <th scope="col">Visit window</th>
                              <th scope="col">{isRadiology ? 'Protocol scan' : 'Task'}</th>
                              <th scope="col">Access check</th>
                            </>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td colSpan={isPathology ? 7 : 6}>
                            <div className="technician-empty-state">
                              <FiShield aria-hidden="true" />
                              <strong>{isPathology ? 'No authorized samples to display' : 'No authorized tasks to display'}</strong>
                              <span>{isPathology ? 'Sample identifiers, participants, and chain-of-custody records remain hidden until access requirements are satisfied.' : 'Patient and study details remain hidden until access requirements are satisfied.'}</span>
                            </div>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>

                {isPathology ? (
                  <section className="technician-secondary-panels technician-pathology-panels">
                    <article className="technician-panel technician-queue-panel">
                      <div className="technician-panel-heading">
                        <div><h2>Shipment to central laboratory</h2><p>Manifest and handoff checklist.</p></div>
                        <FiPackage aria-hidden="true" />
                      </div>
                      <div className="technician-panel-empty">No shipment or manifest is configured. Shipment contents, dry-ice records, and courier handoff are unavailable.</div>
                      <div className="technician-disabled-action" aria-disabled="true">Sign and hand to courier · unavailable</div>
                    </article>
                    <article className="technician-panel technician-queue-panel">
                      <div className="technician-panel-heading">
                        <div><h2>Freezers and equipment</h2><p>Temperature and capacity status.</p></div>
                        <FiThermometer aria-hidden="true" />
                      </div>
                      <div className="technician-equipment-empty">
                        <div><strong>Freezer inventory</strong><span>Not connected</span></div>
                        <div><strong>Temperature monitoring</strong><span>Not connected</span></div>
                        <div><strong>Centrifuge availability</strong><span>Not reported</span></div>
                      </div>
                    </article>
                  </section>
                ) : (
                <section className="technician-secondary-panels">
                  <article className="technician-panel technician-queue-panel">
                    <div className="technician-panel-heading">
                      <div><h2>{isRadiology ? 'Image QC queue' : `${specialtyLabel} review queue`}</h2><p>Authorized items will appear here.</p></div>
                      <FiImage aria-hidden="true" />
                    </div>
                    <div className="technician-panel-empty">Queue unavailable until task access is configured.</div>
                  </article>
                  <article className="technician-panel technician-queue-panel">
                    <div className="technician-panel-heading">
                      <div><h2>{isRadiology ? 'Transfer status' : 'Handoffs'}</h2><p>Transfer and handoff status.</p></div>
                      <FiUploadCloud aria-hidden="true" />
                    </div>
                    <div className="technician-panel-empty">No transfer or handoff data is available.</div>
                  </article>
                </section>
                )}
              </div>

              <aside className="technician-dashboard-aside">
                {isPathology && (
                  <section className="technician-panel technician-access-requirements">
                    <div className="technician-panel-heading">
                      <div><h2>Chain of custody</h2><p>Sample event history is unavailable.</p></div>
                      <FiTag aria-hidden="true" />
                    </div>
                    <div className="technician-custody-empty">
                      <FiShield aria-hidden="true" />
                      <strong>No sample selected</strong>
                      <span>Collection, receipt, processing, storage, and handoff events will be recorded here when authorized sample records are available.</span>
                    </div>
                  </section>
                )}
                <section className="technician-panel technician-access-requirements">
                  <div className="technician-panel-heading">
                    <div><h2>{isPathology ? 'My authorized tasks' : 'Access requirements'}</h2><p>{isPathology ? 'Task-level authorization must be confirmed.' : 'Confirm with your study coordinator before work begins.'}</p></div>
                    <FiShield aria-hidden="true" />
                  </div>
                  {isPathology ? (
                    <>
                      <ul>
                        <li>Sample receipt and accession</li>
                        <li>Authorized processing and aliquoting</li>
                        <li>Storage and freezer-log recording</li>
                        <li>Shipment preparation and custody handoff</li>
                        <li>Flag unsuitable samples for scientist review</li>
                      </ul>
                      <p className="technician-access-footnote">No task delegation or laboratory connection is configured. These permissions are not active.</p>
                    </>
                  ) : (
                    <>
                      <ul>
                        <li>Active study and site assignment</li>
                        <li>PI-approved task delegation</li>
                        <li>Current protocol-specific training</li>
                        <li>Protocol version and blinding access</li>
                      </ul>
                      <p className="technician-access-footnote">These controls are not yet connected to this workspace. No clinical task actions are enabled.</p>
                    </>
                  )}
                </section>
                {isPathology && (
                  <section className="technician-panel technician-access-requirements">
                    <div className="technician-panel-heading">
                      <div><h2>Laboratory scientist only</h2><p>Not available to technician accounts.</p></div>
                      <FiShield aria-hidden="true" />
                    </div>
                    <div className="technician-scientist-actions" aria-label="Restricted scientist actions">
                      <span>Validate or release results</span>
                      <span>Final sample rejection</span>
                      <span>Call critical values to clinicians</span>
                    </div>
                  </section>
                )}
              </aside>
            </div>
          </>
        )}
      </div>
    </SidebarLayout>
  );
};

export default TechnicianDashboard;
