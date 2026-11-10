# Diagnosis and Statistics User-Journey Roadmap

## Purpose

This roadmap converts the diagnosis and statistics requirements in [DIAGNOSIS_STATISTICS_INTEGRATION_PLAN.md](DIAGNOSIS_STATISTICS_INTEGRATION_PLAN.md) into a staged, evidence-based verification sequence.

The target is not to test every feature at once. The target is to maintain one synthetic patient through the complete clinical journey while proving, after every action, that:

1. the UI shows the correct role-specific state;
2. the authoritative record contains the expected state;
3. the audit record identifies the actor, resource, action, and timestamp;
4. refresh and navigation preserve the state;
5. analytics use the correct source for each metric;
6. cross-role visibility and hospital scoping remain correct.

> Clinical actions use synthetic data only. No real patient information is included.

The staged browser QA sequence and current execution checkpoint are tracked in [QA_USER_JOURNEY_RUNBOOK.md](QA_USER_JOURNEY_RUNBOOK.md).

## Current Baseline and Important Findings

- The physician diagnosis endpoint currently persists a free-text diagnosis and treatment data, but it does not yet persist the planned diagnosis metadata atomically.
- The current analytics implementation can derive a physician diagnosis from the encounter diagnosis field, but it also falls back through triage, symptom, or patient-history fields.
- The current analytics UI uses the legacy `diagnoses` series and contains sample mock data as initial state.
- The audit model has a broad action enum, but diagnosis, specialist lifecycle, and work-item events are not yet represented as a complete diagnostic decision contract.
- The plan's implementation and acceptance gates are still approval-dependent, especially catalog scope, terminology, legacy treatment, custom-category governance, and specialist targets.

These findings are why the roadmap begins with a controlled diagnosis and analytics journey before specialist workflows.

## Canonical Test Environment

### Single-patient fixture

Use one synthetic patient and one hospital for the complete diagnosis-and-statistics road map.

| Fixture | Value |
|---|---|
| Hospital | Sandbox QA Hospital |
| Patient | `DIAG-ROADMAP-001` |
| Patient presentation | Acute fever, generalized weakness, and mild cough |
| Triage diagnosis | Fever / viral syndrome |
| Final diagnosis | Community-acquired pneumonia, unless the test explicitly changes it |
| Initial triage urgency | High or medium, according to the seeded encounter |
| Test window | One calendar day, with an explicit start and end timestamp |
| Data state | Synthetic and disposable |

Do not create additional patient encounters until the first diagnosis journey is complete and verified.

### Required role identities

Use only backend-issued test accounts and known test credentials.

| Role | Expected dashboard | Primary actions |
|---|---|---|
| Account owner | Account dashboard | Create/confirm sandbox hospital and admin identities |
| Hospital administrator | Hospital administration | Confirm ownership, assign staff, verify hospital scope |
| Nurse | Triage/Nurse dashboard | Triaged patient, capture symptoms, record triage outcome |
| Physician | Physician dashboard | Review triage, select or enter final diagnosis, record disposition |
| Pharmacist | Pharmacy dashboard | Review medication order, verify, clarify, intervene, complete |
| Radiologist | Radiology dashboard | Accept, complete, report, amend or reject study |
| Pathologist | Pathology dashboard | Accession, process, result, sign out specimen |
| Supervisor/admin | Analytics/operations | Review audit and data-quality state |

Each role should be logged in separately, with the browser session restored to the original test state after the role check.

## Roadmap Principles

1. **One patient, one encounter.** A single encounter is the unit of diagnosis and final-statistics verification.
2. **Triage is never final diagnosis.** Triage findings remain separate from the physician decision.
3. **The final diagnosis is immutable until explicitly corrected.** Correction creates a new version or audit event; it does not overwrite the original source record.
4. **Every clinical action is persisted before the UI is considered complete.** UI state is evidence, not the source of truth.
5. **Every audit event includes actor, resource, hospital, action, decision, timestamp, and authorization context.**
6. **Every metric has a denominator and data-quality state.** Empty, low-sample, and reliable states must be explicit.
7. **Every role is hospital-scoped.** No role may see or act on another hospital's patient or work item.
8. **Refresh and navigation are part of verification.** A successful action is not complete until it survives reload and route changes.

## Hospital Operations and Shift-Coverage Prerequisite

The diagnosis journey depends on a persisted hospital structure and an authorized staffing workflow. Before using a triage nurse in the clinical journey:

- Configure departments, specialties, and wards for the hospital; shift coverage must reference that configured structure rather than free-text labels.
- Staff onboarding position codes, not permission roles, determine eligible shift duties. The server rejects position/duty mismatches; no cross-cover override is permitted. Legacy staff without a position code must choose an explicit compatible duty and be flagged for backfill.
- Shifts persist the configured department ID/code and a department-name snapshot. Existing shifts must continue to load with their original department label after structure changes.
- Use the shift calendar to review coverage month by month, including navigation across year boundaries and returning to the current month.
- Provisioning or publishing a shift does not start it and must not clock staff in automatically.
- An authorized hospital administrator explicitly starts the current-day shift. The persisted shift and audit trail must identify its status, approver, and start time.
- Staff clock-in is permitted only after that explicit start. Assignment changes are limited to shifts that have not started.
- Cancelling a not-yet-started shift requires a reason, preserves the shift record, and records the actor and cancellation in the audit trail.
- Scheduled shift end is not an early hard stop: administrators may finalize during the five-minute window after its scheduled end; otherwise the backend completes it automatically. Staff auto-clock-out remains tied to that scheduled end and grace window.
- Early hard stop is disabled until two distinct administrators authorize it, an emergency reason and attestation are recorded, active chart sessions are checked and handed over, earned hours are preserved, and audit and token-revocation controls are verified.
- Encounter-editing routes maintain a hospital-scoped chart session linked to staff and encounter, with an active shift link where available. Sessions heartbeat and stay open until explicitly closed or transferred; unavailable tracking must fail closed for any future hard stop.
- Keep the Shifts & attendance page focused on its viewing date and summary; open the calendar, roster, attendance, and administrator shift-scheduling functions from the adjacent action buttons in accessible dialogs rather than stacking those workspaces below the header.

Verify these rules through the shared browser and the authoritative API/persisted records. Refresh the calendar and relevant staff views to confirm the same hospital, department, specialty, ward, shift, and lifecycle state remain linked.

## Phase 0: Confirm Product Decisions

### Required approvals before implementation

Confirm the following before changing the plan into code:

- Diagnosis catalog scope: global, hospital-specific, or hybrid.
- Terminology standard: internal catalog only, ICD-10, SNOMED CT, or staged combination.
- Whether free-text custom diagnoses are immediately publishable.
- Who approves custom categories, aliases, and mappings.
- How legacy `encounter.diagnosis` values are classified.
- Whether symptom statistics appear on the main analytics page or a separate clinical-data-quality view.
- Whether specialist telemetry is an event collection, embedded work-item model, or hybrid event/read-model architecture.
- Targets for radiology, pharmacy, and pathology status states.
- Critical-result alert delivery: polling operations feed first, or sub-minute WebSocket/SSE required.
- Ownership of radiology discordance, pharmacy override, and pathology rejection classifications.

### Exit gate

Do not start implementation until the chosen contract is confirmed and recorded in the plan. The browser journey can be prepared, but no production-like test should exercise unapproved classification rules.

## Phase 1: Establish the Patient and Triage Journey

### Journey 1A — Account owner creates the test hospital

1. Sign in as the sandbox account owner.
2. Open the account Hospitals view.
3. Create the test hospital.
4. Confirm the hospital appears in the account's persisted hospital list.
5. Create the first hospital administrator and capture the one-time credential.
6. Sign out and sign in as the new hospital administrator.
7. Confirm the administrator can see the hospital and its account ownership.
8. Add a second test administrator only if the roadmap requires a role handoff.

### Persistence and audit gates

- The hospital has a stable hospital ID and owner account ID.
- The administrator appears in the hospital's admin list.
- The credential is displayed once and is not returned by a later read.
- Authentication works only with the correct hospital and role.
- A failed or unauthorized login does not create a clinical record.
- The account owner and hospital administrator cannot cross tenant boundaries.

### Browser evidence

- Account dashboard shows the correct hospital count and administrator count.
- Hospital admin login returns to the correct hospital dashboard.
- A refresh preserves the hospital and admin list.
- The browser console and network panel contain no 401/403 errors.

## Phase 2: Nurse Triage and Patient Intake

### Journey 2A — Nurse receives the synthetic patient

1. Sign in as the test nurse.
2. Open the nurse/triage dashboard.
3. Confirm the patient is visible according to hospital, department, and role scope.
4. Open the patient encounter.
5. Confirm the patient identifier, age, presentation, priority, and assigned context.
6. Record the triage assessment.
7. Save the triage result.
8. Navigate away and return to the patient.
9. Confirm the saved triage remains visible and unchanged.

### Required triage persistence

The encounter must retain:

- patient identifier and hospital scope;
- source presentation and reported symptoms;
- triage diagnosis or recommended differential;
- assigned clinician and department;
- triage actor and timestamp;
- triage source and model/manual provenance;
- original triage data for audit comparison;
- current encounter status and status history.

### Audit checks

The audit record must show the nurse as the actor and must identify the patient encounter as the resource. It should include:

- action: triage-related event;
- authorization result;
- source and triage provenance;
- timestamp;
- hospital scope;
- actor and role;
- any prior triage state for comparison.

### Browser verification loop

- Refresh the patient page.
- Navigate to the queue and back.
- Open the analytics page in a separate tab.
- Confirm triage-only data is present in triage statistics but does not appear as a physician-final diagnosis.
- Confirm no console or network error occurs.

## Phase 3: Physician Diagnosis Review

### Journey 3A — Physician reviews the triage

1. Sign in as the assigned physician.
2. Open the encounter from the physician dashboard.
3. Confirm the triage diagnosis is labeled explicitly as triage.
4. Confirm the presenting symptoms and priority are visible as review context.
5. Answer the match question:
   - matches;
   - differs;
   - not sufficient/not applicable.
6. Continue without forcing a match decision to be correct.

### Journey 3B — Select a catalog diagnosis

1. Open the guided diagnosis selector.
2. Search for a known diagnosis.
3. Select a catalog diagnosis.
4. Confirm the normalized category and canonical code.
5. Confirm the selected diagnosis preview is shown.
6. Save the diagnosis atomically.
7. Reload the diagnosis page.
8. Confirm the final diagnosis, category, source, match decision, and actor remain.

### Journey 3C — Enter a custom diagnosis

1. Choose the custom diagnosis path.
2. Enter a non-catalog diagnosis.
3. Confirm the diagnosis is marked as custom and not silently mapped to a common category.
4. Save it.
5. Confirm the custom category is present in the unclassified/custom bucket.
6. Verify the original text is retained and the review state is visible.

### Journey 3D — Differentiate triage from final diagnosis

1. Save a final diagnosis that differs from triage.
2. Confirm the triage report remains unchanged.
3. Confirm the final diagnosis is labeled as physician final.
4. Confirm analytics use the final diagnosis only.
5. Confirm the triage diagnostic value remains available for comparison and audit.

### Required diagnosis persistence contract

One atomic save should persist:

- encounter ID and hospital ID;
- final diagnosis text;
- canonical diagnosis code, when available;
- normalized diagnosis category;
- diagnosis source;
- physician/staff actor ID and display name;
- recorded timestamp;
- triage diagnosis at review;
- triage match state: `MATCH`, `DIFFERS`, or `NOT_REVIEWED`;
- optional match note;
- presenting symptom codes/categories;
- custom category metadata, when applicable;
- prior diagnosis and source metadata for correction history;
- treatment and progression records that are independently persisted.

### Audit checks

The diagnosis audit event must record:

- `DIAGNOSIS_RECORDED` or equivalent action;
- actor ID and role;
- encounter and hospital IDs;
- diagnosis text and normalized category;
- source and match decision;
- before/after values;
- authorization result and approver, if applicable;
- user agent and request time;
- immutable timestamp.

A second diagnosis save must create a new decision version or audit event. It must not erase the original diagnosis or triage record.

### Browser verification loop

- Save diagnosis.
- Reload diagnosis page.
- Navigate to patient summary and back.
- Open analytics in a separate tab.
- Confirm the diagnosis chart contains exactly one final diagnosis for the single patient.
- Confirm the chart displays `n=1` and low-confidence context.
- Confirm triage-only data is not counted as final diagnosis.
- Confirm the match-rate and case-review data agree with the persisted event.

## Phase 4: Diagnosis Analytics and Data Quality

### Journey 4A — Final diagnosis distribution

1. Open the analytics page as an administrator or authorized reviewer.
2. Select the same test-day range used for the patient encounter.
3. Confirm the final diagnosis series contains the selected diagnosis only.
4. Confirm the triage series contains the triage diagnosis separately.
5. Confirm the symptom series contains the presenting symptoms separately.
6. Confirm the final diagnosis percentage sums to 100% for the selected final-diagnosis denominator.
7. Confirm the chart state is `LOW_CONFIDENCE` when `n=1`.

### Journey 4B — Triage-only exclusion

1. Use a triage-only encounter or a patient whose final diagnosis is pending.
2. Confirm it appears in triage statistics but not in final diagnosis distribution.
3. Confirm it is not assigned to the final diagnosis chart.
4. Confirm an explicit pending or unavailable state is shown.

### Journey 4C — Custom diagnosis and legacy classification

1. Review the custom diagnosis bucket.
2. Confirm the custom category is not silently relabeled.
3. Review a legacy encounter, if present.
4. Confirm it is marked legacy/unverified, not treated as physician-confirmed.
5. Confirm the legacy count is separately visible.

### Journey 4D — Cross-view consistency

Compare:

- Overview;
- Triage Analytics;
- Physician diagnosis page;
- encounter timeline;
- operations/audit view;
- specialist work-item views, when available.

All views must use the same hospital, time range, encounter identifier, and data source for their metric.

### Required analytics contract

The response should expose explicit fields:

- `finalDiagnosisDistribution`;
- `triageDiagnosisDistribution`;
- `presentingSymptomDistribution`;
- `triageFinalMatchRate`;
- `customDiagnosisRate`;
- `unclassifiedFinalDiagnosisCount`;
- final diagnosis data-quality flags;
- sample size and confidence state;
- legacy/unverified count;
- specialist performance metrics, where applicable.

The legacy `diagnoses` field may remain as a compatibility alias, but it must not be the primary source for final diagnosis statistics.

## Historical EHR seeding and diagnostic context

- Administrator seeding creates only synthetic `DISCHARGED` encounters, with `isWaiting=false`, no staff assignment, no triage report, and no handover. It must never create or advance a live clinical workflow.
- The lookback is explicit and bounded: default 90 days, configurable from 1 to 3650 days. Seeded encounter timestamps fall strictly before the current time and within the selected window.
- A nurse starts the new, active triage through the staffed shift workflow. The new encounter uses the patient's stable hospital-scoped identifier so discharged history can inform the new visit without reusing the prior encounter.
- At scheduled shift end, an administrator has a five-minute window to finalize; otherwise the backend completes the shift and records the scheduled automatic completion. A browser closing or refresh must not cancel the deadline. Early hard stop is a separate, disabled workflow until chart-session tracking and legal/clinical safeguards are verified.
- Chart edit routes must create hospital-scoped staff/encounter sessions with optional active-shift linkage, heartbeats, edit detection, explicit closure or handover, and an administrator query for a staff member's active sessions. Session tracking/audit events are not proof of immutable storage or immediate token revocation.
- A radiology or pathology request snapshots the current encounter and relevant prior discharged EHR context when the order is created. The snapshot, encounter link, patient identifier, capture time, and order lifecycle persist together and are returned to the authorized specialist queue.
- Specialist dashboards display the captured context as history, not as a current diagnosis or new triage result. Context is hospital-scoped and must not be inferred from another hospital or a different patient identifier.

## Phase 5: Specialist Work-Item Lifecycle

Use one test patient for each specialist workflow and create separate work items only when the business process requires it. Do not reuse the same work item across radiology, pharmacy, and pathology.

### Phase 5A — Radiology

#### Journey

1. Physician creates or confirms a radiology request for the patient.
2. Radiology accepts the request; the authorized technician worklist then tracks patient arrival, identity/order verification, and technician handoff.
3. PACS/RIS or modality staff perform acquisition in their vendor system. NurseHelpDesk may record sequential, explicitly manual status observations for acquisition start/completion and image availability; these are not PACS events or proof of machine operation.
4. When images are reported available, an optional organization-approved HTTPS cloud DICOM viewer link may be recorded. Image storage, viewing, and modality control remain outside NurseHelpDesk.
5. A radiology staff member records that the request is ready for interpretation. The radiologist may then start interpretation and submit the report.
6. The report and signature metadata are saved to the encounter/EHR; an authorized user can download a printable PDF. The generated PDF is a workflow copy, not a validated PDF/A document or qualified digital signature.
7. The ordering physician reviews the report. A correction or amendment is recorded separately if the workflow supports it.
8. The radiologist reloads the queue and the persisted report and status history remain visible.

#### Required lifecycle

`REQUESTED -> ACCEPTED -> PATIENT_ARRIVED -> IDENTITY_VERIFIED -> TECHNICIAN_HANDOFF -> ACQUISITION_STARTED -> ACQUISITION_COMPLETED -> IMAGES_AVAILABLE -> READY_FOR_INTERPRETATION -> REPORT_SUBMITTED -> PHYSICIAN_REVIEWED`

Each recorded transition must persist:

- hospital and encounter or study identifier;
- role, staff ID, actor, and authorization context;
- priority and requested modality;
- timestamp and provenance (`MANUAL` or a future authenticated vendor event);
- external acquisition completion time separately from radiologist interpretation and report-submission time;
- optional viewer URL only when images are reported available;
- correction, amendment, rejection, or clarification event;
- turnaround calculations.

The current manual workflow must not be presented as a live PACS/RIS integration. Queue labels, audit history, and analytics must preserve the manual source until an authenticated adapter is configured.

#### Persistence and audit gates

- A study is not counted as completed before a completion event exists.
- A signed report is not counted as verified before the verifier is recorded.
- An amended report is counted separately from a completed report.
- A rejected study has a reason and is not counted as completed.
- Radiology turnaround uses acquisition completion to signed report, not shift start/end.
- Queue age and backlog are derived from persisted timestamps.

### Phase 5B — Pharmacy

#### Journey

1. Physician creates a medication order for the patient.
2. Pharmacist receives the order.
3. Pharmacist accepts and reviews it.
4. Pharmacist verifies or clarifies it.
5. Pharmacist completes or holds it.
6. A high-risk warning or intervention is recorded when applicable.
7. The pharmacist reloads the order and confirms the lifecycle.

#### Required lifecycle

`RECEIVED -> ACCEPTED -> IN_PROGRESS -> COMPLETED -> VERIFIED/REPORTED`

Each transition must persist:

- hospital, encounter, and medication order ID;
- pharmacist ID and role;
- priority and medication details;
- received, reviewed, verified, completed, and released timestamps;
- intervention or warning status;
- override actor, rationale, and authorization;
- rejection, hold, clarification, or cancellation reason;
- turnaround and safety metric values.

#### Persistence and audit gates

- A medication order is not counted as verified until the verification event exists.
- A pharmacist intervention is not counted as accepted until an approval or resolution event exists.
- An override is distinguishable from an unreviewed action.
- A held or rejected order remains a quality metric and is not counted as completed.
- Verification lead time uses physician order signing to pharmacist verification.

### Phase 5C — Pathology

#### Journey

1. Physician requests a specimen for the patient.
2. Pathologist receives the specimen.
3. Pathologist accessiones it.
4. Pathologist processes it.
5. Pathologist publishes the result.
6. The result is signed out and notified to the recipient.
7. A critical-value or rejection event is recorded when applicable.
8. The pathologist reloads the specimen and confirms the lifecycle.

#### Required lifecycle

`RECEIVED -> ACCEPTED -> IN_PROGRESS -> COMPLETED -> VERIFIED/REPORTED`

Each transition must persist:

- hospital, encounter, and specimen ID;
- pathologist ID and role;
- specimen type and priority;
- received, accessioned, processed, result, and notification timestamps;
- critical-value flag and detection time;
- clinician recipient and escalation route;
- rejection or contamination reason;
- result correction or amendment event;
- turnaround and backlog calculations.

#### Persistence and audit gates

- A specimen is not counted as completed until the result is complete.
- A rejected specimen is not counted as completed.
- A critical result creates a persisted alert and mediator event.
- If delivery fails, the underlying event remains and the retry state is visible.
- Critical notification time is measured from detection to clinician notification.

## Phase 6: Specialist Analytics and Data Quality

For each specialist domain, verify:

1. the dashboard shows the same totals as the event-backed work-item list;
2. the selected time range filters both views consistently;
3. the denominator includes only completed or verified work;
4. sample size and confidence state are visible;
5. backlog and queue age use persisted timestamps;
6. waiting-for-upstream-information is separated from active processing time;
7. corrections, amendments, rejections, and clarifications remain visible;
8. cross-hospital data is excluded;
9. refresh and route navigation preserve the metrics;
10. empty, low-sample, and reliable states are explicit.

### Specialist metric gates

| Domain | Required denominator | Required event evidence |
|---|---|---|
| Radiology | Completed and verified studies | Acquisition completion, signed report, reviewer status |
| Pharmacy | Verified and completed orders | Order signing, pharmacist review, intervention, resolution |
| Pathology | Completed and signed results | Accession, processing, result, notification, sign-out |

## Phase 7: Surgery and Appointment Journey

This phase is separate from diagnosis and specialist analytics. It must not begin until the prior phases are complete.

### Surgery journey

1. Create a surgery request for the synthetic patient.
2. Schedule the surgery and confirm the hospital, procedure, priority, surgeon, and theatre.
3. Refresh and confirm the record remains.
4. Move through pre-op ready, in progress, and completed states.
5. Record the outcome and disposition.
6. Confirm the audit history preserves every transition.
7. Test cancellation, postponement, no-show, or abort as a separate exception journey.

### Appointment journey

1. Create a normal appointment for the synthetic patient.
2. Confirm the appointment.
3. Check the patient in.
4. Start the visit or record unable-to-proceed.
5. Complete the appointment and capture the outcome.
6. Refresh and confirm the lifecycle.
7. Test reschedule, cancellation, waitlist, and no-show separately.

### Required lifecycle checks

- A scheduled surgery is not counted as completed without completion.
- A normal appointment is not counted as a completed visit without an explicit visit completion event.
- Reception, clinical, and surgery records use the same appointment identifier when linked.
- Calendar, operations feed, encounter timeline, and analytics agree on status.
- Exception reasons are persisted and counted separately.

## Phase 8: Final Cross-Role Browser Verification Loop

Run the following sequence after each phase is approved:

1. Log in as the account owner and verify the account view.
2. Log in as hospital administrator and verify hospital scope.
3. Log in as nurse and perform the triage action.
4. Log in as physician and perform the diagnosis action.
5. Reload the diagnosis page and analytics page.
6. Log in as pharmacist and complete the work item.
7. Log in as radiologist and complete the work item.
8. Log in as pathologist and complete the work item.
9. Log in as supervisor and compare all metric sources.
10. Log out and verify the session is cleared.
11. Log back in and verify the same records remain.
12. Review the browser console, network responses, persisted documents, and audit records.

## Persistence Gate Matrix

Each action must satisfy all of these checks:

| Gate | Required evidence |
|---|---|
| UI persistence | The visible page shows the saved state after refresh |
| API persistence | The response and database document contain the same canonical identifiers |
| Hospital scope | The record is visible only within the correct hospital |
| Role authority | The actor is authorized for the requested action |
| Audit attribution | The actor, resource, action, and timestamp are present |
| State integrity | The action is represented in the correct lifecycle and status history |
| Analytics consistency | The relevant dashboard and aggregate use the same event or record |
| Refresh safety | The state survives navigation, reload, and log out/in |
| Error safety | No unhandled console, network, authorization, or validation error remains |

## Audit Evidence Checklist

For every clinical action, capture the following evidence:

- action name and stable action code;
- actor ID, role, and display name;
- hospital ID;
- patient or encounter ID;
- work-item, study, order, or specimen ID;
- before and after values;
- decision or reason;
- authorization result;
- approver, if applicable;
- request and response timestamps;
- client user agent or request metadata;
- event ID;
- event persistence status;
- analytics impact or derived metric.

Do not treat a successful UI response as proof that the audit event was created. Query the authoritative audit store and compare event IDs and timestamps.

## Recommended Verification Tools

Use the shared browser for all UI verification. Use the live API and MongoDB inspection for the persistence gates.

The verification record should contain:

- browser URL and page state;
- authenticated role and token identity;
- action timestamp;
- network request and response status;
- UI visible result;
- database document result;
- audit event result;
- analytics result;
- console errors;
- refresh/navigation result;
- pass/fail status and blocker.

Do not run builds as part of this roadmap. The requested verification path is browser, API, database, and audit evidence only.

## Phase Status

| Phase | Status | Exit condition |
|---|---|---|
| Product decisions | Pending | All plan decisions are approved |
| Account and hospital setup | Complete for sandbox workflow | Hospital ownership and admin credentials verified |
| Hospital structure and shift coverage | API/unit tests and frontend build verified; browser verification pending | Position-to-duty rules, configured-department validation, department snapshots, calendar navigation, administrator start/cancel audit, and pre-start clock-in rejection are verified after refresh |
| Nurse triage | Pending | Triage persists, audits, refreshes, and analytics excludes it from final diagnosis |
| Physician diagnosis | Pending | Final diagnosis persists atomically and audits correctly |
| Diagnosis analytics | Pending | Final, triage, symptom, custom, and legacy metrics agree |
| Radiology lifecycle | Pending | Complete work-item lifecycle and metrics are persisted and audited |
| Pharmacy lifecycle | Pending | Verification, intervention, and completion events are persisted and audited |
| Pathology lifecycle | Pending | Result, notification, and rejection events are persisted and audited |
| Specialist analytics | Pending | Event totals, targets, confidence, and data-quality states agree |
| Surgery and appointments | Pending | Lifecycle states, exceptions, and metrics survive refresh |
| Cross-role release gate | Pending | All roles, scopes, audit events, and analytics agree |

## Release Gate

The diagnosis and statistics roadmap is ready to move forward only when:

- one patient completes the full controlled journey;
- final diagnosis, triage, symptoms, and custom categories remain separate;
- every clinical action has a persisted audit event;
- all audit records match the browser/API action and actor;
- refresh/navigation preserve state;
- analytics values match the authoritative records;
- no cross-hospital data is visible;
- no unresolved console or authorization error remains;
- low-sample and empty states are explicit;
- specialist and surgery/appointment work item transitions are verified separately;
- the plan's approval decisions are documented.

The next implementation step should begin with the physician diagnosis metadata contract and the smallest possible diagnosis/analytics browser journey, not with the entire roadmap at once.
