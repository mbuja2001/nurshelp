# Diagnosis and Statistics Integration Plan

## Status

**Draft for user confirmation. No application code changes are included in this document.**

This plan defines the work required before moving on to Labs and Pharmacy. It is based on the request to make analytics count the physician's final diagnosis, explicitly compare it with triage, support guided diagnosis entry, and preserve reliable statistical categories without losing clinical nuance.

## 1. Core product rule

Analytics must distinguish between:

- **Triage suggestion**: AI or nurse triage assessment used to prioritize and route the encounter.
- **Physician final diagnosis**: the diagnosis documented or confirmed by the doctor after reviewing the encounter.
- **Statistical diagnosis category**: a normalized category selected from the diagnosis catalog or created as a governed custom category.
- **Presenting symptom / complaint**: what the patient reported or what was observed at intake.

The physician final diagnosis is the source of truth for final-diagnosis statistics. Triage diagnosis must never silently become the final diagnosis merely because it exists first.

### Hospital schedule and staffing dependency

Hospital departments, specialties, wards, and shift coverage are persisted operational records. A shift must reference configured hospital structure, but staffing or attendance by itself is not evidence that specialist work was completed and must not be used as a specialist performance metric.

Publishing or provisioning coverage does not start a shift. An authorized hospital administrator must explicitly start it, with the actor and timestamp audited; staff clock-in is valid only after that transition. Pre-start cancellation requires a reason and preserves an auditable cancelled record. The calendar must expose the same persisted shift state across month and year navigation. These controls are prerequisites for the browser diagnosis journey and are tracked in the user-journey roadmap.

## 2. Current code baseline

The existing implementation already provides useful foundations:

- `DiagnosePage.jsx` loads an encounter and persists `diagnosis` through `PUT /api/encounters/:id/diagnosis`.
- `Encounter.diagnosis` is documented as the physician-documented diagnosis.
- `analytics.controller.js` reads `encounter.diagnosis` when building the physician final diagnosis for case review.
- Triage data is stored separately in triage/AI/manual report structures.
- The analytics page and overview already consume diagnosis distributions from live analytics responses.

The remaining work is to make the clinical workflow explicit and prevent fallback logic from mixing triage, complaint, and final diagnosis in the same statistic.

## 3. Target data model

### 3.1 Encounter diagnosis fields

Preserve the existing `diagnosis` field as the physician final diagnosis for compatibility, but add explicit metadata:

- `diagnosis`: displayed physician final diagnosis text
- `diagnosisCode`: optional normalized catalog code
- `diagnosisCategory`: normalized statistical category
- `diagnosisSource`: `PHYSICIAN_SELECTED`, `PHYSICIAN_CUSTOM`, or `LEGACY_MIGRATED`
- `diagnosisRecordedBy`: physician/staff identifier
- `diagnosisRecordedAt`: timestamp
- `triageDiagnosisAtReview`: triage diagnosis presented to the physician
- `triageDiagnosisMatch`: `MATCH`, `DIFFERS`, or `NOT_REVIEWED`
- `triageDiagnosisMatchNote`: optional clinical explanation
- `presentingSymptoms`: normalized symptom codes/categories where available
- `customDiagnosisCategory`: governed custom category when no catalog item applies

The system should not overwrite the original triage result when the physician makes a final diagnosis.

### 3.2 Catalog entities

Introduce hospital-aware or platform-wide catalogs for:

- common diagnoses
- presenting symptoms
- synonyms and alternate labels
- diagnosis category mappings
- optional ICD/SNOMED terminology references where available

Each catalog item should include:

- stable identifier
- canonical label
- display label
- type: `DIAGNOSIS` or `SYMPTOM`
- aliases/synonyms
- active status
- sort/search metadata
- created/updated audit metadata
- hospital scope or global scope

Custom categories must be marked as custom and remain auditable. They should not silently merge with an existing category solely because their text looks similar.

## 4. Physician diagnosis user journey

### Step 1: Receive triage

When the physician opens the encounter, show the triage diagnosis and relevant symptom/priority context as an input to clinical review, not as the final diagnosis.

The page should ask explicitly:

> Does the physician diagnosis match the triage diagnosis?

Required choices:

- Yes, it matches
- No, the diagnosis differs
- Triage diagnosis is not sufficient / not applicable

The physician must be able to continue with a final diagnosis regardless of the answer.

### Step 2: Select or enter final diagnosis

Provide a guided diagnosis input with:

- search across common diagnoses
- search across symptoms when appropriate
- synonym-aware matching
- suggested normalized category
- selected diagnosis preview
- free-text/custom diagnosis path

The physician can choose a catalog diagnosis or enter a custom diagnosis when no suggested option is clinically correct.

### Step 3: Capture clinical distinction

If the physician selects “No, the diagnosis differs,” require either:

- a final diagnosis selection, or
- a custom diagnosis label

Optionally request a concise reason for the difference for quality review. Do not block urgent clinical disposition on a long explanation, but preserve the audit event.

### Step 4: Persist one atomic clinical decision

Saving the diagnosis should persist the final diagnosis, normalized category, match decision, source, actor, and timestamps together. The UI should confirm that the physician diagnosis is now the final diagnosis used for downstream reporting.

## 5. Diagnosis page integration points

Primary update surface:

- `nurshelp/frontend/src/PatientSummary/DiagnosePage.jsx`

Related review surfaces:

- `PatientDiagnosis.jsx`
- `DoctorTriageReport.jsx`
- `ConfirmingAIClinicalHandover.jsx`
- `ManualOfficialTriageReport.jsx`
- `OfficialTriageReport.jsx`

Backend update surfaces:

- encounter diagnosis persistence controller
- encounter model
- encounter routes and authorization
- analytics controller
- audit log creation for diagnosis decisions

The existing diagnosis save endpoint should remain compatible with current clients while accepting the new structured fields.

## 6. Analytics processing rules

### 6.1 Final diagnosis distribution

The `diagnoses` analytics series must be calculated only from encounters with a physician final diagnosis or an explicitly governed legacy migration value.

Preferred order:

1. `diagnosisCategory` when present
2. `diagnosisCode` resolved through the catalog
3. `diagnosis` normalized through a controlled alias map
4. no final-diagnosis statistic when none exists

Do not use these as final diagnosis fallbacks:

- AI triage diagnosis
- nurse triage diagnosis
- presenting complaint
- patient history
- generic placeholder text

Those fields may have separate analytics series, but they must not inflate physician diagnosis statistics.

### 6.2 Separate metrics

The analytics payload should expose separate named series where useful:

- `finalDiagnosisDistribution`
- `triageDiagnosisDistribution`
- `presentingSymptomDistribution`
- `triageFinalMatchRate`
- `customDiagnosisRate`
- `unclassifiedFinalDiagnosisCount`

The existing `diagnoses` field can remain temporarily as a compatibility alias for `finalDiagnosisDistribution`, but the frontend should migrate to the explicit name.

### 6.3 Confidence and empty states

A diagnosis chart can render real proportions even when the sample is small. Confidence messaging must describe the limitation without replacing a real chart with an apparently contradictory empty state.

Example behavior:

- `1` physician-final diagnosis: render the exact 100% circle and show `n=1, low confidence`.
- `0` physician-final diagnoses: show an empty state explaining that physician diagnosis has not yet been recorded.
- triage-only data: show that triage data exists but final diagnosis is pending; do not label triage as final diagnosis.

### 6.4 Specialist performance metrics

Radiologists, pharmacists, and pathologists require separate performance domains. Their metrics must be based on completed specialist work items and recorded timestamps, not on staffing shifts alone.

#### Radiologist metrics

- studies received, accepted, completed, and reported
- median and percentile report turnaround time
- urgent-study response time
- report completion rate within configured target
- addenda or amended-report rate
- rejected/returned study rate with reason
- workload by modality and priority
- queue age and unreported-study backlog

#### Pharmacist metrics

- medication orders reviewed, verified, clarified, and completed
- medication verification turnaround time
- high-risk medication review completion rate
- intervention rate and intervention acceptance rate
- dispensing or fulfillment turnaround time
- rejected, held, or returned order rate with reason
- medication reconciliation completion rate where applicable
- queue age and outstanding medication-review backlog

#### Pathologist metrics

- specimens received, accessioned, processed, resulted, and signed out
- specimen-to-result turnaround time
- urgent-result notification time
- result completion rate within configured target
- amended or corrected-result rate
- rejected or insufficient-specimen rate with reason
- workload by specimen type and priority
- queue age and outstanding-result backlog

#### Common specialist metric contract

Every specialist work item should have a traceable lifecycle:

`RECEIVED -> ACCEPTED -> IN_PROGRESS -> COMPLETED -> VERIFIED/REPORTED`

The event record should include:

- hospital and work-item scope
- specialist role and specialist identifier
- source encounter/order/specimen/study identifier
- priority and department
- received, started, completed, and verified timestamps
- status transitions and reason codes
- correction, amendment, rejection, or clarification events

Performance must be attributed to the specialist who performed or verified the work, while preserving team-level metrics for shared workflows. A specialist must not be penalized for time spent waiting on missing clinical information, equipment, or an upstream department; those delays should be separately classified.

Specialist metrics must be configurable by administration for targets and reporting visibility, but the underlying event timestamps and status history remain immutable audit data.

## 7. Normalization and synonym strategy

### Controlled catalog first

Use stable catalog identifiers for statistics. Display labels can change without splitting the historical metric.

### Alias matching second

Map known synonyms to a canonical category only when the mapping is explicitly approved. Examples:

- `viral fever` -> `Fever / viral syndrome` if approved
- `SOB` -> `Shortness of breath` if approved
- `chest discomfort` -> `Chest pain` only if clinically accepted by the catalog owner

### Custom category last

When no catalog or approved alias is appropriate:

- preserve the physician's original text
- create/select a custom category
- mark it as custom and unclassified until reviewed
- include it in custom-category statistics
- allow an administrator or clinical data owner to map it later

Never silently force a custom diagnosis into an incorrect common category for cleaner charts.

## 8. API contract proposal

### Read diagnosis options

`GET /api/diagnosis-catalog?type=DIAGNOSIS|SYMPTOM&search=...`

Returns active catalog options, aliases, and normalized category identifiers.

### Save physician diagnosis

`PUT /api/encounters/:id/diagnosis`

Proposed body additions:

```json
{
  "diagnosis": "Fever and generalized weakness",
  "diagnosisCode": "DX-FEVER-WEAKNESS",
  "diagnosisCategory": "Fever / viral syndrome",
  "diagnosisSource": "PHYSICIAN_SELECTED",
  "triageDiagnosisMatch": "DIFFERS",
  "triageDiagnosisMatchNote": "Clinical review indicates a different final assessment",
  "presentingSymptoms": ["SYM-FEVER", "SYM-WEAKNESS"]
}
```

### Read analytics

`GET /api/analytics/triage?range=...`

Add explicit fields:

```json
{
  "finalDiagnosisDistribution": [],
  "triageDiagnosisDistribution": [],
  "presentingSymptomDistribution": [],
  "triageFinalMatchRate": null,
  "customDiagnosisRate": null,
  "unclassifiedFinalDiagnosisCount": 0,
  "specialistPerformance": {
    "radiology": {},
    "pharmacy": {},
    "pathology": {}
  }
}
```

Recommended specialist endpoints:

- `GET /api/analytics/specialists?role=RADIOLOGIST|PHARMACIST|PATHOLOGIST&range=...`
- `GET /api/specialist-work-items?role=...&status=...`
- `PATCH /api/specialist-work-items/:id/status`

The analytics response should expose both aggregate role metrics and the event counts used to calculate them, including sample size, target configuration, confidence state, and data-quality exclusions.

### Historical EHR context passed to diagnostic work

Synthetic history is for prior EHR context, not active work generation. Seed only discharged encounters in an explicit 1–3650 day window (90-day default); do not seed triage, waiting, assignment, or handover state. A nurse must create the current encounter and triage through the shift workflow.

Scheduled shift end and an early hospital hard stop are separate workflows. At the scheduled end, an administrator may complete the shift during the following five-minute authorization window; otherwise a backend job completes it automatically. Staff clock records retain the same scheduled-end grace and are auto-clocked-out with the existing attendance flags. No early end request or same-admin confirmation is allowed. Early hard stop remains disabled until chart sessions can be reliably queried and all required controls are implemented: distinct dual-admin approval, emergency reason and attestation, chart handover gating, labour-record preservation, and a tested audit/revocation design.

Encounter-editing routes open a hospital-scoped chart session tied to the staff member and encounter, and to an active assigned shift when one exists. Sessions heartbeat, record detected edits, and remain open across browser navigation until explicitly closed or transferred. A failed or unavailable session query must block any future hard-stop execution. Application audit events are not equivalent to immutable storage; do not claim immutability or immediate token revocation without database-level controls and a verified revocation mechanism.

When an authorized clinician creates a radiology or pathology request, persist a bounded, versioned snapshot of the current encounter and same-hospital prior discharged history on that request. Include the patient and encounter identifiers and capture timestamp. Return the snapshot in the matching specialist queue and show it as historical context, separate from the current request and any new diagnostic findings. A later change to the patient's record must not silently rewrite the context originally presented with the order.

## 8A. Trial-readiness ancillary metric matrix

The sandbox administration layer should expose a separate auxiliary diagnostic and therapeutic monitoring panel for Radiology, Pharmacy, and Pathology. These metrics bridge emergency triage with definitive treatment and must be calculated from completed workflow events, not shift attendance.

### Radiology performance and throughput

- modality-stratified turnaround time from acquisition completion to final signed report
- peer-review discrepancy and diagnostic discordance rate
- scanner/equipment utilization rate, separated from transport and technician assignment delays
- unsigned-report stagnation debt: count and age of reports awaiting signature beyond the configured audit window
- report completion within target by modality and priority
- unreported-study backlog and queue age

Recommended formula:

`Radiology TAT = reportSignedAt - acquisitionCompletedAt`

### Clinical pharmacy performance and safety

- allergy/contraindication warning override rate, including critical-history context
- pharmacist-caught near-miss and adverse-drug-event intervention rate
- medication-use-evaluation protocol adherence for configured high-risk therapies
- order-verification lead time from physician order signing to pharmacist review/release
- intervention acceptance rate and unresolved clarification count
- medication-review backlog and queue age

Recommended formula:

`Verification lead time = pharmacistVerifiedAt - physicianOrderSignedAt`

An override must be recorded as an auditable safety event. The metric should distinguish an appropriate documented override from an unreviewed or unexplained override; the system must not label a clinical decision as defiance without the configured governance definition.

### Pathology and laboratory operations

- critical/panic-value turnaround from threshold detection to clinician notification
- specimen rejection and contamination index, with rejection reasons
- order-to-inoculation or order-to-processing sepsis window
- crossmatch-to-transfusion efficiency ratio for blood-bank workflows
- result completion within target by specimen type and priority
- outstanding-result backlog and queue age

Recommended formulas:

`Critical notification time = clinicianNotifiedAt - criticalValueDetectedAt`

`Crossmatch-to-transfusion ratio = unitsCrossmatched / max(unitsTransfused, 1)`

Critical-result notification must create an alert/audit event tied to the encounter, responsible specialist, recipient, and timestamp. An unassigned patient must route to the configured nursing supervisor or clinical escalation queue.

## 8B. Ancillary workflow event model

The proposed telemetry should be implemented as an event-backed, hospital-scoped model rather than a single mutable blob attached to an encounter. This preserves workflow history, supports corrections, and allows each role to be measured independently.

Each work item should include:

- hospital ID and encounter/order/study/specimen identifier
- specialist role and staff ID
- workflow type and priority
- immutable lifecycle timestamps
- status transitions and reason codes
- derived turnaround fields calculated from timestamps
- correction, amendment, rejection, override, intervention, or notification events
- actor and audit metadata

The three telemetry domains should support these fields:

### Radiology telemetry

- imaging order type: `XR`, `CT`, `MRI`, or `US`
- ordered time
- acquisition completed time
- report signed time
- radiologist ID
- peer-review discordance flag and reason
- derived imaging turnaround minutes

### Pharmacy telemetry

- order verification minutes
- critical-history warning triggered
- warning override actor and rationale
- pharmacist intervention recorded
- intercepted prescribing-error description
- medication-use-evaluation non-compliance flag

### Pathology telemetry

- specimen type such as `BLOOD_CULTURE`, `LACTATE_POC`, `FBC`, or `COAG`
- collection and laboratory receipt times
- result publication time
- critical/panic-value flag and detection time
- clinician notification time and recipient
- specimen rejection flag and reason
- contamination classification where applicable

Indexes should support hospital, role, lifecycle status, priority, and the primary turnaround fields. Aggregated dashboards should read these events with explicit sample size and data-quality exclusions.

## 8C. Ancillary dashboard and alert behavior

The sandbox administrator dashboard should provide three sibling panels:

- Radiology Performance
- Pharmacy Vigilance
- Pathology Metrics

Each panel should show the measured value, target, sample size, confidence state, trend, and backlog. Example labels from the supplied design are acceptable as a visual direction, but values must always come from live events.

The system should support real-time escalation for:

- critical pathology/panic results
- overdue radiology reports
- high-risk medication warnings or unresolved pharmacist interventions

Use the existing audit/operations feed where suitable. Add WebSocket or server-sent events only when polling cannot meet the required notification latency; the event must still be persisted before or together with notification delivery so refresh and audit views remain authoritative.

## 9. Administration and governance

Admin configuration should control reporting categories and catalog governance, not physician clinical truth.

Admin capabilities:

- activate/deactivate catalog entries
- approve aliases
- create governed custom categories
- map reviewed custom categories to canonical categories
- view unclassified diagnosis queue
- view triage/final disagreement rates
- audit who changed a category mapping and when

Physicians retain responsibility for the final clinical diagnosis. Administrators configure statistical classification and reporting scope.

## 10. Migration and legacy data

Existing encounters may contain:

- only triage diagnosis
- only free-text `diagnosis`
- diagnosis copied from a complaint
- legacy data without source metadata

Migration rules:

- do not claim legacy text is physician-confirmed unless evidence exists
- mark migrated values as `LEGACY_MIGRATED`
- expose legacy counts separately until reviewed
- keep old analytics values reproducible during the migration period
- add a data-quality indicator for legacy/unclassified diagnosis records

## 11. Acceptance tests

### Physician journey

- [ ] Physician opens an encounter and sees triage diagnosis clearly labeled as triage.
- [ ] Physician is explicitly asked whether the final diagnosis matches triage.
- [ ] Physician can select a common diagnosis from search results.
- [ ] Physician can select a symptom/category when clinically appropriate.
- [ ] Physician can enter a custom diagnosis when no option applies.
- [ ] Diagnosis save persists the final diagnosis and match decision after refresh.
- [ ] Triage diagnosis remains unchanged after a different physician diagnosis is saved.
- [ ] Audit history records the actor, source, match decision, and timestamp.

### Analytics journey

- [ ] Final diagnosis counts use physician-final diagnosis only.
- [ ] Triage diagnosis counts are separate from final diagnosis counts.
- [ ] Presenting symptoms are separate from diagnosis counts.
- [ ] A single final diagnosis renders as 100% of the chart with `n=1` confidence context.
- [ ] Multiple diagnosis categories render exact proportions.
- [ ] Custom diagnoses appear in a governed custom/unclassified bucket without being silently relabeled.
- [ ] Overview and analytics use the same explicit final-diagnosis payload and time range.
- [ ] Refresh and browser navigation preserve the diagnosis decision and its category.

### Regression and persistence

- [ ] Save diagnosis, navigate away, return, and verify the same final diagnosis.
- [ ] Refresh the diagnosis page and verify match status and catalog selection.
- [ ] Refresh analytics and verify final diagnosis distributions remain identical.
- [ ] Verify triage-only encounters do not appear as physician-final diagnoses.
- [ ] Verify a physician correction changes final diagnosis statistics without rewriting triage history.
- [ ] Verify custom category creation and later mapping preserve audit history.

### Specialist performance journey

- [ ] A radiologist can receive, accept, complete, and verify a study while each transition persists after refresh.
- [ ] A pharmacist can review and complete a medication order while verification and intervention events persist after refresh.
- [ ] A pathologist can accession, process, result, and sign out a specimen while the full lifecycle remains auditable.
- [ ] Radiology and pathology requests retain the EHR context captured at order time, and specialists can view it after reloading their queues.
- [ ] Historical seed encounters remain discharged and never appear as active triage work.
- [ ] Specialist dashboards count completed work from work-item events, not from shift attendance alone.
- [ ] Turnaround metrics use recorded lifecycle timestamps and respect priority-specific targets.
- [ ] Waiting-for-upstream-information time is separated from active specialist processing time.
- [ ] Amendments, corrections, rejections, and clarifications are visible as quality metrics rather than silently discarded.
- [ ] Specialist metrics remain scoped to the selected hospital and selected time range.
- [ ] Overview and detailed analytics show the same specialist totals and data-quality state.
- [ ] Empty, low-sample, and reliable states are explicit for each specialist role.
- [ ] Radiology TAT is calculated from acquisition completion to signed report, not shift start/end.
- [ ] Pharmacy safety metrics distinguish warning overrides, pharmacist interventions, and accepted interventions.
- [ ] Pathology critical-value notification time is persisted with recipient and escalation route.
- [ ] Specimen rejection and contamination reasons are visible without being counted as completed results.
- [ ] Crossmatch-to-transfusion ratio handles zero transfusions without division errors.
- [ ] Ancillary panel values remain correct after browser refresh and route navigation.
- [ ] A critical pathology result creates a persisted alert and reaches the configured supervisor/escalation queue.
- [ ] Real-time delivery failure does not erase the persisted critical event.

## 12. Implementation order after approval

1. Confirm the data contract and terminology ownership.
2. Add encounter diagnosis metadata and backward-compatible persistence.
3. Update the physician diagnosis UI with triage comparison and guided catalog selection.
4. Add diagnosis/symptom catalog APIs and admin governance surfaces.
5. Update analytics aggregation to use physician-final diagnosis only.
6. Split final diagnosis, triage diagnosis, and symptom distributions.
7. Align Overview and Analytics time range/source contracts.
8. Run browser user journeys and persistence acceptance tests.
9. Add specialist work-item lifecycle events and role-specific metrics.
10. Migrate or classify legacy encounters and specialist records.
11. Move on to Labs and Pharmacy only after diagnosis and specialist analytics acceptance suites are green.

## 13. Decisions needed before implementation

Please confirm these points before code changes:

- Should the diagnosis catalog be global across all hospitals, hospital-specific, or both with overrides?
- Which terminology standard should be used first: internal catalog only, ICD-10, SNOMED CT, or a staged combination?
- Should a physician be required to choose a catalog category, or may free-text custom diagnoses remain publishable immediately?
- Who approves new custom categories and synonym mappings?
- Should legacy `encounter.diagnosis` values be treated as physician-final, legacy-unverified, or excluded until reviewed?
- Should symptom statistics be visible on the main analytics page or only in an expanded clinical data-quality view?
- Should ancillary telemetry be a new `AncillaryPerformanceTracking` collection, embedded work-item subdocuments, or a hybrid event plus read-model design?
- Which targets define orange/red status for each modality, medication safety workflow, and specimen type?
- Should critical pathology alerts use the existing polling operations feed first, or is sub-minute WebSocket/SSE delivery required for trial readiness?
- Who owns approval of radiology discordance, pharmacy override, and pathology rejection classifications?

## Approval gate

No implementation work should begin until this plan is confirmed, especially the rules for physician-final diagnosis, legacy records, and custom statistical categories.

## Release sequence: analytics, communication, deployment

## 14. Surgery and scheduled appointment workflows

These workflows are required before chat hardening and deployment. The repository contains partial surgery records, scheduled-surgery views, calendar aggregation, and appointment foundations, but the complete user journeys, outcomes, analytics, and persistence acceptance tests still need to be finished.

### 14.1 Surgery lifecycle

Each surgery record must follow an auditable lifecycle:

`REQUESTED -> SCHEDULED -> PREOP_READY -> IN_PROGRESS -> COMPLETED`

Exception states include `CANCELLED`, `POSTPONED`, `NO_SHOW`, and `ABORTED`.

The record should persist hospital and encounter scope, procedure and normalized category, type and priority, scheduled/actual timestamps, surgeon, theatre and team, pre-operative readiness, exception reasons, outcome, complications where appropriate, post-operative disposition, follow-up, actor, and status history.

### 14.2 Surgery outcome metrics

- scheduled-to-start delay
- procedure duration variance against expected duration
- cancellation, postponement, no-show, and abort rates
- completion rate by procedure and priority
- theatre utilization and idle time
- urgent/emergency response time
- pre-operative readiness completion rate
- complication and unplanned-return rate where data is available
- post-operative disposition distribution
- pending surgery backlog and age

A scheduled surgery must never be counted as completed without an explicit completion event and outcome.

### 14.3 Normal patient appointment booking

Appointment booking is a separate receptionist workflow. It must not be implemented as another surgery scheduler, and a surgery schedule must not be treated as a normal patient appointment.

Reception staff should be able to:

- search or register the patient within the hospital scope
- select an appointment type such as consultation, follow-up, referral, procedure review, or specialist visit
- select a department and eligible specialist
- view real availability and book an available slot
- capture referral reason, symptoms, preferred time, accessibility needs, and relevant notes
- confirm the booking and provide the patient with appointment details
- reschedule, cancel, waitlist, or mark a patient as a no-show with a reason
- view the patient’s upcoming and historical appointments
- avoid double-booking a specialist, room, equipment, or required resource

The patient-facing and receptionist-facing records must use the same appointment identifier and lifecycle. Reception staff may manage scheduling facts; clinical staff own clinical findings and final diagnoses.

### 14.4 Specialist appointment fulfillment

The appointment workflow must also record what happens when the scheduled visit actually occurs. This is the missing operational flow.

At the appointment time, the system should support:

1. receptionist check-in
2. queue placement and arrival timestamp
3. specialist acknowledgement or start-of-visit event
4. consultation, procedure, or review work
5. completion or outcome capture
6. follow-up appointment, referral, order, or discharge instruction

The specialist must be able to record that the appointment was:

- completed
- partially completed
- unable to proceed because of missing information or resources
- redirected to another department
- converted into an encounter or linked to an existing encounter

The system must distinguish a booked appointment from a patient who arrived, a visit that started, and a visit that completed. These are separate facts and timestamps.

### 14.5 Scheduled appointment lifecycle

Appointments must be persisted independently from shifts and surgery records and follow this lifecycle:

`REQUESTED -> BOOKED -> CONFIRMED -> CHECKED_IN -> IN_PROGRESS -> COMPLETED`

Exception states include `RESCHEDULED`, `CANCELLED`, `NO_SHOW`, `WAITLISTED`, `PARTIALLY_COMPLETED`, and `UNABLE_TO_PROCEED`.

Each appointment should persist hospital, patient, encounter, provider, department, and resource scope; appointment type; scheduled and actual timestamps; confirmation/reminder status; cancellation, reschedule, and no-show reasons; waitlist position; linked diagnostic or treatment workflow; fulfillment outcome; actor; and audit history.

### 14.6 Appointment metrics

- booking-to-appointment lead time
- confirmation, check-in, and no-show rates
- cancellation and reschedule rate
- appointment start delay
- visit duration
- provider and department throughput
- waitlist conversion rate
- reminder delivery and acknowledgement rate
- appointment backlog and overdue appointments
- booked versus checked-in versus started versus completed counts
- specialist appointment fulfillment rate
- unable-to-proceed and redirected-visit rate
- average time from check-in to specialist start
- appointment outcome and follow-up completion rate

A scheduled appointment is a scheduling fact. It must not be presented as a completed visit until an explicit completion event exists.

### 14.7 UI integration surfaces

Connect these existing or partial surfaces without creating duplicate workflow pages:

- `SurgeryPage.jsx` for encounter-level surgery creation and outcome capture
- `ScheduledSurgeriesPage.jsx` for surgery scheduling and theatre coordination only
- a dedicated receptionist appointment-booking workspace for patient and specialist appointment management
- a specialist appointment queue/workspace for check-in, start, fulfillment, and outcome capture
- calendar and operations-feed views for persisted schedule/status events
- the appointment model and calendar controller for appointment lifecycle
- encounter timeline/EHR views for surgery and appointment history
- analytics overview and specialist panels for surgery and appointment metrics

All views should read the same persisted records with role-appropriate actions.

### 14.8 Surgery and appointment acceptance tests

- [ ] Create a surgery request, schedule it, refresh, and verify the record remains.
- [ ] Move surgery through pre-op, in-progress, and completed states with history preserved.
- [ ] Record completed surgery outcome and verify it appears in the encounter timeline and analytics.
- [ ] Cancel, postpone, no-show, or abort surgery and verify the reason persists and is counted correctly.
- [ ] Create an appointment, confirm it, check the patient in, complete it, and verify the lifecycle after refresh.
- [ ] Receptionist can book a normal patient consultation with an available specialist without creating a surgery record.
- [ ] Receptionist can reschedule, cancel, waitlist, and mark a normal appointment as no-show with persisted reasons.
- [ ] Specialist can see only their assigned/upcoming appointments and record check-in, start, completion, or unable-to-proceed outcome.
- [ ] A completed specialist appointment can create or link an encounter without overwriting existing triage history.
- [ ] A surgery schedule does not appear as a normal appointment unless explicitly linked for display.
- [ ] Normal appointment booking does not create a surgery or procedure outcome by default.
- [ ] Reschedule or cancel an appointment and preserve both old and new schedule facts in audit history.
- [ ] Ensure no-show and cancellation metrics do not inflate completed-visit counts.
- [ ] Ensure calendar, operations feed, encounter timeline, overview, and detailed analytics agree on status.
- [ ] Ensure hospital scoping prevents cross-hospital surgery or appointment visibility.
- [ ] Ensure notification delivery failure does not remove the underlying surgery or appointment event.

The application should move toward deployment in this order:

1. **Complete the clinical and ancillary analytics foundation**
  - physician-final diagnosis is separated from triage diagnosis
  - symptoms, diagnoses, custom categories, and synonyms are governed
  - radiologist, pharmacist, and pathologist workflows emit measurable events
  - Overview and Analytics consume the same live payload and time range
  - empty, low-confidence, and reliable states are truthful

2. **Perfect chat communication**
  - staff-to-staff messages are hospital-scoped and permission-aware
  - message delivery, read state, acknowledgements, and escalation persist after refresh
  - clinical handovers retain encounter, sender, recipient, priority, and timestamps
  - failed delivery is visible and retryable rather than silently lost
  - urgent specialist and pathology alerts appear in the correct operations/chat channel
  - notification counts match the persisted unread messages
  - browser journeys verify sending, receiving, refreshing, navigating away, and returning

3. **Run the complete browser acceptance loop**
  - authenticate with a backend-issued token
  - verify hospital scoping
  - create or confirm live encounters and specialist work items
  - verify analytics, diagnosis decisions, chat messages, and alerts
  - refresh each page and confirm persistence
  - compare Overview, Analytics, specialist panels, and communication surfaces
  - capture and resolve console, network, authorization, and rendering errors

4. **Deploy only after release gates pass**
  - focused tests and builds pass
  - no unresolved 401/403 or background polling errors
  - no mock values are used as live defaults
  - database indexes and migration status are verified
  - audit records and retention behavior are verified
  - environment secrets and backend/frontend URLs are configured for deployment
  - a production smoke test confirms login, live analytics, chat, alerts, and refresh persistence

Deployment is therefore the final step, not a substitute for completing or validating the workflows. The release candidate is ready only when the clinical analytics and communication acceptance suites are green in the shared browser.

## Future integration track: patient voice, emergencies, and AI scheduling

This capability is intentionally **outside the current deployment runway**. It should be designed as a future integration track after the current clinical analytics, specialist workflows, surgery, appointments, communication, and deployment gates are complete.

### Future objective

Allow a patient to use a voice channel to:

- call for a medical emergency
- communicate in supported South African languages
- have speech transcribed into structured text
- receive safe routing to the appropriate emergency or hospital workflow
- request or modify a normal appointment when the situation is not an emergency
- have an AI assistant check specialist availability, schedules, and shift coverage before proposing an appointment

### Emergency boundary

Emergency calls must be handled as a safety-critical pathway, not as ordinary appointment scheduling.

The future design must:

- detect emergency intent and immediately prioritize human/emergency escalation
- capture caller identity, location, hospital, language, transcript, confidence, and timestamp
- preserve the original audio and transcription metadata subject to privacy policy
- never delay emergency escalation while waiting for an AI answer or schedule lookup
- route to the configured emergency desk, nursing supervisor, or emergency services process
- create a persisted emergency event and audit trail
- support human takeover and correction of transcription or classification
- clearly communicate when the system cannot understand or safely classify the call

The AI must not diagnose the patient or provide false reassurance. It may collect and route information, but urgent clinical decisions remain with qualified staff and emergency protocols.

### South African language and transcription integration

The future integration should define support for the required South African languages, dialect variation, code-switching, accents, names, medical terms, and noisy call environments.

Required capabilities:

- language identification with user confirmation
- audio-to-text transcription with timestamps
- confidence scores and unintelligible-segment markers
- medical vocabulary and hospital-name adaptation
- translation or normalized summary only when the original transcript is retained
- speaker and handover context where legally and technically appropriate
- human correction workflow for clinically important transcripts
- explicit consent, retention, access control, and audit rules for recorded audio

Transcripts used for scheduling or clinical routing must remain distinguishable from clinician-authored clinical notes.

### AI appointment scheduling integration

For non-emergency requests, a future AI scheduling assistant may:

- identify the patient or begin a safe registration flow
- understand the requested service, department, specialist, language, location, and preferred time
- inspect live specialist schedules, shifts, leave, appointment capacity, and required resources
- propose available slots without claiming a booking until the booking transaction succeeds
- confirm the appointment details back to the patient in the selected language
- create, reschedule, cancel, or waitlist an appointment through the same canonical receptionist booking API
- hand off to reception when identity, clinical urgency, eligibility, or availability is ambiguous

The assistant must not schedule against a stale cached shift. Availability must be rechecked transactionally before confirmation to prevent double booking.

### Future integration architecture

The future track should be built around these boundaries:

- telephony or voice-channel adapter
- speech recognition and language identification service
- transcript and consent service
- emergency-intent and routing service
- appointment orchestration service
- staff schedule and capacity read model
- human handoff and supervisor queue
- immutable audit and safety-event store

The AI integration should call the existing appointment and staff-schedule contracts rather than creating a second appointment or shift data model.

### Future acceptance tests

- [ ] Patient can initiate an emergency call and reach the configured human escalation path without waiting for scheduling.
- [ ] Emergency event persists with audio/transcript metadata, language, confidence, hospital scope, and timestamps.
- [ ] Human staff can correct the transcript and emergency classification without losing the original record.
- [ ] Supported South African language speech is transcribed with confidence and uncertainty markers.
- [ ] Code-switched or unclear speech triggers clarification or human handoff rather than unsafe assumptions.
- [ ] AI checks live staff shifts and appointment capacity before proposing a slot.
- [ ] A slot is transactionally rechecked before booking confirmation.
- [ ] AI-created appointments appear identically in receptionist, patient, specialist, calendar, and analytics views.
- [ ] Voice scheduling cannot create duplicate appointments during retries or repeated calls.
- [ ] Emergency requests cannot be downgraded into routine appointments by the AI alone.

### Future-track approval gate

Do not include this voice and AI scheduling track in the current deployment readiness claim. Before implementation, approve the supported languages, emergency escalation ownership, audio/transcript retention policy, consent model, AI provider, human handoff requirements, and safety evaluation protocol.
