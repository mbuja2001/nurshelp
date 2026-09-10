# Alert System and Live Operations Plan

Status: Phase 1 authorization foundation and the first operations-feed slice are implemented; queue-engine, laboratory, pharmacy, and staff-message integration remain gated work.

## Goal

Build one live operations feed shared by `SidebarLayout.jsx`, the Nurse Dashboard priority queue, and future physician, pharmacist, radiology, and pathology dashboards.

The feed must support:

- Priority queue changes and patient escalation alerts.
- Patient identity, chief complaint, priority, wait time, and encounter reference.
- Staff messages.
- Clinical handover notifications.
- Pending and completed radiology and pathology work.
- Prescription and pharmacy status changes.
- Accurate alert and message counts.
- Desktop queue lane and responsive emoji modal using the same data.
- Hospital and recipient isolation on every read or write.
- Polling first, with an upgrade path to SSE or WebSockets.

## Current Architecture Findings

### Frontend

- `frontend/src/Dashboard/SidebarLayout.jsx` owns the header Alerts and Messages buttons.
- Sidebar notifications currently poll `/api/notifications/unread`.
- Sidebar notifications fall back to demo alerts when the API returns no items or fails.
- Sidebar message rows are hard-coded.
- Sidebar message count is derived from the alert count and is not authoritative.
- `frontend/src/Dashboard/NurseDashboard.jsx` computes priority from encounter data separately.
- The desktop queue lane and responsive `🎯` modal are separate render paths and can drift.
- Nurse shift status is fetched from `/api/staff/today-record`.
- Scheduled shift data is available from `/api/shifts/assigned/today`.

### Backend

- `backend/src/models/notification.model.js` is designed mainly for clinical handovers.
- `backend/src/routes/notification.routes.js` exposes unread, list, read, acknowledge, dismiss, and encounter notification routes.
- Notification read, acknowledge, and dismiss updates currently query by notification ID without also enforcing recipient and hospital ownership.
- `backend/src/routes/dashboard.route.js` exposes a queue route, but its controller currently returns an empty placeholder queue.
- Waiting encounters are available from `/api/encounters/waiting`.
- Lab routes exist under `/api/lab-requests`.
- Specialist queues exist under `/api/specialist/radiology/queue`, `/api/specialist/pathology/queue`, and `/api/specialist/pharmacy/queue`.
- Pharmacy request routes exist under `/api/pharmacy/requests` and `/api/pharmacy/messages/pending`.
- Prescription routes exist under `/api/prescriptions`.
- Bed statistics exist under `/api/beds/stats`.
- Handover and pharmacy controllers already create some Notification documents.
- Lab status changes do not yet consistently create generalized notifications.

## Canonical Event Contract

All event sources must map to this shape before reaching a dashboard:

```js
{
  id: String,
  type: String,
  category: 'ALERT' | 'MESSAGE' | 'QUEUE',
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO',
  status: 'UNREAD' | 'READ' | 'ACKNOWLEDGED' | 'DISMISSED',
  title: String,
  message: String,
  source: 'PRIORITY_QUEUE' | 'HANDOVER' | 'RADIOLOGY' | 'PATHOLOGY' | 'PHARMACY' | 'PRESCRIPTION' | 'SURGERY' | 'STAFF_MESSAGE' | 'SYSTEM',
  patient: {
    encounterId: String | null,
    mrn: String | null,
    name: String | null,
    chiefComplaint: String | null
  } | null,
  priority: {
    code: 'RED' | 'ORANGE' | 'YELLOW' | 'GREEN' | null,
    score: Number | null, // display-only; never used for ordering
    waitMs: Number | null,
    label: String | null
  } | null,
  metadata: Object,
  action: {
    label: String,
    route: String | null
  } | null,
  createdAt: String,
  updatedAt: String
}
```

Rules:

1. `id` must be stable and unique for the source record plus event type.
2. `patient` is required for patient-related events and null for system events.
3. `metadata` may contain source-specific fields but must not replace canonical fields.
4. Display code must never invent a patient name, count, provider, or clinical event.
5. The same event must render consistently in the sidebar dropdown, desktop lane, and responsive modal.

## Canonical Operations Feed

Add an authenticated endpoint:

```text
GET /api/operations/feed?limit=50&cursor=<optional>&since=<optional>
```

Response:

```js
{
  success: true,
  serverTime: String,
  cursor: String | null,
  feed: {
    alerts: Event[],
    messages: Event[],
    queue: {
      waiting: Number,
      patients: QueuePatient[],
      providers: ProviderLoad[]
    },
    labs: Event[],
    prescriptions: Event[],
    surgeries: Event[]
  }
}
```

The endpoint must:

- Authenticate the user.
- Derive `hospitalId` and user ID from the authenticated token.
- Apply role and recipient visibility rules.
- Query only the hospital's records.
- Return deterministic ordering: severity/priority first, then wait time, then creation time.
- Return empty arrays and zero counts when no data exists.
- Never return demo fallback data.

## Queue Rules and Source of Truth

The backend queue engine, implemented as `executeSATSQueueRanking` or its successor, is the
single source of truth for priority, ordering, breach state, and data-integrity isolation.
Nothing downstream recomputes these values.

`frontend/src/utils/priority.js` is presentation-only: it may map canonical categories to CSS
tokens, labels, and icons, but it must not compare, rank, sort, or derive escalation state from
raw patient fields. Any ranking logic currently there must move server-side before the feed is
considered authoritative.

The operations feed maps the queue engine's `orderedQueueLane` directly into `queue.patients`.
It must not re-sort, re-bucket, or recompute that array. The desktop lane and responsive `🎯`
modal consume that array verbatim and may differ only in layout.

Within a SATS colour bucket, patients are clinically equal. Ordering uses the queue engine's
breach state, resource-count tiebreaker, and FIFO rules, never a continuous score.

`priority.score` remains optional display-only data, such as a raw TEWS total. No consumer may
use it in a sort comparator or any conditional that changes render order.

Queue patient data must include:

- Encounter ID.
- Patient name and complaint.
- Priority category, optional display-only score, and server ordering metadata.
- Wait start and calculated wait time.
- Current queue state: `IN_PROGRESS`, `NEXT_UP`, `WAITING`, or `AI_TRIAGING`.
- Assigned clinician, if present.
- Signals explaining escalation.

Canonical queue field mapping:

| Feed field | Queue engine source | Rule |
|---|---|---|
| `priority.code` | `triageCategory` | `RED`, `ORANGE`, `YELLOW`, `GREEN`, or `NEEDS_REVIEW` |
| `priority.waitMs` | `elapsedWaitMins * 60000` | Computed once at feed-build time |
| `priority.label` | category plus `escalationTriggered` | Presentation only |
| `priority.score` | optional raw clinical score | Never used for ordering |
| queue array order | `orderedQueueLane` | Passed through without re-sorting |

The desktop lane and responsive modal must receive the same normalized `queue.patients` array. They may differ in layout only.

Provider workload must be calculated from live assignments and active statuses. It must not use fixed provider names, fixed percentages, or reference-only sample patients.

Recommended provider load calculation:

```text
active workload = active assigned waiting/in-progress encounters
capacity = configured shift capacity or clinician workload limit
load percent = min(100, round(active workload / capacity * 100))
```

If capacity is unavailable, display `Unconfigured` rather than a fabricated percentage.

## Alert Types

### Priority alerts

- `PRIORITY_ESCALATED`: patient crossed a wait threshold.
- `CRITICAL_TRIAGE`: patient has a critical priority or clinical override.
- `QUEUE_CONGESTION`: a clinician or service exceeds its configured workload threshold.
- `QUEUE_REROUTE_RECOMMENDED`: a different eligible clinician has lower load.
- `DATA_INTEGRITY_ALERT`: `triageCategory` is missing, malformed, or failed queue-engine validation; the record is isolated pending manual review and this is not a clinical urgency judgement.

### Clinical workflow alerts

- `HANDOVER_PENDING`: clinician has an unread handover.
- `HANDOVER_ACKNOWLEDGED`: handover was accepted.
- `RADIOLOGY_PENDING`: imaging order requires action.
- `RADIOLOGY_RESULT_READY`: imaging result is ready for review.
- `PATHOLOGY_PENDING`: laboratory order requires action.
- `PATHOLOGY_RESULT_READY`: laboratory result is ready for review.
- `PRESCRIPTION_PENDING`: prescription requires pharmacy action.
- `PRESCRIPTION_STATUS_CHANGED`: prescription was accepted, dispensed, declined, or completed.
- `PHARMACY_MESSAGE`: pharmacist message requires attention.
- `SURGERY_SCHEDULED`: a procedure has been scheduled and requires operational awareness.
- `SURGERY_STATUS_CHANGED`: a procedure moved to pending, scheduled, in-progress, completed, post-op review, complication, or cancelled.
- `URGENT_SURGERY`: an urgent or emergency procedure needs immediate operating-theatre coordination.
- `SURGERY_CANCELLED`: a scheduled procedure was cancelled and affected staff must be informed.

### Staff and system messages

- `STAFF_MESSAGE` for direct or team messages.
- `SYSTEM_STATUS` for service or operational notices.

### Data integrity alert contract

Each queue-engine `dataIntegrityIssues` entry produces one stable event:

```js
{
  id: `data-integrity:${encounterId}`,
  type: 'DATA_INTEGRITY_ALERT',
  category: 'ALERT',
  severity: 'CRITICAL',
  status: 'UNREAD',
  title: 'Triage data needs review',
  message: 'Triage category could not be validated. Manual re-triage required.',
  source: 'PRIORITY_QUEUE',
  patient: { encounterId, mrn, name, chiefComplaint: null },
  priority: { code: null, score: null, waitMs, label: 'Needs review' },
  metadata: { reason: rawCategoryValueReceived, detectedAt },
  action: { label: 'Re-triage now', route: `/triage/re-assess/${encounterId}` },
  createdAt,
  updatedAt
}
```

This event must reach the charge nurse or shift lead role, in addition to normal hospital scope.
That role must be represented in the role taxonomy before this event ships. Delivery must be
push-capable; polling alone is insufficient for the production guarantee.

The stable ID makes creation idempotent. Repeated evaluations must not create duplicates. The
primary close path is automatic resolution when the encounter no longer appears in
`dataIntegrityIssues`. Manual dismissal should be disallowed or require an explicit reason and
acknowledgment. Add `acknowledgedBy` and `acknowledgedAt` to preserve the audit trail.

Every type needs an explicit source, severity mapping, recipient rule, and action route before it is enabled in production.

## Data Source Mapping

| Event/data | Existing source | Required work |
|---|---|---|
| Handover notifications | `Notification` and handover controllers | Normalize existing records into the event contract |
| Waiting patients | `GET /api/encounters/waiting` | Move or share priority normalization with feed |
| Bed occupancy | `GET /api/beds/stats` | Include in dashboard metrics, not necessarily alert feed |
| Radiology queue/results | specialist and lab routes/controllers | Add hospital-safe aggregation and event creation |
| Pathology queue/results | specialist and lab routes/controllers | Add hospital-safe aggregation and event creation |
| Pharmacy requests/messages | pharmacy routes/controllers | Normalize request and message statuses |
| Prescriptions | prescription/pharmacy models and routes | Align status names and add event emission |
| Scheduled surgeries | `Encounter.surgery_records[]`, `/api/encounters/physician/surgeries` | Normalize upcoming procedures, status changes, urgency, surgeon, patient, and theatre workflow events |
| Staff messages | existing message endpoints where available | Identify authoritative message model; remove hard-coded UI rows |
| Shift timing | `/api/staff/today-record`, `/api/shifts/assigned/today` | Display actual clock-in plus scheduled shift end |
| Data integrity failures | Queue engine `dataIntegrityIssues[]` | Emit one idempotent `DATA_INTEGRITY_ALERT` per encounter and auto-resolve after correction |

## Required Backend Changes

### Phase 1: Schema and security

- Extend Notification with generalized fields:
  - `type`
  - `category`
  - `title`
  - `message`
  - `source`
  - `metadata`
  - `action_route`
  - `action_label`
- Preserve legacy handover fields for compatibility.
- Add indexes for `hospital_id`, recipient, status, type, and created time.
- Add recipient and hospital filters to read, acknowledge, and dismiss updates.
- Add tests proving one hospital or clinician cannot mutate another clinician's notification.
- Add `acknowledgedBy`, `acknowledgedAt`, and resolution fields for auditable critical alerts.

Progress: Generalized event fields, audit fields, and clinician/hospital ownership filters are implemented. Focused `node:test` coverage verifies ownership filters for clinician and hospital isolation.

Gate: schema tests and authorization tests pass.

### Phase 2: Feed service

- Create an operations feed controller/service.
- Normalize persisted notifications.
- Aggregate live waiting encounters and calculate queue metadata.
- Aggregate lab, radiology, pathology, pharmacy, and prescription events allowed for the current role.
- Aggregate scheduled surgery records allowed for the current role, including upcoming procedures and urgent status changes.
- Deduplicate events by stable source key.
- Return server time and cursor/since support.
- Pass through `orderedQueueLane` without reordering and expose integrity issues as canonical events.

Progress: `/api/operations/feed` is registered and currently exposes scoped unread notifications and scoped surgery events. Its queue is explicitly marked `engineReady: false` until the backend SATS queue engine is implemented; it does not fabricate ordering.

Gate: authenticated feed returns correct empty state, seeded state, role visibility, and hospital isolation. A snapshot test proves feed queue order is byte-for-byte traceable to direct `orderedQueueLane` output for the same input state.

### Phase 3: Event creation and status transitions

- Emit or persist events when handovers are created.
- Emit events when priority crosses an escalation threshold.
- Emit events when lab orders are created, received, completed, verified, or reported.
- Emit events when prescriptions are submitted, accepted, declined, dispensed, or completed.
- Emit events when surgery records are created, scheduled, started, completed, moved to post-op review, complicated, or cancelled.
- Mark `URGENT_SURGERY` as critical/high according to `surgery_type` and `priority`, and route it to the assigned surgical team, charge nurse, and relevant operating-theatre role.
- Emit pharmacy messages through the same event contract.
- Make event creation idempotent for repeated status updates.
- For every queue evaluation, create exactly one `DATA_INTEGRITY_ALERT` per issue using
  `data-integrity:${encounterId}` and auto-resolve it when the issue disappears.

Gate: one workflow transition creates one event and repeated polling does not duplicate it. Inject
a malformed `triageCategory`, verify one critical alert appears, correct the record, and verify
the alert auto-resolves within one polling cycle without manual dismissal.

### Phase 4: Live transport

Start with one shared polling interval of 5 to 10 seconds. Use `since` or cursor filtering to reduce payloads.

Then evaluate SSE:

```text
GET /api/operations/stream
```

SSE requirements:

- Authenticated connection.
- Hospital and recipient scope enforced server-side.
- Reconnect with `Last-Event-ID`.
- Polling fallback when the stream fails.
- No duplicate events during reconnect.

WebSockets are not required until bidirectional staff messaging needs them.

Gate: network throttling and reconnect tests preserve alert ordering and counts.

## Required Frontend Changes

### SidebarLayout

- Replace the current notification-only fetch with the operations feed.
- Remove all demo alerts and hard-coded staff messages.
- Use authoritative alert and message counts from the feed.
- Render alert type, severity, patient, complaint, priority, wait time, timestamp, and action.
- Add read, acknowledge, and dismiss actions where the event type permits them.
- Keep the Alerts icon as the primary global entry point.
- Keep dropdown positioning stable at desktop and mobile widths.
- Show a truthful empty state: `No new alerts` or `No new messages`.

### NurseDashboard

- Consume the same feed snapshot as SidebarLayout through a shared hook/context.
- Keep the Queue Manager in the existing right lane.
- Keep the compact `🎯` launcher and modal on narrow screens.
- Use the same queue patient array in both render paths.
- Remove View buttons from the queue if the intended action is an inline queue action; otherwise define one canonical action route.
- Render urgent queue events as structured alert cards rather than a separate legacy alert modal.
- Keep triage ranking sourced from the canonical priority algorithm.
- Render actual clock-in and scheduled shift end in the header:
  - actual `currentRecord.clockInTime`
  - scheduled `shift.shiftEnd` from `/api/shifts/assigned/today`
  - actual `currentRecord.clockOutTime` only after checkout

### Other dashboards

Implement the shared feed consumer after Nurse Dashboard is stable:

1. Physician Dashboard.
2. Pharmacist Dashboard.
3. Radiology Dashboard.
4. Pathology Dashboard.
5. Manager/Admin operational dashboard.

Each dashboard may filter event categories but must not create a second event format.

## Validation Gates

Do not advance a phase until its gate passes.

### Data correctness

- No demo rows or fabricated counts.
- Patient names and complaints match backend records.
- Priority categories match the algorithm.
- Wait timers update without reordering incorrectly.
- Queue count equals the normalized patient array length.
- Lab and prescription statuses match their source records.

### Security

- Hospital A cannot see Hospital B events.
- Clinician A cannot read or mutate Clinician B notifications.
- Role restrictions are enforced in the feed and source routes.
- Patient data is not exposed in events outside the user's permitted workflow.

### UI behavior

- Desktop: Queue Manager occupies the established right lane.
- Narrow viewport: right lane collapses to the `🎯` launcher.
- Modal content matches the desktop queue data.
- Alerts icon opens the current alert feed without stale demo content.
- Messages icon opens only real messages.
- Long patient names and complaints do not overlap controls.
- Loading, empty, error, and reconnect states are visible and truthful.

### Operational testing

- Seed one urgent triage patient.
- Confirm the queue ranking changes.
- Confirm the alert count changes.
- Confirm the sidebar alert shows the patient and complaint.
- Create a lab request and confirm a pending event.
- Complete a lab and confirm a result-ready event.
- Submit a prescription and confirm a pharmacy event.
- Change prescription status and confirm one updated event.
- Schedule an elective surgery and confirm a `SURGERY_SCHEDULED` event.
- Schedule an urgent or emergency surgery and confirm an urgent alert with patient, procedure, surgeon, time, and action route.
- Change a surgery to in-progress, completed, post-op review, and cancelled, confirming one status event per transition.
- Clock a nurse in and verify actual start plus scheduled end.
- Refresh the desktop and narrow layouts and compare the same live data.

## Execution Order

1. Add and test generalized notification fields and ownership filters.
2. Implement the operations feed with empty-state correctness.
3. Add queue normalization and provider workload calculations.
4. Add lab and prescription aggregation.
5. Replace SidebarLayout demo data with the feed.
6. Share the feed state with NurseDashboard desktop lane and `🎯` modal.
7. Correct Nurse Dashboard shift header rendering.
8. Add event creation for missing lab, pathology, radiology, pharmacy, and prescription transitions.
9. Add polling cursor/since support.
10. Add SSE only after polling behavior is proven.
11. Roll out the shared consumer to other dashboards.
12. Run security, data, responsive, and end-to-end sandbox verification.
13. Prepare deployment configuration and production observability.

## Definition Of Done

The alert system is ready for deployment when:

- All dashboard surfaces consume the same canonical feed contract.
- No UI uses demo alerts, hard-coded messages, fixed provider names, or fabricated workload percentages.
- Priority queue and alert popup update from the same live snapshot.
- Lab, pathology, radiology, pharmacy, prescription, handover, and triage events are represented.
- Read/acknowledge/dismiss operations are scoped securely.
- Polling works reliably with a tested fallback/error state.
- Desktop and responsive layouts show identical underlying data.
- Automated tests and browser verification pass for every phase.
