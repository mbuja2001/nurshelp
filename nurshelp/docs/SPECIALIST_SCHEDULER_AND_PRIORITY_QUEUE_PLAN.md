# Specialist Scheduler, Priority Queue, and Role-Based Workflow Plan

## Objective
Redesign the physician, nursing, pharmacy, radiology, pathology, and specialist experience around a unified operational scheduler and priority queue that reflects real hospital activity.

## Core Product Goals
- Replace the physician dashboard refresh behavior with a real scheduling system for planned care.
- Support scheduling for both hospitalized and non-hospitalized patients.
- Track operational events for the day, including:
  - diagnostics and labs
  - medication administration
  - surgeries
  - diagnoses
  - bed changes
  - handoffs and specialist consults
- Preserve urgency and severity as first-class signals in the queue.
- Give every role a consistent operational header with the specialist/department context.
- Move specialist pages from mock shells to real, hospital-scoped workflow views.

## Design Principles
1. Hospital-scoped data only
   - Every operational record must be associated with a hospital and, where relevant, an encounter.
   - New hospital setup must create a clean slate for patients, encounters, labs, admissions, and scheduling events.

2. Role-based access + role-specific UI
   - Physicians, nurses, pharmacists, radiologists, pathologists, and administrators each see a tailored workflow.
   - Specialist roles get a clinical sidebar and operational workspace instead of generic dashboards.

3. Real-time operational workflow
   - The scheduler and priority queue should be driven by live data, not static mock state.
   - Queue ordering should reflect severity, urgency, due time, and specialist availability.

4. Event-led operations model
   - Every significant action becomes an event in a unified timeline.
   - Events can be grouped by patient, encounter, specialty, department, and time.

## Proposed Architecture
### 1. Unified operational event model
Create a shared event structure used across the app:
- eventType: medication, surgery, diagnosis, lab_request, bed_change, discharge, consult, handoff
- patientId / encounterId
- hospitalId
- department / specialty
- priority / severity
- scheduledAt / dueAt / completedAt
- status: scheduled, in_progress, completed, delayed, cancelled
- actorId / actorRole

### 2. Scheduler + priority queue components
Each main dashboard will include:
- a calendar-style scheduler for the day/week
- a priority queue pane showing the next actions by severity/urgency
- a timeline view for scheduled events

### 3. Role-specific dashboard mapping
- Physician dashboard
  - admissions, diagnoses, surgery scheduling, consults, bed changes
- Nurse dashboard
  - triage, bed turnover, medication administration, patient events, escalation queue
- Pharmacy dashboard
  - medication orders, dispensing, refill planning, pharmacist shift context
- Radiology dashboard
  - imaging requests, slot scheduling, report completion, queueing
- Pathology dashboard
  - specimen tracking, result reporting, lab workflow queue

## Header Standardization
Every role-specific dashboard should include a shared header block with:
- role label
- department/specialty context
- current shift or coverage context
- user name
- hospital name

### Header examples
- Physician: “Cardiology Department” or “Emergency Department” depending on specialty/department
- Nurse: “Triage Department”
- Pharmacist: “Pharmacy Department” with pharmacist name and shift context
- Radiologist: “Radiology Department”
- Pathologist: “Pathology Department”

## Specialist Interaction Model
We need a clear relationship map between staff across specialties:
- Physicians create and update diagnosis, treatment, surgery, and consult requests.
- Nurses coordinate bed changes, medications, handoffs, and patient readiness.
- Pharmacy receives medication requests and returns dispensing status.
- Radiology receives imaging orders and returns results.
- Pathology receives lab/specimen requests and returns results.
- Specialists all feed their outcome back into the shared event queue and patient timeline.

### Suggested workflow relationships
- Physician -> Nurse: admission, diagnosis, surgery scheduling, treatment plan
- Physician -> Pharmacy: medication requests
- Physician -> Radiology: imaging orders
- Physician -> Pathology: specimen/lab orders
- Nurse -> Pharmacy: medication administration tracking
- Nurse -> Bed management: movement, discharge prep, transfer coordination
- Specialist -> Physician: diagnostic report / recommendation
- Specialist -> Nurse: status update for patient handoff

## UI/UX Plan
### 1. Physician dashboard redesign
- Replace refresh button logic with a scheduler experience.
- Display a daily agenda with event cards.
- Show a live priority queue driven by severity + urgency + schedule time.
- Heading should derive from role/specialty/department context.

### 2. Nurse dashboard redesign
- Adopt the same shared header as other specialized roles.
- Show triage department header.
- Display comparable scheduler + queue panel to support bed turnover and care coordination.

### 3. Pharmacy, radiology, pathology redesign
- Keep the existing overall layout if it already works, but replace mock data with live data from hospital-scoped endpoints.
- Introduce queue cards, scheduler columns, and status updates.
- Ensure all actions feed back into shared operational events.

### 4. Clinical sidebar
- Add a role-aware clinical sidebar for specialists.
- Restrict visible modules by role.
- Example modules:
  - Overview
  - Schedule
  - Queue
  - Patients
  - Orders / requests
  - Results
  - Handoffs

## Implementation Phases
### Phase 1 — Data model and backend foundations
- Add a shared event model and related API endpoints.
- Ensure all operations are hospital-scoped.
- Ensure encounter relationships remain scoped by hospital and can support multiple notes per day.

### Phase 2 — Shared UI shell
- Build a shared header component.
- Build a shared scheduler + priority queue component.
- Add role-aware sidebar modules.

### Phase 3 — Dashboard integration
- Integrate the scheduler/queue into physician and nurse dashboards first.
- Then wire pharmacy, radiology, and pathology.

### Phase 4 — Real data wiring
- Replace mock data with live encounter, lab, pharmacy, bed, and medication data.
- Ensure specialist pages update from the same backend event stream.

### Phase 5 — Access control and polish
- Restrict UI modules by role and department.
- Add timers, live refresh, and priority ordering rules.

## Suggested MVP Scope
1. Shared header
2. Shared scheduler + priority queue component
3. Physician dashboard schedule/queue integration
4. Nurse dashboard header + queue integration
5. Pharmacy dashboard live queue integration
6. Radiology and pathology live queue integration
7. Role-aware sidebar

## Next Actions
1. Create the shared scheduling event data model and API endpoints.
2. Implement the shared scheduler/priority queue UI component.
3. Update the physician and nurse dashboard headers.
4. Replace mock specialist page data with live backend-driven views.
5. Add role-based sidebar navigation and access control.
