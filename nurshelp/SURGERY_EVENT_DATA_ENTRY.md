# Surgery Event Data Entry

This document defines the missing workflow for recording actual surgical event data after a surgery is scheduled. The goal is to move beyond "scheduled surgery only" and support live capture of surgical start/end times, whether the surgery happened on time, and the total procedure duration.

## Problem

- The system currently schedules surgeries, requests labs, orders medications, and supports discharge workflows.
- What is missing is the ability to record the real surgical event when it actually happens.
- Without actual surgery event data, the system cannot track:
  - whether surgery started as planned
  - surgery duration and overrun
  - actual room usage and OR availability
  - downstream operational analytics and handoff accuracy

## Desired Outcome

When a surgery is scheduled, clinical staff should be able to:
- accept or confirm the scheduled surgery event
- record the actual surgery start time
- record the actual surgery end time
- compute and persist surgery duration
- update the surgery event status from `scheduled` to `in_progress` and `completed`
- surface the captured data in patient timelines, OR schedules, and surgical analytics

## Key Workflow Steps

1. Surgery is created as a scheduled event.
2. The responsible clinician or surgical team opens the surgery event from the OR schedule or encounter workflow.
3. The team marks the surgery as started and records the actual start time.
4. The team marks the surgery as completed and records the actual end time.
5. The system calculates `duration = actualEnd - actualStart`.
6. The event status updates and the surgical record is linked to the encounter/bed/patient.

## Suggested Data Model Additions

- `surgeryEvent` or `surgerySchedule` object should include:
  - `scheduledStart: DateTime`
  - `scheduledEnd: DateTime`
  - `actualStart: DateTime|null`
  - `actualEnd: DateTime|null`
  - `durationMinutes: Number|null`
  - `status: 'scheduled' | 'confirmed' | 'in_progress' | 'completed' | 'cancelled'`
  - `surgeonId: String`
  - `theaterId: String`
  - `patientId: String`
  - `encounterId: String`
  - `notes: String`

## API / Interaction Requirements

- Backend endpoints:
  - `PATCH /api/surgeries/:id/start`
    - body: `{ actualStart: DateTime }`
    - sets `status = 'in_progress'`
  - `PATCH /api/surgeries/:id/complete`
    - body: `{ actualEnd: DateTime }`
    - calculates duration and sets `status = 'completed'`
  - `PATCH /api/surgeries/:id/accept`
    - body: `{ accepted: true }`
    - transitions scheduled surgery into active workflow
- Optional combined endpoint:
  - `PATCH /api/surgeries/:id/event`
    - body: `{ actualStart?, actualEnd?, status? }`
    - supports start/completion in one payload

## Frontend UX

- OR schedule and surgery details should show both scheduled and actual values.
- Add an action button or toggle in the surgery event detail view:
  - `Start Surgery`
  - `Complete Surgery`
- When the event is started, store the timestamp and update the UI to show `In progress`.
- When the event is completed, show `Duration`, `Actual start`, and `Actual end`.
- If actual times are edited manually, recalculate duration immediately.

## Acceptance Criteria

- The surgery event detail view can transition an event from `scheduled` to `in_progress` and `completed`.
- Actual start and end times are persisted in the backend.
- Duration is derived from actual start/end times and stored or returned consistently.
- Scheduled and actual values are displayed together so users can compare plan vs reality.
- The workflow is available from both the OR schedule view and the patient encounter/surgery modal.

## Implementation Notes

- Use the existing surgery or schedule model in the backend if present, otherwise add a dedicated surgery events collection.
- Keep the workflow guarded by role-based auth so only surgical staff or authorized clinicians can move a surgery to `in_progress` or `completed`.
- Record both timestamps even if the surgery start and end are entered after the fact.
- Support manual correction of surgery duration for retrospective documentation if needed.

## Next Tasks

1. Identify the current surgery scheduling model and API in the codebase.
2. Add surgery event fields for `actualStart`, `actualEnd`, and `durationMinutes`.
3. Implement the backend start/complete endpoints.
4. Add the frontend surgery event action buttons and actual time entry fields.
5. Add tests or validation for time ordering and duration computation.

---

This document closes the gap between scheduling and real surgical event reporting, enabling accurate OR productivity tracking and patient care handoffs.