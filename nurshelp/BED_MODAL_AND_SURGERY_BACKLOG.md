# Bed Modal & Surgery Backlog

This file captures the next-stage objectives and tickets for the Physician bed modal, live bed/equipment updates, and acknowledgment flow.

## Goals
- Move the bed modal to a fully editable, live clinical action center.
- Allow physicians to update patient severity in the bed modal and persist it to the encounter.
- Surface live bed equipment status and track equipment usage.
- Ensure acknowledging a patient moves them from the "awaiting clinician" list to the "acknowledged" list.

---

## Tickets (high-level)

### 1) Add editable severity control in Bed Modal
- Path: `frontend/src/components/dashboards/PhysicianDashboard.jsx`
- Description: Show `Name`, `Gender`, `Age`, `Severity` (already displayed). Add a control to modify severity (dropdown) and an `Update severity status` button in the bed modal's Clinical Actions area.
- Acceptance criteria:
  - Dropdown shows severity levels (e.g., CRITICAL, HIGH, MODERATE, STABLE).
  - Clicking `Update severity status` calls the live endpoint and updates UI immediately.
  - Update is recorded into `encounter.status_history`.
- Priority: High

### 2) Implement live severity update endpoint
- Path: `backend/src/controllers/encounter.controller.js` and `backend/src/routes/encounter.route.js`
- Description: Add `PATCH /api/encounters/:id/severity` that accepts `{ severity: 'CRITICAL' }` and updates encounter triage/severity/status and appends a `status_history` record.
- Acceptance criteria:
  - Endpoint requires auth and validates access.
  - Returns updated encounter and new status_history entry.
- Priority: High

### 3) Show bed emoji icon centered in bed tiles
- Path: `frontend/src/components/dashboards/PhysicianDashboard.jsx`
- Description: Render a bed emoji (🛏️) centered in each bed tile prior to opening the modal; keep existing patient info layout when occupied.
- Acceptance criteria:
  - Icon appears visually centered on both occupied and empty tiles.
- Priority: Medium

### 4) Display live bed equipment in Clinical Snapshot
- Path: `frontend/src/components/dashboards/PhysicianDashboard.jsx` (clinical snapshot block)
- Description: Pull live equipment list for the selected bed (from an existing Bed/BedSchedule endpoint or a new `GET /api/beds/:bedId/equipment`) and show equipment status in the clinical snapshot square.
- Acceptance criteria:
  - Equipment list is fetched when opening bed modal and displayed.
  - If equipment status changes in backend, re-fetch on modal refresh or polling.
- Priority: High

### 5) Track equipment usage score per bed/encounter
- Path: `backend/src/models/bed.model.js` or `encounter.model.js`
- Description: Add a lightweight usage counter per equipment item (or per bed) that increments when equipment is used for a procedure; expose via API.
- Acceptance criteria:
  - Backend stores usage counts and provides them via `GET /api/beds/:bedId/equipment`.
  - Frontend shows usage metrics in the snapshot area.
- Priority: Medium

### 6) Move patient to acknowledged table on acknowledge
- Path: `frontend/src/components/dashboards/PhysicianDashboard.jsx`
- Description: When a clinician acknowledges a handover (existing endpoint `/api/encounters/:id/handover/acknowledge`), update frontend lists so the patient moves from the awaiting list to the acknowledged list.
- Acceptance criteria:
  - Acknowledge button calls the endpoint and on success removes item from awaiting list and adds to acknowledged table (or refreshes lists).
- Priority: High

### 7) Wire frontend ↔ backend and test flows end-to-end
- Tasks:
  - Hook `Update severity status` UI element to the severity endpoint.
  - Wire equipment calls and bed emoji UX.
  - End-to-end manual QA: open bed modal, change severity, confirm DB change, confirm patient moves on acknowledge.
- Priority: High

### 8) Add unit/integration tests and QA checklist
- Tests for backend endpoint and minimal frontend smoke test for the bed modal interaction (or manual QA steps if no automated test infra available).
- Priority: Medium

---

## Next steps / Implementation Plan
1. Implement backend severity endpoint and unit tests.
2. Add frontend dropdown + `Update severity status` button and wire to endpoint.
3. Add bed icon UX and ensure resize/center alignment.
4. Implement equipment API or use existing bed/equipment API; add UI display and polling if required.
5. Add equipment usage counters and expose them via API.
6. Update acknowledge flow to move patient between lists.
7. QA and test, then close tickets.

---

If you'd like, I can start implementing ticket #2 (backend severity endpoint) now and mark it `in-progress` in the todo list. Reply with which ticket to start first or say "Start backend severity endpoint" and I will begin the code changes and tests.