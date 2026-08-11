# Nurse Triage EHR Search and Header Sprint

This sprint captures the next feature work needed to perfect nurse triage, complete the EHR quick search path, and mirror the physician dashboard header UI in the nurse dashboard.

## Sprint Goals

- Enable nurses to quickly pull a patient record by typing the patient name during the triage stage.
- Implement a full electronic health record lookup capability for triage nurses.

- Use a calendar-style patient iteration pattern similar to `Inpatient.jsx` for assigned patient navigation page where its a histoy of assigned patients to specialists.

## Feature Description

### 1. Triage EHR Quick Search

A triage nurse should be able to type a patient name and immediately pull the latest encounter record.

- Search field in the triage stage should support partial name matching.
- Search should return the current encounter, active discharge status, bed assignment, and assignment history.
- Search results should link directly to the patient’s EHR summary so the nurse can confirm triage details and handoff status.
- This quick search should be available from the nurse triage landing page and the nurse dashboard header area.
- This function should be in this page: http://localhost:5173/summary/new
- lets compleletly revamp and redesign and renovate this point of entry for triaging patients. This page structure is crucial. We need to make this experience from here the best in the world as this is from here is our competitive advantage. Itt nneds to be SAPHRA clinical trial ready and iso cert ready.
lets firstly make sure where the patient infomation is, we can quickly search for the patient in our EHR meaning they have finished the simulation life cycle
we already have a overall app theme. look at nursedashboard.jsx, physiciandashboard.jsx, look at the navigations like history actions.jsx, ordered meds.jsx, Staffteam.jsx You can see the consistent them designs patterns. that is 10/10 design ui experience and layout . we have to be great.
### 2. Nurse Dashboard Header Parity

The nurse dashboard header must match `PhysicianDashboard.jsx` exactly in layout and style, with nurse-specific content.

#### Header content

- Staff name and role area should use the same font scale and spacing as physician header.
- Shift clock and current timestamp widget should be identical.
- Button row should use the same rounded-pill button style and hover states.
- Buttons should be configured as follows:
  - `Calendar Events` → `Assigned Patients`
  - `History Actions` (unchanged if the physician header includes it)
  - `Lab Requests` → `Available Specialists`
  - `Ordered Meds` → `Bed Allocation`
  - `Scheduled Surgeries` → `Assigned Patients for Today`

#### Button actions

- `Assigned Patients` should route to the assigned patient list or the new assigned patient page.
- `Available Specialists` should route to the specialist availability page.
- `Bed Allocation` should route to the bed allocation / ward assignment page.
- `Assigned Patients for Today` should open the clinician assignment calendar view.

### 3. Assigned Patient Calendar Function

- Use a calendar-like iteration pattern for assigned patients.
- The UI should allow nurses to flip through assigned patients and see assignment timestamp data.
- The timestamp should be rendered as a millisecond-derived schedule marker, similar to how `Inpatient.jsx` renders event dates.
- This view should expose the patient schedule for the day and allow fast switching between patients.

## Implementation Tasks

1. Review `frontend/src/components/dashboards/PhysicianDashboard.jsx` header block and copy the same structure into `NurseDashboard.jsx`.
2. Add nurse-specific labels and routes in the header button array.
3. Ensure the header component is rendered in `NurseDashboard.jsx`, not `SidebarLayout.jsx`.
4. Add a triage-stage patient search input field with backend search support.
5. Implement a nurse EHR lookup endpoint if not already present.
6. Build a calendar/assigned-patient iterator component or reuse `Inpatient.jsx` logic for the assigned patients list.
7. Update the nurse dashboard styling so the header and buttons are visually identical to the physician header.
8. Test the feature with a nurse user and confirm that patient lookup and header navigation are smooth.

## Acceptance Criteria

- The nurse dashboard header matches the physician header in structure and visual style.
- Nurse header buttons are labeled as specified and navigate to the correct pages.
- Triage nurses can search by patient name and pull live EHR records.
- Assigned patients are displayed in a calendar-like navigation view with timestamp markers.
- The header remains in `NurseDashboard.jsx` and does not move into sidebar layout.
- The nurse triage experience is polished and aligned with the physician dashboard design system.

## Notes

- The core foundation already exists; this sprint is primarily a UI/UX parity and data lookup completion step.
- This work should be completed before adding page navigation and lower-level workflow details.
- Once the header and EHR search are in place, the next sprint can connect the assigned patient flow to specialist handoffs and discharge tests.
