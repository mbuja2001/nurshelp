# Staff roles, shift assignments, and dashboard rendering

## Keep these three concepts separate

| Field | Meaning | Example |
|---|---|---|
| `positionCode` / `positionTitle` | The staff member's job, selected from the hospital staff-position catalog | `PHARMACIST` / Pharmacist |
| `systemRole` | The account's broad permission-bearing role | `PHARMACIST` |
| `assignmentRole` | The specific duty the staff member is assigned for a shift | `PHARMACIST` |

Staff onboarding derives the system role and job title from the selected position in `shared/hospitalStaffRoleCatalog.js`. Shift-duty choices must come from the same catalog's allowed duties for that position; a job title or system role should not be replaced with a generic shift duty.

## Scheduling and validation

1. The admin Shifts workspace loads the hospital's staff roster, including position and role details.
2. The scheduler displays the job title and system role and offers only shift duties permitted for that staff member's `positionCode`. For legacy records without a position code, the catalog provides duties based on `systemRole`.
3. `POST /api/shifts/create` and `POST /api/shifts/:shiftID/assign-staff` reload the staff member from the hospital-scoped database record and validate the submitted duty against the same position catalog. The server remains authoritative; a browser cannot assign staff from another hospital, inactive staff, or an incompatible duty.
4. All assigned staff are persisted in the shift's `assignedStaff` roster. Triage nurses, clinical nurses, and physicians are additionally recorded in `teamAssignment` for older clinical workflows that consume those fields.

## Clock-in and dashboard selection

1. `GET /api/shifts/assigned/today` finds a shift for the signed-in staff member. It checks the clinical `teamAssignment` fields first, then falls back to the `assignedStaff` roster.
2. The frontend `DashboardRouter` blocks clinical work queues until the staff member has a shift today and has clocked in. Before then, the staff access tabs provide clock-in, schedule, attendance history, statistics where available, and profile access.
3. After clock-in, `resolveShiftDashboardContext` matches the authenticated user to their `assignedStaff` entry and routes using that entry's `assignmentRole`. This makes the shift duty—not just the broad account role—the primary dashboard-routing signal.
4. Supported contexts include triage nurse, clinical nurse, physician, pharmacy, radiology, pathology, and radiology technician. Technicians require a supported specialty to select the corresponding technician workspace; an unknown technician specialty fails closed.
5. The router renders the corresponding dashboard component: `NurseDashboard`, `ClinicalNurseDashboard`, `PhysicianDashboard`, `PharmacistDashboard`, `RadiologyDashboard`, or `PathologyDashboard`.

The legacy `GET /api/dashboard/config` endpoint builds clinical contexts from `teamAssignment` (triage nurse, clinical nurse, physician). The normal router path uses `/api/shifts/assigned/today` plus the shift roster to support specialty and non-clinical dashboard contexts.

## Source of truth

- Staff position and allowed duties: `shared/hospitalStaffRoleCatalog.js`
- Server-side shift-duty validation: `backend/src/utils/shiftTeamAssignment.js`
- Shift creation and staff assignment APIs: `backend/src/controllers/shift.controller.js`
- Client-side dashboard context resolution: `frontend/src/utils/shiftDashboardContext.js`
- Dashboard loading, clock-in gate, and component selection: `frontend/src/components/dashboards/DashboardRouter.jsx`
