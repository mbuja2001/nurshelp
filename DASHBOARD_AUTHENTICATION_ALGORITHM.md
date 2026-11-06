# Dashboard Authentication and Triage Access Algorithm

## Scope

This describes the current staff-dashboard flow for deciding whether to show a triage nurse the clock-in board or the clinical dashboard. It is not yet a route-wide authorization design: most application routes are not gated by this `Dashboard.jsx` / `NurseDashboard.jsx` flow.

## State and decision sequence

1. **Restore the browser session.** `Dashboard.jsx` reads the token from `localStorage` once when it mounts. If it can decode an ID from the token payload, it restores the dashboard user and view without a full-page reload. Decoding is only client-side state restoration; it does not prove the token is authentic or unexpired.
2. **Ask the server for the assigned clinical duty.** `NurseDashboard` requests the current day's assigned shift and resolves the staff member's explicit duty from that shift. Nurse/physician duties are not inferred from a general JWT role when no explicit assignment exists. Certain specialist roles retain their role-based fallback.
3. **Read persisted attendance.** For triage access, the clock-in board loads the current attendance record from the staff API and reports the record to `NurseDashboard`. The parent stores the record and re-evaluates the access gate.
4. **Choose the triage view.**
   - While assigned-duty resolution is pending, show a loading state.
   - If the user is a triage nurse and the verified attendance status is `CLOCKED_IN` or `ON_BREAK`, render the clinical dashboard.
   - If the user is a triage nurse and attendance is not clocked in, render the clock-in / schedule / history board. Keep patient queues and clinical actions unavailable.
   - Other assigned roles continue to their role-specific dashboard path; this triage clock-in gate is not a substitute for server authorization.
5. **Handle clock-in as a server decision.** Selecting an assigned shift enables the explicit clock-in action. The server may reject it (for example, HTTP 409 when the administrator has not started the shift). A successful response is followed by an attendance refresh; the updated persisted record flows to the parent and unlocks the clinical dashboard in place.
6. **Handle protected API failures without reloading.** `apiFetch` clears the local session and dispatches `nursehelpdesk:logout` for a 401 on protected endpoints. The dashboard listens for that event and switches to login in-app. Public login/registration endpoints are excluded, so a rejected login does not discard another session. This helper is not yet used by every direct `fetch` call in the application.

## Decision sketch

```text
token present?
  no  -> show login
  yes -> decode only to restore local UI -> fetch assigned duty + persisted attendance

assigned duty still loading?
  yes -> show loading
  no  -> triage nurse?
           no  -> use role-specific dashboard routing
           yes -> persisted status CLOCKED_IN or ON_BREAK?
                    yes -> render clinical dashboard
                    no  -> render clock-in/access board

clock-in selected?
  server rejects (e.g. 409) -> remain on access board and show the error
  server accepts -> refresh attendance -> unlock clinical dashboard without navigation/reload
```

## Acceptance evidence from the shared-browser journey

- A triage nurse without a currently eligible, started shift was restricted to the access board. Attempting to clock into the not-yet-started shift returned HTTP 409 and displayed that administrator start was required.
- After the administrator explicitly started the October 16 ED coverage shift, the assigned nurse signed in, selected the shift, and clocked in. The access board changed to the triage clinical dashboard in place.
- A deliberate browser refresh restored the nurse session and the clinical dashboard remained available while the attendance record was still active. Later, the expired overnight shift was shown as completed and did not unlock clinical work.

These are browser acceptance observations, not proof of every API, route, or database authorization boundary.

## Known limitations and next security work

- Client-side JWT decoding is not signature verification. Only the server may authorize protected data and actions.
- The explicit attendance gate currently protects the triage dashboard path, not every `App.jsx` route. Other clinical pages must not be considered authorized merely because a dashboard is hidden.
- Several components still make direct `fetch` calls rather than using `apiFetch`; their session-expiry behavior must be reviewed before extending the route gate.
- The operations feed and several dashboard views poll in the background. Polling refreshes data and must not be implemented by reloading the page.
- Shift creation, administrator approval/start, assignment, and staff clock-in are separate persisted events. Scheduling or starting must never imply a staff clock-in.
- Before generalizing this gate, verify backend middleware on every sensitive endpoint, hospital/staff identity scope, explicit role/duty assignment, attendance lifecycle, and audit actor/resource data. Then add role-by-role route tests and stale/expired-token cases.

## QA continuation

The current next step is one synthetic manual intake on the already clocked-in triage nurse session, followed by verifying its urgency/queue notification and persistence. Continue the same patient and encounter into clinician acknowledgment and the roadmap's diagnosis/statistics lifecycle. Do not seed history or reuse an existing encounter.
