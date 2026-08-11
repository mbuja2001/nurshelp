# Triage Acknowledgment and Discharge Test Plan

This document defines the workflow for testing discharge by re-running an AI triage simulation, generating a triage report, assigning a specialist, and validating the nurse/specialist acknowledgement tables.

## Purpose

- Verify the full handoff workflow from AI triage to clinician acknowledgement.
- Ensure the triage nurse's `awaiting clinician` table and the specialist physician's `acknowledged` table update correctly.
- Confirm the discharge flow only completes once the patient has been properly diagnosed and the care responsibility has passed.
- Fix the current failure mode where handover state transitions were incomplete because the focus was on dashboard relationships rather than the actual clinical workflow.

## Workflow Overview

1. Simulate a mock patient through AI triage.
2. Run inference and produce a triage report.
3. Assign the patient to a specialist physician based on urgency and care level.
4. Deliver the assignment as a notification to the specialist dashboard.
5. Specialist physician acknowledges the patient.
6. Patient moves from the triage nurse’s `awaiting clinician` table to the `acknowledged` table.
7. The acknowledged row holds critical care data while diagnosis and disposition are pending.
8. The patient remains visible until the responsibility is passed to inpatient hospitalization or outpatient discharge.

## Key Concepts

- `awaiting clinician table`: holds patients assigned by triage but not yet acknowledged by the receiving clinician.
- `acknowledged table`: holds patients that have been accepted by the clinician and are actively being managed.
- `critical status data`: care level, saturation score, urgency, hospital admission needs, diagnosis progress.
- `notification`: the real-time event that tells a specialist physician a patient has been assigned and requires acknowledgement.
- `responsibility pass`: the point at which the patient is transferred out of triage/specialist handoff state into hospitalized or outpatient care.

## Data Requirements

The mock patient should include:
- AI triage score and level of care recommendation
- Oxygen saturation (`sats`) and vital signs
- Urgency tags such as `critical`, `emergency`, `surgical`, or `observation`
- Specialist assignment metadata
- Patient encounter reference and triage report id

## Expected Behavior

- After AI triage, the patient appears in the specialist dashboard notification queue.
- The specialist dashboard shows a new pending patient with care level data.
- The triage nurse dashboard shows the patient in `awaiting clinician` until the specialist acknowledges.
- When the specialist acknowledges, the patient moves to the acknowledged table immediately.
- The acknowledged table includes fields such as:
  - `diagnosed` status
  - `hospitalized` or `outpatient` disposition
  - `care level`
  - `sats score`
  - `assigned specialist`
  - `triage nurse`
- The patient should only disappear from both tables once the case is closed by discharge or admission handoff.

## Testing Steps

1. Create a mock patient input for AI triage.
2. Trigger the AI inference path and generate a triage report.
3. Confirm the triage report contains the assigned specialist, sats values, and care urgency.
4. Assert the specialist dashboard receives a notification for the new assigned patient.
5. Assert the triage nurse dashboard adds the patient to the `awaiting clinician` table.
6. Have the specialist click acknowledge.
7. Assert the patient moves to `acknowledged` and is removed from `awaiting clinician`.
8. Verify the acknowledged row contains expected status fields and does not disappear prematurely.
9. Run the discharge or hospitalization handoff path and verify the patient leaves the active tables only after final disposition.

## Failure Modes to Fix

- patient stays stuck in `awaiting clinician` after acknowledgement
- patient disappears from both tables before diagnosis or disposition is complete
- notification arrives but dashboard list does not update
- acknowledged table lacks critical care metadata
- discharge test starts before the patient has been properly acknowledged and diagnosed

## Acceptance Criteria

- AI triage simulation can be run end-to-end for a mock patient.
- Specialist assignment produces a notification and pending patient row.
- `awaiting clinician` → `acknowledged` transition works reliably.
- acknowledged rows contain diagnosis/hospitalization/outpatient fields.
- The patient remains in the workflow until responsibility is explicitly passed.
- Discharge can be tested only after this handoff workflow is functioning.

## Next Actions

1. Implement or verify the notification path from triage assignment to specialist dashboards.
2. Ensure the awaiting/acknowledged tables are driven by the same shared patient handoff state.
3. Add tests for the acknowledgement transition and table state updates.
4. Once stable, run the discharge simulation test using a mock triaged patient.
