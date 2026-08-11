# Discharge Report Workflow

This document captures the low-hanging fruit needed to finish discharge reporting before completing the surgery event capture workflow.

## Why this is first

- The patient has already been discharged in the workflow.
- Discharge is the end state of the encounter and should produce a persistent summary, just like lab reports and medication prescriptions.
- Capturing discharge reports now closes the loop on the encounter lifecycle and makes the application feel complete.
- Surgery event recording is the next step after discharge reporting is in place.

## Goal

Implement a discharge report system that behaves like existing clinical reports:
- lab reports
- medication / prescription reports
- clinical encounter summaries

The system should generate, store, and surface discharge reports for discharged patients.

## Key Requirements

1. Discharge report should be created when an encounter reaches `DISCHARGED`.
2. The report should include:
   - patient identifiers
   - discharge date/time
   - diagnosis and admission summary
   - final medications and prescriptions
   - procedures performed (including surgery when available)
   - follow-up instructions
   - clinician author and signoff
3. Discharge reports should be queryable by patient, encounter, and discharge date.
4. Discharge reports should be displayed in the same report UI family as lab reports and prescription reports.
5. Discharge report storage should be auditable and update-safe.

## Suggested Data Model

Add or extend a discharge report object with fields such as:
- `reportId: String`
- `patientId: String`
- `encounterId: String`
- `dischargeDate: DateTime`
- `status: 'draft' | 'finalized' | 'amended'`
- `primaryDiagnosis: String`
- `procedures: Array<{ procedureName: String, performedAt: DateTime, durationMinutes: Number }>`
- `medications: Array<{ name: String, dose: String, route: String, frequency: String, prescribingClinician: String }>`
- `followUpInstructions: String`
- `notes: String`
- `authorId: String`
- `signedOffBy: String`
- `createdAt: DateTime`
- `updatedAt: DateTime`

## Backend API

Create discharge report APIs analogous to lab and prescription reports:
- `GET /api/reports/discharge?patientId=:patientId`
- `GET /api/reports/discharge/:reportId`
- `POST /api/reports/discharge`
- `PATCH /api/reports/discharge/:reportId`
- `DELETE /api/reports/discharge/:reportId` (optional, if allowed)

The endpoints should enforce auth and role-based permissions.

## UX / Frontend

Add UI surfaces similar to existing report views:
- Discharge Report list panel
- Discharge Report detail page
- Discharge report card in the patient encounter summary
- “Generate discharge report” action available after discharge
- “View discharge report” in patient history and bed modal

Labels and behavior should mirror lab and prescription reports:
- `Discharge Reports Awaiting Review`
- `Discharge Reports Completed`
- `Prescription Report`
- `Lab Report`

## Low-Hanging Implementation Plan

1. Add discharge report schema and model.
2. Implement backend routes and controller methods for discharge reports.
3. Create the frontend report list/detail UI.
4. Add a discharge report creation flow when patient is discharged.
5. Wire discharge report links into patient summary and clinician dashboards.
6. Add validation tests for report creation and retrieval.

## Next Steps After This

- Finish surgery event capturing using the surgery event data-entry workflow.
- Link surgery procedure records into discharge reports.
- Complete the working application by validating end-to-end patient lifecycle: triage → labs → meds → surgery → discharge.

## Acceptance Criteria

- A discharged encounter can produce a discharge report.
- The report is stored and retrievable.
- The discharge report UI exists alongside lab and prescription report workflows.
- The application can finish the encounter lifecycle and move to surgery event capture next.
