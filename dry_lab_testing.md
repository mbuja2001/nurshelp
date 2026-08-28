# Patient History Context-Stuffing Stress-Test Plan

## Purpose

Build a scalable, reproducible test workflow for the EHR Library, human-in-the-loop editing, longitudinal patient-history retrieval, timestamp handling, critical-history safety context, and ESI Engine inference.

This plan uses synthetic patient data only. It is intended to validate system behavior before any clinical-trial workflow is considered.

## Success Criteria

- Patient history can be seeded at increasing scale without corrupting existing data.
- EHR Library search and longitudinal retrieval remain correct and performant.
- Critical history is preserved across retrieval, editing, persistence, and inference.
- Timestamp windows classify encounters deterministically at boundaries.
- Critical history is retained even when it falls outside ordinary recency windows.
- Human edits are visible in the next saved and retrieved encounter.
- The ESI Engine receives structured, source-aware history context.
- The model does not invent undocumented critical facts.
- Deterministic ESI/TEWS scoring remains authoritative over model narrative.
- All model outputs and context decisions are auditable.

## Critical Human-in-the-Loop Rules

Context stuffing is governed by a human-in-the-loop architecture. The ESI Engine must never receive raw, algorithmically assembled clinical-history context without a triage nurse review decision.

The required sequence is:

1. Retrieve the patient's longitudinal history from the EHR Library.
2. Build and chunk the history through the timestamp and context-budget algorithm.
3. Generate a clinical snapshot containing the proposed context, source encounters, timestamps, critical facts, omissions, conflicts, and truncation metadata.
4. Present that snapshot to the triage nurse before inference.
5. Require the triage nurse to approve the snapshot unchanged or edit it.
6. Record the nurse's identity, decision, edit reason, decision timestamp, and approved snapshot version.
7. Send only the approved snapshot to the ESI Engine.
8. Preserve the original algorithm-generated snapshot and the nurse-approved snapshot for audit comparison.

The AI request must be blocked when:

- No nurse approval exists.
- The snapshot has changed after approval.
- Critical-history facts are missing, contradictory, or unresolved without an explicit nurse decision.
- The approval is stale relative to a newly retrieved or edited patient history.
- The snapshot exceeds the configured context budget after nurse edits.

The triage nurse must be able to:

- Approve the generated snapshot unchanged.
- Edit, remove, or add context with a recorded reason.
- See every source encounter and timestamp behind the snapshot.
- See critical-history facts separately from ordinary historical context.
- See what was omitted or truncated by the algorithm.
- Reject the snapshot and prevent AI submission.

The AI must not be able to convert an unapproved snapshot into an approved one. Approval is a clinical workflow decision owned by the triage nurse, not an inference result.

## Current Architecture Anchor

The EHR Library currently searches and loads longitudinal records through `Encounter`:

- `GET /api/patients/search`
- `GET /api/patients/ehr/:mrn`

`PatientEHRRecord` is a separate monitoring-oriented projection. Seed data must therefore enter `Encounter` first. The projection should be synchronized only where the monitoring workflow requires it; it should not become a competing history source.

The existing longitudinal context builder is:

- `backend/src/utils/longitudinalPatientHistory.js`

The ESI Engine prompt and inference path are in:

- `backend/ML/ESI_Engine.py`

The existing LLM-only startup path is:

- `START_ESI_LLM_ONLY.sh`

## Persistence Map Before Stress Testing

Persistence must be designed and verified before load testing. A stress test is not valid if the generated context, nurse decision, approved context, or AI payload cannot be reconstructed after the request completes.

### Source-of-Truth Boundaries

The persistence flow has four distinct responsibilities:

| Data | Primary persistence location | Purpose |
| --- | --- | --- |
| Patient encounters and clinical history | `Encounter` | Canonical EHR Library history and longitudinal source records |
| Monitoring projection | `PatientEHRRecord` | Current patient status, monitoring, assignments, and operational views |
| Generated and approved context snapshots | Dedicated snapshot persistence or versioned encounter subdocuments | Human review gate, replay, comparison, and audit |
| Approval and access events | `AuditLog` plus snapshot decision metadata | Who approved or edited what, when, and why |

`Encounter` remains the source of truth for the seeded EHR history. `PatientEHRRecord` must not silently become an alternate longitudinal-history repository.

### Required Snapshot Lifecycle

Every context-stuffing request must move through explicit persisted states:

```text
GENERATED
  -> PREVIEWED
  -> APPROVED_UNCHANGED
  -> APPROVED_EDITED
  -> SUBMITTED
  -> COMPLETED or FAILED
```

The following transitions are also required:

```text
PREVIEWED -> REJECTED
PREVIEWED -> STALE
APPROVED_* -> STALE when source history changes
APPROVED_* -> BLOCKED when the approved payload hash no longer matches
```

No AI request may be sent from `GENERATED`, `PREVIEWED`, `REJECTED`, `STALE`, or `BLOCKED`.

### Minimum Persisted Snapshot Record

The generated snapshot and nurse-approved snapshot must be persisted as separate immutable versions or equivalent before inference.

Each snapshot record must include:

- Snapshot ID and version
- Patient ID, MRN, encounter ID, and hospital ID
- Seed batch ID and synthetic-data marker when applicable
- Source encounter IDs and normalized UTC timestamps
- `as_of` timestamp
- Recent-window, trend-window, critical-history, and relevant-history metadata
- Trend summary statistics and bounded sequences
- Critical-history facts and their source records
- Omitted and truncated content
- Serialized snapshot character and token estimates
- Snapshot content hash
- Parent/generated snapshot ID for edited versions
- Status: generated, approved, edited, rejected, stale, submitted, completed, or failed
- Nurse actor ID, role, decision timestamp, and edit reason
- Approved snapshot hash and approval version
- AI request ID, submission timestamp, response status, and model mode
- Failure or rejection reason where applicable

The raw generated snapshot must never be overwritten by nurse edits. An edited approval must create a new version linked to the generated snapshot so the system can compare algorithm output with the clinical decision.

### Persistence Sequence

The required write and validation sequence is:

1. Read the EHR history from `Encounter` using hospital isolation.
2. Build the timestamp windows and trend summary.
3. Persist the generated snapshot as `GENERATED` with a content hash.
4. Present that exact version to the triage nurse.
5. Persist an approval, edit, or rejection decision with the nurse identity and timestamp.
6. If edited, persist the edited snapshot as a new version and recompute its hash, budget metrics, and source metadata.
7. Revalidate that source history has not changed since snapshot generation.
8. Revalidate that the nurse-approved snapshot is within budget and contains required critical-history metadata.
9. Atomically mark the approved version as ready for submission.
10. Send only the approved snapshot to the ESI Engine.
11. Persist the request ID, exact submitted hash, prompt metrics, model mode, response, and outcome.
12. Mark the snapshot `COMPLETED` or `FAILED` without changing the approved input.

If any validation or persistence step fails, the AI request must not be sent.

### Idempotency and Concurrency Rules

The stress harness and client must verify:

- Repeating the same seed batch does not create duplicate synthetic records.
- Repeating an approval request does not create duplicate approvals.
- A snapshot can be submitted at most once for a given approval version and request key.
- Two nurse decisions cannot silently overwrite one another.
- A stale client cannot approve an older snapshot after a newer history version exists.
- Retry behavior preserves the original approved payload hash.
- Failed AI requests can be retried without mutating the approved snapshot.
- Partial writes are detectable and recoverable.
- Audit records reference the exact snapshot version, not only the patient or encounter.

### Persistence Acceptance Tests

Before scale testing, prove these cases with a script against the real backend:

- Generated snapshot survives a process restart.
- Nurse approval survives a process restart.
- Edited snapshot is distinct from the generated snapshot.
- Rejected snapshots cannot reach the ESI Engine.
- Missing approval blocks submission.
- Changed source history marks an approval stale.
- Changed approved content hash blocks submission.
- Critical history remains present after serialization and reload.
- Trend summary remains bounded after persistence and reload.
- Submitted prompt metrics match the approved snapshot.
- AI response is linked to the exact submitted snapshot hash.
- Failed requests retain the approved input and failure reason.
- Hospital isolation applies to snapshots and audit records as well as encounters.

This persistence phase must pass before running high-volume seed generation or UI stress testing. Otherwise, throughput results could conceal loss of clinical context or an approval bypass.

## Phase 1: Define the Canonical Data Contract

Standardize the shape used by seed data, API responses, prompt context, and audit records.

Required patient fields:

- Patient ID and MRN
- Name, surname, age, gender, and height
- Synthetic-data marker
- Seed batch ID

Required encounter fields:

- Stable encounter ID
- Encounter timestamp in UTC
- Symptoms and chief complaint
- Duration and pain level
- Medical history
- Medications
- Allergies
- Vitals and vital-recorded timestamp
- ESI level and acuity score
- Triage outcome
- Source and audit metadata

Required critical-history fields:

- `criticalHistoryStatus`
  - `reported`
  - `none_established`
  - missing or incomplete for negative-path tests
- `criticalHistory.allergies`
- `criticalHistory.adverseReactions`
- `criticalHistory.medications`
- `criticalHistory.conditions`
- `criticalHistory.implantsOrProcedures`
- Source encounter ID
- Source timestamp
- Confirmation or review metadata where available

The canonical shape must support compatibility with existing snake_case fields while avoiding silent loss during normalization.

## Phase 2: Create the Scalable Synthetic Seed Script

Add a parameterized backend seed script for the EHR Library.

Suggested command shape:

```text
node seeds/seedPatientHistoryStress.js --patients 100 --encounters-per-patient 20 --batch stress-v1
```

The script should support:

- 10, 100, 1,000, and 10,000 synthetic patients
- Configurable encounters per patient
- Configurable hospital ID
- Deterministic seed value
- Idempotent batch insertion
- Batch cleanup without deleting unrelated records
- Optional projection creation for `PatientEHRRecord`
- Summary output with inserted, skipped, and failed counts
- Duplicate detection by batch ID and stable synthetic identifiers

Each patient should receive varied longitudinal encounters covering:

- Recent encounters
- Older trend-window encounters
- Critical history outside the trend window
- Multiple encounters with the same timestamp
- Timestamp boundary cases
- Conflicting historical facts
- Missing critical-history status
- `none_established` status
- Multiple documented allergies and reactions
- High-risk medications
- Major conditions
- Critical implants or procedures
- Normal, abnormal, and deteriorating vital trends
- Multiple ESI levels and acuity scores

All generated records must be clearly marked as synthetic and must never be used as production clinical data.

## Phase 3: Validate EHR Library Retrieval

Exercise the real API path, not a test-only repository.

Search tests:

- Search by MRN
- Search by patient ID
- Search by first name
- Search by surname
- Search with pagination
- Search with limits
- Search with no matches
- Search with special characters
- Search within one hospital while records exist in another hospital

Longitudinal retrieval tests:

- Load all encounters for one synthetic patient
- Confirm newest-first ordering
- Confirm all critical-history fields survive serialization
- Confirm patient identity is consistent across encounters
- Confirm records from another hospital are excluded
- Confirm large histories do not silently disappear because of a default limit

Performance measurements:

- Search latency
- EHR-load latency
- Database query count where observable
- Response size
- Results returned versus requested
- Behavior at 100, 1,000, and 10,000 patient scale

## Phase 4: Test Timestamp and History Windows

Use fixed `asOf` timestamps so every test is reproducible.

Validate `buildHistoryWindows` and `buildInferenceHistoryPayload` for:

- Recent-window inclusion
- Trend-window inclusion
- Critical-history retention outside the trend window
- Exact recent-window boundary
- Exact trend-window boundary
- Same-timestamp encounters
- Invalid timestamps
- Future timestamps
- UTC and timezone-offset inputs
- Missing dates
- Current-encounter exclusion
- Deterministic tie ordering
- Correct encounter and critical-fact counts

Budget tests must confirm:

- Current encounter is retained
- Critical history is retained
- Relevant history is reduced before safety-critical context
- Recent history is reduced after relevant history
- Truncation metadata is accurate
- Estimated characters and tokens are recorded
- No undocumented facts appear during normalization

### Trend Statistics Contract

The trend window may contain 10, 100, or 10,000 encounters, but raw trend encounters must not be copied into the inference payload. The full trend set is used internally to calculate a bounded clinical summary through `buildTrendSummary()`.

The summary must provide:

- **Averages:** average TEWS across encounters with a valid TEWS value.
- **Counts:** total trend encounters, red cases, orange cases, trauma-present encounters, and valid scored encounters where applicable.
- **Vital trends:** first chronological value, latest chronological value, and numeric change for systolic blood pressure, heart rate, and respiratory rate.
- **Sequences:** a bounded chronological mobility sequence and a bounded sequence of mobility changes.
- **Repeated clinical terms:** bounded unique lists for diagnoses, interventions, and dispositions.
- **Coverage metadata:** the total number of encounters processed, even though the raw encounters are not returned.

The output-size guarantee is an item-count guarantee for summary arrays:

- Every sequence and repeated-term list must contain no more than `maxSummaryItems` entries.
- `maxSummaryItems` must default to the documented value of 25 for the production contract.
- Budget profiles may lower the value for smaller prompt budgets.
- Scalar values such as averages and counts do not grow with encounter volume.
- Individual strings must also be length-limited or normalized if a strict serialized-character guarantee is required.

The trend-statistics tests must prove that identical summary structure and bounded array lengths are produced for equivalent datasets containing 10 and 10,000 trend encounters. Tests must also verify that changing input order does not change chronological results.

`latest_tews` must be selected from the chronologically newest eligible encounter with a valid TEWS value. It must never depend on the original database or array order. This is a required correctness condition for timestamp stress testing.

The summary is descriptive context for the ESI Engine. It must not replace deterministic ESI/TEWS scoring or permit the model to silently change the authoritative score.

## Phase 5: Update the ESI Engine Prompt

Update `backend/ML/ESI_Engine.py` to make longitudinal and critical history explicit in the inference contract.

The prompt must instruct the model to:

- Treat critical history as safety context, not optional background
- Distinguish `reported`, `none_established`, and missing or incomplete status
- Separate allergies, severe adverse reactions, medications, conditions, and implants/procedures
- Preserve source encounter IDs and timestamps when citing history
- Prefer newer confirmed facts when records conflict
- Identify unresolved contradictions
- Avoid inventing undocumented history
- Distinguish historical facts from current symptoms
- Explain how relevant critical history affected reasoning
- State when history is insufficient
- Return structured JSON suitable for audit

The deterministic ESI/TEWS result remains authoritative for scoring. The model must explain and synthesize clinical context without silently overriding deterministic rules.

Prompt tests should verify that:

- Critical history is present in the final prompt
- Older critical history is not dropped solely because it is old
- `none_established` is not converted into an invented negative medical history
- Conflicts are visible to the model
- Prompt metadata includes the `as_of` timestamp and context counts
- The model receives the edited value after a human update

## Phase 6: Human-in-the-Loop Persistence Workflow

This phase is divided into a script-first validation path and a final client-UI path. The UI is not the first stress surface; the underlying approval contract must pass automated and script-based tests first.

### 6A. Script-Based Approval and Persistence Test

For each representative seeded patient, use a test script against the real backend services to:

1. Seed or select a synthetic patient with multiple encounters.
2. Search for the patient in the EHR Library.
3. Load the longitudinal patient history.
4. Build the algorithm-generated clinical snapshot.
5. Verify source encounters, timestamps, critical facts, omissions, and budget metadata.
6. Assert that inference is rejected before approval.
7. Approve the snapshot unchanged and verify an approved payload is produced.
8. Edit the snapshot and Critical History data, then approve the edited version.
9. Confirm the approved payload contains the edited status and facts.
10. Confirm the original and edited snapshots remain auditable.
11. Reload the EHR record and confirm the edited value persists.
12. Run the AI-assisted path and confirm only the approved snapshot is included.
13. Verify the model input hash or snapshot version matches the nurse-approved version.

### 6B. Final Client-UI Approval Test

After the script path passes, exercise the production client experience:

1. Search for the patient in the EHR Library.
2. Review the generated clinical snapshot in the triage nurse approval view.
3. Inspect source encounters, timestamps, critical history, conflicts, and omissions.
4. Approve the snapshot unchanged or edit it with a reason.
5. Confirm the UI shows the approved version and decision state.
6. Submit the approved context to the ESI Engine.
7. Confirm the displayed AI result references the approved context version.
8. Reload the workflow and verify the audit trail.

Negative-path tests:

- Submit with missing critical-history status
- Select `none_established`
- Change from `reported` to `none_established`
- Clear one category while keeping status `reported`
- Submit contradictory historical records
- Close the modal without changing data
- Reload during an in-progress save
- Retry after a failed request
- Attempt AI submission without nurse approval
- Modify the snapshot after approval and verify the request is blocked
- Approve, then retrieve newer history and verify approval becomes stale
- Edit critical history without recording an edit reason
- Reject the snapshot and verify no AI request is sent
- Exceed the context budget through nurse edits and verify revalidation

## Phase 7: Start and Stress the ESI Engine

Start the engine through the existing LLM-only path first:

```text
START_ESI_LLM_ONLY.sh
```

Then exercise the real HTTP inference endpoint with controlled synthetic cases. Every request must use a nurse-approved snapshot produced by Phase 6; direct submission of an unapproved generated context is a required rejection test, not a valid inference path.

Run at increasing levels:

- Smoke test: 1 patient, 1 encounter
- Functional test: 10 patients, multiple encounters
- Baseline load: 100 patients
- Scale test: 1,000 patients
- Extended stress test: 10,000 patients where local resources permit

Capture for every request:

- Request ID
- Patient and encounter IDs
- Seed batch ID
- Context character estimate
- Context token estimate
- Number of source encounters
- Number of recent encounters
- Number of relevant encounters
- Number of critical encounters and facts
- Truncation status
- ESI/TEWS deterministic result
- Model response status
- JSON parse status
- Latency
- Timeout or retry status
- Human-edited field indicator

Run both available modes where supported:

- `LLM_ONLY`
- Hybrid or RAG-enabled mode

Compare whether the two modes preserve the same safety-critical history and whether their narrative conclusions differ.

## Phase 8: Evaluate Model Behavior

Create a labeled synthetic evaluation set with expected safety facts and deterministic triage outcomes.

Measure:

- Critical-fact recall
- Critical-fact precision
- Unsupported-fact hallucination rate
- Correct handling of `none_established`
- Correct handling of contradictions
- Correct source and timestamp attribution
- Agreement with deterministic ESI/TEWS scoring
- JSON schema validity
- Missing-field rate
- Timeout rate
- Retry rate
- Latency distribution
- Prompt-budget overflow rate

A model response must fail evaluation when it:

- Omits a known severe allergy relevant to the case
- Treats an old critical fact as expired without explicit policy support
- Invents an allergy, medication, condition, or procedure
- Changes deterministic ESI/TEWS scoring without an explicit human-reviewed override
- Reports an unsupported certainty level
- Loses the relationship between a fact and its source encounter

## Phase 9: Automated Test Suite and Quality Gates

Add focused automated coverage for:

- Seed generation and idempotency
- Encounter/EHR serialization
- EHR search and load APIs
- Hospital isolation
- Timestamp boundary classification
- Invalid and future timestamps
- Critical-history preservation
- Context-budget reduction
- Prompt construction
- Output schema validation
- Manual edit persistence
- AI context inclusion after an edit
- Load-test result thresholds

Recommended quality gates:

- No critical-history loss in retrieval tests
- No current-encounter loss
- No cross-hospital leakage
- No critical-history omission during budget reduction
- Zero unsupported critical-fact hallucinations in the labeled evaluation set
- Deterministic scoring unchanged by prompt changes
- All seed records marked synthetic
- No seed script execution allowed against production configuration

## Phase 10: Clinical-Trial Readiness Controls

Before trial use is considered:

- Enforce synthetic-data safeguards in seed scripts
- Add production-environment refusal checks
- Add batch cleanup and rollback support
- Protect PHI and secrets from logs and fixtures
- Log prompt inputs and model outputs with appropriate access controls
- Preserve deterministic and AI-generated decisions separately
- Require clinician review before operational action
- Define behavior when EHR retrieval fails
- Define behavior when timestamps are invalid
- Define behavior when the model times out or returns invalid JSON
- Document the human override and audit process
- Review the workflow with clinical, privacy, and safety stakeholders

## Execution Order

1. Finalize the canonical data contract.
2. Add deterministic seed generation for a small dataset.
3. Add EHR retrieval and hospital-isolation tests.
4. Add timestamp boundary and context-budget tests.
5. Define the clinical snapshot and nurse-approval contract.
6. Update the ESI Engine prompt.
7. Add prompt, approval, and inference instrumentation.
8. Validate script-based approval, edit, and persistence flow.
9. Run smoke and functional inference tests using approved snapshots.
10. Validate the final client UI approval experience.
11. Scale to 100, 1,000, and 10,000 synthetic patients.
12. Review stress results and close all safety-critical gaps.
13. Complete clinical-trial readiness review.

## Deliverables

- Parameterized synthetic patient-history seed script
- `backend/scripts/seed-patient-history-stress.js` for deterministic batch seeding
- `backend/scripts/test-patient-history-context-payload.js` for preview-first context generation and approved sending
- Canonical patient-history and critical-history contract
- Timestamp and context-window test suite
- EHR retrieval integration tests
- Updated ESI Engine prompt
- Inference stress harness and metrics output
- Human-in-the-loop persistence tests
- Model evaluation report
- Clinical-trial readiness checklist
