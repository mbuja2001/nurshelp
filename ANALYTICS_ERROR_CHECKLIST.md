# Analytics Error Checklist and Source-of-Truth Review

This checklist captures the current production-readiness concerns for the live hospital analytics flow. It is intended to be a working checklist before touching code again.

## 1) Current problem statement

The analytics foundation exists, but the system is still vulnerable to the following issues:

- live encounter counts are visible in one area, while other metrics incorrectly show insufficient data
- overview statistics and analytics page values are not using the same source of truth
- the diagnosis breakdown can display N/A even when seeded encounters exist with diagnosis data
- charts can refresh without animating consistently from zero to the real value
- some metrics appear to be built from shift-based logic instead of live encounter reality
- there is still risk that the app presents mock or synthetic defaults instead of real hospital activity

## 1A) Live browser verification findings from the shared /analytics page

This is the first real acceptance test we have from the shared browser session, and it already reveals a production problem independent of the dashboard rendering.

### Observed browser evidence

- the app is live on http://localhost:5173/analytics
- repeated console failures are showing 401 authorization responses for the operations feed and consult data
- the repeated messages are:
  - [NotificationService] getOperationsFeed error: Error: Error: 401
  - Failed to fetch on-call consults: 401
- because the sidebar and notifications are polling with expired or missing auth, the page is not in a clean production state even before the analytics chart logic is evaluated

### What this means

The analytics page cannot be declared production-ready while the app is failing authentication in background feed calls. This is a front-end health issue and a data-flow integrity issue, and it must be treated as a blocker for any browser-based acceptance workflow.

### Browser rule

Do not judge analytics correctness until the page is authenticated and its background data sources are not throwing 401s. Once those are resolved, the true dashboard mismatch checks below become valid.

## 2) How analytics receive data today

The live analytics flow should be:

1. hospital seed / live encounter creation produces real patient records
2. backend analytics query reads encounter records in a date range
3. derived metrics are calculated from encounter data, not from static UI defaults
4. confidence gating is applied only after a real sample size is established
5. the rendered dashboard reads the same final payload across overview, diagnostics, and analytics pages

The current concern is that the pipeline is not consistently aligned across components, which means different screens can each derive a different truth from the same hospital state.

## 3) How analytics are processed

The analytics pipeline must be designed as follows:

- Source collection: Encounter documents, AuditLog records, ClinicalStaff data
- Time window: range based on date selection (today / week / month)
- Aggregation: total patients, priority buckets, wait times, diagnosis counts, case reviews
- Confidence evaluation: only publish values when sample size meets threshold; otherwise show a clear insufficient-data state
- Trend comparison: compare current vs previous window using the same aggregation logic
- UI rendering: all cards, charts, diagnosis panels, and overview widgets consume the same payload

The current issue appears to be that some metrics are built from a shifted interpretation of the hospital data model, and some panels are using alternative / partial data sources rather than the canonical encounter-derived results.

## 4) Production-readiness concerns to fix

### A. Single source of truth

- [ ] Overview tab and analytics page must read the same canonical hospital analytics payload
- [ ] No component should derive summary numbers from a different logic branch unless explicitly intended
- [ ] All pages should agree on the same patient count, diagnosis distribution, and wait-time values

### B. Live patient and encounter validation

- [ ] Total patients should be derived from actual encounter records in the selected window
- [ ] If there are seeded encounters, the app must not present insufficient data unless the chosen metric truly fails the confidence threshold
- [ ] The app must show honest empty states only when there are no matching live records for the selected range

### C. Diagnosis / breakdown integrity

- [ ] Diagnosis distribution must aggregate real diagnosis fields from seeded encounter data
- [ ] If diagnosis data is null, the app should show a clear empty state rather than a misleading N/A or default card
- [ ] Admin overview and analytics page must show the same diagnosis mix for the same hospital and date range

### D. Trend and comparison correctness

- [ ] Current vs previous period must be computed from the same range logic
- [ ] Growth and decline should reflect actual difference between current and prior windows
- [ ] Labels like improvement / worsening must not be based on static or stale values

### E. Chart animation and refresh behavior

- [ ] All charts must animate from 0 to the actual value on initial render and on refresh
- [ ] Refresh behavior should not preserve stale intermediate values or silently show a static snapshot
- [ ] The animation must depend on the live numeric payload, not on mock/default placeholders

### F. Live feed persistence and acceptance testing

- [ ] Seeded encounter state must persist into the live hospital store and remain visible across refresh
- [ ] The analytics page must continue to reflect the same data after reload
- [ ] Browser acceptance tests should verify that the analytics page and overview page remain consistent after refresh

## 5) Specific mismatches to validate in browser

- [ ] Patient count in overview matches analytics page
- [ ] Diagnosis breakdown matches between overview and analytics page
- [ ] Insufficient data appears only when sample size is actually below threshold
- [ ] Wait time, throughput, and staffing metrics reflect the same encounter set
- [ ] Charts animate on initial load and after refresh
- [ ] Dashboard behavior is not using a stale seeded default when live records exist

## 6) Acceptance criteria before calling this production-ready

The analytics layer is ready when all of the following are true:

- the analytics page shows real encounter totals when hospital encounters exist
- other metric cards no longer show insufficient data for valid live patient sets
- overview and detail analytics panels display the same numbers for the same time window
- diagnosis chart shows real seeded diagnoses instead of N/A / empty state when the data exists
- refreshes do not regress to stale or default values
- charts animate from zero to the measured value rather than showing a static fetch result
- the system reports only actual hospital event reality, not mock defaults, when live data is present

## 7) Immediate next step

Before code changes, the next action is to align the canonical analytics source and confirm which page is the truth source. Once that is locked, each metric, chart, and overview card should be wired to the same live payload and re-tested in browser persistence loops.

## 8) Recursive browser verification loop (required)

This is the acceptance stage we should use before any patching or refactor effort:

1. Open the shared browser on the live analytics page.
2. Confirm the page is authenticated and the background feed calls are not throwing 401 errors.
3. Seed or confirm the hospital has live encounters.
4. Reload the analytics page and verify the same values remain after refresh.
5. Compare the overview tab against the dedicated analytics page for the same hospital/window.
6. Check each metric card for honest data quality state: real numbers vs clearly labeled insufficient data.
7. Confirm diagnosis distribution shows seeded diagnosis values when records exist.
8. Check charts animate from zero to the actual value after refresh.
9. Export a real snapshot and confirm the payload matches the live hospital state.
10. Only once the browser loop is green should code changes be considered for production readiness.

### Current browser blocker recorded

From the live shared /analytics page, the observed browser log shows repeated 401s:

- [NotificationService] getOperationsFeed error: Error: Error: 401
- Failed to fetch on-call consults: 401

This is a blocker because the page is not in a valid authenticated state. Until that is fixed, the analytics data cannot be trusted as production-ready.

## 9) Notes for the next fix loop

The next round should validate in browser with a seeded hospital, reload the page, confirm that metrics remain consistent, and only then patch code. The rule is: do not change rendering logic until the source-of-truth mismatch is resolved and the 401 blocker is cleared.
