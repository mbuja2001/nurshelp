# Hospital Events Tracker & Analytics Engine — Implementation Plan

Status: This is the core data backbone the analytics dashboard sits on top of. Get this right once, and the confidence-threshold and mistriage-split logic already done becomes trustworthy by construction instead of by manual labeling.

---

## Executive summary

This plan is strong and aligns with the actual architecture problems already discovered in the project: the dashboard needs event-backed metrics, explicit confidence rules, hospital-scoped queries, and a single taxonomy for case review outcomes. I would not change the direction; I would tighten two things before implementation begins:

1. Add a formal MetricDefinition registry so every MetricCard declares exactly which event types it uses and how its denominator is computed.
2. Add event versioning and metric lineage metadata so exports and dashboards are demonstrably derived from the same source-of-truth query path.

The result is a cleaner analytics engine that is easier to trust, easier to validate, and safer to expand into Radiology, Pathology, and Pharmacy reporting.

---

## 1. Single source of truth: one append-only hospital event log

This project already has a canonical event contract from the alert system work. Do not build a second event system for analytics. Extend the existing append-only event log and treat alert events as a subset of hospital events, not a separate parallel stream.

```ts
HospitalEvent {
  id: string
  hospitalId: string
  type: EventType
  timestamp: string
  source: string
  payload: Record<string, any>
  correlationId?: string
  sequenceNumber?: number
  version?: number
}
```

### Design rules
- Immutable append-only record store.
- Corrections are new events pointing to the original event; never an in-place mutation.
- Every analytics metric must trace back to one or more event types in the registry below.
- Every write is filtered to the current hospitalId and re-validated server-side per request.

### Why this matters
The dashboard is currently trying to infer trustworthiness from raw values. Event-backed metrics give us the right foundation to compute denominators honestly, avoid inherited `n` values, and make export and UI consistent by construction.

---

## 2. Event taxonomy and metric lineage

The event log should be extended, not duplicated. The existing alert-category taxonomy is already a good starting point; analytics only needs broader coverage and more structured payloads.

```ts
EventType =
  | 'PATIENT_REGISTERED'
  | 'PATIENT_TRIAGED'
  | 'WAIT_TIME_MILESTONE'
  | 'CASE_ASSIGNED'
  | 'CASE_REVIEWED'
  | 'STAFF_SHIFT_STARTED'
  | 'STAFF_SHIFT_ENDED'
  | 'STAFF_BREAK_STARTED'
  | 'STAFF_BREAK_ENDED'
  | 'LAB_ORDERED'
  | 'LAB_RESULTED'
  | 'PRESCRIPTION_STATUS_CHANGED'
  | 'DISCHARGE_COMPLETED'
  | 'ALERT'
  | 'MESSAGE'
  | 'QUEUE'
```

### Event payload examples

```ts
PATIENT_REGISTERED {
  encounterId: string
}

PATIENT_TRIAGED {
  encounterId: string
  category: 'RED' | 'ORANGE' | 'YELLOW' | 'GREEN'
  tewsScore?: number
  discriminatorFlags?: string[]
}

WAIT_TIME_MILESTONE {
  encounterId: string
  milestone: 'ARRIVAL' | 'TRIAGE_COMPLETE' | 'SEEN_BY_PHYSICIAN' | 'DISPOSITION'
  deltaMinutes?: number
  observedAt: string
}

CASE_ASSIGNED {
  encounterId: string
  staffId: string
  acuityWeight: number
  assignmentType?: string
}

CASE_REVIEWED {
  encounterId: string
  aiSuggestedUrgency?: string
  nurseApprovedUrgency?: string
  physicianFinalUrgency?: string
  aiSuggestedDiagnosis?: string
  nurseApprovedDiagnosis?: string
  physicianFinalDiagnosis?: string
  reviewOutcome?: 'CORRECT' | 'MISTRIAGE_MINOR' | 'MISTRIAGE_MAJOR' | 'MISTRIAGE_CRITICAL' | 'PENDING_CLINICAL_REVIEW'
  reviewerNotes?: string | null
}

STAFF_SHIFT_STARTED / STAFF_SHIFT_ENDED {
  staffId: string
  shiftId?: string
}

STAFF_BREAK_STARTED / STAFF_BREAK_ENDED {
  staffId: string
  breakId?: string
}

LAB_ORDERED / LAB_RESULTED {
  encounterId: string
  department: 'RADIOLOGY' | 'PATHOLOGY'
  orderId?: string
  turnaroundMinutes?: number
}

PRESCRIPTION_STATUS_CHANGED {
  encounterId: string
  status: 'ORDERED' | 'READY' | 'DISPENSED' | 'CANCELLED'
}

DISCHARGE_COMPLETED {
  encounterId: string
  timestamp: string
}
```

### Required rule
Every metric on the dashboard must be explicitly traceable to a defined event-type combination, never an ad hoc query with no event lineage.

---

## 3. Metric registry: the real fix for denominator correctness

This is the key design improvement to prevent the remaining denominator bugs.

```ts
MetricDefinition {
  id: string
  label: string
  unit: '%' | 'min' | 'cases/hr' | 'hrs'
  eventTypesAggregated: EventType[]
  aggregation: 'count' | 'avg' | 'sum' | 'ratio' | 'percent'
  minimumN: number
  formula: string
  outputField: string
  confidenceThresholds: {
    reliable: number
    lowConfidence: number
    insufficient: number
  }
}
```

### Example metric definitions

```ts
const METRIC_DEFINITIONS = {
  totalPatients: {
    id: 'totalPatients',
    label: 'Total Patients',
    unit: 'count',
    eventTypesAggregated: ['PATIENT_REGISTERED'],
    aggregation: 'count',
    minimumN: 1,
  },

  triageAccuracy: {
    id: 'triageAccuracy',
    label: 'Triage Accuracy',
    unit: '%',
    eventTypesAggregated: ['CASE_REVIEWED'],
    aggregation: 'percent',
    minimumN: 30,
    formula: 'correct triage / CASE_REVIEWED count',
  },

  mistriageRate: {
    id: 'mistriageRate',
    label: 'Mistriage Rate',
    unit: '%',
    eventTypesAggregated: ['CASE_REVIEWED'],
    aggregation: 'percent',
    minimumN: 30,
    formula: 'mistriageFlag = true / CASE_REVIEWED count',
  },

  avgWaitTime: {
    id: 'avgWaitTime',
    label: 'Avg Wait Time',
    unit: 'min',
    eventTypesAggregated: ['WAIT_TIME_MILESTONE'],
    aggregation: 'avg',
    minimumN: 1,
    formula: 'average deltaMinutes across WAIT_TIME_MILESTONE records',
  },

  weightedCasesPerHour: {
    id: 'weightedCasesPerHour',
    label: 'Weighted Cases / Hour',
    unit: 'cases/hr',
    eventTypesAggregated: ['CASE_ASSIGNED'],
    aggregation: 'ratio',
    minimumN: 5,
    formula: 'total weighted cases / staff worked hours',
  },
};
```

### Rule for denominator correctness
A MetricCard must always show the actual count of events used in its own query for the selected period. The displayed `n` is computed fresh, never inherited, copied, or assumed from a different metric.

This directly fixes the bug pattern:
- `Avg Wait Time` should reflect the count of `WAIT_TIME_MILESTONE` events for the period, not a random small count from a different metric.
- `Weighted Cases / Hour` should reconcile against the same `CASE_ASSIGNED` population the Workforce Carrying Load table is using.
- `Mistriage Rate` and `Triage Accuracy` should naturally use `CASE_REVIEWED` counts, which are legitimately smaller than patient counts.

---

## 4. MetricCard contract and confidence states

```ts
MetricCard {
  value: number
  unit: '%' | 'min' | 'cases/hr' | 'hrs'
  eventTypesAggregated: EventType[]
  sampleSize: number
  minimumN: number
  confidenceState: 'RELIABLE' | 'LOW_CONFIDENCE' | 'INSUFFICIENT'
  displayLabel: string
}
```

### Rendering behavior
- If `sampleSize < minimumN`, render: `Insufficient data (n=5, minimum n=30)`.
- Always show visible `n` next to the number in the card itself.
- Use distinct styling for low-confidence vs reliable states. Never give the same visual authority to a metric below threshold and one above threshold.

### Data quality rule
If a metric is below threshold, it should be styled as withheld or low confidence, not presented as a headline success metric.

---

## 5. Aggregation approach

For the current team size and evidence volume, compute metrics on demand from the event log with hospital and date-range filters. Do not build materialized views or streaming aggregation yet.

```ts
queryEvents({
  hospitalId: string,
  eventTypes: EventType[],
  startDate: string,
  endDate: string,
  filters?: Record<string, any>
})
```

### Why on-demand aggregation is correct first
- Correctness before optimization is the right move for this project.
- It reduces architectural complexity while getting the metric semantics correct.
- It keeps the event model honest and testable on the same path the dashboard uses.

### Optimization trigger
Only introduce streaming/materialized aggregation when event volume or query latency proves it is a bottleneck.

---

## 6. Shared visualization components

The chart layer should be a reusable component library so every department shares the same visual semantics and logic.

```ts
PieBreakdownChart({ data: { label: string; value: number }[] })
TierBreachChart({ data: { tier: 'Low' | 'Medium' | 'High' | 'Critical'; actual: number; allocated: number }[] })
TimeSeriesChart({ data: { timestamp: string; value: number }[] })
MetricCard({ ...MetricCard })
```

### Reuse requirements
- Diagnosis and alert-type mix should use `PieBreakdownChart`.
- SLA breach charts should use `TierBreachChart`.
- Time-based trends should use `TimeSeriesChart`.
- Every card should use the same `MetricCard` contract and confidence states.

### Department SLA reuse rule
The same `TierBreachChart` should be reused for Radiology, Pathology, and Pharmacy turnaround metrics instead of a bespoke chart per department.

---

## 7. Date-range as a first-class parameter

The Today / Week / Month toggle must change the same date-range input for every metric query and chart, not only for some panels.

```ts
queryEvents({
  hospitalId,
  eventTypes,
  startDate,
  endDate,
})
```

### Implementation requirement
All cards and charts must derive from the same date-range state, and all metric functions must accept a consistent query context. No metric should be using a hardcoded conditional or stale local state.

---

## 8. Export Report: same data path as the dashboard

The export must use the same aggregated metric values as the UI, not a parallel calculation path.

```ts
exportAnalyticsSnapshot({
  hospitalId,
  range,
  metrics: MetricDefinition[]
})
```

### Export contract
- Export calls the same metric functions the UI uses.
- Export output must match what the dashboard is showing in the current session.
- If export values differ from UI values, that is considered a trust bug.

---

## 9. Role gating and hospital scoping

These requirements remain non-negotiable.

- Every event write and every analytics query is hospital-scoped and re-validated server-side per request.
- Case-level `CASE_REVIEWED` detail stays restricted to clinical-lead and QA roles.
- Named staff and diagnosis-chain details never become general admin data without explicit access checks.

### Server-side enforcement
This should live in the route/service layer, not only in the UI.

```ts
if (!userHasRole(req.user, ['CLINICAL_LEAD', 'QA'])) {
  throw new ForbiddenError('Case-level review detail is restricted');
}
```

---

## 10. Backend implementation plan

### 10.1 Event ingestion layer
Create a single hospital event ingestion module that accepts all event types and validates the payload against known schemas.

Planned modules:
- `backend/src/events/eventTypes.ts`
- `backend/src/events/eventSchema.ts`
- `backend/src/events/EventRepository.ts`
- `backend/src/events/appendEvent.ts`
- `backend/src/events/queryEvents.ts`

### 10.2 Metric query layer
Compute metrics from event queries instead of using ad hoc dashboard-local transforms.

Planned modules:
- `backend/src/analytics/metricDefinitions.ts`
- `backend/src/analytics/metricsService.ts`
- `backend/src/analytics/reportBuilder.ts`
- `backend/src/analytics/aggregateMetrics.ts`

### 10.3 Route layer
Add analytics route endpoints that accept hospitalId and date range and return normalized dashboard data.

Planned routes:
- `GET /api/analytics/triage?range=today|week|month`
- `GET /api/analytics/summary?range=...`
- `GET /api/analytics/sla?department=radiology|pathology|pharmacy`
- `POST /api/analytics/export`

### 10.4 Audit and lineage
Each aggregate result should include metadata like:
- `eventTypesAggregated`
- `sampleSize`
- `minimumN`
- `dateRange`
- `generatedAt`
- `hospitalId`

This makes export and UI easier to audit and improves trust with clinical reviewers.

---

## 11. Frontend implementation plan

### 11.1 Shared visualization components
Create a dedicated chart component layer:
- `src/components/visualizations/PieBreakdownChart.jsx`
- `src/components/visualizations/TierBreachChart.jsx`
- `src/components/visualizations/TimeSeriesChart.jsx`
- `src/components/analytics/MetricCard.jsx`

### 11.2 Metric data contract
The dashboard should stop using ad hoc card props and instead render from a uniformly shaped metric object.

```ts
const metric = {
  id: 'triageAccuracy',
  label: 'Triage Accuracy',
  value: 40.0,
  unit: '%',
  eventTypesAggregated: ['CASE_REVIEWED'],
  sampleSize: 5,
  minimumN: 30,
  confidenceState: 'INSUFFICIENT',
};
```

### 11.3 Dashboard data flow
- Range state sits at the top-level dashboard.
- Each panel requests the same normalized analytics payload.
- Cards and charts render from the same metric-service output.
- Export uses the same data model as on-screen content.

---

## 12. Acceptance criteria mapping

This plan satisfies the required decisions already documented:

- [ ] Every MetricCard declares its eventTypesAggregated and a fresh sample size.
- [ ] Avg Wait Time either has a valid capture rate or surfaces the gap explicitly.
- [ ] Weighted Cases / Hour reconciles with Workforce Carrying Load.
- [ ] Mistriage and Triage Accuracy continue to use CASE_REVIEWED counts only.
- [ ] Charts are built from shared visualization primitives, not bespoke implementations.
- [ ] Range changes update all metrics together.
- [ ] Export matches dashboard output exactly.
- [ ] Department SLA metrics reuse the shared tier breach chart.

---

## 13. Suggested implementation sequence

### Phase 1 — event backbone and metric registry
1. Extend canonical event contract with the broader hospital event taxonomy.
2. Create event validation schemas and immutable append-only repository.
3. Define the MetricDefinition registry and tie it to event types.
4. Implement hospital-scoped query APIs.

### Phase 2 — analytics service and dashboard data path
1. Build `aggregateMetrics` for Triage, Wait Time, Workforce Carry, and department SLA metrics.
2. Enforce confidence thresholds and `n` display rules in `MetricCard`.
3. Replace dashboard metric logic with metric registry-driven output.
4. Ensure export uses the same metric service as the UI.

### Phase 3 — visualization library and department reuse
1. Create shared chart components.
2. Migrate diagnosis mix and breach charts to the shared primitives.
3. Reuse `TierBreachChart` for Radiology / Pathology / Pharmacy SLA views.

### Phase 4 — QA and validation
1. Add tests for denominator correctness.
2. Add tests for export-vs-dashboard consistency.
3. Add tests for hospital-scoping and role gating.
4. Run UI-level regression checks for range changes and confidence styling.

---

## 14. Final recommendation

This is already close to the correct architecture. The strongest improvements I would make before coding are:

1. Add a formal metric registry; it reduces drift and makes the project far easier to extend.
2. Track event and metric lineage explicitly; it gives you both trust and future auditability.
3. Keep the event log append-only and hospital-scoped; this is the right foundation for all the reporting and QA work already discussed.

If you implement those three items, this plan becomes the real production backbone for the analytics engine instead of just a dashboard layer on top of partially structured data.
