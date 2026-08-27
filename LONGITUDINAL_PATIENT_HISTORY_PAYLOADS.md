# Longitudinal Patient History Payloads

## Purpose

Patient history can grow beyond the token budget available to an inference request. The solution is to treat history as a time-aware retrieval and summarization problem rather than sending every historical encounter to the model.

The system should preserve the complete clinical record in storage while sending the model only:

- the current encounter in detail
- compact recent history
- deterministic clinical trends
- the most relevant historical events

## Core Design

```text
Structured clinical records
    -> deterministic TEWS/SATS calculation
    -> trend aggregation
    -> relevant-history retrieval
    -> token-budgeted prompt builder
    -> inference
```

The model should interpret and summarize prepared clinical facts. It should not be responsible for recalculating years of raw encounters.

## Structured History Events

Each encounter should retain structured fields that can be filtered, ranked, and summarized:

- encounter date and facility context
- diagnosis and severity
- vital signs
- TEWS/SATS score and component scores
- SATS priority and nurse override
- mobility state and mobility score
- trauma presence and trauma modifier
- treatments and procedures
- medications and allergies
- clinical notes and presenting complaint
- disposition and outcome

Free-text notes remain valuable, but they should be stored alongside structured fields rather than used as the only source of truth.

## History Layers

### 1. Current Encounter

Send the current encounter with full detail, including:

- current symptoms and complaint
- current vital signs
- AVPU, mobility, and trauma status
- TEWS component breakdown
- TEWS total and derived priority
- nurse-selected SATS override, if present

### 2. Recent Window

Include full or near-full detail for the most recent 24 to 72 hours, depending on the workflow. This captures active deterioration, repeated observations, and recent treatment decisions.

### 3. Trend Window

Compress the previous 30 to 90 days into calculated trends:

- encounter count
- latest, highest, and average TEWS
- number of RED and ORANGE cases
- vital-sign deterioration or improvement
- mobility changes
- recurrent trauma presentations
- repeated diagnoses and interventions
- recent admissions, discharges, and escalations

### 4. Long-Term Summary

Keep stable facts and clinically important history across the full record:

- chronic conditions
- allergies
- active and recurring medications
- prior major procedures
- prior ICU or emergency admissions
- persistent mobility limitations
- significant trauma history

### 5. Relevant Historical Events

Retrieve only events related to the current complaint or risk profile. Useful retrieval signals include:

- matching symptoms or diagnosis
- prior trauma
- abnormal vital signs
- previous RED or ORANGE triage
- matching medication or procedure
- mobility deterioration
- similar disposition or outcome

## Deterministic TEWS and SATS Processing

TEWS calculations should happen before inference. Persist both the components and the result:

```json
{
  "components": {
    "temperature": 0,
    "heartRate": 3,
    "respiratoryRate": 1,
    "systolicBP": 1,
    "avpu": 2,
    "mobility": 2,
    "trauma_modifier": 1
  },
  "total": 9,
  "priority": "red"
}
```

The system must keep these concepts separate:

- **TEWS total:** numeric points
- **SATS priority:** green, yellow, orange, or red
- **mobility score:** the mobility component
- **trauma modifier:** an additional `0` or `1`
- **nurse override:** an intentional SATS priority selection

Historical data should expose at least:

```text
Latest TEWS: 9, RED
Highest historical TEWS: 11, RED
Average TEWS across 12 encounters: 5.8
Trend: worsening
Mobility trend: independent walking -> assisted -> immobile
```

## Token Budgeting

The prompt builder should enforce a hard token budget before inference. A practical allocation is:

| Section | Approximate budget |
|---|---:|
| Current encounter | 35% |
| Recent history | 25% |
| Deterministic trends | 20% |
| Relevant historical events | 15% |
| Instructions and output schema | 5% |

The exact percentages can change by workflow, but the total must always be bounded.

When the budget is exceeded, reduce data in this order:

1. remove duplicate notes and repeated measurements
2. replace repeated encounters with summaries
3. retain abnormal values and red flags
4. retain the first occurrence of chronic conditions
5. retain the most recent and most severe event for repeated patterns
6. never remove allergies, active medications, critical diagnoses, or prior life-threatening events

## Example Inference Payload

```json
{
  "patient": {
    "id": "UNKNOWN_MALE_01",
    "age": 30,
    "sex": "Male",
    "allergies": [],
    "active_medications": []
  },
  "current_encounter": {
    "vitals": {
      "temperature": 35.6,
      "heart_rate": 132,
      "respiratory_rate": 28,
      "systolic_bp": 72,
      "avpu": "P",
      "mobility": "Immobile",
      "trauma_present": true
    },
    "tews": {
      "components": {
        "temperature": 0,
        "heart_rate": 3,
        "respiratory_rate": 1,
        "systolic_bp": 1,
        "avpu": 2,
        "mobility": 2,
        "trauma_modifier": 1
      },
      "total": 9,
      "priority": "red"
    }
  },
  "historical_summary": {
    "encounters_count": 12,
    "first_seen": "2021-04-10",
    "last_seen": "2026-08-25",
    "previous_red_cases": 2,
    "previous_trauma_cases": 3,
    "tews_trend": "increasing",
    "mobility_trend": "declined from independent walking to immobile",
    "recurrent_conditions": ["hypertension"],
    "recent_abnormalities": [
      "Repeated systolic blood pressure below 90 mmHg"
    ]
  },
  "relevant_history": [
    {
      "date": "2025-11-18",
      "reason": "motor vehicle trauma",
      "tews_total": 8,
      "priority": "red",
      "mobility": "Immobile",
      "outcome": "ICU admission"
    }
  ]
}
```

## Retrieval and Ranking

Relevant events should be ranked using a combination of:

- recency
- diagnosis or symptom similarity
- severity and priority
- abnormal vital overlap
- trauma or mobility relevance
- disposition significance

A simple ranking model can use weighted scores:

```text
relevance = recency + complaint_match + diagnosis_match + severity + risk_signal
```

The ranking should be deterministic and explainable. The prompt builder should include the reason an event was selected when practical.

## API and Persistence Guidance

The backend should expose separate concepts rather than one oversized history endpoint:

- `GET /patients/:id/history/summary`
- `GET /patients/:id/history/recent`
- `GET /patients/:id/history/relevant`
- `GET /patients/:id/history/trends`

The complete record remains available for authorized EHR views. Inference endpoints should call a token-budgeted history service that returns a bounded payload.

Every generated summary should include:

- source encounter IDs
- source dates
- summary generation timestamp
- model or algorithm version
- whether events were omitted due to token limits

This preserves clinical traceability and makes summaries auditable.

## Safety and Correctness Rules

- Do not let the model invent missing history.
- Mark unavailable values as unknown rather than zero.
- Keep TEWS points separate from SATS priority labels.
- Preserve nurse overrides as explicit clinical decisions.
- Preserve the raw source encounters for audit and reprocessing.
- Show the date range and number of encounters represented by every summary.
- Treat allergies, active medications, critical diagnoses, and life-threatening events as high-priority facts.
- Apply authorization and hospital tenancy filters before retrieval.

## Recommended Implementation Sequence

1. Normalize every encounter into a consistent history-event shape.
2. Persist mobility, trauma, TEWS components, total, and priority together.
3. Build deterministic trend aggregation in the backend.
4. Add recent and relevant-history retrieval endpoints.
5. Add token counting and a hard prompt budget.
6. Add source references and omission metadata to summaries.
7. Test current-only, short-history, and long-history patients.
8. Verify that the same patient history produces consistent results across manual and AI triage workflows.

## Success Criteria

The design is working when:

- the model never receives an unbounded history payload
- current TEWS values are always complete and authoritative
- mobility and trauma remain available across all triage workflows
- long-term trends remain visible without sending every raw encounter
- selected historical events are explainable and source-linked
- allergies, medications, critical diagnoses, and red flags are never lost during compression
- manual and AI reports agree on the same source data

## Synthetic Test Case: Window Organization

This section shows how the current synthetic patient is divided into the recent and trend windows.

### Reference Time

The simulation uses:

```text
as_of = 2026-08-25T12:00:00.000Z
recent window = 72 hours
trend window = 90 days, excluding the recent window
```

The calculated boundaries are:

```text
recent window: 2026-08-22T12:00:00.000Z through 2026-08-25T12:00:00.000Z
trend window: 2026-05-27T12:00:00.000Z through 2026-08-22T12:00:00.000Z
```

The boundary is exclusive for the trend window and inclusive for the recent window. An encounter exactly at the recent cutoff belongs to the recent window.

### Source Encounter Shape

Each synthetic encounter is a plain object with a stable ID, an ISO timestamp, clinical terms, vitals, mobility, trauma state, TEWS data, and disposition:

```json
{
  "id": "enc-current",
  "date": "2026-08-25T10:00:00.000Z",
  "symptoms": ["trauma", "bleeding"],
  "diagnoses": ["traumatic injury"],
  "interventions": ["hemorrhage control"],
  "vitals": {
    "systolic_bp": 72,
    "hr": 138,
    "resp": 28
  },
  "mobility": "Immobile",
  "traumaPresent": true,
  "tews": {
    "total": 9,
    "priority": "red"
  },
  "disposition": "ICU admission"
}
```

The builder accepts legacy and current field locations. For example, mobility can be read from `encounter.mobility`, `encounter.vitals.mobility`, or `encounter.SATS_reasoning.mobility`. TEWS can be read from `encounter.tews.total`, `SATS_reasoning.total`, `tews_total`, or `tewsTotal`.

### Synthetic Encounter Timeline

| ID | UTC date and time | Age at reference time | Window | TEWS | Priority | Mobility | Trauma | Disposition |
|---|---|---:|---|---:|---|---|---|---|
| `enc-current` | 2026-08-25 10:00 | 2 hours | Recent | 9 | RED | Immobile | Yes | ICU admission |
| `enc-recent` | 2026-08-23 08:00 | 52 hours | Recent | 6 | ORANGE | With help | Yes | Emergency discharge |
| `enc-10d` | 2026-08-15 08:00 | 10 days | Trend | 2 | GREEN | Walks | No | Discharged |
| `enc-30d` | 2026-07-26 08:00 | 30 days | Trend | 3 | YELLOW | Walks | No | Discharged |
| `enc-60d` | 2026-06-26 08:00 | 60 days | Trend | 7 | RED | With help | Yes | Admission |
| `enc-old` | 2026-01-01 08:00 | More than 90 days | Excluded | 1 | GREEN | Walks | No | Discharged |

The current encounter is `enc-current`, because the builder orders valid encounters newest first. It is passed separately as `current_encounter` and is also included in the recent window because it occurred within the last 72 hours.

### Recent Window Output

The recent window contains two full encounters:

```json
{
  "from": "2026-08-22T12:00:00.000Z",
  "to": "2026-08-25T12:00:00.000Z",
  "encounter_count": 2,
  "encounters": [
    { "id": "enc-current", "date": "2026-08-25T10:00:00.000Z" },
    { "id": "enc-recent", "date": "2026-08-23T08:00:00.000Z" }
  ]
}
```

The full objects are retained in this window because recent observations may describe active deterioration, repeated treatment, or a current escalation.

### Trend Window Output

The trend window excludes the recent encounters and contains three encounters:

```json
{
  "from": "2026-05-27T12:00:00.000Z",
  "to": "2026-08-22T12:00:00.000Z",
  "summary": {
    "encounter_count": 3,
    "latest_tews": 2,
    "highest_tews": 7,
    "average_tews": 4,
    "red_cases": 1,
    "orange_cases": 0,
    "mobility_sequence": ["With help", "Walks", "Walks"],
    "mobility_changes": ["With help", "Walks"],
    "trauma_present_count": 1,
    "recurrent_diagnoses": ["hypertension", "traumatic injury"],
    "repeated_interventions": ["medication adjustment", "wound care"],
    "dispositions": ["Discharged", "Admission"]
  }
}
```

Trend mobility is ordered chronologically from oldest to newest. In this case, the patient moved from `With help` at 60 days to `Walks` at 30 and 10 days. The latest trend value is therefore `Walks`, while the current recent encounter is `Immobile`.

### Vital Trends

Vital trends use the oldest and newest available value inside the trend window:

```json
{
  "systolic_bp": { "first": 110, "latest": 128, "change": 18 },
  "heart_rate": { "first": 96, "latest": 82, "change": -14 },
  "respiratory_rate": { "first": 20, "latest": 16, "change": -4 }
}
```

The positive systolic blood-pressure change indicates improvement within the historical trend window. It does not override the current encounter, where the latest recent systolic BP is 72.

### Relevant History

Relevant history is selected only from the trend window. The current encounter's symptoms, diagnoses, and interventions are used as matching terms. Each historical event receives a deterministic relevance score based on:

```text
matching terms * 10 + TEWS total + trauma signal
```

For this patient, `enc-60d` is selected because it matches the current trauma presentation, has a high TEWS score, and includes trauma:

```json
{
  "id": "enc-60d",
  "date": "2026-06-26T08:00:00.000Z",
  "tews_total": 7,
  "priority": "red",
  "mobility": "With help",
  "traumaPresent": true,
  "selection_relevance": 32
}
```

### Final Inference Organization

The model-facing payload has four distinct areas:

```text
current_encounter
    full current patient and clinical context

recent_history
    full encounters from the last 72 hours

historical_summary
    compressed 90-day trend statistics excluding the recent window

relevant_history
    ranked historical events selected from the trend window
```

This organization prevents the model from receiving six unstructured encounters. It receives the current case in detail, recent events with detail, long-term behavior as calculated facts, and only the historical events most related to the current presentation.

## Backlog: Validate Longitudinal History With LLM Inference

The first synthetic POST verified that longitudinal history can reach `ESI_Engine.py`, but it also exposed separate inference reliability work. This should be completed before integrating longitudinal history into the production triage workflow.

### Observed Test Result

- The request reached `POST /llm-triage` successfully.
- The history context was extracted and merged into the existing `medical_history` variable.
- The context size was approximately 591 characters.
- The model was running in CPU GGUF / LLM-only mode with a 1,900-token low-memory context configuration.
- GGUF generation returned instruction text instead of a JSON object.
- JSON parsing failed because the response contained no opening `{`.
- The endpoint correctly entered its deterministic fallback path.
- The fallback returned ESI 2, then the later mapping produced SATS YELLOW.

### Issues To Resolve

1. Restart the Python service after TEWS code changes before comparing results. The captured run used the old scoring behavior:
  - mobility `+3` instead of chart-aligned `+2`
  - AVPU P `+3` instead of `+2`
  - trauma `0` because the old request expected `trauma` rather than `traumaPresent`
2. Add a direct unit test for `tews_from_vitals()` using the synthetic trauma case.
3. Add a request-contract test proving `mobility`, `traumaPresent`, and the serialized history reach the Flask handler.
4. Improve GGUF JSON reliability with strict response-format prompting, bounded output tokens, and a repair/validation path that never treats arbitrary instruction text as a clinical report.
5. Ensure deterministic fallback priority is based on the TEWS/SATS safety result when the model response is invalid.
6. Keep the deterministic TEWS result separate from the model-generated ESI result in the response.
7. Return explicit metadata such as `history_context_used`, `history_context_characters`, `model_json_valid`, and `fallback_reason`.

### Acceptance Criteria

- A fresh synthetic POST reports the chart-aligned TEWS components and total.
- The response includes the bounded recent and trend history metadata.
- Invalid GGUF output is marked as invalid JSON and cannot overwrite deterministic safety scoring.
- The fallback response remains clinically traceable and includes the exact TEWS components.
- Manual and AI reports display the same mobility, trauma modifier, TEWS total, and SATS priority.
