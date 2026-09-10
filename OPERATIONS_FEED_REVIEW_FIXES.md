# Operations Feed Review and Corrective Decisions

Status: Corrective changes implemented for the first operations-feed slice.

## Findings

1. The feed sorted notifications and surgery events independently by creation time, then concatenated them. This violated the contract requiring deterministic severity/priority-first ordering.
2. Surgery events were capped independently, allowing the combined alerts array to exceed the requested limit.
3. Surgery records were queried for the whole hospital and returned to every clinician, violating workflow visibility boundaries.
4. Surgical urgency was incorrectly placed in `priority.label`, which belongs to the SATS priority taxonomy.
5. The notification query is recipient-specific, while role-level alerts require recipient fan-out or a separate role-broadcast mechanism.

## Decisions

### Combined ordering

All alert events are normalized first, then sorted together by:

1. Severity rank: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`.
2. Priority wait time, when present.
3. Event creation time, newest first.
4. Stable event ID as the final tie-breaker.

The merged array is truncated once, after sorting, so `limit` is a hard response limit.

The queue patient array remains exempt from this feed alert sort: it must be passed through in the exact order returned by the authoritative backend queue engine.

### Surgery visibility

A surgery event is visible to:

- An administrator or manager with hospital-wide operational permission.
- A clinician whose ID is present in the encounter's assigned clinicians or assigned physician/nurse fields.
- A clinician whose authenticated name or email is explicitly present in the surgery surgeon or surgical-team fields.

A general hospital clinician does not receive every surgery in the hospital. The assignment matching logic is isolated so it can be replaced with the canonical theatre-team/assignment model when that model is finalized.

### Surgery priority separation

Surgery events set `priority: null`. Surgical urgency is stored in:

```js
metadata: {
  surgicalPriority: 'URGENT' | 'HIGH' | 'MEDIUM' | 'LOW',
  surgeryType: 'ELECTIVE' | 'URGENT' | 'EMERGENCY'
}
```

`URGENT_SURGERY` is an event type and severity signal, not a SATS queue category. The UI must never render it as RED/ORANGE/YELLOW/GREEN.

### Role broadcasts

Recipient-specific events continue to use `clinician_id`. Role-level events require one of these explicit mechanisms before production:

- Fan out one persisted event per eligible charge nurse/shift lead recipient.
- Add a separate role-targeted event with `target_role` and enforce hospital plus role filtering in the feed.

The initial implementation uses recipient fan-out as the preferred audit-friendly approach. A role broadcast must not be represented by omitting `clinician_id` from the existing required-recipient schema.

## Verification requirements

- A critical notification created earlier sorts above a high/info surgery event created later.
- The returned alerts array never exceeds `limit`.
- An unrelated clinician cannot receive another clinician's surgery event.
- A related clinician receives the surgery event.
- Surgery `priority` is null and surgical urgency is only in metadata.
- Role broadcasts reach every eligible recipient exactly once.
- Queue ordering is tested separately against the backend queue engine output and is never replaced by the alert sort.
