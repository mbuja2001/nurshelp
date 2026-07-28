# Clinical Handover - Quick Reference

## System Architecture at a Glance

### Two Parallel Triage Paths

| Aspect | Manual Triage | AI-Assisted Triage |
|--------|---------------|-------------------|
| **Input** | Nurse enters vitals manually | ESI_Engine processes vitals + symptoms |
| **Report** | manualTriageReport only | manualTriageReport + aiTriageReport |
| **Processing** | ~1-2 minutes | ~3-5 seconds (HYBRID mode) |
| **Confidence** | Based on nurse judgment | Confidence score (0-1) |
| **Override?** | N/A (nurse is primary) | Yes - nurse can override AI |
| **Use Case** | Complex cases, clinical intuition needed | Standard presentations, speed needed |

---

## Key Files & Their Roles

### Frontend Components

**PatientReport.jsx** (Dashboard & PatientSummary)
- Entry point for triage result display
- Shows AI inference output as preview
- Allows nurse to confirm or manually edit

**ConfirmingAIClinicalHandover.jsx**
- Handles AI-assisted triage confirmation flow
- Displays side-by-side: Manual vs AI recommendations
- Tracks which clinicians to notify

**ConfirmingManualClinicalHandover.jsx**
- Handles manual-only triage confirmation
- Simpler flow, no AI comparison
- Saves manual report fields before handover

**TriageConfirmationPage.jsx** (Components)
- Final confirmation + override step
- Tracks which fields nurse changed and why
- Creates `nurseEditsReport` for audit trail

### Backend Models

**TriageRequest** (src/models/triageRequest.model.js)
```javascript
{
  manualTriageReport: {...},    // Nurse's assessment
  aiTriageReport: {...},        // AI engine output
  computeMode: 'HYBRID',        // LLM_ONLY | RAG_FULL | HYBRID
  status: 'TRIAGED'
}
```

**Encounter** (src/models/encounter.model.js)
```javascript
{
  manual_triage_report: {
    // Includes override tracking
    field_changes: {
      esiLevel: { original: 2, override: 3 },
      physician: { original: "Dr. A", override: "Dr. B" }
    },
    overridden_by: ObjectId,
    overridden_at: Date
  },
  clinical_handovers: [
    {
      clinician_id, clinician_name, clinician_role,
      handover_status: 'NOTIFIED' | 'VIEWED' | 'ACKNOWLEDGED',
      report_type: 'MANUAL' | 'AI',
      viewed_at, acknowledged_at
    }
  ]
}
```

### Backend Engine

**ESI_Engine.py** (backend/ML/ESI_Engine.py)
- Flask server for triage inference
- Supports: LLM_ONLY, RAG_FULL, HYBRID modes
- Returns: esiLevel, specialty, physician recommendation, confidence

---

## The Handover Process (5 Steps)

### Step 1: Triage Confirmation
```
Nurse reviews triage report
├─ Accept AI? → Uses AI recommendations
└─ Override? → Manual selection + reasons → nurseEditsReport created
```

### Step 2: Clinician Assignment
```
Determine assigned clinicians:
├─ From AI recommendation (if accepted)
└─ From manual override (if nurse changed)
```

### Step 3: Handover Creation
```
POST /api/encounters/{id}/handover
├─ For each clinician:
│  └─ Create clinical_handovers entry
│     └─ Set status: 'NOTIFIED'
│        Set report_type: 'MANUAL' or 'AI'
│
└─ Set encounter.handover_completed = true
```

### Step 4: Notifications Sent
```
WebSocket event to clinicians:
{
  type: 'CLINICAL_HANDOVER',
  encounter_id: "...",
  patient_name: "...",
  esi_level: 3,
  bed_location: "ED-7",
  summary: "Acute chest pain, 52M",
  action_url: "/encounters/ENC-123/handover"
}
```

### Step 5: Clinician Acknowledgment
```
PATCH /api/encounters/{id}/acknowledge
├─ Updates clinical_handover entry:
│  ├─ viewed_at = now
│  ├─ acknowledged_at = now
│  └─ status = 'ACKNOWLEDGED'
│
└─ SLA tracked (typically 30 minutes)
```

---

## Override Tracking (Audit Trail)

### Where Overrides Are Stored

```javascript
// In Encounter.manual_triage_report.field_changes
{
  esiLevel: {
    original: 2,           // What AI said
    override: 3,           // What nurse chose
    reason: "Patient appears clinically stable despite high acuity score"
  },
  recommendedPhysician: {
    original: { name: "Dr. Chen", specialty: "EM" },
    override: { name: "Dr. Wilson", specialty: "Cardiology" },
    reason: "Better specialty match for chest pain presentation"
  }
}

// Metadata
overridden_by: ObjectId("triageNurseId")
overridden_at: ISODate("2026-06-05T13:22:00Z")
```

### Why This Matters

**For AI Improvement**
- Capture nurse's reasoning when they disagree with AI
- Later: specialist validates who was right
- Result: training data to improve ESI boundary detection

**For Audit/Compliance**
- Full audit trail: Nurse → AI → Override (if any) → Specialist
- Shows decision-making rationale
- Required for healthcare accreditation

**For Safety**
- Pattern detection: if many nurses override on "chest pain" → investigate
- Early warning: if override leads to adverse outcome → escalate

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  CLINICAL TRIAGE                        │
└─────────────────────────────────────────────────────────┘

MANUAL ENTRY (PatientSummary.jsx)
↓
┌─────────────────────────────────────────────────────────┐
│  triageNurse enters: vitals, symptoms, medical hx      │
│  TriageRequest.manualTriageData saved                  │
└─────────────────────────────────────────────────────────┘
↓
[AI PROCESSING - Background]
ESI_Engine.py running (if enabled)
↓
┌─────────────────────────────────────────────────────────┐
│  TriageRequest saved with both reports:               │
│  ├─ manualTriageReport (nurse's ESI choice)           │
│  └─ aiTriageReport (AI recommendation + confidence)   │
│                                                        │
│  Status: TRIAGED                                      │
└─────────────────────────────────────────────────────────┘
↓
NURSE REVIEW (PatientReport.jsx → TriageConfirmationPage)
├─ Accept AI recommendation?
│  └─→ Use assigned_clinicians from AI
│
└─ Override AI?
   ├─→ Nurse manually selects clinicians
   ├─→ Records reasons in field_changes
   └─→ Creates nurseEditsReport
↓
┌─────────────────────────────────────────────────────────┐
│  POST /api/encounters/{id}/handover                    │
│  ├─ Trigger clinical handover                         │
│  └─ Create clinical_handovers entries                 │
└─────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│  ENCOUNTER.clinical_handovers array populated:        │
│  [                                                     │
│    {                                                   │
│      clinician_id: "Dr. James Wilson",                 │
│      clinician_role: "PHYSICIAN",                      │
│      handover_status: "NOTIFIED",                      │
│      report_type: "MANUAL",                            │
│      viewed_at: null,       // Until clinician views   │
│      acknowledged_at: null  // Until acknowledged      │
│    }                                                   │
│  ]                                                     │
└─────────────────────────────────────────────────────────┘
↓
REAL-TIME NOTIFICATIONS (WebSocket/Event-driven)
Clinicians receive handover alert
↓
CLINICIAN ACKNOWLEDGMENT
├─ Opens handover link
├─ Views patient summary + triage reasoning
├─ Clicks "Acknowledge"
└─ Status updated: ACKNOWLEDGED
```

---

## Common Workflows

### Workflow 1: AI Accepts → Straight Handover
```
✓ Nurse validates AI ESI level
✓ Nurse accepts recommended clinicians
✓ No override reasoning needed
✓ Uses: assigned_clinicians from AI
→ Clinical handover proceeds
```

### Workflow 2: AI Override → Different Physician
```
✓ AI recommends: Dr. Chen, ESI-2
✗ Nurse overrides: Dr. Wilson, ESI-3
✓ Nurse records: "Patient appears clinically stable"
✓ field_changes tracks the delta
→ Later specialist feedback validates nurse was right/wrong
→ Feeds into model improvement
```

### Workflow 3: Manual Triage (No AI)
```
✓ Nurse enters vitals manually
✗ No AI engine available (LLM_ONLY mode or disabled)
✓ Nurse selects ESI level manually
✓ Manually selects clinicians
✓ No override tracking (nurse is primary)
→ Direct handover with manual report
```

### Workflow 4: Complex Case
```
✓ AI suggests ESI-2 based on vitals
✗ Nurse recognizes clinical deterioration pattern
✓ Nurse overrides to ESI-1 (Emergency)
✓ Records: "Patient expressing angina symptoms, diaphoretic"
✓ Selects specialist (Cardiologist) instead of EM
→ High-priority handover with escalation
→ Override reasoning captured for audit
```

---

## Testing Checklist

### Unit Tests
- [ ] Override tracking captures all field deltas
- [ ] nurseEditsReport includes reasoning
- [ ] clinical_handovers array created per clinician
- [ ] Status transitions: TRIAGED → CONFIRMED → HANDED_OVER

### Integration Tests
- [ ] AI triage → override → handover flow
- [ ] Manual triage → direct handover flow
- [ ] Clinician acknowledgment updates status
- [ ] Specialist feedback links back to original triage

### E2E Tests
- [ ] Patient enters triage
- [ ] Nurse reviews + overrides AI
- [ ] Clinician receives notification
- [ ] Clinician acknowledges
- [ ] Specialist validates accuracy
- [ ] All records properly logged

---

## Performance Considerations

| Operation | Time | Bottleneck |
|-----------|------|-----------|
| Manual triage entry | 2-5 min | Nurse data entry |
| AI inference (HYBRID) | 3-5 sec | RAG retrieval |
| AI inference (LLM_ONLY) | 1-2 sec | LLM generation |
| Nurse confirmation | 30 sec | UI review |
| Handover notification | <1 sec | WebSocket broadcast |
| Clinician acknowledgment | 1-30 min | Clinician response time (SLA) |

---

## Schema Relationships

```
TriageRequest (MongoDB)
├─ manualTriageReport
├─ aiTriageReport
├─ computeMode: 'HYBRID' | 'LLM_ONLY' | 'RAG_FULL'
├─ status: 'TRIAGED'
└─ triageNurseId → NurseProfile

Encounter (MongoDB)
├─ manual_triage_report
│  └─ field_changes: { esiLevel, specialty, physician, ... }
│     └─ overridden_by → NurseProfile
│
├─ clinical_handovers: [
│  ├─ clinician_id → ClinicalStaff
│  ├─ handover_status: 'NOTIFIED' | 'VIEWED' | 'ACKNOWLEDGED'
│  └─ report_type: 'MANUAL' | 'AI'
│ ]
│
├─ assigned_clinicians: [ physicians, specialists ]
├─ manually_selected_clinicians: [ nurse-chosen clinicians ]
│
└─ triageRequestId → TriageRequest (links back to full audit)
```

---

## Key Insights

1. **Dual Reports = Better Audit Trail**
   - Manual captures nurse expertise
   - AI captures pattern recognition
   - Override captures reasoning for improvement

2. **Override Tracking Enables ML Feedback Loop**
   - When nurse overrides AI, capture why
   - Specialist validates later
   - Improves model accuracy on future similar cases

3. **Multiple Clinicians = Multiple Handovers**
   - Nurse gets one clinical_handover entry
   - Physician gets separate entry
   - Specialist gets separate entry
   - Each tracks their own acknowledgment SLA

4. **Report Type Tracks Context**
   - Clinician sees `report_type: 'MANUAL'` with override reasoning
   - Or `report_type: 'AI'` with confidence scores
   - Contextual info helps clinician understand why triage happened this way

