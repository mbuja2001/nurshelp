# Clinical Triage & Handover Architecture

## Executive Summary

The clinical triage and handover system is a **dual-track workflow** that combines AI-assisted triage with manual nurse assessment, converging on a unified clinical handover to assigned clinicians. Both pathways produce audit-traceable reports that feed into the clinical handover system.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CLINICAL TRIAGE WORKFLOW                         │
└─────────────────────────────────────────────────────────────────────┘

   MANUAL TRIAGE PATH                    AI-ASSISTED TRIAGE PATH
   ═════════════════════                 ══════════════════════════
   
   PatientSummary.jsx                    ESI_Engine.py (LLM/RAG)
   │                                     │
   ├─ Nurse enters vitals                ├─ LLM-Only Mode
   ├─ Nurse enters symptoms              │  (DISABLE_RAG=1)
   ├─ Nurse selects ESI level            │
   │                                     ├─ RAG-Full Mode
   ↓                                     │  (PubMedBERT + retrieval)
                                         │
                                         ├─ Hybrid Mode (RAG + LLM)
                                         │
                                         ↓
   TriageRequest Model                   
   ├─ manualTriageReport                 ├─ aiTriageReport
   │  (ALWAYS created by nurse)          │  (ESI, specialty, physician rec)
   │  └─ esiLevel, reasoning             │
   │                                     ├─ aiConfidence score
   │                                     │
   │                                     ↓
   │                              PatientReport.jsx
   │                              (Inference Preview)
   │                              │
   │                              ├─ AI recommendations displayed
   │                              ├─ Nurse can accept OR override
   │                              │
   │                              ↓
   │                      TriageConfirmationPage.jsx
   │                      (Confirmation + Edit)
   │                      │
   │                      Path A: ACCEPT AI
   │                      └─ Stores: manualTriageReport + aiTriageReport
   │                         EHR: ["Manual Report", "AI Report"]
   │
   │                      Path B: OVERRIDE AI
   │                      └─ Stores: manualTriageReport + aiTriageReport 
   │                         + nurseEditsReport (tracks changes + reasoning)
   │                         EHR: ["Manual Report", "AI Report", "Nurse Edits"]
   │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ↓
                    ╔════════════════════════╗
                    │  TriageRequest Saved   │
                    │  with audit trail      │
                    ╚════════════════════════╝
                              │
                              ├─ triageRequestId recorded
                              ├─ computeMode tracked
                              ├─ triageType: 'MANUAL_ONLY' | 'AI_ASSISTED'
                              ├─ status: 'TRIAGED'
                              │
                              ↓
                    ╔════════════════════════════════╗
                    │  CLINICAL HANDOVER GENERATION   │
                    ╚════════════════════════════════╝
                              │
                    ┌─────────┴──────────┐
                    │                    │
              AI-Recommended          Manual Override
              Specialist             Assessment
              │                       │
              ├─ ESI recommendation   ├─ Nurse ESI choice
              ├─ Specialty assigned   ├─ Specialty override
              ├─ Physician scored     ├─ Physician manual selection
              ├─ Availability checked ├─ Notes on why override
              │                       │
              ↓                       ↓
         ┌─────────────────────────────────────┐
         │ ENCOUNTER Handover Trigger          │
         │ (ConfirmingAIClinicalHandover.jsx)  │
         │ (ConfirmingManualClinicalHandover) │
         │                                     │
         │ POST /api/encounters/{id}/handover  │
         └─────────────────────────────────────┘
                              │
                              ↓
         ┌─────────────────────────────────────┐
         │  CLINICAL_HANDOVERS Array Created   │
         │  in Encounter Document              │
         │  [per assigned clinician]           │
         │                                     │
         │  Fields:                            │
         │  - clinician_id                     │
         │  - clinician_name                   │
         │  - handover_type: 'INITIAL'         │
         │  - handover_status: 'NOTIFIED'      │
         │  - report_type: 'AI' | 'MANUAL'     │
         │  - handover_at: timestamp           │
         │  - viewed_at: null (until clinician views)
         │  - acknowledged_at: null            │
         │  - acknowledged_notes: null         │
         └─────────────────────────────────────┘
                              │
                              ↓
         ┌──────────────────────────────────┐
         │ REAL-TIME NOTIFICATIONS          │
         │ to assigned clinicians           │
         │ (WebSocket/Event-driven)         │
         │                                  │
         │ Notification includes:           │
         │ - Triage summary                 │
         │ - Patient vitals                 │
         │ - Assigned bed/ward              │
         │ - Link to full report            │
         │ - Acknowledgment deadline        │
         └──────────────────────────────────┘
```

---

## Detailed Data Flow

### 1. TRIAGE PHASE

#### Input Data (PatientSummary.jsx)
```javascript
// Manual Triage Assessment Entry
const manualTriageData = {
  symptoms: ["chest pain", "shortness of breath"],
  vitalSigns: {
    temperature: 37.2,
    heartRate: 92,
    respiratoryRate: 18,
    systolicBP: 145,
    diastolicBP: 88,
    oxygenSaturation: 96
  },
  medicalHistory: ["hypertension", "diabetes"],
  medications: ["metformin", "enalapril"],
  allergies: ["penicillin"],
  chiefComplaint: "Acute onset chest pain with dyspnea",
  triageNotes: "Patient reports pain 8/10, first episode"
};
```

#### ESI Engine Processing (ESI_Engine.py)
```python
# Three execution modes:
# 1. LLM_ONLY: Pure GGUF LLM reasoning (CPU-friendly)
# 2. RAG_FULL: PubMedBERT embeddings + similarity search + LLM
# 3. HYBRID: RAG fallback to LLM if GPU unavailable

class ESIEngine:
  def triage(self, vitals, symptoms):
    # Step 1: Extract pathognomonic markers
    markers = detect_pathognomonic_markers(chief_complaint)
    
    # Step 2: Run diagnostic criteria matching
    criteria_matches = match_diagnostic_criteria(text, vitals)
    
    # Step 3: Calculate SATS discriminators
    sats_score = calibrate_esi_to_vitals(initial_esi, vitals)
    
    # Step 4: If RAG available, retrieve similar cases
    similar_cases = RAG_RETRIEVER.retrieve(
      clinical_text=formatted_narrative,
      top_k=5,
      min_similarity=0.70  # CLEAN index filtering
    )
    
    # Step 5: LLM generates final triage report
    ai_report = llm.generate_triage_report(
      vitals=vitals,
      symptoms=symptoms,
      similar_cases=similar_cases,
      sats_score=sats_score
    )
    
    return ai_report
```

#### TriageRequest Model (Backend Storage)
```javascript
// Stored in MongoDB after triage complete
{
  _id: ObjectId("..."),
  encounterId: "ENC-123456",
  triageNurseId: ObjectId("..."),
  
  // MANUAL DATA (nurse entry)
  manualTriageData: {
    symptoms: [...],
    vitalSigns: {...},
    medicalHistory: [...],
    ...
  },
  
  // MANUAL REPORT (nurse's ESI choice)
  manualTriageReport: {
    esiLevel: 3,
    estimatedAcuity: 65,
    reasoning: "Acute chest pain with stable vitals = ESI-3",
    recommendedDepartment: "Cardiology",
    immediateInterventions: ["EKG", "troponin panel"],
    createdBy: ObjectId("..."),  // triageNurseId
    createdAt: ISODate("...")
  },
  
  // AI PROCESSING MODE
  computeMode: "HYBRID",  // LLM_ONLY | RAG_FULL | HYBRID
  
  // AI REPORT (if computeMode !== "LLM_ONLY")
  aiTriageReport: {
    esiLevel: 2,  // Might differ from manual!
    estimatedAcuity: 78,
    confidence: 0.87,
    reasoning: "Acute presentation + chest pain + elevated BP = ESI-2",
    recommendedDepartment: "Emergency Medicine",
    recommendedPhysician: {
      physicianId: ObjectId("..."),
      name: "Dr. Sarah Chen",
      specialization: "Emergency Medicine",
      availabilityScore: 0.92,
      workloadScore: 0.45
    },
    relevantCasesRetrieved: [
      {
        caseId: "CASE-001",
        similarity: 0.84,
        esiLevel: 2,
        outcome: "MI ruled out, discharged"
      },
      ...
    ],
    createdAt: ISODate("...")
  },
  
  // AUDIT
  status: "TRIAGED",
  triageDuration: 1240,  // ms
  aiProcessingTime: 3200,  // ms
  
  timestamps: {...}
}
```

---

### 2. CONFIRMATION PHASE

#### Scenario A: AI-Assisted Workflow (ConfirmingAIClinicalHandover.jsx)

```javascript
// Step 1: Fetch latest encounter
GET /api/encounters/{encounterId}

// Response includes:
{
  _id: "...",
  triage: {
    // AI-generated report
    esiLevel: 2,
    specialty: "Cardiology",
    physician: "Dr. Sarah Chen",
    ...
  },
  manual_triage_report: {
    // Nurse's manual assessment
    esiLevel: 3,
    notes: "Patient appears stable"
  },
  assigned_clinicians: [
    // AI-recommended clinicians
    {
      id: ObjectId("..."),
      name: "Dr. Sarah Chen",
      role: "PHYSICIAN",
      specialty: "Emergency Medicine",
      score: 0.92
    }
  ]
}

// Step 2: Nurse reviews side-by-side comparison
// Path A: ACCEPT AI RECOMMENDATION
// └─ Confirms handover with assigned_clinicians from AI

const handleConfirmHandover = async () => {
  const response = await fetch(`/api/encounters/${encounterId}/handover`, {
    method: 'POST',
    body: JSON.stringify({
      reportType: 'AI_TRIAGE_REPORT',
      clinicians: encounter.assigned_clinicians,
      // AI recommendation accepted as-is
    })
  });
}

// Path B: OVERRIDE AI (trigger manual selection flow)
navigate(`/scheduler?mode=clinician-override&encounterId=${encounterId}`)
// └─ Nurse manually selects different physician/clinicians
// └─ Records override reasoning in manual_triage_report.field_changes
```

#### Scenario B: Manual-Only Workflow (ConfirmingManualClinicalHandover.jsx)

```javascript
// Step 1: Fetch encounter
GET /api/encounters/{encounterId}

// Step 2: Display manual report form
const manualReport = encounter.manual_triage_report || {}

// Step 3: Nurse confirms manual assessment
const handleConfirmHandover = async () => {
  // Save encounter with manual report fields
  await fetch(`/api/encounters/${encounterId}/ai-fields`, {
    method: 'PUT',
    body: JSON.stringify({
      triage: encounter.triage,
      assigned_clinicians: encounter.assigned_clinicians || [],
      manually_selected_clinicians: encounter.manually_selected_clinicians || [],
      manual_triage_report: encounter.manual_triage_report,
      nurseNotes: encounter.nurseNotes
    })
  });
  
  // Then trigger handover
  await fetch(`/api/encounters/${encounterId}/handover`, {
    method: 'POST',
    body: JSON.stringify({
      reportType: 'MANUAL_TRIAGE_REPORT',
      clinicians: encounter.assigned_clinicians || [],
      manuallySelected: encounter.manually_selected_clinicians || []
    })
  });
}
```

---

### 3. OVERRIDE TRACKING (For Audit & ML Improvement)

#### Override Storage Structure

```javascript
// In manual_triage_report.field_changes
{
  esiLevel: {
    original: 2,        // AI recommendation
    override: 3,        // Nurse's decision
    reason: "Patient appears clinically stable despite high acuity score"
  },
  recommendedPhysician: {
    original: {
      name: "Dr. Sarah Chen",
      specialty: "Emergency Medicine"
    },
    override: {
      name: "Dr. James Wilson",
      specialty: "Cardiology"
    },
    reason: "Dr. Chen at capacity, Dr. Wilson better match for chest pain"
  },
  specialty: {
    original: "Emergency Medicine",
    override: "Cardiology",
    reason: "Patient history indicates cardiac risk factors"
  }
}

// Metadata
{
  overridden_at: ISODate("2026-06-05T13:45:00Z"),
  overridden_by: ObjectId("..."),  // triageNurse._id
  
  // Links back to both reports for audit
  manual_report_id: ObjectId("..."),
  ai_report_id: ObjectId("..."),
  
  // For ML training: was the nurse right or wrong?
  specialist_feedback: null  // Filled in later by physician
}
```

#### Audit Trail Example

```
ENCOUNTER: ENC-12345 | PATIENT: John Doe
═════════════════════════════════════════

TRIAGE TIMELINE:
────────────────

13:00 - AI Engine generates triage report
  ESI Level: 2 (High Acuity)
  Confidence: 0.87
  Recommended Physician: Dr. Sarah Chen (EM)
  Specialty: Emergency Medicine
  Processing Time: 3.2s
  Mode: HYBRID (RAG + LLM)

13:15 - Nurse reviews PatientReport.jsx
  Manual Assessment ESI: 3
  Nurse Notes: "Patient seems comfortable, denies severe pain"

13:18 - Nurse accesses TriageConfirmationPage
  Option 1: ACCEPT AI (ESI-2, Dr. Chen)
  Option 2: OVERRIDE with manual (ESI-3, different physician)

13:22 - Nurse OVERRIDES
  Changes: ESI 2→3, Physician: Chen→Wilson
  Reasoning: "Patient vitals misleading; clinical presentation stable"
  
  ✓ Manual Report Saved
  ✓ Override Tracking Recorded
  ✓ Difference flagged for ML improvement

13:23 - Clinical Handover Triggered
  Handover Type: MANUAL (with nurse override reasoning)
  Assigned Clinician: Dr. James Wilson (Cardiology)
  Report Sent: manualTriageReport (includes override justification)

13:24 - Dr. Wilson Notified
  Notification includes:
  - Patient summary
  - Vitals snapshot
  - Nurse's clinical reasoning
  - Reference to AI recommendation (for context)

13:30 - Dr. Wilson Acknowledges
  Status: ACKNOWLEDGED
  Notes: "Agree with nurse assessment. Ordering EKG."

FINAL DISPOSITION:
──────────────────
Triage Type: AI-ASSISTED (with nurse override)
Final ESI: 3
Final Physician: Dr. James Wilson
Accuracy: ✓ Correct (confirmed by specialist feedback later)
ML Impact: This case tagged for training → improve ESI-2/3 boundary detection
```

---

### 4. CLINICAL HANDOVER PHASE

#### Encounter.clinical_handovers Array

```javascript
// Created when handover is confirmed

encounter.clinical_handovers = [
  {
    _id: ObjectId("..."),
    clinician_id: ObjectId("..."),  // Dr. James Wilson's ID
    clinician_name: "Dr. James Wilson",
    clinician_role: "PHYSICIAN",
    clinician_specialty: "Cardiology",
    
    // Handover metadata
    handover_type: "INITIAL",  // INITIAL | UPDATE | FOLLOW_UP
    handover_status: "NOTIFIED",  // NOTIFIED → VIEWED → ACKNOWLEDGED → IN_PROGRESS → COMPLETED
    handover_at: ISODate("2026-06-05T13:23:00Z"),
    
    // Clinician's interaction timeline
    viewed_at: ISODate("2026-06-05T13:25:30Z"),  // When clinician first viewed
    acknowledged_at: ISODate("2026-06-05T13:30:15Z"),  // When acknowledged
    acknowledged_notes: "Agree with assessment. Starting workup.",
    
    // Report variant tracked
    report_type: "MANUAL",  // Which report version they received
                           // AI-assisted cases get: manualTriageReport + aiTriageReport context
                           // Manual cases get: manualTriageReport only
    
    is_active: true  // Still assigned to this patient
  },
  
  // If multiple clinicians assigned (e.g., nurse + physician)
  {
    clinician_id: ObjectId("..."),  // Assigned Nurse
    clinician_name: "Emma Thompson",
    clinician_role: "NURSE",
    clinician_specialty: "Emergency Department",
    handover_type: "INITIAL",
    handover_status: "ACKNOWLEDGED",
    handover_at: ISODate("2026-06-05T13:23:00Z"),
    viewed_at: ISODate("2026-06-05T13:24:00Z"),
    acknowledged_at: ISODate("2026-06-05T13:26:00Z"),
    report_type: "MANUAL",
    is_active: true
  }
];

encounter.handover_completed = true;
encounter.handover_completed_at = ISODate("2026-06-05T13:30:15Z");
```

#### Handover Content Structure

```javascript
// What clinician receives in their notification

const clinicalHandoverContent = {
  // Patient snapshot
  patient: {
    name: "John Doe",
    age: 52,
    gender: "Male",
    id: "P-12345"
  },
  
  // Chief complaint & presentation
  presentation: {
    chiefComplaint: "Acute onset chest pain with dyspnea",
    onset: "30 minutes ago",
    severity: "8/10 pain"
  },
  
  // Current vitals
  vitals: {
    temperature: 37.2,
    heartRate: 92,
    respiratoryRate: 18,
    systolicBP: 145,
    diastolicBP: 88,
    oxygenSaturation: 96
  },
  
  // Location assignment
  location: {
    ward: "Emergency Department",
    bed: "ED-7",
    roomNumber: "102"
  },
  
  // Triage assessment (depends on report_type)
  triageAssessment: {
    // If report_type === "MANUAL"
    type: "Manual Triage by Nurse",
    esiLevel: 3,
    acuityScore: 65,
    reasoning: "Acute presentation but stable vitals",
    immediateInterventions: ["EKG", "troponin panel"],
    
    // If AI-assisted, also include context
    aiRecommendation: {
      esiLevel: 2,
      confidence: 0.87,
      reasoning: "High acuity pattern detected",
      note: "Nurse override: believes ESI-3 more appropriate"
    }
  },
  
  // Clinical notes from triage nurse
  clinicalNotes: {
    triageNurse: "Emma Thompson",
    notes: "Patient appears clinically stable despite test concerns",
    allergies: ["Penicillin"],
    medicalHistory: ["Hypertension", "Diabetes Type 2"]
  },
  
  // Recommended actions
  recommendations: [
    "12-lead EKG STAT",
    "Troponin panel",
    "Cardiology consult consideration",
    "Keep NPO pending workup"
  ],
  
  // Links to full EHR
  links: {
    fullReport: "/api/encounters/ENC-12345/report",
    acknowledgeHandover: "/api/encounters/ENC-12345/acknowledge",
    updateStatus: "/api/encounters/ENC-12345/status"
  },
  
  // Acknowledgment deadline (SLA)
  acknowledgmentRequired: true,
  acknowledgmentDeadline: ISODate("2026-06-05T14:00:00Z"),  // 30 min SLA
  
  // Audit metadata
  handover: {
    sentAt: ISODate("2026-06-05T13:23:00Z"),
    sentBy: "Triage System",
    encounterId: "ENC-12345"
  }
};
```

---

## API Endpoints

### Triage Confirmation Endpoints

```
POST /api/triage/{triageRequestId}/confirm
├─ Payload:
│  ├─ confirmed: boolean
│  ├─ confirmedAt: ISO timestamp
│  ├─ triageType: 'MANUAL_ONLY' | 'AI_ASSISTED'
│  ├─ manualTriageReport: {...}
│  ├─ aiTriageReport: {...} (optional)
│  └─ nurseEditsReport: {...} (if overridden)
│
└─ Returns: { success, triageRequestId, status }

PUT /api/encounters/{encounterId}/ai-fields
├─ Payload:
│  ├─ triage: {...}
│  ├─ assigned_clinicians: [...]
│  ├─ manually_selected_clinicians: [...]
│  ├─ manual_triage_report: {...}
│  └─ nurseNotes: String
│
└─ Returns: { success, encounter }

POST /api/encounters/{encounterId}/handover
├─ Payload:
│  ├─ reportType: 'MANUAL_TRIAGE_REPORT' | 'AI_TRIAGE_REPORT'
│  ├─ clinicians: [...]
│  └─ manuallySelected: [...]
│
└─ Returns: {
     success,
     handoverInitiatedAt: timestamp,
     cliniciansNotified: number,
     clinicalHandoversCreated: number
   }

PATCH /api/encounters/{encounterId}/acknowledge
├─ Payload:
│  ├─ acknowledgedAt: ISO timestamp
│  ├─ acknowledgmentNotes: String
│  └─ readyToReceivePatient: boolean
│
└─ Returns: { success, acknowledgedAt, clinician }

PATCH /api/triage/{triageRequestId}/specialist-feedback
├─ Payload:
│  ├─ specialistId: ObjectId
│  ├─ actualESILevel: number (1-5)
│  ├─ actualAcuity: number (0-100)
│  ├─ finalDiagnosis: String
│  ├─ wasMisTriaged: boolean
│  ├─ misTriageType: 'UNDER_TRIAGED' | 'OVER_TRIAGED' | 'WRONG_DEPT'
│  └─ feedbackNotes: String
│
└─ Returns: { success, feedbackRecorded, accuracyImprovement }
```

---

## Contextual Reasoning for Schemas

### Why Two Triage Reports?

1. **Manual Report** - Nurse's clinical judgment (gold standard during triage phase)
   - Created regardless of AI availability
   - Represents human expertise
   - Includes immediate clinical reasoning

2. **AI Report** - Machine-generated recommendations (for comparison & improvement)
   - Powered by ESI Engine (LLM/RAG)
   - Includes confidence scores
   - References similar historical cases
   - Useful for alerting nurses to patterns they might miss

### Why Track Overrides?

**ML Improvement**
- If nurse overrides AI, we capture why
- Later specialist feedback validates: was nurse right or AI?
- Feeds back into model retraining

**Audit Trail**
- Compliance: demonstrate decision-making rationale
- Quality: identify systematic biases (e.g., always under-triages certain presentations)
- Training: show nurses why their override was/wasn't appropriate

**Clinical Safety**
- Red flags: if multiple nurses override AI on same symptom pattern → investigate
- Escalation: if override causes poor outcome → feedback to nurse

### Why clinical_handovers Array?

Each clinician gets their own handover entry because:
1. **Multiple recipients**: Nurse + Physician + Specialist might all be assigned
2. **Different timelines**: Each clinician has their own acknowledgment deadline
3. **Role-specific info**: What a nurse needs differs from what a physician needs
4. **Status tracking**: Track who viewed, who acknowledged, who's still active
5. **Report variant**: Manual cases may show different report depth than AI-assisted

---

## Workflow Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                      COMPLETE WORKFLOW                           │
└──────────────────────────────────────────────────────────────────┘

PHASE 1: TRIAGE
├─ Nurse enters vitals + symptoms (PatientSummary.jsx)
├─ If AI enabled: ESI_Engine processes in background
├─ Both manual + AI reports generated (audit trail)
│
└─→ TriageRequest saved to DB
   Status: 'TRIAGED'
   Stores: manualTriageReport, aiTriageReport (if available)

PHASE 2: CONFIRMATION
├─ Nurse reviews reports (PatientReport.jsx)
├─ Choice A: Accept AI recommendation
│  └─→ Use assigned_clinicians from AI
├─ Choice B: Override AI
│  ├─→ Track all changes in field_changes
│  ├─→ Record override reasoning
│  └─→ Manually select clinicians (Scheduler)
│
└─→ TriageConfirmationPage processes confirmation
   Status: TRIAGED → CONFIRMED
   Stores: nurseEditsReport (if overridden)

PHASE 3: HANDOVER
├─ Backend creates clinical_handovers array
│  └─ One entry per assigned clinician
├─ Each clinician receives notification with:
│  ├─ Patient summary
│  ├─ Vitals snapshot
│  ├─ Clinical reasoning (manual + AI context if available)
│  ├─ Assigned location
│  └─ Acknowledgment link
│
├─ Clinician acknowledgment tracked:
│  └─ viewed_at → acknowledged_at → status: ACKNOWLEDGED
│
└─→ Encounter.handover_completed = true
   Status: AWAITING_PHYSICIAN → AWAITING_SPECIALIST (etc.)

PHASE 4: SPECIALIST FEEDBACK (Optional, occurs during care)
├─ Physician documents actual diagnosis
├─ Validates accuracy of triage assessment
├─ Marks: was_triage_correct? under_triaged? over_triaged?
│
└─→ Specialist feedback recorded
   Status: TRIAGE_VALIDATED
   Result fed back to ML model
   Audit trail complete: Manual → AI → Override (if any) → Specialist

PHASE 5: CARE CONTINUATION
├─ Nurse continues monitoring
├─ Orders executed
├─ Patient progresses through care
├─ Updates sent via clinical_handovers (UPDATE type)
│
└─→ Patient completes episode
   All reports archived
   Feedback integrated for next case
```

---

## Implementation Checklist

- [ ] **Backend**: `/api/triage/{id}/confirm` endpoint (accepts manual + AI reports)
- [ ] **Backend**: `/api/encounters/{id}/handover` endpoint (creates clinical_handovers)
- [ ] **Backend**: `/api/encounters/{id}/acknowledge` endpoint (tracks clinician views)
- [ ] **Backend**: Override tracking in manual_triage_report.field_changes
- [ ] **Frontend**: PatientReport.jsx displays AI recommendations vs manual
- [ ] **Frontend**: TriageConfirmationPage for side-by-side comparison
- [ ] **Frontend**: ConfirmingAIClinicalHandover.jsx flow
- [ ] **Frontend**: ConfirmingManualClinicalHandover.jsx flow
- [ ] **WebSocket**: Real-time notifications to assigned clinicians
- [ ] **ML**: Specialist feedback ingestion → model retraining pipeline
- [ ] **Audit**: Full handover lifecycle logged with timestamps

---

## Next Steps

1. **Test AI-Assisted Flow**: Generate triage request via ESI_Engine, confirm with override
2. **Test Manual-Only Flow**: Manual triage confirmation without AI
3. **Test Handover Notifications**: Verify clinicians receive + can acknowledge
4. **Test Override Auditing**: Validate field_changes tracking for ML feedback
5. **Integration Test**: Full end-to-end from triage through specialist feedback

