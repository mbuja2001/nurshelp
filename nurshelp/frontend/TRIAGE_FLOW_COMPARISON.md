# Triage Flow Comparison: Old vs. New Production Architecture

## Current Architecture (Old Flow)

### Entry Point: POST /api/triage/run

**Frontend Call** (RightColumn.jsx → PatientReport.jsx):
```javascript
// This is triggered from somewhere in the UI
const response = await fetch(`${BACKEND_URL}/api/triage/run`, {
  method: 'POST',
  body: JSON.stringify({
    patient: { id_number, name, symptoms, history, ... },
    vitals: { temp, bp, hr, o2, resp, avpu, ... },
    transcript: [...],
    arrival_time, nurseNotes
  })
});
```

**Backend Flow:**
```
POST /api/triage/run
    ↓
triage.route.js → runTriage() from encounter.controller.js
    ↓
encounter.controller.js:
  1. Validates payload (patient + vitals required)
  2. Calls Python ALWAYS: POST http://localhost:5000/triage
  3. Stores full response in triage field
  4. Creates Encounter document with triage data
  5. Returns created encounter
    ↓
Response: { encounter: { _id, patient, vitals, triage, ... } }
```

### Physician Recommendations: GET /api/physicians/recommendations

**Frontend Call** (PatientReport.jsx after encounter created):
```javascript
const response = await fetch(
  `/api/physicians/recommendations?encounterId=${encounterId}`,
  { headers: { Authorization: `Bearer ${token}` } }
);

// Consumes response and displays top 5 physicians
```

**Backend Flow:**
```
GET /api/physicians/recommendations?encounterId=<id>
    ↓
physician.controller.js::getRecommendations()
    ↓
  1. Fetches encounter by ID
  2. Extracts triage data (triage.top_5_composite_physicians, etc.)
  3. If triage stored physician lists: Uses them
  4. If NO physician lists: Calls Python ESI_Engine again
  5. Normalizes + ranks by specialty match + workload
  6. Returns top 5
    ↓
Response: { physicians: [{ id, name, specialty, score, reason }, ...] }
```

### Key Issues with Old Flow:
- ❌ **No separation between manual and AI triage**
- ❌ **Python called IMMEDIATELY** (no option for LLM_ONLY quick path)
- ❌ **No computeMode parameter** (always tries full RAG)
- ❌ **Physician.csv always loads** (resource-heavy)
- ❌ **Recommendations fetched separately** (two endpoints)
- ❌ **No AI confidence/mode tracking** in stored data
- ❌ **Can't distinguish between manual vs AI-suggested ESI**

---

## New Production Architecture

### Two-Workflow Support

#### Workflow 1: Manual Triage (New) - Zero AI
```
Nurse Interface:
  ├─ Enter symptoms, vitals manually
  ├─ GET /api/triage/staff (show available nurses/physicians/beds)
  ├─ Nurse manually decides: ESI level
  ├─ Nurse manually selects: assigned nurse/physician/bed
  ├─ Create manual triage report
  └─ NO computeMode parameter, NO AI API call

Result:
  ├─ TriageRequest stored with:
  │  ├─ manual TriageReport (only)
  │  ├─ aiTriageReport: null
  │  └─ specialistFeedbackReport: null (filled later)
  └─ Fast execution (~500ms, no AI latency)
```

#### Workflow 2: AI-Assisted Triage (New) - With Optional AI
```
Nurse Interface:
  ├─ Enter symptoms, vitals manually
  ├─ GET /api/triage/staff (show available nurses/physicians/beds)
  ├─ Choose computeMode: "RAG_FULL" or "HYBRID" (optional)
  ├─ POST /api/triage with computeMode + staff data
  │  └─ ESI_Engine processes based on mode:
  │     ├─ Mode="RAG_FULL": Load physician.csv + FAISS search (threshold > 0.70)
  │     └─ Mode="HYBRID": Light load + softer threshold (> 0.50)
  ├─ Returns AI suggestions (ESI, acuity, recommended physician, similar cases)
  ├─ Nurse reviews both manual thinking + AI recommendations
  ├─ Nurse confirms or overrides with manual selection
  └─ Create both manual + AI triage reports

Result:
  ├─ TriageRequest stored with:
  │  ├─ manualTriageReport (nurse's initial assessment)
  │  ├─ aiTriageReport (AI suggestions)
  │  └─ specialistFeedbackReport: null (filled later)
  └─ Execution time depends on mode (2-5s)
```

### New Endpoints

#### POST /api/triage - Manual Path (No computeMode)

```javascript
Frontend Request:
{
  triageNurseId: ObjectId,           // Current logged-in nurse
  // NO computeMode - this signals MANUAL-ONLY workflow
  
  // Clinical data (always collected)
  symptoms: ["chest pain", "SOB"],
  vitalSigns: {
    temperature: 98.6,
    heartRate: 102,
    respiratoryRate: 22,
    systolicBP: 145,
    diastolicBP: 88,
    oxygenSaturation: 94
  },
  medicalHistory: [...],
  medications: [...],
  allergies: [...],
  chiefComplaint: "Acute chest pain",
  triageNotes: "Patient states pain started 30min ago",
  
  // Manual ESI from nurse (their judgment, no AI)
  manualESILevel: 2,
  manualAcuity: 75,
  triageReasoningNotes: "Chest pain with vital abnormalities",
  recommendedDepartment: "ED",
  immediateInterventions: ["ECG", "IV", "Monitoring"],
  
  // Nurse picked staff from GET /api/triage/staff
  assignedNurseId: ObjectId,
  assignedPhysicianId: ObjectId,
  bedAssigned: "ED-05"
}
```

#### POST /api/triage - AI-Assisted Path (With computeMode)

```javascript
Frontend Request:
{
  triageNurseId: ObjectId,
  computeMode: "RAG_FULL" | "HYBRID",  // This SIGNALS AI-assisted workflow
  
  // Clinical data
  symptoms: ["chest pain", "SOB"],
  vitalSigns: {...},
  medicalHistory: [...],
  medications: [...],
  allergies: [...],
  chiefComplaint: "Acute chest pain",
  triageNotes: "Patient states pain started 30min ago",
  
  // Nurse's initial assessment (before AI)
  manualESILevel: 2,
  manualAcuity: 75,
  triageReasoningNotes: "Chest pain with vital abnormalities",
  recommendedDepartment: "ED",
  immediateInterventions: ["ECG", "IV", "Monitoring"]
  
  // NO assigned staff yet - nurse will review AI suggestions first
}

Backend Processing:
  ↓
triage.controller.js::submitTriage()
  ├─ Extract computeMode from request
  ├─ Fetch current staff schedule (GET from schedule controller)
  ├─ Build engine payload with:
  │  ├─ patientData
  │  ├─ staffScheduling (data only, not control)
  │  └─ computeMode (controls ESI_Engine behavior)
  ├─ Call ESI_Engine if computeMode !== "LLM_ONLY"
  │  └─ ESI_Engine checks mode and processes accordingly
  ├─ Store TriageRequest with:
  │  ├─ manualTriageReport (always)
  │  ├─ aiTriageReport (if mode !== LLM_ONLY)
  │  ├─ computeMode used (for audit)
  └─ Return response with both reports + metrics
  
Backend Response:
{
  triageRequestId: ObjectId,
  encounterId: "ENC-123",
  computeMode: "RAG_FULL",
  
  manualReport: {
    esiLevel: 2,
    estimatedAcuity: 75,
    reasoning: "Chest pain with vital abnormalities",
    ...
  },
  
  aiReport: {  // Only if computeMode !== "LLM_ONLY"
    esiLevel: 2,
    estimatedAcuity: 78,
    confidence: 0.92,
    reasoning: "ACS concern with troponin risk",
    recommendedPhysician: { physicianId, name, specialty },
    recommendedNurse: { nurseId, name, licenseType },
    relevantCasesRetrieved: [...]
  },
  
  staffAvailable: { nurseCount: 5, physicianCount: 3 },
  processingMetrics: {
    totalTime: 1250,
    aiProcessingTime: 890,
    usedAI: true
  }
}
```

#### GET /api/triage/staff?date=2026-04-11

```javascript
Frontend Request:
  GET /api/triage/staff?date=2026-04-11

Backend Response (from schedule controller):
{
  date: "2026-04-11",
  timestamp: "2026-04-11T08:30:00Z",
  nurses: [
    {
      nurseId: ObjectId,
      name: "Sarah Johnson",
      licenseType: "RN-ICU",
      specialty: "Critical Care",
      currentWorkload: 2,
      maxBedsPerShift: 6,
      isOnBreak: false,
      shift: { start: "08:00", end: "16:00" }
    },
    ...
  ],
  physicians: [
    {
      physicianId: "DR-001",
      name: "Dr. Smith",
      specialty: "Cardiology",
      workloadScore: 0.6,
      isAvailable: true,
      shift: { start: "08:00", end: "17:00" }
    },
    ...
  ]
}
```

#### GET /api/triage/:triageRequestId

```javascript
Frontend Request:
  GET /api/triage/6507abc123def456

Backend Response (TriageRequest document):
{
  triageRequestId: ObjectId,
  encounterId: "ENC-123",
  triageNurseId: ObjectId,
  
  manualTriageReport: {...},
  aiTriageReport: {...},
  specialistFeedbackReport: null,  // Filled after specialist review
  
  computeMode: "RAG_FULL",
  status: "TRIAGED",
  isAttended: false,
  assignedNurseId: null,
  assignedPhysicianId: null,
  
  triageDuration: 1250,
  aiProcessingTime: 890
}
```

#### PATCH /api/triage/:triageRequestId/specialist-feedback

```javascript
Frontend Request (After specialist reviews):
{
  specialistId: ObjectId,
  actualESILevel: 2,
  actualAcuity: 78,
  finalDiagnosis: "Acute Coronary Syndrome",
  wasMisTriaged: false,
  feedbackNotes: "ESI assessment accurate, recommend..." 
}

Backend Creates:
specialistFeedbackReport: {
  specialistId: ObjectId,
  actualESILevel: 2,
  actualAcuity: 78,
  finalDiagnosis: "Acute Coronary Syndrome",
  wasMisTriaged: false,
  misTriageType: null,
  feedbackNotes: "...",
  createdAt: Date
}
```

---

## Server.js Changes Needed

### Old server.js
```javascript
app.use("/api/triage", triageRouter);          // POST /api/triage/run
app.use("/api/physicians", physicianRouter);   // GET /api/physicians/recommendations?encounterId
```

### New server.js
```javascript
// Keep triage router for new endpoints
app.use("/api/triage", triageRouter);          // POST /api/triage (new)
                                                // GET /api/triage/staff (new)
                                                // GET /api/triage/:id (new)
                                                // PATCH /api/triage/:id/specialist-feedback (new)

// Optionally keep physician router for backward compatibility
app.use("/api/physicians", physicianRouter);   // GET /api/physicians/recommendations (legacy)
```

### New routes/triage.route.js
```javascript
import express from "express";
import {
  submitTriage,
  getTriageRequest,
  getUnattendedTriages,
  updateSpecialistFeedback,
  getStaffSchedule,
  markTriageAttended
} from "../controllers/triage.controller.js";

const router = express.Router();

// POST /api/triage - Submit triage (manual or AI-assisted)
router.post("/", submitTriage);

// GET /api/triage/staff?date=2026-04-11 - Get current staff (for UI + AI context)
router.get("/staff", getStaffSchedule);

// GET /api/triage/:id - Retrieve specific triage request
router.get("/:id", getTriageRequest);

// GET /api/triage?status=TRIAGED&isAttended=false - Get unattended triages
router.get("/", getUnattendedTriages);

// PATCH /api/triage/:id/specialist-feedback - Add specialist review
router.patch("/:id/specialist-feedback", updateSpecialistFeedback);

// PATCH /api/triage/:id/attended - Mark as attended (assigned staff)
router.patch("/:id/attended", markTriageAttended);

export default router;
```

---

## Data Model Changes

### Old Encounter Model
```javascript
{
  _id, nurse_id, patient, vitals, transcript,
  triage: {        // Single triage object from Python
    priority_code, priority_colour, ai_summary,
    SATS_reasoning, assigned_specialty, assigned_physician,
    ...
  },
  status, isWaiting, severity, nurseNotes,
  createdAt, updatedAt
}
```

### New TriageRequest Model
```javascript
{
  _id,
  encounterId,              // Links to encounter (if exists)
  triageNurseId,            // Current logged-in nurse
  
  // Nurse's manual entry
  manualTriageData: {
    symptoms, vitalSigns, medicalHistory, medications, allergies,
    chiefComplaint, triageNotes
  },
  
  // Nurse's manual ESI decision
  manualTriageReport: {
    esiLevel, estimatedAcuity, reasoning, recommendedDepartment,
    immediateInterventions, createdBy, createdAt
  },
  
  // AI suggestions (only if computeMode !== LLM_ONLY)
  aiTriageReport: {
    esiLevel, estimatedAcuity, confidence, reasoning,
    recommendedDepartment, recommendedPhysician, recommendedNurse,
    relevantCasesRetrieved, createdAt
  },
  
  // Specialist accuracy tracking
  specialistFeedbackReport: {
    specialistId, actualESILevel, actualAcuity, finalDiagnosis,
    wasMisTriaged, misTriageType, feedbackNotes, createdAt
  },
  
  // Compute mode used (audit trail)
  computeMode: "LLM_ONLY" | "RAG_FULL" | "HYBRID",
  
  // Tracking
  status: "TRIAGED",
  isAttended: false,
  assignedNurseId, assignedPhysicianId, departmentAssigned, bedAssigned,
  
  // Metrics
  triageDuration, aiProcessingTime, rawESIEngineResponse,
  
  timestamps: true
}
```

---

## Flow Diagram Comparison

### OLD FLOW
```
┌─ Nurse enters symptoms/vitals
├─ POST /api/triage/run (patient, vitals, no computeMode)
├─ Python ESI_Engine called IMMEDIATELY
├─ Python returns triage object
├─ Encounter created with triage data
├─ Frontend fetches GET /api/physicians/recommendations?encounterId
├─ Physician.controller calls Python again (if no stored list)
└─ Physicians displayed to nurse
   └─ Nurse manually selects (manual component added as afterthought)

Issues:
- AI is mandatory, not optional
- Physician selection separated from triage
- Can't distinguish manual vs AI decision
- Staff data not passed to AI for ranking
- computeMode not supported
```

### NEW FLOW: Two Distinct Paths
```
┌─ Triage Nurse Interface shows two buttons:
│  ├─ [Manual Triage] (ZERO AI)
│  └─ [AI-Assisted Triage] (With AI Help)
│
├─ PATH A: Manual Triage (ZERO AI)
│  ├─ Nurse enters symptoms/vitals
│  ├─ GET /api/triage/staff → Shows available nurses/physicians/beds
│  ├─ Nurse manually decides ESI level (their judgment)
│  ├─ Nurse manually selects: assigned nurse + physician + bed
│  ├─ POST /api/triage (NO computeMode parameter)
│  │  ├─ Backend checks: no computeMode → skip Python
│  │  ├─ Validate data only
│  │  └─ Store TriageRequest with manualTriageReport only
│  └─ ~500ms, pure human decision
│
├─ PATH B: AI-Assisted Triage (With Optional AI)
│  ├─ Nurse enters symptoms/vitals
│  ├─ GET /api/triage/staff → Shows available nurses/physicians/beds
│  ├─ Nurse chooses: computeMode="RAG_FULL" or "HYBRID"
│  ├─ POST /api/triage { computeMode, symptoms, vitals, ... }
│  │  ├─ Backend checks: computeMode present → call Python
│  │  ├─ Fetch staff schedule + urgency context
│  │  ├─ ESI_Engine processes based on mode:
│  │  │  ├─ RAG_FULL: Full physician.csv + FAISS (threshold > 0.70)
│  │  │  └─ HYBRID: Lighter load + softer threshold (> 0.50)
│  │  ├─ Returns: AI ESI + confidence + recommendations + similar cases
│  │  └─ Store TriageRequest with manualTriageReport + aiTriageReport
│  ├─ Frontend displays side-by-side:
│  │  ├─ Nurse's thinking (manual report)
│  │  ├─ AI's thinking (AI report + confidence + reasoning)
│  │  └─ Nurse confirms AI or overrides with manual selection
│  └─ ~2-5s depending on mode
│
└─ Later: Specialist reviews patient
   ├─ PATCH /api/triage/:id/specialist-feedback
   └─ TriageRequest updated with specialist assessment + accuracy tracking

Key Differences:
- Manual = NO computeMode, NO Python call, ~500ms
- AI-Assisted = WITH computeMode, Python called, ~2-5s
- Nurse chooses per-patient at triage time
- Both paths tracked separately for accuracy measurement
- Staff scheduling always visible (informational)
- Specialist feedback used for model improvement
```

---

## Implementation Checklist

### Backend Changes Needed

- [x] Create TriageRequest model (triageRequest.model.js)
- [x] Create triage controller (triage.controller.js) with:
  - [x] submitTriage() - Main endpoint
  - [x] getTriageRequest() - Fetch by ID
  - [x] getUnattendedTriages() - Find waiting patients
  - [x] updateSpecialistFeedback() - Add specialist review
  - [x] getStaffSchedule() - Fetch available staff
  - [x] markTriageAttended() - Assign staff
- [x] Create triage routes (triage.route.js) with proper HTTP verbs
- [ ] Update ESI_Engine.py to parse computeMode and conditionally load physician.csv + FAISS
- [ ] Update server.js to register new triage routes
- [ ] Add staffSchedule endpoint to schedule.controller.js (if not exists)

### Frontend Changes Needed

- [ ] Create new Triage Interface component with computeMode radio select
- [ ] Create Manual Triage workflow UI
- [ ] Create AI-Assisted Triage workflow UI
- [ ] Create Staff Schedule display (available nurses/physicians)
- [ ] Create TriageRequest view (show all 3 reports)
- [ ] Update PatientReport.jsx to use new endpoints
- [ ] Add specialist feedback form to EHR page

### Integration Points

- [ ] Modify existing encounter flow to support new TriageRequest
- [ ] Backward compatibility: Keep old /api/physicians/recommendations endpoint
- [ ] Analytics: Query TriageRequest to compare manual vs AI accuracy
- [ ] Audit: Log all changes to specialist feedback

---

## Key Differences Summary

| Aspect | Old | New |
|--------|-----|-----|
| **Compute Mode** | None (always full RAG) | Per-request: LLM_ONLY \| RAG_FULL \| HYBRID |
| **Triage Paths** | 1 (AI mandatory) | 2 (Manual or AI-assisted) |
| **Manual vs AI** | No separation | Both stored, compared |
| **Scheduling** | Not used | Fetched + passed to AI |
| **Physician Ranking** | Separate endpoint | Included in POST /api/triage response |
| **Specialist Feedback** | Not tracked | Full feedback report stored |
| **Audit Trail** | Partial | Complete: mode, both reports, specialist review |
| **Speed** | 2-5s (always AI) | 500ms (manual) to 5s (AI) |
| **Resource Use** | High (always FAISS) | Variable (depends on mode) |
