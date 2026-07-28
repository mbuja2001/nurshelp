# Patient Routing Architecture
## Intelligent Clinician Assignment Based on Triage

**Goal:** When a patient completes AI triage or manual triage, automatically route them to an available clinician (Physician or Clinical Nurse) from today's shift schedule.

---

## Current State

### Triage System
- **Manual Triage:** NurseTriageForm captures vitals, creates TriageAssessment
- **AI Triage:** LLM analyzes vitals → generates SATS/TEWS score → recommends priority/context
- **Output:** TriageAssessment stored with patient ID, priority, severity, context

### Staff Scheduling System
- **Shift Creation:** Admin creates shifts via StaffScheduleManager
- **Team Assignment:** Shifts get `teamAssignment` { triageNurseId, clinicalNurseId, physicianId }
- **Context:** Clinical role gets context (ER, ICU, SURGICAL)
- **Daily Status:** Staff clock in/clock out tracked in NurseClockRecord

### Role-Based Dashboards
- **Triage Nurse:** Sees patient queue, triages, needs to route to clinician
- **Clinical Nurse:** Sees assigned patients (hospitalized/observation)
- **Physician:** Sees patient lanes (triage waiting, active assessment, observation)

---

## Problem Statement

**Current Gap:** After triage, patients are stuck—no automatic routing to available clinicians on today's shift.

**Missing Links:**
1. ❌ Triage result → which clinician should handle this patient?
2. ❌ Is that clinician available (clocked in, not on break)?
3. ❌ What patients already assigned to that clinician?
4. ❌ Patient location tracking (ED triage vs hospitalized ward vs ICU)?

---

## Proposed Architecture

### 1. Clinician Availability Query

**Endpoint:** `GET /api/shifts/available-clinicians/today`

Returns all clinicians **currently working today** with their assignment details:

```json
{
  "success": true,
  "clinicians": [
    {
      "staffId": "621a...",
      "name": "Dr. Chen",
      "role": "PHYSICIAN",
      "context": "PHYSICIAN_ER",        // ER or ICU or SURGICAL
      "clockStatus": "CLOCKED_IN",    // CLOCKED_IN or ON_BREAK
      "assignedPatients": ["62...", "63..."],  // Current patient IDs
      "capacity": "available",        // available or at_capacity
      "specializations": ["Cardiology", "Emergency"],
      "department": "ED"
    },
    {
      "staffId": "621b...",
      "name": "Sarah Johnson",
      "role": "CLINICAL_NURSE",
      "context": "CLINICAL_NURSE_ER",
      "clockStatus": "CLOCKED_IN",
      "assignedPatients": ["64..."],
      "capacity": "available"
    }
  ]
}
```

**Query Logic:**
- Find today's shifts with `teamAssignment` (triageNurseId, clinicalNurseId, physicianId non-null)
- Filter by clock status: only CLOCKED_IN (exclude ON_BREAK)
- Get assigned patients from active TriageAssessments/PatientJourney
- Determine capacity: max 3-5 patients per clinician depending on role

---

### 2. Triage → Routing Decision Rules

**When triage completes:**

```
IF SATS_SCORE ≤ 1 (Critical)
  → Route to PHYSICIAN (highest priority)
  → Prefer same context (ER, ICU, SURGICAL)

ELSE IF SATS_SCORE = 2 (High)
  → Route to PHYSICIAN or CLINICAL_NURSE
  → Match by clinical context

ELSE (Low priority)
  → Route to CLINICAL_NURSE
  → Can defer if no capacity

AVAILABILITY CHECK:
  → Pick clinician with lowest current load
  → If all at capacity → queue for next available
```

---

### 3. Patient Location Tracking

Add to **TriageAssessment** schema:

```javascript
const triageSchema = {
  // ... existing fields ...
  
  status: {
    triage: 'COMPLETED',          // PENDING, IN_PROGRESS, COMPLETED
    assignment: 'ASSIGNED',        // UNASSIGNED, ASSIGNED, HANDED_OFF
    location: 'ED_TRIAGE',         // ED_TRIAGE, ED_OBSERVATION, ICU_BED_X, WARD_A, DISCHARGED
  },
  
  assignedClinician: {
    clinicianId: ObjectId,         // Physician or Nurse ID
    clinicianName: String,
    clinicianRole: String,         // PHYSICIAN or CLINICAL_NURSE
    assignedAt: Date,
    context: String,               // PHYSICIAN_ER, CLINICAL_NURSE_ICU, etc
  },
  
  hospitalization: {
    isHospitalized: Boolean,
    bedAssignment: String,         // Bed number if hospitalized
    ward: String,                  // ICU, Surgical, Medical-Surgical
    admissionTime: Date,
  }
};
```

---

### 4. Triage Endpoint Flow

**POST /api/triage** (after manual triage completes):

```
1. Validate triage data (vitals, priority, SATS/TEWS)
2. Create/update TriageAssessment
3. Query available clinicians → GET /api/shifts/available-clinicians/today
4. Apply routing rules → select best clinician
5. Update TriageAssessment.assignedClinician
6. Emit WebSocket event → clinician's dashboard receives new patient
7. Return: { success: true, assignedClinician: {...}, routingReason: "..." }
```

---

### 5. Role-Based Dashboard Updates

#### Triage Nurse Dashboard
- **Current:** Sees triage queue (pending patients)
- **New:** 
  - After triaging a patient, see "✓ Routed to Dr. Chen (ER)"
  - Option to manually reassign if needed

#### Clinical Nurse Dashboard
- **Current:** Sees observation patients
- **New:**
  - Real-time updates: new patients routed by triage
  - Shows patient triage urgency + vital signs
  - Can mark patient as "hospitalized" → bed assignment
  - Tracks patient location (ED observation → ICU bed → discharge)

#### Physician Dashboard
- **Current:** Views triage queue, active cases, disposition
- **New:**
  - Triage queue filtered by assigned patients
  - Shows routed patients from triage (with priority indicators)
  - Can set disposition: admit to ICU/ward, discharge, refer
  - Dashboard lanes: **Triage Queue** (routed to me) | **Active Cases** (I'm assessing) | **Observation** (Clinical nurse monitoring) | **Ready for Disposition**

---

## Implementation Plan

### Phase 1: Availability & Capacity
- [ ] Add `getAvailableCliniciansToday` backend endpoint
- [ ] Track assigned patients per clinician in real-time
- [ ] Define capacity rules per role

### Phase 2: Patient Location & Hospitalization
- [ ] Update TriageAssessment schema with location, hospitalization fields
- [ ] Add bed assignment logic (query available beds)
- [ ] Create admission workflow

### Phase 3: Intelligent Routing
- [ ] Implement routing rules in triage endpoint
- [ ] Route patient to available clinician after triage
- [ ] Store assignedClinician in TriageAssessment

### Phase 4: Dashboard Integration
- [ ] Clinical Nurse: see routed patients, update locations
- [ ] Physician: see triage queue + routed patients
- [ ] Real-time WebSocket updates for new assignments

### Phase 5: Manual Override
- [ ] Triage Nurse: manually reassign patient to different clinician
- [ ] Clinician: accept/decline patient assignment

---

## Data Flow Diagram

```
Manual Triage Form
      ↓
NurseTriageForm → POST /api/triage
      ↓
Backend triage handler:
  1. Create TriageAssessment
  2. Compute SATS score
  3. GET /api/shifts/available-clinicians/today
  4. Apply routing rules
  5. assignedClinician = best match
  6. WebSocket broadcast: "New patient routed to you"
      ↓
Physician/Clinical Nurse Dashboard:
  - Real-time notification
  - Patient appears in "Triage Queue" / "New Assignments"
  - Can view vitals, triage notes, SATS score
  - Take action: accept, view details, mark hospitalized, etc.
```

---

## Database Changes Needed

### TriageAssessment
```javascript
{
  // existing: vital signs, priority, severity, esi_context
  
  // NEW: Routing & Assignment
  assignedClinician: {
    clinicianId: ObjectId,
    clinicianName: String,
    role: String,               // PHYSICIAN or CLINICAL_NURSE
    context: String,            // ER, ICU, SURGICAL
    assignedAt: Date,
    assignmentSource: String,   // AUTOMATIC_ROUTING or MANUAL_OVERRIDE
  },
  
  // NEW: Status tracking
  status: {
    triageStatus: String,       // PENDING, IN_PROGRESS, COMPLETED
    assignmentStatus: String,   // UNASSIGNED, ASSIGNED, HANDOFF_DECLINED
    location: String,           // ED_TRIAGE, ED_OBSERVATION, ICU_BED_3, WARD_A, DISCHARGED
  },
  
  // NEW: Hospitalization
  hospitalization: {
    isHospitalized: Boolean,
    bedId: ObjectId,            // Reference to Bed model
    bedNumber: String,
    ward: String,               // ICU, Surgical, Medical-Surgical
    admissionTime: Date,
  }
}
```

---

## Questions for Discussion

1. **Capacity Limits:** Should physicians have max 5 patients, nurses 3? Adjustable per hospital?
2. **Context Matching:** If only PHYSICIAN_ICU clinician available, should we route a non-ICU patient there?
3. **Queue Strategy:** If no capacity, should patient queue for next available, or notify triage nurse?
4. **Override Authority:** Can triage nurse override routing, or only clinicians?
5. **Bed Assignment:** Should clinical nurse assign bed, or is that a separate admin workflow?
6. **WebSocket Events:** Real-time updates for new assignments, or polling?

---

## Next Steps

1. **Consensus:** Agree on routing rules, capacity, and override authority
2. **Design ER Diagram:** Map out relationships (Shift → Clinician → TriageAssessment)
3. **Implement Phase 1:** Availability endpoint + capacity tracking
4. **Test End-to-End:** Create shift → triage patient → verify routed to clinician
