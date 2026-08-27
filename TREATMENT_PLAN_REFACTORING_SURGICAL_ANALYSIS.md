# 🔧 TREATMENT PLAN REFACTORING - SURGICAL CODE ANALYSIS

## 📍 PROBLEM STATEMENT

**Current Monolith Bottleneck**:
- Treatment plan endpoint (`PUT /api/encounters/{id}/diagnosis`) accepts medication_requests
- Creates medication requests DIRECTLY in encounter during diagnosis
- Bypasses dedicated specialist workflows (bed modal actions)
- Blocks proper tracking and modular specialist interactions

**Desired State**:
- Treatment plan = **JUST STORE THE PLAN** (diagnosis, treatments, recommendations, progression_notes)
- Medication ordering = **SEPARATE dedicated workflow** (bed modal → handleOrderMedication)
- Lab requests = **SEPARATE dedicated workflow** (bed modal → handleRequestLabs)
- Surgery scheduling = **SEPARATE dedicated workflow** (bed modal → handleSurgeryRequest)
- Each action trackable with status bar independently

---

## 🔍 ENDPOINT ANALYSIS

### Current Route Definition
**File**: `nurshelp/backend/src/routes/encounter.route.js`
```javascript
// Line 72: Diagnosis endpoint
router.put("/:id/diagnosis", protect, recordDiagnosis);

// Line 75: Inpatient report endpoint  
router.put("/:id/inpatient-report", protect, createInpatientProgressReport);
```

---

## 🎯 CONTROLLER LOGIC - WHAT NEEDS REMOVAL

### Controller: recordDiagnosis
**File**: `nurshelp/backend/src/controllers/encounter.controller.js`
**Lines**: 1354-1467

**PROBLEMATIC CODE TO REMOVE** (Lines 1408-1418):
```javascript
// ❌ THIS CREATES MEDICATION REQUESTS - SHOULD BE REMOVED
if (Array.isArray(medication_requests) && medication_requests.length > 0) {
  encounter.medication_requests = medication_requests
    .filter(isMedicationRequestValid)
    .map((request) => ({
      ...normalizeMedicationRequest(request),
      requested_by: clinicianHandover.clinician_name,
      requested_at: new Date()
    }));
}
```

**WHAT STAYS** (Should remain as-is):
```javascript
// ✅ TREATMENT PLAN DATA - KEEP THIS
encounter.diagnosis = diagnosis;
encounter.treatment_plan = treatment_plan || '';
encounter.physician_notes = notes || '';

// ✅ TREATMENTS ARRAY (the actual intervention plan)
if (Array.isArray(treatments) && treatments.length > 0) {
  encounter.treatments = treatments.map(t => ({
    ...t,
    prescribed_by: clinicianHandover.clinician_name,
    prescribed_at: new Date()
  }));
}

// ✅ RECOMMENDATIONS (clinical recommendations)
if (Array.isArray(recommendations) && recommendations.length > 0) {
  encounter.recommendations = recommendations.map(r => ({
    ...r,
    recommended_by: clinicianHandover.clinician_name,
    recommended_at: new Date()
  }));
}

// ✅ PROGRESSION NOTES (clinical documentation)
if (Array.isArray(progression_notes) && progression_notes.length > 0) {
  encounter.progression_notes = progression_notes.map(p => ({
    ...p,
    recorded_by: clinicianHandover.clinician_name,
    recorded_at: new Date()
  }));
}
```

---

## 📋 FRONTEND - WHAT NEEDS TO CHANGE

### File: `nurshelp/frontend/src/PatientSummary/DiagnosePage.jsx`

**REMOVE** (Lines 20-21):
```javascript
// ❌ REMOVE: Medication requests should not be collected here
const [medicationRequests, setMedicationRequests] = useState([]);
const [newMedicationRequest, setNewMedicationRequest] = useState({ medication_name: "", dosage: "", frequency: "", route: "Oral", indication: "" });
```

**REMOVE** (Lines 134-162):
```javascript
// ❌ REMOVE: handleAddMedicationRequest function
const handleAddMedicationRequest = async () => { ... }

// ❌ REMOVE: handleRemoveMedicationRequest function
const handleRemoveMedicationRequest = (idx) => { ... }
```

**REMOVE** (Lines 495-570 - entire "Medication Requests for Pharmacy Review" section):
```javascript
{/* ❌ REMOVE: MEDICATION REQUESTS SECTION */}
<section style={{ marginBottom: 28 }}>
  <h3>Medication Requests for Pharmacy Review</h3>
  {/* All medication request UI */}
</section>
```

**UPDATE** (Lines 54-80, persistDiagnosis function):
```javascript
// ❌ BEFORE:
const persistDiagnosis = async (overrideDiagnosis = diagnosisText, requestList = medicationRequests) => {
  const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/diagnosis`, {
    method: "PUT",
    body: JSON.stringify({
      diagnosis: overrideDiagnosis?.trim(),
      treatment_plan: "",
      notes: progressionNote,
      treatments: treatments,
      recommendations: recommendations,
      progression_notes: progressionNote ? [...] : [],
      medication_requests: requestList  // ❌ REMOVE THIS LINE
    })
  });
}

// ✅ AFTER:
const persistDiagnosis = async (overrideDiagnosis = diagnosisText) => {
  const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/diagnosis`, {
    method: "PUT",
    body: JSON.stringify({
      diagnosis: overrideDiagnosis?.trim(),
      treatment_plan: "",
      notes: progressionNote,
      treatments: treatments,
      recommendations: recommendations,
      progression_notes: progressionNote ? [...] : []
      // medication_requests removed
    })
  });
}
```

**UPDATE** (All calls to persistDiagnosis - remove medicationRequests parameter):
```javascript
// ❌ BEFORE:
const savedEncounter = await persistDiagnosis(diagnosisText, nextMedicationRequests);

// ✅ AFTER:
const savedEncounter = await persistDiagnosis(diagnosisText);
```

**REMOVE** from handleDisposition():
```javascript
// Line 41:
setMedicationRequests(enc.medication_requests || []);  // ❌ REMOVE
```

---

## 🛠️ DEDICATED WORKFLOWS - WHERE ACTIONS SHOULD HAPPEN

### 1️⃣ Medication Ordering
**Location**: PhysicianDashboard.jsx → Bed Modal
**Handler**: `handleOrderMedication()` (Line 1040)
**Endpoint**: Existing medication ordering endpoint
```javascript
handleOrderMedication = async () => {
  // ✅ Medication ordering happens here, NOT in treatment plan
  // Specialists interact via bed modal
  // Creates independent medication request with status tracking
}
```

### 2️⃣ Lab Requests  
**Location**: PhysicianDashboard.jsx → Bed Modal
**Handler**: `handleRequestLabs()` (Line 783)
**Endpoint**: `POST /api/lab-requests/{encounterId}/create`
```javascript
handleRequestLabs = async () => {
  // ✅ Lab requests happen here, NOT in treatment plan
  // Radiologist/Pathologist gets independent queue
  // Creates lab_requests[] with status tracking
}
```

### 3️⃣ Surgery Scheduling
**Location**: PhysicianDashboard.jsx → Bed Modal
**Handler**: `handleSurgeryRequest()` (Line 1208)
**Endpoint**: `POST /api/encounters/{id}/surgery` (implied)
```javascript
handleSurgeryRequest = async () => {
  // ✅ Surgery scheduling happens here, NOT in treatment plan
  // Creates surgery_records[] with status tracking
}
```

---

## 📊 DATA FLOW COMPARISON

### ❌ CURRENT (Monolith - Everything in diagnosis):
```
DiagnosePage.jsx
  ↓
  Physician enters diagnosis + treatments + MEDICATION REQUESTS
  ↓
  PUT /api/encounters/{id}/diagnosis
  ↓
  recordDiagnosis controller
  ├─ Stores diagnosis
  ├─ Stores treatments
  ├─ Stores recommendations
  └─ Creates medication_requests directly ❌ MONOLITH
  ↓
  Medication sits in encounter.medication_requests[]
  ❌ No separate pharmacy workflow needed
  ❌ No status tracking per action
  ❌ Blocks specialist interaction
```

### ✅ DESIRED (Modular - Each action separate):
```
STEP 1: Diagnosis Phase
  DiagnosePage.jsx
    ↓
    Physician enters diagnosis + treatments + recommendations
    ↓
    PUT /api/encounters/{id}/diagnosis  ✅ CLEAN - JUST THE PLAN
    ↓
    recordDiagnosis controller
    ├─ Stores diagnosis
    ├─ Stores treatments
    ├─ Stores recommendations
    └─ DONE (no medication requests)

STEP 2: Hospitalization
  Physician clicks "Hospitalize"
    ↓
    Navigate to bed assignment
    ↓
    Assign bed in ward

STEP 3: Specialist Actions (in Bed Modal)
  Physician clicks bed modal actions:
  
  3A. Order Medication
      handleOrderMedication()
        ↓
        POST /api/pharmacy/medication-order  ✅ SEPARATE WORKFLOW
        ↓
        Creates medication with status tracking
        
  3B. Request Labs
      handleRequestLabs()
        ↓
        POST /api/lab-requests/{id}/create  ✅ SEPARATE WORKFLOW
        ↓
        Creates lab_request with status tracking
        
  3C. Schedule Surgery
      handleSurgeryRequest()
        ↓
        POST /api/encounters/{id}/surgery  ✅ SEPARATE WORKFLOW
        ↓
        Creates surgery_record with status tracking
```

---

## 📝 SUMMARY OF CHANGES NEEDED

### Backend Changes:
1. **encounter.controller.js - recordDiagnosis()**
   - Remove medication_requests handling (lines 1408-1418)
   - Keep treatments, recommendations, progression_notes, diagnosis only
   
2. **encounter.controller.js - createInpatientProgressReport()**
   - Same removal as above

### Frontend Changes:
1. **DiagnosePage.jsx**
   - Remove medication request form section
   - Remove medication request state (medicationRequests, newMedicationRequest)
   - Remove handleAddMedicationRequest, handleRemoveMedicationRequest functions
   - Update persistDiagnosis to NOT send medication_requests
   - Update all function calls to persistDiagnosis

### No Changes Needed:
- PhysicianDashboard.jsx bed modal workflows (already separate)
- Lab request workflow (already separate)
- Medication ordering workflow (already separate)
- Surgery workflow (already separate)

---

## 🎯 BENEFITS OF THIS REFACTORING

✅ **Cleaner Separation of Concerns**
- Treatment plan is JUST a plan
- Specialist actions are isolated and trackable

✅ **Modular Status Tracking**
- Each action (medication, lab, surgery) gets independent status bar
- Users see journey visually: PENDING → APPROVED/ACKNOWLEDGED → COMPLETED

✅ **Proper Workflow Flow**
- Diagnose (in DiagnosePage)
- Hospitalize (select bed)
- Interact (bed modal specialists do their actions)

✅ **No Blocking**
- Physicians can diagnose without triggering downstream cascades
- Specialists act independently via bed modal

---

## 🔗 RELATED ENDPOINTS (NOT CHANGING)

These are the CORRECT dedicated workflows that should remain:

1. `POST /api/pharmacy/medication-order` - Order medication
2. `POST /api/lab-requests/{id}/create` - Request lab
3. `POST /api/encounters/{id}/surgery` - Schedule surgery
4. `POST /api/pharmacy/dispense` - Dispense medication
5. `PATCH /api/lab-requests/{id}/acknowledge` - Specialist acknowledges lab
6. `PATCH /api/lab-requests/{id}/submit-report` - Specialist submits report

These remain the PRIMARY paths for specialist actions.
