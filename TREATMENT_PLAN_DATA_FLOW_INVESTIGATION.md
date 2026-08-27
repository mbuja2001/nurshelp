# 🔍 TREATMENT PLAN DATA FLOW INVESTIGATION

## CURRENT DATA FLOW (How Treatment Plan is Pulled & Displayed)

### 1️⃣ FETCH ENDPOINT
**Location**: Backend `GET /api/encounters/{id}`
**Handler**: `encounter.controller.js:getEncounterById()` (line 594)
**Returns**: 
```json
{
  "encounter": {
    "_id": "...",
    "diagnosis": "Clinical diagnosis text",
    "treatment_plan": "Treatment plan text or empty",
    "physician_notes": "Notes",
    "treatments": [
      { "name": "Medication A", "dosage": "500mg", "frequency": "2x daily", "indication": "..." },
      { "name": "Lab Test B", "dosage": null, "indication": "..." }
    ],
    "recommendations": [
      { "description": "Follow up imaging", "recommended_by": "..." }
    ],
    "progression_notes": [
      { "content": "Progress note text", "recorded_by": "...", "recorded_at": "..." }
    ],
    "medication_requests": [
      { "medication_name": "Aspirin", "dosage": "100mg", "status": "PENDING", "requested_by": "..." }
    ],
    "lab_requests": [ ... ],
    "surgery_records": [ ... ]
  }
}
```

**Data Structure**: Full encounter object with ALL fields including medication_requests, treatments, recommendations

---

### 2️⃣ FRONTEND FETCH (PhysicianDashboard.jsx)
**Location**: Line 320 - `fetchEncounterForBed()`
**Endpoint Called**: `GET ${BACKEND_URL}/api/encounters/{encounterId}`
**Data Retrieved**: `data.encounter` or `data.case` or direct return
**Next Step**: Pass to `normalizeBedModalEncounter()` for display normalization

---

### 3️⃣ FRONTEND NORMALIZATION (Line 450-475)
**Function**: `normalizeBedModalEncounter()`
**Purpose**: Transform raw encounter into consistent display structure

**Key Normalization Rules**:
```javascript
{
  treatment_plan: normalizedEncounter.treatment_plan 
    || normalizedEncounter.triage?.aiTriageReport?.recommended_treatment 
    || normalizedEncounter.triage?.recommended_treatment 
    || '',
  
  treatments: normalizedEncounter.treatments 
    || normalizedEncounter.treatment_plan?.treatments 
    || [],
  
  recommendations: normalizedEncounter.recommendations 
    || normalizedEncounter.triage?.aiTriageReport?.recommendations 
    || [],
  
  medication_requests: normalizedEncounter.medication_requests 
    || normalizedEncounter.medicationRequests 
    || [],
  
  progression_notes: normalizedEncounter.progression_notes || []
}
```

**Translation**: Ensures frontend always gets these arrays even if they come from different sources

---

### 4️⃣ FRONTEND DISPLAY (Bed Modal) (Lines 3195-3225)

**Treatment Display Logic**:
```jsx
{bedModalData?.treatments?.length ? (
  // Show treatments array with name and dosage
  bedModalData.treatments.slice(0, 3).map((treatment) => (
    <div>{treatment.name} · {treatment.dosage}</div>
  ))
) : (
  // Fallback to treatment_plan text if no treatments array
  <p>{bedModalData?.treatment_plan || 'Pending'}</p>
)}
```

**Recommendations Display Logic**:
```jsx
{bedModalData?.recommendations?.length ? (
  // Show recommendations array
  bedModalData.recommendations.slice(0, 3).map((recommendation) => (
    <div>• {recommendation.description}</div>
  ))
) : (
  // Fallback to AI triage recommendation if no array
  <p>{bedModalData?.triage?.aiTriageReport?.recommended_treatment || 'Pending'}</p>
)}
```

**Other Displays**:
- Line 3204: `bedModalData?.treatment_plan` (text display)
- Line 3381: `bedModalData?.treatment_plan || bedModalData?.physician_notes` (text display)
- Line 3256: `{bedModalData?.medication_requests?.length || 0}` (count display)
- Line 3250: `{bedModalData?.progression_notes?.length || 0}` (count display)
- Line 3399: `bedModalData.treatments.slice(0, 2)` (treatments array display)

**KEY INSIGHT**: 
- Bed modal expects `bedModalData.treatments[]` as PRIMARY display
- Falls back to `treatment_plan` text if no treatments array
- Also expects `medication_requests[]` to be present (for count/status)
- Also expects `recommendations[]` and `progression_notes[]`

---

## ✅ WHAT MUST BE PRESERVED

The current data contract is:

1. **Fetch Response Structure** (NO CHANGE):
   ```
   GET /api/encounters/{id} returns { encounter: { ... full object ... } }
   ```

2. **Database Schema** (NO CHANGE):
   - `encounter.diagnosis` text field
   - `encounter.treatment_plan` text field
   - `encounter.treatments[]` array
   - `encounter.recommendations[]` array
   - `encounter.medication_requests[]` array
   - `encounter.progression_notes[]` array

3. **Frontend Normalization** (NO CHANGE):
   - `bedModalData.treatment_plan` must exist
   - `bedModalData.treatments[]` must exist
   - `bedModalData.recommendations[]` must exist
   - `bedModalData.medication_requests[]` must exist (for bed modal to display count)
   - `bedModalData.progression_notes[]` must exist

4. **Frontend Display Logic** (NO CHANGE):
   - Show treatments array IF exists
   - Show treatment_plan text as fallback
   - Show recommendations array
   - Show medication_requests count
   - Show progression_notes

---

## 🔄 WHAT CHANGES

### Current Workflow (Monolith):
```
DiagnosePage.jsx
  ↓
  Collect diagnosis + treatments + recommendations + MEDICATION_REQUESTS
  ↓
  PUT /api/encounters/{id}/diagnosis
    body: { diagnosis, treatments, recommendations, medication_requests }
  ↓
  recordDiagnosis()
    └─ Stores medication_requests in encounter.medication_requests[]
  ↓
  GET /api/encounters/{id}
    └─ Returns encounter with all fields populated
  ↓
  Frontend displays from bedModalData
```

### New Workflow (Modular):
```
STEP 1: DiagnosePage.jsx (Diagnosis Only)
  ↓
  Collect diagnosis + treatments + recommendations (NO MEDICATION_REQUESTS)
  ↓
  PUT /api/encounters/{id}/diagnosis
    body: { diagnosis, treatments, recommendations }  ← NO medication_requests
  ↓
  recordDiagnosis()
    └─ Stores diagnosis, treatments, recommendations ONLY
    └─ Ignores medication_requests parameter if sent
  ↓

STEP 2: Patient Hospitalized & Assigned to Bed
  ↓
  Bed modal loaded
  ↓
  GET /api/encounters/{id} (STILL WORKS - returns full data)
    └─ Returns encounter with diagnosis, treatments, recommendations
    └─ medication_requests array may be empty initially
  ↓

STEP 3: Physician Creates Medication Order in Bed Modal
  ↓
  handleOrderMedication()
  ↓
  POST /api/pharmacy/medication-requests
    body: { medication_name, dosage, frequency, route, indication }
  ↓
  Creates medication request in encounter.medication_requests[]
  ↓

STEP 4: Data Refresh (refreshBedModalData)
  ↓
  GET /api/encounters/{id}
    └─ Now returns medication_requests with newly created orders
  ↓
  Frontend displays from bedModalData (SAME DISPLAY LOGIC)
```

---

## 🎯 SUMMARY: WHAT TO PRESERVE

| Component | Action | Reason |
|-----------|--------|--------|
| **Database Schema** | ✅ KEEP | All fields stay, just populated differently |
| **GET Endpoint** | ✅ KEEP | Returns full encounter, no change |
| **Fetch Response** | ✅ KEEP | Same structure `{ encounter: {...} }` |
| **Frontend Normalization** | ✅ KEEP | Same data contract for display |
| **Display Logic** | ✅ KEEP | Same arrays and fallback logic |
| **Bed Modal Refresh** | ✅ KEEP | `refreshBedModalData()` pulls latest data |
| **Count Displays** | ✅ KEEP | `bedModalData?.medication_requests?.length` still works |

---

## 🚫 WHAT TO REMOVE

| Component | Action | Reason |
|-----------|--------|--------|
| **DiagnosePage medication UI** | ❌ REMOVE | Medication ordering happens in bed modal |
| **DiagnosePage medication state** | ❌ REMOVE | No local collection of medication requests |
| **recordDiagnosis medication handling** | ❌ REMOVE | Stop processing medication_requests parameter |
| **persistDiagnosis medication payload** | ❌ REMOVE | Stop sending medication_requests to backend |
| **handleAddMedicationRequest** | ❌ REMOVE | Moved to bed modal (handleOrderMedication) |
| **handleRemoveMedicationRequest** | ❌ REMOVE | Moved to bed modal |

---

## 🔗 THE KEY INSIGHT

**We're not changing WHAT DATA IS STORED or HOW IT'S DISPLAYED.**

We're only changing **WHEN AND WHERE** that data is CREATED:
- ✅ **Was**: Created during diagnosis in DiagnosePage
- ✅ **Now**: Created later during bed modal interaction
- ✅ **Same**: Data ends up in same place, just via different API path

The fetch, display, storage, and schema remain identical. We're just separating the **workflow pathway** from a monolith (create all at once in DiagnosePage) to modular (create each specialist action in bed modal).

---

## 📋 NEXT ACTION

After this refactoring, when physician:
1. Diagnoses patient → Stores diagnosis, treatments, recommendations
2. Hospitalizes → Assigned bed
3. In bed modal → Creates medication requests via pharmacy endpoint
4. Refreshes data → Gets same display as before

**The display experience is IDENTICAL. We just changed the API pathway.**
