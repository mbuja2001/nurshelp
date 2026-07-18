# NurseHelp Refactoring Plan - Schematics (NO IMPLEMENTATION YET)
**Status**: Planning Phase - Review & Approval Required  
**Date**: 10 April 2026

---

## 1. Data Model Schematics

### 1.1 Extended User Model
**Current State**: `Nurse` model has only {name, surname, email, password, timestamps}

**Proposed Extension**: Add compute preferences to User
```
User {
  _id: ObjectId
  name: string
  email: string (unique)
  password: string (hashed)
  role: enum(NURSE, PHYSICIAN, ADMIN)  // NEW
  
  // Compute Preferences (NEW)
  computeMode: enum(LLM_ONLY, RAG_FULL, HYBRID) = RAG_FULL
  enableGPU: boolean = true
  ragTimeout: number (ms) = 30000
  llmModel: enum(MEDGEMMA_7B, QWEN) = MEDGEMMA_7B
  
  // Facility (NEW)
  facilityId: ObjectId (ref Facility)
  department: string (ER, ICU, etc.)
  maxPatientsPerShift: number = 10
  
  // Preferences (NEW)
  preferences: {
    notificationsEnabled: bool,
    specialtyPreferences: [string],
    autoAssignToSpecialty: bool
  }
  
  timestamps: {createdAt, updatedAt}
}
```

**Question**: Should we rename "Nurse" to "User" or create new "User" model alongside existing "Nurse"?

---

### 1.2 NEW: TriageSettings Model
**Purpose**: Store facility-wide compute configuration + per-user overrides

```
TriageSettings {
  _id: ObjectId
  facilityId: ObjectId (ref Facility)
  userId: ObjectId (ref User)
  
  computeMode: enum(LLM_ONLY, RAG_FULL, HYBRID)
  
  ragParams: {
    semanticWeight: number [0.35-0.85] = 0.70
    similarityThreshold: number [0.70-0.95] = 0.85
    topKResults: number = 5
    enableIdfWeighting: boolean = true
  }
  
  llmParams: {
    model: string = MEDGEMMA_7B
    temperature: number [0-1] = 0.3
    maxTokens: number = 512
    systemPrompt: string (optional)
  }
  
  hybridRAGWeight: number [0.0-1.0] = 0.6  // For HYBRID mode
  
  updatedBy: ObjectId (ref User)
  timestamps: {createdAt, updatedAt}
  
  // Index: UNIQUE (userId, facilityId)
}
```

**Question**: One record per user per facility, or facility-wide defaults with user overrides?

---

### 1.3 NEW: PhysicianSchedule Model
**Purpose**: Hot-reloadable daily physician roster

```
PhysicianSchedule {
  _id: ObjectId
  facilityId: ObjectId
  userId: ObjectId (ref User)
  
  shiftDate: Date
  shiftStart: string "08:00"
  shiftEnd: string "16:00"
  specialty: string (Cardiology, ER, etc.)
  subspecialties: [string] (optional)
  maxPatientsPerShift: number = 20
  
  status: enum(AVAILABLE, ON_DUTY, ON_CALL)
  currentPatientCount: number = 0
  activeEncounters: [ObjectId] (ref Encounter)
  onCallForSpecialties: [string]
  
  contactNumber: string
  updatedBy: ObjectId
  source: enum(MANUAL_UPLOAD, AUTO_GENERATED)
  
  timestamps: {createdAt, updatedAt}
  
  // Indices:
  // - (facilityId, shiftDate, specialty)
  // - (userId, shiftDate)
  // - (facilityId, status)
}
```

**Question**: Should we track which physician has how many beds assigned?

---

### 1.4 NEW: BedSchedule Model
**Purpose**: Real-time bed availability tracking

```
BedSchedule {
  _id: ObjectId
  facilityId: ObjectId
  
  bedId: string UNIQUE "ER-01", "ICU-12"
  roomNumber: string
  bedType: enum(ER, ICU, GENERAL, PEDIATRIC, PSYCH)
  
  status: enum(AVAILABLE, OCCUPIED, MAINTENANCE, RESERVED, CLEANING)
  currentEncounterId: ObjectId (ref Encounter) nullable
  occupiedSince: Date nullable
  estimatedDischarge: Date nullable
  actualDischarge: Date nullable
  
  acuityLevel: number [1-5] = 3  // ESI match
  equipment: [string] (VENTILATOR, MONITOR, IV_PUMP)
  isIsolated: boolean = false
  isolationType: string (DROPLET, AIRBORNE, etc.)
  
  assignedNurseId: ObjectId (ref User) nullable
  
  updatedBy: ObjectId
  cleanedAt: Date nullable
  
  timestamps: {createdAt, updatedAt}
  
  // Indices:
  // - (facilityId, bedType, status)
  // - (facilityId, status)
  // - (bedId, facilityId) UNIQUE
}
```

**Question**: Should we track cleaning schedule separately or here?

---

### 1.5 NEW: NurseSchedule Model
**Purpose**: Daily nurse roster with bed assignments

```
NurseSchedule {
  _id: ObjectId
  facilityId: ObjectId
  userId: ObjectId (ref User)
  
  shiftDate: Date
  shiftStart: string "08:00"
  shiftEnd: string "16:00"
  shiftType: enum(MORNING, AFTERNOON, NIGHT, EXTENDED)
  
  assignedBeds: [string] (ref BedSchedule.bedId)
  maxPatientsPerShift: number = 5
  certifications: [string] (ACLS, PALS, etc.)
  
  status: enum(AVAILABLE, ASSIGNED, ON_BREAK, ON_DUTY)
  currentPatientCount: number = 0
  activeEncounters: [ObjectId] (ref Encounter)
  
  breaksTaken: [{breakStart, breakEnd, type}]
  totalBreakMinutes: number = 0
  
  updatedBy: ObjectId
  source: enum(MANUAL_UPLOAD, AUTO_GENERATED)
  
  timestamps: {createdAt, updatedAt}
  
  // Indices:
  // - (facilityId, shiftDate, status)
  // - (userId, shiftDate)
  // - (assignedBeds, status)
}
```

**Question**: Should nurse max patients per shift be configurable or fixed?

---

### 1.6 NEW: Facility Model
**Purpose**: Centralized facility configuration

```
Facility {
  _id: ObjectId
  
  name: string
  facilityType: enum(HOSPITAL, CLINIC, URGENT_CARE, TRAUMA_CENTER)
  
  location: {
    address, city, state, zip, country
    coordinates: {latitude, longitude}
  }
  
  adminEmail: string
  supportPhone: string
  emergencyPhone: string
  
  // Global defaults
  defaultComputeMode: enum(LLM_ONLY, RAG_FULL, HYBRID) = RAG_FULL
  
  bedsTotal: number = 100
  nursesTotal: number = 30
  physiciansTotal: number = 15
  specialtiesAvailable: [string]
  
  ragConfiguration: {
    enableRAG: boolean = true
    semanticWeight: number = 0.70
    similarityThreshold: number = 0.85
    topKResults: number = 5
  }
  
  enabledFeatures: {
    multiLanguageSupport: bool,
    advancedAnalytics: bool,
    automatedScheduling: bool,
    bedManagement: bool,
    physicianReferral: bool
  }
  
  status: enum(ACTIVE, INACTIVE, SUSPENDED, TRIAL)
  licenseExpiry: Date
  
  timestamps: {createdAt, updatedAt}
  
  // Index: name, status
}
```

**Questions**: 
- Should facility have hard capacity limits that trigger automatic mode switches?
- Do we need multi-site support?

---

## 2. API Endpoint Schematics

### 2.1 Settings Endpoints

#### **GET /api/settings/user**
- **Auth**: JWT required
- **Returns**:
```json
{
  "user": {
    "id": "uid123",
    "name": "Jane Doe",
    "email": "jane@hospital.com",
    "role": "NURSE",
    "computeMode": "RAG_FULL",
    "facilityId": "fac123"
  },
  "triageSettings": {
    "computeMode": "RAG_FULL",
    "ragParams": {
      "semanticWeight": 0.70,
      "similarityThreshold": 0.85,
      "topKResults": 5
    },
    "llmParams": {
      "model": "MEDGEMMA_7B",
      "temperature": 0.3
    }
  }
}
```

#### **PATCH /api/settings/user**
- **Auth**: JWT required
- **Body**:
```json
{
  "computeMode": "LLM_ONLY",
  "preferences": {
    "specialtyPreferences": ["Cardiology", "ER"],
    "notificationsEnabled": true
  }
}
```
- **Returns**: Updated settings + success message

---

#### **GET /api/settings/facility**
- **Auth**: JWT required (ADMIN preferred)
- **Returns**:
```json
{
  "facility": {
    "name": "City Hospital",
    "defaultComputeMode": "RAG_FULL",
    "bedsTotal": 100,
    "ragConfiguration": {...}
  }
}
```

#### **PATCH /api/settings/facility**
- **Auth**: JWT + ADMIN role required
- **Body**:
```json
{
  "defaultComputeMode": "HYBRID",
  "ragConfiguration": {
    "semanticWeight": 0.75,
    "similarityThreshold": 0.80
  }
}
```

---

### 2.2 Schedule Endpoints

#### **POST /api/schedules/physicians/upload**
- **Auth**: JWT required (ADMIN)
- **Body**: FormData with CSV file
- **CSV Columns**: `physician_id, name, specialty, shift_start, shift_end, max_patients, on_call`
- **Behavior**: Hot-reload (deletes old schedules for same date, inserts new)
- **Returns**:
```json
{
  "message": "Imported 15 physician schedules",
  "recordsImported": 15,
  "recordsRejected": 2,
  "errors": ["Row 5: Physician john@hospital.com not found"],
  "scheduleDate": "2026-04-10"
}
```

#### **GET /api/schedules/physicians**
- **Query**: `?date=2026-04-10&specialty=Cardiology&status=AVAILABLE`
- **Returns**:
```json
{
  "total": 8,
  "date": "2026-04-10",
  "data": [
    {
      "userId": "uid456",
      "specialty": "Cardiology",
      "shiftStart": "08:00",
      "shiftEnd": "16:00",
      "status": "AVAILABLE",
      "maxPatientsPerShift": 20,
      "currentPatientCount": 3
    }
  ]
}
```

---

#### **POST /api/schedules/beds/upload**
- **Auth**: JWT required (ADMIN)
- **CSV Columns**: `bed_id, room_number, bed_type, acuity_level, equipment`
- **Behavior**: Hot-reload (replaces all facility beds)
- **Returns**: Same structure as physician upload

#### **GET /api/schedules/beds**
- **Query**: `?status=AVAILABLE&bedType=ER&isAvailable=true`
- **Returns**:
```json
{
  "total": 50,
  "available": 8,
  "occupied": 32,
  "maintenance": 5,
  "cleaning": 5,
  "data": [...]
}
```

---

#### **POST /api/schedules/nurses/upload**
- **Auth**: JWT required (ADMIN)
- **CSV Columns**: `nurse_id, shift_start, shift_end, shift_type, assigned_beds, certifications`

#### **GET /api/schedules/nurses**
- **Query**: `?date=2026-04-10&status=AVAILABLE`

---

### 2.3 Resource Dashboard Endpoint

#### **GET /api/resources/summary**
- **Auth**: JWT required
- **Returns**:
```json
{
  "facility": "City Hospital",
  "beds": {
    "total": 100,
    "available": 8,
    "occupied": 72,
    "maintenance": 10,
    "cleaning": 10
  },
  "physicians": {
    "total": 15,
    "onDuty": 8,
    "onCall": 2,
    "off": 5,
    "bySpecialty": {
      "Cardiology": 2,
      "ER": 4,
      "others": 2
    }
  },
  "nurses": {
    "total": 30,
    "available": 5,
    "assigned": 20,
    "onBreak": 3,
    "off": 2
  },
  "performances": {
    "averageWaitTime": "15 minutes",
    "activeEncounters": 12,
    "computeMode": "RAG_FULL",
    "lastRAGUpdate": "2026-04-10T09:30:00Z"
  }
}
```

---

## 3. System Flow Schematics

### 3.1 Current Triage Flow
```
Frontend (TriageBoard.jsx)
  ↓ POST /api/triage {patient, vitals}
Backend (triage.controller.js)
  ↓ Hard-coded to always use RAG
Python Worker (ESI_Engine.py)
  ↓ ALWAYS runs full Graph-RAG pipeline
  ↓ Returns {ESI, specialty, ragCases}
Frontend Display
  ↓ Shows results
```

**Problem**: No mode flexibility, no per-user settings

---

### 3.2 Proposed Triage Flow (WITH COMPUTE MODES)
```
Frontend (TriageBoard.jsx)
  ↓ Reads user.computeMode from Redux/Context
  ↓ POST /api/triage {patient, vitals, computeMode}
  
Backend (triage.controller.js)
  ↓ Gets user settings from JWT
  ↓ Builds TriageSettings record
  ↓ POST http://localhost:5000/triage {
      patient, vitals,
      compute_mode: "LLM_ONLY" | "RAG_FULL" | "HYBRID",
      rag_params: {...},
      llm_params: {...}
    }
  
Python Worker (ESI_Engine.py)
  ↓ IF compute_mode == "LLM_ONLY":
      → Call MedGemma directly (fast, no embeddings)
      → Return {ESI, specialty, reasoning}
  ↓ ELIF compute_mode == "RAG_FULL":
      → Run full Graph-RAG pipeline (current behavior)
      → Return {ESI, specialty, ragCases, reasoning}
  ↓ ELIF compute_mode == "HYBRID":
      → Get top RAG results + LLM reasoning
      → Blend with ragWeight parameter (0.6 default)
      → Return {ESI, specialty, ragCases, reasoning, blendedScore}
  
Frontend Display
  ↓ Shows mode + results
```

---

### 3.3 Hot-Reload Flow (Physician Schedule Update)
```
Admin uploads CSV via Dashboard
  ↓ POST /api/schedules/physicians/upload {file}
  
Backend (schedule.controller.js)
  ↓ Parse CSV
  ↓ DELETE FROM PhysicianSchedule WHERE facilityId=X, shiftDate=2026-04-10
  ↓ INSERT new rows from CSV
  ↓ Return {recordsImported: 15, errors: [...]}
  ↓ NO SERVICE RESTART NEEDED
  
TriageBoard (Real-time)
  ↓ Next time running triage:
      → GET /api/schedules/physicians?date=2026-04-10&specialty=Cardiology
      → Retrieves FRESH data from DB (updated 1 second ago)
      → Auto-assigns specialist based on availability
  ↓ Result: Specialist assignment reflects latest CSV

```

**Key**: No restart, no cache invalidation, just database updates

---

## 4. Frontend Component Schematics

### 4.1 Current Frontend Structure
```
src/
  Dashboard.jsx (main router)
    ├─ TriageBoard.jsx (read-only queue)
    └─ PatientSummary/ (patient details)
```

**Problem**: No settings UI, no resource management

---

### 4.2 Proposed Frontend Structure
```
src/
  Dashboard.jsx (main router)
    ├─ TriageBoard.jsx (MODIFIED - adds mode selector)
    ├─ SettingsPanel.jsx (NEW - user preferences)
    ├─ ScheduleManager.jsx (NEW - CSV uploads)
    ├─ ResourceDashboard.jsx (NEW - real-time beds/physicians/nurses)
    ├─ PatientSummary.jsx
    └─ hooks/
        ├─ useSettings.js (NEW - GET/PATCH /api/settings/user)
        ├─ useResourceDashboard.js (NEW - polling /api/resources/summary)
        └─ useSchedules.js (NEW - GET /api/schedules/*)
```

---

### 4.3 Component Specifications

#### **SettingsPanel.jsx** (NEW)
```jsx
Props:
  user: {id, name, computeMode, preferences}
  onSave: (settings) => void

Features:
  - Radio buttons: LLM-Only | RAG-Full | Hybrid
  - Toggle: Enable GPU
  - Slider: RAG semantic weight (0-1)
  - Select: Specialty preferences
  - Save button (PATCH /api/settings/user)
```

#### **ScheduleManager.jsx** (NEW)
```jsx
Props:
  scheduleType: "physicians" | "beds" | "nurses"
  facilityId: string

Features:
  - Drag-drop CSV upload area
  - Preview of CSV before upload
  - Progress bar during upload
  - Error display (rows rejected)
  - Timestamp of last upload
  - POST /api/schedules/{type}/upload
```

#### **ResourceDashboard.jsx** (NEW)
```jsx
Features:
  - Real-time bed occupancy grid (color-coded)
  - Physician roster table with specialties
  - Nurse availability breakdown
  - Polling GET /api/resources/summary every 5s
  - Quick-assign patient to bed
```

#### **TriageBoard.jsx** (MODIFIED)
```jsx
Changes:
  - Add mode selector at top (LLM-Only | RAG | Hybrid)
  - Store mode in Redux/Context
  - Pass computeMode in triage request
  - Display compute mode in results
  - Show response time metrics
```

---

## 5. Python Backend Schematics (ESI_Engine.py)

### 5.1 Current ESI_Engine Structure
```python
@app.route('/triage', methods=['POST'])
def triage():
    payload = request.json
    patient_text = payload['patient_text']
    vitals = payload['vitals']
    
    # ALWAYS does RAG pipeline
    results = rag_pipeline_triage(patient_text, vitals)
    return results
```

**Problem**: No mode detection or routing

---

### 5.2 Proposed ESI_Engine Structure
```python
@app.route('/triage', methods=['POST'])
def triage():
    payload = request.json
    compute_mode = payload.get('compute_mode', 'RAG_FULL')
    patient_text = payload['patient_text']
    vitals = payload['vitals']
    rag_params = payload.get('rag_params', {})
    llm_params = payload.get('llm_params', {})
    
    if compute_mode == 'LLM_ONLY':
        return llm_only_triage(patient_text, vitals, llm_params)
    elif compute_mode == 'RAG_FULL':
        return rag_full_triage(patient_text, vitals, rag_params)
    elif compute_mode == 'HYBRID':
        return hybrid_triage(patient_text, vitals, rag_params, llm_params)

def llm_only_triage(patient_text, vitals, llm_params):
    """Fast LLM-only (no RAG retrieval)"""
    response = medgemma_reasoning(
        patient_text, 
        vitals,
        temperature=llm_params.get('temperature', 0.3),
        max_tokens=llm_params.get('max_tokens', 512)
    )
    return {
        'ESI': response['esi'],
        'specialty': response['specialty'],
        'reasoning': response['reasoning'],
        'mode': 'LLM_ONLY',
        'processingTime': response['elapsed_ms']
    }

def rag_full_triage(patient_text, vitals, rag_params):
    """Full Graph-RAG pipeline (current behavior)"""
    # ... existing RAG logic ...
    return {
        'ESI': esi_level,
        'specialty': specialty,
        'ragCases': rag_results,
        'reasoning': llm_reasoning,
        'mode': 'RAG_FULL',
        'processingTime': elapsed_ms
    }

def hybrid_triage(patient_text, vitals, rag_params, llm_params):
    """Blend RAG + LLM with weighted scoring"""
    rag_weight = rag_params.get('ragWeight', 0.6)
    llm_weight = 1.0 - rag_weight
    
    rag_esi = rag_full_triage(...)['ESI']
    llm_esi = llm_only_triage(...)['ESI']
    
    blended_esi = (rag_esi * rag_weight) + (llm_esi * llm_weight)
    
    return {
        'ESI': blended_esi,
        'specialty': highest_confidence_specialty,
        'ragCases': rag_results,
        'reasoning': combined_reasoning,
        'mode': 'HYBRID',
        'ragWeight': rag_weight,
        'processingTime': elapsed_ms
    }
```

---

## 6. Database Schema Changes Summary

| Model | Current | Status | Changes |
|-------|---------|--------|---------|
| Nurse/User | Minimal auth | EXTEND | Add computeMode, facilityId, preferences |
| TriageSettings | N/A | NEW | Store per-user RAG+LLM settings |
| PhysicianSchedule | N/A | NEW | Daily physician roster hot-reload |
| BedSchedule | N/A | NEW | Real-time bed availability |
| NurseSchedule | N/A | NEW | Daily nurse roster hot-reload |
| Facility | N/A | NEW | Centralized config + defaults |

---

## 7. Security & Validation Schematics

### 7.1 Role-Based Access Control
```
GET /api/settings/user
  ✅ Anyone can view own settings (JWT validated)
  ❌ Cannot view other users' settings

PATCH /api/settings/user
  ✅ User can update own computeMode
  ❌ User cannot override facility defaults

GET/PATCH /api/settings/facility
  ✅ ADMIN only
  ✅ PHYSICIAN with admin flag
  ❌ NURSE not allowed

POST /api/schedules/physicians/upload
  ✅ ADMIN only
  ❌ Facilities cannot upload schedules for other facilities (facilityId validated)
```

### 7.2 Input Validation
```
computeMode: enum(LLM_ONLY, RAG_FULL, HYBRID) - strict validation
ragParams.semanticWeight: number 0.35-0.85 - range validation
ragParams.topKResults: number 1-20 - reasonable limits
CSV upload: max 1000 rows, required columns checked
```

---

## 8. Implementation Sequence (when approved)

1. **Phase 1**: Create database models (Facility, User ext, TriageSettings, 3x Schedule models)
2. **Phase 2**: Create API controllers (settings.controller.js, schedule.controller.js)
3. **Phase 3**: Create API routes in Express (POST/GET/PATCH endpoints)
4. **Phase 4**: Refactor ESI_Engine.py (add compute_mode parameter + 3 functions)
5. **Phase 5**: Create frontend components (SettingsPanel, ScheduleManager, ResourceDashboard)
6. **Phase 6**: Update TriageBoard to use settings
7. **Phase 7**: Add CSV parsing + hot-reload logic
8. **Phase 8**: End-to-end testing
9. **Phase 9**: Deployment

---

## 9. Questions for Approval

1. **User Model**: Rename "Nurse" to "User" or keep separate?
2. **Facility Management**: Single facility or multi-facility support?
3. **Compute Mode Defaults**: Should facility admins force a mode on all users, or is it per-user preference?
4. **HYBRID Mode Weight**: User-configurable (0.0-1.0 rag_weight) or fixed?
5. **CSV Update Frequency**: Daily? Hourly? On-demand only?
6. **Specialist Assignment**: Should triage auto-assign based on physician availability from schedule?
7. **GPU Detection**: Should Python auto-detect GPU or rely on config?

---

**Ready to implement when you approve!**
