# SCHEMA ARCHITECTURE DIAGNOSIS
## Complete Analysis of Data Structures & Patient Journey

**Status**: 🔍 DIAGNOSTIC ANALYSIS - Comprehensive schema review  
**Date**: 20 April 2026  
**Focus**: How shifts end, staff register, and patients flow through state changes

---

## PART 1: CURRENT MODELS INVENTORY

### ✅ EXISTING MODELS (21 files identified)

```
├─ Shift & Scheduling
│  ├─ shift.model.js (5️⃣ CORE) - Shift creation + staff assignments
│  ├─ nurseSchedule.model.js - Nurse availability schedules
│  ├─ physicianSchedule.model.js - Physician availability
│  └─ nurseClockRecord.model.js (5️⃣ CORE) - Clock in/out tracking
│
├─ Staff
│  ├─ nurse.model.js (5️⃣ CORE) - Nurse/Physician registration + authentication
│  ├─ nurseProfile.model.js - Extended nurse profile data
│  ├─ nurseAssignment.model.js (4️⃣ IMPORTANT) - Patient assignment per shift
│  ├─ manager.model.js - Hospital admin/manager record
│  └─ ward.model.js (5️⃣ CORE) - Ward/department structure
│
├─ Patient & EHR
│  ├─ patientEHRRecord.model.js (5️⃣ CORE) - Main patient record with vitals & status
│  ├─ triageRequest.model.js (5️⃣ CORE) - Triage assessment & AI reports
│  ├─ encounter.model.js (5️⃣ CORE) - Patient encounter (clinical episode)
│  ├─ vitals.model.js (4️⃣ IMPORTANT) - Vital signs time series
│  ├─ handoffReport.model.js - Shift handoff (when shifts end)
│  ├─ nurseConfirmedReport.model.js - Nurse final assessment
│  ├─ aiSuggestedReport.model.js - AI preliminary findings
│  └─ interaction.model.js - Patient/staff interactions
│
├─ Bed Management
│  ├─ bedSchedule.model.js (5️⃣ CORE) - Individual bed status & occupancy
│  ├─ bedStatusHistory.model.js (4️⃣ IMPORTANT) - History of bed status changes
│  └─ hospital.model.js (5️⃣ CORE) - Hospital structure
│
└─ Monitoring
   ├─ acuityAlert.model.js - Alert system for vital derangements
   └─ [MISSING] - ECG/Continuous waveform model
```

---

## PART 2: STAFF REGISTRATION & SHIFT ASSIGNMENT

### 2.1 STAFF MEMBER REGISTRATION (How It Works)

#### Step 1: Nurse Model (Base)
```javascript
// backend/src/models/nurse.model.js

const nurseSchema = {
  hospitalId: ObjectId,          // Which hospital
  name: String,                  // First name
  surname: String,               // Last name
  email: String (indexed),       // Login email
  password: String (bcrypt),     // Encrypted password
  
  // Role & Department
  role: {
    enum: ['NURSE', 'PHYSICIAN', 'SPECIALIST'],
    required: true
  },
  
  department: {
    enum: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'General'],
    default: 'General'
  },
  
  specialty: String,             // e.g., "Emergency Nursing", "Critical Care"
  licenseNumber: String,         // Professional license
  isActive: Boolean,             // Employment status
  
  dateCreated: Date,
  timestamps: { created, updated }
};
```

**Key Insight**: 
- ❌ **MISSING**: `operationalContext` field (this is what we need to add!)
- ❌ **MISSING**: `maxPatientLoad` (how many patients can this nurse handle?)
- ❌ **MISSING**: `certifications` array (advanced life support, trauma, etc.)

---

#### Step 2: Shift Model (Assignment)
```javascript
// backend/src/models/shift.model.js

const shiftSchema = {
  hospitalId: ObjectId,
  shiftID: String,
  shiftName: String,
  
  // Time & Recurrence
  date: Date,
  shiftStart: String,            // "07:00"
  shiftEnd: String,              // "15:00"
  duration: Number,              // 8 hours
  dayOfWeek: String,
  recurrence: {
    enum: ['ONCE', 'DAILY', 'WEEKLY', 'BIWEEKLY', 'MONTHLY'],
    default: 'ONCE'
  },
  
  // Department & Staff
  department: {
    enum: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry'],
    required: true
  },
  
  maxStaffRequired: Number,
  
  assignedStaff: [{
    staffID: ObjectId (ref: Nurse),
    type: {
      enum: ['MANAGER', 'NURSE', 'PHYSICIAN', 'SPECIALIST', 'SUPPORT']
    },
    name: String,
    surname: String,
    specialty: String,
    status: {
      enum: ['ASSIGNED', 'CONFIRMED', 'ABSENT', 'PENDING'],
      default: 'ASSIGNED'
    }
  }],
  
  status: {
    enum: ['DRAFT', 'PUBLISHED', 'CANCELLED', 'COMPLETED'],
    default: 'DRAFT'
  },
  createdBy: ObjectId (ref: Manager)
};
```

**Key Insight**:
- ❌ **MISSING**: `operationalContext` for each assigned staff member
- ❌ **MISSING**: Explicit field to track shift status → shift end
- ✅ **GOOD**: Has `status: COMPLETED` for when shift ends
- ❌ **MISSING**: Handover validation (when previous nurse leaves)

---

#### Step 3: Clock In/Out (NurseClockRecord)
```javascript
// backend/src/models/nurseClockRecord.model.js

const clockRecordSchema = {
  nurseId: ObjectId (ref: Nurse),
  shiftId: ObjectId (ref: Shift),
  date: Date,
  
  // Clock times
  clockInTime: Date,
  clockOutTime: Date (null while active),
  
  // Break tracking
  breakStartTime: Date,
  breakEndTime: Date,
  
  status: {
    enum: ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT', 'ABSENT'],
    default: 'CLOCKED_IN'
  },
  
  location: String,              // 'ED', 'ICU', etc.
  
  // Calculated metrics
  totalShiftDuration: Number,    // milliseconds
  totalBreakTime: Number,
  totalWorkTime: Number,
  
  // Metadata
  metadata: {
    deviceType: String,
    ipAddress: String,
    location: { latitude, longitude }
  },
  
  // Break approval
  breakApprovalStatus: {
    enum: ['NOT_REQUESTED', 'PENDING', 'APPROVED', 'REJECTED'],
    default: 'NOT_REQUESTED'
  }
};
```

**Key Insight**:
- ✅ **GOOD**: Tracks when staffmember clocks out
- ❌ **MISSING**: `operationalContextAtClockIn` (what was their context?)
- ❌ **MISSING**: Validation that all assigned patients are handed over before clock-out

---

### 2.2 THE STAFF REGISTRATION FLOW

```
WORKFLOW: Hospital Admin registers new nurse

┌─────────────────────────────────────┐
│ Admin: Alice Johnson (Manager)       │  
│ Action: "Add Staff Member"           │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ CREATE in nurse.model.js:            │
│ ├─ name: "Bob"                      │
│ ├─ surname: "Martinez"              │
│ ├─ email: "bob@stmercy.hospital"    │
│ ├─ password: bcrypt("bob_123")      │
│ ├─ role: "NURSE"                    │
│ ├─ department: "ED"                 │
│ ├─ specialty: "General"             │
│ ├─ licenseNumber: "LIC-001"         │
│ ├─ isActive: true                   │
│ ├─ createdBy: alice_id              │
│ ├─ [❌ MISSING: operationalContext] │
│ └─ [❌ MISSING: certifications]     │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Admin: "Create Shift"                │
│ Date: 2026-04-20                     │
│ Time: 07:00 - 15:00                  │
│ Department: ED                       │
│ Max Staff: 4                         │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ CREATE in shift.model.js:            │
│ ├─ shiftName: "Tuesday ED Shift"    │
│ ├─ date: 2026-04-20                 │
│ ├─ shiftStart: "07:00"              │
│ ├─ shiftEnd: "15:00"                │
│ ├─ assignedStaff: [                 │
│ │  {                                 │
│ │    staffID: bob_id,               │
│ │    type: "NURSE",                 │
│ │    specialty: "General",          │
│ │    status: "ASSIGNED"             │
│ │    [❌ MISSING: operationalContext]
│ │  }                                 │
│ │]                                   │
│ └─ status: "PUBLISHED"              │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Bob logs in                          │
│ Email: bob@stmercy.hospital         │
│ Password: bob_123                    │
│ System: Authenticates + generates   │
│ JWT token                            │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Bob clicks "Clock In"                │
│ System creates NurseClockRecord:    │
│ ├─ nurseId: bob_id                  │
│ ├─ shiftId: shift_id                │
│ ├─ clockInTime: NOW                 │
│ ├─ status: "CLOCKED_IN"             │
│ └─ location: "ED"                   │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ [❌ PROBLEM HERE]                    │
│ System doesn't know:                 │
│ ├─ Is Bob triaging?                 │
│ ├─ Is Bob watching ER patients?     │
│ ├─ Is Bob managing beds?            │
│ ├─ Which dashboard should Bob see?  │
│ └─ → Result: Generic "NurseDashboard"
└─────────────────────────────────────┘
```

---

## PART 3: PATIENT CREATION & STATE TRANSITIONS

### 3.1 PATIENT ENTERS SYSTEM (Triage)

```javascript
// Patient arrives at ER
// STEP 1: Sarah (Triage Nurse) enters patient data

// CREATE in triageRequest.model.js:
{
  encounterId: "ENC-2026-04-20-0001",
  
  // Who is triaging?
  triageNurseId: sarah_id,
  
  // Patient vitals (manual entry)
  manualTriageData: {
    symptoms: ["chest pain", "shortness of breath"],
    vitalSigns: {
      temperature: 37.2,
      heartRate: 102,
      respiratoryRate: 20,
      systolicBP: 158,
      diastolicBP: 92,
      oxygenSaturation: 96
    },
    chiefComplaint: "Acute chest pain",
    medicalHistory: ["Hypertension"],
    medications: ["Lisinopril"],
    allergies: ["NKDA"]
  },
  
  // Manual triage assessment
  manualTriageReport: {
    esiLevel: 2,
    estimatedAcuity: 85,
    reasoning: "Acute chest pain, elevated vitals, possible ACS",
    recommendedDepartment: "ED",
    immediateInterventions: ["12-lead ECG", "Troponin draw"],
    createdBy: sarah_id,
    createdAt: NOW
  },
  
  // AI processing (if enabled)
  computeMode: 'HYBRID',
  aiTriageReport: {
    esiLevel: 2,
    estimatedAcuity: 88,
    confidence: 0.92,
    reasoning: "...",
    recommendedPhysician: { id, name, specialty },
    recommendedNurse: { id, name, specialty }
  },
  
  // ❌ MISSING in current schema
  needsBed: UNDEFINED,  // Does patient need hospitalization?
  
  status: 'TRIAGED' // What status indicates this?
}

// PROBLEM: triageRequest.model.js doesn't have explicit needsBed field!
```

---

### 3.2 PATIENT WAITS FOR CLINICAL ATTENTION

```javascript
// STEP 2: Patient enters queue, waiting for Clinical Nurse

// CREATE in encounter.model.js:
{
  nurse_id: sarah_id,           // Who triaged?
  assignedNurseId: bob_id,      // Who is now caring?
  assignedPhysicianId: chen_id, // Who will review?
  
  patient: {
    id_number: "P-001",
    name: "John",
    surname: "Smith",
    gender: "M",
    age: 59,
    symptoms: "chest pain",
    painLevel: "8/10",
    history: "Hypertension"
  },
  
  vitals: {
    temp: 37.2,
    hr: 102,
    BP: "158/92",
    o2: 96,
    resp: 20
  },
  
  status: "pending",    // ← PATIENT WAITING FOR CLINICAL NURSE
  severity: 2,          // High priority
  isWaiting: true,      // Still in queue
  submittedAt: NOW,
  attendedAt: null      // Not yet attended
}

// SAME PROBLEM: encounters.status is vague
// "pending" = What exactly? Waiting for nurse? Waiting for physician?
```

---

### 3.3 CLINICAL NURSE RECEIVES PATIENT

```javascript
// STEP 3: Bob (Clinical Nurse) receives John

// UPDATE encounter.model.js:
{
  // ... same as above, but now:
  assignedNurseId: bob_id,
  status: "confirmed",          // Nurse confirmed receipt
  attendedAt: NOW,              // Nurse now caring
  
  // Bob logs initial vitals
  vitals: {
    temp: 37.1,
    hr: 104,
    BP: "160/94",
    o2: 95,
    resp: 22
  }
  
  // ❌ MISSING: What's the next step?
  // ❌ MISSING: Should this patient go to a bed?
  // ❌ MISSING: Is Bob just observing or admitting?
}

// CREATE in vitals.model.js (time series):
{
  encounter_id: enc_id,
  temperature: 37.1,
  heart_rate: 104,
  blood_pressure_systolic: 160,
  blood_pressure_diastolic: 94,
  oxygen_saturation: 95,
  resp: 22,
  timestamps: NOW
}

// Bob logs periodic vitals every 15 minutes
// Vitals table grows with new entries
```

---

### 3.4 BOB ESCALATES TO PHYSICIAN

```javascript
// STEP 4: Bob sees vitals deteriorating, escalates to Dr. Chen

// UPDATE encounter.model.js:
{
  status: "escalated",  // ← NEW STATUS INDICATING PHYSICIAN REVIEW NEEDED
  escalatedAt: NOW,
  escalationReason: "HR increased from 102 to 110, BP elevated"
}

// PROBLEM: Current schema doesn't have "escalated" status
// Current enums: ["pending", "confirmed", "completed", "unassigned"]
// This is incomplete!
```

---

### 3.5 PHYSICIAN REVIEW & DISPOSITION DECISION

```javascript
// STEP 5: Dr. Chen reviews case

// UPDATE encounter.model.js (or create new dispositionDecision.model.js):
{
  // Review data
  assignedPhysicianId: chen_id,
  physicianReviewTime: NOW,
  physicianDecision: {
    disposition: "ADMIT",           // DISCHARGE | ADMIT | REFER | OBSERVE
    
    // Critical: Compare triage vs physician assessment
    triageNeedsBedPrediction: true,  // Sarah said: YES, needs bed
    physicianDecision_NeedsBed: true,// Chen says: YES, needs admitting
    
    // Accuracy tracking
    triageAccuracy: "MATCH",         // TRUE_POSITIVE
    
    // If admitted: which bed?
    admittingWard: "ICU",
    admittingBedId: "ICU-05",
    admittingReason: "Troponin positive, ACS confirmed",
    estimatedLOS: "3-5 days"
  }
}

// ❌ CRITICAL MISSING MODEL: dispositionDecision.model.js!
// ❌ MISSING: How does the system track "DISCHARGE" vs "ADMIT"?
```

---

### 3.6 PATIENT ADMITTED TO BED (State Change)

```
TRANSITION POINT: ER → WARD (Hospitalization)

BEFORE Admission:
  - Patient in encounter.model.js (status: "pending/confirmed")
  - No bed assigned
  - Managed by ER team (Bob, Sarah, Chen)

ADMISSION DECISION:
  Dr. Chen decides: ADMIT to ICU bed
  
DURING Admission:
  - bedSchedule.model.js updated: ICU-05 status = "OCCUPIED"
  - bedStatusHistory.model.js created: Log state change
  - encounter.model.js status = "TRANSFERRED" or "ADMITTED"
  - NEW: Ward encounter created or opened
  
AFTER Admission:
  - Patient now in patientEHRRecord.model.js
  - currentBedId: "ICU-05"
  - assignedNurseId: ward_nurse_id
  - location: "ICU Ward A, Bed 5"
  - Patient appears in Ward Nurse dashboard
  - Vitals still logged to vitals.model.js (continuous)
```

---

## PART 4: COMPLETE PATIENT STATE MACHINE

### 4.1 STATE DIAGRAM (WHAT WE NEED)

```
PATIENT LIFECYCLE:

┌──────────────────────────────────────────────────────────┐
│                    ARRIVAL STATE MACHINE                  │
└──────────────────────────────────────────────────────────┘

1. WAITING_TRIAGE
   ├─ Patient arrived at ED
   ├─ Waiting for Sarah (Triage Nurse)
   ├─ Location: Waiting room
   ├─ System: TriageRequest.status = "PENDING"
   └─ Duration: <5 minutes (target)
        ↓
2. TRIAGING
   ├─ Sarah assesses patient
   ├─ Enters vitals, chief complaint, medical history
   ├─ Makes ESI level decision
   ├─ ❌ DECISION POINT: needsBed: true | false
   ├─ System: TriageRequest.status = "TRIAGED"
   └─ Duration: 3-10 minutes
        ↓
3. AWAITING_CLINICAL_NURSE
   ├─ Patient triaged, assigned to Bob (clinical nurse)
   ├─ Location: Observation area / ER bed
   ├─ System: Encounter.status = "CONFIRMED", Vitals logged every 15 min
   ├─ Bob monitors for changes
   └─ Duration: 15-60 minutes (depends on severity)
        ↓
   [BRANCH: DETERIORATION?]
   │
   ├─ NO: Status stable → continue observation
   │   (vitals logged periodically)
   │
   └─ YES: Bob escalates immediately
        ↓
4. AWAITING_PHYSICIAN
   ├─ Patient escalated to Dr. Chen for review
   ├─ Dr. Chen reviews: Triage data + Vital trends + Clinical notes
   ├─ Dr. Chen makes disposition decision:
   │
   ├─────────── DECISION GATE ──────────────
   │
   ├─ DISCHARGE (GREEN/YELLOW case)
   │  └─ Go to state: DISCHARGED
   │
   ├─ ADMIT (RED case or deterioration)
   │  └─ Go to state: ADMITTED_TO_WARD
   │
   └─ REFER (Specialist needed)
      └─ Go to state: AWAITING_SPECIALIST
        ↓
5a. DISCHARGE (Terminal)
    ├─ Dr. Chen signs discharge order
    ├─ Patient given discharge instructions
    ├─ System: Encounter.status = "COMPLETED"
    ├─ Patient record moved to archive
    └─ End: Patient leaves hospital
        
5b. ADMITTED_TO_WARD (Long-term state)
    ├─ Bed assigned (ICU-05)
    ├─ Patient transferred from ER to ward
    ├─ System: bedSchedule[ICU-05].status = "OCCUPIED"
    ├─ System: patientEHRRecord.currentBedId = "ICU-05"
    ├─ Ward Nurse assigned (takes over from Bob)
    ├─ Vitals logged continuously (every hour)
    ├─ Duration: 24 hours to weeks
    │
    ├─ During hospitalization:
    │  ├─ Ward Nurse logs vitals every 1 hour
    │  ├─ Medications administered per order
    │  ├─ Lab results checked
    │  ├─ Specialist consultations as needed
    │  └─ On-Call Physician monitors for alerts
    │
    └─ Eventually:
       ├─ Improvement: Ready for discharge
       ├─ Deterioration: ICU escalation
       └─ Complication: Specialist involvement
         ↓
5c. DISCHARGE_PENDING
    ├─ Physician orders discharge
    ├─ Patient education completed
    ├─ Follow-up appointments scheduled
    └─ Status ready for physical discharge
         ↓
5d. DISCHARGED (Terminal)
    ├─ Patient physically leaves hospital
    ├─ System: bedSchedule[ICU-05].status = "CLEANING"
    ├─ System: Encounter.status = "CLOSED"
    ├─ Ward Nurse documents discharge summary
    └─ End: Patient leaves hospital
```

---

### 4.2 CURRENT SCHEMA STATUS vs. STATE MACHINE

| State | Currently Tracked? | Model | Issue |
|-------|-------------------|-------|-------|
| WAITING_TRIAGE | ❌ NO | N/A | No queue tracking |
| TRIAGING | ✅ PARTIAL | TriageRequest | Missing `needsBed` decision |
| AWAITING_CLINICAL_NURSE | ✅ PARTIAL | Encounter | Status "pending" is ambiguous |
| AWAITING_PHYSICIAN | ❌ NO | N/A | No escalation tracking |
| DISCHARGE | ❌ NO | N/A | No disposition model |
| ADMITTED_TO_WARD | ✅ GOOD | patientEHRRecord + BedSchedule | But handover missing |
| DISCHARGE_PENDING | ❌ NO | N/A | No pre-discharge tracking |
| DISCHARGED | ✅ PARTIAL | Encounter | "completed" status vague |

---

## PART 5: BED MANAGEMENT & VITALS TRACKING

### 5.1 BED LIFECYCLE

```javascript
// backend/src/models/bedSchedule.model.js

BEDS IN HOSPITAL:

Ward: "ICU Ward A"
├─ Bed 01 (ICU-01): OCCUPIED
│  ├─ currentEncounterId: enc_001
│  ├─ assignedNurseId: ward_nurse_id
│  ├─ occupancyStartTime: 2026-04-20 07:53 (when admitted)
│  ├─ expectedDischargeTime: null (open-ended)
│  ├─ acuityLevel: "CRITICAL"
│  ├─ hasMonitor: true
│  ├─ hasVentilator: false
│  ├─ bedType: "INTENSIVE"
│  └─ status: "OCCUPIED"
│
├─ Bed 02 (ICU-02): AVAILABLE
│  ├─ status: "AVAILABLE"
│  ├─ acuityLevel: "MODERATE"
│  └─ lastCleanedAt: 2026-04-20 06:30
│
├─ Bed 03 (ICU-03): CLEANING
│  ├─ status: "CLEANING"
│  ├─ statusChangedAt: 2026-04-20 07:45 (when patient discharged)
│  ├─ cleanedByStaffId: housekeeper_001
│  ├─ cleaningDurationMinutes: 30
│  └─ estimatedReadyTime: 2026-04-20 08:15 (30 min cleaning)
│
└─ Bed 04 (ICU-04): MAINTENANCE
   ├─ status: "MAINTENANCE"
   ├─ maintenanceNotes: "Monitor cable replacement"
   └─ estimatedReadyTime: 2026-04-20 14:00
```

### 5.2 BED STATUS TRANSITIONS

```javascript
// backend/src/models/bedStatusHistory.model.js

// Tracks EVERY bed status change:

Event 1: 07:53 AM - Patient Admitted
{
  bedId: icu_05_id,
  bedIdString: "ICU-05",
  previousStatus: "AVAILABLE",
  currentStatus: "OCCUPIED",
  reason: "PATIENT_ADMITTED",
  changedBy: chen_id,
  changedByName: "Dr. Chen",
  timestamp: 2026-04-20T07:53:00Z,
  encounterId: john_encounter_id,
  notes: "John Smith - ACS admission"
}

Event 2: 14:30 PM - Patient Discharged
{
  bedId: icu_05_id,
  bedIdString: "ICU-05",
  previousStatus: "OCCUPIED",
  currentStatus: "CLEANING",
  reason: "PATIENT_DISCHARGED",
  changedBy: ward_nurse_id,
  timestamp: 2026-04-20T14:30:00Z,
  durationMinutes: 390  // How long patient was in bed
}

Event 3: 15:00 PM - Cleaning Complete
{
  bedId: icu_05_id,
  bedIdString: "ICU-05",
  previousStatus: "CLEANING",
  currentStatus: "AVAILABLE",
  reason: "CLEANING_COMPLETED",
  changedBy: housekeeper_id,
  timestamp: 2026-04-20T15:00:00Z
}
```

---

### 5.3 VITALS TRACKING (Time Series)

```javascript
// backend/src/models/vitals.model.js

// Every vital check creates a new record:

Patient: John Smith (ICU-05)
Vitals logged by Ward Nurse every 1 hour:

[VITALS ENTRY 1] 08:00 AM
{
  encounter_id: john_enc_id,
  temperature: 37.1,
  blood_pressure_systolic: 158,
  blood_pressure_diastolic: 92,
  heart_rate: 102,
  oxygen_saturation: 96,
  resp: 20,
  createdAt: 2026-04-20T08:00:00Z
}

[VITALS ENTRY 2] 09:00 AM
{
  encounter_id: john_enc_id,
  temperature: 37.0,
  blood_pressure_systolic: 155,
  blood_pressure_diastolic: 90,
  heart_rate: 100,
  oxygen_saturation: 97,
  resp: 18,
  createdAt: 2026-04-20T09:00:00Z,
  note: "Patient more comfortable, pain reduced"
}

[VITALS ENTRY 3] 10:00 AM
{
  encounter_id: john_enc_id,
  temperature: 36.9,
  blood_pressure_systolic: 152,
  blood_pressure_diastolic: 88,
  heart_rate: 98,
  oxygen_saturation: 98,
  resp: 16,
  createdAt: 2026-04-20T10:00:00Z,
  note: "ECG done, troponin results: POSITIVE (NSTEMI confirmed)"
}
```

---

### 5.4 PATIENT EHR RECORD INTEGRATION

```javascript
// backend/src/models/patientEHRRecord.model.js

// MASTER PATIENT RECORD (during hospitalization)

{
  encounterId: "ENC-2026-04-20-0001",
  patientId: "P-001",
  name: "John",
  surname: "Smith",
  age: 59,
  
  // Current Location
  currentBedId: "ICU-05",
  currentDepartment: "ICU",
  admissionTime: 2026-04-20T07:53:00Z,
  
  // Status
  esiLevel: 2,
  acuityScore: 85,
  status: "ADMITTED",  // ✅ Has this
  
  // Latest Vitals (snapshot)
  latestVitals: {
    temperature: 36.9,
    heartRate: 98,
    systolicBP: 152,
    diastolicBP: 88,
    respiratoryRate: 16,
    o2Saturation: 98,
    recordedAt: 2026-04-20T10:00:00Z
  },
  
  // Vital Thresholds (triggers alert if violated)
  vitalThresholds: {
    hr_min: 60,
    hr_max: 100,
    bp_systolic_min: 90,
    bp_systolic_max: 160,
    resp_min: 12,
    resp_max: 20
  },
  
  // Clinical Orders
  activeOrders: [
    {
      orderId: "ORD-001",
      orderType: "medication",
      description: "Aspirin 500mg + Clopidogrel 600mg loading dose",
      orderedAt: 2026-04-20T08:00:00Z
    },
    {
      orderId: "ORD-002",
      orderType: "lab",
      description: "Troponin I levels (serial)",
      orderedAt: 2026-04-20T07:55:00Z
    }
  ],
  
  // Alert System
  surveillanceFlags: [
    {
      flag: "cardiac_arrest_risk",
      severity: "high",
      reason: "NSTEMI, frequent PVCs on monitor",
      flaggedAt: 2026-04-20T10:05:00Z
    }
  ],
  
  // Assignments
  assignedNurseId: ward_nurse_id,
  assignedPhysicianId: chen_id,
  assignmentStartTime: 2026-04-20T07:53:00Z,
  
  // Trend Analysis
  vitalTrendDirection: "stable",  // 'stable', 'improving', 'deteriorating'
  lastVitalUpdateAt: 2026-04-20T10:00:00Z
}

// ❌ MISSING FIELDS:
// - ecgData: { waveform, results, timestamp }
// - continuousMonitoring: { enabled, alerts }
// - bedTimelineEvents: [] // Complete bed timeline
```

---

## PART 6: MISSING MODELS & FIELDS

### 6.1 CRITICAL MISSING: Patient State Tracker

```javascript
// ❌ MISSING from current schema

// NEW MODEL NEEDED: patientState.model.js
{
  encounterId: String,
  patientId: String,
  
  // Current state in journey
  currentState: {
    enum: [
      'WAITING_TRIAGE',
      'TRIAGING',
      'AWAITING_CLINICAL_NURSE',
      'AWAITING_PHYSICIAN',
      'DISPOSITION_PENDING',
      'ADMITTED_TO_WARD',
      'DISCHARGE_PENDING',
      'DISCHARGED',
      'REFERRED'
    ],
    required: true,
    index: true
  },
  
  // State timeline (audit trail)
  stateHistory: [{
    state: String,
    enteredAt: Date,
    exitedAt: Date,
    duration: Number // seconds
  }],
  
  // Transitions and timestamps
  transitions: {
    triageCompletedAt: Date,
    clinicalNurseAssignedAt: Date,
    physicianReviewStartedAt: Date,
    dispositionDecidedAt: Date,
    admittedToBedAt: Date,
    dischargeOrderedAt: Date,
    physicallyDischargedAt: Date
  }
}
```

---

### 6.2 CRITICAL MISSING: Disposition Decision Model

```javascript
// ❌ MISSING from current schema

// NEW MODEL NEEDED: dispositionDecision.model.js
{
  encounterId: String,
  patientId: String,
  
  // Initial triage prediction
  triageAssessment: {
    nurseName: String,
    needsBedPrediction: Boolean,  // ← KEY: Did triage nurse say YES/NO?
    reasoning: String,
    createdAt: Date
  },
  
  // Physician final decision
  physicianDecision: {
    physicianName: String,
    disposition: {
      enum: ['DISCHARGE', 'ADMIT', 'REFER', 'OBSERVE'],
      required: true
    },
    needsBedActual: Boolean,  // ← KEY: Did physician actually admit?
    reasoning: String,
    decidedAt: Date
  },
  
  // Accuracy tracking (for QA)
  triageAccuracy: {
    enum: ['TRUE_POSITIVE', 'TRUE_NEGATIVE', 'FALSE_POSITIVE', 'FALSE_NEGATIVE'],
    calculated: function() {
      // TRUE_POSITIVE: triage said YES, physician said YES
      // TRUE_NEGATIVE: triage said NO, physician said NO
      // FALSE_POSITIVE: triage said YES, physician said NO
      // FALSE_NEGATIVE: triage said NO, physician said YES ← MARIA'S CASE
    }
  },
  
  // If admitted
  admissionDetails: {
    wardId: String,
    bedId: String,
    admittedAt: Date
  },
  
  // If discharged
  dischargeDetails: {
    dischargeSummary: String,
    dischargeMedications: [String],
    followUpInstructions: String,
    dischargedAt: Date
  }
}
```

---

### 6.3 CRITICAL MISSING: Shift End & Handover

```javascript
// ❌ INCOMPLETE in current schema

// ENHANCEMENT NEEDED: nurseClockRecord.model.js
{
  // ... existing fields
  
  // ❌ MISSING: Handover validation
  handoverDetails: {
    patientsHandedOver: [{
      encounterId: String,
      patientName: String,
      receivingNurseId: String,
      handoverTime: Date,
      handoverNotes: String,
      status: { enum: ['PENDING', 'CONFIRMED', 'REJECTED'] }
    }],
    
    // Can nurse clock out?
    canClockOut: Boolean,  // Validates all patients handed over
    allPatientsHandedOver: Boolean,
    handoverCompletedAt: Date
  }
}

// ❌ MISSING MODEL: shiftEndSummary.model.js
{
  shiftId: String,
  nurseId: String,
  
  // Summary at shift end
  patientsSeenTotal: Number,
  patientsAdmitted: Number,
  patientsDischarged: Number,
  
  // Metrics
  averageTimePerPatient: Number,
  escalationsInitiated: Number,
  adverseEvents: Number,
  
  // Handover report (who takes over)
  incomingNurseId: String,
  handoverCompletedAt: Date,
  
  opsNotes: String  // What happened this shift?
}
```

---

## PART 7: ECG & CONTINUOUS MONITORING

### 7.1 MISSING ECG MODEL

```javascript
// ❌ MISSING from current schema

// NEW MODEL NEEDED: ecgData.model.js
{
  encounterId: String,
  patientId: String,
  
  // ECG timing
  recordedAt: Date,           // When ECG was taken
  recordedBy: String,         // Which nurse/tech
  
  // ECG waveform data
  waveformData: {
    duration: Number,         // seconds (usually 10)
    sampleRate: Number,       // Hz (usually 500)
    leads: {
      I: [Number],           // Array of voltage readings
      II: [Number],
      III: [Number],
      aVR: [Number],
      aVL: [Number],
      aVF: [Number],
      V1: [Number],
      V2: [Number],
      V3: [Number],
      V4: [Number],
      V5: [Number],
      V6: [Number]
    },
    rawWaveform: String       // Binary/base64 encoded raw ECG
  },
  
  // Automated interpretation
  automatedInterpretation: {
    heartRate: Number,
    rhythm: String,           // 'NSR', 'AF', 'VT', etc.
    measurementPoints: {
      pr_interval: Number,    // milliseconds
      qrs_duration: Number,
      qt_interval: Number,
      st_segment: String      // 'normal', 'elevated', 'depressed'
    },
    abnormalFindings: [String],
    clinicalSummary: String
  },
  
  // Physician interpretation
  physicianInterpretation: {
    physicianId: String,
    interpretedAt: Date,
    findings: String,
    clinicalDecision: String,
    isAbnormal: Boolean,
    needsFollowUp: Boolean
  },
  
  // Comparison
  priorECGId: String,         // For trend analysis
  changesSincePrior: String
}
```

---

### 7.2 CONTINUOUS MONITORING DATA

```javascript
// ❌ MISSING from current schema

// NEW MODEL NEEDED: continuousMonitoring.model.js
{
  bedId: String,
  encounterId: String,
  patientId: String,
  
  // Monitoring session
  monitoringStartTime: Date,
  monitoringEndTime: Date,
  
  // Monitor type
  monitorType: {
    enum: ['CARDIAC', 'CAPNOGRAPHY', 'PULSE_OXIMETRY', 'MULTI_PARAMETER'],
    required: true
  },
  
  // Heart rate data (every beat)
  heartRateStream: [{
    bpm: Number,
    timestamp: Date,
    quality: String  // 'good', 'poor', 'artifact'
  }],
  
  // Rhythm strips
  rhythmStrips: [{
    recordedAt: Date,
    duration: Number,  // seconds
    interpretation: String,  // 'NSR', 'PVC', 'AF', 'VT', etc.
    isAbnormal: Boolean,
    alertSent: Boolean
  }],
  
  // Alerts
  monitorAlerts: [{
    alert: String,           // 'Asystole', 'Severe Tachy', 'Lead Off'
    severity: { enum: ['INFO', 'WARNING', 'CRITICAL'] },
    triggeredAt: Date,
    duration: Number,        // How long alert lasted
    acknowledgedBy: String,
    acknowledgedAt: Date,
    notes: String
  }]
}
```

---

## PART 8: SUMMARY TABLE - WHAT EXISTS VS. WHAT'S MISSING

| Function | Model | Current Status | Missing |
|----------|-------|-----------------|---------|
| **Shift Management** | shift.model.js | ✅ Good | `operationalContext` per staff |
| **Staff Registration** | nurse.model.js | ✅ Good | `operationalContext`, `certifications` |
| **Clock In/Out** | nurseClockRecord.model.js | ✅ Good | Handover validation |
| **Patient Arrival** | triageRequest.model.js | ⚠️ Partial | `needsBed` decision field |
| **Patient State** | ❌ NONE | ❌ MISSING | Complete state machine tracker |
| **Disposition** | ❌ NONE | ❌ MISSING | Decision + accuracy tracking |
| **Bed Assignment** | bedSchedule.model.js | ✅ Good | Nothing major |
| **Bed History** | bedStatusHistory.model.js | ✅ Good | Nothing major |
| **Vitals Tracking** | vitals.model.js | ⚠️ Minimal | Threshold alerts, ECG |
| **ECG Data** | ❌ NONE | ❌ MISSING | Waveform + interpretation |
| **Continuous Monitoring** | ❌ NONE | ❌ MISSING | Rhythm, alerts, trending |
| **Patient EHR** | patientEHRRecord.model.js | ✅ Good | ECG, monitoring links |
| **Encounter** | encounter.model.js | ⚠️ Vague | State clarity, escalation tracking |
| **Shift End/Handover** | ❌ Partial | ⚠️ Incomplete | Handover validation |

---

## PART 9: WHEN SHIFTS END (The Complete Handover)

### 9.1 SHIFT COMPLETION WORKFLOW

```
SCENARIO: Bob's shift ends at 15:00 (3:00 PM)
Next Shift: Another Clinical Nurse starts

┌─────────────────────────────────────┐
│ 14:45 PM: Bob sees clock approaching │
│ System: "Your shift ends in 15 min"  │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ Bob reviews: Patients under his care │
│ ├─ John (ICU-05): Stable            │
│ ├─ Sarah (Obs-02): Awaiting results │
│ └─ Michael (Obs-03): Ready discharge│
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ Bob must HAND OVER before clock-out │
│                                     │
│ For each patient:                   │
│ ├─ Provide current status           │
│ ├─ Critical findings               │
│ ├─ Pending orders                  │
│ ├─ Next incoming nurse confirms    │
│ └─ [BOTH sign off]                 │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ ❌ PROBLEM: Current system doesn't  │
│ validate handover before clock-out! │
│                                     │
│ What stops Bob from clocking out   │
│ with unhandled patients?           │
│ → NOTHING!                         │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ Bob clicks "Clock Out"              │
│ System creates:                     │
│ ├─ NurseClockRecord.clockOutTime   │
│ ├─ Calculates totalWorkTime        │
│ ├─ Status: "CLOCKED_OUT"           │
│ └─ [❌ No validation that patients  │
│     are handed over]                │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ Shift ends                          │
│ shift.model.js status = "COMPLETED" │
└─────────────────────────────────────┘
```

---

### 9.2 WHAT SHOULD HAPPEN (IMPROVED)

```javascript
// ENHANCED HANDOVER WORKFLOW

// 1. Check all patients assigned to Bob
const bobPatients = await Encounter.find({
  assignedNurseId: bob_id,
  status: { $in: ['pending', 'confirmed', 'escalated'] }
});

if (bobPatients.length > 0) {
  // 2. Force handover before clock-out
  showModal('Handover Required', {
    message: `You have ${bobPatients.length} patients assigned.
              Each must be handed over to next nurse.`,
    
    patients: bobPatients.map(p => ({
      encounterId: p.encounterId,
      patientName: p.patient.name,
      currentStatus: p.status,
      handoverTo: [list of available nurses],
      handoverNotes: [text field]
    })),
    
    onComplete: async (handoverData) => {
      // 3. Create handover records
      for (let p of handoverData) {
        await Handover.create({
          fromNurseId: bob_id,
          toNurseId: p.handoverTo,
          encounterId: p.encounterId,
          notes: p.handoverNotes,
          acknowledgedBy: p.handoverTo,  // Other nurse confirms!
          handoverTime: NOW
        });
        
        // 4. Update encounter
        await Encounter.updateOne(
          { encounterId: p.encounterId },
          {
            assignedNurseId: p.handoverTo,
            status: 'HANDOVERED'
          }
        );
      }
      
      // 5. NOW Bob can clock out
      await NurseClockRecord.create({
        nurseId: bob_id,
        status: 'CLOCKED_OUT',
        clockOutTime: NOW,
        validatedHandover: true  // ← NEW FIELD
      });
    }
  });
} else {
  // No patients, clock out immediately
  await NurseClockRecord.create({...});
}
```

---

## PART 10: RECOMMENDATIONS FOR YOUR HOSPITAL OS

### 10.1 PRIORITY 1 (CRITICAL - Build These First)

```
1. Add operationalContext to Shift model
   └─ Makes dashboard routing work

2. Create PatientState model + state machine
   └─ Tracks patient journey: WAITING → TRIAGING → etc.

3. Add needsBed field to TriageRequest
   └─ Records triage nurse's bed prediction

4. Create DispositionDecision model
   └─ Tracks physician decision + accuracy

5. Enhance Encounter status enum
   └─ Add: 'triaged', 'escalated', 'handed_over', 'transferred'

6. Add handover validation to NurseClockRecord
   └─ Prevents clock-out with unhandled patients
```

### 10.2 PRIORITY 2 (IMPORTANT - After Priority 1)

```
7. Create ECG model (ecgData.model.js)
   └─ Stores 12-lead waveforms + interpretation

8. Create ContinuousMonitoring model
   └─ Tracks heart rate stream, rhythms, alerts

9. Create ShiftEndSummary model
   └─ Metrics when shift ends

10. Enhance VitalThresholds in PatientEHRRecord
    └─ Auto-alert if vitals outside range
```

### 10.3 OPTIONAL (LATER - Nice to Have)

```
11. Create AcuityAlert model (ALREADY EXISTS!)
    └─ Alert when patient acuity changes

12. Enhance bedStatusHistory with occupancy analytics
    └─ Track: bed turnover time, occupancy rates
```

---

## FINAL CHECKLIST: ST. MERCY HOSPITAL SCENARIO

### Before We Run the Simulation:

- [ ] **Backend**: Add `operationalContext` to shift.assignedStaff
- [ ] **Backend**: Create PatientState model (state machine tracker)
- [ ] **Backend**: Add `needsBed: Boolean` to TriageRequest
- [ ] **Backend**: Create DispositionDecision model
- [ ] **Backend**: Update Encounter status enum (add missing states)
- [ ] **Backend**: Add handover validation to clock-out
- [ ] **Frontend**: Create DashboardRouter with operationalContext switch
- [ ] **Frontend**: Create 4 role-specific dashboards
- [ ] **Frontend**: Show bed map for Ward Nurse
- [ ] **Frontend**: Show vitals graph for monitoring
- [ ] **Testing**: Verify John's journey: WAITING → TRIAGING → CLINICAL → ADMITTED
- [ ] **Testing**: Verify Maria's journey: FALSE NEGATIVE detection
- [ ] **Testing**: Verify shift handover validation

---

**Ready to build this architecture?** 🚀
