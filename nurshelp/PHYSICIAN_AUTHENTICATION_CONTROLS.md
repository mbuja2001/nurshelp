# PHYSICIAN AUTHENTICATION & OPERATIONAL CONTROLS
## Complete Schematics for Physician Registration, Clock-In, and Role-Based Permissions

**Status**: 🔍 DIAGNOSTIC - Physician system analysis  
**Date**: 20 April 2026  

---

## PART 1: PHYSICIAN REGISTRATION & AUTHENTICATION

### 1.1 PHYSICIAN AS EXTENSION OF NURSE MODEL

**Key Finding**: Physicians are NOT in a separate model!

```javascript
// backend/src/models/nurse.model.js
// Physicians are stored in the SAME collection as nurses!

const nurseSchema = {
  hospitalId: ObjectId,
  name: String,
  surname: String,
  email: String (indexed, unique per hospital),
  password: String (bcrypt),
  
  // ← THIS determines if they're a physician or nurse
  role: {
    enum: ['NURSE', 'PHYSICIAN', 'SPECIALIST'],
    required: true
  },
  
  department: {
    enum: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'General'],
    default: 'General'
  },
  
  specialty: String,           // For physicians: "Emergency Medicine", "Cardiology", etc.
  licenseNumber: String,       // Professional license (doctor must have one)
  isActive: Boolean,
  dateCreated: Date,
  timestamps: { created, updated }
};

// ❌ MISSING FIELDS FOR PHYSICIANS:
// - medicalDoctorLicense: String  // Must verify
// - boardCertifications: [String] // e.g., ["ABIM", "ACEP"]
// - specialization: String        // e.g., "ER", "Cardiology", "Trauma"
// - operationalContext: String    // ← CRITICAL! (e.g., ER_CLINICAL_DECISION, HOSPITALIZED_CARE)
```

### 1.2 PHYSICIAN REGISTRATION FLOW

```
WORKFLOW: Hospital Admin (Alice) registers new physician

┌─────────────────────────────────────────┐
│ Admin: Alice Johnson (Manager login)     │
│ Action: "Add Staff Member"               │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ SELECT: Staff Type                      │
│ ├─ [ ] Nurse                            │
│ ├─ [✓] Physician / Doctor               │
│ └─ [ ] Support Staff                    │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ FORM: New Physician Registration        │
│ ├─ First Name: [Dr. Chen           ]    │
│ ├─ Last Name:  [MD                  ]    │
│ ├─ Email:      [dr.chen@...        ]    │
│ ├─ Password:   [***                ]    │
│ ├─ License #:  [MD-12345           ]    │  ← REQUIRED
│ ├─ Role:       [PHYSICIAN          ] ✓  │
│ ├─ Department: [ED                 ] ✓  │
│ ├─ Specialty:  [Emergency Medicine ]    │
│ ├─ Board Certs:[❌ MISSING FIELD] │
│ └─ [Register Physician]                │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ CREATE in nurse.model.js:               │
│ ├─ name: "Chen"                        │
│ ├─ surname: "MD"                       │
│ ├─ email: "dr.chen@stmercy.hospital"   │
│ ├─ password: bcrypt("chen_secure_123") │
│ ├─ role: "PHYSICIAN"  ← KEY FIELD     │
│ ├─ department: "ED"                    │
│ ├─ specialty: "Emergency Medicine"     │
│ ├─ licenseNumber: "MD-12345"           │
│ ├─ isActive: true                      │
│ ├─ createdBy: alice_id                 │
│ └─ [❌ MISSING: operationalContext]    │
└─────────────────────────────────────────┘
                  ↓
✅ Physician now can login using:
   Email: dr.chen@stmercy.hospital
   Password: chen_secure_123
```

### 1.3 PHYSICIAN LOGIN (UNIFIED AUTH)

```javascript
// backend/src/controllers/auth.controller.js - UNIFIED LOGIN

export const login = async (req, res) => {
  // Step 1: Try Manager collection first
  let user = await Manager.findOne({ email });
  let userType = 'manager';
  
  // Step 2: If not found, try Nurse collection (includes PHYSICIANS)
  if (!user) {
    user = await Nurse.findOne({ email });  // ← Physicians found here!
    userType = 'nurse';
  }
  
  // Step 3: Verify password
  const isPasswordValid = await bcrypt.compare(password, user.password);
  
  if (!isPasswordValid) {
    return error: 'Invalid credentials'
  }
  
  // Step 4: Generate JWT token
  const token = jwt.sign({
    id: user._id,
    email: user.email,
    name: user.name,
    role: user.role,            // ← "PHYSICIAN" | "NURSE" | "SPECIALIST"
    department: user.department, // ← "ED", "ICU", etc.
    hospitalId: user.hospitalId,
    hospitalName: user.hospitalId.name
  }, JWT_SECRET, { expiresIn: '24h' });
  
  // Step 5: Return response
  return {
    token,
    user: {
      id: user._id,
      name: user.name,
      email: user.email,
      role: user.role,         // ← JWT will have this
      department: user.department,
      hospitalId: user.hospitalId,
      speciality: user.specialty // ← Available for physicians
    }
  };
}
```

**JWT Token Contents** (Dr. Chen's token):
```json
{
  "id": "phys_chen_001",
  "email": "dr.chen@stmercy.hospital",
  "name": "Chen",
  "role": "PHYSICIAN",           // ← KEY: Identifies as physician
  "department": "ED",             // ← Where they work
  "hospitalId": "hosp_001",
  "hospitalName": "St. Mercy Hospital",
  "iat": 1713607200,
  "exp": 1713693600
}
```

---

## PART 2: PHYSICIAN CLOCK-IN/OUT (OR LACK THEREOF)

### 2.1 CURRENT STATE: ❌ PHYSICIANS DON'T HAVE EXPLICIT CLOCK RECORDS

```javascript
// backend/src/models/nurseClockRecord.model.js
// This is ONLY for nurses!

const clockRecordSchema = {
  nurseId: ObjectId,           // ← References Nurse model
  shiftId: ObjectId,
  date: Date,
  clockInTime: Date,
  clockOutTime: Date,
  breakStartTime: Date,
  breakEndTime: Date,
  status: {
    enum: ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT', 'ABSENT'],
    default: 'CLOCKED_IN'
  },
  location: String,            // 'ED', 'ICU', etc.
  totalShiftDuration: Number,
  totalBreakTime: Number,
  totalWorkTime: Number,
  metadata: {
    deviceType: String,
    ipAddress: String,
    location: { latitude, longitude }
  }
};

// ❌ PROBLEM: No equivalent for physicians!
// ❌ Cannot track: When Dr. Chen clocks in? How long he works?
// ❌ Cannot track: When does he become available for consults?
```

### 2.2 WHAT PHYSICIANS ACTUALLY DO (Current System)

```
OBSERVED PHYSICIAN WORKFLOW:

1. Dr. Chen logs in via /api/auth/login
   └─ Gets JWT token with role: "PHYSICIAN"
   ├─ System knows he's logged in
   └─ ❌ But no "clock in" event recorded

2. Frontend detects role: "PHYSICIAN"
   └─ Routes to: PhysicianDashboard (if it exists)
   ├─ Shows: Recommended patients for review
   ├─ Shows: Encounter queue
   └─ ❌ No explicit "shift started" event

3. Dr. Chen reviews patient (John in ER)
   └─ Calls: PUT /api/encounters/:id/attend
   ├─ Updates: encounter.assignedPhysicianId = chen_id
   ├─ Updates: encounter.assignedPhysicianName = "Dr. Chen"
   ├─ Status: encounter.status = "confirmed"
   └─ Response: Physician has "reviewed" patient

4. But NO RECORD OF:
   ❌ When Dr. Chen started working
   ❌ How many patients he reviewed
   ❌ How long he spent on each patient
   ❌ When does his shift end
   ❌ Workload calculation

COMPARISON:

Nurse Bob:
  ├─ Logs in
  ├─ Clocks in via NurseClockRecord
  ├─ System: "Bob started shift at 07:01"
  ├─ System: "Bob is currently CLOCKED_IN"
  ├─ Receives patient assignments
  └─ Must clock out to end shift

Physician Dr. Chen:
  ├─ Logs in
  ├─ ❌ NO CLOCK RECORD
  ├─ System: "Chen logged in but when did shift start?"
  ├─ Receives patient recommendations
  └─ ❌ No explicit shift end
```

---

## PART 3: PHYSICIAN SCHEDULING (PhysicianSchedule Model)

### 3.1 THE SEPARATE SCHEDULING MODEL

```javascript
// backend/src/models/physicianSchedule.model.js

const physicianScheduleSchema = {
  physicianId: String,         // String, not ObjectId! (⚠️ inconsistency)
  name: String,
  surname: String,
  email: String,
  specialty: String,           // e.g., "Cardiology", "ER"
  
  // Daily schedule
  date: Date,
  status: {
    enum: ['ON_DUTY', 'ON_CALL', 'OFF', 'VACATION'],
    default: 'OFF'
  },
  
  // Shift timing
  shiftStart: String,          // "08:00"
  shiftEnd: String,            // "16:00"
  breakStart: String,          // "12:00"
  breakEnd: String,            // "13:00"
  
  // Capacity
  maxPatients: Number,         // e.g., 20 patients/shift
  currentPatientCount: Number, // How many assigned now
  
  // Availability
  availabilityMask: [Number],  // 24-hour availability (1=available, 0=not)
  workloadScore: Number,       // ESI-based workload calculation
  
  // Performance metrics
  averageTriageTime: Number,   // minutes per patient
  consultationRate: Number,    // % of cases needing consult
  averageConfirmationTime: Number,  // minutes from assignment to confirmation
  
  notes: String,               // "Limited to trauma cases"
  timestamps: { created, updated }
};

// THIS model is SEPARATE from nurse.model.js!
// Creating inconsistency:
// - Physicians registered in nurse.model.js
// - Physicians scheduled in physicianSchedule.model.js
// - NOT linked directly!
```

### 3.2 PHYSICIAN SCHEDULE DATA

```javascript
// EXAMPLE: Dr. Chen's schedule for 2026-04-20

{
  physicianId: "phys_chen_001",
  name: "Chen",
  surname: "MD",
  email: "dr.chen@stmercy.hospital",
  specialty: "Emergency Medicine",
  
  date: 2026-04-20,
  status: "ON_DUTY",
  
  shiftStart: "07:00",
  shiftEnd: "15:00",
  breakStart: "12:00",
  breakEnd: "13:00",
  
  maxPatients: 20,
  currentPatientCount: 0,  // Will increase as patients assigned
  
  availabilityMask: [
    0,0,0,0,0,0,0,   // Midnight-7am: not available
    1,1,1,1,1,1,1,1, // 7am-3pm: available
    1,1,1,1,1,1,1    // 3pm-11pm: available
  ],
  
  workloadScore: 0,
  averageTriageTime: 15,  // avg 15 min per patient
  consultationRate: 0.25, // 25% of cases need consult
  
  notes: "Emergency specialist - trauma focus"
}
```

---

## PART 4: PHYSICIAN CONTROLS & PERMISSIONS

### 4.1 WHAT PHYSICIANS CAN DO (From Routes)

```javascript
// backend/src/routes/encounter.route.js

// 1. GET ENCOUNTER QUEUE
router.get("/waiting", protect, getWaitingEncounters);
// Dr. Chen can see: All patients waiting for physician review

// 2. ATTEND / CONFIRM ENCOUNTER
router.put("/:id/attend", protect, attendEncounter);
// Dr. Chen updates:
// - encounter.assignedPhysicianId = chen_id
// - encounter.assignedPhysicianName = "Dr. Chen"
// - encounter.status = "confirmed"

// 3. ASSIGN PHYSICIAN / BED / WARD
router.patch("/:id/assign", protect, assignPhysician);
// Dr. Chen decides:
// - Disposition: DISCHARGE | ADMIT | REFER
// - Bed assignment (if admitting)
// - Ward assignment (ED, ICU, Ward, etc.)

// 4. GET RECOMMENDATIONS (Specialty-based)
router.get("/recommendations", getRecommendations);
// For a patient, system recommends which physician should handle:
// - Ranked by specialty match
// - Ranked by availability
// - Ranked by current workload
```

### 4.2 PHYSICIAN PERMISSIONS IN SYSTEM

```javascript
// ❌ PROBLEM: No explicit permissions model for physicians!
// Manager model HAS permissions:

const managerPermissions = {
  canCreateShifts: Boolean,
  canEditShifts: Boolean,
  canDeleteShifts: Boolean,
  canAssignStaff: Boolean,
  canApproveLeaves: Boolean,
  canManageRoles: Boolean,
  canViewReports: Boolean
};

// ❌ But Nurse/Physician model doesn't have any!

// IMPLIED permissions for PHYSICIAN role:
// ✅ Can review triage assessments
// ✅ Can attend patients (confirm they're reviewing)
// ✅ Can assign beds/wards
// ✅ Can make disposition decisions
// ❌ Cannot create shifts (manager only)
// ❌ Cannot approve leaves (manager only)
// ❌ Cannot manage roles (admin only)
// ❓ Can they escalate to specialist? (unclear)
// ❓ Can they order medications? (unclear - not in schema)
// ❓ Can they order labs? (unclear - not in schema)
```

---

## PART 5: PHYSICIAN WORKFLOW IN PATIENT JOURNEY

### 5.1 PHYSICIAN ENTERS THE PATIENT SYSTEM

```
STEP 1: Patient triaged by Sarah (triage nurse)
  └─ TriageRequest created with:
     ├─ esiLevel: 2
     ├─ recommendedSpecialty: "Emergency Medicine"
     ├─ recommendedPhysician (from AI):
     │  ├─ physicianId: phys_chen_001
     │  ├─ name: "Dr. Chen"
     │  ├─ specialty: "Emergency Medicine"
     │  └─ matchScore: 0.92
     └─ Status: "TRIAGED"

STEP 2: System recommends Dr. Chen
  └─ GET /api/physicians/recommendations?encounterId=enc_001
  ├─ Input: Encounter data (vitals, symptoms, chief complaint)
  ├─ Calls: Python triage service (if enabled)
  ├─ Returns: Top 5 physicians ranked by:
  │  ├─ Specialty match to patient case
  │  ├─ Current workload score
  │  ├─ Availability now
  │  └─ Performance metrics
  └─ Result: Dr. Chen ranked #1

STEP 3: Bob (clinical nurse) requests Dr. Chen
  └─ Frontend shows: "Request Physician Review"
  ├─ Calls: PUT /api/encounters/enc_001/attend
  ├─ Payload: { assignedPhysicianId: "phys_chen_001" }
  └─ Status: encounter.assignedPhysicianId = phys_chen_001

STEP 4: Dr. Chen receives notification (WebSocket?)
  └─ ❌ MISSING: How does Dr. Chen know he's assigned?
  ├─ ❌ No explicit notification system
  ├─ ❌ No workload update to physicianSchedule
  └─ ❌ Would need to refresh dashboard manually?

STEP 5: Dr. Chen reviews John's case
  └─ Sees: Triage assessment + Bob's vitals recordings
  ├─ Makes diagnosis: "Likely NSTEMI"
  ├─ Decision: ADMIT to ICU
  └─ ❌ MISSING: Where does Dr. Chen document this?
     ├─ No "PhysicianAssessment" model
     ├─ No "DispositionDecision" model
     ├─ No "ClinicalDiagnosis" model

STEP 6: Dr. Chen assigns bed
  └─ Calls: PATCH /api/encounters/enc_001/assign
  ├─ Payload: {
  │   disposition: "ADMIT",
  │   wardId: "ICU",
  │   bedId: "ICU-05",
  │   reason: "Troponin positive, ACS"
  │  }
  └─ Result: Patient moves to bed, Ward Nurse assigned
```

### 5.2 WHAT'S MISSING: PHYSICIAN ASSESSMENT CAPTURE

```javascript
// ❌ CRITICAL MISSING: Where does physician document decision?

// Current Encounter model has:
{
  assignedPhysicianId: String,
  assignedPhysicianName: String,
  // ❌ NO fields for:
  // - physicianAssessment: String
  // - physicianDiagnosis: String
  // - physicianDecision: String (ADMIT | DISCHARGE | REFER)
  // - dispositionReasoning: String
  // - accuracyVsTriage: Boolean (did physician match triage prediction?)
}

// NEEDED: PhysicianAssessment model
const physicianAssessmentSchema = {
  encounterId: String,
  physicianId: String,
  physicianName: String,
  
  // Physician findings
  clinicalAssessment: String,  // "Based on ECG, troponin, chest pain..."
  preliminaryDiagnosis: String, // "Acute coronary syndrome (NSTEMI)"
  severity: String,             // "High", "Moderate", "Low"
  
  // Comparison with triage
  triageNeedsBedPrediction: Boolean, // What Sarah said
  physicianDecision: {
    disposition: {
      enum: ['DISCHARGE', 'ADMIT', 'REFER'],
      required: true
    },
    ward: String,
    bed: String,
    reasoning: String
  },
  
  // Accuracy (for QA)
  triageAccuracy: String,  // 'TRUE_POSITIVE', 'FALSE_NEGATIVE', etc.
  
  // Timing
  reviewStartedAt: Date,
  reviewCompletedAt: Date,
  timeToDecision: Number   // seconds
};
```

---

## PART 6: OPERATIONAL CONTEXTS FOR PHYSICIANS

### 6.1 TWO OPERATIONAL CONTEXTS FOR PHYSICIANS

Based on the hospital scenario, physicians need TWO contexts:

#### Context 1: ER_CLINICAL_DECISION
**Staff Member**: Dr. Chen MD  
**Title**: Emergency Physician  
**Department**: ED  
**Responsibility**: "Review ER cases, make disposition decisions"

**What Dr. Chen sees** (ER Context):
```
Dashboard: "🩺 PHYSICIAN COMMAND CENTRE - ER MODE"

┌─────────────────────────────────────┐
│ Active ER Cases (Needs Decision)    │
├─────────────────────────────────────┤
│ 🔴 RED PRIORITY:                   │
│ ├─ John Smith (59M, Chest pain)   │
│ │  └─ [📋 Review] [✓ Admit] [✗ Disch]
│ └─ [2 more critical cases]         │
│                                     │
│ 🟡 YELLOW OBSERVATION:             │
│ ├─ Maria Rodriguez (71F, Confusion) │
│ │  └─ [📋 Review] [✓ Admit] [✗ Disch]
│ └─ [5 more moderate cases]         │
│                                     │
│ 🟢 GREEN (Routine):                │
│ └─ [Monitor list, lower priority]  │
│                                     │
│ 📊 WORKLOAD: 8/20 patients reviewed │
│ ⏱️  AVG REVIEW TIME: 12 min/patient │
└─────────────────────────────────────┘

Controls:
├─ [📋 Review Details] - See full triage + vitals
├─ [✓ Admit] - Order bed assignment
├─ [✗ Discharge] - Send home
├─ [↗️ Refer] - Transfer to specialist
└─ [✎ Notes] - Add physician assessment
```

**Permissions in ER_CLINICAL_DECISION context**:
```
✅ Can review triage assessments
✅ Can view patient vitals + trends
✅ Can access clinical history
✅ Can recommend bed/ward
✅ Can make disposition decision (ADMIT/DISCHARGE)
✅ Can document clinical findings
✅ Can request specialist consult
✅ Can escalate urgent cases
❌ Cannot modify nurse notes
❌ Cannot change vital signs (read-only)
❌ Cannot discharge against nursing recommendation
```

---

#### Context 2: ON_CALL_PHYSICIAN (Hybrid View)
**Staff Member**: [Second Physician or Attending]  
**Title**: On-Call Physician (Command Centre)  
**Department**: All  
**Responsibility**: "See ER crisis + Ward alerts simultaneously"

**What On-Call Physician sees**:
```
Dashboard: "🏥 COMMAND CENTRE - HYBRID MODE"

Top Half (ER Activity):
┌──────────────────────────┐
│ 🩺 ACTIVE ER             │
│ ├─ 3 RED cases waiting   │
│ ├─ 7 YELLOW observing    │
│ └─ 2 ready for discharge │
└──────────────────────────┘

Bottom Half (Ward Status):
┌──────────────────────────┐
│ 🛏️ WARD ALERTS            │
│ ├─ ICU-05: HR elevated   │
│ ├─ Ward-12: Temp spike   │
│ └─ ICU-02: Pain complaint│
└──────────────────────────┘

Decision Point:
├─ Focus on ER (most ER cases waiting)
├─ Focus on Ward (critical alert detected)
└─ Call in reinforcements? (No available physicians)
```

---

## PART 7: COMPARISON: NURSE vs. PHYSICIAN SCHEMATICS

| Aspect | Nurse | Physician |
|--------|-------|-----------|
| **Storage Model** | nurse.model.js (role: NURSE) | nurse.model.js (role: PHYSICIAN) |
| **Schedule Model** | nurseSchedule.model.js | physicianSchedule.model.js |
| **Clock Record** | ✅ nurseClockRecord.model.js | ❌ MISSING |
| **Authentication** | Unified auth.login | Unified auth.login |
| **Token Has Role** | role: "NURSE" | role: "PHYSICIAN" |
| **Workload Tracking** | nurseAssignment.model.js | ❌ MISSING |
| **Patient Assignment** | assignedNurseId (Encounter) | assignedPhysicianId (Encounter) |
| **Assessment Model** | nurseConfirmedReport.model.js | ❌ MISSING PhysicianAssessment |
| **Permissions Model** | ❌ No explicit field | ❌ No explicit field |
| **Operational Context** | ❌ MISSING in Shift | ❌ MISSING in PhysicianSchedule |
| **Dashboard** | ✅ NurseDashboard (exists) | ❌ PhysicianDashboard (missing) |
| **Vitals Access** | Can log vitals (Vitals.model.js) | Can VIEW vitals (read-only) |
| **Medication Orders** | Cannot order | ❌ MISSING order model |
| **Lab Orders** | Cannot order | ❌ MISSING order model |

---

## PART 8: CRITICAL GAPS IN PHYSICIAN SYSTEM

### 8.1 MISSING MODELS

```
❌ PhysicianAssessment.model.js
   └─ Should store: Clinical findings, diagnosis, disposition decision

❌ PhysicianClockRecord.model.js (or extend NurseClockRecord)
   └─ Should track: When physician started, how many patients, end time

❌ PhysicianWorkload.model.js
   └─ Should track: Current patient count, response time metrics

❌ DispositionDecision.model.js
   └─ Should store: Physician decision + accuracy tracking

❌ MedicationOrder.model.js
   └─ Should store: Medications ordered by physician

❌ LabOrder.model.js
   └─ Should store: Lab tests ordered by physician

❌ ConsultationRequest.model.js
   └─ Should store: Specialist consults ordered by physician
```

### 8.2 MISSING FIELDS

```
nurse.model.js (for PHYSICIAN role):
❌ operationalContext: String
   └─ "ER_CLINICAL_DECISION" | "ON_CALL_PHYSICIAN"

physicianSchedule.model.js:
❌ Link to actual Nurse record (inconsistent physicianId type)
❌ onCallAvailability: { startTime, endTime }
❌ consultationSpecialties: [String]

Encounter.model.js:
❌ physicianAssessment: String
❌ physicianDiagnosis: String
❌ disposition: {
     decision: String,
     reasoning: String,
     decidedAt: Date
   }
❌ accuracyVsTriage: String (TRUE_POSITIVE, FALSE_NEGATIVE, etc.)
```

### 8.3 MISSING WORKFLOWS

```
❌ Physician receiving patient assignment
   └─ How does Dr. Chen know he's assigned? (No notification)

❌ Real-time workload updates
   └─ How does system track Dr. Chen is now reviewing patient #5?

❌ Physician clock in/out workflow
   └─ When does Dr. Chen's shift officially start/end?

❌ Specialist consultation requests
   └─ Can Dr. Chen request Cardiology consult? (No model)

❌ Medication ordering workflow
   └─ Where do prescriptions get stored?

❌ Lab ordering workflow
   └─ Where do lab requests get stored?

❌ Physician-to-physician handover
   └─ When next physician takes over, what handoff happens?
```

---

## PART 9: BUILD PRIORITIES FOR PHYSICIAN SYSTEM

### PRIORITY 1 (CRITICAL - Before St. Mercy scenario runs)

```
1. Add operationalContext field to Shift.assignedStaff
   └─ Include physicians with context: ER_CLINICAL_DECISION

2. Create PhysicianAssessment model
   └─ Captures physician decision + accuracy

3. Enhance Encounter.status enum
   └─ Add: "awaiting_physician", "physician_reviewed", "disposition_decided"

4. ❓ Create PhysicianClockRecord (equivalent to NurseClockRecord)
   └─ Track when Dr. Chen starts/ends shift
   └─ Track workload metrics

5. Create DispositionDecision model
   └─ Track ADMIT vs. DISCHARGE decisions
   └─ Compare against triage prediction (for accuracy)
```

### PRIORITY 2 (IMPORTANT - After core runs)

```
6. Create PhysicianDashboard component
   └─ 3-lane view: RED waiting, YELLOW observing, GREEN ready

7. Create real-time notification system
   └─ Alert physician when assigned patient

8. Enhance PhysicianSchedule linking
   └─ Link to actual Nurse record (consistency)

9. Create consultationRequest model
   └─ Support specialist referrals
```

### PRIORITY 3 (OPTIONAL - Later)

```
10. Create MedicationOrder model
11. Create LabOrder model
12. Create PhysicianPerformanceMetrics model
```

---

## PART 10: RECOMMENDED PHYSICIAN ARCHITECTURE

### 10.1 UNIFIED CLOCK RECORD (For Both Nurses and Physicians)

```javascript
// ENHANCED: clockRecord.model.js (supports both)

const clockRecordSchema = {
  // Who is clocking in?
  staffId: ObjectId,           // References Nurse model
  staffName: String,
  staffRole: {
    enum: ['NURSE', 'PHYSICIAN', 'SPECIALIST'],
    required: true
  },
  
  // When?
  shiftId: ObjectId,
  date: Date,
  clockInTime: Date,
  clockOutTime: Date,
  
  // Where?
  location: String,
  department: String,
  
  // Operational context
  operationalContext: String,  // e.g., "TRIAGE_NURSE", "ER_CLINICAL_DECISION"
  
  // Activity metrics (filled at clock-out)
  patientsHandled: Number,
  timeSpent: Number,            // seconds
  tasksCompleted: Number,
  escalationsInitiated: Number,
  
  // Shift metrics
  status: {
    enum: ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT'],
    default: 'CLOCKED_IN'
  },
  
  // Handover (before clocking out)
  handoverDetails: {
    patientsHandedOver: [{
      patientId: String,
      receivingStaffId: String,
      confirmed: Boolean
    }],
    allHandedOver: Boolean,
    handoverCompletedAt: Date
  }
};
```

### 10.2 PHYSICIAN IN ST. MERCY SCENARIO

```
ST. MERCY SCENARIO - Dr. Chen's Full Day:

07:00 AM
├─ Dr. Chen logs in
├─ Shift announced: "ED Shift, 07:00-15:00"
├─ Clocks in via unified endpoint
└─ System: "Dr. Chen is now CLOCKED_IN, operational context: ER_CLINICAL_DECISION"

07:15 AM
├─ John arrives + triaged by Sarah
├─ System recommends: Dr. Chen (specialty match: 0.92)
├─ Bob requests physician review
└─ Dr. Chen notified (WebSocket): "John Smith assigned, urgent review needed"

07:50 AM
├─ Dr. Chen reviews John
├─ Creates: PhysicianAssessment
│  ├─ Diagnosis: "Likely NSTEMI"
│  ├─ Decision: ADMIT to ICU
│  ├─ Reasoning: "Troponin positive, elevated troponin trajectory"
│  └─ AccuracyVsTriage: "TRUE_POSITIVE" (triage predicted bed: YES, physician admits: YES)
├─ Updates: Encounter with disposition
└─ Assigns: Bed ICU-05

08:00 AM - 14:00 PM
├─ Reviews 5 more patients
├─ Sarah (GREEN): Discharged ✓
├─ Michael (YELLOW): Discharged ✓
├─ Maria (FALSE NEGATIVE): 
│  └─ Triage said NO bed, but Dr. Chen admits to ICU
│  └─ system flags: "FALSE_NEGATIVE - Training opportunity"
└─ System tracking: "Dr. Chen reviewed 8 patients, 3 admitted, 2 discharged, 1 referred"

14:30 PM
├─ No more patients waiting
├─ Begins clock-out process
├─ Must hand over any ongoing cases to next physician
└─ System validates: All patients assigned/dispositioned

14:45 PM
├─ Dr. Chen ready to clock out
├─ Calls: /api/clockout
├─ System updates: clockOutTime = NOW
├─ Shift summary generated:
│  ├─ Duration: 7h 45m (minus 1h break)
│  ├─ Patients reviewed: 8
│  ├─ Admissions: 3
│  ├─ Discharges: 2
│  ├─ Referrals: 1
│  ├─ Response time: avg 8 min per patient
│  └─ Accuracy: 7/8 correct (87.5%, Maria was false negative)
└─ Status: "CLOCKED_OUT"

15:00 PM
├─ Next physician clocks in
├─ Takes over any ongoing patients
└─ System updates: New physician now ER_CLINICAL_DECISION context
```

---

## SUMMARY TABLE

| Item | Current Status | Needed for St. Mercy |
|------|-----------------|---------------------|
| Physician Registration | ✅ Works | ✅ Working |
| Physician Login | ✅ Works | ✅ Working |
| Physician Role (PHYSICIAN) | ✅ In nurse.model.js | ✅ Using |
| Physician Schedule | ✅ physicianSchedule.model.js exists | ✅ Using |
| Physician Clock In/Out | ❌ MISSING | 🔴 CRITICAL - Need to build |
| Operational Context | ❌ MISSING | 🔴 CRITICAL - Need to add |
| Assessment Capture | ❌ MISSING | 🔴 CRITICAL - Need model |
| Disposition Decision | ❌ MISSING | 🔴 CRITICAL - Need model |
| Accuracy Tracking | ❌ MISSING | 🔴 CRITICAL - Need logic |
| Dashboard | ❌ MISSING | 🟡 IMPORTANT - Need component |
| Real-time Notifications | ❌ MISSING | 🟡 IMPORTANT - Need system |

---

**Ready to build unified physician framework?** 🏥
