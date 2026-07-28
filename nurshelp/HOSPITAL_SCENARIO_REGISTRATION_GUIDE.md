# ST. MERCY HOSPITAL SCENARIO - MANUAL REGISTRATION GUIDE

**Objective**: Register hospital, staff, and patients to match the simulation scenario while respecting existing software workflows

**Status**: Ready for manual implementation  
**Date**: 20 April 2026  
**System**: Nurse Helpdesk v1.0

---

## PART 1: HOSPITAL REGISTRATION

### Step 1: Hospital Registration Page

**URL**: `http://localhost:5181/register` (or equivalent)

**Form Data to Enter**:

```
Step 1: Hospital Information
┌─────────────────────────────────────────┐
│ Hospital Name:                          │
│ [St. Mercy Hospital                  ]  │
│                                         │
│ Country: [South Africa              ]  │
│ City: [Cape Town                    ]  │
│ State: [Western Cape                ]  │
│ Zip: [8000                          ]  │
│ Phone: [+27 21 555 0100             ]  │
│ Email: [admin@stmercy.hospital      ]  │
└─────────────────────────────────────────┘

Step 2: Admin Manager (Hospital Admin)
┌─────────────────────────────────────────┐
│ First Name:  [Alice              ]     │
│ Last Name:   [Johnson            ]     │
│ Email:       [alice.johnson@...  ]     │
│ Password:    [alice_secure_123   ]     │
│ Department:  [All                ]     │
│ Role:        [ADMIN              ] ✓   │
└─────────────────────────────────────────┘

Step 3: Departments
┌─────────────────────────────────────────┐
│ Available Departments to Add:           │
│ [ ] ED (Emergency Department)    ✓     │
│ [ ] ICU (Intensive Care Unit)    ✓     │
│ [ ] Ward (General Hospital Ward) ✓     │
│ [ ] OR (Operating Room)                 │
│                                         │
│ Selected: ED, ICU, Ward                │
└─────────────────────────────────────────┘

[Cancel] [← Previous Step] [Next Step →] [Submit Registration]
```

**Click**: Submit Registration

**Expected Response**:
```
✅ Hospital "St. Mercy Hospital" registered successfully
✅ Admin Manager "Alice Johnson" created
✅ You can now login with your admin credentials
```

---

## PART 2: STAFF MEMBER REGISTRATION

### Staff to Register (from simulation)

| # | Name | Role | Specialty | Department | Email | Password | Shift Context | Notes |
|---|------|------|-----------|-----------|-------|----------|----------------|-------|
| 1 | Bob | NURSE | General | ED | bob.martinez@stmercy.hospital | bob_secure_123 | CLINICAL_NURSE_ER | Manages observation patients in ER |
| 2 | Sarah | NURSE | Emergency Medicine | ED | sarah.triage@stmercy.hospital | sarah_secure_123 | TRIAGE_NURSE | Processes incoming triage patients |
| 3 | Dr. Chen | PHYSICIAN | Emergency Medicine | ED | dr.chen@stmercy.hospital | chen_secure_123 | PHYSICIAN_ER | Makes disposition decisions |
| 4 | Ward Nurse | NURSE | General | Ward | ward.nurse@stmercy.hospital | ward_secure_123 | CLINICAL_NURSE_WARD | Manages hospitalized patients in beds |

---

### Method 1: Register via Frontend (Manual)

**You have two options**:
1. Create via existing RegisterForm multi-step process (recommended for realistic testing)
2. Create via backend API call (faster for testing)

#### Option A: Frontend Registration Flow

**For each staff member**:

1. **Login as Admin** (Alice Johnson)
   - Email: `alice.johnson@stmercy.hospital`
   - Password: `alice_secure_123`

2. **Navigate to**: Staff Management or Add Staff page (check UI flow)

3. **Fill Staff Register Form**:
   ```
   Name: Bob
   Surname: Martinez
   Email: bob.martinez@stmercy.hospital
   Password: bob_secure_123
   Role: NURSE
   Department: ED
   Specialty: General
   License Number: LIC-NURSE-001
   
   [Add Staff]
   ```

4. **Repeat for each staff member**

#### Option B: Backend API Call (for testing)

**For testing/seed purposes**:

```bash
# Terminal: Create staff via API
curl -X POST http://localhost:5001/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "hospitalId": "hospital_stmercy",
    "name": "Bob",
    "surname": "Martinez",
    "email": "bob.martinez@stmercy.hospital",
    "password": "bob_secure_123",
    "role": "NURSE",
    "department": "ED",
    "specialty": "General",
    "licenseNumber": "LIC-NURSE-001",
    "isActive": true
  }'
```

---

## PART 3: STAFF ROLE CONTEXT ASSIGNMENT

### Critical: Assign Shift Roles (Manual Step)

Once staff are registered, they need to be assigned to **specific shift contexts** for the simulation to work.

**This requires**:
1. Creating Shift records
2. Assigning staff to shifts with specific contexts
3. Defining their role during that shift

### Current System Flow vs. Required Flow

**Current System** (What exists):
- Staff register with generic role (NURSE, PHYSICIAN)
- Staff assigned to department (ED, ICU, Ward)
- Shifts are created with assigned staff

**Required for Simulation** (What we need):
- Same staff in same encounter but different dashboards
- Alice = TRIAGE_NURSE (processes queue)
- Bob = CLINICAL_NURSE_ER (manages observation)
- Dr. Chen = PHYSICIAN_ER (reviews all cases)
- Ward Nurse = CLINICAL_NURSE_WARD (manages beds)

### Implementation: Shift Assignment (Manual)

**Step 1: Create Today's Shift**

As Admin (Alice), create a shift:

```
Shift Name: "Tuesday 20th April - ED Shift"
Date: 2026-04-20
Department: ED
Start Time: 07:00
End Time: 15:00
Duration: 8 hours

Assign Staff:
┌─────────────────────────────┐
│ ✓ Sarah (Triage Nurse)      │
│ ✓ Bob (Clinical Nurse)      │
│ ✓ Dr. Chen (Physician)      │
│ ✓ Ward Nurse (Ward)         │
└─────────────────────────────┘

[Create Shift]
```

**Step 2: Each Staff Member Clocks In**

```
7:00 AM - Sarah (Triage Nurse) logs in
├─ Email: sarah.triage@stmercy.hospital
├─ Password: sarah_secure_123
├─ System shows: "Clock In - ED Shift"
└─ [Clock In]

7:01 AM - Bob (Clinical Nurse) logs in
├─ Email: bob.martinez@stmercy.hospital
├─ Password: bob_secure_123
├─ System shows: "Clock In - ED Shift"
└─ [Clock In]

7:02 AM - Dr. Chen (Physician) logs in
├─ Email: dr.chen@stmercy.hospital
├─ Password: chen_secure_123
├─ System shows: "Clock In - ED Shift"
└─ [Clock In]
```

---

## PART 4: PATIENT REGISTRATION

### Patients for Scenario

| # | Name | DOB | Age | Arrived | Chief Complaint | Priority | Needs Bed | Outcome |
|---|------|-----|-----|---------|-----------------|----------|-----------|---------|
| 1 | John Smith | 1965-03-15 | 59 | 7:15 AM | Acute chest pain | RED | YES | ADMIT to ICU |
| 2 | Sarah Wong | 1992-07-22 | 34 | 7:35 AM | Ankle sprain | GREEN | NO | DISCHARGE |
| 3 | Michael Chen | 1978-11-05 | 48 | 8:15 AM | Mild fever | YELLOW | NO | DISCHARGE |
| 4 | Maria Rodriguez | 1955-08-12 | 71 | 11:30 AM | Confusion, headache | YELLOW | NO | ADMIT (FALSE NEG) |

---

### Method 1: Patient Arrival via Frontend (VoiceIntake?)

**Check if patient intake is available**:

1. Find **Patient Arrival** or **Triage Queue** page
2. Look for "New Patient" button
3. Enter patient information:

```
Patient 1: John Smith (RED - Chest Pain)
┌─────────────────────────────────────┐
│ First Name:    [John            ]   │
│ Last Name:     [Smith           ]   │
│ Date of Birth: [03/15/1965      ]   │
│ Age:           [59              ]   │
│ Gender:        [Male            ]   │
│ Chief Complaint: [Chest pain    ]   │
│ Temperature:   [37.2            ]   │
│ Heart Rate:    [102             ]   │
│ BP (Sys/Dia):  [158/92          ]   │
│ Resp Rate:     [20              ]   │
│ O2 Saturation: [96              ]   │
│ Pain Level:    [8/10            ]   │
│                                     │
│ Medical History: [Hypertension  ]   │
│ Current Meds: [Lisinopril, ...  ]   │
│ Allergies: [NKDA               ]   │
│                                     │
│ [Cancel] [Add to Queue]             │
└─────────────────────────────────────┘
```

**Click**: Add to Queue

**Expected Result**:
```
✅ Patient added to triage queue
✅ Patient appears in Triage Nurse dashboard
✅ Sarah sees John in her queue
```

---

### Method 2: Patient Arrival via Backend Seed Script

**For rapid testing**, create a seed script:

**File**: `backend/src/seeds/seedHospitalScenario.js`

```javascript
/**
 * Seed: St. Mercy Hospital Scenario
 * 
 * Creates:
 * - Hospital: St. Mercy Hospital
 * - Staff: Sarah (Triage), Bob (Clinical), Dr. Chen (Physician), Ward Nurse
 * - Shifts: Today's ED shift with all staff
 * - Patients: 4 patients with specific scenarios
 */

import mongoose from "mongoose";
import bcrypt from "bcrypt";
import dotenv from "dotenv";
import { Hospital } from "../src/models/hospital.model.js";
import { Manager } from "../src/models/manager.model.js";
import { Nurse } from "../src/models/nurse.model.js";
import { Shift } from "../src/models/shift.model.js";
import { TriageRequest } from "../src/models/triageRequest.model.js";
import { Encounter } from "../src/models/encounter.model.js";

dotenv.config();

const HOSPITAL_DATA = {
  hospitalID: "HOSP-STMERCY-001",
  name: "St. Mercy Hospital",
  country: "South Africa",
  city: "Cape Town",
  state: "Western Cape",
  zipCode: "8000",
  phone: "+27 21 555 0100",
  email: "admin@stmercy.hospital",
};

const ADMIN_DATA = {
  name: "Alice",
  surname: "Johnson",
  email: "alice.johnson@stmercy.hospital",
  password: "alice_secure_123",
};

const STAFF_DATA = [
  {
    name: "Sarah",
    surname: "Triage",
    email: "sarah.triage@stmercy.hospital",
    password: "sarah_secure_123",
    role: "NURSE",
    department: "ED",
    specialty: "Triage",
    licenseNumber: "LIC-TRIAGE-001",
    shiftContext: "TRIAGE_NURSE",
  },
  {
    name: "Bob",
    surname: "Martinez",
    email: "bob.martinez@stmercy.hospital",
    password: "bob_secure_123",
    role: "NURSE",
    department: "ED",
    specialty: "Clinical Nursing",
    licenseNumber: "LIC-CLINICAL-001",
    shiftContext: "CLINICAL_NURSE_ER",
  },
  {
    name: "Dr. Chen",
    surname: "MD",
    email: "dr.chen@stmercy.hospital",
    password: "chen_secure_123",
    role: "PHYSICIAN",
    department: "ED",
    specialty: "Emergency Medicine",
    licenseNumber: "MD-CHEN-001",
    shiftContext: "PHYSICIAN_ER",
  },
  {
    name: "Ward",
    surname: "Nurse",
    email: "ward.nurse@stmercy.hospital",
    password: "ward_secure_123",
    role: "NURSE",
    department: "Ward",
    specialty: "General Nursing",
    licenseNumber: "LIC-WARD-001",
    shiftContext: "CLINICAL_NURSE_WARD",
  },
];

const PATIENT_DATA = [
  {
    firstName: "John",
    lastName: "Smith",
    dob: new Date("1965-03-15"),
    chiefComplaint: "Acute chest pain",
    initialVitals: {
      temperature: 37.2,
      heartRate: 102,
      respiratoryRate: 20,
      systolicBP: 158,
      diastolicBP: 92,
      oxygenSaturation: 96,
    },
    medicalHistory: ["Hypertension"],
    medications: ["Lisinopril 10mg", "Aspirin 81mg daily"],
    allergies: ["NKDA"],
    triagePriority: "RED",
    triageReasoning: "Acute chest pain radiating to arm, elevated vitals",
    needsBed: true,
    bedNeedReasoning: "Needs cardiac workup, probable admission",
    arrivedAt: new Date("2026-04-20T07:15:00"),
  },
  // ... more patients
];

async function seedScenario() {
  try {
    console.log("🔄 Seeding St. Mercy Hospital Scenario...");
    
    // Create hospital
    // Create staff
    // Create shift
    // Create patients
    
    console.log("✅ Scenario seeded successfully");
  } catch (error) {
    console.error("❌ Seed error:", error);
  }
}

seedScenario();
```

---

## PART 5: TESTING THE FULL FLOW

### Timeline: Simulating the Hospital Day

Once all staff and patients are registered, test the simulation:

---

### **7:00 AM - Shift Setup**

#### Step 1: Staff Clock In

1. **Sarah** (Triage Nurse) logs in to dashboard
   - URL: `http://localhost:5181/dashboard`
   - Email: `sarah.triage@stmercy.hospital`
   - Password: `sarah_secure_123`

2. **Bob** (Clinical Nurse) logs in to browser tab
   - Email: `bob.martinez@stmercy.hospital`
   - Password: `bob_secure_123`

3. **Dr. Chen** (Physician) logs in to browser tab
   - Email: `dr.chen@stmercy.hospital`
   - Password: `chen_secure_123`

---

#### Step 2: Team Assignment (NEW - Feature Missing)

**Expected Behavior** (from simulation):
```
Sarah's Dashboard → MODAL: "Assign Clinical Team"
  ┌─────────────────────────┐
  │ Select Clinical Nurse:  │
  │ [▼ Bob Martinez    ]    │
  │                         │
  │ Select Physician:       │
  │ [▼ Dr. Chen        ]    │
  │                         │
  │ [✓ Confirm Assignment]  │
  └─────────────────────────┘

RESULT:
├─ Sarah's dashboard: "Team Assigned" ✓
├─ Bob's dashboard: Changes to CLINICAL_NURSE_ER context ✓
└─ Dr. Chen's dashboard: Changes to PHYSICIAN_ER context ✓
```

**Current System Status**: ❌ FEATURE MISSING - Team assignment modal needs to be built

---

### **7:15 AM - First Patient: John (RED - Chest Pain)**

#### Step 1: Patient John Arrives

**Method 1 - Manual Entry in UI**:
- Sarah enters patient info manually (if UI exists)

**Method 2 - Direct paste into database**:
- Use seed script to create patient

**Result**:
```
Sarah's Screen: Patient Queue Updates
┌──────────────────────────────┐
│ #1 John Smith, 59 years old  │
│    Chief: Chest pain         │
│    ⏱️ Waiting: 0m              │
│    [📋 Triage]              │
└──────────────────────────────┘
```

---

#### Step 2: Sarah Triages John

**Current System**: Need to check if TriageAssessmentForm exists

**Expected Flow**:
1. Sarah clicks [📋 Triage]
2. TriageAssessmentForm opens
3. Sarah fills:
   - Vitals: T 37.2, HR 102, RR 20, BP 158/92, O2 96%
   - Priority: RED
   - Reasoning: "Acute chest pain radiating to arm"
   - **⭐ CRITICAL: Needs Bed? → YES**
   - Bed Reasoning: "Cardiac workup needed"
4. Sarah submits

**Expected Result**:
```
✅ POST /api/triage/submit-assessment

TriageAssessment Created:
├─ patientId: john_001
├─ triagePriority: RED
├─ needsBed: true ⭐
├─ triageReasoning: "Acute chest pain radiating..."
└─ Status: "AWAITING_CLINICAL_NURSE"

WEBSOCKET Events:
├─ to Bob: "patient_ready_for_clinical"
│  └─ Shows John in observation with "needsBed: true" flag
│
└─ to Dr. Chen: "new_case_awaiting_review"
   └─ Shows John as RED priority case
```

---

#### Step 3: Bob Receives John in Clinical Mode

**Bob's Dashboard Updates** (real-time):
```
⚕️ CLINICAL MODE - Non-Hospitalized Patients
┌────────────────────────────────┐
│ John Smith (59M)               │
│ 🔴 RED Priority - Chest Pain   │
│ Bed: OBSERVATION_1             │
│ Triage: NEEDS BED ✓            │
│ [📊 Log Vitals]               │
│ [🩺 Request Physician Review]   │
└────────────────────────────────┘
```

**Bob's Action**: Click [🩺 Request Physician Review]

---

#### Step 4: Dr. Chen Reviews and Decides

**Dr. Chen's Dashboard** (real-time alert):
```
🩺 PHYSICIAN COMMAND CENTRE - ER

🔴 ACTIVE ER CASES (1)
┌────────────────────────────────┐
│ ⚡ John Smith - RED Priority   │
│ Chest Pain, awaiting review    │
│                                │
│ [📋 Review] [✓ Admit] [✗ Disch]
└────────────────────────────────┘
```

**Dr. Chen clicks** [📋 Review]:

```
ASSESSMENT MODAL Shows:
├─ Triage Assessment: RED, "Chest pain radiating..."
├─ Vitals: HR 102↑, BP 158/92↑, O2 96%
├─ Triage Decision: "NEEDS BED" ✓
│
├─ PHYSICIAN DECISION:
│  ├─ Clinical Assessment: "ACS - likely NSTEMI"
│  ├─ Decision: [✓ ADMIT]
│  ├─ Ward: [ICU/CCU]
│  ├─ Reason: "Needs troponin levels, continuous monitoring"
│
└─ [Submit Disposition]
```

**Dr. Chen clicks** [Submit Disposition]:

```
✅ POST /api/disposition

Disposition Created:
├─ physicianDecision: "ADMITTED"
├─ triageRecommendation: true
├─ discrepancy: null (MATCH! ✓)
│  "Triage said needs bed, physician admitted"
│
├─ Patient State Updated: "HOSPITALIZED"
├─ Location: ICU/CCU Bed 05
│
└─ EHR Bundle Created: INPATIENT
   └─ Ready for ongoing care documentation

Metrics Logged:
├─ Decision to Admit Latency: 9m 30s ✓
├─ Physician Response Time: 4m 30s ✓
├─ Triage Accuracy: TRUE POSITIVE ✓
└─ Outcome: ADMISSION (ICU - HIGH ACUITY)
```

**Dashboard Updates** (all parties):
```
Sarah's Screen: John removed from queue ✓
Bob's Screen: John removed from clinical list (transferred to ICU)
Dr. Chen's Screen: John removed from active cases ✓
```

---

### **7:35 AM - Second Patient: Sarah Wong (GREEN - Ankle)**

#### Similar flow but:
- GREEN priority
- `needsBed: false` ("Can be managed in observation")
- Outcome: DISCHARGE (same-day)
- Triage Accuracy: TRUE NEGATIVE ✓

---

### **8:15 AM - Third Patient: Michael Chen (YELLOW - Fever)**

#### Similar flow:
- YELLOW priority
- `needsBed: false` (stable, likely viral)
- Outcome: DISCHARGE (same-day)
- Triage Accuracy: TRUE NEGATIVE ✓

---

### **11:30 AM - Fourth Patient: Maria Rodriguez (YELLOW → FALSE NEGATIVE)**

#### Tricky case:
1. **Triage** (11:33 AM):
   - Priority: YELLOW
   - `needsBed: false` ("Mild symptoms, stable")

2. **Clinical Observation** (11:45 AM):
   - Bob discovers: On warfarin + confusion worsening + head bruise
   - Bob escalates immediately

3. **Physician Review** (11:50 AM):
   - Dr. Chen sees: Potential intracranial bleed
   - Decision: ADMIT to ICU (neuro monitoring)

4. **Discrepancy Detected**:
   ```
   ⚠️ FALSE NEGATIVE
   ├─ Triage said: NO BED NEEDED
   ├─ Physician said: ADMIT to ICU
   ├─ Reason: Warfarin + head trauma + confusion → neurology concern
   └─ Cost Impact: $3,200 (potential readmit)
   ```

5. **QA Alert Generated**:
   ```
   Discrepancy Log Entry:
   ├─ Type: FALSE_NEGATIVE
   ├─ Root Cause: Medication interaction not assessed
   ├─ Action: Flag Alice for coaching on neuro screening
   └─ Training: Add warfarin + head trauma → auto-escalate rule
   ```

---

### **2:00 PM - Shift Metrics Summary**

**After shift ends**:

```
GET /api/analytics/triage-accuracy

RESPONSE:
{
  shift: "ed_shift_20apr2026",
  date: "2026-04-20",
  
  SUMMARY:
  ├─ Patients Triaged: 4
  ├─ TRUE POSITIVES: 1 (John → admitted as predicted)
  ├─ TRUE NEGATIVES: 2 (Sarah, Michael → discharged as predicted)
  ├─ FALSE NEGATIVES: 1 (Maria → missed admission need)
  ├─ FALSE POSITIVES: 0
  │
  └─ ACCURACY: 75% (3/4 correct)

  PHYSICIAN RESPONSE:
  ├─ Average Response Time: 27 minutes
  ├─ RED Priority (<5 min): 4m 30s ✓
  ├─ All cases COMPLIANT
  │
  └─ DECISION TO ADMIT LATENCY: 13m 15s (CEO KPI)

  DISREPANCIES:
  ├─ FALSE_NEGATIVE: Maria Rodriguez
  ├─ Action Items: 3
  ├─ QA Follow-up: Required
  │
  └─ TRAINING RECOMMENDATION: 
     Enhance neuro assessment competency
}
```

---

## PART 6: WHAT EXISTS vs WHAT'S NEEDED

### ✅ ALREADY EXISTS (in current system)

- [x] Hospital registration (RegisterForm)
- [x] Staff registration (Nurses, Physicians)
- [x] Shift creation and assignment
- [x] Clock in/out system
- [x] TriageRequest model
- [x] Encounter model
- [x] Basic dashboard routing

### ❌ NEED TO BUILD (for full simulation)

**Backend**:
- [ ] TriageAssessment.model.js (new - tracks needsBed)
- [ ] Disposition.model.js (new - physician decisions)
- [ ] TriageDiscrepancyLog.model.js (new - accuracy tracking)
- [ ] PatientJourney.model.js (new - state machine)
- [ ] Shift model enhancements (team assignment fields)
- [ ] `POST /api/shift/assign-clinical-team` endpoint
- [ ] `GET /api/user/dashboard-config` endpoint
- [ ] `POST /api/triage/submit-assessment` endpoint (with needsBed)
- [ ] `GET /api/nurse/clinical/non-hospitalized` endpoint
- [ ] `POST /api/disposition` endpoint
- [ ] `GET /api/analytics/triage-accuracy` endpoint
- [ ] WebSocket events for real-time updates

**Frontend**:
- [ ] DashboardRouter.jsx (main routing component)
- [ ] TriageNurseDashboard.jsx (queue + team assignment)
- [ ] TriageAssessmentForm.jsx (with needsBed UI)
- [ ] ClinicalNurseDashboard_ER.jsx (observation patients)
- [ ] PhysicianClinicalDashboard_ER.jsx (3-lane view)
- [ ] DispositionControl.jsx (DISCHARGE/ADMIT buttons)
- [ ] TriageAccuracyDashboard.jsx (analytics)

---

## PART 7: QUICK START CHECKLIST

### To manually test this scenario TODAY:

1. [ ] **Register Hospital** via frontend
   - Name: St. Mercy Hospital
   - Admin: Alice Johnson
   - Departments: ED, ICU, Ward

2. [ ] **Register Staff** (4 people):
   - Sarah (Triage Nurse)
   - Bob (Clinical Nurse)
   - Dr. Chen (Physician)
   - Ward Nurse (Ward Staff)

3. [ ] **Create Shift**:
   - Date: Today
   - Time: 07:00-15:00
   - Assign all 4 staff

4. [ ] **Staff Clock In**:
   - Each person logs in and clocks in

5. [ ] **Add Patients** (4 patients):
   - John (RED - chest pain)
   - Sarah Wong (GREEN - ankle)
   - Michael (YELLOW - fever)
   - Maria (YELLOW - confusion)

6. [ ] **Test Triage Workflow**:
   - Sarah triages each patient
   - Bob receives patients in clinical mode
   - Dr. Chen reviews and decides disposition

7. [ ] **Verify Metrics**:
   - Check triage accuracy
   - Verify Maria's false negative is logged
   - See shift summary

---

## PART 8: CURRENT BLOCKER ANALYSIS

**To fully execute this scenario, we need**:

1. **Team Assignment Feature** (CRITICAL)
   - Sarah needs to assign Bob + Dr. Chen
   - All dashboards update based on assignment

2. **Dashboard Router** (CRITICAL)
   - Context detection: Do I see TRIAGE, CLINICAL, or PHYSICIAN view?
   - Conditional rendering based on shift context

3. **TriageAssessmentForm** (CRITICAL)
   - Form with `needsBed` decision (YES/NO)
   - Submit creates TriageAssessment record

4. **Disposition Endpoint** (CRITICAL)
   - Physician submits decision: DISCHARGE vs ADMIT
   - Compares against triage prediction
   - Detects discrepancies

5. **Analytics Dashboard** (NICE-TO-HAVE)
   - Shows triage accuracy metrics
   - Displays discrepancies for QA review

---

## DECISION POINT FOR USER

**Current System** can support 80% of this scenario if we add:
- Team assignment modal
- Dashboard router
- Triage assessment form
- Disposition endpoint

**OR** we can build a simplified version that:
- Uses existing TriageRequest model
- Routes dashboards manually by URL parameter
- Tracks dispositions in existing Encounter model

**Recommendation**: Build the 4 features above (roughly 3-4 hours of work) for production-ready system.

**Question for user**: 
- Want to build the missing features now?
- Or test with existing system first (with workarounds)?

