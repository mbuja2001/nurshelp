# Clock-In & Real-Time EHR Integration Architecture

**Status**: Intensive Planning Phase  
**Date**: 10 April 2026  
**Objective**: Transform NurseHelp from static scheduling to dynamic mission control center

---

## 1. Strategic Overview

### Current State
- ✅ PhysicianSchedule model (stores who works when)
- ✅ NurseSchedule model (stores nurse shifts + ESI levels)
- ✅ BedSchedule model (stores bed status)
- ✅ Staff Schedule UI (add/edit/import schedules)
- ✅ Break times support (08:00-16:00 with 12:00-13:00 break)

### Target State (This Phase)
- **Real-Time Triage Dashboard**: Upon clock-in, nurse sees high-priority ESI Level 1-2 patients
- **Credential Verification**: "Can this RN handle ICU ventilator patients?"
- **Smart Auto-Assignment**: Algorithm suggests best nurse-to-patient match
- **Vital Sign Monitoring**: Auto-flag when HR/BP deviate from ranges
- **SBAR Handoff Synthesis**: Automated shift change reports
- **Surveillance Boards**: Watchlists for sepsis/falls/cardiac arrest risk

---

## 2. Data Schema Architecture

### NEW MODELS

#### 2.1 NurseProfile (extends NurseSchedule data)
```javascript
// backend/src/models/nurseProfile.model.js
// Captures what each nurse is QUALIFIED to do

{
  nurseId: String,                              // Links to NurseSchedule
  name: String,
  surname: String,
  licenseNumber: String,
  licenseType: {                                // NEW: Category level
    type: String,
    enum: ['CNA', 'LPN', 'RVN', 'APRN'],
    required: true
  },
  specialty: {                                   // NEW: Where they work
    type: String,
    enum: ['ER', 'ICU', 'OR', 'L&D', 'NICU', 'Pediatrics', 'Oncology', 'Cardio', 'Psych', 'Nephrology', 'Gerontology', 'MedSurg'],
    required: true
  },
  credentials: [                                 // NEW: What they can do
    {
      name: String,                              // e.g. "ICU Ventilator Certified", "ACLS", "PALS", "IV Therapy"
      verifiedDate: Date,
      expiryDate: Date,
      isActive: Boolean
    }
  ],
  maxESILevel: {                                 // NEW: Highest acuity they can handle
    type: Number,                                // 1 (critical) to 5 (stable)
    min: 1,
    max: 5,
    default: 5
  },
  maxBedsPerShift: Number,                       // How many patients they can manage
  experienceYears: Number,
  performanceScore: Number,                     // Based on feedback from physicians
  certifications: [String],                     // Array of cert names
  createdAt: Date,
  updatedAt: Date
}

// Indexes
db.nurseProfile.createIndex({ nurseId: 1, specialty: 1 })
db.nurseProfile.createIndex({ licenseType: 1, maxESILevel: 1 })
db.nurseProfile.createIndex({ specialty: 1, isActive: 1 })
```

#### 2.2 PatientEHRRecord (mock EHR patient data)
```javascript
// backend/src/models/patientEHRRecord.model.js
// Simulates real hospital EHR—what a nurse sees when they clock in

{
  encounterId: String,                          // Unique to THIS hospital visit
  patientId: String,
  name: String,
  surname: String,
  age: Number,
  gender: String,
  mrn: String,                                  // Medical Record Number
  
  // Current Location
  currentBedId: String,                         // Links to BedSchedule
  currentDepartment: String,                    // ER, ICU, etc
  admissionTime: Date,
  
  // Clinical Status
  esiLevel: {                                   // Emergency Severity Index
    type: Number,
    enum: [1, 2, 3, 4, 5],                      // 1=Critical, 5=Stable
    required: true
  },
  acuityScore: {                                // 0-100 scale
    type: Number,
    min: 0,
    max: 100
  },
  
  // Current Vitals (last recorded)
  latestVitals: {
    temperature: Number,                        // Celsius
    heartRate: Number,
    systolicBP: Number,
    diastolicBP: Number,
    respiratoryRate: Number,
    o2Saturation: Number,
    recordedAt: Date
  },
  
  // Vital Thresholds (alert if outside range)
  vitalThresholds: {
    hr_min: Number,                             // Default: 60
    hr_max: Number,                             // Default: 100
    bp_systolic_min: Number,                    // Default: 90
    bp_systolic_max: Number,                    // Default: 160
    resp_min: Number,                           // Default: 12
    resp_max: Number                            // Default: 20
  },
  
  // Clinical Orders
  activeOrders: [                               // Monitoring flags
    {
      orderId: String,
      orderType: String,                        // 'medication', 'procedure', 'imaging', 'lab'
      description: String,
      isHighRisk: Boolean,                      // e.g., IV antibiotics during sepsis watch
      orderedAt: Date
    }
  ],
  
  // Surveillance (Watchlist flags)
  surveillanceFlags: [
    {
      flag: String,                             // 'sepsis_risk', 'fall_risk', 'cardiac_arrest_risk', 'delirium_risk'
      severity: String,                         // 'low', 'medium', 'high'
      reason: String,
      flaggedAt: Date
    }
  ],
  
  // Chief Complaint
  chiefComplaint: String,
  symptoms: String,
  painLevel: Number,                            // 0-10
  
  // Nursing Notes (pulled from EHR)
  nursingNotes: [
    {
      noteId: String,
      createdBy: String,                        // Nurse name
      content: String,                          // SOAP format (Subjective, Objective, Assessment, Plan)
      createdAt: Date
    }
  ],
  
  // Allergies & Contraindications
  allergies: [String],
  contraindications: [String],
  
  // Assignment Status
  assignedNurseId: String,                      // Who is currently caring
  assignedPhysicianId: String,
  assignmentStartTime: Date,
  
  // Streams (for real-time monitoring)
  lastVitalUpdateAt: Date,
  vitalTrendDirection: String,                  // 'stable', 'improving', 'deteriorating'
  
  createdAt: Date,
  updatedAt: Date
}

// Indexes
db.patientEHRRecord.createIndex({ encounterId: 1 })
db.patientEHRRecord.createIndex({ currentBedId: 1 })
db.patientEHRRecord.createIndex({ esiLevel: 1, currentDepartment: 1 })
db.patientEHRRecord.createIndex({ assignedNurseId: 1 })
db.patientEHRRecord.createIndex({ "surveillanceFlags.severity": 1 })
```

#### 2.3 NurseAssignment (active shift assignments)
```javascript
// backend/src/models/nurseAssignment.model.js
// Tracks: Nurse X is assigned to patients Y and Z on shift date

{
  assignmentId: String,
  nurseId: String,                              // Links to NurseSchedule
  nurseProfile: {                               // Denormalized copy of NurseProfile
    name: String,
    licenseType: String,
    specialty: String,
    maxESILevel: Number
  },
  
  shiftDate: Date,
  shiftStartTime: Date,
  shiftEndTime: Date,
  
  assignedPatients: [
    {
      encounterId: String,                      // Links to PatientEHRRecord
      bedId: String,
      patientName: String,
      esiLevel: Number,
      acuityScore: Number,
      assignmentTime: Date,                     // When nurse took this patient
      
      // Tracking
      lastVitalCheckTime: Date,
      tasksCompleted: Number,
      tasksPending: Number
    }
  ],
  
  maxPatients: Number,                          // From NurseProfile.maxBedsPerShift
  currentPatientCount: Number,
  workloadScore: Number,                        // 0-100: 100 = fully loaded
  
  // Performance during this shift
  shiftMetrics: {
    averageResponseTime: Number,                // Minutes to respond to alert
    alertsAcknowledged: Number,
    mistakesLogged: Number,
    patientSatisfactionScore: Number
  },
  
  clockInTime: Date,
  clockOutTime: Date,
  status: {                                     // Shift progress
    type: String,
    enum: ['WAITING_TO_START', 'ACTIVE', 'ON_BREAK', 'WRAPPING_UP', 'CLOCKED_OUT']
  },
  
  createdAt: Date,
  updatedAt: Date
}

// Indexes
db.nurseAssignment.createIndex({ nurseId: 1, shiftDate: 1 })
db.nurseAssignment.createIndex({ shiftDate: 1, status: 1 })
db.nurseAssignment.createIndex({ "assignedPatients.encounterId": 1 })
```

#### 2.4 AcuityAlert (real-time vital sign violations)
```javascript
// backend/src/models/acuityAlert.model.js
// Triggers when: vital goes out of range, new order placed, surveillance flag raised

{
  alertId: String,
  encounterId: String,                          // Which patient
  nurseId: String,                              // Assigned nurse
  departmentManager: String,                    // ER Head, ICU Head, etc
  
  alertType: {
    type: String,
    enum: ['VITAL_DEVIATION', 'NEW_ORDER', 'SURVEILLANCE_FLAG', 'ACUITY_ESCALATION'],
    required: true
  },
  
  severity: {                                   // Priority for alert queue
    type: String,
    enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
    required: true
  },
  
  // Alert Details
  details: {
    parameter: String,                          // 'Heart Rate', 'BP Systolic', 'Respiratory Rate', etc
    currentValue: Number,
    normalRange: { min: Number, max: Number },
    threshold: String,                          // 'exceeded max' or 'below min'
    
    // For ORDER alerts
    orderDescription: String,
    
    // For SURVEILLANCE alerts
    riskType: String                            // 'sepsis_risk', 'fall_risk', etc
  },
  
  // Response
  createdAt: Date,
  acknowledgedAt: Date,                         // When nurse saw it
  acknowledgedBy: String,                       // Nurse name
  actionTaken: String,                          // "Checked BP manually", "Notified physician", etc
  resolvedAt: Date,
  
  // Escalation
  escalatedToPhysician: Boolean,
  escalationTime: Date,
  physicianResponse: String,
  
  status: {
    type: String,
    enum: ['NEW', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'ESCALATED'],
    default: 'NEW'
  }
}

// Indexes
db.acuityAlert.createIndex({ encounterId: 1, status: 1 })
db.acuityAlert.createIndex({ nurseId: 1, status: 1 })
db.acuityAlert.createIndex({ severity: 1, createdAt: -1 })
```

#### 2.5 HandoffReport (shift change SBAR synthesis)
```javascript
// backend/src/models/handoffReport.model.js
// Structured handoff between shift nurses (SBAR format)

{
  handoffId: String,
  date: Date,
  
  // Shift Info
  fromNurse: {
    nurseId: String,
    name: String,
    shiftEnd: Date
  },
  toNurse: {
    nurseId: String,
    name: String,
    shiftStart: Date
  },
  
  // Handoff Data (one entry per patient assigned to this nurse's shift)
  patientHandoffs: [
    {
      encounterId: String,
      patientName: String,
      bedId: String,
      
      // SBAR = Situation, Background, Assessment, Recommendation
      situation: String,                        // "64M admitted yesterday with fever, chest pain"
      background: String,                       // "PMHx: HTN, diabetes. Recent echo shows..."
      assessment: String,                       // "Currently stable, ESI 3, on oxygen"
      recommendation: String,                   // "Monitor O2 sat q1h, alert if drops below 94%"
      
      // Trend Analysis
      trendDirection: String,                   // 'improving', 'stable', 'deteriorating'
      esiLevelAtHandoff: Number,
      acuityScoreAtHandoff: Number,
      keyAlerts: [String],                      // Summary of active alerts
      
      // Current Orders
      activeOrders: [String],
      highlightedOrders: [String],               // Critical new orders
      
      // Metrics from outgoing nurse
      vitalChecksPerformed: Number,
      interventionsCompleted: [String],
      unfinishedTasks: [String]
    }
  ],
  
  // Manager Notes
  managerNotes: String,
  bedAvailability: {
    totalBeds: Number,
    occupiedBeds: Number,
    cleaningBeds: Number
  },
  
  // Acknowledgment
  acknowledgedAt: Date,
  acknowledgedBy: String,                       // Incoming nurse
  
  createdAt: Date
}

// Indexes
db.handoffReport.createIndex({ date: 1, toNurse: 1 })
db.handoffReport.createIndex({ "fromNurse.nurseId": 1, date: -1 })
```

### UPDATED MODELS

#### 2.6 Update NurseSchedule (add link to NurseProfile)
```javascript
// backend/src/models/nurseSchedule.model.js - ADD THIS FIELD

// Add field to existing schema:
nurseProfileId: {                               // NEW: Link to NurseProfile
  type: mongoose.Schema.Types.ObjectId,
  ref: 'NurseProfile'
}

// This allows: NurseSchedule → (looks up) → NurseProfile for qualifications
```

#### 2.7 Update PatientEHRRecord (add physician link)
```javascript
// Add to existing PatientEHRRecord schema:

assignedPhysicianId: String,                    // NEW: Primary physician
consultingSpecialties: [String],                // NEW: e.g., ['Cardiology', 'ID'], etc
```

---

## 3. User Flow Architecture

### FLOW 1: Clock-In & Smart Assignment

```
ACTOR: Nurse arriving for shift

1. NURSE LOGS IN
   ↓
2. SYSTEM:
   - Checks NurseSchedule for today (is nurse scheduled?)
   - Retrieves NurseProfile (what can this nurse do?)
   - Generates credential list
   ↓
3. NURSE SEES:
   → "Welcome, Sarah (RN, ICU Specialty)"
   → "Your Credentials: ACLS, PALS, ICU Ventilator Certified"
   → "Max ESI Level: 1 (Critical)"
   → Ready to Clock In? [YES] [NO]
   ↓
4. NURSE CLICKS "CLOCK IN"
   ↓
5. SYSTEM:
   - Creates NurseAssignment record (status: WAITING_TO_START)
   - Queries PatientEHRRecord for ALL patients in ICU
   - Filters: ESI Level ≤ nurse's maxESILevel (so can only see 1-3)
   - Sorts by: acuityScore DESC (sickest first)
   - Generates SUGGESTED ASSIGNMENTS
   ↓
6. NURSE SEES:
   → REAL-TIME TRIAGE DASHBOARD
   → Red banner: "ESI Level 1 Critical Patients (2)"
   → Table of available patients:
     ┌─────────────────────────────────────┐
     │ Patient | Bed  | ESI | Acuity | Dept│
     ├─────────────────────────────────────┤
     │ Mr. Chen│ICU-01│ 1  │ 95/100 │Resp │
     │ Ms. Baker│ICU-02│ 1  │ 88/100 │Cardio│
     │ Mr. Diaz│ICU-03│ 2  │ 72/100 │Post-Op│
     └─────────────────────────────────────┘
   → UI shows: "Sarah: [+Assign] [+Assign] [Accept All]"
   ↓
7. NURSE CLICKS "ASSIGN" or "ACCEPT ALL"
   ↓
8. SYSTEM:
   - Calculates workload (checking maxBedsPerShift)
   - Updates NurseAssignment.assignedPatients[]
   - Updates NurseAssignment.status = 'ACTIVE'
   - Updates PatientEHRRecord.assignedNurseId
   - Initializes monitoring (vital sign checks scheduled)
   ↓
9. NURSE ENTERS:
   → SHIFT DASHBOARD
   → Shows assigned patients with current vitals
   → Alerts panel (any deviations or new orders?)
```

### FLOW 2: Vital Sign Monitoring & Auto-Alert

```
ACTOR: System (background process)

Timing: Every 1 minute (MVPs can use every 5 min)
        Or triggered when EHR pushes new vital

EVERY MINUTE:
1. SYSTEM queries all PatientEHRRecord with assignedNurseId = nurse working
2. FOR EACH PATIENT:
   - Get latestVitals
   - Check against vitalThresholds
   
   IF vital OUT OF RANGE:
   ↓
   - Create AcuityAlert
     {
       alertType: 'VITAL_DEVIATION',
       severity: 'CRITICAL' (if hr < 40 or > 130)
                 'HIGH' (if BP systolic > 200 or < 80)
                 'MEDIUM' (other deviations),
       details: { parameter: 'Heart Rate', currentValue: 135, normalRange: {min: 60, max: 100} }
     }
   - Push to AlertQueue on nurse's dashboard
   
   IF surveillanceFlag TRIGGERED:
   ↓
   - Create AcuityAlert
     {
       alertType: 'SURVEILLANCE_FLAG',
       severity: determined by flag urgency,
       details: { riskType: 'sepsis_risk' | 'fall_risk' | 'cardiac_arrest_risk' }
     }
3. NURSE SEES:
   → Dashboard alerts section shows newest alerts first
   → Color-coded: Red (CRITICAL), Orange (HIGH), Yellow (MEDIUM)
   → Nurse clicks "Acknowledge" → sets status='ACKNOWLEDGED'
4. IF NURSE DOESN'T ACKNOWLEDGE WITHIN 5 MIN:
   → Alert escalates to Department Manager
   → System sends SMS/push alert to manager
```

### FLOW 3: Shift Handoff & SBAR

```
ACTOR: Outgoing Nurse (end of shift)
SECONDARY ACTOR: Incoming Nurse (start of shift)

1. OUTGOING NURSE:
   - Completes shift tasks
   - Clicks "Prepare Handoff"
   ↓
2. SYSTEM:
   - Gathers all patient data from NurseAssignment.assignedPatients
   - FOR EACH PATIENT:
     - Pulls latest PatientEHRRecord
     - Extracts last nursing note (for SBAR)
     - Calculates trend (improving/stable/deteriorating)
     - Lists unfinished tasks
   ↓
3. SYSTEM GENERATES:
   → HandoffReport with SBAR format:
     Patient: Mr. Chen
     Situation: "64M on day 3 post-cardiac arrest, on mechanical ventilation"
     Background: "Admitted with STEMI, underwent emergent PCI yesterday"
     Assessment: "Currently sedated, stable vitals last hour, slight improvement in O2 sat"
     Recommendation: "Continue current vent settings, monitor for arrhythmias, prepare for extubation eval tomorrow"
   ↓
4. INCOMING NURSE:
   - Clocks in (same as Flow 1)
   - Sees "Pending Handoff" notification
   - Reviews HandoffReport in SBAR format
   - Acknowledges: "I received handoff for [patient names]"
   ↓
5. SYSTEM:
   - Updates HandoffReport.acknowledgedAt
   - Updates NurseAssignment with new nurse
   - Transfers patients to incoming nurse's assignment
   - Alerts monitoring system to watch for new nurse's qualifications
```

---

## 4. Backend Architecture (New Endpoints)

### 4.1 Mock EHR Service
```
GET  /api/mockehr/patients?department=ICU&esiLevel=1,2
     - Returns all high-priority patients in ICU
POST /api/mockehr/vitals/simulate?encounterId=ID123
     - Generates realistic vital sign trends
POST /api/mockehr/order/new?encounterId=ID123
     - Simulates new clinical order being placed
```

### 4.2 NurseProfile Endpoints
```
GET  /api/nurses/profiles/:nurseId
     - Get full profile (credentials, specialty, max ESI)
POST /api/nurses/profiles
     - Create new nurse profile
PUT  /api/nurses/profiles/:nurseId
     - Update credentials, specialties
```

### 4.3 NurseAssignment Endpoints
```
POST /api/assignments/clock-in
     - Nurse clocks in, system suggests assignments
GET  /api/assignments/suggestions?nurseId=ID&date=YYYY-MM-DD
     - Get smart suggested patient assignments
POST /api/assignments/assign
     - Assign patient to nurse
GET  /api/assignments/current?nurseId=ID
     - Get nurse's current assignments & workload
POST /api/assignments/clock-out
     - Nurse ends shift, triggers handoff prep
```

### 4.4 PatientEHRRecord Endpoints
```
GET  /api/patients/ehr/:encounterId
     - Full EHR record (vitals, orders, notes, surveillance)
GET  /api/patients/high-priority?department=ER&limit=10
     - ESI 1-2 patients needing immediate attention
POST /api/patients/ehr/:encounterId/vital
     - Log new vital sign (from mock EHR or real integration)
GET  /api/patients/:encounterId/trend
     - Get acuity trend (improving/stable/deteriorating)
```

### 4.5 AcuityAlert Endpoints
```
GET  /api/alerts/active?nurseId=ID
     - Get unacknowledged alerts for nurse
POST /api/alerts/:alertId/acknowledge
     - Nurse acknowledges alert
GET  /api/alerts/:alertId/escalate
     - Escalate to physician/manager
```

### 4.6 HandoffReport Endpoints
```
POST /api/handoff/prepare?assignmentId=ID
     - System generates SBAR handoff for incoming nurse
GET  /api/handoff/:handoffId
     - Incoming nurse reviews SBAR
POST /api/handoff/:handoffId/acknowledge
     - Incoming nurse accepts handoff
```

---

## 5. Frontend Architecture (New Pages)

### 5.1 ClockInHub PAGE
```
Route: /nurse-dashboard/clock-in

Components:
- NurseWelcome Card
  Shows: Name, License Type, Specialty, Credentials, Max ESI Level
  
- Triage Dashboard (Real-Time)
  Table: All available patients in nurse's specialty
  Sortable: ESI Level, Acuity Score, Department
  Color-coded flags for RED (ESI 1-2)
  
- Smart Assignment Panel
  Shows system-recommended patients to assign
  Rationale: "Dr. Chen (RN, ICU) can handle ESI 1-2, currently has 2/4 beds"
  Quick buttons: [+Assign] [+Assign] [Accept All]
  
- Clock In Button
  [CLOCK IN] → confirms assignment → enters shift
```

### 5.2 ShiftDashboard PAGE
```
Route: /nurse-dashboard/shift

Components:
- Header
  Nurse name, Current time, Shift time remaining
  
- Assigned Patients Panel (Cards)
  ┌──────────────────────────────────┐
  │ Mr. Chen | ICU-01 | ESI 1, 95/100│
  │ Last vitals: HR 98, BP 130/80    │
  │ O2 Sat 94%, Resp 18              │
  │ Status: Stable ✓                 │
  │ [View Full | Acknowledge Alert]  │
  └──────────────────────────────────┘
  
- Alerts Queue (Sorted by Severity)
  [CRITICAL] HR 145 (Mr. Chen) 2 min ago - [ACK] [ESCALATE]
  [HIGH] New Order: IV Insulin (Ms. Garcia) 1 min ago - [ACK] [VIEW]
  
- Vital Signs Monitoring
  Real-time chart of assigned patients' vitals
  
- Workload Indicator
  "You have 3/4 beds assigned"
  Progress bar: ████░
```

### 5.3 AlertMonitoring PAGE
```
Route: /nurse-dashboard/alerts

Components:
- Alert History
  All alerts from today, sorted by newest
  Status badges: NEW | ACKNOWLEDGED | IN_PROGRESS | ESCALATED
  
- Real-Time Alert Feed
  Live stream as alerts are generated
  
- Escalation View
  If manager or physician, shows all escalated alerts across department
```

### 5.4 StaffHandoff PAGE
```
Route: /nurse-dashboard/handoff

Components:
- Incoming Handoff Section (for starting nurse)
  "You have a handoff from Sarah"
  [Review Handoff]
  
  Handoff Report (SBAR Format):
  ┌──────────────────────────────────┐
  │ Patient: Mr. Chen (ICU-01)       │
  │                                  │
  │ Situation:                       │
  │ 64M post-cardiac arrest, day 3,  │
  │ on mechanical ventilation        │
  │                                  │
  │ Background:                      │
  │ STEMI, emergent PCI yesterday    │
  │                                  │
  │ Assessment:                      │
  │ Sedated, stable, improving O2    │
  │                                  │
  │ Recommendation:                  │
  │ Monitor vent settings, watch for │
  │ arrhythmias, prep extubation...  │
  │                                  │
  │ Trend: ↗ Improving               │
  │ Unfinished Tasks:                │
  │ - Monitor vent every 2 hours     │
  │ - Check labs at 2pm              │
  │                                  │
  │ [I Acknowledge This Handoff]     │
  └──────────────────────────────────┘
  
- Outgoing Handoff Section (for ending nurse)
  "Ready to hand off your patients?"
  Auto-generated SBAR for each patient
  [Prepare & Send Handoff]
```

---

## 6. Implementation Roadmap

### Phase 1: Data Layers (Weeks 1-2)
- [ ] Create NurseProfile.model.js
- [ ] Create PatientEHRRecord.model.js  
- [ ] Create NurseAssignment.model.js
- [ ] Create AcuityAlert.model.js
- [ ] Create HandoffReport.model.js
- [ ] Update NurseSchedule to link NurseProfile
- [ ] Create MongoDB indexes
- [ ] Seed test data

### Phase 2: Mock EHR (Week 2-3)
- [ ] Create mockEHR.controller.js with:
  - Patient generator (50 realistic test patients)
  - Vital sign simulator (trends: improving/stable/deteriorating)
  - Clinical order simulator
  - Surveillance flag generator (sepsis, falls, etc)
- [ ] Create mockEHR.route.js endpoints
- [ ] Wire into server.js

### Phase 3: Business Logic (Week 3-4)
- [ ] Create nurseProfile.controller.js
- [ ] Create assignment.controller.js with smart matching algorithm
- [ ] Create acuityAlert.controller.js with alert triggering
- [ ] Create handoffReport.controller.js with SBAR synthesis
- [ ] Implement vital monitoring service (background job)

### Phase 4: API Routes (Week 4)
- [ ] Create nurseProfile.route.js
- [ ] Create assignment.route.js
- [ ] Create acuityAlert.route.js
- [ ] Create handoffReport.route.js
- [ ] Wire all into server.js

### Phase 5: Frontend (Weeks 5-6)
- [ ] Create ClockInHub.jsx
- [ ] Create ShiftDashboard.jsx
- [ ] Create AlertMonitoring.jsx
- [ ] Create StaffHandoff.jsx
- [ ] Create NurseProfile setup flow
- [ ] Integrate with navigation

### Phase 6: Testing & Integration (Week 7)
- [ ] Unit tests for assignment algorithm
- [ ] Integration tests: clock-in → assignment → alert → handoff
- [ ] Stress test: 100 patients + 20 nurses
- [ ] UI/UX testing with sample users

---

## 7. Key Algorithms

### 7.1 Smart Assignment Algorithm
```
INPUT: nurseId, date
OUTPUT: list of recommended patients to assign sorted by priority

FOR each patient in PatientEHRRecord WHERE department = nurse's specialty AND unassigned:
  
  1. CAPABILITY CHECK:
     IF patient.esiLevel > nurse.maxESILevel:
       SKIP (nurse not qualified)
  
  2. CERTIFICATION CHECK:
     IF patient has specialty requirement (e.g., "requires ACLS"):
       IF nurse doesn't have this credential:
         SKIP (nurse not certified)
  
  3. WORKLOAD CHECK:
     nurse_workload_score = (nurse.currentPatientCount / nurse.maxBedsPerShift) * 100
     IF nurse_workload_score > 85:
       Priority = 'LOW' (nurse nearly full)
     ELSE IF nurse_workload_score > 50:
       Priority = 'MEDIUM'
     ELSE:
       Priority = 'HIGH' (nurse has capacity)
  
  4. ACUITY PRIORITY:
     SORT by patient.acuityScore DESC (sickest first)
  
  5. COMPATIBILITY SCORE:
     base_score = 100
     - IF patient.esiLevel requires specialty expertise: +20 points
     - IF nurse has relevant certifications: +15 points
     - IF nurse has lower workload: +10 × (1 - workload_ratio)
     
  6. RECOMMEND:
     Return top 3-5 with highest compatibility scores
```

### 7.2 Vital Sign Threshold Alert Logic
```
EVERY 1 MINUTE (configurable):
  
FOR each patient with assignedNurseId:
  
  IF new vital signs received:
    FOR each vital parameter (HR, BP, RR, O2, Temp):
      
      IF value < patient.vitalThresholds[param_min]:
        severity = 'CRITICAL' if value < (min - 20)
                 = 'HIGH' if value < min
        CREATE AcuityAlert with severity
        
      IF value > patient.vitalThresholds[param_max]:
        severity = 'CRITICAL' if value > (max + 20)
                 = 'HIGH' if value > max
        CREATE AcuityAlert with severity
  
  ALERT ESCALATION:
    IF alert unacknowledged for 5 minutes:
      escalate to department manager
    IF alert unacknowledged for 15 minutes:
      escalate to physician + manager
```

### 7.3 Surveillance Flag Trigger Logic
```
SURVEILLANCE RULES (run on patient admission + every 30 min):

SEPSIS RISK:
  IF temp > 38.5 AND (HR > 100 OR RR > 20 OR WBC abnormal):
    CREATE flag 'sepsis_risk' with severity 'HIGH'

FALL RISK:
  IF age > 65 AND (taking sedatives OR on pain meds OR weak lower extremity):
    CREATE flag 'fall_risk' with severity 'MEDIUM'

CARDIAC ARREST RISK:
  IF (recent MI OR rhythm abnormality) AND acuity > 80:
    CREATE flag 'cardiac_arrest_risk' with severity 'CRITICAL'

DELIRIUM RISK:
  IF age > 65 AND (on multiple medications OR ICU admission OR critical illness):
    CREATE flag 'delirium_risk' with severity 'MEDIUM'
```

---

## 8. Data Integration Points

### 8.1 Real-Time Vital Sign Flow
```
Mock EHR generates vitals every ~2-5 minutes
    ↓
POST /api/mockehr/vitals → PatientEHRRecord updated
    ↓
Background job checks thresholds
    ↓
If out of range → AcuityAlert created
    ↓
Socket.io/WebSocket pushes alert to nurse's dashboard
    ↓
Nurse sees alert in real-time
```

### 8.2 Patient Assignment Flow
```
Nurse clocks in
    ↓
GET /api/assignments/suggestions returns smart matches
    ↓
Nurse clicks "Assign"
    ↓
NurseAssignment + PatientEHRRecord updated
    ↓
Vital monitoring system begins watching this patient for this nurse
    ↓
Alerts start routing to this nurse
```

### 8.3 Handoff Flow
```
Outgoing nurse ends shift
    ↓
POST /api/handoff/prepare synthesizes SBAR from nursing notes + acuity data
    ↓
Incoming nurse reviews handoff
    ↓
POST /api/handoff/acknowledge transfers patient to new nurse
    ↓
Active alert rules re-evaluate based on new nurse's qualifications
```

---

## 9. Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Clock-In Time** | < 2 min | Faster shift start = more patient contact |
| **Assignment Accuracy** | > 95% | Algorithm should rarely suggest unqualified nurse |
| **Alert Response Time** | < 5 min | Nurse acknowledges critical alert within 5 min |
| **Escalation Rate** | < 5% | Most issues resolved by nurse without escalation |
| **Handoff Completeness** | 100% | Every shift change documented in SBAR format |
| **Patient Safety Events** | Decrease 20% | Better monitoring catches deterioration faster |

---

## 10. Next Immediate Steps

1. **User Decision**: Confirm this architecture aligns with hospital requirements
2. **Phase 1 Start**: Create 5 new Mongoose models with proper indexes
3. **Mock EHR**: Generate realistic patient/vital data for testing
4. **Backend Routes**: Implement assignment suggestion algorithm
5. **Frontend**: Build ClockInHub page with real-time triage view

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
