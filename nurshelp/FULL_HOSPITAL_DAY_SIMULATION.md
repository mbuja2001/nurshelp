# FULL HOSPITAL DAY SIMULATION
## "Perfect Tuesday at St. Mercy Hospital" - 20 April 2026

**Objective**: Design complete 12-hour shift with:
- ✅ Staff arrivals + role assignments
- ✅ Patient arrivals + triage + clinical handoffs
- ✅ Physician dispositions (DISCHARGE vs ADMIT)
- ✅ Dashboard rendering by context
- ✅ Identify missing code/features

---

## PART 1: STAFF SETUP

### Hospital: St. Mercy Hospital (ID: hospital_001)
### Date: Tuesday, 20 April 2026

---

## 7:00 AM - SHIFT START (CASUAL/ED SHIFT)

### Staff Clocking In:

```
7:00:15 AM - Alice Johnson (NURSE) clocks in
├─ Hospital: St. Mercy
├─ Role: NURSE
├─ Shift: CASUALTY/ED, 7:00 AM - 3:00 PM
├─ Shift Status: "AWAITING_TEAM_ASSIGNMENT"
├─ System State: PENDING (no assignment yet)
└─ Database: Shift {
    _id: shift_001_tue_7am,
    hospitalId: hospital_001,
    date: 2026-04-20,
    shiftContext: "TRIAGE_NURSE",
    personnel: {
      triageNurseId: alice_id,
      triageNurseName: "Alice Johnson",
      clinicalNurseId: null,  ← MISSING
      physicianId: null,       ← MISSING
    },
    status: "PENDING",
    shiftStart: 2026-04-20 07:00:00
  }

7:01:30 AM - Dr. Chen (PHYSICIAN) clocks in
├─ Hospital: St. Mercy
├─ Role: PHYSICIAN
├─ Specialty: Emergency Medicine
├─ On-Call Status: AVAILABLE
├─ System State: Not assigned to shift yet
└─ Waiting in staff pool

7:03:00 AM - Bob Martinez (NURSE) clocks in
├─ Hospital: St. Mercy
├─ Role: NURSE
├─ System State: Not assigned to shift yet
└─ Waiting in staff pool
```

---

## 7:05 AM - ALICE SEES ASSIGNMENT MODAL

**Alice's Browser Experience**:

```
URL: http://localhost:5181/dashboard
GET /api/user/dashboard-config

Response: {
  error: "NO_ACTIVE_SHIFT",
  message: "No team assigned yet. Assign your clinical team to begin shift."
}

UI RENDERS: TEAM ASSIGNMENT MODAL
┌─────────────────────────────────────────┐
│ 👥 Assign Clinical Team for Today       │
├─────────────────────────────────────────┤
│                                          │
│ Select Clinical Nurse:                  │
│ [▼ Bob Martinez (LN-2847)           ]   │
│ [  Sarah Wong (LN-2849)             ]   │
│ [  James Park (LN-2851)             ]   │
│                                          │
│ Select Physician (ER Specialty):        │
│ [▼ Dr. Chen (MD-4521, Emergency)   ]   │
│ [  Dr. Patel (MD-4523, Emergency)  ]   │
│                                          │
│ [ Cancel ]  [ ✓ Confirm Team Assignment ]
└─────────────────────────────────────────┘
```

**Alice's Action** (7:06 AM):
```
POST /api/shift/assign-clinical-team
{
  shiftId: "shift_001_tue_7am",
  clinicalNurseId: "bob_id",
  clinicalNurseContext: "ER",
  physicianId: "chen_id",
  physicianContext: "ER",
  physicianSpecialty: "Emergency Medicine"
}

RESPONSE: {
  status: "TEAM_ASSIGNED",
  shift: {
    _id: "shift_001_tue_7am",
    personnel: {
      triageNurseId: "alice_id",
      triageNurseName: "Alice Johnson",
      clinicalNurseId: "bob_id",
      clinicalNurseName: "Bob Martinez",
      clinicalNurseContext: "ER",
      physicianId: "chen_id",
      physicianName: "Dr. Chen",
      physicianContext: "ER"
    },
    status: "ACTIVE",
    teamAssignedAt: "2026-04-20 07:06:00"
  },
  message: "Bob Martinez and Dr. Chen are now active for this shift"
}

WEBSOCKET EVENTS:
├─ broadcast to bob_id: 
│  { event: "shift_assigned", 
│    context: "CLINICAL_NURSE_ER",
│    dashboard: "ClinicalNurseDashboard_ER" }
│
└─ broadcast to chen_id:
   { event: "shift_assigned",
     context: "PHYSICIAN_ER",
     dashboard: "PhysicianClinicalDashboard_ER" }
```

---

## 7:07 AM - THREE DASHBOARDS NOW ACTIVE

### Dashboard 1: Alice's Dashboard (TRIAGE NURSE)
```
GET /api/user/dashboard-config
RESPONSE: {
  context: "TRIAGE_NURSE",
  component: "TriageNurseDashboard",
  shift: shift_001,
  controls: {
    canAssignTeam: true,
    canTriage: true,
    canViewClinicalMode: false,
    canRequestPhysician: false
  }
}

BROWSER RENDERS:
┌───────────────────────────────────────────────┐
│ 🏥 TRIAGE QUEUE - Emergency Department         │
├───────────────────────────────────────────────┤
│                                               │
│ Clinical Team:                                │
│ 👩‍⚕️ Clinical: Bob Martinez                    │
│ 🩺 Physician: Dr. Chen                       │
│ [Change Team]                                │
│                                               │
│ 👥 Waiting for Triage (0)                    │
│ ─────────────────────────────────────────── │
│ No patients waiting                           │
│                                               │
└───────────────────────────────────────────────┘
```

### Dashboard 2: Bob's Dashboard (CLINICAL NURSE - ER)
```
GET /api/user/dashboard-config
RESPONSE: {
  context: "CLINICAL_NURSE_ER",
  component: "ClinicalNurseDashboard_ER",
  shift: shift_001,
  controls: {
    canViewObservationPatients: true,
    canLogVitals: true,
    canRequestPhysician: true,
    canDischargeFromObservation: false
  }
}

BROWSER RENDERS:
┌───────────────────────────────────────────────┐
│ ⚕️ CLINICAL MODE - Non-Hospitalized Patients  │
├───────────────────────────────────────────────┤
│ Managing observation & awaiting disposition   │
│                                               │
│ Assigned Staff: Dr. Chen (ER)                │
│                                               │
│ ⚕️ Observation Patients (0)                   │
│ ─────────────────────────────────────────── │
│ No patients in observation                    │
│                                               │
└───────────────────────────────────────────────┘
```

### Dashboard 3: Dr. Chen's Dashboard (PHYSICIAN - ER)
```
GET /api/user/dashboard-config
RESPONSE: {
  context: "PHYSICIAN_ER",
  component: "PhysicianClinicalDashboard_ER",
  shift: shift_001,
  controls: {
    canViewClinicalMode: true,
    canReceiveEscalations: true,
    canAssignDisposition: true,
    canCreateEHRBundle: true
  }
}

BROWSER RENDERS:
┌───────────────────────────────────────────────┐
│ 🩺 PHYSICIAN COMMAND CENTRE - ER              │
├───────────────────────────────────────────────┤
│                                               │
│ 3-LANE VIEW:                                 │
│ 🔴 Active ER Cases (0)                       │
│ 🟡 Observation/Awaiting Disposition (0)     │
│ 🟣 Ward Escalations (0)                      │
│                                               │
│ Assigned Clinical Team: Bob Martinez          │
│                                               │
│ Response Time Target: <5 minutes              │
│ Compliance: --                                │
│                                               │
└───────────────────────────────────────────────┘
```

---

## 7:15 AM - FIRST PATIENT ARRIVES

### Patient 1: John Smith (DOB: 1965-03-15)

```
ARRIVAL AT TRIAGE DESK: 7:15:30 AM

Chief Complaint: "Acute chest pain for 2 hours"
Vitals (measured by receptionist):
- Temp: 37.2°C
- HR: 102 bpm (elevated)
- RR: 20 breaths/min (elevated)
- BP: 158/92 mmHg (elevated)
- O2 Sat: 96% (normal)
- Pain Scale: 8/10

Database State:
TriageRequest {
  patientId: "patient_john_001",
  firstName: "John",
  lastName: "Smith",
  chiefComplaint: "Acute chest pain",
  arrivedAt: 2026-04-20 07:15:30,
  status: "WAITING_TRIAGE",
  priority: null,  ← Not yet triaged
  needsBed: null,   ← Not yet assessed
}
```

**Alice's Queue Updates**:
```
WEBSOCKET: { event: "patient_arrived", patient: {...} }

Alice's Screen NOW SHOWS:
┌───────────────────────────────────────────────┐
│ 👥 Waiting for Triage (1)                    │
├───────────────────────────────────────────────┤
│ #1  John Smith, 59 years old                 │
│     Chief: Acute chest pain                  │
│     ⏱️  Waiting: 0m                            │
│     [📋 Triage]                              │
└───────────────────────────────────────────────┘
```

---

## 7:18 AM - ALICE TRIAGES JOHN

**Alice clicks [📋 Triage]**:

```
FORM: TriageAssessmentForm opens

Alice fills out:
├─ Chief Complaint: "Chest pain, radiating to left arm"
├─ Symptoms: ["chest pain", "shortness of breath", "nausea"]
├─ Vitals:
│  ├─ Temp: 37.2°C
│  ├─ HR: 102 bpm
│  ├─ RR: 20
│  ├─ BP: 158/92
│  ├─ O2: 96%
│
├─ Medical History: "Hypertension x10 years, smoking history"
├─ Current Meds: ["Lisinopril 10mg", "Aspirin 81mg daily"]
├─ Allergies: "NKDA"
├─ SATS Priority: RED (acute cardiac chest pain)
├─ Reasoning: "Acute onset chest pain radiating to arm, elevated vitals"
│
├─ ⭐ CRITICAL DECISION:
│  ├─ "Does patient NEED BED?" 
│  └─ Alice selects: ✓ YES
│     Reasoning: "Likely needs cardiac workup, probable admission"
│
└─ [✓ Complete Triage & Handoff to Clinical Nurse]

POST /api/triage/submit-assessment
{
  patientId: "patient_john_001",
  chiefComplaint: "Chest pain, radiating to left arm",
  symptoms: ["chest pain", "shortness of breath", "nausea"],
  vitalSigns: { temp: 37.2, hr: 102, rr: 20, systolic: 158, diastolic: 92, o2: 96 },
  medicalHistory: "Hypertension x10 years, smoking history",
  medications: ["Lisinopril 10mg", "Aspirin 81mg daily"],
  allergies: ["NKDA"],
  triagePriority: "RED",
  triageReasoning: "Acute onset chest pain radiating to arm, elevated vitals",
  needsBed: true,          ⭐ CRITICAL
  bedNeedReasoning: "Likely needs cardiac workup, probable admission"
}

BACKEND Creates:
TriageAssessment {
  _id: "triage_john_001",
  patientId: "patient_john_001",
  triageNurseId: "alice_id",
  triagePriority: "RED",
  triageReasoning: "Acute onset chest pain radiating to arm, elevated vitals",
  needsBed: true,          ⭐ KEY FIELD
  bedNeedReasoning: "Likely needs cardiac workup, probable admission",
  triageStartedAt: 2026-04-20 07:18:00,
  triageCompletedAt: 2026-04-20 07:19:30,
  status: "AWAITING_CLINICAL_NURSE",
  physicianDecision: { status: "PENDING" }
}

RESPONSE: {
  status: "TRIAGED",
  message: "Patient triaged as RED, needs bed: true"
}

WEBSOCKET EVENTS:
├─ to bob_id: { 
│   event: "patient_ready_for_clinical",
│   patientId: "patient_john_001",
│   triagePriority: "RED",
│   needsBed: true,        ⭐ Bob sees this
│   bedNeedReasoning: "Likely needs cardiac workup, probable admission",
│   assessment: "triage_john_001"
│  }
│
└─ to chen_id: {
   event: "new_case_awaiting_review",
   patientId: "patient_john_001",
   priority: "RED",
   needsAssessment: true
  }
```

**Alice's Dashboard Updates**:
```
Queue now shows:
┌───────────────────────────────────────────────┐
│ 👥 Waiting for Triage (0)                    │
├───────────────────────────────────────────────┤
│ ✓ John Smith triaged as RED, handed to Bob    │
└───────────────────────────────────────────────┘
```

---

## 7:20 AM - BOB RECEIVES JOHN IN CLINICAL MODE

**Bob's Dashboard Updates**:

```
GET /api/nurse/clinical/non-hospitalized

RESPONSE: {
  patients: [
    {
      patientId: "patient_john_001",
      firstName: "John",
      lastName: "Smith",
      currentState: "TRIAGING",
      triagePriority: "RED",
      bedNumber: "OBSERVATION_1",
      triageAssessment: {
        needsBed: true,        ⭐ Bob sees what Alice decided
        bedNeedReasoning: "Likely needs cardiac workup, probable admission",
        triagePriority: "RED"
      },
      vitals: { temp: 37.2, hr: 102, rr: 20, systolic: 158, diastolic: 92, o2: 96 },
      notes: "Acute onset chest pain radiating to arm"
    }
  ]
}

Bob's Screen NOW SHOWS:
┌─────────────────────────────────────────────────────┐
│ ⚕️ CLINICAL MODE - Non-Hospitalized Patients        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ John Smith (59M) - BED: OBSERVATION_1             │
│ 🔴 RED Priority - Chest Pain                      │
│ ─────────────────────────────────────────────     │
│ Triage Assessment: NEEDS BED (cardiac workup)     │
│ Vitals: HR 102↑, BP 158/92↑, O2 96%              │
│ Time in Queue: 4 minutes                          │
│                                                     │
│ [📊 Log Vitals] [🩺 Request Physician Review]    │
│                                                     │
└─────────────────────────────────────────────────────┘

Bob's Action: Immediately clicks [🩺 Request Physician Review]
to escalate RED priority case
```

---

## 7:22 AM - DR. CHEN GETS ALERT

**Dr. Chen's Dashboard Updates (REAL-TIME)**:

```
WEBSOCKET: { 
  event: "escalation_alert",
  patientId: "patient_john_001",
  priority: "RED",
  requestedBy: "bob_id",
  requestedAt: 2026-04-20 07:22:15,
  triageAssessment: {
    needsBed: true,
    reason: "Cardiac chest pain, elevated vitals"
  }
}

Dr. Chen's Screen NOW SHOWS:
┌─────────────────────────────────────────────────────┐
│ 🩺 PHYSICIAN COMMAND CENTRE - ER                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 🔴 ACTIVE ER CASES (1)                             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                     │
│ ⚡ URGENT - John Smith (59M)  [⏱️ 7m in triage]   │
│    Chest Pain - RED Priority                       │
│    Vitals: HR 102↑ BP 158/92↑ O2 96%              │
│                                                     │
│    ← TRIAGE NOTE: "Needs Bed" (cardiac workup)    │
│    ← CLINICAL NOTE: "Requested immediate review"  │
│                                                     │
│    [📋 Review Full Assessment]                    │
│    [✓ Admit to Ward] [✗ Discharge]               │
│                                                     │
│ 🟡 OBSERVATION (0)                                │
│ 🟣 WARD ESCALATIONS (0)                           │
│                                                     │
└─────────────────────────────────────────────────────┘

RESPONSE TIME: 7:22 - 7:18 = 4 MINUTES ✅ (Target: <5 min)
```

---

## 7:25 AM - DR. CHEN REVIEWS & DECIDES

**Dr. Chen clicks [📋 Review Full Assessment]**:

```
FULL ASSESSMENT MODAL:
┌─────────────────────────────────────────────────────┐
│ TRIAGE ASSESSMENT & PHYSICIAN REVIEW               │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Patient: John Smith, 59M                           │
│ Priority: RED (Acute Chest Pain)                   │
│                                                     │
│ Vitals: T 37.2°, HR 102↑, RR 20↑, BP 158/92↑, O2 96%
│ History: HTN x10y, smoking                         │
│ Meds: Lisinopril, Aspirin                          │
│                                                     │
│ Triage Assessment (Alice):                         │
│   "Acute onset chest pain radiating to arm,        │
│    elevated vitals. Likely needs cardiac workup,   │
│    probable admission."                            │
│                                                     │
│ Triage Decision: NEEDS BED ✓                       │
│                                                     │
│ ─────────────────────────────────────────────────  │
│ PHYSICIAN ASSESSMENT:                              │
│                                                     │
│ Clinical Impression: "Acute Coronary Syndrome     │
│ (ACS) - likely NSTEMI. Chest pain with            │
│ radiation, elevated HR/BP, risk factors present.   │
│ Needs ICU/CCU bed for continuous monitoring."     │
│                                                     │
│ DISPOSITION DECISION:                              │
│ [  ] DISCHARGE - Too risky                        │
│ [✓] ADMIT to ICU/CCU                             │
│                                                     │
│ ICU/CCU Bed Requested? [Yes - ✓ Critical Care]   │
│                                                     │
│ Notes: "ACS protocol, troponin levels needed"    │
│                                                     │
│ [Submit Disposition]                              │
│                                                     │
└─────────────────────────────────────────────────────┘

POST /api/disposition
{
  patientId: "patient_john_001",
  decision: "ADMITTED",
  wardIfAdmit: {
    wardName: "ICU/CCU",
    bedNumber: "ICU_05",
    bedType: "CRITICAL_CARE"
  },
  notes: "ACS protocol, troponin levels needed, continuous monitoring"
}

BACKEND PROCESSING:
1. Get TriageAssessment
   └─ needsBed: true ✓ (MATCHES physician decision)
   └─ No discrepancy

2. Create Disposition Record
   Disposition {
     patientId: "patient_john_001",
     physicianDecision: "ADMITTED",
     triageRecommendation: true,
     discrepancy: null,           ← No discrepancy
     decidedAt: 2026-04-20 07:25:00,
     decidedBy: "chen_id"
   }

3. Update Patient State
   PatientJourney {
     patientId: "patient_john_001",
     currentState: "HOSPITALIZED",
     location: {
       type: "WARD_BED",
       wardName: "ICU/CCU",
       bedNumber: "ICU_05"
     }
   }

4. Create EHR Bundle (INPATIENT)
   - Includes all 5 objects
   - Stays open (not finalized)
   - Ready for ongoing care documentation

5. Notify Ward Nurses
   WEBSOCKET to ICU_ward_nurses: {
     event: "new_patient_admitted",
     patientId: "patient_john_001",
     bedNumber: "ICU_05",
     triageAssessment: { priority: "RED", needsBed: true }
   }

RESPONSE: {
  status: "DISPOSITION_RECORDED",
  decision: "ADMITTED",
  discrepancy: null,
  message: "✓ Patient admitted to ICU/CCU"
}
```

**Dr. Chen's Dashboard Updates**:
```
John Smith REMOVED from "🔴 ACTIVE ER CASES"
Now shows: 🔴 ACTIVE ER CASES (0)

System broadcasts to Bob:
├─ event: "patient_admitted"
├─ patientId: "patient_john_001"
├─ bedNumber: "ICU_05"
└─ message: "John Smith admitted to ICU/CCU - transferred out of your observation"

Bob's Dashboard Updates:
┌─────────────────────────────────────────────────────┐
│ ⚕️ CLINICAL MODE - Non-Hospitalized Patients        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ ✓ John Smith admitted to ICU/CCU (Bed ICU_05)     │
│                                                     │
│ No patients in observation                          │
│                                                     │
└─────────────────────────────────────────────────────┘

Metrics Recorded:
├─ Decision to Admit Latency: 9 minutes 30 seconds
│  (from triage start 07:18 to disposition 07:25)
├─ Physician Response Time: 4 minutes ✅
├─ Triage Accuracy: TRUE POSITIVE (triage said needs bed, physician admitted)
└─ Metrics Dashboard entry
```

---

## 7:35 AM - SECOND PATIENT ARRIVES

### Patient 2: Sarah Wong (DOB: 1992-07-22)

```
ARRIVAL AT TRIAGE: 7:35:00 AM

Chief Complaint: "Ankle sprain from fall at home"
Vitals:
- Temp: 36.8°C
- HR: 78 bpm (normal)
- RR: 16 (normal)
- BP: 120/80 (normal)
- O2: 98% (normal)
- Pain: 5/10

Alice's Queue:
┌───────────────────────────────────────────────┐
│ 👥 Waiting for Triage (1)                    │
├───────────────────────────────────────────────┤
│ #1  Sarah Wong, 34 years old                 │
│     Chief: Ankle sprain from fall             │
│     ⏱️  Waiting: 0m                            │
│     [📋 Triage]                              │
└───────────────────────────────────────────────┘
```

---

## 7:38 AM - ALICE TRIAGES SARAH

```
Alice fills Assessment Form:
├─ Chief: "Ankle sprain, unable to bear weight"
├─ Symptoms: ["ankle pain", "swelling", "difficulty walking"]
├─ Vitals: All normal
├─ History: "No medical conditions, no meds, NKDA"
├─ SATS Priority: GREEN (minor injury, stable vitals)
├─ Reasoning: "Ankle sprain, vital signs normal, no signs of serious injury"
│
├─ ⭐ CRITICAL DECISION: "Does patient NEED BED?"
│  └─ Alice selects: ✗ NO
│     Reasoning: "Ankle sprain can be managed in observation, likely discharged same day"
│
└─ [✓ Complete Triage & Handoff to Clinical Nurse]

POST /api/triage/submit-assessment
{
  patientId: "patient_sarah_001",
  chiefComplaint: "Ankle sprain from fall",
  ...vitals and history...,
  triagePriority: "GREEN",
  triageReasoning: "Ankle sprain, vital signs normal, no serious injury",
  needsBed: false,         ⭐ CRITICAL - "NO BED NEEDED"
  bedNeedReasoning: "Can be managed in observation, likely discharged same day"
}

WEBSOCKET to bob_id: {
  event: "patient_ready_for_clinical",
  patientId: "patient_sarah_001",
  triagePriority: "GREEN",
  needsBed: false,         ⭐ Bob sees this
  bedNeedReasoning: "Can be managed in observation, likely same-day discharge"
}
```

**Bob's Dashboard Updates**:
```
Now shows 2 patients:

┌─────────────────────────────────────────────────────┐
│ ⚕️ CLINICAL MODE - Non-Hospitalized Patients        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Sarah Wong (34F) - BED: OBSERVATION_2              │
│ 🟢 GREEN Priority - Ankle Sprain                   │
│ ─────────────────────────────────────────────────  │
│ Triage Assessment: NO BED NEEDED (same-day d/c)   │
│ Vitals: All normal (HR 78, BP 120/80, O2 98%)     │
│ Time in Queue: 2 minutes                           │
│                                                     │
│ [📊 Log Vitals] [🩺 Request Physician Review]     │
│                                                     │
│ (John Smith - ADMITTED to ICU/CCU)                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 8:15 AM - THIRD PATIENT: DISCHARGE CASE

### Patient 3: Michael Chen (DOB: 1978-11-05)

```
ARRIVAL: 8:15 AM
Chief: "Mild fever, body aches, no respiratory distress"
Vitals: Temp 38.2°C, HR 85, BP 125/81, O2 97%

Alice triages (8:18 AM):
├─ Priority: YELLOW (infection, stable), no bed needed
├─ Alice decides: ✗ NO BED NEEDED
│  "Appears to be viral infection, vitals stable, can go home with rest"
│
└─ Hands to Bob for observation

Bob monitors (8:20-9:30 AM):
├─ Patient rests in observation
├─ Vitals remain stable
├─ No deterioration

Dr. Chen reviews (9:35 AM):
├─ Notes: "Stable, likely viral, safe to discharge"
├─ Decision: DISCHARGE
├─ Triage said: NO BED NEEDED ✓
├─ Physician decision: DISCHARGE ✓
│
└─ RESULT: TRUE NEGATIVE (both agree: no admission)
    ✓ Agreement on disposition
    ✓ Perfect triage prediction

EHR Bundle: AMBULATORY
└─ Patient goes home with discharge instructions
```

---

## 11:30 AM - FOURTH PATIENT: THE TRICKY ONE (DISCREPANCY)

### Patient 4: Maria Rodriguez (DOB: 1955-08-12)

```
ARRIVAL: 11:30 AM
Chief: "Mild headache, confusion for 2 hours"
Vitals: Temp 37.5°C, HR 88, BP 138/85, O2 96%

Alice triages (11:33 AM):
├─ Priority: YELLOW (confusion needs evaluation)
├─ Initial assessment: "Probably dehydration & headache"
├─ Alice decides: ✗ NO BED NEEDED
│  "Mild symptoms, vitals stable, can observe and discharge"
│
└─ Hands to Bob

Medical Context Bob Discovers (11:45 AM):
├─ Bob logs vitals more carefully
├─ Finds: Patient on warfarin (blood thinner)
├─ Finds: Recent fall yesterday, slight bruising on head
├─ Patient mentions: "Feeling more confused now"
│
└─ Bob escalates immediately: "Physician review urgent"

Dr. Chen reviews (11:50 AM):
├─ Sees: Warfarin + confusion + head trauma = potential INTRACRANIAL BLEED
├─ Clinical concern: "Cannot safely discharge, needs CT head + ICU hold"
├─ Decision: ADMIT to ICU for neuro monitoring
│
├─ Triage said: NO BED NEEDED
├─ Physician decided: ADMIT to ICU
│
└─ ⚠️ DISCREPANCY DETECTED!

DISCREPANCY ANALYSIS:
Disposition {
  patientId: "patient_maria_001",
  physicianDecision: "ADMITTED",
  triageRecommendation: false,
  discrepancy: {
    hasDiscrepancy: true,
    triagePrediction: "NO_BED_NEEDED",
    physicianDecision: "ADMITTED",
    type: "FALSE_NEGATIVE",
    reason: "Condition deteriorated after triage: warfarin + head trauma + worsening confusion → neurology concern",
    costImpact: 3200  ← Potential readmit cost
  },
  decidedAt: 2026-04-20 11:50:00
}

TriageDiscrepancyLog.create({
  triageAssessmentId: "triage_maria_001",
  dispositionId: "disposition_maria_001",
  discrepancy: { type: "FALSE_NEGATIVE", ... },
  hospitalId: "hospital_001",
  date: 2026-04-20,
  confidenceAdjustment: -0.15  ← Triage model confidence down 15%
})

Dr. Chen's Notes: "Maria's confusion + warfarin + head trauma not caught in triage. 
Recommend: (1) Better neuro screening in triage, (2) Medication interaction check"
```

**QA Alert**:
```
TRIAGE ACCURACY DASHBOARD (11:52 AM):
┌─────────────────────────────────────────────────────┐
│ ⚠️ DISCREPANCY DETECTED                             │
├─────────────────────────────────────────────────────┤
│ Patient: Maria Rodriguez                            │
│ Time: 11:50 AM (17 min after triage)                │
│ Issue: FALSE NEGATIVE (missed admission need)       │
│                                                     │
│ Triage: "No bed needed"                             │
│ Physician: "Admit ICU - neuro workup"              │
│                                                     │
│ Root Cause: Warfarin + confusion + head trauma      │
│            not fully assessed in triage             │
│                                                     │
│ Action Item: Team huddle on medication interaction  │
│ QA Follow-up: Flag triage nurse Alice for coaching  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 2:00 PM - END OF SHIFT METRICS

### Summary Dashboard (Alice's Shift: 7:00 AM - 3:00 PM)

```
POST /api/analytics/triage-accuracy
GET /api/metrics/shift-performance

RESPONSE: {
  shift: "shift_001_tue_7am",
  hospital: "St. Mercy Hospital",
  date: "2026-04-20",
  shiftDuration: "8 hours",
  
  TRIAGE METRICS:
  ├─ Total Patients Triaged: 4
  ├─ RED Priority: 1 (John)
  ├─ YELLOW Priority: 2 (Michael, Maria)
  ├─ GREEN Priority: 1 (Sarah)
  │
  └─ BED REQUIREMENT PREDICTIONS:
     ├─ Said "Needs Bed": 1 (John)
     │  └─ Physician admitted: 1 ✓ TRUE POSITIVE (100%)
     │
     └─ Said "No Bed": 3 (Sarah, Michael, Maria)
        ├─ Physician discharged: 2 ✓ TRUE NEGATIVE (Sarah, Michael)
        ├─ Physician admitted: 1 ✗ FALSE NEGATIVE (Maria)
        └─ FALSE NEGATIVE RATE: 33%

  PHYSICIAN RESPONSE METRICS:
  ├─ John (RED): 4 minutes 30 sec → ✓ COMPLIANT (<5 min)
  ├─ Sarah (GREEN): Not escalated → N/A
  ├─ Michael (YELLOW): 1h 15m → ✓ COMPLIANT (<15 min)
  ├─ Maria (YELLOW → URGENT): 17 min response → ✓ COMPLIANT
  │
  └─ Average Response Time: 27 minutes

  DECISION TO ADMIT LATENCY:
  ├─ John: 9m 30s (triaged 07:18, admitted 07:25) ✓ Excellent
  ├─ Maria: 17m 0s (triaged 11:33, admitted 11:50) ✓ Excellent
  │
  └─ Average Decision to Admit: 13m 15s ★ CEO KPI

  DISPOSITION SUMMARY:
  ├─ Discharged: 2 patients (Sarah, Michael)
  ├─ Admitted: 2 patients (John→ICU/CCU, Maria→ICU)
  ├─ Admission Rate: 50%
  │
  └─ Total Bed Hours: ICU: 16 (projected for John), ICU: 8 (Maria)

  DISREPANCIES:
  ├─ FALSE NEGATIVES: 1 (Maria)
  ├─ FALSE POSITIVES: 0
  ├─ ACCURACY RATE: 75% (3 of 4 correct predictions)
  │
  └─ ACTION ITEMS:
     ├─ Flag: Alice needs coaching on medication interaction screening
     ├─ Improvement: Add warfarin + head trauma + confusion → auto-escalate
     └─ Training: Neuro assessment competency review

  EHR SYNC:
  ├─ AMBULATORY bundles: 2 (Sarah, Michael)
  ├─ INPATIENT bundles: 2 (John, Maria)
  ├─ EHR export scheduled: 3:15 PM
  └─ Med-e-Mass integration: queue 4 records for sync
}
```

---

## PART 2: MISSING CODE/FEATURES CHECKLIST

### ✅ ALREADY EXISTS (or stubbed)

- [x] Shift model with basic timestamps
- [x] Nurse model with roles
- [x] Patient arrival flow
- [x] Basic vitals logging

### ❌ NEED TO BUILD (for this simulation to work)

```
BACKEND CODE NEEDED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DATABASE MODELS
   [ ] Shift.model.js - Enhance with team assignment fields
   [ ] TriageAssessment.model.js - NEW (with needsBed field)
   [ ] PatientJourney.model.js - NEW (tracks state machine)
   [ ] Disposition.model.js - NEW (physician decisions)
   [ ] TriageDiscrepancyLog.model.js - NEW (accuracy tracking)

2. BACKEND APIS
   [ ] POST /api/shift/assign-clinical-team
   [ ] GET /api/user/dashboard-config
   [ ] GET /api/staff/available-for-shift
   [ ] POST /api/triage/patient-arrived (add to queue)
   [ ] POST /api/triage/submit-assessment (with needsBed)
   [ ] GET /api/nurse/triage/queue
   [ ] GET /api/nurse/clinical/non-hospitalized
   [ ] POST /api/disposition
   [ ] GET /api/analytics/triage-accuracy

3. WEBSOCKET EVENTS
   [ ] shift_assigned
   [ ] patient_arrived
   [ ] patient_ready_for_clinical
   [ ] new_case_awaiting_review
   [ ] escalation_alert
   [ ] patient_admitted
   [ ] patient_discharged
   [ ] discrepancy_alert

FRONTEND CODE NEEDED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DASHBOARD ROUTER
   [ ] DashboardRouter.jsx - Main entry point, route by context
   [ ] dashboardRouter.config.js - Role → dashboard mapping

2. TRIAGE COMPONENTS
   [ ] TriageNurseDashboard.jsx - Queue + team assignment modal
   [ ] TriageAssessmentForm.jsx - Form with needsBed decision
   [ ] BedRequirementModal.jsx - Yes/No bed decision UI

3. CLINICAL COMPONENTS
   [ ] ClinicalNurseDashboard_ER.jsx - Observation patient list
   [ ] WardNurseDashboard.jsx - Bed map grid
   [ ] VitalsLogger.jsx - Log vitals for patient

4. PHYSICIAN COMPONENTS
   [ ] PhysicianClinicalDashboard_ER.jsx - 3-lane view
   [ ] DispositionControl.jsx - DISCHARGE vs ADMIT buttons
   [ ] PhysicianAssessmentModal.jsx - Full review + decision

5. ANALYTICS/QA
   [ ] TriageAccuracyDashboard.jsx - Show discrepancies
   [ ] ShiftMetricsDashboard.jsx - KPIs at end of shift

6. UTILITIES
   [ ] dashboardRouter.service.js - Helper logic
   [ ] TriageAssessmentService.js - CRUD for assessments
   [ ] WebSocket/event handlers - Real-time updates
```

---

## PART 3: CRITICAL DEPENDENCIES (Must build in this order)

```
1. DATABASE LAYER
   ├─ Shift model enhancements
   ├─ TriageAssessment.model.js creation
   ├─ PatientJourney.model.js creation
   └─ Disposition.model.js creation
   
   BLOCKED BY: DB connection, mongoose setup
   PRIORITY: 🔴 CRITICAL - builds foundation

2. BACKEND API LAYER
   ├─ GET /api/staff/available-for-shift (simple queries)
   ├─ POST /api/shift/assign-clinical-team (update shift)
   ├─ GET /api/user/dashboard-config (routing logic)
   ├─ POST /api/triage/submit-assessment (create assessment)
   ├─ GET /api/nurse/clinical/non-hospitalized (query)
   └─ POST /api/disposition (complex logic + EHR integration)
   
   BLOCKED BY: Models layer complete
   PRIORITY: 🔴 CRITICAL - powers dashboards

3. FRONTEND ROUTING LAYER
   ├─ DashboardRouter.jsx (calls dashboard-config endpoint)
   └─ dashboardRouter.config.js (mapping)
   
   BLOCKED BY: Backend APIs complete
   PRIORITY: 🟡 HIGH - enables conditional rendering

4. DASHBOARD COMPONENT LAYER
   ├─ TriageNurseDashboard.jsx (uses /api/nurse/triage/queue)
   ├─ TriageAssessmentForm.jsx (posts to /api/triage/submit-assessment)
   ├─ ClinicalNurseDashboard_ER.jsx (uses /api/nurse/clinical/non-hospitalized)
   ├─ PhysicianClinicalDashboard_ER.jsx (uses /api/physician/clinical/active-cases)
   └─ DispositionControl.jsx (posts to /api/disposition)
   
   BLOCKED BY: Routing layer complete + Backend APIs
   PRIORITY: 🟡 HIGH - main UI

5. WEBSOCKET/REAL-TIME LAYER
   ├─ Event handlers for all dashboard updates
   ├─ Real-time patient queue updates
   └─ Escalation notifications
   
   BLOCKED BY: Dashboard components
   PRIORITY: 🟢 NICE-TO-HAVE - polish/UX

6. ANALYTICS LAYER
   ├─ TriageAccuracyDashboard.jsx
   ├─ GET /api/analytics/triage-accuracy endpoint
   └─ Discrepancy tracking
   
   BLOCKED BY: All other layers
   PRIORITY: 🟢 NICE-TO-HAVE - reporting
```

---

## PART 4: TEST SCENARIOS READY

### Scenario 1 (Already Described Above)
✅ Full 8-hour shift with 4 patients
✅ Multiple priorities (RED, YELLOW, GREEN)
✅ Mix of outcomes (ADMIT, DISCHARGE)
✅ Shows discrepancy detection (Maria case)

### Scenario 2: Multiple Shifts (Afternoon)
- 3:00 PM - Night shift crew arrives
- Another Triage Nurse, different Clinical Nurse, different Physician
- System should handle multiple concurrent shifts
- Test that dashboards are isolated per shift

### Scenario 3: Escalation Flow
- Patient arriving as GREEN (low priority)
- Vitals deteriorate in observation
- Bob escalates to Dr. Chen
- Dr. Chen promotes patient to RED lane
- Admission follows

### Scenario 4: Bed Capacity
- 4 patients arrive
- Only 2 ICU beds available
- System should queue admission requests
- Beds release when patients discharge/transfer

---

## HOW TO USE THIS DOCUMENT

**Step 1**: Review this full simulation above
**Step 2**: User tells us which layer to implement first
**Step 3**: We build real code for that layer
**Step 4**: We test with simulated data matching this scenario
**Step 5**: Repeat for next layer

**Example**:
- User says: "Build the data models first"
- We create: Shift.model.js, TriageAssessment.model.js, etc.
- We test with: Seed 4 patients from scenario above
- We verify: Dashboard configs can be fetched, queues populate, etc.

---

## READY TO BUILD?

**Question for user**: Which layer should we build first?

1. **Database Models** (10 min) - Foundation
2. **Backend APIs** (20 min) - Powers everything
3. **Frontend Router** (10 min) - Dashboard routing
4. **Dashboard Components** (30 min) - Actual UI
5. **WebSocket/Real-time** (15 min) - Polish
6. **Analytics** (10 min) - Reporting

**My Recommendation**: Build in order 1→2→3→4→5→6
- Each layer unblocks the next
- We can test after each layer
- Simulation scenario validates everything

Pick a layer! 🚀
