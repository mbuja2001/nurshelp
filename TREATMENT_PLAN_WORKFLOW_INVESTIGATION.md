# 🔍 TREATMENT PLAN WORKFLOW - COMPLETE INVESTIGATION

## 📋 Overview

When a physician diagnoses a patient on **DiagnosePage.jsx**, they can create a comprehensive treatment plan with:
- **Treatments/Interventions** (Medications, Procedures, Therapy, Monitoring, Imaging, Labs)
- **Recommendations** (Admission, Discharge, Follow-up, etc.)
- **Medication Requests** (for pharmacy workflow)
- **Progression Notes** (clinical documentation)

This document traces the COMPLETE workflow from UI input → Backend processing → Department communications.

---

## 🎯 WORKFLOW ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHYSICIAN DIAGNOSIS FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

1. PHYSICIAN ACTION (DiagnosePage.jsx)
   ↓
   ├─ Enters Diagnosis text
   ├─ Adds Treatment/Interventions (with dosage, frequency, route)
   ├─ Adds Recommendations
   ├─ Adds Medication Requests (for pharmacy)
   └─ Adds Progression Notes (clinical documentation)
   ↓
2. SAVE TRIGGERS (Two options)
   ├─ persistDiagnosis() → PUT /api/encounters/{id}/diagnosis
   └─ persistInpatientProgressReport() → PUT /api/encounters/{id}/inpatient-report
   ↓
3. BACKEND PROCESSING (recordDiagnosis controller)
   ├─ Validates physician has authority over encounter
   ├─ Updates encounter.diagnosis with physician clinical notes
   ├─ Stores treatments array with prescribed_by, prescribed_at metadata
   ├─ Stores medication_requests array with pharmacy workflow metadata
   ├─ Stores recommendations with recorded_at timestamp
   ├─ Updates status to 'DIAGNOSED'
   ├─ Records in status_history
   └─ Returns encounter with all new data
   ↓
4. POST-DIAGNOSIS ACTIONS (Triggered separately)
   ├─ Medication Ordering (PhysicianDashboard.jsx)
   │  └─ handleOrderMedication() → calls /api/pharmacy endpoints
   │
   ├─ Lab Requests (PhysicianDashboard.jsx)
   │  └─ handleRequestLabs() → POST /api/lab-requests/{id}/create
   │
   ├─ Surgery Requests (PhysicianDashboard.jsx)
   │  └─ handleSurgeryRequest() → creates surgery_records in encounter
   │
   └─ Disposition (DiagnosePage.jsx)
      └─ handleDisposition() → HOSPITALIZE, DISCHARGE, or OBSERVE
```

---

## 📊 DETAILED ENDPOINT FLOW

### 1️⃣ TREATMENT PLAN SUBMISSION

#### **Frontend - DiagnosePage.jsx**
```javascript
// State at submission:
const [diagnosisText, setDiagnosisText] = useState("");
const [treatments, setTreatments] = useState([]);  // Array of intervention objects
const [recommendations, setRecommendations] = useState([]);
const [medicationRequests, setMedicationRequests] = useState([]);
const [progressionNote, setProgressionNote] = useState("");

// Treatment object structure:
{
  name: "Ibuprofen",              // e.g., "Chest X-ray", "ECG Monitoring"
  intervention_type: "MEDICATION",  // MEDICATION | PROCEDURE | THERAPY | MONITORING | IMAGING | LAB
  dosage: "500mg",                // e.g., "600mA", "Continuous"
  frequency: "Q4-6h",             // e.g., "Every 4-6 hours", "Stat", "PRN"
  route: "Oral",                  // Oral | IV | IM | SC | Topical | Inhalation
  indication: "Fever reduction",  // Clinical reason for this intervention
  status: "PRESCRIBED"            // Set on backend
}

// Medication request structure:
{
  medication_name: "Aspirin",
  dosage: "325mg",
  frequency: "Once daily",
  route: "Oral",
  indication: "Cardiac prophylaxis"
}

// Call to backend:
const persistDiagnosis = async () => {
  const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/diagnosis`, {
    method: "PUT",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      diagnosis: diagnosisText,
      treatment_plan: "",  // Built from treatments array
      treatments: treatments,
      recommendations: recommendations,
      medication_requests: medicationRequests,
      progression_notes: [{ content: progressionNote, note_type: "ASSESSMENT" }]
    })
  });
}
```

#### **Backend - recordDiagnosis Controller (encounter.controller.js:1354)**

**Endpoint**: `PUT /api/encounters/{id}/diagnosis`

**Flow**:
```javascript
export const recordDiagnosis = async (req, res) => {
  // 1. AUTHORIZATION CHECK
  // ✓ Validate user has clinical_handover for this encounter
  const clinicianHandover = encounter.clinical_handovers?.find(h => 
    h.clinician_id?.toString() === req.user._id.toString()
  );
  if (!clinicianHandover) throw "Not authorized";

  // 2. UPDATE ENCOUNTER FIELDS
  encounter.diagnosis = diagnosis;  // Physician's clinical notes
  encounter.treatment_plan = treatment_plan || '';
  encounter.physician_notes = notes || '';
  encounter.status = 'DIAGNOSED';

  // 3. PROCESS TREATMENTS ARRAY
  if (Array.isArray(treatments) && treatments.length > 0) {
    encounter.treatments = treatments.map(t => ({
      ...t,
      prescribed_by: clinicianHandover.clinician_name,  // Physician name from handover
      prescribed_at: new Date()
    }));
  }
  // treatments stored at: encounter.treatments[]
  // Each with fields: intervention_type, name, dosage, frequency, route, indication, prescribed_by, prescribed_at, status

  // 4. PROCESS RECOMMENDATIONS ARRAY
  if (Array.isArray(recommendations) && recommendations.length > 0) {
    encounter.recommendations = recommendations.map(r => ({
      ...r,
      recommended_by: clinicianHandover.clinician_name,
      recommended_at: new Date()
    }));
  }

  // 5. PROCESS MEDICATION REQUESTS ARRAY (⭐ KEY FOR PHARMACY WORKFLOW)
  if (Array.isArray(medication_requests) && medication_requests.length > 0) {
    encounter.medication_requests = medication_requests
      .filter(isMedicationRequestValid)  // Validation utility
      .map((request) => ({
        ...normalizeMedicationRequest(request),  // Normalize data
        requested_by: clinicianHandover.clinician_name,
        requested_at: new Date()
        // Status: PENDING (awaiting pharmacy review)
      }));
  }
  // medication_requests stored at: encounter.medication_requests[]
  // Schema fields: medication_name, dosage, frequency, route, indication, requested_by, requested_at, status: PENDING|APPROVED|DISPENSED|DECLINED, pharmacy_reference

  // 6. UPDATE PHYSICIAN REFERENCE
  encounter.assigned_physician = {
    id: req.user._id,
    name: clinicianHandover.clinician_name,
    specialty: clinicianHandover.clinician_specialty,
    diagnosed_at: new Date()
  };

  // 7. RECORD STATUS CHANGE
  encounter.status_history.push({
    status: 'DIAGNOSED',
    changed_by: req.user._id,
    changed_at: new Date(),
    notes: `${treatments?.length || 0} treatments + ${recommendations?.length || 0} recommendations`
  });

  // 8. MARK HANDOVER AS IN_PROGRESS
  clinicianHandover.handover_status = 'IN_PROGRESS';

  await encounter.save();
  return { message: "Diagnosis recorded successfully", encounter };
}
```

**Data Stored in MongoDB**:
```javascript
encounter = {
  _id: ObjectId,
  status: 'DIAGNOSED',
  diagnosis: "Patient presenting with acute bronchitis, fever, cough...",
  treatment_plan: "",
  physician_notes: "...",
  
  treatments: [
    {
      intervention_type: "MEDICATION",
      name: "Amoxicillin",
      dosage: "500mg",
      frequency: "Every 8 hours",
      route: "Oral",
      indication: "Bacterial infection treatment",
      prescribed_by: "Dr. Sarah Chen",
      prescribed_at: 2026-08-12T10:30:00Z,
      status: "PRESCRIBED",  // Can change to: ADMINISTERED, COMPLETED, HOLD, DISCONTINUED
    },
    {
      intervention_type: "IMAGING",
      name: "Chest X-ray",
      dosage: "",
      frequency: "STAT",
      route: "N/A",
      indication: "Confirm pneumonia vs bronchitis",
      prescribed_by: "Dr. Sarah Chen",
      prescribed_at: 2026-08-12T10:30:00Z,
      status: "PRESCRIBED"
    }
  ],
  
  medication_requests: [
    {
      medication_name: "Aspirin",
      dosage: "325mg",
      frequency: "Once daily",
      route: "Oral",
      indication: "Cardiac prophylaxis",
      requested_by: "Dr. Sarah Chen",
      requested_at: 2026-08-12T10:30:00Z,
      status: "PENDING",  // Pharmacy will change to APPROVED/DECLINED/DISPENSED
      pharmacy_reference: "",  // Filled by pharmacist
      review_notes: ""  // Pharmacist adds notes if declined
    }
  ],
  
  recommendations: [
    {
      description: "Admit to general ward for observation",
      recommendation_type: "ADMISSION",
      priority: "HIGH",
      recommended_by: "Dr. Sarah Chen",
      recommended_at: 2026-08-12T10:30:00Z
    }
  ],
  
  assigned_physician: {
    id: ObjectId,
    name: "Dr. Sarah Chen",
    specialty: "Internal Medicine",
    diagnosed_at: 2026-08-12T10:30:00Z
  },
  
  clinical_handovers: [
    {
      clinician_id: ObjectId,
      clinician_name: "Dr. Sarah Chen",
      clinician_specialty: "Internal Medicine",
      handover_status: "IN_PROGRESS",  // ← Updated by recordDiagnosis
      handover_type: "PHYSICIAN_DIAGNOSIS",
      assigned_at: 2026-08-12T09:00:00Z
    }
  ],
  
  status_history: [
    { status: 'TRIAGE_COMPLETE', ... },
    { status: 'SEEN', ... },
    { status: 'DIAGNOSED', changed_by: ..., changed_at: ..., notes: "3 treatments + 2 recommendations" }
  ]
}
```

---

### 2️⃣ PHARMACY WORKFLOW (Medication Requests)

After diagnosis is recorded, medication_requests are now in **PENDING** status.

#### **Pharmacy Review Queue**
```
GET /api/pharmacy/pending-medication-requests
  → Pharmacist sees all PENDING medication_requests from all encounters
  → Shows: medication name, dosage, frequency, patient, indication, requested_by, timestamp
```

#### **Pharmacist Action**
```javascript
// Pharmacist reviews and approves/declines medication request
PUT /api/pharmacy/encounters/{encounterId}/medication-request/{requestId}
  body: {
    status: "APPROVED" | "DECLINED",  // ← Change from PENDING
    review_notes: "Dose appropriate, in stock"
  }
```

**Backend Handler** (pharmacy.controller.js:605):
```javascript
export const updateMedicationRequestStatus = async (req, res) => {
  // Find encounter
  const encounter = await Encounter.findById(encounterId);
  
  // Find medication request
  const request = encounter.medication_requests.find(r => r._id === requestId);
  
  // Update status
  request.status = status;  // PENDING → APPROVED/DECLINED/DISPENSED
  request.review_notes = review_notes;
  request.reviewed_at = new Date();
  request.pharmacy_reference = req.user?.name;  // Pharmacist name
  
  // 🔔 SEND NOTIFICATION TO PHYSICIAN
  if (status !== 'PENDING') {
    const assignedPhysician = encounter.assigned_physician;
    await Notification.create({
      encounter_id: encounter._id,
      clinician_id: assignedPhysician.id,
      clinician_name: assignedPhysician.name,
      title: status === 'DECLINED' 
        ? 'Medication request DECLINED by pharmacy'
        : `Medication request ${status} by pharmacy`,
      priority_code: status === 'DECLINED' ? 'RED' : 'YELLOW',
      status: 'UNREAD'
    });
  }
  
  await encounter.save();
}
```

**Lifecycle**:
```
PENDING → APPROVED → DISPENSED (by nurse) → END
       ↘ DECLINED → Pharmacist sends notification to physician
```

---

### 3️⃣ LAB REQUEST WORKFLOW

#### **Physician Creates Lab Request**
```
POST /api/lab-requests/{encounterId}/create
  body: {
    specialty: "RADIOLOGY" | "PATHOLOGY",
    lab_type: "Chest X-ray" | "Complete Blood Count",
    description: "Confirm pneumonia, rule out viral etiology",
    priority: "URGENT" | "HIGH" | "MEDIUM" | "LOW"
  }
```

**Backend Handler** (lab.controller.js:createLabRequest):
```javascript
export const createLabRequest = async (req, res) => {
  const encounter = await Encounter.findById(encounterId);
  
  const newLabRequest = {
    request_id: new ObjectId(),
    specialty: "RADIOLOGY",  // or PATHOLOGY
    lab_type: "Chest X-ray",
    description: "...",
    priority: "MEDIUM",
    requested_by: "Dr. Sarah Chen",
    requested_at: new Date(),
    status: "PENDING",  // Specialist will change this
    created_at: new Date(),
    updated_at: new Date()
  };
  
  encounter.lab_requests.push(newLabRequest);
  
  // Update status history
  encounter.status_history.push({
    status: 'LAB_REQUEST_CREATED_RADIOLOGY',
    changed_by: req.user.id,
    notes: 'RADIOLOGY lab requested: Chest X-ray'
  });
  
  await encounter.save();
}
```

**Data in Encounter**:
```javascript
encounter.lab_requests = [
  {
    request_id: ObjectId,
    specialty: "RADIOLOGY",
    lab_type: "Chest X-ray",
    description: "Confirm pneumonia, rule out viral etiology",
    priority: "MEDIUM",
    requested_by: "Dr. Sarah Chen",
    requested_at: 2026-08-12T10:30:00Z,
    status: "PENDING",  // → ACKNOWLEDGED → COMPLETED
    created_at: 2026-08-12T10:30:00Z,
    updated_at: 2026-08-12T10:30:00Z,
    
    // When specialist completes:
    acknowledged_at: 2026-08-12T10:35:00Z,
    acknowledged_by: "Dr. Radiologist Name",
    
    report: "Findings: Bilateral pneumonic infiltrates...",
    report_submitted_at: 2026-08-12T11:15:00Z,
    
    reviewed_by_physician: false,
    reviewed_at: null
  }
]
```

#### **Specialist Workflow**

1️⃣ **Radiologist sees pending queue**:
```
GET /api/lab-requests?specialty=RADIOLOGY&status=PENDING
  → Shows all pending RADIOLOGY requests across hospital
```

2️⃣ **Radiologist acknowledges request**:
```
PATCH /api/lab-requests/{encounterId}/{requestId}/acknowledge
  → Status: PENDING → ACKNOWLEDGED
  → Radiologist name recorded
```

3️⃣ **Radiologist completes report**:
```
PATCH /api/lab-requests/{encounterId}/{requestId}/submit-report
  body: {
    report: "Findings: Clear bilateral lung fields, no evidence of pneumonia"
  }
  → Status: ACKNOWLEDGED → COMPLETED
  → Report stored in encounter.lab_requests[].report
```

4️⃣ **Physician reviews report**:
```
PATCH /api/lab-requests/{encounterId}/{requestId}/review
  body: {
    reviewed: true,
    review_notes: "Normal findings, discharge patient"
  }
```

**Lifecycle**:
```
PENDING (physician requested) 
  → ACKNOWLEDGED (specialist accepted)
  → COMPLETED (specialist submitted report)
  → REVIEWED (physician reviewed findings)
```

---

### 4️⃣ SURGERY REQUEST WORKFLOW

#### **Physician Creates Surgery Request**
```
POST /api/encounters/{encounterId}/surgery-record
  body: {
    procedure_name: "Appendectomy",
    surgery_type: "EMERGENCY" | "URGENT" | "ELECTIVE",
    priority: "URGENT" | "HIGH" | "MEDIUM" | "LOW",
    rationale: "Acute appendicitis with risk of rupture",
    scheduled_at: "2026-08-12T14:00:00Z",
    surgeon: "Dr. James Smith",
    surgical_team: "2 Surgical Nurses, 1 Anesthesiologist",
    notes: "Patient prepped, NPO since midnight"
  }
```

**Data in Encounter**:
```javascript
encounter.surgery_records = [
  {
    surgery_id: ObjectId,
    procedure_name: "Appendectomy",
    surgery_type: "EMERGENCY",
    rationale: "Acute appendicitis with risk of rupture",
    priority: "URGENT",
    requested_by: "Dr. Sarah Chen",
    requested_at: 2026-08-12T10:30:00Z,
    scheduled_at: 2026-08-12T14:00:00Z,
    surgeon: "Dr. James Smith",
    surgical_team: "2 Surgical Nurses, 1 Anesthesiologist",
    status: "SCHEDULED",  // → IN_PROGRESS → COMPLETED
    notes: "Patient prepped, NPO since midnight",
    
    started_at: null,
    completed_at: null,
    outcome: null,
    complications: null
  }
]
```

---

## 🎨 TREATMENT PLAN TYPES & ROUTES

### Treatment Types:
| Type | Examples | Route Options | Dosage Format |
|------|----------|---------------|---------------|
| **MEDICATION** | Ibuprofen, Aspirin | Oral, IV, IM, SC, Topical, Inhalation | "500mg", "5ml", "1 tablet" |
| **PROCEDURE** | Catheter insertion, Wound care | N/A | N/A |
| **THERAPY** | Physical therapy, Occupational therapy | N/A | Sessions/day |
| **MONITORING** | ECG, Vital signs, Blood glucose | Continuous, Periodic | Frequency |
| **IMAGING** | Chest X-ray, CT Scan, MRI, Ultrasound | STAT, Routine | mA, Contrast needed |
| **LAB** | Blood work, Urinalysis, Culture | Routine, STAT | Sample size |

### Medication Routes:
- **Oral** - By mouth
- **IV** - Intravenous
- **IM** - Intramuscular
- **SC** - Subcutaneous
- **Topical** - On skin/surface
- **Inhalation** - Breathed in

---

## 🔄 COMPLETE WORKFLOW SUMMARY

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHYSICIAN TREATMENT PLAN → SPECIALIST COMMUNICATION                          │
└──────────────────────────────────────────────────────────────────────────────┘

STEP 1: Physician enters diagnosis & treatments
        ↓
STEP 2: PUT /api/encounters/{id}/diagnosis
        └─ Backend stores treatments[], medication_requests[], recommendations[]
        └─ Status → DIAGNOSED
        ↓
STEP 3A: Medication Request Workflow
         │
         ├─ Medication stored in encounter.medication_requests[]
         ├─ Pharmacist reviews: GET pending requests
         ├─ Pharmacist approves/declines: PUT status update
         ├─ Notification sent to physician
         ├─ Nurse dispenses: POST /api/pharmacy/dispense
         └─ Status: PENDING → APPROVED → DISPENSED or DECLINED
         
STEP 3B: Lab Request Workflow
         │
         ├─ Lab request stored in encounter.lab_requests[]
         ├─ Radiologist/Pathologist reviews queue
         ├─ Specialist acknowledges: PATCH /acknowledge
         ├─ Specialist submits report: PATCH /submit-report
         ├─ Physician reviews: PATCH /review
         └─ Status: PENDING → ACKNOWLEDGED → COMPLETED → REVIEWED
         
STEP 3C: Surgery Request Workflow
         │
         ├─ Surgery stored in encounter.surgery_records[]
         ├─ Surgeon reviews pending surgeries
         ├─ Surgery scheduled and confirmed
         └─ Status: SCHEDULED → IN_PROGRESS → COMPLETED

STEP 4: Treatments Administered
        ├─ Nurse administers medication: recorded in medication_administrations[]
        ├─ Nurse performs procedure
        ├─ Clinician records vital signs for monitoring
        └─ Status updates: PRESCRIBED → ADMINISTERED → COMPLETED
```

---

## 🔌 API ENDPOINT REFERENCE

### Treatment Plan Submission:
- `PUT /api/encounters/{id}/diagnosis` - Record diagnosis with treatments
- `PUT /api/encounters/{id}/inpatient-report` - Progress report for hospitalized patients

### Medication Workflow:
- `POST /api/pharmacy/pending-medication-requests` - Get queue for pharmacist
- `PUT /api/pharmacy/encounters/{encounterId}/medication-request/{requestId}` - Approve/Decline
- `POST /api/pharmacy/dispense` - Dispense medication to patient

### Lab Workflow:
- `POST /api/lab-requests/{encounterId}/create` - Physician requests lab
- `GET /api/lab-requests?specialty=RADIOLOGY&status=PENDING` - Specialist queue
- `PATCH /api/lab-requests/{encounterId}/{requestId}/acknowledge` - Specialist accepts
- `PATCH /api/lab-requests/{encounterId}/{requestId}/submit-report` - Specialist submits
- `PATCH /api/lab-requests/{encounterId}/{requestId}/review` - Physician reviews

### Surgery Workflow:
- `POST /api/encounters/{id}/surgery-record` - Physician requests surgery
- `PATCH /api/encounters/{id}/surgery/{surgeryId}` - Update surgery status

---

## ❓ KEY OBSERVATIONS & QUESTIONS

### ✅ What's Working:
1. Treatment plans are being stored correctly in database
2. Medication requests flow to pharmacist queue
3. Lab requests flow to specialist department
4. Status tracking and audit trails are comprehensive
5. Notifications are sent when actions occur

### ❓ What Needs Investigation:
1. **Acknowledgment Logic**: When physician diagnosing patient → should this auto-create handover?
   - Currently: Requires pre-existing clinical_handover to record diagnosis
   - Question: Can a physician start diagnosis workflow without prior handover?

2. **Treatment Plan Display**: In confirmation pages, are treatments being properly normalized?
   - Check: Are routes (Oral, IV, etc.) being displayed correctly?
   - Check: Are dosages/frequencies showing?

3. **Communication Chain**: When medication is approved by pharmacy, does it flow back to nurse for administration?
   - Currently: Notification sent to physician, but does nurse get notified?

4. **Lab Priority Handling**: Does "URGENT" lab request get prioritized in specialist queue?
   - Currently: Stored in database, but no priority-based sorting in API

5. **Medication Request Validation**: What validation happens when physician creates requests?
   - Currently: Basic existence checks, but no drug interaction checking

---

## 🎯 NEXT STEPS FOR PERFECTING WORKFLOW

1. **Automatic Handover Creation** - When physician starts diagnosis, ensure clinical_handover exists
2. **Nurse Notification** - Notify nurse when medication is approved so they can administer
3. **Lab Priority Queue** - Sort pending labs by priority in specialist dashboard
4. **Drug Interaction Checking** - Add validation when requesting medications
5. **Treatment Status UI** - Display live status of all treatments (PRESCRIBED → ADMINISTERED → COMPLETED)
6. **Specialist Assignment** - Allow physician to specify which radiologist/pathologist should handle lab
7. **Surgery Coordination** - Integrate with OR scheduling system
