# Shift-Based Conditional Dashboard System

## 🎯 How It Works (Complete Flow)

### The Problem We're Solving
Staff members have a **general role** (NURSE or PHYSICIAN), but on any given day they can have **different responsibilities** based on their shift assignment.

**Example:**
- Alice is a NURSE (general role)
- On Tuesday, she's assigned as a **TRIAGE_NURSE** 
- So she should see the Triage dashboard, NOT the clinical nurse dashboard
- On Wednesday, she might be a **CLINICAL_NURSE_ER**
- So she should see the clinical nursing dashboard, NOT triage

**The solution: Dashboard is determined by shift assignment, not general role**

---

## 🔄 Complete Workflow

### STEP 1: Manager Assigns Clinical Team (7:06 AM)
```bash
POST /api/shift/assign-clinical-team
```

**Request:**
```json
{
  "shiftId": "shift_67890",
  "triageNurseId": "alice_5c1f2",
  "clinicalNurseId": "bob_5c1f3",
  "clinicalNurseContext": "CLINICAL_NURSE_ER",
  "physicianId": "dr_chen_5c1f4",
  "physicianContext": "PHYSICIAN_ER"
}
```

**Backend Updates Shift Document:**
```javascript
// In MongoDB:
shift = {
  _id: "shift_67890",
  shiftName: "ED-MORNING-TUESDAY",
  status: "PENDING" → "ACTIVE",  // ← Update
  teamAssignment: {               // ← NEW
    triageNurseId: "alice_5c1f2",
    clinicalNurseId: "bob_5c1f3",
    clinicalNurseContext: "CLINICAL_NURSE_ER",
    physicianId: "dr_chen_5c1f4",
    physicianContext: "PHYSICIAN_ER",
    teamAssignedAt: "2026-04-21T07:06:00Z"
  },
  shiftContext: "TRIAGE"
}
```

---

### STEP 2: Staff Member Logs In (7:15 AM)
```bash
POST /api/auth/login
{
  "email": "alice.chen@stmercy.com",
  "password": "***"
}
```

**Returns JWT Token:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "alice_5c1f2",
    "name": "Alice Chen",
    "role": "NURSE",  // ← General role
    "department": "EMERGENCY"
  }
}
```

---

### STEP 3: Frontend Calls Dashboard Config Endpoint
```bash
GET /api/dashboard/config
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

### STEP 4: Backend Decodes JWT & Checks Shift Assignment ⭐

**Backend Logic in `getDashboardConfig()`:**

```javascript
// 1. Extract user ID from JWT token (middleware did this)
const userId = req.user.id;  // "alice_5c1f2"
const hospitalId = req.hospitalId;

// 2. Get staff member's basic info
const staff = await ClinicalStaff.findById(userId);
// staff.role = "NURSE" (general role)

// 3. Find TODAY's shift for the hospital
const today = new Date();
today.setHours(0, 0, 0, 0);
const shift = await Shift.findOne({ 
  hospitalId, 
  date: { $gte: today, $lt: tomorrow } 
});
// shift.teamAssignment = { triageNurseId: "alice_5c1f2", ... }

// 4. CHECK WHICH ROLE USER HAS IN SHIFT ASSIGNMENT ⭐
const userIdString = userId.toString();  // "alice_5c1f2"
const triageNurseId = shift.teamAssignment?.triageNurseId?.toString();

if (userIdString === triageNurseId) {
  assignedRole = "TRIAGE_NURSE";  // ← This is what determines dashboard!
  dashboardContext = "TRIAGE_NURSE";
} else if (userIdString === shift.teamAssignment?.clinicalNurseId?.toString()) {
  assignedRole = "CLINICAL_NURSE";
  dashboardContext = "CLINICAL_NURSE_ER";
} else if (userIdString === shift.teamAssignment?.physicianId?.toString()) {
  assignedRole = "PHYSICIAN";
  dashboardContext = "PHYSICIAN_ER";
} else {
  // User NOT assigned to any role - BLOCK ACCESS
  return 403 Forbidden "Not assigned to any role in today's shift"
}
```

---

### STEP 5: Return Role-Specific Dashboard Config

**For Alice (Triage Nurse):**
```json
{
  "success": true,
  "config": {
    "context": "TRIAGE_NURSE",
    "components": [
      "TriageHeader",
      "PatientQueue",
      "TriageForm",
      "VitalsCapture"
    ],
    "controls": {
      "acceptPatient": true,
      "triagePatient": true,
      "routeToPhysician": true,
      "recordVitals": true
    }
  }
}
```

**For Bob (Clinical Nurse):**
```json
{
  "success": true,
  "config": {
    "context": "CLINICAL_NURSE_ER",
    "components": [
      "ClinicalNurseHeader",
      "ObservationPatientList",
      "VitalsMonitoring",
      "MedicationLog",
      "PhysicianNotifications"
    ],
    "controls": {
      "recordVitals": true,
      "escalateConcerns": true,
      "markForDisposition": true
    }
  }
}
```

**For Dr. Chen (Physician):**
```json
{
  "success": true,
  "config": {
    "context": "PHYSICIAN_ER",
    "components": [
      "PhysicianHeader",
      "TriagePatientLane",
      "ActiveCasesLane",
      "ObservationPatientLane",
      "PatientDetails",
      "DispositionPanel"
    ],
    "controls": {
      "setDisposition": true,
      "orderTests": true,
      "prescribeMedications": true
    }
  }
}
```

---

### STEP 6: Frontend Renders Conditional Dashboard

**DashboardRouter.jsx:**
```javascript
function DashboardRouter() {
  const [config, setConfig] = useState(null);

  useEffect(() => {
    // Call backend to get config
    axios.get('/api/dashboard/config').then(res => {
      setConfig(res.data.config);
    });
  }, []);

  // Render based on context
  switch (config.context) {
    case 'TRIAGE_NURSE':
      return <TriageNurseDashboard config={config} />;
    case 'CLINICAL_NURSE_ER':
      return <ClinicalNurseDashboard config={config} />;
    case 'PHYSICIAN_ER':
      return <PhysicianDashboard config={config} />;
    case 'NO_SHIFT':
      return <OffDutyDashboard config={config} />;
  }
}
```

---

## 🔐 Permission Matrix - What Each Role Sees

### Alice (TRIAGE_NURSE)
| Component | Visible | Reason |
|-----------|:-------:|--------|
| Patient Queue | ✅ | Part of TriageNurseDashboard |
| Triage Form | ✅ | Can perform triage assessment |
| Vitals Capture | ✅ | Can record vital signs |
| Medication Log | ❌ | Not in component list |
| Disposition Panel | ❌ | Only for physicians |
| Test Orders | ❌ | Only for physicians |

### Bob (CLINICAL_NURSE_ER)
| Component | Visible | Reason |
|-----------|:-------:|--------|
| Patient Queue | ❌ | Not in component list |
| Triage Form | ❌ | Not in component list |
| Vital Signs Monitoring | ✅ | Part of ClinicalNurseDashboard |
| Medication Log | ✅ | Can record meds given |
| Disposition Panel | ❌ | Only for physicians |
| Escalate to Physician | ✅ | Control for escalation |

### Dr. Chen (PHYSICIAN_ER)
| Component | Visible | Reason |
|-----------|:-------:|--------|
| Triage Queue (Lane 1) | ✅ | Can see waiting patients |
| Active Cases (Lane 2) | ✅ | Can see assessing patients |
| Observation (Lane 3) | ✅ | Can see observation patients |
| Disposition Panel | ✅ | ONLY role with disposition |
| Test Orders | ✅ | Can order tests |
| Prescribe Meds | ✅ | Can prescribe |

---

## ⚠️ Error Scenarios

### Scenario 1: Staff Not Assigned to Any Role
```
GET /api/dashboard/config
Authorization: Bearer <alice_token>
```

**Response (403 Forbidden):**
```json
{
  "success": false,
  "message": "You are not assigned to any role in today's shift. Contact your manager.",
  "shift": {
    "shiftName": "ED-MORNING-TUESDAY",
    "shiftStart": "07:00",
    "shiftEnd": "19:00"
  }
}
```

**Frontend shows:** "You're not assigned to today's shift. Contact your manager."

---

### Scenario 2: Staff Has No Shift Today
```
GET /api/dashboard/config
Authorization: Bearer <staff_token>
```

**Response (200 OK, but NO_SHIFT context):**
```json
{
  "success": true,
  "config": {
    "context": "NO_SHIFT",
    "components": ["OffDutyMessage", "UpcomingShifts"]
  }
}
```

**Frontend shows:** Off-duty dashboard with upcoming shifts list

---

### Scenario 3: Invalid JWT Token (Missing or Expired)
```
GET /api/dashboard/config
Authorization: Bearer invalid_token
```

**Response (401 Unauthorized):**
```json
{
  "success": false,
  "message": "User not authenticated"
}
```

**Frontend redirects to:** Login page

---

## 📋 Database State Changes

### Before Team Assignment (7:00 AM)
```javascript
{
  shiftName: "ED-MORNING-TUESDAY",
  status: "PENDING",
  teamAssignment: null
}
```

### After Team Assignment (7:06 AM)
```javascript
{
  shiftName: "ED-MORNING-TUESDAY",
  status: "ACTIVE",
  teamAssignment: {
    triageNurseId: ObjectId("alice_5c1f2"),
    clinicalNurseId: ObjectId("bob_5c1f3"),
    physicianId: ObjectId("dr_chen_5c1f4"),
    teamAssignedAt: ISODate("2026-04-21T07:06:00Z")
  }
}
```

---

## 🔄 What Happens Next?

### When Alice Refreshes Page (7:15 AM)
1. Frontend calls `GET /api/dashboard/config`
2. Backend:
   - Gets Alice's ID from JWT
   - Finds Tuesday shift
   - Checks: `shift.teamAssignment.triageNurseId === alice_id` ✅
   - Returns `context: "TRIAGE_NURSE"`
3. Frontend renders TriageNurseDashboard
4. Alice sees: Patient Queue + Triage Form + Vitals

### When Bob Refreshes Page (7:15 AM)
1. Frontend calls `GET /api/dashboard/config`
2. Backend:
   - Gets Bob's ID from JWT
   - Finds Tuesday shift
   - Checks: `shift.teamAssignment.clinicalNurseId === bob_id` ✅
   - Returns `context: "CLINICAL_NURSE_ER"`
3. Frontend renders ClinicalNurseDashboard
4. Bob sees: Observation Patients + Vitals Monitoring + Medication Log

### When Dr. Chen Refreshes Page (7:15 AM)
1. Frontend calls `GET /api/dashboard/config`
2. Backend:
   - Gets Dr. Chen's ID from JWT
   - Finds Tuesday shift
   - Checks: `shift.teamAssignment.physicianId === dr_chen_id` ✅
   - Returns `context: "PHYSICIAN_ER"`
3. Frontend renders PhysicianDashboard
4. Dr. Chen sees: 3-lane layout (Triage Queue, Active Cases, Observation)

---

## ✅ Key Implementation Points

✓ **JWT decoding** happens in auth middleware → `req.user.id`
✓ **Shift lookup** by today's date and hospital
✓ **Role check** compares `req.user.id` against `shift.teamAssignment` IDs
✓ **Context-specific config** returned based on matched role
✓ **Frontend conditional rendering** based on `config.context`
✓ **Error handling** for unassigned staff (403 Forbidden)
✓ **Audit logging** of team assignments

---

## 📝 Testing Validation

To validate this system works:

1. ✅ Assign team via `POST /api/shift/assign-clinical-team`
2. ✅ Login as Alice → Call `GET /api/dashboard/config` → Should get TRIAGE_NURSE
3. ✅ Logout, Login as Bob → Call `GET /api/dashboard/config` → Should get CLINICAL_NURSE_ER
4. ✅ Logout, Login as Dr. Chen → Call `GET /api/dashboard/config` → Should get PHYSICIAN_ER
5. ✅ Frontend renders correct dashboard for each role
6. ✅ Controls/permissions enforced based on role

---

## 🎬 The "Perfect Tuesday" Journey

- **7:00 AM**: Alice logs in → calls dashboard/config → NO shift yet → sees OffDutyDashboard
- **7:06 AM**: Manager calls assign-clinical-team → updates shift.teamAssignment
- **7:15 AM**: 
  - Alice refreshes → NOW sees TriageNurseDashboard
  - Bob refreshes → NOW sees ClinicalNurseDashboard
  - Dr. Chen refreshes → NOW sees PhysicianDashboard
- **7:20 AM**: Patient arrives → Alice triages → routes to Dr. Chen
- **7:20-7:30 AM**: Dr. Chen assesses patient
- **8:00 AM**: Diagnosis recorded, patient goes to observation with Bob
- **8:30 AM**: Patient disposition complete

---

**Status:** Implementation ready for testing ✅
**Testing Location:** TESTING_AND_VALIDATION_GUIDE.md
