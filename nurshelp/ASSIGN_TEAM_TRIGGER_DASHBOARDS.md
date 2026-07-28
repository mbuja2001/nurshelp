# ASSIGN CLINICAL TEAM - Triggers the Different Dashboards

## What Backend is Waiting For

When you click **"Confirm Team Assignment"** in the UI, it sends:

```
POST /api/shift/assign-clinical-team
```

### Expected Request Body

```json
{
  "shiftId": "shift_001_mon",
  "triageNurseId": "alice_id",
  "clinicalNurseId": "bob_id",
  "clinicalNurseContext": "CLINICAL_NURSE_ER",
  "physicianId": "chen_id",
  "physicianContext": "PHYSICIAN_ER",
  "physicianSpecialty": "Emergency Medicine"
}
```

---

## What Happens Inside Backend

### File: `backend/src/controllers/shift.controller.js` (Line 626)

**Function:** `assignClinicalTeam()`

**Steps:**
1. ✅ Validates shift exists
2. ✅ Verifies all staff members exist in database
3. ✅ **Updates `shift.teamAssignment`** with the three assigned roles
4. ✅ Changes shift status: `PENDING` → `ACTIVE`
5. ✅ Sets `shiftContext` based on contexts (TRIAGE, ICU, SURGICAL, WARD)
6. ✅ Returns updated shift

**Database Change:**
```javascript
// BEFORE
Shift {
  status: "PENDING",
  teamAssignment: null  // or empty
}

// AFTER
Shift {
  status: "ACTIVE",           // ← Changed from PENDING
  teamAssignment: {
    triageNurseId: ObjectId("alice_id"),
    clinicalNurseId: ObjectId("bob_id"),
    clinicalNurseContext: "CLINICAL_NURSE_ER",
    physicianId: ObjectId("chen_id"),
    physicianContext: "PHYSICIAN_ER",
    teamAssignedAt: 2026-04-21T07:06:00.000Z
  }
}
```

---

## How This Triggers Different Dashboards

### The Golden Rule
The dashboard shown depends on **which ID in `teamAssignment` matches your user ID**.

### Step 1: Alice Logs In
```
GET /api/dashboard/config
Authorization: Bearer <alice_jwt_token>

Backend Logic:
├─ Extracts userId from JWT (alice_id)
├─ Finds today's shift
├─ Checks shift.teamAssignment:
│  ├─ Is alice_id === teamAssignment.triageNurseId? YES
│  └─ → Assign role: "TRIAGE_NURSE"
└─ Returns config with context: "TRIAGE_NURSE"

Response: {
  context: "TRIAGE_NURSE",
  components: ["TriageHeader", "PatientQueue", "TriageForm", ...],
  controls: {
    triagePatient: true,
    acceptPatient: true,
    ...
  }
}

Frontend: Renders TriageNurseDashboard ✓
```

### Step 2: Bob Logs In
```
GET /api/dashboard/config
Authorization: Bearer <bob_jwt_token>

Backend Logic:
├─ Extracts userId from JWT (bob_id)
├─ Finds today's shift
├─ Checks shift.teamAssignment:
│  ├─ Is bob_id === teamAssignment.triageNurseId? NO
│  ├─ Is bob_id === teamAssignment.clinicalNurseId? YES
│  └─ → Assign role: "CLINICAL_NURSE" with context: "CLINICAL_NURSE_ER"
└─ Returns config

Response: {
  context: "CLINICAL_NURSE_ER",
  components: ["ClinicalHeader", "ObservationList", "VitalsMonitoring", ...],
  controls: {
    recordVitals: true,
    escalateConcerns: true,
    ...
  }
}

Frontend: Renders ClinicalNurseDashboard (ER version) ✓
```

### Step 3: Dr. Chen Logs In
```
GET /api/dashboard/config
Authorization: Bearer <chen_jwt_token>

Backend Logic:
├─ Extracts userId from JWT (chen_id)
├─ Finds today's shift
├─ Checks shift.teamAssignment:
│  ├─ Is chen_id === teamAssignment.triageNurseId? NO
│  ├─ Is chen_id === teamAssignment.clinicalNurseId? NO
│  ├─ Is chen_id === teamAssignment.physicianId? YES
│  └─ → Assign role: "PHYSICIAN" with context: "PHYSICIAN_ER"
└─ Returns config

Response: {
  context: "PHYSICIAN_ER",
  components: ["PhysicianHeader", "ActiveCases3Lane", "Escalations", ...],
  controls: {
    canAssignDisposition: true,
    receiveEscalations: true,
    ...
  }
}

Frontend: Renders PhysicianDashboard (ER 3-lane view) ✓
```

### Step 4: Someone Else Logs In (Not Assigned)
```
GET /api/dashboard/config
Authorization: Bearer <unassigned_person_jwt>

Backend Logic:
├─ Extracts userId from JWT (unassigned_id)
├─ Finds today's shift
├─ Checks shift.teamAssignment:
│  ├─ Is unassigned_id === triageNurseId? NO
│  ├─ Is unassigned_id === clinicalNurseId? NO
│  ├─ Is unassigned_id === physicianId? NO
│  └─ No match!
└─ Returns 403 Forbidden

Response: {
  success: false,
  message: "You are not assigned to any role in today's shift. Contact your manager.",
  code: 403
}

Frontend: Shows error message "Contact your manager" ✗
```

---

## API Testing Timeline

### Precondition
- ✅ Backend running on port 5001
- ✅ Hospital registered
- ✅ Alice (NURSE), Bob (NURSE), Dr. Chen (PHYSICIAN) registered
- ✅ Shift created (status: PENDING)

### Test Sequence

**BEFORE team assignment:**
```bash
curl -X GET http://localhost:5001/api/dashboard/config \
  -H "Authorization: Bearer <alice_token>"

# Returns 403: "You are not assigned to any role"
```

**ASSIGN THE TEAM:**
```bash
curl -X POST http://localhost:5001/api/shift/assign-clinical-team \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "shiftId": "SHIFT-xxx",
    "triageNurseId": "alice_id",
    "clinicalNurseId": "bob_id",
    "clinicalNurseContext": "CLINICAL_NURSE_ER",
    "physicianId": "chen_id",
    "physicianContext": "PHYSICIAN_ER"
  }'

# Returns: shift with updated teamAssignment, status: ACTIVE
```

**NOW Alice sees her dashboard:**
```bash
curl -X GET http://localhost:5001/api/dashboard/config \
  -H "Authorization: Bearer <alice_token>"

# Returns: { context: "TRIAGE_NURSE", components: [...] }
```

**NOW Bob sees his dashboard:**
```bash
curl -X GET http://localhost:5001/api/dashboard/config \
  -H "Authorization: Bearer <bob_token>"

# Returns: { context: "CLINICAL_NURSE_ER", components: [...] }
```

**NOW Dr. Chen sees his dashboard:**
```bash
curl -X GET http://localhost:5001/api/dashboard/config \
  -H "Authorization: Bearer <chen_token>"

# Returns: { context: "PHYSICIAN_ER", components: [...] }
```

---

## Error Scenarios

### ❌ Error 1: shiftId not found
```
POST /api/shift/assign-clinical-team with non-existent shiftId

Response:
{
  success: false,
  message: "Shift not found"
}
```

### ❌ Error 2: Staff member doesn't exist
```
POST /api/shift/assign-clinical-team with invalid staff IDs

Response:
{
  success: false,
  message: "One or more staff members not found in the system"
}
```

### ❌ Error 3: Staff not assigned (calling dashboard endpoint)
```
GET /api/dashboard/config as staff member NOT in teamAssignment

Response:
{
  success: false,
  message: "You are not assigned to any role in today's shift. Contact your manager.",
  code: 403
}
```

---

## The Complete Flow In Order

```
1. Manager creates shift (status: PENDING)
   ↓
2. Manager assigns team:
   POST /api/shift/assign-clinical-team
   ↓
3. Backend updates shift.teamAssignment & status → ACTIVE
   ↓
4. Alice refreshes / logs in
   ├─ Calls GET /api/dashboard/config
   ├─ Match found: alice_id === teamAssignment.triageNurseId
   ├─ Returns: { context: "TRIAGE_NURSE" }
   └─ Sees: TriageNurseDashboard ✓
   ↓
5. Bob refreshes / logs in
   ├─ Calls GET /api/dashboard/config
   ├─ Match found: bob_id === teamAssignment.clinicalNurseId
   ├─ Returns: { context: "CLINICAL_NURSE_ER" }
   └─ Sees: ClinicalNurseDashboard ✓
   ↓
6. Dr. Chen refreshes / logs in
   ├─ Calls GET /api/dashboard/config
   ├─ Match found: chen_id === teamAssignment.physicianId
   ├─ Returns: { context: "PHYSICIAN_ER" }
   └─ Sees: PhysicianDashboard ✓
```

---

## Key Implementation Details

### File: `backend/src/routes/dashboard.route.js`
- Route: `GET /api/dashboard/config`
- Protected: ✅ Requires valid JWT token
- Header: `Authorization: Bearer <token>`

### File: `backend/src/routes/shift.route.js`  
- Route: `POST /api/shift/assign-clinical-team`
- Protected: ✅ Requires valid JWT token
- Protected: ✅ Requires MANAGER or ADMIN role
- Endpoint line: 113

### Database Model: `backend/src/models/shift.model.js`
- Field: `teamAssignment` (Object)
  - `triageNurseId` (ObjectId reference to ClinicalStaff)
  - `clinicalNurseId` (ObjectId reference to ClinicalStaff)
  - `clinicalNurseContext` (String: "CLINICAL_NURSE_ER", "CLINICAL_NURSE_ICU", etc)
  - `physicianId` (ObjectId reference to ClinicalStaff)
  - `physicianContext` (String: "PHYSICIAN_ER", "PHYSICIAN_ICU", "PHYSICIAN_SURGICAL")
  - `teamAssignedAt` (Date)

### Dashboard Logic: `backend/src/controllers/dashboard.controller.js`
- Function: `getDashboardConfig()` (Line 18)
- Check: Compares `userId` from JWT against all three IDs in `shift.teamAssignment`
- Returns: Role-specific configuration based on which ID matched
- If no match: Returns 403 Forbidden

---

## Summary

**The user story:**
1. Manager clicks "Assign Team" in UI with selected staff
2. Frontend sends POST to `/api/shift/assign-clinical-team`
3. Backend writes `shift.teamAssignment` to database
4. Each staff member's GET `/api/dashboard/config` now returns DIFFERENT config
5. Frontend renders the appropriate dashboard based on config.context

**The trigger:** Setting `shift.teamAssignment` in the database enables the different dashboards to appear.
