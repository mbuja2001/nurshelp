# Full Day Simulation - Testing & Validation Guide

**IMPORTANT:** Do not commit until all tests pass and user journey is validated!

## 🧪 Testing Phases

### PHASE 1: Backend Setup & Validation (No Commits)

#### Step 1: Verify Backend Starts ✅
```bash
cd backend
npm install  # If needed
node src/server.js
```

**Check:**
- [ ] Server starts without errors
- [ ] MongoDB connects: `✅ MongoDB connected`
- [ ] Routes register properly
- [ ] No syntax errors in console

**If fails:**
- [ ] Check `.env` file has `MONGO_URI`
- [ ] MongoDB service is running
- [ ] Node version compatibility

---

#### Step 2: Seed Test Data ✅
```bash
# In a new terminal while server is running:
cd backend
node scripts/seed-full-day-simulation.js
```

**Check:**
- [ ] Hospital created: "St. Mercy Hospital"
- [ ] 3 Staff members created (Alice, Bob, Dr. Chen)
- [ ] Staff IDs printed in output
- [ ] Shift created for Tuesday
- [ ] No database errors

**Expected Output:**
```
✅ PERFECT TUESDAY SCENARIO CREATED SUCCESSFULLY
📋 TEST DATA SUMMARY:
  Hospital: St. Mercy Hospital (ID: xxxxxx)
  👥 TEAM ASSIGNMENT:
    🔴 Triage Nurse: Alice Chen (ID: xxxxxx)
    🟢 Clinical Nurse: Bob Johnson (ID: xxxxxx)
    🔵 Physician: Dr. Chen (ID: xxxxxx)
```

**Save these IDs** - you'll need them for testing!

---

#### Step 3: Test Authentication Endpoints ✅
```bash
# Get JWT token for Alice
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice.chen@stmercy.com",
    "password": "your_password"  # May fail if no password set
  }'
```

**Check:**
- [ ] Returns JWT token
- [ ] Token can be used in Authorization header
- [ ] Token includes user ID and role

---

### PHASE 2: Endpoint Testing (No Commits)

#### Test 1: Assign Clinical Team Endpoint ✅
```bash
# First, get manager token (may need to create manager user)
MANAGER_TOKEN="your_manager_jwt_token"
SHIFT_ID="shift_id_from_seed"
ALICE_ID="alice_id_from_seed"
BOB_ID="bob_id_from_seed"
DR_CHEN_ID="dr_chen_id_from_seed"

curl -X POST http://localhost:3000/api/shift/assign-clinical-team \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MANAGER_TOKEN" \
  -d "{
    \"shiftId\": \"$SHIFT_ID\",
    \"triageNurseId\": \"$ALICE_ID\",
    \"clinicalNurseId\": \"$BOB_ID\",
    \"clinicalNurseContext\": \"CLINICAL_NURSE_ER\",
    \"physicianId\": \"$DR_CHEN_ID\",
    \"physicianContext\": \"PHYSICIAN_ER\"
  }"
```

**Check:**
- [ ] Endpoint exists (404 error means not implemented)
- [ ] Returns 200 success response
- [ ] Response includes updated shift
- [ ] `teamAssignment` object populated
- [ ] `status` changed to "ACTIVE"

**Expected Response:**
```json
{
  "success": true,
  "message": "Clinical team assigned to shift successfully",
  "shift": {
    "teamAssignment": {
      "triageNurseId": "alice_id",
      "clinicalNurseId": "bob_id",
      "physicianId": "dr_chen_id",
      "teamAssignedAt": "2026-04-21T..."
    },
    "status": "ACTIVE"
  }
}
```

---

#### Test 2: Dashboard Config - Triage Nurse ✅
```bash
ALICE_TOKEN="alice_jwt_token"

curl -X GET http://localhost:3000/api/dashboard/config \
  -H "Authorization: Bearer $ALICE_TOKEN"
```

**Check:**
- [ ] Endpoint exists
- [ ] Returns 200 success
- [ ] `context` is "TRIAGE_NURSE"
- [ ] `components` array includes:
  - [ ] "TriageHeader"
  - [ ] "PatientQueue"
  - [ ] "TriageForm"
  - [ ] "VitalsCapture"
- [ ] `controls.triagePatient` is true
- [ ] `controls.setDisposition` is false (triage nurse shouldn't have this)

---

#### Test 3: Dashboard Config - Clinical Nurse ✅
```bash
BOB_TOKEN="bob_jwt_token"

curl -X GET http://localhost:3000/api/dashboard/config \
  -H "Authorization: Bearer $BOB_TOKEN"
```

**Check:**
- [ ] `context` is "CLINICAL_NURSE_ER"
- [ ] `components` includes "VitalsMonitoring", "MedicationLog"
- [ ] `controls.recordVitals` is true
- [ ] `controls.triagePatient` is false
- [ ] `controls.setDisposition` is false

---

#### Test 4: Dashboard Config - Physician ✅
```bash
DR_CHEN_TOKEN="dr_chen_jwt_token"

curl -X GET http://localhost:3000/api/dashboard/config \
  -H "Authorization: Bearer $DR_CHEN_TOKEN"
```

**Check:**
- [ ] `context` is "PHYSICIAN_ER"
- [ ] `components` includes all 3 lanes:
  - [ ] "TriagePatientLane"
  - [ ] "ActiveCasesLane"
  - [ ] "ObservationPatientLane"
  - [ ] "DispositionPanel"
- [ ] `controls.setDisposition` is true (ONLY physician)
- [ ] `permissions.disposition` is true

---

### PHASE 3: Frontend Testing (No Commits)

#### Step 1: Start Frontend ✅
```bash
cd frontend
npm install  # If needed
npm run dev
```

**Check:**
- [ ] Frontend starts on http://localhost:5173
- [ ] No console errors
- [ ] React dev server running

---

#### Step 2: Login & Check Dashboard Routing ✅

1. **Login as Alice (Triage Nurse)**
   - [ ] Login successful
   - [ ] DashboardRouter mounts
   - [ ] Calls `/api/dashboard/config`
   - [ ] TriageNurseDashboard renders
   - [ ] Shows "Alice Chen"
   - [ ] Shows role badge "TRIAGE_NURSE"

2. **Logout & Login as Bob (Clinical Nurse)**
   - [ ] Login successful
   - [ ] ClinicalNurseDashboard renders
   - [ ] Shows "Bob Johnson"
   - [ ] Shows role badge "CLINICAL_NURSE_ER"

3. **Logout & Login as Dr. Chen (Physician)**
   - [ ] Login successful
   - [ ] PhysicianDashboard renders
   - [ ] Shows "Dr. Chen"
   - [ ] Shows role badge "PHYSICIAN_ER"
   - [ ] 3-lane layout visible
   - [ ] All controls present in disposition panel

---

### PHASE 4: User Journey Testing 🎬

**The "Perfect Tuesday" workflow at 7:00 AM - 8:30 AM:**

#### 7:00 AM - Staff Arrives
- [ ] Alice logs in
- [ ] Dashboard shows waiting/off-duty state (no shift assigned yet)
- [ ] Bob logs in
- [ ] Bob sees same off-duty state

#### 7:06 AM - Manager Assigns Team
- [ ] Manager (or automated system) calls `/api/shift/assign-clinical-team`
- [ ] API returns success with updated teamAssignment
- [ ] Database shows shift status = "ACTIVE"

#### 7:15 AM - Staff Refresh Dashboard
1. **Alice's perspective:**
   - [ ] Refreshes page (or automatic refresh)
   - [ ] Calls `/api/dashboard/config` again
   - [ ] NOW shows `context: "TRIAGE_NURSE"`
   - [ ] NOW shows triage-specific components
   - [ ] Patient queue visible (or placeholder)

2. **Bob's perspective:**
   - [ ] Refreshes page
   - [ ] NOW shows `context: "CLINICAL_NURSE_ER"`
   - [ ] Observation patient list visible (or placeholder)
   - [ ] Vitals monitoring interface visible

3. **Dr. Chen's perspective:**
   - [ ] Refreshes page
   - [ ] NOW shows 3-lane physician dashboard
   - [ ] All disposition controls visible
   - [ ] Admission ward dropdown functional

---

### PHASE 5: Permission Validation ✅

**Check what each role CAN see:**

| Control | Alice (Triage) | Bob (Clinical) | Dr. Chen (Physician) |
|---------|:---------:|:---------:|:---------:|
| Patient Queue | ✅ Visible | ❌ Hidden | ❌ Hidden |
| Triage Form | ✅ Visible | ❌ Hidden | ❌ Hidden |
| Vitals Form | ✅ Visible | ✅ Visible | ❌ Hidden |
| Disposition Panel | ❌ Hidden | ❌ Hidden | ✅ Visible |
| Test Orders | ❌ Hidden | ❌ Hidden | ✅ Visible |

**In browser DevTools:**
- [ ] Open Console → No permission errors
- [ ] Open Network → Check API responses
- [ ] Verify controls match permissions object

---

### PHASE 6: Error Scenarios ✅

#### Test Missing Data
```bash
# Try without required fields
curl -X POST http://localhost:3000/api/shift/assign-clinical-team \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MANAGER_TOKEN" \
  -d "{ }"
```

**Check:**
- [ ] Returns 400 Bad Request
- [ ] Provides clear error message
- [ ] Doesn't crash server

---

#### Test Invalid Staff IDs
```bash
curl -X POST http://localhost:3000/api/shift/assign-clinical-team \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MANAGER_TOKEN" \
  -d "{
    \"shiftId\": \"$SHIFT_ID\",
    \"triageNurseId\": \"invalid_id_12345\"
  }"
```

**Check:**
- [ ] Returns 400 error
- [ ] Message: "One or more staff members not found"
- [ ] Shift NOT modified

---

#### Test Without Authentication
```bash
curl -X GET http://localhost:3000/api/dashboard/config
```

**Check:**
- [ ] Returns 401 Unauthorized
- [ ] No sensitive data exposed

---

## 📋 Testing Checklist - COMPLETE

- [ ] Phase 1: Backend Setup ✅
  - [ ] Server starts
  - [ ] MongoDB connects
  - [ ] Seed data loads
  
- [ ] Phase 2: Endpoint Testing ✅
  - [ ] assignClinicalTeam works
  - [ ] dashboard/config works for each role
  - [ ] Correct permissions returned
  
- [ ] Phase 3: Frontend Testing ✅
  - [ ] Frontend starts
  - [ ] Components render
  - [ ] Role-based routing works
  
- [ ] Phase 4: User Journey ✅
  - [ ] Perfect Tuesday workflow tested
  - [ ] Dashboard updates after team assignment
  - [ ] All roles see correct interface
  
- [ ] Phase 5: Permission Validation ✅
  - [ ] Each role sees only their controls
  - [ ] No permission errors in console
  
- [ ] Phase 6: Error Scenarios ✅
  - [ ] Invalid input handled
  - [ ] Missing data rejected
  - [ ] Unauthorized requests blocked

---

## ✅ READY TO COMMIT ONLY WHEN:

- [x] All 6 phases complete
- [x] No errors in console
- [x] No network errors in DevTools
- [x] User journey flows naturally
- [x] All roles see correct dashboards
- [x] Permissions enforced correctly
- [x] Error handling works
- [x] Database records match expected state

---

## 🚀 Commit Command (AFTER all tests pass)

```bash
git add -A
git commit -m "feat: implement full day simulation role-based dashboard system

Test Status: ✅ ALL PHASES PASSED
- Backend endpoints: assign-clinical-team, dashboard/config working
- Frontend components: DashboardRouter routing correctly
- User journey: Perfect Tuesday workflow validated
- Permissions: Role-based access control enforced
- Error handling: All error scenarios handled

Implements:
- POST /api/shift/assign-clinical-team
- GET /api/dashboard/config
- 5 frontend dashboard components (Router, Triage, Clinical, Physician, OffDuty)
- Role-based permission system"
```

---

## 📝 Test Report Template

When all tests pass, fill this out:

```
TEST REPORT - Full Day Simulation Implementation
================================================

Date: 2026-04-21
Tester: [Your Name]

PHASE 1 - Backend Setup: ✅ PASS / ⚠️ ISSUES
PHASE 2 - Endpoint Testing: ✅ PASS / ⚠️ ISSUES
PHASE 3 - Frontend Testing: ✅ PASS / ⚠️ ISSUES
PHASE 4 - User Journey: ✅ PASS / ⚠️ ISSUES
PHASE 5 - Permission Validation: ✅ PASS / ⚠️ ISSUES
PHASE 6 - Error Scenarios: ✅ PASS / ⚠️ ISSUES

Overall: ✅ READY FOR COMMIT

Issues Found: [List any]
Recommendations: [List any improvements]
```
