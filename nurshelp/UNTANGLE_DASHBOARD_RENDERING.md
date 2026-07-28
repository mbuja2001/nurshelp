# ARCHITECTURE: UNTANGLING ROLE-BASED DASHBOARD RENDERING
## Complete Diagnostic of Current Wiring & Refactoring Plan

**Status**: 🔍 DIAGNOSTIC - Current state analysis + untangling blueprint  
**Date**: 20 April 2026  
**Goal**: Make role-based rendering work WITHOUT breaking existing clock-in/scheduling

---

## PART 1: CURRENT STATE - THE TANGLED WIRES

### 1.1 BACKEND CLOCK-IN FLOW (Currently Working ✅)

```
ENDPOINT: POST /api/nurses/clock-in

Flow:
├─ 1. receive { shiftId, location }
├─ 2. Auth check: req.user._id (from JWT middleware)
├─ 3. Lookup Shift by ID
├─ 4. Validate shift timing:
│  ├─ Can't clock in >5 mins before start
│  ├─ Can't clock in after shift end
│  └─ ⚠️ NO CHECK: Does staff member belong to this shift?
├─ 5. Check for existing active clock:
│  ├─ Find record with status: CLOCKED_IN or ON_BREAK today
│  ├─ If shift has expired, AUTO_CLOCKOUT
│  └─ If already clocked in, error: "Already clocked in today"
├─ 6. Check if shift already covered:
│  ├─ Find record with status: CLOCKED_OUT for this shift today
│  └─ If found, error: "Already covered this shift"
├─ 7. Create NurseClockRecord:
│  └─ nurseId, shiftId, date, clockInTime, location, status: CLOCKED_IN, flags: []
├─ 8. Add flags if timing weird:
│  ├─ EARLY_CLOCKIN (if >5 mins before start)
│  └─ LATE_CLOCKIN (if >10 mins after start)
└─ 9. Return success + record

❌ PROBLEM 1: No verification that nurse is assigned to this shift
❌ PROBLEM 2: No operationalContext captured from Shift
```

### 1.2 BREAK MANAGEMENT (Working ✅)

```
ENDPOINTS: 
├─ POST /api/nurses/break-start
└─ POST /api/nurses/break-end

Flow:
├─ Find active clock record (status: CLOCKED_IN)
├─ Update status: CLOCKED_IN → ON_BREAK
├─ Set breakStartTime
├─ (same for break-end, set breakEndTime)
└─ Calculate totalBreakTime

❌ PROBLEM: No role checks, everyone can break
❌ PROBLEM: No operationalContext tracking during break
```

### 1.3 FRONTEND CLOCK-IN FLOW (Currently Working ✅)

```
COMPONENT: frontend/src/Dashboard/Pages/ClockInOutBoard.jsx

Flow:
├─ useEffect on mount:
│  ├─ fetchTodayRecord() → GET /api/nurses/today-record
│  ├─ fetchTodayShift() → GET /api/shifts/assigned/today
│  ├─ fetchAssignedShifts() → GET /api/shifts/assigned
│  └─ (Sets up live clock every 500ms)
│
├─ Check shift eligibility:
│  ├─ Is it already covered? (look at attendance history)
│  ├─ Can I clock in? (within 5 mins before to end)
│  └─ Color code: 🟢 Ready, 🟡 Late, 🔴 Expired
│
├─ User selects shift from dropdown
├─ Clicks "Clock In" button
└─ POST /api/nurses/clock-in { shiftId, location }

❌ PROBLEM 1: No role-based UI hiding
   - Everyone (NURSE, PHYSICIAN, SPECIALIST) sees same interface
   - Physicians should NOT see triage queue or bed management
❌ PROBLEM 2: No operationalContext selection
   - Frontend doesn't ask: "Which context are you working in?"
   - Backend clock record doesn't capture context
```

### 1.4 DASHBOARD ROUTING (BROKEN ❌)

```
CURRENT ARCHITECTURE: frontend/src/App.jsx

Routes defined:
├─ "/" → LandingPage (public)
├─ "/dashboard" → Dashboard component
├─ "/clock-in-out" → ClockInOutBoard component
├─ "/profile", "/ehr", "/analytics", etc. → Specific pages
└─ (All protected by checking localStorage.token)

❌ PROBLEM 1: Dashboard.jsx doesn't distinguish roles
   - Lines 113-116: Checks for "ADMIN" or "MANAGER" role
   - Everything else treats as "nurse"
   - NO routing for PHYSICIAN, NO routing for SPECIALIST

❌ PROBLEM 2: No operationalContext routing
   - System doesn't know if user is:
     ├─ TRIAGE_NURSE (triage queue)
     ├─ CLINICAL_NURSE_ER (observation patients)
     ├─ CLINICAL_NURSE_WARD (bed map)
     ├─ PHYSICIAN_ER (3-lane review)
     └─ ON_CALL_PHYSICIAN (hybrid view)

❌ PROBLEM 3: NurseDashboard.jsx is the fallback for everyone
   - ADMIN sees NurseDashboard (confusing!)
   - PHYSICIAN sees NurseDashboard (confusing!)
   - No role-specific features
```

---

## PART 2: WHAT'S TANGLED

### 2.1 THE WIRING MESS

```
Current Data Flow (Tangled):

┌─ Browser
│  ├─ User logs in → token stored
│  ├─ Token has role: NURSE | PHYSICIAN | MANAGER | ADMIN
│  └─ ❌ NO operationalContext in token!
│
├─ Dashboard.jsx
│  ├─ Reads role from token
│  ├─ Checks: isManagerRole? (lines 113-116)
│  │  ├─ YES → Default to manager view
│  │  └─ NO → Falls through to EVERYTHING else
│  └─ ❌ NO ELSE-IF for PHYSICIAN, SPECIALIST
│
├─ NurseDashboard.jsx (Fallback)
│  ├─ Called when role = NURSE (correct!)
│  ├─ Called when role = PHYSICIAN (wrong!)
│  ├─ Called when role = SPECIALIST (wrong!)
│  └─ Shows generic nurse interface to everyone
│
├─ Clock-in continues to work
│  ├─ No role checks (authorize('NURSE', 'PHYSICIAN'))
│  ├─ No context capture
│  └─ ✅ Functional but incomplete
│
└─ Backend clock record
   └─ NurseClockRecord.model.js:
      ├─ nurseId ✅
      ├─ shiftId ✅
      ├─ location ✅ (ED, ICU, etc.)
      ├─ status ✅ (CLOCKED_IN, ON_BREAK, CLOCKED_OUT)
      ├─ ❌ NO operationalContext field!
      ├─ ❌ NO staffType field (NURSE vs PHYSICIAN)
      └─ ❌ NO dashboard preference/context
```

### 2.2 WHAT'S WORKING (DO NOT BREAK!)

```
✅ Clock-in endpoint logic (POST /api/nurses/clock-in)
   └─ Manages timing, duplicate checks, auto-clockout
   
✅ Break management (breakStart, breakEnd)
   └─ Updates status correctly
   
✅ Clock history fetching
   └─ Attendance history loads
   
✅ Shift assignment queries
   └─ Staff can find their shifts
   
✅ JWT authentication
   └─ Token stored, read, validated
   
✅ Frontend clock selection UI
   └─ Shifts populate correctly
   
✅ Real-time clock (updates every 500ms)
   └─ Eligibility checks work
```

### 2.3 WHAT'S BROKEN (NEEDS FIXING)

```
❌ Role-based dashboard routing
   └─ Everyone sees same dashboard
   
❌ No physician-specific interface
   └─ Dr. Chen sees triage queue (wrong!)
   
❌ No specialist-specific interface
   └─ Nobody can see specialist dashboard
   
❌ No operationalContext tracking
   └─ Can't distinguish TRIAGE_NURSE vs CLINICAL_NURSE_ER
   
❌ No role-based view hiding
   └─ Triage form shows to everyone
   └─ Bed allocation shows to everyone
   └─ Analytics shows to everyone
   
❌ Frontend doesn't ask for operational context
   └─ No UI to select: "I'm triaging" vs "I'm monitoring patients"
   
❌ Backend doesn't capture operational context
   └─ Clock record doesn't store: which context was selected?
```

---

## PART 3: DIAGNOSIS - ENDPOINTS & FLOWS

### 3.1 BACKEND CLOCK-IN ENDPOINTS (Current)

```javascript
// ROUTE: backend/src/routes/nurse.route.js

POST /api/nurses/clock-in (Protected)
├─ Middleware: protect (auth check)
├─ Middleware: authorize('NURSE', 'PHYSICIAN')
├─ Middleware: auditLog('CLOCK_IN')
└─ Handler: clockIn (nurse.controller.js)

POST /api/nurses/clock-out (Protected)
├─ Middleware: protect
├─ Middleware: authorize('NURSE', 'PHYSICIAN')
├─ Middleware: auditLog('CLOCK_OUT')
└─ Handler: clockOut

POST /api/nurses/break-start (Protected)
├─ Middleware: protect
├─ Middleware: authorize('NURSE', 'PHYSICIAN')
└─ Handler: breakStart

POST /api/nurses/break-end (Protected)
├─ Middleware: protect
├─ Middleware: authorize('NURSE', 'PHYSICIAN')
└─ Handler: breakEnd

GET /api/nurses/today-record (Protected)
├─ Middleware: protect
└─ Handler: getTodayRecord

GET /api/shifts/assigned (Protected)
├─ Middleware: protect
└─ Handler: Looks up shifts for current user

GET /api/shifts/assigned/today (Protected)
├─ Middleware: protect
└─ Handler: Returns today's assigned shift

GET /api/nurses/attendance-history (Protected)
├─ Middleware: protect
└─ Handler: Returns today's attendance records
```

### 3.2 FRONTEND CLOCK-IN FLOW (Current)

```javascript
// COMPONENT: frontend/src/Dashboard/Pages/ClockInOutBoard.jsx

useEffect (on mount/tab change):
├─ fetchTodayRecord()
│  └─ GET /api/nurses/today-record
│  └─ Returns: { success, record: { clockInTime, status: CLOCKED_IN|ON_BREAK|CLOCKED_OUT, ... } }
│
├─ fetchTodayShift()
│  └─ GET /api/shifts/assigned/today
│  └─ Returns: { success, shift: { _id, shiftName, startTime, endTime, ... } }
│
├─ fetchAssignedShifts()
│  └─ GET /api/shifts/assigned
│  └─ Returns: { success, shifts: [...] }
│
├─ fetchAttendanceHistory()
│  └─ GET /api/nurses/attendance-history
│  └─ Returns: { success, records: [...] }
│
└─ setInterval for live time (500ms)

User Action: Select shift + click "Clock In"
├─ POST /api/nurses/clock-in
│  ├─ Body: { shiftId, location }
│  └─ Returns: { success, record: { _id, clockInTime, status, flags } }
└─ Update UI: Show "Clocked In" state

User Action: Click "Break Start"
├─ POST /api/nurses/break-start
│  └─ Body: {} or { reason }
│  └─ Returns: { success, record: { status: ON_BREAK, breakStartTime } }
└─ Update UI: Show "On Break" state
```

### 3.3 DASHBOARD ROUTING (Current)

```javascript
// COMPONENT: frontend/src/Dashboard/Dashboard.jsx

getInitial state:
├─ view = "home"
├─ nurse = null
└─ (other state)

useEffect (session restore):
├─ const token = localStorage.getItem("token")
├─ if token:
│  ├─ Decode JWT
│  ├─ setNurse({ id, email, name })
│  └─ setView("dashboard")
└─ else:
   └─ Show login form

getUserRole():
├─ Read token from localStorage
├─ Decode JWT
└─ Return decoded.role (NURSE | PHYSICIAN | MANAGER | ADMIN)

isManagerRole():
├─ const role = getUserRole()
├─ return role === "ADMIN" || role === "MANAGER"

useEffect (polling):
├─ if (!nurse) return
├─ const isAdmin = nurse.role === "ADMIN" || nurse.role === "MANAGER"
├─ if (!isAdmin):
│  ├─ fetchWaiting() → GET /api/encounters/waiting
│  ├─ fetchMyEncounters() → GET /api/encounters
│  └─ setInterval to re-fetch waiting every 5 sec
└─ else:
   └─ Don't fetch patient data (admin view)

Render Logic (❌ BROKEN):
├─ if (!nurse) → <LoginForm />
├─ else if view === "dashboard":
│  └─ <NurseDashboard /> (for EVERYONE except manager)
│      ├─ Shows patient queue ✅ for nurses
│      ├─ Shows patient queue ❌ for physicians (wrong!)
│      ├─ Shows triage form ✅ for nurses
│      ├─ Shows triage form ❌ for physicians (wrong!)
│      └─ No special physician interface
├─ else if view === "schedule":
│  └─ <StaffScheduleManager /> (admin only, but not role-gated)
├─ else if view === "clock":
│  └─ <ClockInOutBoard /> (admin only, but not role-gated)
└─ else if ...other views
```

---

## PART 4: REFACTORING PLAN (Without Breaking)

### 4.1 PHASE 1: Capture operationalContext (With backward compatibility)

**Goal**: Make clock-in capture operational context WITHOUT breaking existing flow

#### Step 1A: Update NurseClockRecord model

```javascript
// Backend: backend/src/models/nurseClockRecord.model.js

// ADD these fields (don't remove existing):
operationalContext: {
  type: String,
  enum: [
    'TRIAGE_NURSE',
    'CLINICAL_NURSE_ER',
    'CLINICAL_NURSE_WARD',
    'PHYSICIAN_ER',
    'ON_CALL_PHYSICIAN',
    'SPECIALIST'
  ],
  default: null  // ← Nullable for backward compat
},
staffRole: {
  type: String,
  enum: ['NURSE', 'PHYSICIAN', 'SPECIALIST'],
  default: 'NURSE'
},
dashboardType: {
  type: String,
  default: null  // Will be derived from operationalContext
}
```

#### Step 1B: Get operationalContext from Shift (don't change anything else)

```javascript
// Backend: backend/src/controllers/nurse.controller.js - clockIn function

// AFTER shift lookup (around line 145):
const shift = await Shift.findById(shiftId);

// ADD: Extract context from shift if exists
const operationalContext = shift?.assignedStaff?.find(staff => 
  staff.staffID.toString() === nurseId.toString()
)?.context || null;

// Later, when creating clockRecord (around line 265):
const clockRecord = new NurseClockRecord({
  nurseId,
  shiftId,
  date: new Date(),
  clockInTime: new Date(),
  location: location || "ED",
  status: "CLOCKED_IN",
  flags: [],
  operationalContext: operationalContext,  // ← NEW (may be null)
  staffRole: staffRole || "NURSE"  // ← NEW (from where?)
});
```

**Problem**: We don't have staffRole at clock-in time. Need to fix...

#### Step 1C: Get staffRole from Nurse record

```javascript
// Backend: Same clockIn function

// BEFORE creating clockRecord:
const nurseRecord = await Nurse.findById(nurseId);  // New lookup
const staffRole = nurseRecord?.role || "NURSE";  // NURSE | PHYSICIAN | SPECIALIST

// Then when creating clockRecord:
const clockRecord = new NurseClockRecord({
  nurseId,
  shiftId,
  date: new Date(),
  clockInTime: new Date(),
  location: location || "ED",
  status: "CLOCKED_IN",
  flags: [],
  operationalContext: operationalContext,
  staffRole: staffRole  // ✅ From Nurse.role
});
```

**Status**: ✅ Gets operationalContext from Shift (if it exists)
**Backward Compat**: ✅ If no context in shift, defaults to null

---

### 4.2 PHASE 2: Frontend ask for context (Optional selection)

**Goal**: If backend clock has null operationalContext, ask user to choose

#### Step 2A: Update ClockInOutBoard to ask for context

```javascript
// Frontend: frontend/src/Dashboard/Pages/ClockInOutBoard.jsx

// ADD state:
const [selectedContext, setSelectedContext] = useState(null);

// ADD context selector UI (before Clock In button):
{selectedShift && !selectedShift.hasOperationalContext && (
  <div>
    <label>What are you doing today?</label>
    <select value={selectedContext} onChange={(e) => setSelectedContext(e.target.value)}>
      <option value="">-- Select your role --</option>
      {user?.role === 'NURSE' && (
        <>
          <option value="TRIAGE_NURSE">Triaging patients</option>
          <option value="CLINICAL_NURSE_ER">Monitoring ER patients</option>
          <option value="CLINICAL_NURSE_WARD">Managing ward beds</option>
        </>
      )}
      {user?.role === 'PHYSICIAN' && (
        <>
          <option value="PHYSICIAN_ER">Reviewing ER cases</option>
          <option value="ON_CALL_PHYSICIAN">On-call (hybrid view)</option>
        </>
      )}
    </select>
  </div>
)}

// UPDATE handleClockIn:
const handleClockIn = async () => {
  const clockInData = {
    shiftId: selectedShift._id,
    location: user?.location || 'ED',
    operationalContext: selectedContext || null  // ← NEW
  };
  
  const res = await fetch(`${BACKEND_URL}/api/nurses/clock-in`, {
    method: 'POST',
    headers: { ... },
    body: JSON.stringify(clockInData)
  });
};
```

**Status**: ✅ Optional (context selector only if needed)
**Backward Compat**: ✅ POST still works without operationalContext

---

### 4.3 PHASE 3: Update dashboard routing (The BIG refactoring)

**Goal**: Route based on operationalContext INSTEAD OF just role

#### Step 3A: Create new DashboardRouter component

```javascript
// Frontend: frontend/src/Dashboard/DashboardRouter.jsx (NEW)

import React, { useEffect, useState } from 'react';
import { decodeJwt } from '../utils/jwt';

import TriageNurseDashboard from './Dashboards/TriageNurseDashboard';
import ClinicalNurseDashboard_ER from './Dashboards/ClinicalNurseDashboard_ER';
import WardNurseDashboard from './Dashboards/WardNurseDashboard';
import PhysicianDashboard_ER from './Dashboards/PhysicianDashboard_ER';
import OnCallPhysicianDashboard from './Dashboards/OnCallPhysicianDashboard';
import AdminDashboard from './Dashboards/AdminDashboard';

export default function DashboardRouter() {
  const [operationalContext, setOperationalContext] = useState(null);
  const [staffRole, setStaffRole] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) return;

    const decoded = decodeJwt(token);
    setStaffRole(decoded?.role);

    // Fetch current clock record to get operationalContext
    fetchCurrentContext();
  }, []);

  const fetchCurrentContext = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('${BACKEND_URL}/api/nurses/today-record', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();

      if (data.record?.operationalContext) {
        setOperationalContext(data.record.operationalContext);
      }
    } catch (err) {
      console.error('Failed to fetch context:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>Loading dashboard...</div>;

  // Route based on role + operationalContext
  if (staffRole === 'ADMIN' || staffRole === 'MANAGER') {
    return <AdminDashboard />;
  }

  if (staffRole === 'PHYSICIAN') {
    if (operationalContext === 'ON_CALL_PHYSICIAN') {
      return <OnCallPhysicianDashboard />;
    }
    return <PhysicianDashboard_ER />;
  }

  if (staffRole === 'NURSE') {
    if (operationalContext === 'TRIAGE_NURSE') {
      return <TriageNurseDashboard />;
    }
    if (operationalContext === 'CLINICAL_NURSE_ER') {
      return <ClinicalNurseDashboard_ER />;
    }
    if (operationalContext === 'CLINICAL_NURSE_WARD') {
      return <WardNurseDashboard />;
    }
  }

  // Fallback (for backward compat with no context)
  return <TriageNurseDashboard />;
}
```

#### Step 3B: Update App.jsx to use DashboardRouter

```javascript
// Frontend: frontend/src/App.jsx

import DashboardRouter from "./Dashboard/DashboardRouter.jsx";  // NEW

// Replace old dashboard route:
// OLD:
// <Route path="/dashboard" element={<Dashboard />} />

// NEW:
<Route path="/dashboard" element={<DashboardRouter />} />
```

**Status**: ✅ Clean separation by operationalContext
**Backward Compat**: ✅ Defaults to TriageNurseDashboard if no context

---

### 4.4 PHASE 4: Create dashboard stubs (Don't break existing)

**Goal**: Create placeholder dashboards for each role/context

#### Step 4A: Create stub components

```javascript
// Frontend: frontend/src/Dashboard/Dashboards/TriageNurseDashboard.jsx (NEW)

export default function TriageNurseDashboard() {
  return (
    <div>
      <h1>🏥 TRIAGE QUEUE</h1>
      <p>Rapid assessment and team assignment</p>
      {/* Will migrate existing NurseDashboard queue logic here */}
    </div>
  );
}

// Frontend: frontend/src/Dashboard/Dashboards/PhysicianDashboard_ER.jsx (NEW)

export default function PhysicianDashboard_ER() {
  return (
    <div>
      <h1>🩺 PHYSICIAN COMMAND CENTRE</h1>
      <p>3-lane view: Active ER | Observation | Escalations</p>
      {/* NEW: Show physician-specific view */}
    </div>
  );
}

// Similar for: ClinicalNurseDashboard_ER, WardNurseDashboard, etc.
```

**Status**: ✅ Stubs created
**Backward Compat**: ✅ Existing Dashboard.jsx still exists (unused)

---

## PART 5: STEP-BY-STEP IMPLEMENTATION (Without breaking)

### Step 1: Backup current system (git commit)

```bash
git add -A
git commit -m "Checkpoint: Before operationalContext refactoring"
```

### Step 2: Update backend models (backward compat)

```javascript
// backend/src/models/nurseClockRecord.model.js

// Just ADD new fields, don't remove anything
operationalContext: { type: String, enum: [...], default: null }
staffRole: { type: String, default: 'NURSE' }
```

**Why safe**: 
- Existing clock records have `undefined` for new fields (fine)
- New clock records will have values for new fields
- Backend handles both

### Step 3: Update clock-in endpoint (backward compat)

```javascript
// backend/src/controllers/nurse.controller.js - clockIn function

// Extract operationalContext from shift.assignedStaff if available
const operationalContext = /* extract from shift */ || null;

// Set staffRole from Nurse record
const staffRole = nurseRecord?.role || "NURSE";

// Pass to NurseClockRecord (new fields are optional)
```

**Why safe**:
- If no operationalContext found, defaults to null (backward compat)
- If no staffRole found, defaults to "NURSE"
- Existing code doesn't know about these fields

### Step 4: Update frontend ClockInOutBoard (optional context selector)

```javascript
// Add optional context selector (only if needed)
// If shift already has context, don't ask user
// If no context, optionally ask user to select
```

**Why safe**:
- Only affects frontend UI
- Doesn't change POST to /api/nurses/clock-in
- Existing flow still works

### Step 5: Create DashboardRouter (new file, no changes to existing)

```javascript
// frontend/src/Dashboard/DashboardRouter.jsx (NEW FILE)

// Just routes based on role + operationalContext
// Doesn't modify existing Dashboard.jsx
```

**Why safe**:
- New file doesn't affect existing code
- Old Dashboard.jsx still exists
- Can be tested in isolation

### Step 6: Gradually migrate dashboard logic

```javascript
// For each role/context:
// 1. Create new sub-component (e.g., TriageNurseDashboard.jsx)
// 2. Copy logic from NurseDashboard.jsx into it
// 3. Test thoroughly
// 4. Update DashboardRouter to use new component
// 5. Keep old NurseDashboard.jsx as fallback
```

**Why safe**:
- Each component created independently
- DashboardRouter can fallback to old NurseDashboard if something breaks
- Incremental migration

---

## PART 6: TESTING STRATEGY (No broken flow)

### Test 1: Clock-in still works (backward compat)

```bash
# Test existing clock-in WITHOUT context selector
POST /api/nurses/clock-in
├─ Body: { shiftId, location }
├─ Expected: 201, record created
└─ operationalContext field should be null (fine)
```

### Test 2: Clock-in WITH context (new feature)

```bash
# Test new clock-in WITH context
POST /api/nurses/clock-in
├─ Body: { shiftId, location, operationalContext: "TRIAGE_NURSE" }
├─ Expected: 201, record created
└─ operationalContext field should be "TRIAGE_NURSE"
```

### Test 3: Dashboard routing (role-based)

```
# Sarah (TRIAGE_NURSE context) logs in
├─ /dashboard → DashboardRouter
├─ Fetches /api/nurses/today-record
├─ operationalContext = "TRIAGE_NURSE"
├─ Routes to: <TriageNurseDashboard />
└─ ✅ Shows triage queue

# Dr. Chen (PHYSICIAN_ER context) logs in
├─ /dashboard → DashboardRouter
├─ Fetches /api/nurses/today-record
├─ operationalContext = "PHYSICIAN_ER"
├─ Routes to: <PhysicianDashboard_ER />
└─ ✅ Shows 3-lane physician view
```

### Test 4: Fallback (backward compat)

```
# Old nurse (no operationalContext in clock record)
├─ /dashboard → DashboardRouter
├─ Fetches /api/nurses/today-record
├─ operationalContext = null
├─ Routes to: <TriageNurseDashboard /> (fallback)
└─ ✅ Still works (backward compat)
```

---

## PART 7: TIMELINE & EFFORT

### Phase 1 (1 hour): Backend model + endpoint update
- [ ] Add operationalContext, staffRole fields to NurseClockRecord
- [ ] Update clockIn to extract context from shift
- [ ] Test backward compatibility

### Phase 2 (1 hour): Frontend context selector (optional)
- [ ] Add optional context selector to ClockInOutBoard
- [ ] Test that POST still works

### Phase 3 (3 hours): DashboardRouter + stubs
- [ ] Create DashboardRouter.jsx
- [ ] Create 5 dashboard stub components
- [ ] Update App.jsx routing
- [ ] Test role-based routing

### Phase 4 (4+ hours): Migrate dashboard logic (incremental)
- [ ] Migrate TriageNurseDashboard logic ✅ FIRST (most critical)
- [ ] Migrate ClinicalNurseDashboard_ER logic
- [ ] Migrate PhysicianDashboard_ER logic (NEW)
- [ ] Migrate WardNurseDashboard logic

**Total**: ~9 hours (can be done incrementally without breaking)

---

## PART 8: CHECKLIST (DO NOT BREAK!)

### Before You Start
- [ ] `git commit` current working state
- [ ] All clock-in tests pass
- [ ] All shift scheduling tests pass
- [ ] All break management tests pass

### During Development
- [ ] Clock-in endpoint still works (test every commit)
- [ ] Break start/end still work
- [ ] Attendance history still loads
- [ ] Old NurseDashboard still rendered (fallback)

### After Each Phase
- [ ] Test with Postman: POST /api/nurses/clock-in (old style)
- [ ] Test with Postman: POST /api/nurses/clock-in (new style with context)
- [ ] Test frontend: Clock-in button works
- [ ] Test frontend: Dashboard routes correctly

### Final Validation
- [ ] Sarah can log in → Sees TriageNurseDashboard
- [ ] Bob can log in → Sees ClinicalNurseDashboard_ER
- [ ] Dr. Chen can log in → Sees PhysicianDashboard_ER
- [ ] Ward Nurse can log in → Sees WardNurseDashboard
- [ ] Admin can log in → Sees AdminDashboard
- [ ] All can clock in/out
- [ ] All can take breaks
- [ ] All can view history

---

## SUMMARY

**What's tangled**: Dashboard renders same interface for all roles + no operationalContext tracking

**What's working**: Clock-in, break management, shift queries, authentication

**Untangling strategy**:
1. Add operationalContext fields to clock record (backward compat)
2. Extract context from shift when clocking in (optional)
3. Create DashboardRouter that routes by role + context
4. Create role-specific dashboard components
5. Migrate logic incrementally (don't break existing)

**Key principle**: Add new capability WITHOUT removing old code until tested

**Effort**: ~9 hours of development, can be done incrementally

Ready to start untangling? 🚀
