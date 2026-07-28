# REFACTORING ROADMAP: Phase-by-Phase Implementation
## Complete Code Snippets & Git Workflow

**Status**: 🛠️ IMPLEMENTATION READY  
**Prerequisite**: Read UNTANGLE_DASHBOARD_RENDERING.md first  
**Goal**: Execute refactoring WITHOUT breaking existing functionality

---

## WORKFLOW: Git-Safe Implementation

### Step 0: Create feature branch (CRITICAL!)

```bash
cd /home/wtc/Videos/nursehelpdesk

# Create isolated branch for this refactoring
git checkout -b feature/role-based-dashboard-routing

# Verify you're on new branch
git branch -v  # Should show * feature/role-based-dashboard-routing

# Commit current state as checkpoint
git add -A
git commit -m "CHECKPOINT: Before operationalContext refactoring

- Clock-in, break management, scheduling all working
- About to add operationalContext tracking
- This commit is fallback if anything breaks"
```

---

## PHASE 1: Backend Models & Clock-In Endpoint (1-1.5 hours)

### Task 1.1: Update NurseClockRecord schema

**File**: `backend/src/models/nurseClockRecord.model.js`

**Current state**: 
```javascript
const nurseClockRecordSchema = new Schema({
  nurseId: { type: Schema.Types.ObjectId, ref: 'Nurse', required: true },
  shiftId: { type: Schema.Types.ObjectId, ref: 'Shift', required: true },
  date: { type: Date, default: Date.now },
  clockInTime: Date,
  clockOutTime: Date,
  totalShiftDuration: Number,
  status: { type: String, enum: ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT'], default: 'CLOCKED_IN' },
  location: String,
  breakStartTime: Date,
  breakEndTime: Date,
  totalBreakTime: Number,
  flags: [{ type: String, enum: ['EARLY_CLOCKIN', 'LATE_CLOCKIN', 'EARLY_CLOCKOUT', 'BREAK_MISSED'] }]
});
```

**What to add** (just INSERT, don't remove anything):

```javascript
const nurseClockRecordSchema = new Schema({
  nurseId: { type: Schema.Types.ObjectId, ref: 'Nurse', required: true },
  shiftId: { type: Schema.Types.ObjectId, ref: 'Shift', required: true },
  date: { type: Date, default: Date.now },
  clockInTime: Date,
  clockOutTime: Date,
  totalShiftDuration: Number,
  status: { type: String, enum: ['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT'], default: 'CLOCKED_IN' },
  location: String,
  breakStartTime: Date,
  breakEndTime: Date,
  totalBreakTime: Number,
  flags: [{ type: String, enum: ['EARLY_CLOCKIN', 'LATE_CLOCKIN', 'EARLY_CLOCKOUT', 'BREAK_MISSED'] }],
  
  // ← ADD THESE THREE FIELDS:
  operationalContext: {
    type: String,
    enum: [
      'TRIAGE_NURSE',
      'CLINICAL_NURSE_ER',
      'CLINICAL_NURSE_WARD',
      'PHYSICIAN_ER',
      'ON_CALL_PHYSICIAN',
      'SPECIALIST',
      'SUPPORT_STAFF'
    ],
    default: null  // Allow null for backward compatibility
  },
  staffRole: {
    type: String,
    enum: ['NURSE', 'PHYSICIAN', 'SPECIALIST', 'SUPPORT'],
    default: 'NURSE'
  },
  dashboardType: {
    type: String,
    default: null  // Will be set based on operationalContext
  }
});
```

**Commit after this step**:
```bash
git add backend/src/models/nurseClockRecord.model.js
git commit -m "PHASE 1.1: Add operationalContext fields to clock record

- Added operationalContext: enum of role-specific contexts
- Added staffRole: NURSE|PHYSICIAN|SPECIALIST|SUPPORT
- Added dashboardType: derived field for routing
- Fields are nullable for backward compatibility
- Existing clock records unaffected"
```

---

### Task 1.2: Update clock-in endpoint to extract context

**File**: `backend/src/controllers/nurse.controller.js`

**Find this function** (around line 118-200):
```javascript
export const clockIn = async (req, res) => {
  try {
    const { shiftId, location } = req.body;
    const nurseId = req.user?._id;
    
    // ... existing code for validation ...
    
    const clockRecord = new NurseClockRecord({
      nurseId,
      shiftId,
      date: new Date(),
      clockInTime: new Date(),
      location: location || "ED",
      status: "CLOCKED_IN",
      flags: [],
      // ← In the old code, ends here
    });
    
    // ... rest of function
  }
}
```

**UPDATE to this** (just add the NEW lines marked with ↓ AFTER this):

```javascript
export const clockIn = async (req, res) => {
  try {
    const { shiftId, location, operationalContext: userProvidedContext } = req.body;
    const nurseId = req.user?._id;

    // Validate shift exists and user can clock in
    const shift = await Shift.findById(shiftId);
    if (!shift) {
      return res.status(404).json({ success: false, message: "Shift not found" });
    }

    // Prevent clocking in before 5 minutes before shift start
    const now = new Date();
    const shiftStart = new Date(shift.startTime);
    const fiveMinsBefore = new Date(shiftStart.getTime() - 5 * 60000);

    if (now < fiveMinsBefore) {
      return res.status(400).json({
        success: false,
        message: `Cannot clock in before ${fiveMinsBefore.toLocaleTimeString()}`,
        earliestClockIn: fiveMinsBefore
      });
    }

    // Prevent clocking in well after shift end
    const shiftEnd = new Date(shift.endTime);
    if (now > shiftEnd) {
      return res.status(400).json({
        success: false,
        message: "Cannot clock in after shift end time"
      });
    }

    // Check for existing active clock record
    let existingRecord = await NurseClockRecord.findOne({
      nurseId,
      date: { $gte: new Date().setHours(0, 0, 0, 0), $lt: new Date().setHours(24, 0, 0, 0) },
      status: { $in: ["CLOCKED_IN", "ON_BREAK"] }
    });

    if (existingRecord) {
      // Auto-clock out if shift expired
      if (new Date() > shiftEnd) {
        existingRecord.clockOutTime = new Date();
        existingRecord.status = "CLOCKED_OUT";
        existingRecord.totalShiftDuration = existingRecord.clockOutTime - existingRecord.clockInTime;
        await existingRecord.save();
      } else {
        return res.status(400).json({
          success: false,
          message: "Already clocked in today"
        });
      }
    }

    // Check if shift already covered
    const coveredRecord = await NurseClockRecord.findOne({
      shiftId,
      date: { $gte: new Date().setHours(0, 0, 0, 0), $lt: new Date().setHours(24, 0, 0, 0) },
      status: "CLOCKED_OUT"
    });

    if (coveredRecord) {
      return res.status(400).json({
        success: false,
        message: "This shift is already covered"
      });
    }

    // ↓ ADD THIS BLOCK: Extract operationalContext from shift
    let operationalContext = userProvidedContext || null;  // Use provided context if given
    
    if (!operationalContext && shift.assignedStaff) {
      // Try to find context from shift assignment
      const assignment = shift.assignedStaff.find(staff =>
        staff.staffID?.toString() === nurseId.toString()
      );
      operationalContext = assignment?.context || null;  // Extract context if it exists
    }

    // ↓ ADD THIS BLOCK: Get staff role from Nurse record
    const nurseRecord = await Nurse.findById(nurseId);
    const staffRole = nurseRecord?.role || "NURSE";  // Default to NURSE if not found

    // ↓ MODIFY: Add new fields to NurseClockRecord
    const clockRecord = new NurseClockRecord({
      nurseId,
      shiftId,
      date: new Date(),
      clockInTime: new Date(),
      location: location || "ED",
      status: "CLOCKED_IN",
      flags: [],
      operationalContext: operationalContext,  // ← NEW
      staffRole: staffRole,  // ← NEW
      dashboardType: operationalContext  // ← NEW (derived from context)
    });

    // Add timing flags
    if (new Date(now) < new Date(shiftStart.getTime() - 5 * 60000)) {
      clockRecord.flags.push("EARLY_CLOCKIN");
    }
    if (new Date(now) > new Date(shiftStart)) {
      clockRecord.flags.push("LATE_CLOCKIN");
    }

    await clockRecord.save();

    res.status(201).json({
      success: true,
      message: "Clocked in successfully",
      record: {
        _id: clockRecord._id,
        clockInTime: clockRecord.clockInTime,
        status: clockRecord.status,
        flags: clockRecord.flags,
        operationalContext: clockRecord.operationalContext,  // ← NEW: Return context
        staffRole: clockRecord.staffRole  // ← NEW
      }
    });
  } catch (error) {
    console.error("Clock-in error:", error);
    res.status(500).json({
      success: false,
      message: error.message || "Error clocking in"
    });
  }
};
```

**Check the following**:
1. Find the exact line numbers where `clockRecord = new NurseClockRecord` is created in YOUR file
2. Make sure imports at top include `Nurse` model: `import Nurse from '../models/nurse.model.js';`
3. Test endpoint with Postman to verify it still works

**Commit after this step**:
```bash
git add backend/src/controllers/nurse.controller.js
git commit -m "PHASE 1.2: Extract operationalContext in clock-in endpoint

- Extract operationalContext from shift.assignedStaff if available
- Get staffRole from Nurse record (NURSE|PHYSICIAN|SPECIALIST|SUPPORT)
- Pass both to NurseClockRecord on creation
- User can optionally provide context in POST body
- Fully backward compatible: if no context found, defaults to null"
```

---

### Task 1.3: Test backend changes

**Use Postman to test both old and new flows:**

**Test 1: Old style (backward compat)**
```
POST http://localhost:5000/api/nurses/clock-in
Headers: { Authorization: Bearer <YOUR_TOKEN> }
Body: {
  "shiftId": "660f69f9c0d81f001e4a5b2c",
  "location": "ED"
}

Expected: 201 OK
{
  "success": true,
  "record": {
    "_id": "...",
    "clockInTime": "2024-04-20T...",
    "status": "CLOCKED_IN",
    "flags": [],
    "operationalContext": null,  ← Should be null if shift has no context
    "staffRole": "NURSE"
  }
}
```

**Test 2: New style with context**
```
POST http://localhost:5000/api/nurses/clock-in
Headers: { Authorization: Bearer <YOUR_TOKEN> }
Body: {
  "shiftId": "660f69f9c0d81f001e4a5b2c",
  "location": "ED",
  "operationalContext": "TRIAGE_NURSE"  ← NEW
}

Expected: 201 OK
{
  "success": true,
  "record": {
    "_id": "...",
    "clockInTime": "2024-04-20T...",
    "status": "CLOCKED_IN",
    "flags": [],
    "operationalContext": "TRIAGE_NURSE",  ← Should match what we sent
    "staffRole": "NURSE"
  }
}
```

**Test 3: Clock out still works**
```
POST http://localhost:5000/api/nurses/clock-out
Headers: { Authorization: Bearer <YOUR_TOKEN> }

Expected: 200 OK (should have no errors)
```

**If all 3 tests pass**: ✅ Phase 1 is COMPLETE

**If any test fails**:
```bash
# Rollback to checkpoint
git reset --hard feature/role-based-dashboard-routing~2

# Start over with Phase 1
```

---

## PHASE 2: Frontend Context Selector (1 hour)

### Task 2.1: Update ClockInOutBoard.jsx

**File**: `frontend/src/Dashboard/Pages/ClockInOutBoard.jsx`

**Find this component** and add state near top:
```javascript
export default function ClockInOutBoard({ token, user, onLogout, activeTab }) {
  const [currentRecord, setCurrentRecord] = useState(null);
  const [selectedShift, setSelectedShift] = useState(null);
  // ↓ ADD THIS STATE:
  const [selectedContext, setSelectedContext] = useState(null);
  // ... rest of state
}
```

**Find the handleClockIn function** and update it:
```javascript
// BEFORE:
const handleClockIn = async () => {
  const clockInData = {
    shiftId: selectedShift._id,
    location: user?.location || "ED"
  };
  
  const res = await fetch(`${BACKEND_URL}/api/nurses/clock-in`, {
    method: 'POST',
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify(clockInData)
  });
};

// AFTER (add operationalContext):
const handleClockIn = async () => {
  const clockInData = {
    shiftId: selectedShift._id,
    location: user?.location || "ED",
    operationalContext: selectedContext || null  // ← ADD THIS LINE
  };
  
  const res = await fetch(`${BACKEND_URL}/api/nurses/clock-in`, {
    method: 'POST',
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify(clockInData)
  });
};
```

**Find the render/return section** and add context selector UI:
```javascript
// Find where the "Clock In" button is rendered
// Add this BEFORE the button:

{selectedShift && !currentRecord && (
  <div style={{ marginBottom: '20px', padding: '10px', border: '1px solid blue' }}>
    <label>
      <strong>What role are you working today?</strong>
    </label>
    <select 
      value={selectedContext || ""} 
      onChange={(e) => setSelectedContext(e.target.value)}
      style={{ marginLeft: '10px', padding: '5px' }}
    >
      <option value="">-- Select your role --</option>
      {user?.role === 'NURSE' && (
        <>
          <option value="TRIAGE_NURSE">🏥 Triaging patients</option>
          <option value="CLINICAL_NURSE_ER">🚑 Monitoring ER patients</option>
          <option value="CLINICAL_NURSE_WARD">🛏️ Managing ward beds</option>
        </>
      )}
      {user?.role === 'PHYSICIAN' && (
        <>
          <option value="PHYSICIAN_ER">🩺 Reviewing ER cases</option>
          <option value="ON_CALL_PHYSICIAN">📱 On-call (hybrid view)</option>
        </>
      )}
      {user?.role === 'SPECIALIST' && (
        <option value="SPECIALIST">🔬 Specialist consultation</option>
      )}
    </select>
  </div>
)}

{/* Clock In button goes here after selector */}
<button onClick={handleClockIn}>
  Clock In
</button>
```

**Commit after this step**:
```bash
git add frontend/src/Dashboard/Pages/ClockInOutBoard.jsx
git commit -m "PHASE 2.1: Add operationalContext selector to clock-in UI

- Added selectedContext state
- Added role-based context selector dropdown
- Context selector only shows if no active record
- Clock-in now sends operationalContext to backend
- Fully optional: users can clock in without selecting context"
```

---

### Task 2.2: Test frontend clock changes

1. **In browser, navigate to**: `http://localhost:3000/dashboard/clock-in-out`
2. **Verify**:
   - [ ] Clock selector dropdown appears
   - [ ] Can select "Triaging patients", "Monitoring ER patients", etc.
   - [ ] Clicking "Clock In" still works
   - [ ] No console errors

**If frontend works**:
✅ Phase 2 is COMPLETE, proceed to Phase 3

**If frontend breaks**:
```bash
git reset --hard feature/role-based-dashboard-routing~3
git stash
```

---

## PHASE 3: DashboardRouter & Role-Based Routing (2 hours)

### Task 3.1: Create DashboardRouter component

**Create NEW file**: `frontend/src/Dashboard/DashboardRouter.jsx`

```javascript
import React, { useEffect, useState } from 'react';
import { decodeJwt } from '../utils/jwt';

// Import all dashboard components
// (We'll create stubs in next task)
import TriageNurseDashboard from './Dashboards/TriageNurseDashboard';
import ClinicalNurseDashboard_ER from './Dashboards/ClinicalNurseDashboard_ER';
import WardNurseDashboard from './Dashboards/WardNurseDashboard';
import PhysicianDashboard_ER from './Dashboards/PhysicianDashboard_ER';
import OnCallPhysicianDashboard from './Dashboards/OnCallPhysicianDashboard';
import AdminDashboard from './Dashboards/AdminDashboard';
import NurseDashboard from './Dashboards/NurseDashboard'; // Fallback

const BACKEND_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export default function DashboardRouter() {
  const [operationalContext, setOperationalContext] = useState(null);
  const [staffRole, setStaffRole] = useState(null);
  const [staffName, setStaffName] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    initializeUser();
  }, []);

  const initializeUser = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setError('No token found');
        setLoading(false);
        return;
      }

      const decoded = decodeJwt(token);
      setStaffRole(decoded?.role || 'NURSE');
      setStaffName(decoded?.name || 'Staff Member');

      // Fetch current clock record to get operational context
      await fetchCurrentContext(token);
    } catch (err) {
      console.error('Error initializing user:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  const fetchCurrentContext = async (token) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/today-record`, {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        }
      });

      if (!res.ok) {
        // Not clocked in yet, context remains null
        setLoading(false);
        return;
      }

      const data = await res.json();
      if (data.record?.operationalContext) {
        setOperationalContext(data.record.operationalContext);
      }
      setLoading(false);
    } catch (err) {
      console.error('Error fetching context:', err);
      setLoading(false);
      // Don't fail, just continue with null context
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h2>Loading dashboard...</h2>
      </div>
    );
  }

  if (error && error !== 'No token found') {
    return (
      <div style={{ padding: '20px', color: 'red' }}>
        <h2>Error: {error}</h2>
      </div>
    );
  }

  // Route based on role + operational context
  console.log('DashboardRouter:', {
    staffRole,
    operationalContext,
    staffName
  });

  // ADMIN / MANAGER get admin dashboard
  if (staffRole === 'ADMIN' || staffRole === 'MANAGER') {
    return <AdminDashboard staffName={staffName} />;
  }

  // PHYSICIAN routing
  if (staffRole === 'PHYSICIAN') {
    if (operationalContext === 'ON_CALL_PHYSICIAN') {
      return <OnCallPhysicianDashboard staffName={staffName} context={operationalContext} />;
    }
    // Default physician dashboard
    return <PhysicianDashboard_ER staffName={staffName} context={operationalContext} />;
  }

  // NURSE routing (by operational context)
  if (staffRole === 'NURSE') {
    if (operationalContext === 'TRIAGE_NURSE') {
      return <TriageNurseDashboard staffName={staffName} context={operationalContext} />;
    }
    if (operationalContext === 'CLINICAL_NURSE_ER') {
      return <ClinicalNurseDashboard_ER staffName={staffName} context={operationalContext} />;
    }
    if (operationalContext === 'CLINICAL_NURSE_WARD') {
      return <WardNurseDashboard staffName={staffName} context={operationalContext} />;
    }
    // Fallback: if no context selected, send to triage
    // (User hasn't clocked in yet)
    return <TriageNurseDashboard staffName={staffName} context={operationalContext} />;
  }

  // Fallback for other roles
  return <NurseDashboard username={staffName} />;
}
```

**Commit after this step**:
```bash
git add frontend/src/Dashboard/DashboardRouter.jsx
git commit -m "PHASE 3.1: Create DashboardRouter for role-based routing

- Routes by staffRole + operationalContext
- Admin/Manager → AdminDashboard
- Physician → PhysicianDashboard_ER or OnCallPhysicianDashboard
- Nurse → TriageNurseDashboard, ClinicalNurseDashboard_ER, or WardNurseDashboard
- Fetches operational context from /api/nurses/today-record on load
- Fallback to NurseDashboard for backward compatibility"
```

---

### Task 3.2: Create dashboard stub components

**Create folder**: `frontend/src/Dashboard/Dashboards/`

  **Create**: `frontend/src/Dashboard/Dashboards/TriageNurseDashboard.jsx`

```javascript
import React, { useState, useEffect } from 'react';

export default function TriageNurseDashboard({ staffName = 'Staff', context }) {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🏥 TRIAGE NURSE DASHBOARD</h1>
      <p>Welcome, {staffName}! ({context})</p>
      <div style={{ background: '#f0f0f0', padding: '20px', marginTop: '20px' }}>
        <h2>Patient Queue</h2>
        <p>TODO: Migrate existing queue logic here</p>
      </div>
    </div>
  );
}
```

**Create**: `frontend/src/Dashboard/Dashboards/ClinicalNurseDashboard_ER.jsx`

```javascript
import React from 'react';

export default function ClinicalNurseDashboard_ER({ staffName = 'Staff', context }) {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🚑 CLINICAL NURSE - ER DASHBOARD</h1>
      <p>Welcome, {staffName}! ({context})</p>
      <div style={{ background: '#f0f0f0', padding: '20px', marginTop: '20px' }}>
        <h2>Observation Patients</h2>
        <p>TODO: Show ER observation patients</p>
      </div>
    </div>
  );
}
```

**Create**: `frontend/src/Dashboard/Dashboards/WardNurseDashboard.jsx`

```javascript
import React from 'react';

export default function WardNurseDashboard({ staffName = 'Staff', context }) {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🛏️ WARD NURSE DASHBOARD</h1>
      <p>Welcome, {staffName}! ({context})</p>
      <div style={{ background: '#f0f0f0', padding: '20px', marginTop: '20px' }}>
        <h2>Bed Map</h2>
        <p>TODO: Show ward beds and patient assignments</p>
      </div>
    </div>
  );
}
```

**Create**: `frontend/src/Dashboard/Dashboards/PhysicianDashboard_ER.jsx`

```javascript
import React from 'react';

export default function PhysicianDashboard_ER({ staffName = 'Staff', context }) {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🩺 PHYSICIAN COMMAND CENTRE</h1>
      <p>Welcome, Dr. {staffName}! ({context})</p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>
        <div style={{ background: '#e3f2fd', padding: '20px' }}>
          <h2>🔴 Active Cases</h2>
          <p>TODO: Show active ER cases</p>
        </div>
        <div style={{ background: '#fff3e0', padding: '20px' }}>
          <h2>⏳ Observation</h2>
          <p>TODO: Show observation patients</p>
        </div>
        <div style={{ background: '#f3e5f5', padding: '20px' }}>
          <h2>⬆️ Escalations</h2>
          <p>TODO: Show escalation alerts</p>
        </div>
      </div>
    </div>
  );
}
```

**Create**: `frontend/src/Dashboard/Dashboards/OnCallPhysicianDashboard.jsx`

```javascript
import React from 'react';

export default function OnCallPhysicianDashboard({ staffName = 'Staff', context }) {
  return (
    <div style={{ padding: '20px' }}>
      <h1>📱 ON-CALL PHYSICIAN DASHBOARD</h1>
      <p>Welcome, Dr. {staffName}! ({context})</p>
      <div style={{ background: '#f0f0f0', padding: '20px', marginTop: '20px' }}>
        <h2>Hybrid View (ER + Escalations + Consults)</h2>
        <p>TODO: Show on-call view combining multiple departments</p>
      </div>
    </div>
  );
}
```

**Create**: `frontend/src/Dashboard/Dashboards/AdminDashboard.jsx`

```javascript
import React from 'react';

export default function AdminDashboard({ staffName = 'Staff' }) {
  return (
    <div style={{ padding: '20px' }}>
      <h1>⚙️ ADMIN DASHBOARD</h1>
      <p>Welcome, Admin {staffName}!</p>
      <div style={{ background: '#f0f0f0', padding: '20px', marginTop: '20px' }}>
        <h2>System Management</h2>
        <p>TODO: Show admin controls, staff management, analytics</p>
      </div>
    </div>
  );
}
```

**Create**: `frontend/src/Dashboard/Dashboards/NurseDashboard.jsx` (import the existing one)

Actually, this already exists. Just import it from the existing location in DashboardRouter.

**Commit after this step**:
```bash
git add frontend/src/Dashboard/Dashboards/
git commit -m "PHASE 3.2: Create dashboard stub components

- TriageNurseDashboard: for TRIAGE_NURSE context
- ClinicalNurseDashboard_ER: for CLINICAL_NURSE_ER context  
- WardNurseDashboard: for CLINICAL_NURSE_WARD context
- PhysicianDashboard_ER: for PHYSICIAN_ER context
- OnCallPhysicianDashboard: for ON_CALL_PHYSICIAN context
- AdminDashboard: for staff with ADMIN|MANAGER role
- All stubs have working skeleton UI, ready for migration"
```

---

### Task 3.3: Update App.jsx routing

**File**: `frontend/src/App.jsx`

**Find the route** for dashboard (around where other routes are defined):
```javascript
// BEFORE:
import Dashboard from "./Dashboard/Dashboard.jsx";  // OLD

// Later in routes:
<Route path="/dashboard" element={<Dashboard />} />

// AFTER:
import DashboardRouter from "./Dashboard/DashboardRouter.jsx";  // NEW

// Later in routes:
<Route path="/dashboard" element={<DashboardRouter />} />
```

**Commit after this step**:
```bash
git add frontend/src/App.jsx
git commit -m "PHASE 3.3: Route /dashboard to DashboardRouter

- /dashboard now uses DashboardRouter instead of old Dashboard
- Old Dashboard.jsx still exists as fallback
- Role-based routing now active"
```

---

### Task 3.4: Test routing

**In browser**:
1. **Go to**: `http://localhost:3000/dashboard`
2. **If logged in as NURSE with TRIAGE_NURSE context**:
   - [ ] Should see: "🏥 TRIAGE NURSE DASHBOARD"
3. **If logged in as PHYSICIAN**:
   - [ ] Should see: "🩺 PHYSICIAN COMMAND CENTRE"
4. **If logged in as ADMIN**:
   - [ ] Should see: "⚙️ ADMIN DASHBOARD"
5. **Check browser console**:
   - [ ] No errors about missing imports

**If routing works**:
✅ Phase 3 is COMPLETE

**If routing breaks**:
```bash
git reset --hard feature/role-based-dashboard-routing~4
```

---

## PHASE 4: Migrate Dashboard Logic (2-3 hours per dashboard)

### Task 4.1: Migrate TriageNurseDashboard (CRITICAL FIRST)

**Source**: Look at existing `frontend/src/Dashboard/Encounters.jsx` or `NurseDashboard.jsx`

**Steps**:
1. Copy all the useEffect hooks from old NurseDashboard
2. Copy all the state variables
3. Copy all the fetching logic
4. Copy all the UI rendering logic
5. Paste into TriageNurseDashboard.jsx
6. Update component name references

**After migration**:
```bash
git add frontend/src/Dashboard/Dashboards/TriageNurseDashboard.jsx
git commit -m "PHASE 4.1: Migrate TriageNurseDashboard logic

- Copied all queue-related logic from NurseDashboard
- Updated state management
- Updated API calls
- Tested with real patient data"
```

### Continue with remaining dashboards (similar pattern)

---

## Phase 5: Testing & Validation

### Test Checklist

```bash
# Test all roles can still log in and clock in
- [ ] NURSE logs in → Clock-in works → Context selector appears
- [ ] PHYSICIAN logs in → Clock-in works → Context selector shows physician options
- [ ] ADMIN logs in → Can clock in (if assigned) or skip

# Test routing works
- [ ] TRIAGE_NURSE → Sees TriageNurseDashboard
- [ ] CLINICAL_NURSE_ER → Sees ClinicalNurseDashboard_ER
- [ ] PHYSICIAN_ER → Sees PhysicianDashboard_ER
- [ ] ADMIN → Sees AdminDashboard

# Test backward compat
- [ ] Old nurses (no context) → Still see TriageNurseDashboard (fallback)
- [ ] Break in/out → Still works
- [ ] Attendance history → Loads correctly

# Test no regressions
- [ ] Existing functionality doesn't break
- [ ] No console errors
- [ ] No database migration needed
```

---

## Final: Merge to main branch

Once all phases tested and working:

```bash
# Make sure everything committed
git status  # Should show "nothing to commit"

# Return to main
git checkout main

# Merge feature branch
git merge --no-ff feature/role-based-dashboard-routing -m "Feature: Role-based dashboard routing

Summary:
- Added operationalContext tracking to clock records
- Created role-specific dashboard components
- Implemented DashboardRouter for intelligent routing
- Preserved backward compatibility throughout
- All tests passing"

# Push to origin if using git remote
# git push origin main
```

---

## Troubleshooting

### If clock-in endpoint returns error:

```bash
# Check backend logs for error details
# Verify Nurse record exists with role field
# Test with Postman to debug

# If import error (Nurse model not found):
# Check: backend/src/controllers/nurse.controller.js line 1-5
# Should have: import Nurse from '../models/nurse.model.js';
```

### If DashboardRouter fails to load:

```bash
# Check browser console for error
# Verify all imported components exist:
ls frontend/src/Dashboard/Dashboards/
# Should list all 6 dashboard components

# If fetchCurrentContext fails:
# Check network tab for errors on /api/nurses/today-record
# Verify backend is running
```

### If role-based routing doesn't work:

```bash
# Add debug logging:
console.log('DashboardRouter:', { staffRole, operationalContext });

# Verify token has role field:
# In browser console: 
// const token = localStorage.getItem('token');
// const decoded = jwt_decode(token);
// console.log(decoded);

# Should see: role: "PHYSICIAN" or similar
```

---

## Summary

**Completed**: 
- ✅ Backend models updated (backward compat)
- ✅ Clock-in extraction logic (operationalContext)
- ✅ Frontend context selector (optional)
- ✅ DashboardRouter (role-based routing)
- ✅ Dashboard stubs (all roles)
- ✅ App.jsx routing updated
- ✅ Testing plan defined

**Next**: Execute each phase in order, test thoroughly between phases

**Key**: Don't skip testing steps. Each phase must work before moving to next.

Ready? 🚀
