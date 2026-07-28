# ROLE-BASED CONDITIONAL RENDERING - IMPLEMENTATION PLAN

**Objective**: When staff members clock in, system detects their role and conditionally renders the appropriate dashboard

**Current Checkpoint**: Staff assignments defined in HOSPITAL_SCENARIO_REGISTRATION_GUIDE.md

**Timeline**: This session

---

## PART 1: UNDERSTAND THE ROUTING FLOW

### Current System Flow (What Exists)

```
User Login
  ↓
Verify credentials (email + password)
  ↓
User Profile Loaded (role: NURSE | PHYSICIAN)
  ↓
Dashboard.jsx rendered
  ↓
❓ QUESTION: What dashboard should they see?
  ← THIS IS WHERE WE NEED TO IMPLEMENT LOGIC
```

### Desired System Flow (What We're Building)

```
User Login (email + password)
  ↓
Load User Profile + Shift Context
  ↓
GET /api/user/dashboard-config
  ↓ Returns:
  {
    context: "TRIAGE_NURSE" | "CLINICAL_NURSE_ER" | "PHYSICIAN_ER" | "CLINICAL_NURSE_WARD",
    role: "NURSE" | "PHYSICIAN",
    shift: { ... },
    dashboardType: "TriageNurseDashboard" | "ClinicalNurseDashboard" | ...
  }
  ↓
DashboardRouter.jsx detects context
  ↓
Conditionally render:
  ├─ If TRIAGE_NURSE → <TriageNurseDashboard />
  ├─ If CLINICAL_NURSE_ER → <ClinicalNurseDashboard_ER />
  ├─ If PHYSICIAN_ER → <PhysicianDashboard_ER />
  └─ If CLINICAL_NURSE_WARD → <WardNurseDashboard />
  ↓
User sees their specific dashboard with their specific controls
```

---

## PART 2: CURRENT SYSTEM AUDIT

### What Already Exists

**Backend**:
- ✅ User authentication (login)
- ✅ Role field in Nurse model: `role: ['NURSE', 'PHYSICIAN', 'SPECIALIST']`
- ✅ Department field in Nurse model
- ✅ Shift model with basic assignment
- ✅ Clock in/out system (NurseClockRecord)

**Frontend**:
- ✅ LoginForm.jsx (authentication)
- ✅ Dashboard.jsx (main dashboard component)
- ✅ ClockInOut.jsx (clock-in workflow)
- ✅ NurseDashboard.jsx (general dashboard)
- ✅ ShiftSelectionModal.jsx (shift picker)

**What's Missing**:
- ❌ `shiftContext` field in Shift model (NEW - CRITICAL)
- ❌ `GET /api/user/dashboard-config` endpoint (NEW - CRITICAL)
- ❌ DashboardRouter.jsx component (NEW - CRITICAL)
- ❌ Individual dashboard components (NEW)
- ❌ Conditional rendering logic based on context

---

## PART 3: DETAILED IMPLEMENTATION PLAN

### Layer 1: Database Models (Backend)

#### 1a. Enhance Shift Model

**File**: `backend/src/models/shift.model.js`

**Current**: Basic shift with start/end times, assigned staff

**Add these fields**:

```javascript
// NEW: Context for each assigned staff member
shiftContextAssignments: [
  {
    staffId: mongoose.Schema.Types.ObjectId,
    staffName: String,
    role: String,  // NURSE | PHYSICIAN
    
    // ⭐ THE KEY FIELD: What dashboard should this person see?
    context: {
      type: String,
      enum: [
        "TRIAGE_NURSE",           // Processes triage queue
        "CLINICAL_NURSE_ER",      // Manages observation patients
        "CLINICAL_NURSE_WARD",    // Manages hospitalized patients
        "PHYSICIAN_ER",           // Reviews ER cases + observation
        "PHYSICIAN_WARD",         // Ward rounds
      ]
    },
    
    // Optional: Team leader assignment
    isTeamLead: { type: Boolean, default: false },
    assignedTeam: [mongoose.Schema.Types.ObjectId],  // If this person is a team lead
  }
]

// Example Shift Document:
{
  _id: "shift_001",
  shiftName: "Tuesday ED Shift",
  date: "2026-04-20",
  startTime: "07:00",
  endTime: "15:00",
  
  // OLD (still there)
  assignedStaff: ["sarah_id", "bob_id", "chen_id"],
  
  // NEW (what we're adding)
  shiftContextAssignments: [
    {
      staffId: "sarah_id",
      staffName: "Sarah Triage",
      role: "NURSE",
      context: "TRIAGE_NURSE",    ← Sarah sees TRIAGE dashboard
      isTeamLead: true,            ← Sarah is team lead (assigns others)
      assignedTeam: ["bob_id", "chen_id"]
    },
    {
      staffId: "bob_id",
      staffName: "Bob Martinez",
      role: "NURSE",
      context: "CLINICAL_NURSE_ER",  ← Bob sees CLINICAL dashboard
      isTeamLead: false
    },
    {
      staffId: "chen_id",
      staffName: "Dr. Chen MD",
      role: "PHYSICIAN",
      context: "PHYSICIAN_ER",      ← Dr. Chen sees PHYSICIAN dashboard
      isTeamLead: false
    }
  ]
}
```

**Migration Plan**:
1. Add `shiftContextAssignments` array field to schema
2. Make backward compatible (existing shifts still work)
3. When creating new shift, populate this field

---

#### 1b. New Model (Optional): ShiftContext

**Alternative Approach** (cleaner):

```javascript
// backend/src/models/shiftContext.model.js

const shiftContextSchema = new mongoose.Schema({
  shiftId: mongoose.Schema.Types.ObjectId,
  staffId: mongoose.Schema.Types.ObjectId,
  
  // USER CONTEXT (determines dashboard)
  context: {
    type: String,
    enum: ["TRIAGE_NURSE", "CLINICAL_NURSE_ER", "PHYSICIAN_ER", "CLINICAL_NURSE_WARD"],
    required: true
  },
  
  // TEAM INFO (if applicable)
  isTeamLead: Boolean,
  assignedTeam: [mongoose.Schema.Types.ObjectId],
  
  // WHEN THIS WAS ASSIGNED
  assignedAt: Date,
  assignedBy: mongoose.Schema.Types.ObjectId,
  
  // TRACKING
  clockedInAt: Date,
  clockedOutAt: Date,
  status: { type: String, enum: ["PENDING", "ACTIVE", "COMPLETED"] }
});

export const ShiftContext = mongoose.model("ShiftContext", shiftContextSchema);
```

**I recommend**: Use the Shift model enhancement (simpler, less schema fragmentation)

---

### Layer 2: Backend API

#### 2a. New Endpoint: `GET /api/user/dashboard-config`

**File**: `backend/src/routes/user.route.js` (create if doesn't exist)

```javascript
/**
 * GET /api/user/dashboard-config
 * 
 * Returns dashboard configuration based on user's shift context
 * Called when user logs in or navigates to dashboard
 * 
 * Request: 
 *   Headers: Authorization: Bearer {token}
 * 
 * Response:
 *   {
 *     userId: string,
 *     name: string,
 *     role: "NURSE" | "PHYSICIAN",
 *     context: "TRIAGE_NURSE" | "CLINICAL_NURSE_ER" | ...,
 *     shift: { _id, name, date, startTime, endTime },
 *     dashboard: "TriageNurseDashboard" | "ClinicalNurseDashboard_ER" | ...,
 *     permissions: { canAssignTeam, canTriagePatients, ... },
 *     isClocked: boolean,
 *     clockRecord: { clockedInAt, breakStartAt, ... }
 *   }
 */

export async function getDashboardConfig(req, res) {
  const userId = req.user._id;
  const hospitalId = req.user.hospitalId;

  try {
    // 1. Get user full profile
    const user = await Nurse.findById(userId).populate('hospitalId');
    
    if (!user || !user.hospitalId) {
      return res.status(404).json({ error: "User or hospital not found" });
    }

    // 2. Find active shift for today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const activeShift = await Shift.findOne({
      hospitalId,
      date: { $gte: today, $lt: tomorrow },
      assignedStaff: userId  // User is assigned to this shift
    });

    if (!activeShift) {
      return res.status(404).json({
        error: "NO_ACTIVE_SHIFT",
        message: "You have no assigned shift for today"
      });
    }

    // 3. Determine user's context in this shift
    let userContext = null;
    let isTeamLead = false;
    let assignedTeam = [];

    // Check shiftContextAssignments (NEW approach)
    if (activeShift.shiftContextAssignments && activeShift.shiftContextAssignments.length > 0) {
      const contextAssignment = activeShift.shiftContextAssignments.find(
        ctx => ctx.staffId.toString() === userId.toString()
      );
      
      if (contextAssignment) {
        userContext = contextAssignment.context;
        isTeamLead = contextAssignment.isTeamLead || false;
        assignedTeam = contextAssignment.assignedTeam || [];
      }
    }

    // Fallback: Derive context from role + department (BACKWARD COMPAT)
    if (!userContext) {
      userContext = deriveContextFromRoleAndDepartment(user.role, user.department);
    }

    // 4. Check if user is clocked in
    const clockRecord = await NurseClockRecord.findOne({
      nurseId: userId,
      date: today,
      clockOutTime: null
    });

    const isClocked = !!clockRecord;

    // 5. Map context to dashboard component
    const dashboardMap = {
      TRIAGE_NURSE: "TriageNurseDashboard",
      CLINICAL_NURSE_ER: "ClinicalNurseDashboard_ER",
      CLINICAL_NURSE_WARD: "WardNurseDashboard",
      PHYSICIAN_ER: "PhysicianClinicalDashboard_ER",
      PHYSICIAN_WARD: "WardRoundsPhysician"
    };

    // 6. Get permissions based on context
    const permissions = getPermissionsForContext(userContext);

    // 7. Return config
    res.json({
      userId: user._id,
      name: `${user.name} ${user.surname}`,
      role: user.role,
      context: userContext,
      shift: {
        _id: activeShift._id,
        name: activeShift.shiftName,
        date: activeShift.date,
        startTime: activeShift.shiftStart,
        endTime: activeShift.shiftEnd,
        department: activeShift.department,
        isTeamLead,
        assignedTeam
      },
      dashboard: dashboardMap[userContext] || "NurseDashboard",  // Fallback
      permissions,
      isClocked,
      clockRecord: clockRecord ? {
        clockedInAt: clockRecord.clockInTime,
        breakStartAt: clockRecord.breakStartTime,
        current: "WORKING"  // or "ON_BREAK"
      } : null
    });

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}

// ====== HELPER FUNCTIONS ======

function deriveContextFromRoleAndDepartment(role, department) {
  // Fallback logic if shiftContextAssignments not set
  
  if (role === "PHYSICIAN") {
    return department === "Ward" ? "PHYSICIAN_WARD" : "PHYSICIAN_ER";
  }
  
  if (role === "NURSE") {
    if (department === "ED") {
      // For ED nurses, default to CLINICAL
      // (Triage must be explicitly assigned in shiftContextAssignments)
      return "CLINICAL_NURSE_ER";
    }
    if (department === "Ward") {
      return "CLINICAL_NURSE_WARD";
    }
  }
  
  return "NurseDashboard";  // Generic fallback
}

function getPermissionsForContext(context) {
  const permissionsMap = {
    TRIAGE_NURSE: {
      canCanAssignTeam: true,
      canPerformTriage: true,
      canViewClinicalMode: false,
      canViewWards: false,
      canDischargePatients: false
    },
    CLINICAL_NURSE_ER: {
      canAssignTeam: false,
      canPerformTriage: false,
      canViewClinicalMode: true,
      canViewWards: false,
      canRequestPhysician: true,
      canLogVitals: true
    },
    CLINICAL_NURSE_WARD: {
      canAssignTeam: false,
      canPerformTriage: false,
      canViewClinicalMode: true,
      canViewWards: true,
      canLogVitals: true,
      canManageMedications: true
    },
    PHYSICIAN_ER: {
      canViewClinicalMode: true,
      canMakeDispo: true,      // Disposition (ADMIT/DISCHARGE)
      canSignOff: true,
      canCreateEHRBundle: true
    },
    PHYSICIAN_WARD: {
      canViewWards: true,
      canMakeDispo: true,
      canOrderDischarge: true
    }
  };

  return permissionsMap[context] || {};
}
```

**Endpoint Registration**:

```javascript
// backend/src/routes/index.js or main router file

import { getDashboardConfig } from './user.route.js';

router.get('/api/user/dashboard-config', protect, getDashboardConfig);
```

---

### Layer 3: Frontend Router Component

#### 3a. New Component: DashboardRouter.jsx

**File**: `frontend/src/Dashboard/DashboardRouter.jsx`

```javascript
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// Import all dashboard variants
import TriageNurseDashboard from './Dashboards/TriageNurseDashboard';
import ClinicalNurseDashboard_ER from './Dashboards/ClinicalNurseDashboard_ER';
import WardNurseDashboard from './Dashboards/WardNurseDashboard';
import PhysicianClinicalDashboard_ER from './Dashboards/PhysicianClinicalDashboard_ER';
import WardRoundsPhysician from './Dashboards/WardRoundsPhysician';
import NurseDashboard from './NurseDashboard'; // Fallback generic dashboard

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

/**
 * DASHBOARD ROUTER
 * 
 * Main component that:
 * 1. Fetches user's dashboard config
 * 2. Detects their role/context
 * 3. Conditionally renders appropriate dashboard
 * 
 * This replaces the need for multiple manual routing
 */
const DashboardRouter = ({ token, user }) => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!token) {
      navigate('/login');
      return;
    }
    
    fetchDashboardConfig();
  }, [token]);

  const fetchDashboardConfig = async () => {
    try {
      const response = await fetch(
        `${BACKEND_URL}/api/user/dashboard-config`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.status === 404) {
        const data = await response.json();
        if (data.error === 'NO_ACTIVE_SHIFT') {
          setError('You have no active shift for today. Please check with your manager.');
          return;
        }
      }

      if (!response.ok) throw new Error('Failed to fetch config');

      const dashboardConfig = await response.json();
      setConfig(dashboardConfig);
      console.log('✅ Dashboard Config:', dashboardConfig);

    } catch (err) {
      setError(err.message);
      console.error('❌ Error fetching dashboard config:', err);
    } finally {
      setLoading(false);
    }
  };

  // ====== RENDER LOGIC ======

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loadingSpinner}>
          <h2>Loading your dashboard...</h2>
          <p>Detecting your role and shift context</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.errorBox}>
          <h2>⚠️ Dashboard Error</h2>
          <p>{error}</p>
          <button 
            onClick={() => navigate('/')}
            style={styles.button}
          >
            Return to Home
          </button>
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div style={styles.container}>
        <div style={styles.errorBox}>
          <h2>⚠️ No Configuration</h2>
          <p>Unable to determine your dashboard type</p>
        </div>
      </div>
    );
  }

  // ====== CONDITIONAL RENDERING BASED ON CONTEXT ======

  const renderDashboard = () => {
    console.log(`🚀 Rendering dashboard for context: ${config.context}`);

    switch (config.context) {
      case 'TRIAGE_NURSE':
        return <TriageNurseDashboard config={config} token={token} />;

      case 'CLINICAL_NURSE_ER':
        return <ClinicalNurseDashboard_ER config={config} token={token} />;

      case 'CLINICAL_NURSE_WARD':
        return <WardNurseDashboard config={config} token={token} />;

      case 'PHYSICIAN_ER':
        return <PhysicianClinicalDashboard_ER config={config} token={token} />;

      case 'PHYSICIAN_WARD':
        return <WardRoundsPhysician config={config} token={token} />;

      default:
        console.warn(`⚠️ Unknown context: ${config.context}, using generic dashboard`);
        return <NurseDashboard token={token} />;
    }
  };

  // ====== RENDER WITH HEADER ======

  return (
    <div style={styles.wrapper}>
      {/* DASHBOARD HEADER */}
      <div style={styles.header}>
        <div>
          <h1>👤 {config.name}</h1>
          <p style={styles.subtitle}>
            {config.role} | {config.context} 
            {config.shift.isTeamLead && ' | 👑 TEAM LEAD'}
          </p>
          {config.isClocked && (
            <p style={styles.clockedIn}>
              ✅ Clocked In: {new Date(config.clockRecord.clockedInAt).toLocaleTimeString()}
            </p>
          )}
        </div>
        <div style={styles.shiftInfo}>
          <p><strong>Shift:</strong> {config.shift.startTime} - {config.shift.endTime}</p>
          <p><strong>Department:</strong> {config.shift.department}</p>
          {config.shift.assignedTeam && config.shift.assignedTeam.length > 0 && (
            <p><strong>Team Size:</strong> {config.shift.assignedTeam.length + 1}</p>
          )}
        </div>
      </div>

      {/* MAIN DASHBOARD CONTENT */}
      <div style={styles.content}>
        {renderDashboard()}
      </div>
    </div>
  );
};

const styles = {
  wrapper: {
    minHeight: '100vh',
    backgroundColor: '#f3f4f6'
  },
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: '100vh',
    backgroundColor: '#f3f4f6'
  },
  loadingSpinner: {
    textAlign: 'center',
    padding: '40px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
  },
  errorBox: {
    textAlign: 'center',
    padding: '40px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    border: '2px solid #dc2626'
  },
  button: {
    marginTop: '20px',
    padding: '10px 20px',
    backgroundColor: '#0284c7',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px 30px',
    backgroundColor: 'white',
    borderBottom: '1px solid #e5e7eb',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
  },
  subtitle: {
    color: '#6b7280',
    fontSize: '14px',
    marginTop: '4px'
  },
  clockedIn: {
    color: '#16a34a',
    fontSize: '13px',
    fontWeight: '600',
    marginTop: '4px'
  },
  shiftInfo: {
    textAlign: 'right',
    fontSize: '14px'
  },
  content: {
    padding: '20px',
    maxWidth: '1400px',
    margin: '0 auto'
  }
};

export default DashboardRouter;
```

---

#### 3b. Update App.jsx to Use Router

**File**: `frontend/src/App.jsx`

```javascript
// BEFORE (old way)
import Dashboard from './Dashboard/Dashboard';

// AFTER (new way)
import DashboardRouter from './Dashboard/DashboardRouter';  // ← NEW

function App() {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);

  // ... existing code ...

  return (
    <Routes>
      <Route path="/login" element={<LoginForm ... />} />
      {/* OLD */}
      {/* <Route path="/dashboard" element={<Dashboard token={token} />} /> */}
      
      {/* NEW - Router handles conditional rendering */}
      <Route 
        path="/dashboard" 
        element={<DashboardRouter token={token} user={user} />} 
      />
      {/* ... rest of routes ... */}
    </Routes>
  );
}
```

---

### Layer 4: Individual Dashboard Components (Stubs)

#### 4a. TriageNurseDashboard.jsx

**File**: `frontend/src/Dashboard/Dashboards/TriageNurseDashboard.jsx`

```javascript
import React, { useState } from 'react';
import SidebarLayout from '../SidebarLayout';

const TriageNurseDashboard = ({ config, token }) => {
  const [queue, setQueue] = useState([]);

  return (
    <SidebarLayout username={config.name}>
      <div style={styles.container}>
        <h1>🏥 TRIAGE QUEUE - Emergency Department</h1>
        
        <div style={styles.infoBox}>
          <p>✨ <strong>Role Context:</strong> {config.context}</p>
          <p>💼 <strong>Status:</strong> {config.isTeamLead ? '👑 TEAM LEAD' : 'Triage Nurse'}</p>
          {config.shift.isTeamLead && (
            <p>👥 <strong>Assigned Team:</strong> {config.shift.assignedTeam.length} people</p>
          )}
        </div>

        <div style={styles.queue}>
          <h2>📋 Patients Waiting ({queue.length})</h2>
          {queue.length === 0 ? (
            <p style={styles.empty}>No patients waiting</p>
          ) : (
            queue.map(patient => (
              <div key={patient._id} style={styles.patientCard}>
                {/* Patient card content */}
              </div>
            ))
          )}
        </div>

        {config.shift.isTeamLead && (
          <div style={styles.controlsBox}>
            <h3>👥 Team Assignment</h3>
            <p>Button to assign clinical team would go here</p>
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

const styles = {
  container: { padding: '20px' },
  infoBox: { 
    backgroundColor: '#ecfdf5', 
    padding: '15px', 
    borderRadius: '8px',
    marginBottom: '20px',
    border: '1px solid #86efac'
  },
  queue: { marginBottom: '20px' },
  empty: { color: '#9ca3af', fontStyle: 'italic' },
  patientCard: { 
    padding: '12px', 
    border: '1px solid #e5e7eb',
    borderRadius: '6px',
    marginBottom: '10px',
    backgroundColor: 'white'
  },
  controlsBox: {
    backgroundColor: '#eff6ff',
    padding: '15px',
    borderRadius: '8px',
    border: '1px solid #bfdbfe'
  }
};

export default TriageNurseDashboard;
```

#### 4b. ClinicalNurseDashboard_ER.jsx

```javascript
import React, { useState, useEffect } from 'react';
import SidebarLayout from '../SidebarLayout';

const ClinicalNurseDashboard_ER = ({ config, token }) => {
  const [patients, setPatients] = useState([]);

  return (
    <SidebarLayout username={config.name}>
      <div style={styles.container}>
        <h1>⚕️ CLINICAL MODE - Observation Patients</h1>
        
        <div style={styles.infoBox}>
          <p>✨ <strong>Role Context:</strong> {config.context}</p>
          <p>💼 <strong>Department:</strong> {config.shift.department}</p>
          <p>📍 <strong>Assigned Physician:</strong> Waiting for team assignment</p>
        </div>

        <div style={styles.patientList}>
          <h2>🛏️ Patients in Observation ({patients.length})</h2>
          {patients.length === 0 ? (
            <p style={styles.empty}>No patients currently in observation</p>
          ) : (
            patients.map((patient, idx) => (
              <div key={idx} style={styles.patientCard}>
                {/* Patient observation card */}
              </div>
            ))
          )}
        </div>
      </div>
    </SidebarLayout>
  );
};

const styles = {
  container: { padding: '20px' },
  infoBox: { 
    backgroundColor: '#eff6ff', 
    padding: '15px', 
    borderRadius: '8px',
    marginBottom: '20px',
    border: '1px solid #bfdbfe'
  },
  patientList: { marginBottom: '20px' },
  empty: { color: '#9ca3af', fontStyle: 'italic' },
  patientCard: { 
    padding: '12px', 
    border: '1px solid #e5e7eb',
    borderRadius: '6px',
    marginBottom: '10px',
    backgroundColor: 'white'
  }
};

export default ClinicalNurseDashboard_ER;
```

#### 4c. PhysicianClinicalDashboard_ER.jsx

```javascript
import React from 'react';
import SidebarLayout from '../SidebarLayout';

const PhysicianClinicalDashboard_ER = ({ config, token }) => {
  return (
    <SidebarLayout username={config.name}>
      <div style={styles.container}>
        <h1>🩺 PHYSICIAN COMMAND CENTRE - Emergency Department</h1>
        
        <div style={styles.infoBox}>
          <p>✨ <strong>Role Context:</strong> {config.context}</p>
          <p>🔬 <strong>Specialty:</strong> Emergency Medicine</p>
          <p>⏱️ <strong>Response Time Target:</strong> &lt;5 minutes</p>
        </div>

        <div style={styles.laneContainer}>
          <div style={styles.lane}>
            <h3>🔴 Active ER Cases (0)</h3>
            <p style={styles.empty}>No active cases</p>
          </div>
          
          <div style={styles.lane}>
            <h3>🟡 Observation/Awaiting Disposition (0)</h3>
            <p style={styles.empty}>No patients in observation</p>
          </div>
          
          <div style={styles.lane}>
            <h3>🟣 Ward Escalations (0)</h3>
            <p style={styles.empty}>No escalations</p>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
};

const styles = {
  container: { padding: '20px' },
  infoBox: { 
    backgroundColor: '#fff7ed', 
    padding: '15px', 
    borderRadius: '8px',
    marginBottom: '20px',
    border: '1px solid #fed7aa'
  },
  laneContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '15px'
  },
  lane: {
    backgroundColor: 'white',
    padding: '15px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    minHeight: '250px'
  },
  empty: { color: '#9ca3af', fontStyle: 'italic' }
};

export default PhysicianClinicalDashboard_ER;
```

#### 4d. WardNurseDashboard.jsx (stub)

```javascript
import React from 'react';
import SidebarLayout from '../SidebarLayout';

const WardNurseDashboard = ({ config, token }) => {
  return (
    <SidebarLayout username={config.name}>
      <div style={styles.container}>
        <h1>🛏️ WARD NURSING - Hospitalized Patients</h1>
        
        <div style={styles.infoBox}>
          <p>✨ <strong>Role Context:</strong> {config.context}</p>
          <p>🏥 <strong>Ward:</strong> {config.shift.department}</p>
        </div>

        <p style={styles.placeholder}>Ward bed map and patient management would display here</p>
      </div>
    </SidebarLayout>
  );
};

const styles = {
  container: { padding: '20px' },
  infoBox: { 
    backgroundColor: '#fae8ff', 
    padding: '15px', 
    borderRadius: '8px',
    marginBottom: '20px',
    border: '1px solid #f0d9ff'
  },
  placeholder: { color: '#9ca3af', fontStyle: 'italic' }
};

export default WardNurseDashboard;
```

---

## PART 4: IMPLEMENTATION SEQUENCE

### Phase 1: Backend (2-3 hours)

```
1. Enhance Shift model
   ├─ Add shiftContextAssignments array field
   └─ Make backward compatible

2. Create GET /api/user/dashboard-config endpoint
   ├─ Fetch user's shift
   ├─ Determine context
   ├─ Return dashboard config + permissions
   └─ Test with Postman

3. Test Endpoint
   ├─ Login as Sarah → GET config → Returns TRIAGE_NURSE
   ├─ Login as Bob → GET config → Returns CLINICAL_NURSE_ER
   ├─ Login as Dr. Chen → GET config → Returns PHYSICIAN_ER
   └─ Verify all fields correct
```

### Phase 2: Frontend (2-3 hours)

```
1. Create DashboardRouter.jsx component
   ├─ Fetch dashboard config on mount
   ├─ Handle loading states
   ├─ Handle error states
   └─ Display header with user info

2. Create individual dashboard stubs
   ├─ TriageNurseDashboard.jsx
   ├─ ClinicalNurseDashboard_ER.jsx
   ├─ PhysicianClinicalDashboard_ER.jsx
   ├─ WardNurseDashboard.jsx
   ├─ WardRoundsPhysician.jsx (optional)
   └─ Each displays their context

3. Update App.jsx router
   ├─ Replace Dashboard with DashboardRouter
   └─ Pass token and user props

4. Test flow
   ├─ Login as Sarah → Clock in → See TRIAGE dashboard
   ├─ Login as Bob → Clock in → See CLINICAL dashboard
   ├─ Verify header shows correct role/context
   └─ Test error handling (no shift assigned)
```

### Phase 3: Integration (1-2 hours)

```
1. Connect shift assignment to database
   ├─ When creating shift, populate shiftContextAssignments
   └─ Ensure all 4 staff have contexts assigned

2. End-to-end test
   ├─ Register hospital and staff (already done)
   ├─ Create shift with context assignments
   ├─ All staff clock in
   ├─ Verify each sees their dashboard
   └─ Test context switching

3. Error handling
   ├─ What if staff not in shiftContextAssignments?
   ├─ What if no active shift?
   ├─ What if contexts not defined?
   └─ Graceful fallbacks
```

---

## PART 5: CURRENT SYSTEM AUDIT - WHAT EXISTS

### Already Built

**Nursing Models**:
- ✅ Nurse.model.js with role/department/specialty

**Authentication**:
- ✅ LoginForm.jsx
- ✅ JWT tokens + Bearer auth

**Shift Management**:
- ✅ Shift.model.js (basic)
- ✅ NurseSchedule, NurseAssignment models
- ✅ Clock in/out system (NurseClockRecord)

**Dashboards (General)**:
- ✅ Dashboard.jsx (main entry point)
- ✅ NurseDashboard.jsx (generic nurse view)
- ✅ SidebarLayout.jsx (layout wrapper)

### What Needs Adding

**Models**:
- ❌ shiftContextAssignments array in Shift

**APIs**:
- ❌ GET /api/user/dashboard-config

**Frontend**:
- ❌ DashboardRouter.jsx
- ❌ TriageNurseDashboard.jsx
- ❌ ClinicalNurseDashboard_ER.jsx
- ❌ PhysicianClinicalDashboard_ER.jsx
- ❌ WardNurseDashboard.jsx

---

## PART 6: BUILD CHECKLIST

### Backend Checklist

```
☐ Step 1: Enhance Shift model
  ☐ Add shiftContextAssignments field with full schema
  ☐ Keep backward compatible
  ☐ Test migration on existing shifts

☐ Step 2: Build getDashboardConfig API
  ☐ Create endpoint handler
  ☐ Get user profile
  ☐ Find active shift for today
  ☐ Determine context
  ☐ Check clock status
  ☐ Return config JSON

☐ Step 3: Register endpoint in router
  ☐ Add route: GET /api/user/dashboard-config
  ☐ Add protect middleware
  ☐ Test with Postman

☐ Step 4: Test all 4 staff scenarios
  ☐ Sarah → TRIAGE_NURSE
  ☐ Bob → CLINICAL_NURSE_ER
  ☐ Dr. Chen → PHYSICIAN_ER
  ☐ Ward Nurse → CLINICAL_NURSE_WARD
```

### Frontend Checklist

```
☐ Step 1: Create DashboardRouter.jsx
  ☐ useEffect to fetch config on mount
  ☐ Handle loading, error, success states
  ☐ Render header with user info
  ☐ Switch/case on context
  ☐ Conditional render dashboards

☐ Step 2: Create 4 dashboard components
  ☐ TriageNurseDashboard.jsx (basic stub)
  ☐ ClinicalNurseDashboard_ER.jsx (basic stub)
  ☐ PhysicianClinicalDashboard_ER.jsx (basic stub)
  ☐ WardNurseDashboard.jsx (basic stub)
  ☐ Each shows their context in header

☐ Step 3: Update App.jsx
  ☐ Import DashboardRouter
  ☐ Replace Dashboard route
  ☐ Pass token and user props
  ☐ Test import/export

☐ Step 4: Test e2e flow
  ☐ Login as Sarah
  ☐ Clock in to shift
  ☐ Verify TRIAGE dashboard renders
  ☐ Repeat for Bob, Dr. Chen, Ward Nurse
  ☐ Verify permissions shown correctly
  ☐ Test error: no shift assigned
```

---

## PART 7: EXPECTED BEHAVIOR AFTER IMPLEMENTATION

### Sarah (Triage Nurse) Flow

```
1. Sarah logs in
   Email: sarah.triage@stmercy.hospital
   Password: sarah_secure_123

2. Redirected to /dashboard

3. DashboardRouter fetches config
   GET /api/user/dashboard-config
   
4. Backend returns:
   {
     name: "Sarah Triage",
     role: "NURSE",
     context: "TRIAGE_NURSE",  ← KEY
     dashboard: "TriageNurseDashboard",
     shift: { startTime: "07:00", endTime: "15:00", isTeamLead: true },
     permissions: { canAssignTeam: true, canPerformTriage: true, ... },
     isClocked: false
   }

5. Router renders: <TriageNurseDashboard />

6. Sarah sees:
   ┌─────────────────────────────────┐
   │ 🏥 TRIAGE QUEUE                 │
   │ Role: TRIAGE_NURSE              │
   │ Status: TEAM LEAD 👑            │
   │ Assigned Team: 2 people         │
   │                                 │
   │ 👥 Patients Waiting (0)         │
   │                                 │
   │ 👥 Team Assignment Controls     │
   │ [Assign Clinical Nurse]         │
   │ [Assign Physician]              │
   └─────────────────────────────────┘
```

### Bob (Clinical Nurse) Flow

```
1. Bob logs in
   Email: bob.martinez@stmercy.hospital
   Password: bob_secure_123

2. Redirected to /dashboard

3. DashboardRouter fetches config
   GET /api/user/dashboard-config
   
4. Backend returns:
   {
     name: "Bob Martinez",
     role: "NURSE",
     context: "CLINICAL_NURSE_ER",  ← KEY (different from Sarah!)
     dashboard: "ClinicalNurseDashboard_ER",
     shift: { startTime: "07:00", endTime: "15:00", isTeamLead: false },
     permissions: { canLogVitals: true, canRequestPhysician: true, ... },
     isClocked: false
   }

5. Router renders: <ClinicalNurseDashboard_ER />

6. Bob sees:
   ┌─────────────────────────────────┐
   │ ⚕️ CLINICAL MODE                │
   │ Role: CLINICAL_NURSE_ER         │
   │ Department: ED                  │
   │ Assigned Physician: Dr. Chen    │
   │                                 │
   │ 🛏️ Patients in Observation (0) │
   │                                 │
   │ [📊 Log Vitals]                 │
   │ [🩺 Request Physician Review]   │
   └─────────────────────────────────┘
```

### Dr. Chen (Physician) Flow

```
1. Dr. Chen logs in
   Email: dr.chen@stmercy.hospital
   Password: chen_secure_123

2. Redirected to /dashboard

3. Backend returns:
   {
     context: "PHYSICIAN_ER",  ← Different dashboard type!
     dashboard: "PhysicianClinicalDashboard_ER",
     ...
   }

4. Router renders: <PhysicianClinicalDashboard_ER />

5. Dr. Chen sees:
   ┌──────────────────────────────────────────┐
   │ 🩺 PHYSICIAN COMMAND CENTRE              │
   │                                          │
   │ 3-LANE VIEW:                             │
   │ 🔴 Active ER Cases (0)                   │
   │ 🟡 Observation/Awaiting Dispo (0)       │
   │ 🟣 Ward Escalations (0)                 │
   │                                          │
   │ Response Time Target: <5 minutes         │
   └──────────────────────────────────────────┘
```

---

## DECISION FOR USER

**Should we**:

1. **Build in this order** (recommended):
   - Backend: Enhance Shift model + create endpoint
   - Frontend: Create router + dashboard stubs
   - Integration: Test all 4 scenarios
   - Timeline: 4-5 hours

2. **Build just the backend first**:
   - Test endpoint with Postman
   - Verify contexts return correctly
   - Then build frontend
   - Timeline: 2 hours now + 2 hours later

3. **Build a simplified version**:
   - Use URL parameter `?context=TRIAGE_NURSE`
   - Hardcode contexts for testing
   - Refactor to proper backend config later
   - Timeline: 1 hour

**Recommendation**: Option 1 - build properly now, you'll use it throughout development

Ready to build? 🚀
