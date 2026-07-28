# DEEP REFACTORING DIAGNOSIS: Untangle Core Architecture
## Complete Schema Redesign for Staff/Schedule/Routes

**Status**: 🔴 CRITICAL DESIGN ISSUES IDENTIFIED  
**Date**: 20 April 2026  
**Goal**: Transform from hardcoded "nurse" terminology to unified "staff" architecture

---

## PART 1: THE ARCHITECTURAL MESS (What's Broken)

### 1.1 Collection Names Don't Match Reality

```
CURRENT STATE:
├─ Collection: "nurses"
│  └─ Contains: nurses (role: NURSE)
│  └─ Contains: physicians (role: PHYSICIAN) ← WRONG PLACE!
│  └─ Contains: specialists (role: SPECIALIST) ← WRONG PLACE!
│  └─ Model called: `Nurse` everywhere
│
├─ Collection: "managers"
│  └─ Contains: MANAGER, ADMIN, SHIFT_MANAGER
│  └─ Separate from "nurses" (confusing!)
│
├─ Collection: "nurseschedules"
│  └─ For nurses ONLY
│  └─ References nurseId as STRING (manual, fragile)
│
└─ Collection: "physicianschedules"
   └─ For physicians ONLY
   └─ References physicianId as STRING (manual, fragile)
   └─ Different structure than nurseSchedule

PROBLEM: Physicians are in "nurses" collection but have separate schedule table!
```

### 1.2 Routes Hardcode "nurse" for All Staff Types

```
CURRENT:
POST /api/nurses/clock-in (nurses ONLY - works for NURSE + PHYSICIAN)
POST /api/nurses/break-start (nurses ONLY)
POST /api/manager/staff/register (creates NURSE records)

LOGIC:
- Route says /nurses but PHYSICIANS use the same endpoint!
- Routes don't differentiate by staff type
- Manager registration endpoint assumes NURSE model

WHAT FRONTEND DOES:
- Doesn't look at role for routing clock-in
- All staff use /api/nurses/* endpoints
- Treats request handling identically for role=NURSE vs role=PHYSICIAN

PROBLEM: Semantic lie - routes say "nurse" but mean "staff"
```

### 1.3 Schedule Tables Are Siloed

```
CURRENT:
NurseSchedule:
├─ nurseId: STRING (manual FK!)
├─ maxBeds: 5
├─ assignedBeds: [String]
├─ Qualifications snapshot fields
└─ Indexed by: { nurseId, date }

PhysicianSchedule:
├─ physicianId: STRING (manual FK!)
├─ maxPatients: 20
├─ No bed concept
├─ specialty: String (required)
└─ Indexed by: { physicianId, date }

PROBLEM 1: No proper MongoDB references (STRING IDs!)
PROBLEM 2: Different schemas for same concept = different queries
PROBLEM 3: Frontend must know which endpoint to use

FRONTEND LOGIC:
if (role === NURSE) fetch(/api/schedules/nurses)
else if (role === PHYSICIAN) fetch(/api/schedules/physicians)
else if (role === SUPPORT) ??? (no endpoint!)
```

### 1.4 Manager is Separate from Staff

```
CURRENT:
Manager model: Separate collection
├─ managerID: String
├─ permissions object
├─ Can register staff (creates Nurse records)

Nurse model: Contains all staff
├─ No permissions object
├─ role: NURSE | PHYSICIAN | SPECIALIST
├─ createdBy: ObjectId (refs Manager)

PROBLEM: Managers not in Nurse collection but supervise from it
CONSISTENCY: Managers have different schema structure
ROUTING: /api/manager/... vs /api/nurses/...
```

### 1.5 Frontend Doesn't Use Role Consistently

```
CURRENT FLOW:
1. Login → JWT contains { id, email, role, department, ... }
2. Store in state: { id, email, name } ← ROLE THROWN AWAY!
3. On route /dashboard:
   - Check role ONLY for admin vs not admin
   - If NOT admin → NurseDashboard (EVERYONE!)
   - No role-based endpoint selection

PROBLEM: Role information travels but never gets USED
REDUNDANCY: getUserRole() re-decodes token instead of using stored role
CONFUSION: Frontend doesn't differentiate which staff type to route to

Example of frontend confusion:
```javascript
// Backend says PHYSICIAN uses /api/nurses/clock-in
// Frontend doesn't check role to know this
// Just blindly calls same endpoint for nurses, physicians, specialists
```
```

---

## PART 2: REFACTORING STRATEGY (Complete Redesign)

### 2.1 NEW MODEL: Unified `Staff` Model

Instead of: `Nurse` model with physicians inside  
New: `Staff` model that explicitly represents ALL staff types

```javascript
// backend/src/models/staff.model.js

const staffSchema = new Schema({
  hospitalId: ObjectId (refs Hospital) - MULTI-TENANT
  staffID: String (required, unique per hospital) - e.g., "STAFF-ED-001"
  
  // Identity
  firstName: String (required)
  lastName: String (required)
  email: String (required, unique per hospital)
  password: String (hashed)
  
  // Role & Permissions
  staffType: String (enum: STAFF, MANAGER, ADMIN) - HIGH LEVEL
  role: String (enum: NURSE, PHYSICIAN, SPECIALIST, SUPPORT) - SPECIFIC ROLE
  department: String (enum: ED, ICU, OR, OB, Pediatrics, Psychiatry, Medical-Surgical, General)
  specialty: String (e.g., "Emergency Medicine", "Cardiology") - For physicians
  
  // Permissions (applicable to all staff, structure varies by role)
  permissions: {
    canClockInOut: Boolean (default: true)
    canAccessTriage: Boolean (default: false for physicians)
    canManagePatients: Boolean (default: true for nurses)
    canViewAnalytics: Boolean (default: false)
    canManageStaff: Boolean (default: false) - MANAGER only
    canViewReports: Boolean (default: false) - MANAGER only
  }
  
  // Qualifications (applicable mostly to clinical staff)
  licenseNumber: String
  licenseType: String (enum: RN, PHC, SPECIALIST, MD)
  nqfLevel: Number (enum: 6, 7, 8, 9) - Medical credential level
  specializations: [String] (e.g., ["ICU Nurse", "Emergency Nurse"])
  credentials: [{
    name: String
    verifiedDate: Date
    expiryDate: Date
  }]
  
  // Capacity & Capabilities
  maxCapacity: Number - Role-specific (5 for nurses, 20 for physicians, etc.)
  canTriageRed: Boolean (default: false) - Nurses only
  canTriageYellow: Boolean (default: true)
  canTriageGreen: Boolean (default: true)
  
  // Status
  isActive: Boolean (default: true)
  lastLogin: Date
  lastClockIn: Date
  lastClockOut: Date
  
  // Audit
  createdBy: ObjectId (refs Staff) - Admin who created
  createdAt: Date
  updatedAt: Date
  deactivatedAt: Date (if removed)
  
  // Multi-version support
  legacyNurseId: String (for data migration) - ← TEMPORARY
});

// Indexes
staffSchema.index({ hospitalId: 1, email: 1 }, { unique: true })
staffSchema.index({ hospitalId: 1, role: 1 })
staffSchema.index({ hospitalId: 1, staffType: 1, isActive: 1 })
staffSchema.index({ hospitalId: 1, department: 1 })
```

**Collection name**: `staff` (not "nurses")  
**Imports**: Use this model everywhere instead of "Nurse"

---

### 2.2 UNIFIED `Schedule` Model (Replace Nurse + Physician Schedules)

Instead of: Separate NurseSchedule and PhysicianSchedule  
New: Single Schedule model that works for any staff type

```javascript
// backend/src/models/schedule.model.js

const scheduleSchema = new Schema({
  hospitalId: ObjectId (refs Hospital) - MULTI-TENANT
  staffId: ObjectId (refs Staff) ← PROPER MONGODB REFERENCE!
  staffType: String (enum: NURSE, PHYSICIAN, SPECIALIST, SUPPORT)
  
  // Schedule Details
  date: Date (required, index)
  shiftStart: String (HH:MM format)
  shiftEnd: String (HH:MM format)
  duration: Number (minutes, calculated)
  
  // Status
  status: String (enum: ON_DUTY, ON_CALL, OFF, VACATION, CALL_OUT, STANDBY)
  breakStart: String
  breakEnd: String
  
  // Capacity & Assignment (flexible for any role)
  maxCapacity: Number - Role-specific
  currentCapacity: Number - How many assigned
  assignedItems: [{
    itemId: ObjectId (patient for nurse, case for physician)
    assignedAt: Date
    priority: Number
  }]
  
  // Performance Tracking
  metrics: {
    avgResponseTime: Number
    tasksCompleted: Number
    // Role-specific metrics added as needed
  }
  
  // Qualifications snapshot (at time of scheduling)
  qualificationsRequired: [String]
  meetsCriteria: Boolean
  
  notes: String
  timestamps: { createdAt, updatedAt }
});

scheduleSchema.index({ hospitalId: 1, staffId: 1, date: 1 })
scheduleSchema.index({ date: 1, status: 1 })
scheduleSchema.index({ hospitalId: 1, staffType: 1, date: 1 })
```

**Replaces**: NurseSchedule + PhysicianSchedule (both deleted)  
**Migration**: Copy existing records, set staffType appropriately

---

### 2.3 UNIFIED Clock Record Model (Add staffType)

Keep NurseClockRecord but rename and extend:

```javascript
// backend/src/models/clockRecord.model.js (renamed from nurseClockRecord)

const clockRecordSchema = new Schema({
  hospitalId: ObjectId (refs Hospital)
  staffId: ObjectId (refs Staff) ← PROPER REFERENCE
  staffType: String (enum: NURSE, PHYSICIAN, SPECIALIST, SUPPORT)
  shiftId: ObjectId (refs Shift)
  scheduleId: ObjectId (refs Schedule) ← NEW: Link to schedule
  
  date: Date (required, index)
  clockInTime: Date (required)
  clockOutTime: Date
  location: String (ED, ICU, OR, etc.)
  
  status: String (enum: CLOCKED_IN, ON_BREAK, CLOCKED_OUT, ABSENT)
  
  breakStartTime: Date
  breakEndTime: Date
  totalBreakTime: Number
  
  totalShiftDuration: Number
  totalWorkTime: Number (shift - breaks)
  
  // Flags applicable to all staff
  flags: [String] (enum: EARLY_CLOCKIN, LATE_CLOCKIN, LATE_CLOCKOUT, EARLY_CLOCKOUT, UNAUTHORIZED_BREAK)
  
  // Approval workflow
  lateArrivalMinutes: Number
  lateApprovalStatus: String (enum: PENDING, APPROVED, REJECTED)
  breakApprovalStatus: String (enum: PENDING, APPROVED, REJECTED)
  breakReason: String
  approvedBy: ObjectId (refs Staff) ← Manager who approved
  approvalNotes: String
  
  metadata: {
    deviceType: String (mobile, tablet, desktop)
    ipAddress: String
  }
  timestamps: { createdAt, updatedAt }
});

clockRecordSchema.index({ hospitalId: 1, staffId: 1, date: 1 })
```

---

### 2.4 NEW ROUTING: Semantic `/api/staff/` Routes

Instead of: `/api/nurses/`, `/api/manager/`, `/api/physicians/`  
New: `/api/staff/{staffType}/{action}` or `/api/staff/{staffId}/{action}`

```javascript
// backend/src/routes/staff.route.js (NEW, unified)

// ===== STAFF OPERATIONS (All staff types) =====
POST /api/staff/register
  ├─ Protected: MANAGER, ADMIN
  ├─ Body: { firstName, lastName, email, password, role, department, specialty }
  ├─ Creates: Staff record with appropriate role
  └─ Returns: { success, staff: { id, email, role, staffType } }

POST /api/staff/login
  ├─ Public
  ├─ Body: { email, password }
  └─ Returns: { token, user: { id, email, role, permissions } }

GET /api/staff/list
  ├─ Protected: MANAGER, ADMIN
  ├─ Query: ?role=NURSE&department=ED&isActive=true&hospitalId=...
  └─ Returns: [{ id, name, email, role, department, ... }]

GET /api/staff/{staffId}
  ├─ Protected: Self or MANAGER/ADMIN
  └─ Returns: Complete staff record

// ===== CLOCK OPERATIONS (All staff types) =====
POST /api/staff/{staffId}/clock-in
  ├─ Protected: Staff with MANAGE own role + MANAGER/ADMIN
  ├─ Body: { shiftId, location, operationalContext }
  └─ Creates: ClockRecord entry
  └─ Returns: { success, record: { id, clockInTime, staffType } }

POST /api/staff/{staffId}/clock-out
  ├─ Protected: Own staff or MANAGER
  ├─ Body: {}
  └─ Returns: { success, record: { clockInTime, clockOutTime, totalDuration } }

POST /api/staff/{staffId}/break-start
  ├─ Protected: Own staff or MANAGER
  ├─ Body: { reason }
  └─ Returns: { success, status: ON_BREAK }

POST /api/staff/{staffId}/break-end
  ├─ Protected: Own staff or MANAGER
  ├─ Returns: { success, status: CLOCKED_IN }

GET /api/staff/{staffId}/today-record
  ├─ Protected: Own staff or MANAGER
  └─ Returns: { clockInTime, status, operationalContext }

GET /api/staff/{staffId}/clock-history
  ├─ Protected: Own staff or MANAGER
  ├─ Query: ?startDate=...&endDate=...
  └─ Returns: [{ date, clockInTime, clockOutTime, ... }]

// ===== SCHEDULE OPERATIONS (All staff types) =====
GET /api/staff/{staffType}/schedules?date=YYYY-MM-DD&status=ON_DUTY
  ├─ Protected: MANAGER/ADMIN
  ├─ staffType: NURSE | PHYSICIAN | SPECIALIST | SUPPORT
  └─ Returns: [{ staffId, date, status, maxCapacity, ... }]

POST /api/staff/schedule
  ├─ Protected: MANAGER/ADMIN
  ├─ Body: { staffId, date, status, shiftStart, shiftEnd, maxCapacity }
  └─ Creates: Schedule entry (works for any staff type)

GET /api/staff/{staffId}/schedule?date=YYYY-MM-DD
  ├─ Protected: Own staff or MANAGER
  └─ Returns: Staff's schedule for that date

// ===== MANAGEMENT OPERATIONS (MANAGER/ADMIN only) =====
GET /api/staff/management/dashboard
  ├─ Protected: MANAGER, ADMIN
  ├─ Query: ?department=ED&location=ED&date=...
  └─ Returns: Dashboard with all staff status + pending approvals

GET /api/staff/management/pending-approvals
  ├─ Protected: MANAGER, ADMIN
  └─ Returns: [{ staffId, staffName, reason, type: LATE_ARRIVAL|BREAK, ... }]

POST /api/staff/management/approve-bulk
  ├─ Protected: MANAGER, ADMIN
  ├─ Body: { approvalIds: [id1, id2], decision: APPROVED|REJECTED, notes }
  └─ Updates multiple clock record approvals

POST /api/staff/management/deactivate/{staffId}
  ├─ Protected: MANAGER, ADMIN
  └─ Soft-delete (sets isActive: false)
```

**Key improvements**:
- All routes start with `/api/staff/`
- Explicit staffId parameter for personal operations
- Separate management routes with clear RBAC
- Semantic endpoint naming

---

### 2.5 NEW CONTROLLER: Unified `StaffController.js`

```javascript
// backend/src/controllers/staff.controller.js

export const registerStaff = async (req, res) => {
  // Create Staff record (works for NURSE, PHYSICIAN, SPECIALIST, etc.)
  // Input: { firstName, lastName, email, password, role, department, specialty }
  // Output: { success, new staff record }
}

export const loginStaff = async (req, res) => {
  // Unified login (searches Staff collection, returns role-aware JWT)
  // Input: { email, password }
  // Output: { token, user with permissions based on role }
}

export const clockInStaff = async (req, res) => {
  // Works for any staffType (NURSE, PHYSICIAN, etc.)
  // Input: { staffId, shiftId, location, operationalContext }
  // Creates: ClockRecord with staffType preserved
}

export const clockOutStaff = async (req, res) => {
  // Works for any staffType
  // Automatically calculates totalDuration based on staffType's role
}

export const getTodayRecord = async (req, res) => {
  // Works for any staffType
  // Returns: Clock record + operational context from Schedule + Staff role
}

// Similar for: breakStart, breakEnd, getClockHistory, etc.
```

---

## PART 3: WHAT CHANGES AT FRONTEND

### 3.1 Single Endpoint Selection (Not Role-Based Branching)

```javascript
// BEFORE (hardcoded /nurses):
fetch('/api/nurses/clock-in', { ... })
fetch('/api/schedules/nurses', { ... })

// AFTER (staffId-based):
const staffId = loggedInUser.id;
fetch(`/api/staff/${staffId}/clock-in`, { ... })
fetch(`/api/staff/${staffId}/today-record`, { ... })

// Works for NURSES, PHYSICIANS, all staff types the same!
```

### 3.2 Dashboard Routing Uses operationalContext (Not Just Role)

```javascript
// BEFORE:
if (isManagerRole()) → AdminDashboard
else → NurseDashboard (everyone!)

// AFTER:
const context = clockRecord.operationalContext; // From today-record
const staffRole = user.role; // From JWT

if (staffRole === 'ADMIN') → AdminDashboard
else if (staffRole === 'PHYSICIAN' && context === 'PHYSICIAN_ER') → PhysicianDashboard_ER
else if (staffRole === 'NURSE' && context === 'TRIAGE_NURSE') → TriageNurseDashboard
else → FallbackNurseDashboard
```

### 3.3 No Role-Based Condition For Endpoints (Always Use StaffId)

```javascript
// BEFORE:
if (user.role === 'NURSE') {
  fetch('/api/schedules/nurses')
} else if (user.role === 'PHYSICIAN') {
  fetch('/api/schedules/physicians')
}

// AFTER:
// Always same endpoint:
fetch(`/api/staff/${staffId}/today-record`)
// Backend handles role logic internally!
```

---

## PART 4: DATA MIGRATION PLAN

### 4.1 Phase 1: Create New `Staff` Model (Parallel)

```bash
# Don't delete Nurse model yet
# Create Staff model alongside Nurse
# Test both exist simultaneously
```

### 4.2 Phase 2: Migrate Data

```javascript
// Script: backend/scripts/migrateNurseToStaff.js

// For each Nurse document:
// 1. Create Staff record with same data
// 2. Map role: NURSE/PHYSICIAN/SPECIALIST → role field
// 3. Map staffType based on role
// 4. Set createdBy from nurse.createdBy
// 5. Add legacyNurseId for cross-referencing

// For each NurseSchedule document:
// 1. Find corresponding staff (via nurseId → now use staffId ObjectId)
// 2. Create Schedule record
// 3. Set staffType: NURSE

// For each PhysicianSchedule:
// 1. Find corresponding staff with role=PHYSICIAN
// 2. Create Schedule record
// 3. Set staffType: PHYSICIAN
```

### 4.3 Phase 3: Update Routes Incrementally

```bash
# Old routes (deprecated but working):
POST /api/nurses/clock-in → Still works, redirects to /api/staff/{id}/clock-in

# New routes (active):
POST /api/staff/{staffId}/clock-in → Uses new Staff model
```

### 4.4 Phase 4: Frontend Migration

```bash
# Update component imports one at a time
# Old endpoints still work during transition
# New endpoints work in parallel
# Gradually switch frontend to new routes
```

### 4.5 Phase 5: Delete Old Models

```bash
# Only after all frontend/backend using new routes:
# Delete Nurse model
# Delete NurseSchedule model
# Delete PhysicianSchedule model
# Delete legacy routes
# Full cleanup
```

---

## PART 5: BENEFITS OF NEW ARCHITECTURE

### 5.1 Clarity

```
BEFORE:
- /api/nurses/ "for nurses" but used by physicians
- Nurse model contains physicians
- NurseSchedule and PhysicianSchedule separate

AFTER:
- /api/staff/{id}/ "for any staff"
- Staff model contains all staff with staffType
- Single Schedule model works for any role
- Clear semantic meaning
```

### 5.2 Extensibility

```
NEW ROLES (future):
+ Radiologist (SPECIALIST)
+ Administrative Assistant (SUPPORT)
+ Paramedic (SUPPORT)

Just add to role enum, no new models/tables needed!
Old architecture would need new NurseSchedule variant.
```

### 5.3 Consistency

```
NurseClockRecord → ClockRecord
├─ Works for nurses
├─ Works for physicians
├─ Works for any staff type

Same for Schedule, same for endpoints
Frontend doesn't care about internal role logic
```

### 5.4 References

```
BEFORE:
nurseId: String (manual, fragile)
physicianId: String (manual, fragile)
Must manually join in code

AFTER:
staffId: ObjectId (refs Staff) (proper MongoDB reference)
Can use populate()
Referential integrity!
```

---

## PART 6: COMPLETE REFACTORING CHECKLIST

```
☐ PHASE 1: Models
  ☐ Create Staff model (keep Nurse temporarily)
  ☐ Create unified Schedule model (keep Nurse/Physician schedules)
  ☐ Update ClockRecord to reference Staff + Schedule
  ☐ Test all models create/read successfully

☐ PHASE 2: Controllers
  ☐ Create StaffController.js (new unified controller)
  ☐ Keep old NurseController.js (deprecated but working)
  ☐ All functions work with Staff model
  ☐ Test all CRUD operations

☐ PHASE 3: Routes
  ☐ Create staff.route.js (new routes)
  ☐ Map old /api/nurses/* → new /api/staff/* (redirect)
  ☐ Test both old and new routes work

☐ PHASE 4: Frontend
  ☐ Store user role in state (don't throw away!)
  ☐ Update API calls to use /api/staff/{staffId}/
  ☐ No more role-based endpoint branching
  ☐ Test clock-in works for all roles

☐ PHASE 5: Dashboard Routing
  ☐ Fetch operationalContext from /api/staff/{staffId}/today-record
  ☐ Route based on operationalContext + role (not just role!)
  ☐ Create role-specific dashboard components
  ☐ Test each role sees correct dashboard

☐ PHASE 6: Data Migration
  ☐ Run migration script (Nurse → Staff)
  ☐ Verify all records migrated
  ☐ Spot-check permissions preserved

☐ PHASE 7: Cleanup
  ☐ Remove deprecated Nurse model
  ☐ Remove deprecated controllers
  ☐ Remove old routes
  ☐ Remove legacy code
```

---

## PART 7: TIMELINE & EFFORT

```
Phase 1 (1 hour): Models - Create Staff + Schedule models
Phase 2 (1.5 hours): Controllers - StaffController.js
Phase 3 (1.5 hours): Routes - New /api/staff/* endpoints
Phase 4 (2 hours): Frontend - Update API calls, store role
Phase 5 (2 hours): Dashboard Routing - operationalContext-based
Phase 6 (1 hour): Data Migration - Run migration script
Phase 7 (30 min): Cleanup - Delete old code

TOTAL: ~9 hours (can be done incrementally!)
```

---

## SUMMARY: Why This Matters

### Current Problem
- Routes lie: `/api/nurses/*` used by physicians
- Models confuse: Nurse has physicians, separate schedule tables
- Frontend confused: Must re-check role for endpoint selection
- References broken: String IDs instead of ObjectId
- No extensibility: Adding new roles requires new models

### New Solution
- Honest routes: `/api/staff/{staffId}/*` (works for any role)
- Clear models: Staff has staffType, unified Schedule
- Smart frontend: Use staffId parameter, endpoint is always same
- Proper references: ObjectId + populate()
- Extensible: Add roles to enum, everything works

### Result
- 🎯 Single source of truth for staff
- 🎯 Consistent API regardless of role
- 🎯 Easy to add new staff types
- 🎯 Eliminates confusion in codebase
- 🎯 Better for multi-role future (person can be NURSE and MANAGER)
