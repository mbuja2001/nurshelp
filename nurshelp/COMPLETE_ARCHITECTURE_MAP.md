# Complete Nurse Helpdesk Architecture Map

**Date:** April 20, 2026  
**Status:** Diagnostic snapshot of current system architecture  
**Purpose:** Understand all staff and schedule models, APIs, and controllers

---

## SECTION 1: STAFF-RELATED MODELS

### 1.1 NURSE Model (Multi-role staff)
**File:** [backend/src/models/nurse.model.js](backend/src/models/nurse.model.js)

**Purpose:** Master staff record for nurses, physicians, and specialists. Uses single model with role-based differentiation.

**Key Schema Fields:**
```javascript
hospitalId: ObjectId (refs Hospital) - MULTI-TENANT KEY
name: String (required)
surname: String (required)
email: String (required, unique per hospital)
password: String (hashed by pre-save hook)
role: String (enum: NURSE, PHYSICIAN, SPECIALIST) - CRITICAL FIELD
department: String (enum: ED, ICU, OR, OB, Medical-Surgical, Pediatrics, Psychiatry, General)
specialty: String (e.g., "Emergency Medicine", "Cardiology")
licenseNumber: String
isActive: Boolean (default: true)
createdBy: ObjectId (refs Manager)
dateCreated: Date
timestamps: { createdAt, updatedAt }
```

**Indexes:**
```javascript
{ hospitalId: 1, email: 1 } - UNIQUE (allows same email across hospitals)
{ hospitalId: 1, isActive: 1 }
{ hospitalId: 1, department: 1 }
{ hospitalId: 1, role: 1 }
```

**Collection Name:** `nurses` (in MongoDB)

**Key Insight:** 🚨 **Single Nurse model handles all staff types (NURSE, PHYSICIAN, SPECIALIST).** There's NO separate Physician model. Role is differentiated via `role` field.

---

### 1.2 NURSE PROFILE Model
**File:** [backend/src/models/nurseProfile.model.js](backend/src/models/nurseProfile.model.js)

**Purpose:** Detailed qualification and capability profile for nurses. Tracks certifications, specializations, and triage permissions.

**Key Schema Fields:**
```javascript
nurseId: String (required, unique)
name: String
surname: String
licenseNumber: String (required, unique)
licenseType: String (enum: RN, PHC, SPECIALIST)
nqfLevel: Number (enum: 6, 7, 8) // 6=Diploma, 7=Degree, 8=Post-Basic
specializations: [String] (enum: 'ICU Nurse', 'Emergency Nurse', 'Midwife', 'Trauma Nurse', etc.)
primarySpecialty: String
credentials: [{
  name: String
  verifiedDate: Date
  expiryDate: Date
  isActive: Boolean
}]
maxSatsLevel: String (enum: green, yellow, red)
canTriageRed: Boolean
canTriageYellow: Boolean
canTriageGreen: Boolean
maxBedsPerShift: Number (default: 4)
experienceYears: Number
```

**Key Insight:** 📋 **Separate table for detailed qualifications.** Links to NurseSchedule but NOT directly to Nurse model—uses `nurseId` string, not ObjectId.

---

### 1.3 MANAGER Model
**File:** [backend/src/models/manager.model.js](backend/src/models/manager.model.js)

**Purpose:** Administrative staff who manage hospitals, create shifts, register staff, approve requests.

**Key Schema Fields:**
```javascript
hospitalId: ObjectId (refs Hospital) - Multi-tenant
managerID: String (unique)
name: String
surname: String
email: String (required, unique per hospital)
password: String (hashed)
phone: String
department: String (enum: ED, ICU, OR, OB, Medical-Surgical, Pediatrics, Psychiatry, All)
role: String (enum: MANAGER, SHIFT_MANAGER, DEPARTMENT_MANAGER, ADMIN)
permissions: {
  canCreateShifts: Boolean
  canEditShifts: Boolean
  canDeleteShifts: Boolean
  canAssignStaff: Boolean
  canApproveLeaves: Boolean
  canManageRoles: Boolean
  canViewReports: Boolean
}
isActive: Boolean
lastLogin: Date
timestamps: { createdAt, updatedAt }
```

**Indexes:**
```javascript
{ hospitalId: 1, isActive: 1 }
{ hospitalId: 1, email: 1 } - UNIQUE
```

**Collection Name:** `managers`

**Key Relationships:**
- Creates staff: Nurse model has `createdBy: ObjectId (refs Manager)`
- Hospital: references Hospital._id

---

## SECTION 2: SCHEDULE-RELATED MODELS

### 2.1 NURSE SCHEDULE Model
**File:** [backend/src/models/nurseSchedule.model.js](backend/src/models/nurseSchedule.model.js)

**Purpose:** Daily schedule for nurses. What date/time they work, beds assigned, capabilities on that day.

**Key Schema Fields:**
```javascript
nurseId: String (required, unique) // String ID, not MongoDB ObjectId!
nurseProfileId: ObjectId (refs NurseProfile) // Qualified info
name: String
surname: String
email: String
licenseNumber: String

// Daily Schedule
date: Date (required) // The day scheduled
status: String (enum: ON_DUTY, ON_CALL, OFF, VACATION, CALL_OUT) - DEFAULT: OFF
shiftStart: String (e.g., "07:00")
shiftEnd: String (e.g., "19:00")
breakStart: String
breakEnd: String

// Bed Assignment
maxBeds: Number (default: 5)
assignedBeds: [String] // e.g., ["ER-01", "ER-02"]
currentBedCount: Number

// Professional Qualifications (Snapshot)
licenseType: String (enum: RN, PHC, SPECIALIST)
nqfLevel: Number (enum: 6, 7, 8)
specializations: [String]
canTriageRed: Boolean (default: false)
canTriageYellow: Boolean (default: true)
canTriageGreen: Boolean (default: true)

// Performance Metrics
averageAssessmentTime: Number (minutes)
triageAccuracy: Number (%)
averageChargeTime: Number (minutes)
availabilityMask: [Number] // 24-hour array
certifications: [String]
notes: String
timestamps: { createdAt, updatedAt }
```

**Indexes:**
```javascript
{ date: 1, status: 1 }
{ nurseId: 1, date: 1 }
{ date: 1 }
```

**Key Insight:** ⚠️ **nurseId is a STRING, not ObjectId reference. Manual relationship.** Uses both `nurseId` (string) and `nurseProfileId` (ObjectId ref).

---

### 2.2 PHYSICIAN SCHEDULE Model
**File:** [backend/src/models/physicianSchedule.model.js](backend/src/models/physicianSchedule.model.js)

**Purpose:** Daily schedule for physicians. Similar to NurseSchedule but tailored for doctors with different capacity metrics.

**Key Schema Fields:**
```javascript
physicianId: String (required, unique) // String ID like NurseSchedule
name: String
surname: String
email: String
specialty: String (required) // e.g., "Cardiology", "ER", "Orthopedics"

// Daily Schedule
date: Date (required)
status: String (enum: ON_DUTY, ON_CALL, OFF, VACATION) // No CALL_OUT
shiftStart: String
shiftEnd: String
breakStart: String
breakEnd: String

// Workload Capacity
maxPatients: Number (default: 20) // vs nurses' maxBeds: 5
currentPatientCount: Number
availabilityMask: [Number] // 24-hour availability

// Performance Metrics
averageTriageTime: Number (minutes)
consultationRate: Number (%)
averageConfirmationTime: Number (minutes)
workloadScore: Number

notes: String
timestamps: { createdAt, updatedAt }
```

**Indexes:**
```javascript
{ date: 1, status: 1, specialty: 1 }
{ physicianId: 1, date: 1 }
{ date: 1 }
```

**Differences from NurseSchedule:**
| Field | Nurse | Physician |
|-------|-------|-----------|
| capacity | `maxBeds: 5` | `maxPatients: 20` |
| status options | ON_DUTY, ON_CALL, OFF, VACATION, CALL_OUT | ON_DUTY, ON_CALL, OFF, VACATION |
| qualifications | licenseType, nqfLevel, specializations | specialty only |
| metrics | averageAssessmentTime, triageAccuracy | averageTriageTime, consultationRate |

**Key Insight:** 🔴 **No "Physician" document collection.** Physicians stored as Nurse records with `role: PHYSICIAN`. PhysicianSchedule is separate schedule table. **Doctors are in Nurse model with role=PHYSICIAN.**

---

### 2.3 SHIFT Model
**File:** [backend/src/models/shift.model.js](backend/src/models/shift.model.js)

**Purpose:** Generic shift template. Managers create shifts, then assign staff (nurses/physicians/managers) to them.

**Key Schema Fields:**
```javascript
hospitalId: ObjectId (refs Hospital) - Multi-tenant
shiftID: String (generated: "SHIFT-{timestamp}-{random}")
shiftName: String
department: String (enum: ED, ICU, OR, OB, Medical-Surgical, Pediatrics, Psychiatry)
shiftStart: String (HH:MM format)
shiftEnd: String (HH:MM format)
duration: Number (in hours, calculated)

date: Date
dayOfWeek: String (enum: Monday-Sunday)
recurrence: String (enum: ONCE, DAILY, WEEKLY, BIWEEKLY, MONTHLY) - default: ONCE
recurrenceEndDate: Date (only if recurrence != ONCE)

maxStaffRequired: Number (how many staff needed for this shift)
assignedStaff: [{
  staffID: ObjectId (refs Nurse) // Can be nurse OR physician
  type: String (enum: MANAGER, NURSE, PHYSICIAN, SPECIALIST, SUPPORT)
  name: String
  surname: String
  specialty: String
  status: String (enum: ASSIGNED, CONFIRMED, ABSENT, PENDING) - default: ASSIGNED
}]

status: String // Track shift publication state
createdBy: ObjectId (refs Manager)
notes: String
timestamps: { createdAt, updatedAt }
```

**Indexes:**
```javascript
{ hospitalId: 1, index: true }
{ shiftID: 1, index: true }
{ department: 1, index: true }
{ date: 1, index: true }
```

**Key Insight:** 📅 **Shift is recurrence-aware.** Can repeat daily/weekly/monthly. Staff array stores type of staff (NURSE/PHYSICIAN/MANAGER) and their current status.

---

### 2.4 NURSE ASSIGNMENT Model
**File:** [backend/src/models/nurseAssignment.model.js](backend/src/models/nurseAssignment.model.js)

**Purpose:** Track individual nurse's patients during a shift. Maps patients to nurse + shift metadata.

**Key Schema Fields:**
```javascript
assignmentId: String (required, unique, index)
nurseId: String (required, index)
nurseProfile: {
  name: String
  licenseType: String
  specialty: String
  maxESILevel: Number
}

shiftDate: Date (required, index)
shiftStartTime: Date
shiftEndTime: Date

assignedPatients: [{
  encounterId: String
  bedId: String
  patientName: String
  esiLevel: Number
  acuityScore: Number
  assignmentTime: Date
  lastVitalCheckTime: Date
  tasksCompleted: Number
  tasksPending: Number
}]

maxPatients: Number
currentPatientCount: Number
workloadScore: Number (0-100)

shiftMetrics: {
  averageResponseTime: Number (minutes)
  alertsAcknowledged: Number
  mistakesLogged: Number
  patientSatisfactionScore: Number
}

clockInTime: Date
clockOutTime: Date
status: String (enum: WAITING_TO_START, ACTIVE, ON_BREAK, WRAPPING_UP, CLOCKED_OUT)
breakStartTime: Date
breakEndTime: Date
timestamps: { createdAt, updatedAt }
```

**Indexes:**
```javascript
{ nurseId: 1, shiftDate: 1 }
{ shiftDate: 1, status: 1 }
{ 'assignedPatients.encounterId': 1 }
```

**Key Insight:** 👥 **Manages patient assignment for individual shifts.** Different from NurseSchedule (which is daily schedule). NurseAssignment is per-shift patient tracking.

---

### 2.5 NURSE CLOCK RECORD Model
**File:** [backend/src/models/nurseClockRecord.model.js](backend/src/models/nurseClockRecord.model.js)

**Purpose:** Tracks clock-in/out events during a shift for time tracking and attendance.

**Key Schema Fields:**
```javascript
nurseId: ObjectId (refs Nurse, required, index)
shiftId: ObjectId (refs Shift, required)
date: Date (required, index) // Shift date
clockInTime: Date (required)
clockOutTime: Date (default: null) // Null while active

breakStartTime: Date
breakEndTime: Date

status: String (enum: CLOCKED_IN, ON_BREAK, CLOCKED_OUT, ABSENT)
location: String (enum: ED, ICU, OR, MATERNITY, NICU, PEDIATRICS, PSYCH, GENERAL)

// Calculated metrics
totalShiftDuration: Number (milliseconds)
totalBreakTime: Number (milliseconds)
totalWorkTime: Number (shift time - breaks)

metadata: {
  deviceType: String // mobile, tablet, desktop
  // ... security/device tracking info
}

// Approval tracking
lateArrivalMinutes: Number
lateArrivalApprovalStatus: String (enum: PENDING, APPROVED, REJECTED)
breakApprovalStatus: String (enum: PENDING, APPROVED, REJECTED)
breakReason: String

timestamps: { createdAt, updatedAt }
```

**Key Insight:** ⏱️ **Records actual time punches.** Used by auto clock-out job (runs every 30 min) to auto punch out staff after shift ends.

---

## SECTION 3: CONTROLLERS & BUSINESS LOGIC

### 3.1 NURSE Controller
**File:** [backend/src/controllers/nurse.controller.js](backend/src/controllers/nurse.controller.js)

**What it handles:**
- Staff login (nurse & physician)
- Clock in/out operations
- Break start/end
- Clock status queries
- Attendance history
- Shift queries

**Key Functions:**
```javascript
loginNurse(req, res)           // POST /api/nurses/login - Public
clockIn(req, res)              // POST /api/nurses/clock-in - Protected (NURSE, PHYSICIAN)
clockOut(req, res)             // POST /api/nurses/clock-out - Protected
breakStart(req, res)           // POST /api/nurses/break-start - Protected
breakEnd(req, res)             // POST /api/nurses/break-end - Protected
getClockStatus(req, res)       // GET /api/nurses/clockstatus - Protected
getTodayRecord(req, res)       // GET /api/nurses/today-record - Protected
getClockHistory(req, res)      // GET /api/nurses/clock-history - Protected
getAvailableShifts(req, res)   // GET /api/nurses/available-shifts - Protected
autoClockOutExpiredShifts()    // Scheduled job (30 min interval)
```

**Models Used:**
- Nurse (authentication)
- NurseClockRecord (time tracking)
- Shift (shift validation)
- NurseSchedule (optional)

---

### 3.2 SCHEDULE Controller
**File:** [backend/src/controllers/schedule.controller.js](backend/src/controllers/schedule.controller.js)

**What it handles:**
- Get available physicians for a date
- Get physician's schedule
- Create/update physician schedules
- Import physician schedules from CSV
- Get available nurses for a date
- Get nurse's schedule
- Create/update nurse schedules
- Import nurse schedules from CSV
- Bed assignment operations

**Key Functions:**
```javascript
// Physician
getAvailablePhysicians(req, res)        // GET /api/schedules/physicians?date,specialty,status
getPhysicianSchedule(req, res)          // GET /api/schedules/physicians/me?date
upsertPhysicianSchedule(req, res)       // POST /api/schedules/physicians
importPhysicianSchedulesCSV(req, res)   // POST /api/schedules/physicians/import

// Nurse
getAvailableNurses(req, res)            // GET /api/schedules/nurses?date,status,esiLevel
getNurseSchedule(req, res)              // GET /api/schedules/nurses/me?date
upsertNurseSchedule(req, res)           // POST /api/schedules/nurses
importNurseSchedulesCSV(req, res)       // POST /api/schedules/nurses/import

// Beds
getAvailableBeds(req, res)              // GET /api/schedules/beds?department,bedType,acuityLevel
getAllBeds(req, res)                    // GET /api/schedules/beds/all?department
assignBed(req, res)                     // POST /api/schedules/beds/assign
releaseBed(req, res)                    // (implied)
initializeBeds(req, res)                // (implied)
```

**Models Used:**
- PhysicianSchedule
- NurseSchedule
- BedSchedule

**Note:** ⚠️ **Date parsing utility** to avoid timezone issues:
```javascript
parseDateString(dateStr) // YYYY-MM-DD → UTC Date object
```

---

### 3.3 SHIFT Controller
**File:** [backend/src/controllers/shift.controller.js](backend/src/controllers/shift.controller.js)

**What it handles:**
- Create/update/delete shifts
- Get shifts by date range
- Publish shifts
- Assign staff to shifts
- Remove staff from shifts
- Query available staff

**Key Functions:**
```javascript
createShift(req, res)              // POST /api/shifts/create - Protected (MANAGER, ADMIN)
getShifts(req, res)                // GET /api/shifts/list - Protected
getShiftsForDay(req, res)          // GET /api/shifts/day?date - Protected
getShiftsForWeek(req, res)         // GET /api/shifts/week - Protected
getShiftsForMonth(req, res)        // GET /api/shifts/month - Protected
updateShift(req, res)              // PUT /api/shifts/:shiftID - Protected
deleteShift(req, res)              // DELETE /api/shifts/:shiftID - Protected
publishShift(req, res)             // POST /api/shifts/:shiftID/publish - Protected
assignStaffToShift(req, res)       // POST /api/shifts/:shiftID/assign-staff - Protected
removeStaffFromShift(req, res)     // (implied)
getAvailableStaff(req, res)        // Get staff available for period
getAssignedShiftsForUser(req, res) // GET /api/shifts/assigned - Staff view
getTodayAssignedShift(req, res)    // GET /api/shifts/assigned/today - Staff view
```

**Models Used:**
- Shift
- Nurse (for staff availability)

**Key Logic:**
- Generates shiftID: `SHIFT-{timestamp}-{random}`
- Calculates duration from start/end times
- Filters by hospital (multi-tenant)

---

### 3.4 MANAGER Controller
**File:** [backend/src/controllers/manager.controller.js](backend/src/controllers/manager.controller.js)

**What it handles:**
- Register new staff (nurses/physicians)
- Get staff list
- Update staff
- Delete/deactivate staff
- View staff status dashboard
- View pending approvals
- Approve/reject break and late arrival requests

**Key Functions:**
```javascript
// Inline in route:
registerStaff(req, res)            // POST /api/manager/staff/register
getStaffList(req, res)             // GET /api/manager/staff/list

getStaffStatus(req, res)           // GET /api/manager/staff-status - Dashboard
getPendingApprovals(req, res)      // GET /api/manager/pending-approvals
bulkApproveApprovals(req, res)     // POST /api/manager/bulk-approvals
```

**Models Used:**
- Nurse (for staff CRUD)
- Manager (authorization)
- NurseClockRecord (for status dashboard)

**Key Insight:** 🔑 Staff registration creates records in **Nurse model** with password hashing via pre-save hook. Multi-tenant validation ensures email uniqueness per hospital.

---

### 3.5 AUTH Controller
**File:** [backend/src/controllers/auth.controller.js](backend/src/controllers/auth.controller.js)

**What it handles:**
- Unified login for all user types (managers and staff)
- JWT token generation
- Role extraction for RBAC

**Key Functions:**
```javascript
login(req, res) // POST /api/auth/login - Public
```

**Login Logic:**
1. Try to find user in Manager collection first
2. If not found, try Nurse collection
3. Compare password (bcrypt)
4. Generate JWT with: id, email, name, role, department, hospitalId, hospitalName
5. Return token + user object

**Models Used:**
- Manager
- Nurse

**Full List of Response Fields:**
```javascript
{
  success: true,
  message: "Login successful",
  token: "eyJhbGciOi...",
  user: {
    id: "...",
    name: "John",
    email: "john@hospital.com",
    role: "NURSE" | "PHYSICIAN" | "MANAGER" | "ADMIN",
    department: "ED",
    hospitalId: "...",
    hospitalName: "St. Mary's Hospital",
    surname: "Smith" // (if staff)
    permissions: {...} // (if manager)
  }
}
```

---

### 3.6 PHYSICIAN Controller
**File:** [backend/src/controllers/physician.controller.js](backend/src/controllers/physician.controller.js)

**What it handles:**
- Get physician recommendations for a specific encounter
- Calls Python triage service for ranking

**Key Functions:**
```javascript
getRecommendations(req, res) // GET /api/physicians/recommendations?encounterId
```

**How it works:**
1. Looks up Encounter record
2. Extracts triage data
3. Normalizes stored physician lists from triage
4. If no candidates, calls Python service at `http://localhost:5000/triage`
5. Ranks and returns physicians with scores

**Models Used:**
- Encounter
- (No direct Nurse/Physician model queries for ranking)

---

## SECTION 4: ROUTES ARCHITECTURE

### 4.1 Authentication Routes
**File:** [backend/src/routes/auth.route.js](backend/src/routes/auth.route.js)

```
POST /api/auth/login
  ├─ Public (no auth required)
  ├─ Unified for managers and staff
  └─ Response: { token, user with role }
```

---

### 4.2 Hospital Routes
**File:** [backend/src/routes/hospital.route.js](backend/src/routes/hospital.route.js)

```
POST /api/hospital/register
POST /api/hospital/login (deprecated, use /api/auth/login)
```

---

### 4.3 Manager Routes
**File:** [backend/src/routes/manager.route.js](backend/src/routes/manager.route.js)

```
POST /api/manager/staff/register
  ├─ Protected: MANAGER, ADMIN
  ├─ Middleware: protect, authorize, restrictByDepartment, auditLog
  └─ Creates: Nurse record with role (NURSE|PHYSICIAN|SPECIALIST)

GET /api/manager/staff/list
  ├─ Protected: MANAGER, ADMIN
  ├─ Query: ?role=NURSE&department=ED&isActive=true&search=john
  └─ Returns: staff array filtered by hospital and department

GET /api/manager/staff-status
  ├─ Protected: MANAGER, ADMIN
  ├─ Query: ?location=ED&status=CLOCKED_IN&showPendingApprovals=true
  └─ Returns: Dashboard with clock records

GET /api/manager/pending-approvals
  ├─ Protected: MANAGER, ADMIN
  └─ Returns: Pending break/late arrival requests

POST /api/manager/bulk-approvals
  ├─ Protected: MANAGER, ADMIN
  └─ Approves multiple requests in batch
```

---

### 4.4 Nurse/Staff Clock Routes
**File:** [backend/src/routes/nurse.route.js](backend/src/routes/nurse.route.js)

```
POST /api/nurses/login
  ├─ Public
  └─ Body: { email, password }

POST /api/nurses/clock-in
  ├─ Protected: NURSE, PHYSICIAN
  ├─ Middleware: protect, authorize, auditLog
  └─ Body: { location, shiftID, notes }

POST /api/nurses/clock-out
  ├─ Protected: NURSE, PHYSICIAN
  └─ Body: { notes }

POST /api/nurses/break-start
  ├─ Protected: NURSE, PHYSICIAN
  └─ Body: { reason }

POST /api/nurses/break-end
  ├─ Protected: NURSE, PHYSICIAN
  └─ Body: {}

GET /api/nurses/today-record
  ├─ Protected: NURSE, PHYSICIAN
  └─ Returns: Today's clock record

GET /api/nurses/clockstatus
  ├─ Protected: NURSE, PHYSICIAN
  └─ Returns: Current clock status

GET /api/nurses/available-shifts
  ├─ Protected: NURSE, PHYSICIAN
  └─ Returns: Shifts available for assignment

GET /api/nurses/attendance-history
  └─ Returns: Past attendance records

GET /api/nurses/clock-history
  └─ Returns: Clock-in/out history
```

---

### 4.5 Schedule Routes
**File:** [backend/src/routes/schedule.route.js](backend/src/routes/schedule.route.js)

```
// PHYSICIAN SCHEDULES
GET /api/schedules/physicians?date=YYYY-MM-DD&specialty=ED&status=ON_DUTY
POST /api/schedules/physicians
  └─ Body: { physicianId, date, status, shiftStart, shiftEnd, maxPatients }
POST /api/schedules/physicians/import
  └─ Upload CSV file

// NURSE SCHEDULES
GET /api/schedules/nurses?date=YYYY-MM-DD&status=ON_DUTY&esiLevel=2
POST /api/schedules/nurses
  └─ Body: { nurseId, date, status, shiftStart, shiftEnd, maxBeds }
POST /api/schedules/nurses/import
  └─ Upload CSV file

// BED AVAILABILITY
GET /api/schedules/beds?department=ED&bedType=standard&acuityLevel=high
GET /api/schedules/beds/all?department=ED
POST /api/schedules/beds/assign
POST /api/schedules/beds/release (implied)
```

---

### 4.6 Shift Routes
**File:** [backend/src/routes/shift.route.js](backend/src/routes/shift.route.js)

```
POST /api/shifts/create
  ├─ Protected: MANAGER, ADMIN
  ├─ Middleware: protect, authorize, restrictByDepartment, auditLog(SHIFT_CREATED)
  └─ Body: { shiftName, department, shiftStart, shiftEnd, date, maxStaffRequired, assignedStaff }

GET /api/shifts/list
  ├─ Protected: MANAGER, ADMIN
  └─ Query: ?department=ED&startDate=...&endDate=...&status=PUBLISHED

GET /api/shifts/day?date=YYYY-MM-DD
GET /api/shifts/week?startDate=YYYY-MM-DD
GET /api/shifts/month?month=MM&year=YYYY

PUT /api/shifts/:shiftID
  └─ Update shift

DELETE /api/shifts/:shiftID
  └─ Delete shift

POST /api/shifts/:shiftID/publish
  └─ Mark shift as published

POST /api/shifts/:shiftID/assign-staff
  ├─ Protected: MANAGER, ADMIN
  └─ Body: { staffID, type (NURSE|PHYSICIAN|MANAGER), status }

// Staff view
GET /api/shifts/assigned
  ├─ Protected: NURSE, PHYSICIAN
  └─ Returns: All shifts assigned to logged-in staff

GET /api/shifts/assigned/today
  ├─ Protected: NURSE, PHYSICIAN
  └─ Returns: Today's assigned shift
```

---

### 4.7 Other Routes (Brief)
```
POST /api/encounters/...
POST /api/triage/...
GET  /api/physicians/recommendations?encounterId=...
POST /api/beds/...
GET  /api/reports/nurse-confirmed/...
```

---

## SECTION 5: MIDDLEWARE & AUTHORIZATION

### 5.1 Authentication Middleware
**File:** [backend/src/middleware/authmiddleware.js](backend/src/middleware/authmiddleware.js)

**Key Middleware:**
```javascript
protect(req, res, next) // Verifies JWT token
  ├─ Extracts token from Authorization header
  ├─ Verifies with JWT_SECRET
  ├─ Populates req.user with decoded token data
  └─ Sets req.hospitalId
```

---

### 5.2 RBAC Middleware
**File:** [backend/src/middleware/rbac.js](backend/src/middleware/rbac.js)

**Key Middleware:**
```javascript
authorize(...roles)(req, res, next) // Checks user role
  └─ Example: authorize('MANAGER', 'ADMIN')
  
restrictByDepartment()(req, res, next) // Limits manager view by department
  ├─ Managers only see their department (req.departmentFilter)
  └─ Admins see all departments
  
auditLog(action)(req, res, next) // Logs sensitive operations
  └─ Records who did what when (action: STAFF_REGISTERED, SHIFT_CREATED, etc.)
```

---

## SECTION 6: FRONTEND API CALLS

### 6.1 Login API Calls

**File:** [frontend/src/Dashboard/Dashboard.jsx](frontend/src/Dashboard/Dashboard.jsx)

```javascript
POST /api/auth/login
├─ Body: { email, password }
├─ Response: { token, user }
└─ Stored in localStorage: 'token'
```

---

### 6.2 Staff Clock Operations

**File:** [frontend/src/Dashboard/Pages/ClockInOutBoard.jsx](frontend/src/Dashboard/Pages/ClockInOutBoard.jsx)

```javascript
GET /api/nurses/today-record
  └─ Fetches today's clock record to enforce triage permissions

POST /api/nurses/clock-in
  ├─ Body: { shiftID, location, notes }
  └─ Called when staff clicks "Clock In"

POST /api/nurses/clock-out
  ├─ Body: { notes }
  └─ Called at end of shift

POST /api/nurses/break-start
  ├─ Body: { reason }
  └─ When staff takes break

POST /api/nurses/break-end
  └─ When break ends

GET /api/shifts/assigned
  └─ Lists all assigned shifts for staff selection

GET /api/shifts/assigned/today
  └─ Gets today's shift

GET /api/nurses/attendance-history
  └─ Shows attendance records (refreshed every 30 sec)
```

---

### 6.3 Manager Dashboard API Calls

**File:** [frontend/src/Dashboard/Pages/StaffScheduleManager.jsx](frontend/src/Dashboard/Pages/StaffScheduleManager.jsx)

```javascript
GET /api/manager/staff/list?role=NURSE&department=ED
  └─ Lists all staff for selection

GET /api/shifts/day?date=YYYY-MM-DD
  └─ Gets today's shifts

POST /api/shifts/create
  ├─ Body: { shiftName, department, shiftStart, shiftEnd, maxStaffRequired, assignedStaff }
  └─ Creates new shift

POST /api/shifts/:shiftId/assign-staff
  ├─ Body: { staffID, type }
  └─ Assigns staff to shift
```

---

### 6.4 Bed Management API Calls

**File:** [frontend/src/Dashboard/Pages/BedsAllocation.jsx](frontend/src/Dashboard/Pages/BedsAllocation.jsx)

```javascript
GET /api/beds/wards
  └─ Fetch all wards

GET /api/beds/ward/:wardId
  └─ Fetch beds in specific ward (refreshed every 10 sec)

POST /api/beds
  └─ Create new bed

PUT /api/beds/:bedId
  └─ Update bed

POST /api/beds/:bedId/assign
  └─ Assign bed to patient/nurse

POST /api/beds/:bedId/unassign
  └─ Release bed

DELETE /api/beds/:bedId
  └─ Delete bed
```

---

## SECTION 7: KEY ARCHITECTURAL PATTERNS & ANOMALIES

### 🚨 Critical Design Decisions

#### 1. **Single Nurse Model for All Staff**
- **Decision:** One `Nurse` model with `role` field instead of separate Nurse/Physician/Specialist models
- **Why:** Simpler authentication, unified login endpoint
- **Implication:** Frontend must check `user.role` to determine what features to show
- **Risk:** Confusing because collection is called "nurses" but contains physicians too

#### 2. **Separate Schedule Tables**
- **Decision:** PhysicianSchedule and NurseSchedule are separate collections
- **Why:** Different capacity metrics (maxPatients vs maxBeds)
- **Implication:** Frontend needs to call different endpoints based on role
- **Code Pattern:**
  ```javascript
  if (user.role === 'NURSE') {
    fetch('/api/schedules/nurses')
  } else if (user.role === 'PHYSICIAN') {
    fetch('/api/schedules/physicians')
  }
  ```

#### 3. **String nurseId vs ObjectId**
- **Decision:** NurseSchedule and PhysicianSchedule use STRING `nurseId` and `physicianId`, not MongoDB ObjectId
- **Why:** Manual relationship management, easier CSV import
- **Risk:** Orphaned records if staff are deleted; fragility
- **Query Pattern:**
  ```javascript
  // Can't use populate() with string IDs
  // Must manually join in code
  ```

#### 4. **Multi-tenant via hospitalId**
- **Decision:** Every major model has `hospitalId: ObjectId` field
- **Why:** Support multiple hospital instances in single codebase
- **Implication:** ALL queries must filter by `hospitalId` (enforced by middleware)
- **Security:** `restrictByDepartment()` middleware filters by hospital + department

#### 5. **Manual Approval Workflow**
- **Decision:** NurseClockRecord has `lateArrivalApprovalStatus` and `breakApprovalStatus` fields
- **Why:** Managers can override clock-in/out and break times
- **Workflow:**
  1. Staff clocks in/takes break
  2. System flags if late/unusual
  3. Manager reviews in `/api/manager/pending-approvals`
  4. Manager bulk approves in `/api/manager/bulk-approvals`

#### 6. **Auto Clock-Out Job**
- **Decision:** Scheduled job every 30 minutes auto-clocks out staff after shift ends
- **Why:** Prevent stuck records if staff forget to clock out
- **Code:** Runs in server.js scheduled job, updates NurseClockRecord with `AUTO_CLOCKOUT` flag

---

### ⚠️ Confusing Naming & Terminology

| Term | Location | Actual Meaning |
|------|----------|----------------|
| **Nurse model** | `backend/src/models/nurse.model.js` | All staff (nurses, physicians, specialists) |
| **Nurse controller** | `backend/src/controllers/nurse.controller.js` | Staff authentication & clocking (works for physicians too) |
| **NurseSchedule collection** | MongoDB | Daily schedule for nurses ONLY |
| **PhysicianSchedule collection** | MongoDB | Daily schedule for physicians ONLY |
| **Physician** | Code | Really means `role: PHYSICIAN` in Nurse model |
| **nurseId** (string) | NurseSchedule schema | Manual ID, not MongoDB reference |

---

### 🔗 Key Relationships Diagram

```
Hospital (1)
  ├─ many Nurse (role: NURSE|PHYSICIAN|SPECIALIST)
  │   └─ many NurseClockRecord (time tracking)
  ├─ many Manager
  ├─ many Shift
  │   └─ many assignedStaff: [{ staffID: ObjectId, type, status }]
  ├─ many NurseSchedule (daily schedule per nurse)
  ├─ many PhysicianSchedule (daily schedule per physician)
  └─ many NurseAssignment (patient mapping per shift)

NurseProfile (separate from Nurse)
  ├─ qualifications & certifications
  └─ linked by manual nurseId (string), NOT ObjectId

Encounter (patient visit)
  └─ triage: { assigned_specialty, top_5_composite_physicians, ... }
```

---

## SECTION 8: TESTING ENDPOINTS

### Quick Manual Tests (sample commands)

```bash
# Login
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"nurse@hospital.com","password":"nurse123"}'

# Clock In
curl -X POST http://localhost:3000/api/nurses/clock-in \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"shiftID":"SHIFT-123","location":"ED","notes":"On time"}'

# Get Today Record
curl -X GET http://localhost:3000/api/nurses/today-record \
  -H "Authorization: Bearer <token>"

# Manager: Register Staff
curl -X POST http://localhost:3000/api/manager/staff/register \
  -H "Authorization: Bearer <manager-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"Sarah",
    "surname":"Johnson",
    "email":"sarah@hospital.com",
    "password":"nurse123",
    "role":"NURSE",
    "department":"ED",
    "specialty":"Emergency"
  }'

# Create Shift
curl -X POST http://localhost:3000/api/shifts/create \
  -H "Authorization: Bearer <manager-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "shiftName":"ED Morning",
    "department":"ED",
    "shiftStart":"07:00",
    "shiftEnd":"15:00",
    "maxStaffRequired":4
  }'
```

---

## SECTION 9: COMPLETENESS CHECKLIST

✅ = Implemented | ❌ = Missing | ⚠️ = Partial

| Feature | Status | Notes |
|---------|--------|-------|
| Staff registration | ✅ | Via manager dashboard `/api/manager/staff/register` |
| Multi-tenant isolation | ✅ | hospitalId filtering on all routes |
| Staff login | ✅ | Unified `/api/auth/login` |
| Clock in/out | ✅ | Full time tracking with breaks |
| Shift scheduling | ✅ | Recurring shifts supported |
| Staff-to-shift assignment | ✅ | Bulk assignment possible |
| Attendance dashboard | ✅ | Manager view of all staff status |
| Approval workflow | ✅ | Late arrival & break approvals |
| Schedule CSV import | ✅ | For physician & nurse schedules |
| Bed management | ✅ | Allocate/release beds |
| Patient assignment | ✅ | NurseAssignment model tracks |
| ESI triage ranking | ✅ | Python service integration |
| Real-time clock status | ✅ | Frontend polls every 5 sec |
| Department filtering | ✅ | Manager restricted by dept |
| Separate physician flow | ⚠️ | PhysicianSchedule exists but no separate controller |
| Audit logging | ✅ | auditLog middleware tracks actions |

---

## CONCLUSION

This diagnostic map captures the **current architectural state** as of April 2026:

- **21 models** total (with Nurse being the master staff model)
- **13 controllers** handling different domains
- **8 route files** organized by feature area
- **Multi-tenant support** via hospitalId filtering
- **Unified authentication** with role-based access control
- **Complex schedule management** with separate tables for nurses vs physicians
- **Manual relationship management** using string IDs (not MongoDB foreign keys)

The system supports a **complete workflow** from hospital registration → staff management → shift scheduling → clock in/out → attendance approval.

**Next steps for fixes/improvements would involve:**
1. Understanding why schedule logic differs between nurse/physician
2. Identifying missing physician-specific controllers/endpoints
3. Clarifying the role of NurseAssignment vs NurseSchedule
4. Analyzing the auto clock-out job for timing issues
