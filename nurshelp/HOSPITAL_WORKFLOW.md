# Hospital Management Workflow - Staff Cannot Self-Register

## Overview

The system uses a **unified Dashboard.jsx** with **strict role-based registration**:

- **Only Hospital Admin/Manager can register** new hospitals via public registration
- **Staff CANNOT self-register** - managers create staff accounts with passwords
- **All staff login** with credentials provided by manager

The application checks the JWT token role and renders appropriate content automatically.

---

## Key Principle: Manager-Controlled Staff Creation

✅ **Managers control everything:**
- Create staff accounts with passwords
- Define specialty and department
- Assign shifts
- Approve breaks and late arrivals

❌ **Staff cannot:**
- Self-register 
- Create their own accounts
- Change their own passwords (manager creates them)
- Define their own roles or departments

---

## User Registration & Login Flow

### Phase 1: Hospital Registration
When a user selects "Hospital Admin/Manager" on the registration form:

```
POST /api/hospital/register
{
  "hospitalName": "St. Mary's Hospital",
  "managerName": "John",
  "managerSurname": "Smith", 
  "managerEmail": "john@stmarys.com",
  "managerPassword": "secure123",
  "department": "All"
}

Response: {
  "token": "JWT_TOKEN",
  "user": { "id", "name", "role": "ADMIN", "department", "permissions" },
  "hospital": { "name": "St. Mary's Hospital", "adminManager": {...} }
}
```

User is logged in immediately with ADMIN role and has full permissions.

### Phase 1B: Admin/Manager Login
```
POST /api/hospital/login
{
  "email": "john@stmarys.com",
  "password": "secure123"
}

Response: {
  "token": "JWT_TOKEN",
  "user": { "id", "name", "role": "ADMIN", "department", "permissions" }
}
```

---

## Admin/Manager Dashboard Features

After login, admin/managers see the unified Dashboard with tabs:

### Tab 1: Overview
- Statistics dashboard
- Staff count
- Active shifts count
- Pending approvals

### Tab 2: Staff Management (Phase 2)
**Add new staff members directly in the dashboard:**

```
POST /api/manager/staff/register
Headers: Authorization: Bearer {token}
{
  "name": "Sarah",
  "surname": "Johnson",
  "email": "sarah@hospital.com",
  "password": "nurse123",
  "role": "NURSE",
  "department": "ED",
  "specialty": "Emergency Medicine"
}

Response: {
  "success": true,
  "message": "NURSE registered successfully",
  "staff": { "id", "name", "email", "role", "department", "specialty" }
}
```

**View all staff:**
```
GET /api/manager/staff/list
Headers: Authorization: Bearer {token}

Response: {
  "success": true,
  "count": 21,
  "staff": [...]
}
```

**Update staff:**
```
PUT /api/manager/staff/:staffID
Headers: Authorization: Bearer {token}
{ "department": "ICU", "specialty": "Critical Care" }
```

**Deactivate staff:**
```
DELETE /api/manager/staff/:staffID
Headers: Authorization: Bearer {token}
```

### Tab 3: Shift Management (Phase 3)
Coming soon - UI for creating and managing shifts

### Tab 4: Approvals
View pending break requests and late arrival approvals

---

## Staff Registration (Phase 2 Alternative)

IF a staff member registers themselves (before being added by manager):

```
POST /api/nurses/register
{
  "name": "Sarah",
  "surname": "Johnson",
  "email": "sarah@hospital.com",
  "password": "nurse123",
  "role": "NURSE"
}

Response: {
  "success": true,
  "nurse": { "id", "name", "email" }
}
```

---

## Phase 4: Staff Login & Clock In/Out

After the manager has created shifts, staff members can:

### Login
```
POST /api/nurses/login
{
  "email": "sarah@hospital.com",
  "password": "nurse123"
}

Response: {
  "token": "JWT_TOKEN",
  "nurse": { "id", "name", "email" }
}
```

When staff logs in, they see the **NurseDashboard** (not admin dashboard) with:
- Available shifts for the day
- Clock in/out button
- Break request options
- Clock history

### Clock In
```
POST /api/nurses/clockin
Headers: Authorization: Bearer {token}
{
  "location": "ED",
  "shiftID": "...",
  "notes": "Arrived on time"
}
```

### Clock Out
```
POST /api/nurses/clockout
Headers: Authorization: Bearer {token}
{ "notes": "Shift completed" }
```

### Break Start/End
```
POST /api/nurses/break-start
Headers: Authorization: Bearer {token}
{ "reason": "Lunch break" }

POST /api/nurses/break-end
Headers: Authorization: Bearer {token}
```

---

## RBAC Middleware (All Routes Protected)

### Hospital Routes (Public)
- POST /api/hospital/register - Anyone
- POST /api/hospital/login - Anyone

### Manager Routes (Protected)
```javascript
// Requires: protect (JWT), authorize('MANAGER', 'ADMIN')
POST   /api/manager/staff/register
GET    /api/manager/staff/list
PUT    /api/manager/staff/:staffID
DELETE /api/manager/staff/:staffID
GET    /api/manager/staff-status
GET    /api/manager/pending-approvals
POST   /api/manager/bulk-approvals
```

### Nurse Routes (Mixed)
```javascript
// Public endpoints
POST /api/nurses/register
POST /api/nurses/login

// Protected: authorize('NURSE', 'PHYSICIAN')
POST /api/nurses/clockin
POST /api/nurses/clockout
POST /api/nurses/break-start
POST /api/nurses/break-end
GET  /api/nurses/clockstatus
GET  /api/nurses/clockhistory
GET  /api/nurses/available-shifts
POST /api/nurses/break-request

// Protected: authorize('MANAGER', 'ADMIN')
POST /api/nurses/approve-late-arrival/:recordId
POST /api/nurses/approve-break/:recordId
```

---

## Frontend Implementation Detail

### Dashboard.jsx Logic

```javascript
// 1. User registers/logs in
handleRegister() / handleLogin()

// 2. JWT stored in localStorage
const token = localStorage.getItem("token")
const decoded = decodeJwt(token)  // Get role

// 3. Check role and render
if (decoded.role === "ADMIN" || decoded.role === "MANAGER") {
  return <AdminManagerDashboard />
} else if (decoded.role === "NURSE" || decoded.role === "PHYSICIAN") {
  return <NurseDashboard />
}
```

### AdminManagerDashboard Features

- **State Management**: staffList, staffForm, activeTab, loading, messages
- **Tabs**: overview, staff, shifts, approvals
- **Staff Form**: All fields in one form (name, email, password, role, department, specialty)
- **Staff List**: Table with name, role, department, email
- **API Integration**: Fetches from /api/manager/* endpoints

---

## Testing the Workflow

### Step 1: Register Hospital Admin
1. Go to Dashboard
2. Click "Register"
3. Select "Hospital Admin/Manager"
4. Fill in:
   - Hospital Name: "Test Hospital"
   - Name: "John"
   - Surname: "Admin"
   - Email: "john@hospital.com"
   - Password: "admin123"
5. Click Register
6. You're logged in as ADMIN and land on Admin Dashboard

### Step 2: Add Staff Members
1. Click "Staff Management" tab
2. Click "+ Add Staff"
3. Fill in nurse details:
   - Name: "Sarah", Surname: "Nurse"
   - Email: "sarah@hospital.com", Password: "nurse123"
   - Role: NURSE, Department: ED
4. Click "Add Staff Member"
5. Sarah appears in the staff list

### Step 3: Staff Login
1. Click Logout
2. Click "Login"
3. Enter Sarah's credentials (sarah@hospital.com / nurse123)
4. Sarah lands on NurseDashboard (different UI from admin)

### Step 4: Clock In
1. On NurseDashboard, click "Clock In"
2. Select shift and location
3. Clock in is recorded

---

## Key Benefits of Unified Dashboard

✅ **Single point of entry** - Dashboard.jsx handles all roles
✅ **Role-based rendering** - Automatically shows correct UI based on JWT role
✅ **Simplified state management** - No need for separate dashboard components
✅ **Easy to extend** - Add more roles/tabs without creating new files
✅ **Backend RBAC** - All endpoints protected with middleware
✅ **Audit trails** - All actions logged via auditLog middleware
✅ **Department filtering** - Managers see only their department staff

---

## File Structure

```
backend/
  src/
    controllers/
      ✓ hospital.controller.js (NEW - register/login)
      ✓ manager.controller.js (staff status, approvals)
      ✓ nurse.controller.js (clock in/out)
    routes/
      ✓ hospital.route.js (NEW - public routes)
      ✓ manager.route.js (UPDATED - full RBAC)
      ✓ nurse.route.js (UPDATED - RBAC + comments)
      ✓ shift.route.js (UPDATED - full RBAC)
    middleware/
      ✓ rbac.js (authorize, restrictByDepartment, auditLog, etc.)
    server.js (UPDATED - imports hospital routes)

frontend/
  src/
    Dashboard/
      ✓ Dashboard.jsx (UPDATED - unified with AdminManagerDashboard)
      ✓ RegisterForm.jsx (UPDATED - role selector)
      ✓ LoginForm.jsx (unchanged)
      ✓ NurseDashboard.jsx (unchanged)
      ✗ ManagerDashboard.jsx (REMOVED - functionality moved to Dashboard)
```

---

## Next Steps

1. **Create shift UI** - Dashboard tab for creating/managing shifts
2. **Implement clock in UI** - Full interface for staff clock in/out
3. **Add approval workflow** - UI for managers to approve break requests
4. **Permission system** - Fine-grained permissions beyond just role
5. **Audit log viewer** - Dashboard to view all actions taken
