# Hospital Staff & Shift Management System Setup

## 📋 Overview

This system implements real-world hospital staff scheduling with:
- **Role-Based Access Control (RBAC)**
- **Multiple Staff Types** (Nurses, Physicians, Managers)
- **Shift Management** (Daily, Weekly, Monthly scheduling)
- **Staff Assignment** to shifts
- **Professional Authorization** for different user roles

---

## 🚀 Quick Start

### Step 1: Seed the Database

Clear all existing data and populate with realistic hospital staff:

```bash
cd backend
node src/seeds/seedHospitalStaff.js
```

**What this does:**
- ✅ Clears all patients and triage records
- ✅ Clears all nurses and physicians
- ✅ Clears all schedules and clock records
- ✅ Populates 4 Managers (different roles)
- ✅ Populates 20 Nurses (various specialties)
- ✅ Populates 17 Specialist Physicians

### Step 2: Default Credentials

After seeding, use these credentials to log in:

#### 🔐 Managers
| Role | Email | Password |
|------|-------|----------|
| Admin | david.brown@hospital.com | admin123 |
| Shift Manager (ED) | john.anderson@hospital.com | manager123 |
| Dept Manager (ICU) | patricia.williams@hospital.com | manager123 |
| Shift Manager (OR) | susan.miller@hospital.com | manager123 |

#### 👩‍⚕️ Nurses
- Email pattern: `{firstName}.{number}@hospital.com`
- Password: `nurse123`
- Example: `sarah.1@hospital.com`

#### 👨‍⚕️ Physicians
- Email pattern: `{doctorname}@hospital.com`
- Password: `physician123`
- Example: `william@hospital.com`

---

## 📊 Database Schema Changes

### New Models

#### 1. **Manager Model** (`manager.model.js`)
```javascript
{
  managerID: String (unique),
  name: String,
  surname: String,
  email: String (unique),
  password: String (hashed),
  department: String, // ED, ICU, OR, OB, Medical-Surgical, etc.
  role: ENUM [SHIFT_MANAGER, DEPARTMENT_MANAGER, ADMIN],
  permissions: {
    canCreateShifts: Boolean,
    canEditShifts: Boolean,
    canDeleteShifts: Boolean,
    canAssignStaff: Boolean,
    canApproveLeaves: Boolean,
    canManageRoles: Boolean,
    canViewReports: Boolean
  },
  isActive: Boolean,
  lastLogin: Date
}
```

#### 2. **Shift Model** (`shift.model.js`)
```javascript
{
  shiftID: String (unique),
  shiftName: String,
  department: String, // ED, ICU, OR, etc.
  shiftStart: String, // HH:MM format
  shiftEnd: String,   // HH:MM format
  duration: Number,   // in hours
  date: Date,         // for specific-date shifts
  dayOfWeek: String,  // Monday-Sunday (for recurring)
  recurrence: ENUM [ONCE, DAILY, WEEKLY, BIWEEKLY, MONTHLY],
  recurrenceEndDate: Date,
  maxStaffRequired: Number,
  assignedStaff: [
    {
      staffID: String,
      type: ENUM [NURSE, PHYSICIAN, SPECIALIST, SUPPORT],
      name: String,
      surname: String,
      specialty: String,
      status: ENUM [ASSIGNED, CONFIRMED, ABSENT, PENDING]
    }
  ],
  status: ENUM [DRAFT, PUBLISHED, CANCELLED, COMPLETED],
  createdBy: String, // Manager ID
  notes: String
}
```

---

## 🔌 API Endpoints

### Shift Management Routes (`/api/shifts/`)

#### Create Shift
```
POST /api/shifts/create
Protected: Yes (Manager/Admin only)

Body:
{
  "shiftName": "Morning Shift - ED",
  "department": "ED",
  "shiftStart": "08:00",
  "shiftEnd": "16:00",
  "date": "2026-04-15",  // Optional for recurring
  "dayOfWeek": "Monday",  // Optional, required for recurring
  "recurrence": "ONCE",   // ONCE, DAILY, WEEKLY, BIWEEKLY, MONTHLY
  "recurrenceEndDate": "2026-05-15",
  "maxStaffRequired": 5,
  "notes": "Additional notes"
}

Response:
{
  "success": true,
  "message": "Shift created successfully",
  "shift": { ...shiftObject }
}
```

#### Get All Shifts (with filters)
```
GET /api/shifts/list?department=ED&status=PUBLISHED&startDate=2026-04-01&endDate=2026-04-30
Protected: Yes

Query Parameters:
- department: ED, ICU, OR, OB, Medical-Surgical, Pediatrics, Psychiatry
- status: DRAFT, PUBLISHED, CANCELLED, COMPLETED
- dayOfWeek: Monday-Sunday
- startDate: YYYY-MM-DD
- endDate: YYYY-MM-DD

Response:
{
  "success": true,
  "count": 15,
  "shifts": [ ...shiftArray ]
}
```

#### Get Shifts for Specific Day
```
GET /api/shifts/day?date=2026-04-15&department=ED
Protected: Yes

Response:
{
  "success": true,
  "date": "2026-04-15",
  "count": 5,
  "shifts": [ ...shiftArray ]
}
```

#### Get Shifts for Week
```
GET /api/shifts/week?startDate=2026-04-07&department=ED
Protected: Yes

Response:
{
  "success": true,
  "startDate": "2026-04-07",
  "endDate": "2026-04-14",
  "count": 35,
  "shifts": [ ...shiftArray ],
  "shiftsByDay": {
    "Mon, Apr 7": [ ...shifts ],
    "Tue, Apr 8": [ ...shifts ],
    ...
  }
}
```

#### Get Shifts for Month
```
GET /api/shifts/month?year=2026&month=4&department=ED
Protected: Yes

Response:
{
  "success": true,
  "year": 2026,
  "month": 4,
  "count": 120,
  "shifts": [ ...shiftArray ],
  "shiftsByWeek": {
    "Week 1": [ ...shifts ],
    "Week 2": [ ...shifts ],
    ...
  }
}
```

#### Update Shift
```
PUT /api/shifts/{shiftID}
Protected: Yes

Body: { Any fields to update }

Response: Updated shift object
```

#### Delete Shift
```
DELETE /api/shifts/{shiftID}
Protected: Yes

Response: Deleted shift object
```

#### Publish Shift (make available for staff)
```
POST /api/shifts/{shiftID}/publish
Protected: Yes

Response: Updated shift with status=PUBLISHED
```

#### Assign Staff to Shift
```
POST /api/shifts/{shiftID}/assign-staff
Protected: Yes

Body:
{
  "staffID": "NURSE-001",
  "staffType": "NURSE",  // NURSE or PHYSICIAN
  "name": "Sarah",
  "surname": "Johnson",
  "specialty": "Emergency Nurse"
}

Response: Updated shift with new staff assignment
```

#### Remove Staff from Shift
```
POST /api/shifts/{shiftID}/remove-staff
Protected: Yes

Body:
{
  "staffID": "NURSE-001"
}

Response: Updated shift with staff removed
```

#### Get Available Staff
```
GET /api/shifts/staff/available?department=ED&specialty=Emergency Nurse&type=NURSE
Protected: Yes

Query Parameters:
- department: Filter by department
- specialty: Filter by specialty
- type: NURSE or PHYSICIAN

Response:
{
  "success": true,
  "count": 10,
  "staff": [
    {
      "_id": "...",
      "name": "Sarah",
      "surname": "Johnson",
      "email": "sarah.1@hospital.com",
      "department": "ED",
      "specialty": "Emergency Nurse",
      "role": "NURSE"
    }
  ]
}
```

---

## 👥 Staff Types & Specialties

### Nurses by Department

#### Emergency Department (ED)
- Emergency Nurse (3 staff)
- Trauma Nurse (1 staff)

#### ICU
- ICU Nurse (3 staff)
- Critical Care Nurse (varies)

#### Operating Room (OR)
- Theatre Nurse (3 staff)
- Perioperative Nurse (varies)

#### Obstetrics (OB)
- Midwife (2 staff)
- Neonatal Nurse (1 staff)

#### Medical-Surgical
- Dialysis Nurse (1 staff)
- Oncology Nurse (1 staff)
- General Nurse (1 staff)

#### Pediatrics
- Paediatric Nurse (2 staff)
- Neonatal Nurse (varies)

#### Psychiatry
- Mental Health Nurse (3 staff)

### Physicians by Specialty

- Emergency Medicine (2)
- Critical Care (2)
- General Surgery (1)
- Orthopaedic Surgery (1)
- Cardiovascular Surgery (1)
- Obstetrics & Gynecology (2)
- Cardiology (1)
- Pulmonology (1)
- Gastroenterology (1)
- Nephrology (1)
- Pediatrics (1)
- Neonatology (1)
- Psychiatry (2)

---

## 🔐 Role-Based Permissions

### Shift Manager
- ✅ Can create shifts
- ✅ Can edit shifts
- ✅ Can delete shifts
- ✅ Can assign staff
- ❌ Cannot approve leaves
- ❌ Cannot manage roles

### Department Manager
- ✅ Can create shifts
- ✅ Can edit shifts
- ✅ Can delete shifts
- ✅ Can assign staff
- ✅ Can approve leaves
- ❌ Cannot manage roles

### Admin
- ✅ Full permissions
- ✅ Can manage all roles
- ✅ Can approve leaves
- ✅ Can view all reports

---

## 📱 Frontend Integration

### Manager Dashboard Components (To Be Created):

1. **ShiftCreationPage**
   - Form to create shifts
   - Select department, times, recurrence
   - Set max staff required

2. **ShiftCalendarView**
   - Monthly/weekly calendar view
   - Color-coded by department
   - Click to view/edit shifts

3. **StaffAssignmentPanel**
   - View available staff for a shift
   - Drag-drop or click to assign
   - Remove staff functionality

4. **ShiftManagementDashboard**
   - Overview of all shifts
   - Filter by department/status
   - Publish/draft shifts
   - View staff assignments

---

## 🧪 Testing

### Example: Create a Shift for the ED on April 15, 2026

```bash
curl -X POST http://localhost:5001/api/shifts/create \
  -H "Authorization: Bearer YOUR_MANAGER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "shiftName": "Morning ED Shift",
    "department": "ED",
    "shiftStart": "08:00",
    "shiftEnd": "16:00",
    "date": "2026-04-15",
    "maxStaffRequired": 5,
    "notes": "Hospital-wide drill day"
  }'
```

### Example: Get Shifts for April 2026

```bash
curl -X GET 'http://localhost:5001/api/shifts/month?year=2026&month=4' \
  -H "Authorization: Bearer YOUR_MANAGER_TOKEN"
```

---

## 📝 Notes

1. **Passwords are hashed** using bcrypt (10 rounds)
2. **All endpoints require Bearer token authentication**
3. **Managers can only manage shifts in their assigned department** (implement in controllers)
4. **Recurring shifts** can be configured for daily, weekly, biweekly, or monthly
5. **Staff can only clock in for their assigned shifts** (frontend validation)

---

## 🔄 Next Steps

1. ✅ Database seeded with realistic data
2. ✅ Shift management APIs created
3. ⏳ Create Manager Dashboard UI (React)
4. ⏳ Implement staff shift selection in clock-in
5. ⏳ Create shift approval workflow
6. ⏳ Add shift conflict detection
7. ⏳ Generate shift reports

---

## 📞 Support

For issues or questions:
1. Check the Shift model schema
2. Review the controller for endpoint details
3. Verify Bearer token is valid
4. Check user role has required permissions
