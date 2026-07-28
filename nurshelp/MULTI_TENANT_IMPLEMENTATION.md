# Multi-Tenant Hospital Management System - Implementation Guide

## Architecture Overview

This is a **multi-tenant SaaS application** where each hospital operates independently within the same system. Key principles:

- **Hospital Isolation**: Data is partitioned by `hospitalId`
- **Shared Infrastructure**: All hospitals share the same database and servers (more cost-effective)
- **Scalability**: Easy to add new hospitals without infrastructure changes
- **Security**: Proper indexing and access control prevent cross-hospital data leakage

---

## Database Schema & Models

### 1. Hospital Model
**Reference**: `backend/src/models/hospital.model.js`

```javascript
{
  hospitalID: "HOSP-1234567890",        // Unique identifier
  name: "St. Mary's Hospital",
  country: "USA",
  city: "New York",
  state: "NY",
  zipCode: "10001",
  phone: "+1-555-0123",
  email: "contact@stmarys.com",
  
  // Primary administrator
  adminManagerId: ObjectId,              // Reference to Manager
  adminManagerName: "John Smith",
  adminManagerEmail: "john@stmarys.com",
  
  // Hospital status
  isActive: true,
  
  // Departments
  departments: [
    { name: "ED", isActive: true },
    { name: "ICU", isActive: true }
  ],
  
  // Subscription info
  subscription: {
    plan: "PROFESSIONAL",                // FREE, BASIC, PROFESSIONAL, ENTERPRISE
    maxManagers: 10,
    maxStaff: 500,
    maxPatients: 5000,
    startDate: ISODate,
    expiryDate: ISODate
  },
  
  createdAt: ISODate,
  updatedAt: ISODate
}
```

### 2. Manager Model
**Reference**: `backend/src/models/manager.model.js`

Managers/Admins belong to ONE hospital and manage scheduling/staff.

```javascript
{
  hospitalId: ObjectId,                  // Tenant isolation - REQUIRED
  managerID: "MGR-1234567890",          // Unique globally
  name: "John",
  surname: "Smith",
  email: "john@stmarys.com",            // Unique PER HOSPITAL (not global)
  password: "hashed_password",
  phone: "+1-555-0200",
  
  // Department responsibility
  department: "All" | "ED" | "ICU" | "OR" | "OB" | "Medical-Surgical" | "Pediatrics" | "Psychiatry",
  
  // Access level within hospital
  role: "SHIFT_MANAGER" | "DEPARTMENT_MANAGER" | "ADMIN",
  
  // Fine-grained permissions
  permissions: {
    canCreateShifts: true,
    canEditShifts: true,
    canDeleteShifts: true,
    canAssignStaff: true,
    canApproveLeaves: true,
    canManageRoles: true,
    canViewReports: true
  },
  
  isActive: true,
  lastLogin: ISODate,
  createdAt: ISODate,
  updatedAt: ISODate
}
```

**Important**: Email is unique per hospital (not globally):
```javascript
// Index in database
managerSchema.index({ hospitalId: 1, email: 1 }, { unique: true });
```

This allows:
- `alice@site.com` at Hospital A
- `alice@site.com` at Hospital B (different person, same email)

### 3. Nurse Model
**Reference**: `backend/src/models/nurse.model.js`

Nurses/staff members belong to ONE hospital.

```javascript
{
  hospitalId: ObjectId,                  // Tenant isolation
  staffID: "NURSE-1234567890",          // Unique globally
  name: "Alice",
  surname: "Johnson",
  email: "alice@stmarys.com",
  phone: "+1-555-0300",
  
  // Professional info
  role: "RN" | "LPN" | "CNA" | "NURSE_AIDE",
  specialization: "Emergency Medicine" | "Critical Care" | "Pediatrics" | etc,
  department: "ED" | "ICU" | etc,
  licenseNumber: "RN123456",
  
  // Credentials
  certifications: [
    { name: "BLS", expiryDate: ISODate },
    { name: "ACLS", expiryDate: ISODate }
  ],
  
  // Employment status
  isActive: true,
  hireDate: ISODate,
  employmentStatus: "FULL_TIME" | "PART_TIME" | "CONTRACT",
  
  // Current assignment
  currentShift: ObjectId,                // Reference to Shift
  leaveBalance: 15,                      // Days remaining
  
  // Audit
  createdAt: ISODate,
  updatedAt: ISODate
}
```

### 4. Patient Model
**Reference**: `backend/src/models/patient.model.js`

Patients are specific to a hospital.

```javascript
{
  hospitalId: ObjectId,
  mrn: "MRN-1234567",                   // Medical Record Number (unique per hospital)
  
  // Demographics
  firstName: "Robert",
  lastName: "Brown",
  dateOfBirth: ISODate,
  gender: "MALE" | "FEMALE" | "OTHER",
  phone: "+1-555-0400",
  email: "robert@email.com",
  
  // Address
  address: {
    street: "123 Main St",
    city: "New York",
    state: "NY",
    zipCode: "10001"
  },
  
  // Medical
  allergies: ["Penicillin", "Shellfish"],
  medications: ["Aspirin 81mg", "Lisinopril 10mg"],
  medicalHistory: ["Diabetes", "Hypertension"],
  bloodType: "O+",
  
  // Emergency contact
  emergencyContact: {
    name: "Jane Brown",
    relationship: "Spouse",
    phone: "+1-555-0401"
  },
  
  // Insurance
  insurance: {
    provider: "Blue Cross",
    memberId: "BC123456",
    groupNumber: "G789"
  },
  
  // Status
  isActive: true,
  admissionDate: ISODate,
  dischargeDate: ISODate,
  
  createdAt: ISODate,
  updatedAt: ISODate
}
```

---

## API Endpoints - Phase 1 (Implemented)

### Register New Hospital

```http
POST /api/hospital/register
Content-Type: application/json

{
  "hospitalName": "St. Mary's Hospital",
  "managerName": "John",
  "managerSurname": "Smith",
  "managerEmail": "john@stmarys.com",
  "managerPassword": "SecurePassword123!",
  "department": "All"
}
```

**Response** (201 Created):
```json
{
  "success": true,
  "message": "Hospital registered successfully",
  "hospital": {
    "hospitalID": "HOSP-1234567890",
    "name": "St. Mary's Hospital",
    "adminManager": {
      "managerID": "MGR-1234567890",
      "name": "John",
      "email": "john@stmarys.com",
      "role": "ADMIN"
    }
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "ObjectId",
    "name": "John",
    "email": "john@stmarys.com",
    "role": "ADMIN",
    "department": "All",
    "hospitalId": "ObjectId",
    "hospitalName": "St. Mary's Hospital"
  }
}
```

### Manager Login

```http
POST /api/hospital/login
Content-Type: application/json

{
  "email": "john@stmarys.com",
  "password": "SecurePassword123!"
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Login successful",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "ObjectId",
    "name": "John",
    "email": "john@stmarys.com",
    "role": "ADMIN",
    "department": "All",
    "hospitalId": "ObjectId",
    "hospitalName": "St. Mary's Hospital",
    "permissions": {
      "canCreateShifts": true,
      "canEditShifts": true,
      "canDeleteShifts": true,
      "canAssignStaff": true,
      "canApproveLeaves": true,
      "canManageRoles": true,
      "canViewReports": true
    }
  }
}
```

---

## JWT Token Structure

The JWT includes hospital context for easy filtering at the database level:

```javascript
{
  id: "ObjectId",                      // Manager ID
  email: "john@stmarys.com",
  name: "John",
  role: "ADMIN",
  department: "All",
  hospitalId: "ObjectId",              // KEY: Used to filter all queries
  hospitalName: "St. Mary's Hospital",
  iat: 1234567890,                     // Issued at
  exp: 1234671490                      // Expires in 24 hours
}
```

**Usage in Requests**:
- Token is extracted from `Authorization: Bearer <token>`
- Backend decodes token to get `hospitalId`
- ALL database queries are filtered: `{ hospitalId: decoded.hospitalId, ... }`

---

## Security & Multi-Tenant Isolation

### 1. Database Level
- Every data model has `hospitalId` field
- Compound unique indexes prevent duplicates within hospital:
  ```javascript
  // Managers in same hospital can't have same email
  managerSchema.index({ hospitalId: 1, email: 1 }, { unique: true });
  
  // Nurses in same hospital can't have same email
  nurseSchema.index({ hospitalId: 1, email: 1 }, { unique: true });
  ```

### 2. Query Level
All queries MUST include `hospitalId` filter:

```javascript
// CORRECT - Filtered by hospital
const managers = await Manager.find({ 
  hospitalId: req.user.hospitalId  // From JWT
});

// WRONG - Could expose other hospitals' data
const managers = await Manager.find({});

// CORRECT - Get specific manager only if in same hospital
const manager = await Manager.findOne({
  _id: managerId,
  hospitalId: req.user.hospitalId  // Essential protection
});
```

### 3. Middleware (To Be Implemented)
```javascript
// Authentication middleware
const auth = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  const decoded = jwt.verify(token, JWT_SECRET);
  req.user = decoded;  // Contains hospitalId
  next();
};

// Authorization middleware
const authorize = (permissions) => (req, res, next) => {
  const hasPermission = permissions.some(
    p => req.user.permissions[p]
  );
  if (!hasPermission) return res.status(403).json({ error: 'Forbidden' });
  next();
};

// Tenant isolation middleware
const tenantIsolation = (req, res, next) => {
  // Automatically add hospitalId to all responses
  res.locals.hospitalId = req.user.hospitalId;
  next();
};
```

---

## Development Notes

### Running the Application

1. **Start MongoDB**:
   ```bash
   mongod
   ```

2. **Install dependencies**:
   ```bash
   cd backend
   npm install
   ```

3. **Set environment variables**:
   ```bash
   # .env
   JWT_SECRET=your_secret_key
   MONGODB_URI=mongodb://localhost:27017/nursehelpdesk
   PORT=5000
   ```

4. **Start server**:
   ```bash
   npm run dev
   ```

5. **Test registration**:
   ```bash
   curl -X POST http://localhost:5000/api/hospital/register \
     -H "Content-Type: application/json" \
     -d '{
       "hospitalName": "Test Hospital",
       "managerName": "John",
       "managerSurname": "Doe",
       "managerEmail": "john@testhospital.com",
       "managerPassword": "password123",
       "department": "All"
     }'
   ```

### File Structure
```
backend/
├── src/
│   ├── models/
│   │   ├── hospital.model.js       ✅ Implemented
│   │   ├── manager.model.js        ✅ Implemented  
│   │   ├── nurse.model.js          ✅ Implemented
│   │   ├── patient.model.js        ✅ Implemented
│   │   ├── physician.model.js      ✅ Implemented
│   │   ├── shift.model.js          ✅ Implemented
│   │   └── schedule.model.js       ✅ Implemented
│   ├── controllers/
│   │   └── hospital.controller.js  ✅ Phase 1 Done
│   │       (nurse, shift, schedule controllers TBD)
│   ├── routes/
│   │   └── hospital.route.js       ✅ Phase 1 Done
│   │       (nurse, shift, schedule routes TBD)
│   ├── middleware/                 ❓ TBD
│   └── server.js
└── package.json
```

---

## Next Phase Features (Phase 2)

1. **Nurse Management Routes**
   - `POST /api/nurse/register` - Register new nurse to hospital
   - `GET /api/nurse` - List all nurses (filtered by hospital)
   - `PUT /api/nurse/:id` - Update nurse details
   - `DELETE /api/nurse/:id` - Deactivate nurse

2. **Shift Management Routes**
   - `POST /api/shift/create` - Create new shift
   - `GET /api/shift` - List shifts for hospital
   - `PUT /api/shift/:id` - Edit shift
   - `DELETE /api/shift/:id` - Cancel shift

3. **Scheduling Routes**
   - `POST /api/schedule/generate` - Auto-generate weekly schedule
   - `GET /api/schedule` - View current schedule
   - `PATCH /api/schedule/assign` - Assign nurse to shift

4. **Leave Management**
   - `POST /api/leave/request` - Nurse requests leave
   - `GET /api/leave/requests` - Manager reviews requests
   - `PUT /api/leave/approve` - Approve/reject leave

5. **Reports**
   - `GET /api/reports/attendance` - Staff attendance
   - `GET /api/reports/utilization` - Shift utilization
   - `GET /api/reports/payroll` - Payroll data

---

## Troubleshooting

### Issue: Email already exists error on registration
**Cause**: Global uniqueness check is too strict or email exists in different hospital
**Fix**: Verify compound index is properly applied: `managerSchema.index({ hospitalId: 1, email: 1 }, { unique: true });`

### Issue: Data from Hospital A visible in Hospital B
**Cause**: Query missing hospitalId filter
**Fix**: Always include `hospitalId: req.user.hospitalId` in find() queries

### Issue: JWT token doesn't include hospitalId
**Cause**: Controller not populating hospital info on login
**Fix**: See loginManager() in hospital.controller.js - ensure `manager.populate('hospitalId')`

---

## References

- MongoDB Indexing: https://docs.mongodb.com/manual/indexes/
- JWT Best Practices: https://tools.ietf.org/html/rfc7519
- Multi-Tenancy Patterns: https://docs.microsoft.com/en-us/aspnet/core/fundamentals/configuration/options#isample-configuring-multiple-options

