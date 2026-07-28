# Hospital ID Isolation - Diagnosis & Fixes

## Problem Statement
When a manager registers a hospital and adds staff members, the staff is not appearing when assigning them to shifts. The issue is that **hospitalId** is not being properly propagated and filtered across the system.

---

## Root Causes Identified

### CRITICAL BUG #1: Staff Registration Missing hospitalId
**File:** `backend/src/routes/manager.route.js` (Line ~68-80)

**Current Code:**
```javascript
const newStaff = new Nurse({
  name,
  surname,
  email,
  password: hashedPassword,
  role: role.toUpperCase(),
  department: department || req.departmentFilter || 'General',
  specialty: specialty || 'General',
  licenseNumber: `LIC-${Date.now()}`,
  isActive: true,
  createdBy: req.user._id  // ← Missing hospitalId!
});
```

**Issue:** New staff member is created WITHOUT a hospitalId, even though it's a required field in the Nurse schema with an index. Staff belongs to the hospital but the reference is missing.

**Impact:**
- Staff can be created (but malformed)
- Staff won't be found when filtering by hospitalId
- Database index `hospitalId: 1, email: 1` won't work properly
- Queries filtered by hospitalId won't return this staff

---

### CRITICAL BUG #2: getAvailableStaff Not Filtering by hospitalId
**File:** `backend/src/controllers/shift.controller.js` (Line ~555-570)

**Current Code:**
```javascript
export const getAvailableStaff = async (req, res) => {
  try {
    const { department, specialty, type } = req.query;

    const query = { isActive: true };  // ← Missing hospitalId filter!
    if (department) query.department = department;
    if (specialty) query.specialty = specialty;
    if (type) query.role = type;

    const staff = await Nurse.find(query)
      .select('_id name surname email department specialty role')
      .limit(100);
```

**Issue:** When fetching available staff to assign to a shift, the query only filters by `isActive`, `department`, `specialty`, and `role`. It does NOT filter by the manager's `hospitalId`, so it fetches staff from ALL hospitals.

**Impact:**
- Newly registered staff from the same hospital won't appear (if they don't have hospitalId, they won't match)
- If multiple hospitals exist, staff lists get mixed
- A manager can't see their own staff

---

### BUG #3: assignStaffToShift Not Filtering by hospitalId
**File:** `backend/src/controllers/shift.controller.js` (Line ~380-410)

**Current Code:**
```javascript
export const assignStaffToShift = async (req, res) => {
  try {
    const { shiftID } = req.params;
    const { staffID, staffType, name, surname, specialty } = req.body;

    if (!staffID || !staffType) {
      return res.status(400).json({
        success: false,
        message: 'staffID and staffType are required'
      });
    }

    const shift = await Shift.findOne({ shiftID });  // ← Missing hospitalId filter!
    if (!shift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found'
      });
    }
```

**Issue:** Finding the shift only filters by `shiftID`, not by `hospitalId`. A manager from Hospital A could theoretically access/modify shifts from Hospital B.

**Impact:**
- Data isolation breach (security risk)
- Shift operations not scoped to hospital

---

### BUG #4: removeStaffFromShift Not Filtering by hospitalId
**File:** `backend/src/controllers/shift.controller.js` (Line ~450-470)

**Current Code:**
```javascript
const shift = await Shift.findOne({ shiftID });  // ← Missing hospitalId filter!
```

**Same issue as assignStaffToShift**

---

### BUG #5: publishShift Not Filtering by hospitalId
**File:** `backend/src/controllers/shift.controller.js` (Line ~385-395)

**Current Code:**
```javascript
const shift = await Shift.findOneAndUpdate(
  { shiftID },  // ← Missing hospitalId filter!
  { status: 'PUBLISHED' },
  { new: true }
);
```

**Same issue**

---

### BUG #6: registerNurse Not Setting hospitalId
**File:** `backend/src/controllers/nurse.controller.js` (Line ~10-40)

**Current Code:**
```javascript
const nurse = new Nurse({ name, surname, email, password });  // ← No hospitalId or createdBy!
```

**Issue:** Direct nurse registration endpoint doesn't set hospitalId. However, this endpoint shouldn't be used in production (should only use manager registration endpoint).

---

## Impact Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Data Isolation Failure Chain                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Manager registers hospital                             │
│     ✓ Hospital.hospitalId set correctly                    │
│     ✓ Manager.hospitalId set correctly                     │
│                                                             │
│  2. Manager adds staff member                              │
│     ✗ Staff.hospitalId NOT SET                             │
│     ✗ Staff becomes "orphaned" (no hospital)               │
│                                                             │
│  3. Manager tries to add staff to shift                    │
│     ✗ API fetches ALL available staff (no filter)          │
│     ✗ Newly added staff doesn't appear (missing hospitalId)│
│     ✗ Only shows staff from seed data or other hospitals   │
│                                                             │
│  4. Security Issue                                          │
│     ✗ Shifts can be accessed cross-hospital               │
│     ✗ Staff assignments not scoped to hospital             │
│     ✗ Managers can potentially see/modify other data       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Fix Strategy

### Fix Priority:
1. **CRITICAL**: Set hospitalId when registering staff (Bug #1)
2. **CRITICAL**: Filter getAvailableStaff by hospitalId (Bug #2)
3. **CRITICAL**: Filter all shift operations by hospitalId (Bugs #3-5)
4. **HIGH**: Verify all other endpoints that read from DB are hospital-scoped

---

## Required Code Changes

### FIX #1: Staff Registration - Set hospitalId
**File:** `backend/src/routes/manager.route.js`

Change from line ~68 to include `hospitalId`:

```javascript
const newStaff = new Nurse({
  hospitalId: req.user.hospitalId,  // ← ADD THIS
  name,
  surname,
  email,
  password: hashedPassword,
  role: role.toUpperCase(),
  department: department || req.departmentFilter || 'General',
  specialty: specialty || 'General',
  licenseNumber: `LIC-${Date.now()}`,
  isActive: true,
  createdBy: req.user._id
});
```

---

### FIX #2: getAvailableStaff - Filter by hospitalId
**File:** `backend/src/controllers/shift.controller.js`

Change from line ~560 to add hospitalId:

```javascript
export const getAvailableStaff = async (req, res) => {
  try {
    const { department, specialty, type } = req.query;

    const query = { 
      isActive: true,
      hospitalId: req.hospitalId  // ← ADD THIS (from middleware)
    };
    if (department) query.department = department;
    if (specialty) query.specialty = specialty;
    if (type) query.role = type;

    const staff = await Nurse.find(query)
      .select('_id name surname email department specialty role')
      .limit(100);
    
    // ... rest of function
  }
};
```

---

### FIX #3: assignStaffToShift - Filter by hospitalId
**File:** `backend/src/controllers/shift.controller.js`

Change from line ~390 to add hospitalId:

```javascript
const shift = await Shift.findOne({ 
  shiftID,
  hospitalId: req.hospitalId  // ← ADD THIS
});
```

---

### FIX #4: removeStaffFromShift - Filter by hospitalId
**File:** `backend/src/controllers/shift.controller.js`

Change from line ~458 to add hospitalId:

```javascript
const shift = await Shift.findOne({ 
  shiftID,
  hospitalId: req.hospitalId  // ← ADD THIS
});
```

---

### FIX #5: publishShift - Filter by hospitalId
**File:** `backend/src/controllers/shift.controller.js`

Change from line ~393 to add hospitalId:

```javascript
const shift = await Shift.findOneAndUpdate(
  { 
    shiftID,
    hospitalId: req.hospitalId  // ← ADD THIS
  },
  { status: 'PUBLISHED' },
  { new: true }
);
```

---

### FIX #6: updateShift - Filter by hospitalId
**File:** `backend/src/controllers/shift.controller.js`

Change from line ~520 to add hospitalId:

```javascript
const shift = await Shift.findOneAndUpdate(
  { 
    shiftID,
    hospitalId: req.hospitalId  // ← ADD THIS
  },
  updateData,
  { new: true, runValidators: true }
);
```

---

### FIX #7: deleteShift - Filter by hospitalId
**File:** `backend/src/controllers/shift.controller.js`

Change from line ~545 to add hospitalId:

```javascript
const shift = await Shift.findOneAndDelete({ 
  shiftID,
  hospitalId: req.hospitalId  // ← ADD THIS
});
```

---

## Verification Checklist

After applying fixes:

- [ ] Verify `req.hospitalId` is available in all routes (check authmiddleware & protect)
- [ ] Check if `req.hospitalId` exists or if we need to add it from `req.user.hospitalId`
- [ ] Test staff registration: Staff should have hospitalId set
- [ ] Test staff retrieval: Only staff from the manager's hospital should appear
- [ ] Test shift operations: Shifts should be scoped to hospital
- [ ] Test multi-tenant: Create 2 hospitals, verify data isolation
- [ ] Check all other schemas (Shifts, Encounters, Triages, etc.) for similar missing hospitalId filters

---

## Database Index Verification

Once fixes are applied, verify these indexes exist:

**Nurse Model:**
```javascript
nurseSchema.index({ hospitalId: 1, email: 1 }, { unique: true });
nurseSchema.index({ hospitalId: 1, isActive: 1 });
nurseSchema.index({ hospitalId: 1, department: 1 });
nurseSchema.index({ hospitalId: 1, role: 1 });
```

**Shift Model:**
```javascript
shiftSchema.index({ hospitalId: 1, department: 1 });
shiftSchema.index({ hospitalId: 1, status: 1 });
shiftSchema.index({ hospitalId: 1, date: 1 });
```

These indexes ensure efficient filtering by hospital.

---

## Testing Plan

```javascript
// Test 1: Create hospital + staff
POST /api/hospital/register
  Body: { hospitalName, managerName, ... }
  Expected: Hospital created, Manager has hospitalId

// Test 2: Add staff
POST /api/manager/staff/register
  Body: { name, surname, email, password, role, department, specialty }
  Expected: Staff.hospitalId === Manager.hospitalId

// Test 3: Get available staff
GET /api/shift/staff/available?department=ED
  Expected: Only staff from manager's hospital (with hospitalId)

// Test 4: Create shift
POST /api/shift/create
  Body: { shiftName, department, shiftStart, shiftEnd, maxStaffRequired }
  Expected: Shift.hospitalId === Manager.hospitalId

// Test 5: Assign staff to shift
POST /api/shift/:shiftID/assign-staff
  Body: { staffID, staffType, name, surname, specialty }
  Expected: Staff assignment succeeds, staff appears in shift.assignedStaff
```

---

## Additional Recommendations

1. **Add hospitalId to all queries in shift.controller.js**
   - Need to check every Shift.find(), Shift.findOne(), shifts retrieval
   - Should already be done in getShifts, getShiftsForDay, getShiftsForWeek, getShiftsForMonth (via req.hospitalId in query)

2. **Add hospitalId to other controllers**
   - encounter.controller.js
   - triage.controller.js
   - nurse.controller.js (for clock records, schedules)
   - All analytics endpoints

3. **Middleware verification**
   - Ensure `protect` middleware sets both `req.user` and `req.hospitalId` correctly
   - Check if `req.hospitalId` comes from middleware or needs to be extracted from `req.user.hospitalId`

4. **Compound unique indexes**
   - Ensure email is unique per hospital, not globally
   - Shift names might need to be unique per hospital + department

5. **Testing in staging**
   - Multi-tenant test: Create 2 hospitals with same staff names
   - Verify staff from Hospital A never appears in Hospital B
   - Verify Manager from Hospital A can't access shifts from Hospital B
