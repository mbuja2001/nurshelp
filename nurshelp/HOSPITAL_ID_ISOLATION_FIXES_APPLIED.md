# Hospital ID Isolation - Fixes Applied & Testing Guide

## Summary of Fixes Applied

All critical fixes have been applied to enforce hospital-level data isolation across the system.

### Fix #1: Staff Registration Now Sets hospitalId ✅
**File:** `backend/src/routes/manager.route.js` (line ~68)
**Change:** Added `hospitalId: req.hospitalId` when creating new Nurse record
**Status:** APPLIED ✅

```javascript
// BEFORE:
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
  createdBy: req.user._id
});

// AFTER:
const newStaff = new Nurse({
  hospitalId: req.hospitalId,  // ← ADDED
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

**Result:** New staff members now belong to a specific hospital.

---

### Fix #2: getAvailableStaff Now Filters by hospitalId ✅
**File:** `backend/src/controllers/shift.controller.js` (line ~560)
**Change:** Added `hospitalId: req.hospitalId` to query filter
**Status:** APPLIED ✅

```javascript
// BEFORE:
const query = { isActive: true };
if (department) query.department = department;
if (specialty) query.specialty = specialty;
if (type) query.role = type;

// AFTER:
const query = { 
  isActive: true,
  hospitalId: req.hospitalId  // ← ADDED
};
if (department) query.department = department;
if (specialty) query.specialty = specialty;
if (type) query.role = type;
```

**Result:** Only staff from the manager's hospital will appear in the available staff list.

---

### Fix #3: assignStaffToShift Now Filters by hospitalId ✅
**File:** `backend/src/controllers/shift.controller.js` (line ~393)
**Change:** Added `hospitalId: req.hospitalId` to shift query
**Status:** APPLIED ✅

```javascript
// BEFORE:
const shift = await Shift.findOne({ shiftID });

// AFTER:
const shift = await Shift.findOne({ shiftID, hospitalId: req.hospitalId });
```

**Result:** Shifts are scoped to manager's hospital. Cross-hospital data access prevented.

---

### Fix #4: removeStaffFromShift Now Filters by hospitalId ✅
**File:** `backend/src/controllers/shift.controller.js` (line ~458)
**Change:** Added `hospitalId: req.hospitalId` to shift query
**Status:** APPLIED ✅

```javascript
// BEFORE:
const shift = await Shift.findOne({ shiftID });

// AFTER:
const shift = await Shift.findOne({ shiftID, hospitalId: req.hospitalId });
```

**Result:** Prevents manager from removing staff from shifts in other hospitals.

---

### Fix #5: updateShift Now Filters by hospitalId ✅
**File:** `backend/src/controllers/shift.controller.js` (line ~520)
**Change:** Added `hospitalId: req.hospitalId` to update query
**Status:** APPLIED ✅

```javascript
// BEFORE:
const shift = await Shift.findOneAndUpdate(
  { shiftID },
  updateData,
  { new: true, runValidators: true }
);

// AFTER:
const shift = await Shift.findOneAndUpdate(
  { shiftID, hospitalId: req.hospitalId },
  updateData,
  { new: true, runValidators: true }
);
```

**Result:** Prevents manager from updating shifts in other hospitals.

---

### Fix #6: deleteShift Now Filters by hospitalId ✅
**File:** `backend/src/controllers/shift.controller.js` (line ~545)
**Change:** Added `hospitalId: req.hospitalId` to delete query
**Status:** APPLIED ✅

```javascript
// BEFORE:
const shift = await Shift.findOneAndDelete({ shiftID });

// AFTER:
const shift = await Shift.findOneAndDelete({ shiftID, hospitalId: req.hospitalId });
```

**Result:** Prevents manager from deleting shifts from other hospitals.

---

### Fix #7: publishShift Now Filters by hospitalId ✅
**File:** `backend/src/controllers/shift.controller.js` (line ~347)
**Change:** Added `hospitalId: req.hospitalId` to update query
**Status:** APPLIED ✅

```javascript
// BEFORE:
const shift = await Shift.findOneAndUpdate(
  { shiftID },
  { status: 'PUBLISHED' },
  { new: true }
);

// AFTER:
const shift = await Shift.findOneAndUpdate(
  { shiftID, hospitalId: req.hospitalId },
  { status: 'PUBLISHED' },
  { new: true }
);
```

**Result:** Prevents manager from publishing shifts in other hospitals.

---

## How It Works Now

```
Step 1: Manager registers hospital
  POST /api/hospital/register
  Response includes: token with hospitalId

Step 2: Manager logs in
  POST /api/hospital/login
  Token includes: hospitalId in JWT payload
  
Step 3: Manager's request arrives
  Headers: Authorization: Bearer <token>
  
Step 4: Middleware (protect) processes request
  Decodes JWT
  Extracts: hospitalId from token
  Sets: req.hospitalId = decoded.hospitalId
  (Also sets req.user from database)
  
Step 5: Manager adds staff member
  POST /api/manager/staff/register
  Data: { name, surname, email, password, role, department }
  Middleware sets: req.hospitalId ✓
  Endpoint now creates: Nurse({ hospitalId: req.hospitalId, ... }) ✓
  Result: Staff record has hospitalId set ✓
  
Step 6: Manager creates shift
  POST /api/shift/create
  Endpoint creates: Shift({ hospitalId: req.hospitalId, ... }) ✓
  (Already working, shift.controller.js line ~18)
  
Step 7: Manager assigns staff to shift
  GET /api/shift/staff/available
  Query now includes: { hospitalId: req.hospitalId } ✓
  Result: Only staff from manager's hospital returned ✓
  
Step 8: Manager assigns specific staff
  POST /api/shift/:shiftID/assign-staff
  Query now includes: { shiftID, hospitalId: req.hospitalId } ✓
  Result: Staff can only be assigned to their hospital's shifts ✓
```

---

## Testing Checklist

### Test 1: Single Hospital Registration
```
1. Go to registration page
2. Register Hospital A with Manager A
3. Log in as Manager A
4. Add Staff Member S1
5. Create Shift SH1
6. Assign Staff S1 to Shift SH1
7. Verify: S1 appears in the shift's assigned staff
EXPECTED: ✅ Staff appears and can be assigned
```

### Test 2: Multi-Hospital Data Isolation

```
Hospital A:
├─ Manager A1
├─ Staff S1, S2, S3
└─ Shifts SH1, SH2

Hospital B:
├─ Manager B1
├─ Staff S4, S5, S6
└─ Shifts SH3, SH4

Test Scenario:
1. Manager A1 logs in
2. Manager A1 tries to GET available staff
   EXPECTED: ✅ Only S1, S2, S3 (not S4, S5, S6)
   
3. Manager A1 tries to assign S1 to SH1
   EXPECTED: ✅ Success (both same hospital)
   
4. Manager B1 logs in
5. Manager B1 tries to GET available staff
   EXPECTED: ✅ Only S4, S5, S6 (not S1, S2, S3)
   
6. Manager B1 tries to GET shift SH1 (Hospital A)
   Expected: ✅ Error "Shift not found" (data isolation)
   
7. Database check:
   ```bash
   db.nurses.find({ hospitalId: hospitalA_id })
   # Should return: 3 staff
   
   db.nurses.find({ hospitalId: hospitalB_id })
   # Should return: 3 staff
   ```
```

### Test 3: Cross-Hospital Access Prevention

```
1. Manager A logs in
2. Get JWT token (contains hospitalId for Hospital A)
3. Try to access shift from Hospital B
   POST /api/shift/:shiftID_B/assign-staff
   Headers: Authorization: Bearer <hospitalA_token>
   Expected: ❌ 404 "Shift not found" (cannot see other hospital's shifts)
   
4. Try to create staff for Hospital B
   (Not fully preventable via middleware, but staff will have wrong hospitalId)
   
5. Try to remove staff from Hospital B's shift
   POST /api/shift/:shiftID_B/remove-staff
   Expected: ❌ 404 "Shift not found"
```

### Test 4: Staff Registration with hospitalId

```
1. Manager A logs in (gets token with hospitalId_A)
2. Register new staff member S_new
   POST /api/manager/staff/register
   Body: { name: "John", email: "john@hospital.com", ... }
   
3. Query database:
   db.nurses.findById(objectId_of_S_new)
   
   EXPECTED RESULT:
   {
     _id: ObjectId(...),
     hospitalId: hospitalA_id,  ✅ Should be set!
     name: "John",
     email: "john@hospital.com",
     ...
   }
   
   VERIFY:
   - hospitalId is NOT null/undefined ✅
   - hospitalId matches request token hospitalId ✅
```

### Test 5: Available Staff Query

```
1. Manager A logged in (hospitalId_A)
2. Call GET /api/shift/staff/available
   
3. Response should include:
   {
     success: true,
     count: 3,
     staff: [
       {
         _id: objectId,
         name: "Staff1",
         department: "ED",
         role: "NURSE"
       },
       ...
     ]
   }
   
4. Verify ALL returned staff have correct hospitalId:
   db.nurses.findById(staff[0]._id)
   # Should have hospitalId: hospitalA_id
```

---

## Rollback Instructions (If Needed)

If something breaks and you need to revert:

```bash
# Revert all changes
git checkout -- backend/src/routes/manager.route.js
git checkout -- backend/src/controllers/shift.controller.js

# Or manually revert each change by removing:
hospitalId: req.hospitalId
```

---

## Next Steps

### Immediate (Critical Path):
- [x] Fix staff registration to set hospitalId
- [x] Fix getAvailableStaff to filter by hospitalId
- [x] Fix all shift operations to filter by hospitalId
- [ ] **TEST**: Run through all 5 test scenarios above
- [ ] **RESTART SERVERS**: NodeJS backend needs restart

### Short-term (Other Controllers):
- [ ] Apply same hospitalId filtering to: encounter.controller.js
- [ ] Apply same hospitalId filtering to: triage.controller.js
- [ ] Apply same hospitalId filtering to: bed.controller.js
- [ ] Apply same hospitalId filtering to: nurse.controller.js

### Medium-term (Scalability):
- [ ] Create middleware function `filterByHospital()` to avoid repetition
- [ ] Add hospitalId to all new models automatically
- [ ] Create comprehensive hospital isolation test suite
- [ ] Document this in architecture guide

### Long-term (Compliance):
- [ ] Audit all database queries for hospital scoping
- [ ] Create security compliance checklist
- [ ] Document multi-tenant architecture
- [ ] Add hospitalId validation at schema level

---

## Error Messages After Fix

Users will now see these error messages if they try to access data from other hospitals:

```
// When trying to access non-existent shift:
{
  success: false,
  message: "Shift not found"
}

// When trying to assign staff from wrong hospital:
(getAvailableStaff won't return the staff - error)

// When trying to delete shift from wrong hospital:
{
  success: false,
  message: "Shift not found"
}
```

These error messages preserve security (don't reveal data exists elsewhere).

---

## Server Restart Required

**IMPORTANT**: After applying these fixes, **restart your backend server**:

```bash
# Kill existing node process
pkill -f "node src/server.js"

# Start fresh
cd backend
node src/server.js 2>&1
```

The code changes won't take effect until the server restarts.

---

## Verification Commands

```bash
# Check if fixes were applied:
grep -n "hospitalId: req.hospitalId" backend/src/routes/manager.route.js
grep -n "hospitalId: req.hospitalId" backend/src/controllers/shift.controller.js

# Expected: Multiple matches showing filters are in place

# Check database structure:
mongosh  # or mongo
use nursehelpdeskdb
db.nurses.findOne()  # Should show hospitalId field
db.shifts.findOne()  # Should show hospitalId field
```
