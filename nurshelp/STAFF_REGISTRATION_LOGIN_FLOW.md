# Staff Registration and Login Flow - Diagnostic Report

## Issue Identified

**Problem**: Staff members registered during hospital setup (Step 2 of registration) could not login with their credentials. Only the admin could login.

**Root Cause**: **Double Password Hashing** - the staff password was being hashed twice:
1. First hash in `manager.route.js` (manual bcrypt.hash)
2. Second hash in Nurse model's pre-save hook

This caused login to fail because the plaintext password was only hashed once, while the database had a double-hashed version.

---

## Complete Staff Registration & Login Flow

### Phase 1: Hospital Registration (Admin Creation)
```
User Form → POST /api/hospital/register → hospital.controller.js
├─ Create Hospital document
│  └─ Save Hospital (gets MongoDB _id)
├─ Create Manager with hospitalId reference
│  ├─ Hash password once: bcrypt.hash(password, 10) ✅
│  └─ Save Manager
└─ Send JWT token with hospitalId to client
```

**Manager Model**: No pre-save hook, so manual hashing is correct ✅

---

### Phase 2: Staff Registration (During Hospital Setup)
```
User Form → POST /api/manager/staff/register → manager.route.js
├─ Extract staff credentials from request body
├─ Hash password: bcrypt.hash(password, 10) ❌ PROBLEM
├─ Create Nurse document with hashed password
└─ Save Nurse
   └─ Nurse model pre-save hook fires
      └─ Hash password AGAIN: bcrypt.hash(passwordAlreadyHashed, 10) ❌ DOUBLE HASH
```

**Result**: 
- Plaintext "test123" → hashed once in manager.route.js
- Already hashed value → hashed again in pre-save hook
- Database stores: double-hashed password ❌

**Nurse Model**: HAS pre-save hook (line 79 of nurse.model.js):
```javascript
nurseSchema.pre("save", async function () {
  if (this.isModified("password")) {
    this.password = await bcrypt.hash(this.password, 10);
  }
});
```

---

### Phase 3: Staff Login attempt
```
Staff Login Form → POST /api/auth/login → auth.controller.js
├─ Search Manager collection - NOT found
├─ Search Nurse collection - FOUND ✓
├─ Compare password:
│  ├─ Plaintext input "test123"
│  ├─ Hash once: bcrypt.hash("test123", 10) → hash1
│  └─ Compare with DB: bcrypt.compare(hash1, doubleHash) → FALSE ❌
└─ Return 401 Invalid credentials
```

---

## The Fix Applied ✅

**File**: [backend/src/routes/manager.route.js](backend/src/routes/manager.route.js#L75-L100)

**Change**: Remove manual password hashing since Nurse model does it automatically.

### Before (Broken):
```javascript
const hashedPassword = await bcrypt.hash(password, 10);  // ❌ Manual hash

const newStaff = new Nurse({
  password: hashedPassword,  // ❌ Pass hashed password
  ...
});
// Nurse pre-save hook hashes again ❌ DOUBLE HASH
```

### After (Fixed):
```javascript
// ✅ Remove manual hashing entirely

const newStaff = new Nurse({
  password,  // ✅ Pass plaintext password
  ...
});
// Nurse pre-save hook hashes once ✅ CORRECT
```

---

## Verification: Password Hashing in Each Model

| Model | Pre-save Hook | Manual Hash in Route | Result |
|-------|---------------|------------------|--------|
| Manager | ❌ None | ✅ Yes (hospital.controller.js) | ✅ Correct - hashed once |
| Nurse | ✅ Yes (line 79) | ❌ Was Yes (manager.route.js) | ❌ Was double-hashed |
| Nurse | ✅ Yes (line 79) | ✅ Removed | ✅ Now correct - hashed once |

---

## Complete Flow After Fix

### Staff Registration:
```
1. User submits form → manager.route.js
2. Plaintext password received: "staffPassword123"
3. Create Nurse object with plaintext password
4. Nurse.save() triggered
5. Pre-save hook: bcrypt.hash("staffPassword123", 10)
6. Store hashed result in database ✅
```

### Staff Login:
```
1. User submits form → auth.controller.js
2. Plaintext password entered: "staffPassword123"
3. Query Nurse collection by email ✓
4. bcrypt.compare("staffPassword123", dbHashedPassword)
5. Returns TRUE ✅
6. Generate JWT token
7. Staff can access dashboard ✅
```

---

## Key Implementation Details

### Nurse Model Pre-save Hook (UNCHANGED - Correct)
**File**: [backend/src/models/nurse.model.js](backend/src/models/nurse.model.js#L79-L83)
```javascript
nurseSchema.pre("save", async function () {
  if (this.isModified("password")) {
    this.password = await bcrypt.hash(this.password, 10);
  }
});
```

✅ Correctly hashes password once, only when password is modified

### Manager Model (CORRECT - No pre-save hook)
Since Manager doesn't have a pre-save hook, manual hashing in hospital.controller.js is required and correct.

### Unified Login Endpoint
**File**: [backend/src/controllers/auth.controller.js](backend/src/controllers/auth.controller.js#L25-L51)
```javascript
// Try Manager first
let user = await Manager.findOne({ email }).populate('hospitalId');

// If not found, try Nurse
if (!user) {
  user = await Nurse.findOne({ email }).populate('hospitalId');
}

// Verify password (works for both now)
const isPasswordValid = await bcrypt.compare(password, user.password);
```

✅ Correctly checks both collections
✅ bcrypt.compare works with single-hashed passwords

---

## Testing Steps

After applying the fix:

1. **Register Hospital**
   - Hospital Name: "Test Hospital"
   - Admin Email: admin@test.com
   - Admin Password: admin123

2. **Add Staff During Setup**
   - Staff Name: John Doe
   - Staff Email: john@test.com
   - Staff Password: john123
   - Click "Add Staff"

3. **Test Admin Login**
   - Email: admin@test.com
   - Password: admin123
   - ✅ Should login successfully

4. **Test Staff Login**
   - Email: john@test.com
   - Password: john123
   - ✅ Should login successfully (NOW FIXED!)

5. **Dashboard Access**
   - Admin sees: Hospital dashboard with staff management
   - Staff sees: Patient triage, encounters, clock-in/out

---

## Files Modified

### backend/src/routes/manager.route.js
- **Lines 75-100**: Removed manual `bcrypt.hash()` call
- **Changed**: `password: hashedPassword` → `password`
- **Impact**: Lets pre-save hook handle password hashing correctly

### No changes needed:
- backend/src/models/nurse.model.js (pre-save hook is correct)
- backend/src/models/manager.model.js (correct approach for managers)
- backend/src/controllers/auth.controller.js (works with single-hashed passwords)
- backend/src/controllers/hospital.controller.js (correct for managers)

---

## Architecture Summary

```
Registration:
  Admin: Pass plaintext → manual hash in controller → store
  Staff: Pass plaintext → auto-hash in pre-save hook → store

Login:
  Both types: bcrypt.compare(plaintext, dbHash) → generates JWT

Expected behavior:
  - Admin can login with admin credentials ✅
  - Staff can login with their credentials ✅ (NOW FIXED)
  - Different hospitals have isolated staff ✅
  - Same email allowed in different hospitals ✅
```

---

## Related Issues & Fixes

1. **Hospital Foreign Key Integrity** ✅
   - Fixed: Hospital saved before Manager to ensure valid reference

2. **Email Uniqueness Per Hospital** ✅
   - Fixed: Changed from global to compound `{hospitalId, email}` unique index

3. **Staff Password Double Hashing** ✅
   - Fixed: Removed manual hash, use model pre-save hook

4. **Required AdminManagerId** ✅
   - Fixed: Changed to `required: false` in Hospital model

---

## Testing Checklist

- [ ] Hospital registration completes successfully
- [ ] Admin can login with their credentials
- [ ] Staff created during registration can login
- [ ] Staff credentials work in subsequent logins
- [ ] Admin dashboard shows registered staff
- [ ] Staff dashboard shows patient encounters
- [ ] Multiple hospitals don't interfere
- [ ] Same email works in different hospitals
- [ ] Passwords are properly hashed (one hash only)

