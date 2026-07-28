# Login Controllers Breakdown

## 🏥 ADMIN/MANAGER LOGIN

### Controller File
**Location:** `/backend/src/controllers/hospital.controller.js`

### Function Name
```javascript
export const loginManager = async (req, res)
```

### Route
**Endpoint:** `POST /api/hospital/login`
**Route File:** `/backend/src/routes/hospital.route.js`

### Model Used
```javascript
Manager.findOne({ email })  // Search in Manager collection
```

### Complete Flow

```
1. USER SUBMITS
   ↓
   { email: "john.smith@stmarys.hospital", password: "admin123" }

2. VALIDATION
   ↓
   - Check if email & password provided
   - If not → 400 Bad Request

3. DATABASE LOOKUP
   ↓
   const manager = await Manager.findOne({ email }).populate('hospitalId')
   - Searches in MANAGERS collection
   - Populates hospitalId with Hospital details
   - If not found → 401 Invalid credentials

4. PASSWORD VERIFICATION
   ↓
   const isPasswordValid = await bcrypt.compare(password, manager.password)
   - Compare plain password with hashed password in DB
   - If false → 401 Invalid credentials

5. UPDATE LAST LOGIN
   ↓
   manager.lastLogin = new Date()
   await manager.save()

6. CREATE JWT TOKEN
   ↓
   jwt.sign({
     id: manager._id,
     email: manager.email,
     name: manager.name,
     role: manager.role,              // ← ADMIN, MANAGER, etc
     department: manager.department,
     hospitalId: manager.hospitalId._id,
     hospitalName: manager.hospitalId.name,
   }, JWT_SECRET, { expiresIn: '24h' })

7. RETURN RESPONSE
   ↓
   {
     success: true,
     message: "Login successful",
     token: "eyJhbGci...",
     user: {
       id, name, email, role, department,
       hospitalId, hospitalName, permissions
     }
   }
```

### Key Details
- **Collection:** `managers`
- **JWT Expiry:** 24 hours
- **Includes:** permissions object
- **Status Code:** 200 OK
- **Updates:** lastLogin timestamp

---

## 👨‍⚕️ STAFF/NURSE LOGIN

### Controller File
**Location:** `/backend/src/controllers/nurse.controller.js`

### Function Name
```javascript
export const loginNurse = async (req, res)
```

### Route
**Endpoint:** `POST /api/nurses/login`
**Route File:** `/backend/src/routes/nurse.route.js`

### Model Used
```javascript
Nurse.findOne({ email })  // Search in Nurse collection
```

### Complete Flow

```
1. USER SUBMITS
   ↓
   { email: "sarah.johnson@stmarys.hospital", password: "nurse123" }

2. VALIDATION
   ↓
   - Check if email & password provided
   - If not → 400 Bad Request

3. DATABASE LOOKUP
   ↓
   const nurse = await Nurse.findOne({ email })
   - Searches in NURSES collection
   - If not found → 401 Invalid credentials

4. PASSWORD VERIFICATION
   ↓
   const valid = await bcrypt.compare(password, nurse.password)
   - Compare plain password with hashed password in DB
   - If false → 401 Invalid credentials

5. POPULATE HOSPITAL INFO
   ↓
   const populatedNurse = await Nurse.findById(nurse._id).populate('hospitalId')
   - Fetch Hospital details
   - Extract hospitalName and hospitalId

6. CREATE JWT TOKEN
   ↓
   jwt.sign({
     id: nurse._id,
     email: nurse.email,
     name: nurse.name,
     role: nurse.role,                // ← NURSE, PHYSICIAN, SPECIALIST
     department: nurse.department,
     hospitalId: hospitalId,
     hospitalName: hospitalName,
   }, process.env.JWT_SECRET || "supersecretkey", { expiresIn: "1d" })

7. RETURN RESPONSE
   ↓
   {
     success: true,
     message: "Login successful",
     token: "eyJhbGci...",
     user: {
       id, name, surname, email, role, department,
       hospitalId, hospitalName
     }
   }
```

### Key Details
- **Collection:** `nurses`
- **JWT Expiry:** 1 day (same as 24h)
- **Includes:** surname field
- **Status Code:** 200 OK
- **Does NOT update:** timestamp

---

## 📊 SIDE-BY-SIDE COMPARISON

| Feature | Admin Login | Staff Login |
|---------|-------------|-------------|
| **Endpoint** | `/api/hospital/login` | `/api/nurses/login` |
| **Controller File** | `hospital.controller.js` | `nurse.controller.js` |
| **Function Name** | `loginManager` | `loginNurse` |
| **Searches Collection** | `Manager` | `Nurse` |
| **Possible Roles** | ADMIN, MANAGER, SHIFT_MANAGER, DEPARTMENT_MANAGER | NURSE, PHYSICIAN, SPECIALIST |
| **Populates Hospital** | During query | After auth |
| **Updates Timestamp** | ✓ Yes (lastLogin) | ✗ No |
| **Includes Permissions** | ✓ Yes | ✗ No |
| **Includes Surname** | ✗ No | ✓ Yes |
| **JWT Secret Source** | `const JWT_SECRET = process.env.JWT_SECRET \|\| 'supersecretkey'` | `process.env.JWT_SECRET \|\| "supersecretkey"` |
| **Response Structure** | Identical | Identical (NOW) |

---

## 🔐 FRONTEND LOGIN FLOW

**File:** `/frontend/src/Dashboard/Dashboard.jsx`

### handleLogin Function
```javascript
const handleLogin = async ({ email, password }) => {
  try {
    // Try hospital admin login FIRST
    let endpoint = "/api/hospital/login";
    let data;

    try {
      data = await apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify({ email, password })
      });
    } catch (err) {
      // If admin login fails, try staff login SECOND
      endpoint = "/api/nurses/login";
      data = await apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify({ email, password })
      });
    }

    if (data.token) {
      // Store token
      localStorage.setItem("token", data.token);
      
      // Decode token to get user info
      const decoded = decodeJwt(data.token);
      
      // Store role for later checks
      localStorage.setItem("userRole", decoded.role);
      
      // Set user in state
      setNurse({
        id: decoded.id,
        email: decoded.email,
        name: decoded.name || "Staff",
        role: decoded.role
      });

      setView("dashboard");
    }
  } catch (err) {
    alert(err.message);
  }
}
```

### Login Flow Chart
```
User enters email + password
        ↓
TRY admin login (/api/hospital/login)
        ↓
    SUCCESS? Store token, show dashboard
        ↓
    FAILED? Try staff login (/api/nurses/login)
        ↓
    SUCCESS? Store token, show dashboard
        ↓
    FAILED? Show error message
```

---

## 🎯 KEY DIFFERENCE: Why Two Controllers?

### Different User Types = Different Collections
- **Managers** → Store in `Manager` collection
  - Have permissions, department management, full hospital control
  - Login through `/api/hospital/login`
  
- **Staff** → Store in `Nurse` collection
  - Have roles (NURSE, PHYSICIAN), assigned to departments
  - Can clock in/out, request breaks
  - Login through `/api/nurses/login`

### Same User Can't Be Both
- A person is EITHER a Manager OR Nursing Staff
- Frontend tries Manager first (admin login), then Staff (nurse login)
- This allows one login form to work for all users

---

## 🐛 Debug Logs Available

### Admin Login
```
[No specific prefix] - Uses generic error handling
```

### Staff Login
```
[NURSE LOGIN] Attempting login with: email
[NURSE LOGIN] Nurse found: Name (Role: ROLE)
[NURSE LOGIN] Password valid: true/false
[NURSE LOGIN] Token created with role: ROLE
[NURSE LOGIN] ERROR: error message (if error)
```

---

## Routes Summary

### Hospital Routes (`/api/hospital/`)
```
POST /register  → registerHospital()     [Create hospital + admin]
POST /login     → loginManager()         [Admin login]
GET  /debug/managers → Debug endpoint   [Show all managers]
```

### Nurse Routes (`/api/nurses/`)
```
POST /login     → loginNurse()           [Staff login]
POST /clockin   → clockIn()              [Staff clock in]
POST /clockout  → clockOut()             [Staff clock out]
... other staff endpoints
```
