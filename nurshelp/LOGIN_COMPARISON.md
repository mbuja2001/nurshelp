# Admin vs Staff Login Comparison

## 📊 ADMIN/MANAGER LOGIN
**Endpoint:** `POST /api/hospital/login`
**Controller:** `hospital.controller.js`

### Request
```json
{
  "email": "john.smith@stmarys.hospital",
  "password": "admin123"
}
```

### Response Structure
```javascript
{
  "success": true,                    // ✓ Has success flag
  "message": "Login successful",      // ✓ Has message
  "token": "eyJhbGci...",
  "user": {                           // ✓ Uses 'user' key
    "id": "69de519c3dc2c8866769cd76",
    "name": "John",
    "email": "john.smith@stmarys.hospital",
    "role": "ADMIN",                  // ✓ ADMIN role
    "department": "All",
    "hospitalId": "69de519d3dc2c8866769cd79",
    "hospitalName": "St. Mary's Medical Center",
    "permissions": {...}              // ✓ Includes permissions
  }
}
```

### Token Payload
```javascript
{
  "id": "manager._id",
  "email": "manager.email",
  "name": "manager.name",
  "role": "ADMIN",                    // ✓ Role included
  "department": "All",
  "hospitalId": "hospital._id",
  "hospitalName": "hospital.name",
  "iat": 1776177612,
  "exp": 1776264012
}
```

---

## 👨‍⚕️ STAFF/NURSE LOGIN
**Endpoint:** `POST /api/nurses/login`
**Controller:** `nurse.controller.js`

### Request
```json
{
  "email": "sarah.johnson@stmarys.hospital",
  "password": "nurse123"
}
```

### Response Structure
```javascript
{
  "token": "eyJhbGci...",
  "nurse": {                          // ✗ Uses 'nurse' key (DIFFERENT!)
    "_id": "69de57d366354072ae45ac47",
    "name": "Sarah",
    "surname": "Johnson",
    "email": "sarah.johnson@stmarys.hospital",
    "role": "NURSE",                  // ✓ NURSE role
    "department": "ED",
    "hospitalId": "69de57d366354072ae45ac40",
    "hospitalName": "St. Mary's Medical Ce..."
  }
}
```

### Token Payload
```javascript
{
  "id": "nurse._id",
  "email": "nurse.email",
  "name": "nurse.name",
  "role": "NURSE",                    // ✓ Role included
  "department": "ED",
  "hospitalId": "hospital._id",
  "hospitalName": "hospital.name",
  "iat": 1776179234,
  "exp": 1776265634
}
```

---

## ⚠️ KEY DIFFERENCES

| Feature | Admin Login | Staff Login |
|---------|------------|------------|
| **Response has `success` flag** | ✓ Yes | ✗ No |
| **Response has `message`** | ✓ Yes | ✗ No |
| **User object key** | `user` | `nurse` (inconsistent!) |
| **Includes permissions** | ✓ Yes | ✗ No |
| **Response structure** | Consistent/formal | Minimal |
| **Token includes role** | ✓ Yes | ✓ Yes |
| **Token includes hospitalId** | ✓ Yes | ✓ Yes |

---

## 🔴 ISSUES FOUND

### 1. Response Structure Mismatch
- Admin: Returns `{ success, message, token, user }`
- Staff: Returns `{ token, nurse }`
- **Frontend might expect one or the other**

### 2. Missing Success Flag
- Nurse login doesn't include `success: true`
- Frontend error handling might not work properly

### 3. Inconsistent Object Keys
- Admin uses `user` → frontend expects `data.user`
- Staff uses `nurse` → but frontend might expect `data.user` or `data.nurse`

---

## ✅ SOLUTION

Make nurse login response match manager login response structure:

```javascript
// CURRENT (nurse login)
res.json({
  token,
  nurse: { ... }
})

// SHOULD BE (to match manager login)
res.json({
  success: true,
  message: "Login successful",
  token,
  user: {
    id: nurse._id,
    ...
  }
})
```

This way both logins have identical response structures and the frontend doesn't need to handle different keys.
