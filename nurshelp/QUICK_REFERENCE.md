# Quick Reference - Common Operations

## Database Queries Pattern

### ✅ CORRECT - With Hospital Isolation

```javascript
// Find all nurses in a hospital
const nurses = await Nurse.find({ 
  hospitalId: req.user.hospitalId 
});

// Find specific nurse - verify hospital
const nurse = await Nurse.findOne({
  _id: nurseId,
  hospitalId: req.user.hospitalId  // Essential!
});

// Update nurse - only if in same hospital
const nurse = await Nurse.findOneAndUpdate(
  {
    _id: nurseId,
    hospitalId: req.user.hospitalId
  },
  { $set: { isActive: false } },
  { new: true }
);

// Count nurses in hospital
const staffCount = await Nurse.countDocuments({
  hospitalId: req.user.hospitalId
});
```

### ❌ WRONG - Missing Hospital Filter

```javascript
// SECURITY RISK - Could expose all nurses from all hospitals
const nurses = await Nurse.find({});

// SECURITY RISK - Could access nurses from other hospitals
const nurse = await Nurse.findById(nurseId);

// SECURITY RISK - Could delete nurses from other hospitals
await Nurse.deleteOne({ _id: nurseId });
```

---

## Controller Pattern for Phase 2+

```javascript
import { Nurse } from '../models/nurse.model.js';

/**
 * Register new nurse to hospital
 */
export const registerNurse = async (req, res) => {
  try {
    const { name, surname, email, phone, role, specialization, department } = req.body;
    
    // Validate input
    if (!name || !email || !role) {
      return res.status(400).json({
        success: false,
        message: 'Missing required fields'
      });
    }

    // Check email uniqueness IN THIS HOSPITAL only
    const existingNurse = await Nurse.findOne({
      hospitalId: req.user.hospitalId,  // Critical: Check in hospital context
      email: email.toLowerCase()
    });

    if (existingNurse) {
      return res.status(400).json({
        success: false,
        message: 'Email already registered in this hospital'
      });
    }

    // Create nurse
    const nurse = new Nurse({
      hospitalId: req.user.hospitalId,  // Auto-assign from JWT
      staffID: `NURSE-${Date.now()}`,
      name,
      surname,
      email: email.toLowerCase(),
      phone,
      role,
      specialization,
      department,
      isActive: true
    });

    await nurse.save();

    res.status(201).json({
      success: true,
      message: 'Nurse registered successfully',
      nurse: {
        id: nurse._id,
        staffID: nurse.staffID,
        name: nurse.name,
        email: nurse.email,
        role: nurse.role
      }
    });

  } catch (err) {
    console.error('Registration error:', err);
    res.status(500).json({
      success: false,
      message: 'Registration failed',
      error: err.message
    });
  }
};

/**
 * Get all nurses in hospital
 */
export const getNurses = async (req, res) => {
  try {
    const nurses = await Nurse.find({
      hospitalId: req.user.hospitalId  // Filter by hospital
    })
    .select('-password')  // Remove sensitive fields
    .sort({ createdAt: -1 });

    res.status(200).json({
      success: true,
      count: nurses.length,
      nurses
    });

  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message
    });
  }
};

/**
 * Update nurse (only their hospital's nurse)
 */
export const updateNurse = async (req, res) => {
  try {
    const { id } = req.params;
    const updates = req.body;

    const nurse = await Nurse.findOneAndUpdate(
      {
        _id: id,
        hospitalId: req.user.hospitalId  // Ensure same hospital
      },
      { $set: updates },
      { new: true, runValidators: true }
    );

    if (!nurse) {
      return res.status(404).json({
        success: false,
        message: 'Nurse not found'
      });
    }

    res.status(200).json({
      success: true,
      nurse
    });

  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message
    });
  }
};
```

---

## Routes Pattern for Phase 2+

```javascript
import express from 'express';
import { 
  registerNurse, 
  getNurses, 
  updateNurse 
} from '../controllers/nurse.controller.js';
import { auth, authorize } from '../middleware/auth.js';  // TBD

const router = express.Router();

// All routes protected - require valid JWT
router.use(auth);  // Middleware ensures req.user.hospitalId is set

/**
 * POST /api/nurse/register
 * Register new nurse to hospital
 * Permission: MANAGER or ADMIN
 */
router.post('/register',
  authorize(['canAssignStaff']),  // Check permission
  registerNurse
);

/**
 * GET /api/nurse
 * Get all nurses in hospital
 * Permission: Any authenticated user
 */
router.get('/', getNurses);

/**
 * PUT /api/nurse/:id
 * Update nurse details
 * Permission: MANAGER or ADMIN
 */
router.put('/:id',
  authorize(['canAssignStaff']),
  updateNurse
);

export default router;
```

---

## Middleware - Authentication (To Be Implemented)

```javascript
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'supersecretkey';

/**
 * Authentication Middleware
 * Verifies JWT and sets req.user with decoded data
 * Includes hospitalId from token
 */
export const auth = (req, res, next) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];

    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    const decoded = jwt.verify(token, JWT_SECRET);
    
    // Key: decoded includes hospitalId
    req.user = {
      id: decoded.id,
      email: decoded.email,
      name: decoded.name,
      role: decoded.role,
      department: decoded.department,
      hospitalId: decoded.hospitalId,        // ← Multi-tenant context
      hospitalName: decoded.hospitalName,
      permissions: decoded.permissions
    };

    next();

  } catch (err) {
    res.status(401).json({
      success: false,
      message: 'Invalid token',
      error: err.message
    });
  }
};

/**
 * Authorization Middleware
 * Checks if user has required permissions
 */
export const authorize = (requiredPermissions) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
    }

    const hasPermission = requiredPermissions.some(
      perm => req.user.permissions[perm] === true
    );

    if (!hasPermission) {
      return res.status(403).json({
        success: false,
        message: 'Insufficient permissions'
      });
    }

    next();
  };
};

/**
 * Optional: Role-based authorization
 */
export const authorizeRole = (allowedRoles) => {
  return (req, res, next) => {
    if (!allowedRoles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        message: 'Insufficient role'
      });
    }
    next();
  };
};
```

---

## Frontend - Storing & Using JWT

```javascript
// After successful login/registration
const response = await fetch('http://localhost:5000/api/hospital/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password })
});

const data = await response.json();

if (data.success) {
  // Store token
  localStorage.setItem('token', data.token);
  localStorage.setItem('user', JSON.stringify(data.user));
  
  // Store hospital context
  localStorage.setItem('hospitalId', data.user.hospitalId);
  localStorage.setItem('hospitalName', data.user.hospitalName);
}

// Using token in subsequent requests
const response = await fetch('http://localhost:5000/api/nurse', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`,
    'Content-Type': 'application/json'
  }
});
```

---

## Testing New Endpoints

### 1. Register Hospital
```bash
curl -X POST http://localhost:5000/api/hospital/register \
  -H "Content-Type: application/json" \
  -d '{
    "hospitalName": "General Hospital",
    "managerName": "Jane",
    "managerSurname": "Doe",
    "managerEmail": "jane@generalhospital.com",
    "managerPassword": "SecurePass123!",
    "department": "All"
  }'
```

### 2. Login Manager
```bash
curl -X POST http://localhost:5000/api/hospital/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "jane@generalhospital.com",
    "password": "SecurePass123!"
  }'
```

Save the token from response.

### 3. Register Nurse (Phase 2)
```bash
TOKEN="<token_from_login>"

curl -X POST http://localhost:5000/api/nurse/register \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice",
    "surname": "Smith",
    "email": "alice@generalhospital.com",
    "phone": "+1-555-0100",
    "role": "RN",
    "specialization": "Emergency Medicine",
    "department": "ED"
  }'
```

### 4. Get All Nurses (Phase 2)
```bash
TOKEN="<token_from_login>"

curl -X GET http://localhost:5000/api/nurse \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

---

## Error Handling Patterns

```javascript
// Input validation
if (!email || !password) {
  return res.status(400).json({
    success: false,
    message: 'Validation error',
    errors: {
      email: !email ? 'Email is required' : null,
      password: !password ? 'Password is required' : null
    }
  });
}

// Duplicate check
const exists = await Manager.findOne({
  hospitalId: req.user.hospitalId,
  email: email.toLowerCase()
});
if (exists) {
  return res.status(409).json({  // 409 Conflict
    success: false,
    message: 'Email already registered'
  });
}

// Not found
const nurse = await Nurse.findOne({
  _id: id,
  hospitalId: req.user.hospitalId
});
if (!nurse) {
  return res.status(404).json({
    success: false,
    message: 'Nurse not found'
  });
}

// Server error
try {
  // ...
} catch (err) {
  console.error('Operation error:', err);
  return res.status(500).json({
    success: false,
    message: 'Operation failed',
    error: process.env.NODE_ENV === 'development' ? err.message : 'Internal error'
  });
}
```

---

## Useful MongoDB Commands

```javascript
// Connect to mongosh
mongosh

// Use database
use nursehelpdesk

// View all collections
show collections

// Find all hospitals
db.hospitals.find()

// Find all managers
db.managers.find()

// Check indexes
db.managers.getIndexes()

// Count documents
db.nurses.countDocuments({ hospitalId: ObjectId("...") })

// Delete test data
db.managers.deleteMany({ email: "test@example.com" })

// Export data
mongoexport --db nursehelpdesk --collection hospitals --out hospitals.json
```

