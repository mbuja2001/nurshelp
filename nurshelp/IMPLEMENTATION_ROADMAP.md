# Implementation Roadmap - Phase 2 & Beyond

## Phase 1: COMPLETED ✅
- [x] Database models (Hospital, Manager, Nurse, Patient, Physician, Shift, Schedule)
- [x] Hospital registration and admin manager creation
- [x] Manager/admin authentication (login/JWT)
- [x] Multi-tenant architecture with hospitalId isolation
- [x] Email uniqueness per hospital (compound index)
- [x] Documentation and guides

---

## Phase 2: Nurse & Staff Management (NEXT)

### 2.1 Create Nurse Controller
**File**: `backend/src/controllers/nurse.controller.js`

Functions needed:
```javascript
registerNurse(req, res)         // POST /api/nurse/register
getNurses(req, res)             // GET /api/nurse
getNurseById(req, res)          // GET /api/nurse/:id
updateNurse(req, res)           // PUT /api/nurse/:id
deleteNurse(req, res)           // DELETE /api/nurse/:id
getNursesByDepartment(req, res) // GET /api/nurse/department/:dept
getActiveNurses(req, res)       // GET /api/nurse/active
```

Key logic:
- Hash passwords before storing
- Validate license numbers format
- Check certification expiry dates
- Track hire dates and employment status
- Support mass upload from CSV (later phase)

### 2.2 Create Nurse Routes
**File**: `backend/src/routes/nurse.route.js`

```javascript
import express from 'express';
import { auth, authorize } from '../middleware/auth.js';
import * as nurseController from '../controllers/nurse.controller.js';

const router = express.Router();
router.use(auth); // Protect all routes

router.post('/register', authorize(['canAssignStaff']), nurseController.registerNurse);
router.get('/', nurseController.getNurses);
router.get('/:id', nurseController.getNurseById);
router.put('/:id', authorize(['canAssignStaff']), nurseController.updateNurse);
router.delete('/:id', authorize(['canAssignStaff']), nurseController.deleteNurse);
router.get('/department/:dept', nurseController.getNursesByDepartment);
router.get('/active', nurseController.getActiveNurses);

export default router;
```

### 2.3 Create Authentication Middleware
**File**: `backend/src/middleware/auth.js`

Functions needed:
```javascript
auth(req, res, next)                    // Verify JWT, set req.user
authorize(permissions[])(req, res, next) // Check permissions
authorizeRole(roles[])(req, res, next)   // Check role
tenantFilter(req, res, next)             // Auto-add hospitalId filter
```

### 2.4 Test Nurse Management
- Unit tests for nurse controller
- Integration tests with database
- JWT token validation tests
- Multi-tenant isolation tests

---

## Phase 3: Shift & Schedule Management

### 3.1 Create Shift Controller
**File**: `backend/src/controllers/shift.controller.js`

Functions needed:
```javascript
createShift(req, res)           // POST /api/shift
getShifts(req, res)             // GET /api/shift
getShiftById(req, res)          // GET /api/shift/:id
updateShift(req, res)           // PUT /api/shift/:id
deleteShift(req, res)           // DELETE /api/shift/:id
assignNurseToShift(req, res)    // POST /api/shift/:id/assign
removeNurseFromShift(req, res)  // DELETE /api/shift/:id/nurse/:nurseId
getAvailableNurses(req, res)    // GET /api/shift/:id/available
getShiftsByDate(req, res)       // GET /api/shift/date/:date
```

Key logic:
- Validate shift times (start < end)
- Check for overlapping shifts
- Track shift status (SCHEDULED, ONGOING, COMPLETED)
- Prevent assigning inactive nurses
- Calculate shift utilization

### 3.2 Create Schedule Controller
**File**: `backend/src/controllers/schedule.controller.js`

Functions needed:
```javascript
generateWeeklySchedule(req, res)  // POST /api/schedule/generate
getSchedule(req, res)             // GET /api/schedule
updateSchedule(req, res)          // PUT /api/schedule/:id
checkConflicts(req, res)          // GET /api/schedule/conflicts
publishSchedule(req, res)         // POST /api/schedule/:id/publish
```

Key logic:
- Auto-assign nurses based on availability
- Detect scheduling conflicts
- Check staff-to-patient ratios
- Rotate day/evening/night shifts
- Balance workload across nurses

### 3.3 Create Leave Management Controller
**File**: `backend/src/controllers/leave.controller.js`

Functions needed:
```javascript
requestLeave(req, res)          // POST /api/leave/request
getLeaveRequests(req, res)      // GET /api/leave/requests
approveLeaveRequest(req, res)   // PUT /api/leave/:id/approve
rejectLeaveRequest(req, res)    // PUT /api/leave/:id/reject
getNurseLeaveBalance(req, res)  // GET /api/leave/balance/:nurseId
updateLeaveBalance(req, res)    // PUT /api/leave/balance/:nurseId
```

---

## Phase 4: Patient & Clinical Integration

### 4.1 Create Patient Controller
**File**: `backend/src/controllers/patient.controller.js`

Functions needed:
```javascript
registerPatient(req, res)           // POST /api/patient/register
getPatients(req, res)               // GET /api/patient
getPatientById(req, res)            // GET /api/patient/:id
updatePatient(req, res)             // PUT /api/patient/:id
searchPatient(req, res)             // GET /api/patient/search?term=
getPatientsByDepartment(req, res)   // GET /api/patient/dept/:dept
updateMedicalHistory(req, res)      // PUT /api/patient/:id/history
```

### 4.2 Create Triage Controller
**File**: `backend/src/controllers/triage.controller.js`

For emergency department patient prioritization:

Functions needed:
```javascript
createTriageRecord(req, res)    // POST /api/triage
getTriageRecords(req, res)      // GET /api/triage
updateTriageSeverity(req, res)  // PUT /api/triage/:id/severity
calculateTriageScore(req, res)  // POST /api/triage/:id/score
getWaitingPatients(req, res)    // GET /api/triage/waiting
getByUrgency(req, res)          // GET /api/triage/urgency
```

### 4.3 Create Vitals Controller
**File**: `backend/src/controllers/vitals.controller.js`

For tracking patient vital signs:

Functions needed:
```javascript
recordVitals(req, res)                  // POST /api/vitals
getPatientVitals(req, res)              // GET /api/vitals/patient/:id
getLatestVitals(req, res)               // GET /api/vitals/patient/:id/latest
checkAbnormalVitals(req, res)           // GET /api/vitals/abnormal
getTrendData(req, res)                  // GET /api/vitals/:id/trend
```

### 4.4 Create Encounter Controller
**File**: `backend/src/controllers/encounter.controller.js`

For hospital encounters (visits, admissions):

Functions needed:
```javascript
createEncounter(req, res)       // POST /api/encounter
getEncounters(req, res)         // GET /api/encounter
updateEncounter(req, res)       // PUT /api/encounter/:id
closeEncounter(req, res)        // POST /api/encounter/:id/close
getActiveEncounters(req, res)   // GET /api/encounter/active
getPatientEncounters(req, res)  // GET /api/encounter/patient/:id
```

---

## Phase 5: Integration & Advanced Features

### 5.1 Physician & Order Management
```javascript
// Physician Controller
createPhysician(req, res)
getPhysicians(req, res) 
assignPatientToPhysician(req, res)

// Order Controller (Lab, Imaging, etc.)
createOrder(req, res)
getOrders(req, res)
updateOrderStatus(req, res)
trackOrderProgress(req, res)
```

### 5.2 RAG Integration (Clinical Decision Support)
```javascript
// RAG retrieval for clinical data
async function getClinicalGuidance(req, res) {
  // Query RAG system in ML backend
  // Return relevant clinical guideline
}

// Integrate with python backend ML service
const mlServiceURL = 'http://localhost:8000/api/rag';
```

### 5.3 Real-time Notifications
```javascript
// WebSocket or Socket.IO for:
// - Staff availability changes
// - New admissions
// - Critical alerts
// - Shift assignments
```

### 5.4 Reports & Analytics
```javascript
// Reports Controller
getAttendanceReport(req, res)
getStaffUtilization(req, res)
getPatientOutcomes(req, res)
getPayrollData(req, res)
getERMetrics(req, res)
```

---

## Tech Stack Required

### Backend
- [x] Node.js + Express
- [x] MongoDB + Mongoose
- [x] JWT for authentication
- [x] bcrypt for passwords
- [ ] Socket.IO (Phase 5)
- [ ] Bull queue (Phase 5 - async jobs)

### Frontend (React/Vite)
- [ ] React Context for state management
- [ ] Axios for API requests
- [ ] React Router for navigation
- [ ] Tailwind CSS for styling
- [ ] Chart.js for analytics
- [ ] React Table for data grids
- [ ] Socket.IO client (Phase 5)

### ML Backend (Python)
- [x] Already implemented in ML/
- [ ] API integration needed (Phase 4)
- [ ] RAG system setup
- [ ] Model deployment (GGUF, FAISS)

---

## Implementation Checklist

### Phase 2 Tasks
- [ ] 2.1.1 Create nurse.controller.js
- [ ] 2.1.2 Implement registerNurse()
- [ ] 2.1.3 Implement getNurses() with filtering
- [ ] 2.1.4 Implement updateNurse() with validation
- [ ] 2.2.1 Create nurse.route.js
- [ ] 2.2.2 Wire up all nurse endpoints
- [ ] 2.3.1 Create auth.js middleware
- [ ] 2.3.2 Implement auth() middleware
- [ ] 2.3.3 Implement authorize() middleware
- [ ] 2.4.1 Write unit tests for nurse controller
- [ ] 2.4.2 Write integration tests
- [ ] 2.4.3 Test multi-tenant isolation
- [ ] Code review & merge to main

### Phase 3 Tasks
- [ ] 3.1.1 Create shift.controller.js
- [ ] 3.1.2 Implement shift management functions
- [ ] 3.2.1 Create schedule.controller.js
- [ ] 3.2.2 Implement schedule generation algorithm
- [ ] 3.3.1 Create leave.controller.js
- [ ] 3.3.2 Implement leave request workflow
- [ ] Write tests for Phase 3

### Phase 4 Tasks
- [ ] 4.1 Create patient.controller.js
- [ ] 4.2 Create triage.controller.js
- [ ] 4.3 Create vitals.controller.js
- [ ] 4.4 Create encounter.controller.js
- [ ] Integrate with ML backend (RAG)
- [ ] Write tests for clinical features

### Phase 5 Tasks
- [ ] 5.1 Physician and ordering system
- [ ] 5.2 WebSocket real-time updates
- [ ] 5.3 Advanced reporting
- [ ] Performance optimization
- [ ] Load testing & scaling

---

## Time Estimates

| Phase | Feature | Est. Hours | Priority |
|-------|---------|-----------|----------|
| 1 | Database & Auth | 8h | HIGH ✅ |
| 2 | Nurse Management | 12h | HIGH |
| 3 | Shift & Scheduling | 20h | HIGH |
| 4 | Patient & Clinical | 16h | MEDIUM |
| 5 | Integration & Real-time | 24h | MEDIUM |
| Testing | Unit & Integration | 20h | HIGH |
| DevOps | Docker, Deployment | 12h | MEDIUM |
| **Total** | | **112h** | |

---

## Key Success Criteria

✅ **Phase 1 (Current)**
- Hospital can register with one admin
- Admin can login with JWT token
- Multi-tenant isolation in place
- Security best practices implemented

🎯 **Phase 2 (Next)**
- Admins can manage staff roster
- Nurses can be assigned to hospital
- Staff availability tracked
- Permission-based access working

🎯 **Phase 3 (Key)**
- Managers can create shifts
- Automated schedule generation
- Conflict detection working
- Admin approval workflow

🎯 **Phase 4 (Clinical)**
- Patient records managed
- Triage scoring in ED
- Clinical data linked
- RAG lookup working

🎯 **Phase 5 (Final)**
- Real-time notifications
- Comprehensive reporting
- Performance optimized
- Production ready

---

## Notes for Developers

1. **Always filter by hospitalId** - Every query must:
   ```javascript
   { hospitalId: req.user.hospitalId, ... }
   ```

2. **Test multi-tenant isolation** - When implementing endpoints, test:
   - Manager from Hospital A cannot see Hospital B's data
   - Same email can exist in different hospitals
   - IDs are globally unique across hospitals

3. **Error messages** - Be clear but don't expose internal details:
   ```javascript
   // Good
   { success: false, message: 'Nurse not found' }
   
   // Bad
   { error: 'MongoError: no document with id xyz' }
   ```

4. **Maintain backward compatibility** - When updating models:
   - Don't remove fields (mark as deprecated)
   - Add new fields as optional initially
   - Plan migrations for data updates

5. **Performance** - Use indexes for:
   - Frequently queried fields
   - Foreign keys (hospitalId)
   - Date ranges
   - Status fields

---

## Questions & Support

For implementation questions:
- Check MULTI_TENANT_IMPLEMENTATION.md for architecture
- Check QUICK_REFERENCE.md for code patterns
- Review phase 1 implementation for patterns
- Ask team for clarification on requirements

