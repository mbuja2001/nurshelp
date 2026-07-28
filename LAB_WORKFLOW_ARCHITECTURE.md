# Lab Workflow Architecture
## Complete Specialty-Based Care Coordination System

### Overview
This document defines the complete lab request workflow connecting Physicians → Specialty Services (Radiology, Pathology, Pharmacist) → Back to Physicians.

---

## 1. DATA MODEL EXTENSIONS

### 1.1 Encounter.model.js - New lab_requests Array
```javascript
lab_requests: [
  {
    request_id: ObjectId (auto-generated),
    specialty: ENUM ['RADIOLOGY', 'PATHOLOGY'],
    lab_type: String, // e.g., "Chest X-ray", "CT Scan", "Blood Work", "Urinalysis"
    description: String, // Indication for the test
    priority: ENUM ['URGENT', 'HIGH', 'MEDIUM', 'LOW'],
    requested_by: String, // Physician name
    requested_at: Date,
    
    // Acknowledgment stage
    assigned_technician: String, // Radiologist/Pathologist who took the request
    assigned_at: Date,
    assigned_status: ENUM ['PENDING', 'ASSIGNED', 'IN_PROGRESS', 'COMPLETED', 'DECLINED'],
    
    // Report/Result stage
    report_content: String, // Findings, observations
    report_images: [String], // URLs to uploaded images/documents
    clinical_impression: String, // Radiologist/Pathologist conclusion
    recommendations: [String], // Follow-up recommendations
    submitted_at: Date,
    submitted_by: String, // Who submitted the report
    
    // Status tracking
    status: ENUM ['PENDING', 'ACKNOWLEDGED', 'IN_PROGRESS', 'COMPLETED', 'REVIEWED'],
    
    // Physician review
    reviewed_at: Date,
    reviewed_by: String, // Physician who reviewed
    review_notes: String,
    
    // Metadata
    created_at: Date,
    updated_at: Date
  }
]
```

---

## 2. WORKFLOW STAGES

### 2.1 Stage 1: Physician Requests Lab
```
Physician Dashboard → "Request Labs" button
  ↓
  Opens Lab Request Modal
  - Select specialty: RADIOLOGY or PATHOLOGY
  - Select lab type: "Chest X-ray", "CT Scan", "Blood Work", etc.
  - Enter indication/description
  - Select priority: URGENT/HIGH/MEDIUM/LOW
  ↓
  POST /api/encounters/:id/lab-request
  {
    specialty: "RADIOLOGY",
    lab_type: "Chest X-ray",
    description: "Follow-up for pneumonia assessment",
    priority: "HIGH"
  }
  ↓
  Status: "PENDING" → Assigned to queue
```

### 2.2 Stage 2: Radiologist/Pathologist Dashboard
```
Specialty Dashboard (Radiology/Pathology)
  ↓
  Shows "Pending Lab Requests" queue (status = PENDING)
  - Patient name, age, indication
  - Priority badge (URGENT/HIGH/MEDIUM/LOW)
  - Ordered by priority and time
  ↓
  Radiologist clicks "Accept Lab Request"
  PUT /api/encounters/:id/lab-request/:requestId
  { assigned_status: "ASSIGNED", assigned_technician: "Dr. Smith" }
  ↓
  Status changes: "PENDING" → "ASSIGNED"
  - Shows in "In Progress" section
  - Technician can update progress
```

### 2.3 Stage 3: Performing the Lab
```
In Radiology/Pathology Lab
  ↓
  Technician documents:
  - Start performing exam/test
  - Status: "IN_PROGRESS"
  ↓
  Technician completes exam
  - Uploads images/documents
  - Writes findings
  - Status: "COMPLETED" (but not submitted to physician yet)
  ↓
  Radiologist/Pathologist reviews internal findings
  - Writes clinical impression
  - Adds recommendations
  - Status: "COMPLETED"
```

### 2.4 Stage 4: Submit Lab Report
```
Report Submission
  ↓
  PATCH /api/encounters/:id/lab-request/:requestId
  {
    report_content: "Multiple findings noted...",
    report_images: ["url1", "url2"],
    clinical_impression: "Consistent with...",
    recommendations: ["Follow-up CT in 3 months"],
    status: "COMPLETED",
    submitted_by: "Dr. Radiologist"
  }
  ↓
  Status: "COMPLETED" → Now appears in Physician's dashboard as "New Lab Report"
  - Physician gets notification
```

### 2.5 Stage 5: Physician Reviews Lab Report
```
Physician Dashboard
  ↓
  Shows "Lab Reports Awaiting Review" section
  - New reports from radiology/pathology
  - Click to view full report
  ↓
  Physician reviews findings and recommendations
  ↓
  Physician can:
  - Acknowledge report
  - Add clinical notes about report
  - Request follow-up tests
  - Update patient care plan based on findings
  ↓
  PATCH /api/encounters/:id/lab-request/:requestId
  { 
    status: "REVIEWED",
    reviewed_at: Date,
    reviewed_by: "Dr. Physician",
    review_notes: "Findings confirm diagnosis, starting treatment protocol X"
  }
```

---

## 3. BACKEND ENDPOINTS

### 3.1 Physician Endpoints

#### POST /api/encounters/:id/lab-request
**Create new lab request**
```javascript
// Request body
{
  specialty: "RADIOLOGY" | "PATHOLOGY",
  lab_type: "Chest X-ray",
  description: "Post-op imaging",
  priority: "HIGH"
}

// Response
{
  request_id: "...",
  status: "PENDING",
  created_at: "..."
}
```

#### GET /api/encounters/:id/lab-requests
**Get all lab requests for an encounter**
```javascript
// Response
[
  {
    request_id: "...",
    specialty: "RADIOLOGY",
    lab_type: "Chest X-ray",
    status: "PENDING",
    requested_at: "...",
    ...
  },
  ...
]
```

### 3.2 Radiologist/Pathologist Endpoints

#### GET /api/lab-requests?specialty=RADIOLOGY&status=PENDING
**Get pending lab requests for specialty**
```javascript
// Response - list of encounters with pending requests
[
  {
    encounterId: "...",
    patientName: "John Doe",
    patientAge: 45,
    specialty: "RADIOLOGY",
    lab_type: "Chest X-ray",
    description: "...",
    priority: "HIGH",
    requested_at: "...",
    ...
  },
  ...
]
```

#### PATCH /api/encounters/:id/lab-request/:requestId/acknowledge
**Acknowledge & assign lab request**
```javascript
// Request body
{
  status: "ASSIGNED",
  assigned_technician: "Dr. Smith"
}

// Response - updated lab request
```

#### PATCH /api/encounters/:id/lab-request/:requestId/submit-report
**Submit completed lab report**
```javascript
// Request body
{
  report_content: "Findings...",
  report_images: ["url1", "url2"],
  clinical_impression: "...",
  recommendations: ["Follow-up recommended"],
  status: "COMPLETED",
  submitted_by: "Dr. Radiologist"
}

// Response - updated lab request
```

### 3.3 Physician Review Endpoint

#### PATCH /api/encounters/:id/lab-request/:requestId/review
**Physician reviews completed lab report**
```javascript
// Request body
{
  status: "REVIEWED",
  reviewed_by: "Dr. Physician",
  review_notes: "Patient will proceed with treatment A"
}

// Response - updated lab request
```

---

## 4. FRONTEND COMPONENTS

### 4.1 RadiologyDashboard.jsx
**Location:** `nurshelp/frontend/src/components/dashboards/RadiologyDashboard.jsx`

**Sections:**
- Header: "🔬 RADIOLOGY DEPARTMENT"
- Left Sidebar: Current shift info, team statistics
- Main Content:
  - Tab 1: "Pending Orders" (status = PENDING)
    - Card grid showing all pending radiology requests
    - Priority badges, patient info
    - Action button: "Accept Order"
  - Tab 2: "In Progress" (status = ASSIGNED)
    - Shows labs currently being performed
    - Action button: "View Details" / "Update Progress"
  - Tab 3: "Completed" (status = COMPLETED, before submission)
    - Shows completed exams ready for report writing
    - Action button: "Write & Submit Report"
  - Tab 4: "Submitted" (status = SUBMITTED/REVIEWED)
    - Shows submitted reports and their acknowledgment status
- Right Panel: Current selected lab details with full patient context

**Key Features:**
- Queue management with priority sorting
- Real-time updates when new orders arrive
- Report submission modal
- Image upload capability
- Timestamp tracking for SLA compliance

### 4.2 PathologyDashboard.jsx
**Location:** `nurshelp/frontend/src/components/dashboards/PathologyDashboard.jsx`

**Sections:** Same structure as RadiologyDashboard but for pathology requests
- Pending blood work, urinalysis, cultures, biopsies
- Result entry instead of image upload
- Similar workflow

### 4.3 PharmacistDashboard.jsx
**Location:** `nurshelp/frontend/src/components/dashboards/PharmacistDashboard.jsx`

**Sections:**
- Header: "💊 PHARMACY DEPARTMENT"
- Tab 1: "Pending Medication Reviews" (medication_requests status = PENDING)
  - Shows medication orders from physicians awaiting approval
  - Drug interaction checking
  - Action: Approve or Decline with reason
- Tab 2: "Approved & Dispensed" (status = APPROVED/DISPENSED)
- Tab 3: "Medication Administration Tracking"
  - Shows when approved meds were administered by nurses
  - Tracks compliance with orders

---

## 5. UI/UX DESIGN PATTERNS (Matching PhysicianDashboard)

### 5.1 Color Scheme
- Radiology: Blue/Cyan (#3b82f6, #06b6d4)
- Pathology: Purple/Magenta (#a855f7, #ec4899)
- Pharmacy: Green/Emerald (#10b981, #059669)

### 5.2 Card Components
- Pending card: Status badge "PENDING", priority indicator
- In-progress card: Progress bar, assigned staff name
- Completed card: Status badge "READY FOR REVIEW", submit button
- Submitted card: Status badge "SUBMITTED", physician review status

### 5.3 Buttons
- "Accept Order" - converts to assigned, opens detail view
- "Update Progress" - modal for entering findings
- "Write & Submit Report" - opens report composition modal
- "Review Report" (for physician) - view report, acknowledge, add notes

---

## 6. NOTIFICATION SYSTEM

### 6.1 Lab Order Created
- Radiologist/Pathologist receives notification
- Alert in their dashboard queue
- Email/SMS to assigned technician (if exists)

### 6.2 Lab Report Submitted
- Physician receives notification
- "New Lab Reports" badge on dashboard
- Alert panel showing which labs are ready

### 6.3 Lab Report Reviewed
- Radiologist/Pathologist sees "Acknowledged" status
- Completes the handoff cycle

---

## 7. IMPLEMENTATION SEQUENCE

### Phase 1: Data Model & Endpoints (Week 1)
- [ ] Update encounter.model.js with lab_requests schema
- [ ] Create lab.controller.js with all CRUD operations
- [ ] Create lab.routes.js with all endpoints
- [ ] Test endpoints with Postman

### Phase 2: Radiologist Dashboard (Week 2)
- [ ] Create RadiologyDashboard.jsx component
- [ ] Implement queue fetching and display
- [ ] Implement accept/acknowledge workflow
- [ ] Implement report submission modal
- [ ] Test with sample data

### Phase 3: Pathology Dashboard (Week 2-3)
- [ ] Create PathologyDashboard.jsx (similar to Radiology)
- [ ] Test pathology-specific workflows

### Phase 4: Pharmacist Dashboard (Week 3)
- [ ] Create PharmacistDashboard.jsx
- [ ] Integrate with existing medication_requests
- [ ] Test medication review workflow

### Phase 5: Integration & Testing (Week 4)
- [ ] Update PhysicianDashboard with lab request modal
- [ ] Update PhysicianDashboard to show "Lab Reports Awaiting Review"
- [ ] Add notification system
- [ ] End-to-end testing of complete workflow
- [ ] Performance optimization

---

## 8. DATABASE MIGRATION

Before implementing, run:
```javascript
// Add lab_requests to existing encounters
db.encounters.updateMany(
  {},
  { $set: { lab_requests: [] } }
);
```

---

## 9. SECURITY & ACCESS CONTROL

- Radiologist can only view and modify RADIOLOGY requests
- Pathologist can only view and modify PATHOLOGY requests
- Pharmacist can only view and modify medication_requests
- Physicians can view all lab_requests, medication_requests for their encounters
- Cross-role access requires explicit permission

---

## 10. AUDIT TRAIL

Every lab_request tracks:
- Creation: who requested, when, why
- Acknowledgment: who accepted, when
- Work: who performed, timeline
- Completion: who submitted, what findings
- Review: who reviewed, notes, timestamp

All changes logged in encounter.status_history
