# Staff Scheduling & Clock-In/Out System - API Documentation

## Backend Implementation Complete ✅

All data layer and API endpoints have been implemented and tested. This document describes all available endpoints for the staff scheduling system.

---

## Authentication

All protected endpoints require a Bearer token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:5001/api/...
```

Token is obtained from login endpoint: `POST /api/nurses/login`

---

## Nurse Endpoints

### 1. GET `/api/nurses/available-shifts`
**Protected**: Yes | **Method**: GET

Get scheduled shifts for the authenticated nurse (next 7 days)

**Response**:
```json
{
  "success": true,
  "shifts": [
    {
      "_id": "ObjectId",
      "date": "2026-04-14T00:00:00.000Z",
      "status": "ON_DUTY",
      "shiftStart": "07:00",
      "shiftEnd": "19:00",
      "breakStart": "12:00",
      "breakEnd": "13:00",
      "location": "ED",
      "maxBeds": 5,
      "assignedBeds": ["ED-01", "ED-02"],
      "currentBedCount": 2,
      "licenseType": "RN",
      "specializations": ["Emergency", "Trauma"]
    }
  ],
  "message": "Found 3 scheduled shift(s)"
}
```

**Use Case**: Show nurse their available shifts when they first open the app to select which shift to work.

---

### 2. POST `/api/nurses/break-request`
**Protected**: Yes | **Method**: POST

Request a break (requires manager approval)

**Body**:
```json
{
  "reason": "Lunch break"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Break request submitted. Awaiting manager approval.",
  "clockRecord": {
    "_id": "ObjectId",
    "breakApprovalStatus": "PENDING",
    "breakReason": "Lunch break"
  }
}
```

**Use Case**: Nurse submits break request while clocked in. Manager gets notification and can approve/reject.

---

### 3. POST `/api/nurses/clockin`
**Protected**: Yes | **Method**: POST

Clock in to a shift

**Body**:
```json
{
  "shiftId": "ObjectId",
  "location": "ED"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Clocked in successfully",
  "clockRecord": {
    "_id": "ObjectId",
    "clockInTime": "2026-04-14T07:05:00.000Z",
    "shiftStart": "07:00",
    "shiftEnd": "19:00",
    "location": "ED",
    "status": "CLOCKED_IN",
    "lateArrivalApprovalStatus": "PENDING"  // Only if >15 min late
  }
}
```

**Logic**:
- If nurse clocks in >15 minutes late, `lateArrivalApprovalStatus` is set to `PENDING` (requires manager approval)
- `lateArrivalMinutes` calculated automatically
- System prevents double clock-in (already clocked in today)

---

### 4. POST `/api/nurses/clockout`
**Protected**: Yes | **Method**: POST

Clock out from shift

**Body**:
```json
{
  "notes": "Optional notes about shift"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Clocked out successfully",
  "summary": {
    "clockInTime": "2026-04-14T07:05:00.000Z",
    "clockOutTime": "2026-04-14T19:30:00.000Z",
    "totalShiftDuration": 43500000,  // milliseconds
    "totalBreakTime": 3600000,       // milliseconds
    "totalWorkTime": 39900000        // shift - break
  }
}
```

---

### 5. POST `/api/nurses/break-start`
**Protected**: Yes | **Method**: POST

Start a break (immediate, manual)

**Response**:
```json
{
  "success": true,
  "message": "Break started",
  "breakStartTime": "2026-04-14T12:00:00.000Z"
}
```

**Note**: Used for immediate breaks without approval. For approval-required breaks, use `/break-request` instead.

---

### 6. POST `/api/nurses/break-end`
**Protected**: Yes | **Method**: POST

End a break

**Response**:
```json
{
  "success": true,
  "message": "Break ended",
  "breakDuration": 60  // minutes
}
```

---

### 7. GET `/api/nurses/clockstatus`
**Protected**: Yes | **Method**: GET

Get current clock status for authenticated nurse

**Response**:
```json
{
  "success": true,
  "clockRecord": {
    "_id": "ObjectId",
    "status": "CLOCKED_IN",
    "clockInTime": "2026-04-14T07:05:00.000Z",
    "clockOutTime": null,
    "breakStartTime": null,
    "breakEndTime": null,
    "elapsedTime": 120,  // minutes
    "location": "ED"
  }
}
```

Or if not clocked in:
```json
{
  "success": true,
  "status": "NOT_CLOCKED_IN",
  "message": "Not clocked in today"
}
```

---

### 8. GET `/api/nurses/clockhistory`
**Protected**: Yes | **Method**: GET

Get historical clock records for authenticated nurse

**Query Parameters**:
- `startDate` (optional): YYYY-MM-DD
- `endDate` (optional): YYYY-MM-DD

**Response**:
```json
{
  "success": true,
  "records": [
    {
      "_id": "ObjectId",
      "date": "2026-04-13T00:00:00.000Z",
      "clockInTime": "2026-04-13T07:00:00.000Z",
      "clockOutTime": "2026-04-13T19:30:00.000Z",
      "breakStartTime": "2026-04-13T12:00:00.000Z",
      "breakEndTime": "2026-04-13T13:00:00.000Z",
      "totalShiftDuration": 45000000,
      "totalBreakTime": 3600000,
      "totalWorkTime": 41400000,
      "status": "CLOCKED_OUT",
      "location": "ED",
      "lateArrivalMinutes": 5,
      "lateArrivalApprovalStatus": "APPROVED",
      "breakApprovalStatus": "APPROVED"
    }
  ]
}
```

---

## Manager Approval Endpoints

### 9. POST `/api/nurses/approve-late-arrival/:recordId`
**Protected**: Yes | **Method**: POST

Approve or reject a late arrival

**Body**:
```json
{
  "approved": true,
  "reason": "Optional rejection reason"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Late arrival approved",
  "clockRecord": {
    "_id": "ObjectId",
    "nurseId": "ObjectId",
    "lateArrivalMinutes": 20,
    "lateArrivalApprovalStatus": "APPROVED",
    "lateArrivalApprovedBy": "managerId",
    "lateArrivalApprovalTime": "2026-04-14T07:30:00.000Z"
  }
}
```

---

### 10. POST `/api/nurses/approve-break/:recordId`
**Protected**: Yes | **Method**: POST

Approve or reject a break request

**Body**:
```json
{
  "approved": true,
  "reason": "Optional rejection reason"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Break request approved",
  "clockRecord": {
    "_id": "ObjectId",
    "nurseId": "ObjectId",
    "breakApprovalStatus": "APPROVED",
    "breakApprovedBy": "managerId",
    "breakApprovalTime": "2026-04-14T12:00:30.000Z",
    "breakStartTime": "2026-04-14T12:00:30.000Z",  // Auto-started
    "status": "ON_BREAK"
  }
}
```

---

## Manager Dashboard Endpoints

### 11. GET `/api/manager/staff-status`
**Protected**: Yes | **Method**: GET

Real-time view of all staff status (manager dashboard)

**Query Parameters**:
- `location` (optional): "ED", "ICU", "OR", etc.
- `status` (optional): "CLOCKED_IN", "ON_BREAK", "CLOCKED_OUT"
- `showPendingApprovals` (optional): "true" - only show staff with pending approvals

**Response**:
```json
{
  "success": true,
  "summary": {
    "total": 15,
    "clockedIn": 8,
    "onBreak": 2,
    "clockedOut": 5,
    "pendingApprovals": 3,
    "lateArrivals": 1,
    "pendingBreaks": 2
  },
  "staff": [
    {
      "_id": "ObjectId",
      "nurse": {
        "_id": "ObjectId",
        "name": "Test",
        "surname": "Nurse",
        "email": "testnurse@example.com"
      },
      "clockInTime": "2026-04-14T07:05:00.000Z",
      "clockOutTime": null,
      "status": "CLOCKED_IN",
      "location": "ED",
      "elapsedMinutes": 120,
      "breakStartTime": null,
      "breakEndTime": null,
      "breakDurationMinutes": 0,
      "lateArrivalMinutes": 5,
      "lateArrivalApprovalStatus": "PENDING",
      "breakApprovalStatus": "APPROVED",
      "breakReason": "Lunch",
      "hasPendingApprovals": true
    }
  ],
  "timestamp": "2026-04-14T09:05:00.000Z"
}
```

---

### 12. GET `/api/manager/pending-approvals`
**Protected**: Yes | **Method**: GET

Get all pending break and late arrival approvals

**Response**:
```json
{
  "success": true,
  "count": 3,
  "pending": [
    {
      "_id": "ObjectId",
      "type": "BREAK_REQUEST",
      "nurse": {
        "_id": "ObjectId",
        "name": "Test Nurse",
        "email": "testnurse@example.com"
      },
      "date": "2026-04-14T00:00:00.000Z",
      "detail": {
        "reason": "Lunch break",
        "clockInTime": "2026-04-14T07:05:00.000Z",
        "requestTime": "2026-04-14T12:00:00.000Z"
      }
    },
    {
      "_id": "ObjectId",
      "type": "LATE_ARRIVAL",
      "nurse": {
        "_id": "ObjectId",
        "name": "Other Nurse",
        "email": "othernurse@example.com"
      },
      "date": "2026-04-14T00:00:00.000Z",
      "detail": {
        "minutesLate": 20,
        "reason": "Traffic",
        "clockInTime": "2026-04-14T07:20:00.000Z"
      }
    }
  ]
}
```

---

### 13. POST `/api/manager/bulk-approvals`
**Protected**: Yes | **Method**: POST

Batch approve/reject multiple requests

**Body**:
```json
{
  "approvals": [
    {
      "recordId": "ObjectId",
      "type": "BREAK_REQUEST",
      "approved": true,
      "reason": null
    },
    {
      "recordId": "ObjectId",
      "type": "LATE_ARRIVAL",
      "approved": false,
      "reason": "Excessive tardiness"
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "message": "Processed 2 approvals",
  "results": [
    {
      "recordId": "ObjectId",
      "success": true,
      "type": "BREAK_REQUEST",
      "approved": true,
      "message": "Updated"
    },
    {
      "recordId": "ObjectId",
      "success": true,
      "type": "LATE_ARRIVAL",
      "approved": false,
      "message": "Updated"
    }
  ]
}
```

---

## Data Models

### NurseClockRecord Schema

```javascript
{
  nurseId: ObjectId (ref: Nurse),
  shiftId: ObjectId (ref: NurseSchedule),
  date: Date,
  
  // Clock times
  clockInTime: Date,
  clockOutTime: Date,
  
  // Break tracking
  breakStartTime: Date,
  breakEndTime: Date,
  breakApprovalStatus: "NOT_REQUESTED" | "PENDING" | "APPROVED" | "REJECTED",
  breakApprovedBy: ObjectId,
  breakApprovalTime: Date,
  breakReason: String,
  breakRejectionReason: String,
  
  // Late arrival tracking
  lateArrivalMinutes: Number,
  lateArrivalApprovalStatus: "NOT_REQUIRED" | "PENDING" | "APPROVED" | "REJECTED",
  lateArrivalApprovedBy: ObjectId,
  lateArrivalApprovalTime: Date,
  lateArrivalReason: String,
  lateArrivalRejectionReason: String,
  
  // Status & location
  status: "CLOCKED_IN" | "ON_BREAK" | "CLOCKED_OUT",
  location: "ED" | "ICU" | "OR" | "MATERNITY" | "NICU" | "PEDIATRICS" | "PSYCH" | "GENERAL",
  
  // Calculated metrics
  totalShiftDuration: Number (milliseconds),
  totalBreakTime: Number (milliseconds),
  totalWorkTime: Number (milliseconds),
  
  // Metadata
  metadata: {
    deviceType: String,
    userAgent: String,
    ipAddress: String,
    location: { latitude, longitude }
  },
  
  notes: String,
  createdAt: Date,
  updatedAt: Date
}
```

---

## Workflow Examples

### Nurse Clock-In With Late Arrival Approval

```
1. Nurse calls POST /api/nurses/clockin
   - clockInTime = 07:20 (20 minutes late)
   - System calculates lateArrivalMinutes = 20
   - Since 20 > 15, lateArrivalApprovalStatus = "PENDING"

2. Manager sees in GET /api/manager/pending-approvals
   - Notification shows "Test Nurse - 20 minutes late"

3. Manager calls POST /api/nurses/approve-late-arrival/:recordId
   - Body: { approved: true }
   - Response includes lateArrivalApprovalStatus = "APPROVED"

4. Nurse continues shift normally
```

### Nurse Break Request & Approval

```
1. Nurse calls POST /api/nurses/break-request
   - Body: { reason: "Lunch break" }
   - breakApprovalStatus = "PENDING"

2. Manager sees in GET /api/manager/pending-approvals
   - Notification shows "Test Nurse - Lunch break request"

3. Manager calls POST /api/nurses/approve-break/:recordId
   - Body: { approved: true }
   - System auto-starts break (breakStartTime set, status = "ON_BREAK")

4. Nurse ends break with POST /api/nurses/break-end
   - Status returns to "CLOCKED_IN"

5. At clock-out, breakDuration is calculated
   - totalWorkTime = totalShiftDuration - totalBreakTime
```

### Manager Monitoring Staff

```
1. Manager opens Staff Status Dashboard

2. Dashboard polls GET /api/manager/staff-status every 10 seconds
   - Shows real-time list of all staff
   - Color codes: Green (clocked in), Orange (on break), Gray (clocked out)
   - Red badge for pending approvals

3. Manager sees "1 pending approval" summary
   - Clicks on staff member with pending approval

4. Modal opens showing break/late arrival request

5. Manager clicks "Approve" or "Reject"
   - System calls POST /api/manager/bulk-approvals or individual approval endpoint
   - Dashboard refreshes after ~5 seconds
```

---

## Testing Commands

### Test Login & Get Token
```bash
curl -X POST http://localhost:5001/api/nurses/login \
  -H "Content-Type: application/json" \
  -d '{"email":"testnurse@example.com","password":"password123"}'
```

### Test Available Shifts
```bash
TOKEN="<your-token>" && \
curl -X GET http://localhost:5001/api/nurses/available-shifts \
  -H "Authorization: Bearer $TOKEN"
```

### Test Request Break
```bash
TOKEN="<your-token>" && \
curl -X POST http://localhost:5001/api/nurses/break-request \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"Lunch break"}'
```

### Test Manager Staff Status
```bash
TOKEN="<manager-token>" && \
curl -X GET "http://localhost:5001/api/manager/staff-status?showPendingApprovals=true" \
  -H "Authorization: Bearer $TOKEN"
```

### Test Bulk Approvals
```bash
TOKEN="<manager-token>" && \
curl -X POST http://localhost:5001/api/manager/bulk-approvals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "approvals": [
      {"recordId":"<id>","type":"BREAK_REQUEST","approved":true},
      {"recordId":"<id>","type":"LATE_ARRIVAL","approved":false,"reason":"Too late"}
    ]
  }'
```

---

## Important Notes

1. **Late Arrival Auto-Detection**: Calculated in pre-save hook by comparing clock-in time with scheduled shift start time
2. **Break Approval Auto-Start**: When manager approves a break, `breakStartTime` is automatically set
3. **Data Retention**: All records kept indefinitely (as per requirements)
4. **Status Transitions**: 
   - Clock-in → CLOCKED_IN (or PENDING if late)
   - Request break → breakApprovalStatus = PENDING
   - Approve break → CLOCKED_IN → ON_BREAK (auto-started)
   - End break → CLOCKED_IN
   - Clock-out → CLOCKED_OUT
5. **Metadata**: Device type, IP, and location can be captured for audit purposes
6. **Indexes**: Added for efficient queries on:
   - nurseId + date
   - breakApprovalStatus
   - lateArrivalApprovalStatus
   - Status queries for real-time dashboards

