# Staff Scheduling System - Frontend Component Roadmap

## Overview
Backend APIs are complete and tested. Frontend components are ready to be built to consume these APIs. This document outlines the 5 components needed for a complete staff scheduling system.

---

## Component 1: ShiftSelectionModal

**Location**: `frontend/src/Dashboard/ShiftSelectionModal.jsx`

**Purpose**: Display available shifts for the nurse and allow them to select one to clock into

**Features**:
- List all available shifts for today and next 7 days
- Show shift time (07:00-19:00), location (ED, ICU, etc.), breaks (12:00-13:00)
- Highlight today's shifts at the top
- Show if shift is already started/ended
- Click to select → opens clock-in flow
- Cancel button to close modal

**API Call**: `GET /api/nurses/available-shifts`

**Response Data Used**:
- `shifts[].date` - Format as date picker
- `shifts[].shiftStart` / `shifts[].shiftEnd` - Display times
- `shifts[].breakStart` / `shifts[].breakEnd` - Show break window
- `shifts[].location` - Display department
- `shifts[].assignedBeds` - Show bed assignments
- `shifts[].licenseType` / `shifts[].specializations` - Show qualifications

**Interaction Flow**:
```
Nurse clicks "Select Shift" 
→ Modal opens with shift list
→ Today's shifts highlighted in green
→ Nurse clicks shift
→ Modal shows "Confirm Selection" with shift details
→ Nurse clicks "Clock In"
→ Calls POST /api/nurses/clockin with shiftId
→ Modal closes, shows success toast
```

**State Management**:
- `selectedShift` - Currently selected shift
- `showModal` - Modal visibility
- `loading` - API call in progress
- `error` - Error message if API fails
- `shifts` - List of available shifts

**Styling**:
- Card layout for each shift
- Color code: Green (today), Blue (upcoming)
- Badge for "Started" or "Ended" shifts
- Hover effect on cards
- Large "Clock In" button on confirmation

---

## Component 2: ClockInWithApprovals

**Location**: `frontend/src/Dashboard/ClockInWithApprovals.jsx`

**Purpose**: Enhanced clock-in/out component with approval status display and break request UI

**Features**:
- Large "CLOCK IN" / "CLOCK OUT" button based on status
- Shows current shift time, elapsed time
- Break request button (requires approval)
- Display pending approval status with badge
- Show late arrival notification if >15 min late
- Auto-refresh clock status every 5 seconds
- Notes field for clock-out

**API Calls**:
- `POST /api/nurses/clockin` - Clock in
- `POST /api/nurses/clockout` - Clock out  
- `GET /api/nurses/clockstatus` - Check current status
- `POST /api/nurses/break-request` - Request break

**Response Data Used**:
- `clockRecord.status` - Determine button state (CLOCKED_IN/CLOCKED_OUT/ON_BREAK)
- `clockRecord.lateArrivalApprovalStatus` - Show pending badge
- `clockRecord.lateArrivalMinutes` - Show how many minutes late
- `clockRecord.breakApprovalStatus` - Show approved/pending/rejected
- `elapsedTime` - Real-time elapsed display

**Interaction Flow**:
```
Clock In:
1. Nurse clicks "CLOCK IN"
2. Selects shift from modal (ShiftSelectionModal)
3. System submits POST /api/nurses/clockin
4. If >15 min late: Badge shows "Pending Manager Approval"
5. Button changes to "CLOCK OUT" / "REQUEST BREAK"

Request Break:
1. Nurse clicks "REQUEST BREAK"
2. Modal opens with reason input
3. Submits POST /api/nurses/break-request
4. Badge shows "Break Pending Approval"
5. Button disabled until manager approves

Clock Out:
1. Nurse clicks "CLOCK OUT"
2. Optional notes dialog
3. Submits POST /api/nurses/clockout
4. Shows summary: Hours worked, Break time, Total time
```

**State Management**:
- `clockStatus` - Current clock record
- `elapsedTime` - Elapsed minutes (updated every second)
- `showNotes` - Notes dialog visibility
- `showBreakModal` - Break request modal visibility
- `selectedShift` - For clock-in flow
- `loading` - API call in progress

**Styling**:
- Large circular buttons (80px diameter)
- Green for CLOCK IN, Red for CLOCK OUT
- Yellow badge for pending approvals
- Real-time elapsed time display
- Toast notifications for success/errors

---

## Component 3: BreakRequestModal

**Location**: `frontend/src/Dashboard/BreakRequestModal.jsx`

**Purpose**: Modal for nurses to request breaks with reason

**Features**:
- Text area for break reason (required)
- Estimated break duration (pre-filled from shift)
- Show current status (approved/pending/rejected)
- Submit button with loading state
- Cancel button
- Show approval history if break already approved/rejected

**API Calls**:
- `POST /api/nurses/break-request` - Submit request
- `GET /api/nurses/clockstatus` - Check current approval status

**Response Data Used**:
- `clockRecord.breakApprovalStatus` - Show current status
- `clockRecord.breakReason` - Display if already submitted
- `clockRecord.breakRejectionReason` - Show if rejected

**Interaction Flow**:
```
1. Modal opens
2. Show current break approval status
3. If PENDING: Show "Awaiting manager approval" with timestamp
4. If APPROVED: Show "Break approved" with "Start Break" button
5. If REJECTED: Show rejection reason with option to re-request
6. If NOT_REQUESTED: Show form to submit break request
   - Text area for reason
   - Pre-filled shift break times
   - Submit button
```

**State Management**:
- `reason` - Break reason text
- `status` - Current approval status
- `loading` - API call in progress
- `submitted` - Form submitted, awaiting approval
- `rejectionReason` - If rejected, show reason

**Styling**:
- Modal with 500px width
- Green for approved, Orange for pending, Red for rejected
- Status badge prominently displayed
- Text area with character counter

---

## Component 4: StaffStatusDashboard (Manager Page)

**Location**: `frontend/src/Dashboard/Pages/StaffStatusPage.jsx`

**Purpose**: Real-time manager dashboard showing all staff clock status and pending approvals

**Features**:
- Real-time grid of all staff with status badges
- Filters: Department/Location, Status (Clocked In/On Break/Out)
- Summary cards: Total staff, Clocked In count, On Break, Pending Approvals
- Quick action buttons for pending approvals
- Color-coded staff status
- Real-time refresh every 10 seconds
- Sort by: Name, Clock-in time, Elapsed time
- Toggle to show only pending approvals

**API Calls**:
- `GET /api/manager/staff-status` - Get all staff status with filters
- `GET /api/manager/pending-approvals` - Get pending requests
- `POST /api/manager/bulk-approvals` - Approve/reject multiple
- `POST /api/nurses/approve-late-arrival/:recordId` - Individual late approval
- `POST /api/nurses/approve-break/:recordId` - Individual break approval

**Response Data Used**:
- `summary` - Summary statistics
- `staff[].nurse` - Staff name/email
- `staff[].status` - Clock status
- `staff[].elapsedMinutes` - Time elapsed since clock-in
- `staff[].hasPendingApprovals` - Show if needs action
- `staff[].breakApprovalStatus` - Show break status
- `staff[].lateArrivalApprovalStatus` - Show late arrival status

**Interaction Flow**:
```
1. Page loads, fetches GET /api/manager/staff-status
2. Shows grid of all staff with status badges
3. Summary cards show counts and pending approvals
4. User clicks filter: Location = "ICU"
5. Table updates to show only ICU staff
6. Manager sees "2 pending approvals" badge on staff member
7. Clicks on staff member → Approval modal opens
8. Shows: 20 minutes late, Break request approved
9. Manager clicks "Approve Late Arrival"
10. Sends POST /api/nurses/approve-late-arrival/:recordId
11. Toast success, staff member removed from pending list
12. Auto-refresh every 10 seconds updates display
```

**State Management**:
- `staffList` - All staff and their clock records
- `filters` - Active filters (location, status)
- `sortBy` - Sort column and direction
- `selectedStaff` - For approval modal
- `showPendingOnly` - Toggle pending approvals only
- `loading` - API call in progress
- `summary` - Summary statistics

**UI Layout**:
- **Top Section**:
  - Summary cards: Total, Clocked In, On Break, Clocked Out, Pending Approvals
  - Location filter dropdown
  - Status filter (checkboxes)
  - "Show Pending Only" toggle
  - Sort dropdown

- **Main Section**:
  - Table of staff:
    - Name | Status | Location | Clock-in Time | Elapsed | Actions
    - Color-code rows: Green (clocked in), Orange (on break), Gray (out)
    - Red badge on rows with pending approvals
    - Quick action button (hover to reveal)

- **Approval Modal** (when staff selected):
  - Staff name and time clocked in
  - Break request details (if pending)
  - Late arrival details (if pending)
  - Approve/Reject buttons
  - Notes/reason input

**Styling**:
- Dashboard layout with header and sidebar
- Cards for summary statistics
- Professional data grid
- Color-coded status indicators
- Modal for approvals
- Toast notifications

---

## Component 5: MyClockHistory

**Location**: `frontend/src/Dashboard/Pages/MyClockHistoryPage.jsx`

**Purpose**: Personal clock record history for nurses and managers

**Features**:
- Table of all clock records with date range filter
- Columns: Date, Clock In Time, Clock Out Time, Break Time, Total Hours, Approval Status
- Filter by date range
- Download as CSV
- Color-code approval statuses
- Sort by any column
- Show approval reason details on hover/modal

**API Calls**:
- `GET /api/nurses/clockhistory?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD` - Get history

**Response Data Used**:
- `records[].date` - Shift date
- `records[].clockInTime` / `clockOutTime` - Times
- `records[].totalBreakTime` - Break duration
- `records[].totalWorkTime` - Work hours  
- `records[].lateArrivalApprovalStatus` - Late approval status
- `records[].breakApprovalStatus` - Break approval status
- `records[].lateArrivalMinutes` - How many minutes late
- `records[].notes` - Any shift notes

**Interaction Flow**:
```
1. Page loads with default date range (last 30 days)
2. Table shows all clock records for that period
3. Rows color-coded: Green (approved), Orange (pending), Red (rejected)
4. User clicks date range picker
5. Selects custom dates (e.g., April 1-14, 2026)
6. Table filters to show only those dates
7. User clicks "Download CSV" button
8. System generates CSV: Date, In, Out, Break, Total, Approval Status
9. CSV downloads to user's computer
10. User hovers over cell with "PENDING" status
11. Tooltip shows: "Late Arrival - Pending Manager Approval since 12:00 PM"
```

**State Management**:
- `records` - Clock history records
- `startDate` / `endDate` - Date range filters
- `sortBy` - Current sort column
- `sortOrder` - Ascending/descending
- `loading` - API call in progress
- `selectedRecord` - For detail modal

**UI Layout**:
- **Header**:
  - Date range picker (From/To dates)
  - Download CSV button
  - Search/filter bar

- **Table**:
  - Date | In | Out | Break | Total Hours | Late | Break Status | Actions
  - Sortable columns (click header)
  - Color-coded rows for approval status
  - Hover row to show detail button

- **Detail Modal**:
  - Full record details
  - Approval history with timestamps
  - Manager approval reason/notes

**Styling**:
- Clean data table layout
- Color-coded status badges
- Date picker Calendar UI
- PDF/CSV export button
- Responsive table for mobile

---

## Integration Points

### Update Dashboard.jsx
Add navigation to these new pages/components:
```jsx
// In Dashboard.jsx sidebar or main menu
<Link to="/dashboard/staff-status">Staff Status (Manager)</Link>
<Link to="/dashboard/clock-history">My Clock History</Link>
```

### Update SidebarLayout.jsx
Add menu items for:
- Clock-In/Out button (top/prominent)
- Available Shifts link
- Clock History link
- Staff Status (only if manager role)

### Update main routing in App.jsx or Dashboard.jsx
```jsx
<Route path="/dashboard/staff-status" element={<StaffStatusPage />} />
<Route path="/dashboard/clock-history" element={<MyClockHistoryPage />} />
```

### Update user role checking
Implement manager role check (currently uses JWT auth):
```jsx
// Check if user has manager permissions
const isManager = user.role === 'manager' || user.permissions.includes('manage_staff');
{isManager && <StaffStatusLink />}
```

---

## Data Flow Diagram

```
User Login
└─ Get JWT Token
   │
   ├─→ Nurse Flow:
   │   └─ Dashboard opens
   │      ├─ ClockInWithApprovals component shows
   │      ├─ Calls GET /api/nurses/available-shifts
   │      ├─ ShiftSelectionModal displays shifts
   │      ├─ Nurse clicks shift → POST /api/nurses/clockin
   │      ├─ During shift: POST /api/nurses/break-request
   │      ├─ Manager approves: POST /api/nurses/approve-break/:id
   │      ├─ Clock out: POST /api/nurses/clockout
   │      └─ View history: GET /api/nurses/clockhistory
   │
   └─→ Manager Flow:
       └─ Dashboard opens
          ├─ StaffStatusDashboard shows
          ├─ Polls GET /api/manager/staff-status every 10s
          ├─ Sees pending approvals badge
          ├─ Clicks staff member
          ├─ Modal shows break/late arrival request
          ├─ Manager clicks Approve
          ├─ POST /api/nurses/approve-break/:id or approve-late-arrival/:id
          ├─ Dashboard refreshes
          └─ Can view history: GET /api/nurses/clockhistory (filtered)
```

---

## Styling/Design Notes

1. **Color Scheme**:
   - Clocked In: Green (#10b981)
   - On Break: Orange (#f59e0b)
   - Clocked Out: Gray (#6c757d)
   - Pending Approval: Yellow (#fbbf24)
   - Approved: Green (#10b981)
   - Rejected: Red (#ef4444)

2. **Icons** (Use React Icons):
   - FaClock - Clock-in/out
   - FaPlay - Start break
   - FaPause - End break
   - FaCheckCircle - Approved
   - FaTimesCircle - Rejected
   - FaHourglassHalf - Pending
   - FaUser - Staff member
   - FaUsers - Team/All staff

3. **Responsive Design**:
   - Modals: Mobile-friendly (stack on small screens)
   - Tables: Horizontal scroll on mobile
   - Buttons: Touch-friendly size (min 44px)
   - Dashboard: Single column on mobile

4. **Real-time Updates**:
   - ClockInWithApprovals: Refresh every 1-5 seconds
   - StaffStatusDashboard: Refresh every 10 seconds
   - (Avoid high-frequency polling to prevent server load)

---

## Testing Checklist

- [ ] ShiftSelectionModal displays today's shifts first
- [ ] ClockInWithApprovals shows late arrival badge if >15 min late
- [ ] Break request requires manager approval
- [ ] StaffStatusDashboard shows real-time staff status
- [ ] Manager can approve/reject breaks in bulk
- [ ] MyClockHistory downloads CSV with proper formatting
- [ ] Date range filters work correctly
- [ ] All timestamps are in correct timezone
- [ ] Approval notifications are real-time (~5-10 sec delay)
- [ ] Error handling for API failures (show toast notifications)

---

## Next Steps

1. Create ShiftSelectionModal.jsx
2. Create ClockInWithApprovals.jsx (enhanced version)
3. Create BreakRequestModal.jsx
4. Create Pages/StaffStatusPage.jsx
5. Create Pages/MyClockHistoryPage.jsx
6. Update routing and sidebar navigation
7. Test all components with real backend APIs
8. Add role-based access control (manager vs nurse)

