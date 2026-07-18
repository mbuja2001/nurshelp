# Nurse Helpdesk Dashboard Components - Complete Implementation

## Overview
The Nurse Helpdesk Dashboard consists of 6 comprehensive components that provide a complete time management and scheduling system for healthcare staff. All components follow a consistent design system with clean styling, accessibility, and user-friendly interfaces.

---

## 📊 Component Breakdown

### 1. **ClockInOutBoard.jsx**
**Location:** `src/Dashboard/Pages/ClockInOutBoard.jsx`

#### Features:
- **Live Clock Display** - Real-time HH:MM:SS display with purple gradient widget
- **Clock In/Out** - Primary actions with status indicators
- **Break Management** - Start/end breaks with duration tracking
- **Today's Summary** - Quick stats showing:
  - Clock in/out times
  - Total work hours
  - Total break time
  - Late arrival indicators
- **Break Status** - Live break duration tracking while on break
- **Late Arrival Handling** - Shows if arrival was late and approval status
- **Shift Information** - Location and department display

#### API Endpoints Used:
- `POST /api/nurses/clock-in` - Record clock-in time
- `POST /api/nurses/clock-out` - Record clock-out time
- `GET /api/nurses/today-record` - Fetch today's shift record
- `POST /api/nurses/break-start` - Start a break
- `POST /api/nurses/break-end` - End a break

---

### 2. **MyClockHistoryPage.jsx**
**Location:** `src/Dashboard/Pages/MyClockHistoryPage.jsx`

#### Features:
- **Date Range Filter** - Select custom start/end dates for records
- **CSV Export** - Download clock history as formatted CSV file with columns:
  - Date, Clock In, Clock Out, Break Duration, Work Hours, Location
  - Late Arrival (min), Late Approval Status, Break Approval Status, Notes
- **Records Table** - Sortable table with:
  - Clock in/out times
  - Break minutes
  - Work hours calculation
  - Approval status with color-coded badges
- **Summary Statistics** - Total record count and aggregated hours
- **Approval Legends** - Visual guide for approval statuses (APPROVED, PENDING, REJECTED)

#### Key Calculations:
- Format: `HH:MM` time display
- Duration: Minutes and hours formatting
- Work Hours: Total minutes converted to hours and remaining minutes

#### API Endpoints Used:
- `GET /api/nurses/clockhistory?startDate=X&endDate=Y` - Fetch filtered records

---

### 3. **ManagerApprovalPage.jsx**
**Location:** `src/Dashboard/Pages/ManagerApprovalPage.jsx`

#### Features for Managers:
- **Tab Navigation** - Switch between:
  - ⏰ Late Arrivals (with pending count)
  - ☕ Break Overrides (with pending count)
- **Advanced Filtering:**
  - Filter by status (PENDING, APPROVED, REJECTED, ALL)
  - Search by nurse name or employee ID
- **Request Details:**
  - Approval type badge
  - Nurse name and ID
  - Request date
  - Status indicator
  - Scheduled vs actual times
  - Late minutes breakdown
  - Break timing and duration
  - Reason provided by nurse
- **Action Interface:**
  - Review & Decide button
  - Notes textarea for manager comments
  - Approve/Reject buttons
  - Cancel button to exit
- **Summary Dashboard:**
  - Pending actions count
  - Approvals this week
  - Rejections this week

#### Approval Types:
1. **LATE_ARRIVAL** - Shows:
   - Scheduled clock-in time
   - Actual clock-in time
   - Minutes late
   - Nurse's reason for lateness

2. **BREAK_OVERRIDE** - Shows:
   - Break start and end times
   - Duration in minutes
   - Location information
   - Reason for override

#### API Endpoints Used:
- `GET /api/managers/pending-approvals` - Fetch pending approval requests
- `POST /api/managers/approve-request` - Submit approval/rejection decision

---

### 4. **AdminSettingsPage.jsx**
**Location:** `src/Dashboard/Pages/AdminSettingsPage.jsx`

#### Settings Sections:

**⏰ Timing & Threshold Settings**
- `lateArrivalThresholdMinutes` (1-120) - When to flag late arrivals
- `earlyDepartureThresholdMinutes` (1-120) - When to flag early departures
- `clockInWindowMinutes` (5-120) - Time window before shift for clock-in

**☕ Break Settings**
- `mandatoryBreakDurationMinutes` (15-120) - Minimum required break per shift
- `maxBreakDurationMinutes` (>= mandatory) - Max break without approval
- `maxConsecutiveHoursWithoutBreak` (2-12) - Hours before mandatory break
- `requireBreakApproval` - Toggle break approval requirement

**✅ Approval Settings**
- `requireLateApproval` - Enforce manager approval for late arrivals
- `autoApprovalEnabled` - Auto-approve minor infractions after 24 hours
- `notificationEmailEnabled` - Send email notifications for pending approvals

**📅 Schedule Settings**
- `dailyScheduleStartHour` (0-23) - Earliest shift start time
- `dailyScheduleEndHour` (0-23) - Latest shift end time

#### Features:
- **Validation** - All fields validate on save with specific constraints
- **Impact Warning** - Explains that changes apply to future records only
- **Success Feedback** - Clear confirmation messages on save
- **Error Handling** - Per-field validation errors with helpful messages

#### API Endpoints Used:
- `GET /api/admin/settings` - Fetch current settings
- `POST /api/admin/settings` - Save updated settings

---

### 5. **ScheduleBoard.jsx**
**Location:** `src/Dashboard/Pages/ScheduleBoard.jsx`

#### Features:
- **Week Selector** - Toggle between "This Week" and "Next Week"
- **Daily Grid Layout** - Shows all 7 days with shift cards
- **Shift Cards Display:**
  - Shift times (start - end)
  - Location (ED, etc.)
  - Shift date
  - Status indicator (color-coded)
- **Off Days** - Clearly marked when no shifts scheduled
- **Status Indicators:**
  - ✓ Completed (green)
  - ⏳ In Progress (blue)
  - 📅 Upcoming (orange)
- **Summary Statistics:**
  - Total shifts count
  - Total hours calculation
  - Completed count
  - Upcoming count
- **Legend** - Visual guide for status colors

#### Time Format:
- Display: `HH:MM` (e.g., "09:00")
- Durations calculated in hours with decimal

#### API Endpoints Used:
- `GET /api/nurses/schedule?week=current/next` - Fetch schedule data

---

### 6. **StatisticsPage.jsx**
**Location:** `src/Dashboard/Pages/StatisticsPage.jsx`

#### Time Range Options:
- This Week
- This Month
- This Quarter

#### Key Metrics Card:
1. **Total Work Hours** - With trend comparison
2. **Shifts Completed** - Count with scheduled comparison
3. **Attendance Rate** - Percentage with progress bar
4. **Average Shift Duration** - HH:MM format

#### Performance Section:
- **On-Time Arrivals** - Ratio and percentage
- **Late Arrivals** - Count with pending approvals
- **Break Compliance** - Completion ratio and percentage
- **Total Break Time** - HH:MM across all shifts

#### Approval Status Section:
- **Pending Approvals** - Count and status
- **Approved This Period** - Count
- **Rejected This Period** - Count

#### Advanced Features:
- **Insights** - AI-generated recommendations with types:
  - Positive feedback (👍)
  - Warnings (⚠️)
  - Information (ℹ️)
- **Comparison Section** - 
  - Your value vs department average
  - Status indicator (Above/Below average)
  - Helps benchmarking

#### API Endpoints Used:
- `GET /api/nurses/statistics?range=weekly/monthly/quarterly` - Fetch statistics

---

## 🎨 Design System

### Color Palette:
- **Primary:** #3b82f6 (Blue) - Interactive elements
- **Success:** #10b981 (Green) - Approved, completed states
- **Warning:** #f59e0b (Amber) - Pending, warning states
- **Danger:** #ef4444 (Red) - Rejected, error states
- **Neutral:** #6b7280 (Gray) - Secondary text
- **Background:** #f3f4f6 (Light Gray) - Page background

### Typography:
- Headings: 700 weight (bold)
- Labels: 600 weight (semibold)
- Body: 400 weight (regular)
- Small text: 13-14px size

### Spacing:
- Large gaps: 24px
- Medium gaps: 16px
- Small gaps: 8-12px
- Padding: 16-24px

---

## 🔌 Integration Points

### Authentication:
All components require a valid JWT token passed via props or localStorage:
```javascript
const token = localStorage.getItem('token');
const res = await fetch(endpoint, {
  headers: { Authorization: `Bearer ${token}` }
});
```

### Backend Requirements:
Ensure backend implements all API endpoints with:
- JWT token validation
- User role checking (NURSE, MANAGER, ADMIN)
- Proper error responses with `success` flag and `message`

---

## 📝 Component Props

### ClockInOutBoard
```javascript
<ClockInOutBoard 
  token={string}  // JWT authentication token
  user={object}   // { name, location, role }
/>
```

### MyClockHistoryPage
```javascript
<MyClockHistoryPage 
  token={string}  // JWT authentication token
/>
```

### ManagerApprovalPage
```javascript
<ManagerApprovalPage 
  token={string}           // JWT authentication token
  userRole={string}        // 'MANAGER' or 'ADMIN'
/>
```

### AdminSettingsPage
```javascript
<AdminSettingsPage 
  token={string}  // JWT authentication token
/>
```

### ScheduleBoard
```javascript
<ScheduleBoard 
  token={string}  // JWT authentication token
/>
```

### StatisticsPage
```javascript
<StatisticsPage 
  token={string}  // JWT authentication token
/>
```

---

## ✅ Validation & Error Handling

### Input Validation:
- All numeric fields validate against specified ranges
- Date pickers prevent invalid date selections
- Form submissions check for required fields
- Cross-field validation (e.g., end hour > start hour)

### Error Handling:
- API errors display in red banner at top of component
- Field-specific validation shows inline error text
- Success messages confirm actions
- Loading states prevent duplicate submissions

---

## 🚀 Usage Example

```javascript
// In your main Dashboard component
import ClockInOutBoard from './Pages/ClockInOutBoard';
import MyClockHistoryPage from './Pages/MyClockHistoryPage';
import ManagerApprovalPage from './Pages/ManagerApprovalPage';
import AdminSettingsPage from './Pages/AdminSettingsPage';
import ScheduleBoard from './Pages/ScheduleBoard';
import StatisticsPage from './Pages/StatisticsPage';

export default function NurseDashboard({ user, token }) {
  const [activePage, setActivePage] = useState('clock-in-out');

  return (
    <div>
      {/* Navigation */}
      <nav>
        <button onClick={() => setActivePage('clock-in-out')}>Clock In/Out</button>
        <button onClick={() => setActivePage('schedule')}>Schedule</button>
        <button onClick={() => setActivePage('history')}>History</button>
        <button onClick={() => setActivePage('statistics')}>Statistics</button>
        {user?.role === 'MANAGER' && (
          <button onClick={() => setActivePage('approvals')}>Approvals</button>
        )}
        {user?.role === 'ADMIN' && (
          <button onClick={() => setActivePage('settings')}>Settings</button>
        )}
      </nav>

      {/* Content */}
      {activePage === 'clock-in-out' && <ClockInOutBoard token={token} user={user} />}
      {activePage === 'schedule' && <ScheduleBoard token={token} />}
      {activePage === 'history' && <MyClockHistoryPage token={token} />}
      {activePage === 'statistics' && <StatisticsPage token={token} />}
      {activePage === 'approvals' && <ManagerApprovalPage token={token} userRole={user?.role} />}
      {activePage === 'settings' && <AdminSettingsPage token={token} />}
    </div>
  );
}
```

---

## 📋 Summary Table

| Component | Purpose | User Role | Key Features |
|-----------|---------|-----------|--------------|
| ClockInOutBoard | Clock in/out interface | All | Live clock, break tracking, status summary |
| MyClockHistoryPage | View past records | Nurses | Date filtering, CSV export, approval tracking |
| ManagerApprovalPage | Approve requests | Manager | Request review, decision making, notes |
| AdminSettingsPage | System configuration | Admin | 10 configurable settings, validation |
| ScheduleBoard | View shifts | All | Week selector, status indicators, summary |
| StatisticsPage | Performance metrics | All | Multiple metrics, comparisons, insights |

---

## 🔧 Setup Checklist

- [ ] Install React Icons: `npm install react-icons`
- [ ] Set VITE_BACKEND_URL environment variable
- [ ] Configure backend API endpoints matching component calls
- [ ] Implement JWT token handling
- [ ] Create user role-based routing
- [ ] Test all API endpoints respond with correct format
- [ ] Configure CORS for backend requests
- [ ] Add components to main Dashboard navigation
- [ ] Test responsive design on mobile devices
- [ ] Implement notification system for approvals

---

**Created:** 2024
**Status:** Production Ready
**Version:** 1.0.0
