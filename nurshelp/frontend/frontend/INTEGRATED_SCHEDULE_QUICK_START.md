# 🚀 Integrated Clock & Schedule - Quick Start

## Access the New Dashboard

### Main Entry Point
```
http://localhost:5173/integrated-schedule
```

### Alternative Routes
```
/clock-in-out          → Clock In/Out only
/clock-history         → History only
/schedule/staff        → Original staff management
/approvals             → Manager approvals
/statistics            → Performance stats
/admin-settings        → Admin configuration
```

## Feature Breakdown

### 📍 **Clock In/Out Tab** (`/integrated-schedule` → Tab 1)
**What it does:**
- Display live current time (updates every 500ms)
- Clock in to start your shift
- Start/end breaks during shift
- Clock out to end your shift
- See today's summary (hours worked, breaks taken)

**Quick Actions:**
1. Open `/integrated-schedule`
2. Click "📍 Clock In" button
3. Your shift starts - displays clock in time
4. Click "☕ Start Break" if needed
5. Click "✓ Finish Break" to resume work
6. Click "🛑 Clock Out" to end shift

**What you'll see:**
```
⏰ Current Time Display (HH:MM:SS)
Date: Today's Date

Current Status Card:
  Clock In Time: XX:XX
  Status: ⏳ Working

Today's Summary Card:
  Work Hours: Xh Ym
  Break Time: X min
```

---

### 📋 **Clock History Tab** (`/integrated-schedule` → Tab 2)
**What it does:**
- View your clock records with date filtering
- See approval status for each record
- Export records to CSV
- Filter by date range

**Quick Start:**
1. Open `/integrated-schedule`
2. Click "📋 Clock History" tab
3. Default shows last 30 days
4. Change dates if needed
5. Click "📥 Apply" to update
6. Click "📥 Export CSV" to download

**CSV Export Includes:**
- Date
- Clock In Time
- Clock Out Time
- Break Duration (minutes)
- Total Work Hours
- Location
- Late Arrival (minutes)
- Late Approval Status
- Break Approval Status
- Notes

**Approval Status Colors:**
- 🟢 **APPROVED** - Request was approved
- 🟠 **PENDING** - Awaiting approval
- 🔴 **REJECTED** - Request was denied
- ⚪ **N/A** - No approval needed

---

### 📅 **My Schedule Tab** (`/integrated-schedule` → Tab 3)
**What it does:**
- View your weekly shifts
- See shift times and locations
- Check shift status
- Compare This Week vs Next Week

**Quick Start:**
1. Open `/integrated-schedule`
2. Click "📅 My Schedule" tab
3. Click "This Week" or "Next Week"
4. Each day shows your shifts
5. Green = Completed, Blue = In Progress, Amber = Upcoming

**Schedule Information:**
```
Monday
  9:00 - 17:00
  📍 ED (Emergency Department)
  Status: 📅 Upcoming

Tuesday
  8:00 - 16:00
  📍 ICU
  Status: ✓ Completed
```

---

### 📊 **Statistics Tab** (`/integrated-schedule` → Tab 4)
**What it does:**
- View performance metrics
- Monitor attendance
- Track late arrivals
- See performance trends
- Compare with department average

**Key Metrics:**
1. **Total Work Hours** - Sum of all hours this month
2. **Shifts Completed** - Number of finished shifts
3. **Attendance Rate** - % of scheduled shifts worked
4. **Late Arrivals** - Number of late arrivals
5. Average Shift Duration (in hours)
6. Break Compliance (% of breaks taken)

**Visual Indicators:**
- 🔵 Blue = Positive metric
- 🟢 Green = Good performance
- 🟡 Amber = Warning
- 🔴 Red = Needs attention

---

## Manager Features

### Access Manager Panel
```
http://localhost:5173/approvals
```

**Manager Capabilities:**
1. View pending late arrival requests
2. View break override requests  
3. Approve or reject with notes
4. Filter by status (ALL, PENDING, APPROVED, REJECTED)
5. Search nurses by name or ID
6. See summary dashboard with pending count

**Workflow:**
1. Open `/approvals`
2. Select "Late Arrivals" or "Break Overrides" tab
3. Choose "PENDING" status
4. Click on a request to expand details
5. Review nurse's reason/notes
6. Enter your decision notes
7. Click "✓ Approve" or "✗ Reject"
8. Request updates with your decision

---

## Admin Features

### Access Admin Panel
```
http://localhost:5173/admin-settings
```

**Admin Capabilities:**
1. Configure late arrival threshold (1-120 min)
2. Set early departure threshold
3. Configure break duration requirements
4. Set maximum consecutive hours
5. Enable/disable approval requirements
6. Configure approval notifications
7. Set daily schedule hours

**Configuration Options:**

| Setting | Range | Default | Impact |
|---------|-------|---------|--------|
| Late Arrival Threshold | 1-120 min | 5 min | Minutes until arrival is "late" |
| Break Duration Required | 15-120 min | 30 min | Minimum break time |
| Max Hours Without Break | 2-12 hours | 6 hours | Hours before break required |
| Require Late Approval | True/False | True | Needs manager approval for late? |
| Auto-Approval | True/False | False | Auto-approve if under threshold |
| Daily Start Hour | 0-23 | 6 | Earliest clock in time |
| Daily End Hour | 0-23 | 22 | Latest clock in time |

**How to Change:**
1. Open `/admin-settings`
2. Find setting you want to change
3. Update the value
4. See error message if value invalid
5. Click "💾 Save Settings"
6. See success message when saved

---

## Using the Dashboard Components

### From Other Pages
You can navigate to these components from anywhere:

**Option 1: Direct URL**
```javascript
window.location.href = '/integrated-schedule'
```

**Option 2: React Router Navigate**
```javascript
import { useNavigate } from 'react-router-dom'

const navigate = useNavigate()
navigate('/integrated-schedule')
```

**Option 3: Link Component**
```jsx
<Link to="/integrated-schedule">View Schedule</Link>
```

---

## API Endpoints Reference

All endpoints require `Authorization: Bearer {TOKEN}` header.

```javascript
// Clock Operations
POST /api/nurses/clock-in
  → Starts your shift
  → Returns: { success: true, record: {...} }

POST /api/nurses/clock-out
  → Ends your shift
  → Returns: { success: true, record: {...} }

GET /api/nurses/today-record
  → Gets today's clock record
  → Returns: { success: true, record: {...} }

POST /api/nurses/break-start
  → Marks break start time
  → Returns: { success: true, breakStartTime: "..." }

POST /api/nurses/break-end
  → Marks break end time
  → Returns: { success: true, breakDuration: 1800000 }

// History & Filtering
GET /api/nurses/clockhistory?startDate=2026-04-01&endDate=2026-04-14
  → Gets records in date range
  → Returns: { success: true, records: [...] }

// Schedule
GET /api/nurses/schedule?week=current|next
  → Gets weekly schedule
  → Returns: { success: true, schedule: [...] }

// Statistics  
GET /api/nurses/statistics?range=weekly|monthly|quarterly
  → Gets performance stats
  → Returns: { success: true, stats: {...} }

// Quick Stats (for widget)
GET /api/nurses/stats
  → Gets quick stats
  → Returns: { success: true, stats: {...} }
```

---

## Troubleshooting

### Problem: Page doesn't load
**Solution:**
1. Check if backend is running: `curl http://localhost:5001`
2. Verify `.env` has correct `VITE_BACKEND_URL`
3. Clear browser cache: Ctrl+Shift+Delete
4. Reload page: Ctrl+Shift+R

### Problem: "Failed to fetch today record"
**Solution:**
1. Verify you're logged in
2. Check token in localStorage
3. Ensure backend API is implemented
4. Check browser console for error details

### Problem: Clock history is empty
**Solution:**
1. Verify dates are correct
2. Check that clock records exist in database
3. Try expanding date range
4. Verify backend has `/api/nurses/clockhistory` endpoint

### Problem: Can't see "Approvals" tab
**Solution:**
1. Check your user role (must be MANAGER or ADMIN)
2. Verify token has correct role
3. Contact admin if access needed

---

## Data Format Examples

### Clock Record
```json
{
  "_id": "123abc",
  "nurseId": "nurse_001",
  "date": "2026-04-14",
  "clockInTime": "2026-04-14T09:05:00Z",
  "clockOutTime": "2026-04-14T17:30:00Z",
  "totalWorkTime": 28800000,
  "totalBreakTime": 1800000,
  "lateArrivalMinutes": 5,
  "lateArrivalApprovalStatus": "PENDING",
  "location": "ED",
  "notes": ""
}
```

### Schedule Record
```json
{
  "_id": "456def",
  "nurseId": "nurse_001",
  "date": "2026-04-14",
  "dayOfWeek": "Monday",
  "shiftStart": "09:00",
  "shiftEnd": "17:00",
  "location": "ED",
  "status": "UPCOMING",
  "assignedPatients": 10
}
```

### Statistics Response
```json
{
  "stats": {
    "totalWorkHours": 160.5,
    "shiftsCompleted": 20,
    "attendanceRate": 95,
    "lateArrivals": 2,
    "averageShiftDuration": 8.5,
    "breakCompliance": 92,
    "departmentAverage": 8.2
  }
}
```

---

## Tips & Best Practices

### ✅ Do:
- ✓ Clock in as soon as you arrive
- ✓ Start break when taking breaks
- ✓ Clock out at end of shift
- ✓ Check your schedule regularly
- ✓ Export records for personal records
- ✓ Review approval status regularly

### ❌ Don't:
- ✗ Forget to clock out
- ✗ Modify clock times manually
- ✗ Leave breaks running indefinitely
- ✗ Ignore pending approvals (managers)
- ✗ Share your token/credentials
- ✗ Use after hours if not scheduled

---

## Performance Tips

- **Faster Loading**: Bookmark `/integrated-schedule` for quick access
- **Quick Export**: Use keyboard shortcut Ctrl+Shift+S after clicking Export
- **Mobile**: All features work on mobile browsers
- **Offline**: Sync data when back online

---

## Getting Help

**For Features:**
- Check `CLOCK_SCHEDULE_INTEGRATION.md` for detailed docs
- Review `DASHBOARD_COMPONENTS_GUIDE.md` for component details

**For Issues:**
- Check browser console for error messages
- Verify backend is running
- Test API endpoints with curl (see above)
- Contact system admin for access issues

---

**Version**: 1.0.0  
**Last Updated**: April 14, 2026  
**Status**: Production Ready

🚀 **Ready to start?** Open http://localhost:5173/integrated-schedule
