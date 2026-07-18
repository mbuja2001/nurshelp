# Dashboard Components - Quick Reference

## 🚀 Quick Start Commands

```bash
# Install dependencies
npm install react-icons

# Start development server
npm run dev

# Build for production
npm run build
```

---

## 📂 File Locations

```
frontend/src/Dashboard/Pages/
├── ClockInOutBoard.jsx          (580 lines)
├── MyClockHistoryPage.jsx       (450 lines)
├── ManagerApprovalPage.jsx      (520 lines)
├── AdminSettingsPage.jsx        (480 lines)
├── ScheduleBoard.jsx            (340 lines)
└── StatisticsPage.jsx           (510 lines)

frontend/
├── DASHBOARD_COMPONENTS_GUIDE.md
├── INTEGRATION_GUIDE.md
└── QUICK_REFERENCE.md (this file)
```

---

## 🎯 Component Purpose & Route

| Component | Route | Purpose | Who Uses |
|-----------|-------|---------|----------|
| ClockInOutBoard | `/dashboard?page=clock-in-out` | Clock in/out interface with breaks | All Staff |
| ScheduleBoard | `/dashboard?page=schedule` | View upcoming shifts | All Staff |
| MyClockHistoryPage | `/dashboard?page=history` | View past records + CSV export | All Staff |
| StatisticsPage | `/dashboard?page=statistics` | Performance metrics & insights | All Staff |
| ManagerApprovalPage | `/dashboard?page=approvals` | Review & approve requests | Managers |
| AdminSettingsPage | `/dashboard?page=settings` | Configure system settings | Admins |

---

## 🔑 Key Features

### ClockInOutBoard
```javascript
✓ Live clock (HH:MM:SS)
✓ Clock In/Out actions
✓ Break Start/End tracking
✓ Today's summary metrics
✓ Late arrival indicators
✓ Break status display
```

### MyClockHistoryPage
```javascript
✓ Date range filter
✓ CSV export with headers
✓ Sortable data table
✓ Approval status colors
✓ Work hours calculation
✓ Break time tracking
```

### ManagerApprovalPage
```javascript
✓ Tab navigation (Late/Breaks)
✓ Status filtering
✓ Nurse search
✓ Approval details display
✓ Notes textarea
✓ Approve/Reject actions
✓ Summary statistics
```

### AdminSettingsPage
```javascript
✓ 10 configurable settings
✓ Range validation
✓ Cross-field validation
✓ Save/Load functionality
✓ Change impact warning
✓ Success notifications
```

### ScheduleBoard
```javascript
✓ Week selector (This/Next)
✓ 7-day grid layout
✓ Color-coded status
✓ Hours summary
✓ Off-day indicators
```

### StatisticsPage
```javascript
✓ 4 main metric cards
✓ Performance section
✓ Approval status breakdown
✓ AI insights
✓ Comparison to department
✓ 3 time ranges (W/M/Q)
```

---

## 🔌 API Endpoints Reference

### Clock Management
```rest
POST /api/nurses/clock-in
  Body: { location: string }
  
POST /api/nurses/clock-out
  
GET /api/nurses/today-record

POST /api/nurses/break-start

POST /api/nurses/break-end
  Body: { breakDuration: number }
```

### Records & History
```rest
GET /api/nurses/clockhistory?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD

GET /api/nurses/schedule?week=current|next

GET /api/nurses/stats

GET /api/nurses/statistics?range=weekly|monthly|quarterly
```

### Manager Functions
```rest
GET /api/managers/pending-approvals

POST /api/managers/approve-request
  Body: { recordId, action, notes }
```

### Admin Functions
```rest
GET /api/admin/settings

POST /api/admin/settings
  Body: { [key]: value }
```

---

## 📊 Response Format

All endpoints return:
```json
{
  "success": true|false,
  "message": "Optional message",
  "data": { /* varies by endpoint */ },
  "record": { /* for single records */ },
  "records": [ /* for multiple records */ ],
  "stats": { /* for statistics */ },
  "settings": { /* for admin settings */ }
}
```

---

## 🎨 Color Reference

```javascript
// Status Colors
#10b981 - Success/Approved/Green
#3b82f6 - Primary/Info/Blue
#f59e0b - Warning/Pending/Amber
#ef4444 - Error/Rejected/Red
#6b7280 - Secondary text/Gray
#1f2937 - Primary text/Dark Gray
#f3f4f6 - Background/Light Gray
```

---

## 💾 State Pattern

All components follow this pattern:
```javascript
const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  if (token) {
    fetchData();
  }
}, [token]);

const fetchData = async () => {
  setLoading(true);
  try {
    const res = await fetch(url, { 
      headers: { Authorization: `Bearer ${token}` } 
    });
    const data = await res.json();
    if (data.success) {
      setData(data.data || data.record);
    } else {
      setError(data.message);
    }
  } catch (err) {
    setError(err.message);
  } finally {
    setLoading(false);
  }
};
```

---

## 🔐 Authentication

```javascript
// Getting token
const token = localStorage.getItem('token');

// Using in requests
const res = await fetch(url, {
  headers: { 
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
});

// Getting user
const user = JSON.parse(localStorage.getItem('user'));
// user = { id, name, email, role: 'NURSE'|'MANAGER'|'ADMIN' }
```

---

## 📱 Responsive Breakpoints

Components use CSS Grid with `minmax()`:
```javascript
gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))'
// Mobile: 1 column (280px)
// Tablet: 2-3 columns
// Desktop: 4+ columns
```

---

## ⚠️ Common Errors

### Error 1: Invalid Token
```
Solution: Re-login, check token in localStorage
```

### Error 2: 404 Not Found
```
Solution: Verify API endpoint URL and VITE_BACKEND_URL
```

### Error 3: CORS Error
```
Solution: Add CORS headers to backend
  Access-Control-Allow-Origin: *
  Access-Control-Allow-Methods: GET, POST, etc.
```

### Error 4: Undefined Values
```
Solution: Check for null/undefined before accessing properties
  const name = user?.name || 'Unknown';
```

---

## 🧪 Testing Command

```bash
# Test component loads
npm run dev

# Check for errors in browser console
# Test API calls in Network tab
# Verify component renders without crashing
```

---

## 📋 Checklist: Adding a New Page

1. Create `src/Dashboard/Pages/NewPage.jsx`
2. Import required components
3. Add state management (loading, error, data)
4. Implement useEffect for data fetching
5. Add error handling and loading states
6. Build UI with inline styles
7. Add to NurseDashboard navigation
8. Test with token and user
9. Verify API calls work
10. Check responsive design

---

## 🎯 Code Snippets

### Export CSV Function
```javascript
const handleDownloadCSV = () => {
  const headers = ['Col1', 'Col2', 'Col3'];
  const rows = data.map(d => [d.field1, d.field2, d.field3]);
  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `file-${new Date().toISOString().split('T')[0]}.csv`;
  link.click();
  window.URL.revokeObjectURL(url);
};
```

### Time Formatting
```javascript
const formatTime = (dateStr) => {
  if (!dateStr) return '--:--';
  return new Date(dateStr).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  });
};

const formatDate = (dateStr) => {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric'
  });
};
```

### Conditional Styling
```javascript
style={{
  ...styles.button,
  ...(isActive ? styles.buttonActive : {}),
  backgroundColor: error ? '#ef4444' : '#10b981'
}}
```

### API Call Pattern
```javascript
const response = await fetch(`${BACKEND_URL}/api/endpoint`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(payload)
});
const data = await response.json();
if (data.success) {
  // Handle success
} else {
  setError(data.message);
}
```

---

## 📞 Module Imports

```javascript
// React
import React, { useState, useEffect } from 'react';

// React Icons (used throughout)
import { 
  FaClock, FaPlay, FaStop, FaCoffee,
  FaDownload, FaFilter, FaCalendarAlt,
  FaSearch, FaCheck, FaTimes,
  FaSave, FaExclamationTriangle,
  FaChartBar, FaHistory, FaCheckCircle,
  FaCog, FaSignOutAlt
} from 'react-icons/fa';
```

---

## 🔄 Data Flow

```
Login Screen
    ↓
  Token → localStorage
    ↓
NurseDashboard (passes token to all components)
    ↓
  Component fetches from API
    ↓
  Display data / Show errors
    ↓
  User interacts → setState
    ↓
  Submit to API
    ↓
  Refresh data / Show success
```

---

## 💡 Best Practices

1. ✅ Always check `data.success` before using response
2. ✅ Always pass token in headers
3. ✅ Handle loading and error states
4. ✅ Use optional chaining (`user?.name`)
5. ✅ Debounce search/filter inputs
6. ✅ Show feedback after actions
7. ✅ Validate form inputs
8. ✅ Use meaningful variable names
9. ✅ Keep components under 600 lines
10. ✅ Document complex logic

---

## 🚨 Important Notes

- Components use inline styles (easier to customize)
- All times formatted as HH:MM in 24-hour format
- Dates formatted as MM-DD-YYYY or Day, Month DD
- CSV exports include all necessary fields
- Approval system requires manager role
- Settings changes apply to future records only
- Backend must validate all user inputs

---

## 📚 Documentation Files

1. **DASHBOARD_COMPONENTS_GUIDE.md** - Full component documentation
2. **INTEGRATION_GUIDE.md** - Step-by-step integration instructions
3. **QUICK_REFERENCE.md** - This file (quick lookup)

---

## 🎓 Learning Resources

- React Hooks: [react.dev/reference/react](https://react.dev)
- React Icons: [react-icons.github.io](https://react-icons.github.io)
- Fetch API: [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- CSS Grid: [CSS-Tricks](https://css-tricks.com/snippets/css/complete-guide-grid)

---

**Last Updated:** 2024
**Version:** 1.0.0
**All Components:** ✅ Complete & Ready for Integration
