# Nurse Helpdesk Dashboard - Integration Guide

## Quick Start

This guide walks you through integrating the new dashboard components into your existing Nurse Helpdesk application.

---

## 1. File Structure

Ensure the following directory structure exists:

```
frontend/
├── src/
│   └── Dashboard/
│       ├── Pages/
│       │   ├── ClockInOutBoard.jsx          ✓ Created
│       │   ├── MyClockHistoryPage.jsx       ✓ Created
│       │   ├── ManagerApprovalPage.jsx      ✓ Created
│       │   ├── AdminSettingsPage.jsx        ✓ Created
│       │   ├── ScheduleBoard.jsx            ✓ Created
│       │   └── StatisticsPage.jsx           ✓ Created
│       └── NurseDashboard.jsx               (Existing - needs update)
├── DASHBOARD_COMPONENTS_GUIDE.md            ✓ Created
└── INTEGRATION_GUIDE.md                     ✓ Created (this file)
```

---

## 2. Dependencies

Add React Icons to your project (required by components):

```bash
npm install react-icons
```

The components use these icons:
- `FaClock`, `FaPlay`, `FaStop`, `FaCoffee` - Clock/break actions
- `FaDownload`, `FaFilter`, `FaCalendarAlt` - History filtering
- `FaSearch`, `FaCheck`, `FaTimes` - Approval handling
- `FaSave`, `FaExclamationTriangle` - Settings UI

---

## 3. Component Integration into Main Dashboard

Create or update `src/Dashboard/NurseDashboard.jsx` with proper routing:

```javascript
// src/Dashboard/NurseDashboard.jsx
import React, { useState, useEffect } from 'react';
import { FaClock, FaHistory, FaCheckCircle, FaCog, FaChartBar, FaSignOutAlt } from 'react-icons/fa';

// Import all pages
import ClockInOutBoard from './Pages/ClockInOutBoard';
import MyClockHistoryPage from './Pages/MyClockHistoryPage';
import ManagerApprovalPage from './Pages/ManagerApprovalPage';
import AdminSettingsPage from './Pages/AdminSettingsPage';
import ScheduleBoard from './Pages/ScheduleBoard';
import StatisticsPage from './Pages/StatisticsPage';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function NurseDashboard({ user, token, onLogout }) {
  const [activePage, setActivePage] = useState('clock-in-out');
  const [userStats, setUserStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetchUserStats();
    }
  }, [token]);

  const fetchUserStats = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurses/stats`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setUserStats(data.stats);
      }
    } catch (err) {
      console.error('Error fetching stats:', err);
    } finally {
      setLoading(false);
    }
  };

  const getNavItems = () => {
    const items = [
      { id: 'clock-in-out', label: 'Clock In/Out', icon: FaClock },
      { id: 'schedule', label: 'My Schedule', icon: FaChartBar },
      { id: 'history', label: 'Clock History', icon: FaHistory },
      { id: 'statistics', label: 'Statistics', icon: FaChartBar }
    ];

    if (user?.role === 'MANAGER' || user?.role === 'ADMIN') {
      items.push({ id: 'approvals', label: 'Approvals', icon: FaCheckCircle });
    }

    if (user?.role === 'ADMIN') {
      items.push({ id: 'settings', label: 'Settings', icon: FaCog });
    }

    return items;
  };

  const renderPage = () => {
    switch (activePage) {
      case 'clock-in-out':
        return <ClockInOutBoard token={token} user={user} />;
      case 'schedule':
        return <ScheduleBoard token={token} />;
      case 'history':
        return <MyClockHistoryPage token={token} />;
      case 'statistics':
        return <StatisticsPage token={token} />;
      case 'approvals':
        return <ManagerApprovalPage token={token} userRole={user?.role} />;
      case 'settings':
        return <AdminSettingsPage token={token} />;
      default:
        return <ClockInOutBoard token={token} user={user} />;
    }
  };

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <div style={styles.sidebar}>
        <div style={styles.header}>
          <h2 style={styles.title}>🏥 Nurse Helpdesk</h2>
        </div>

        <div style={styles.userProfile}>
          <div style={styles.userAvatar}>
            {user?.name?.charAt(0).toUpperCase()}
          </div>
          <div>
            <p style={styles.userName}>{user?.name}</p>
            <p style={styles.userRole}>{user?.role}</p>
          </div>
        </div>

        <nav style={styles.nav}>
          {getNavItems().map(item => (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              style={{
                ...styles.navItem,
                ...(activePage === item.id ? styles.navItemActive : {})
              }}
            >
              <item.icon style={{ marginRight: '8px' }} />
              {item.label}
            </button>
          ))}
        </nav>

        {userStats && (
          <div style={styles.statsWidget}>
            <h4>This Week</h4>
            <p>Hours: {userStats.weeklyHours || 0}h</p>
            <p>Late: {userStats.weeklyLateArrivals || 0}</p>
          </div>
        )}

        <button onClick={onLogout} style={styles.logoutBtn}>
          <FaSignOutAlt /> Logout
        </button>
      </div>

      {/* Main Content */}
      <div style={styles.main}>
        {renderPage()}
      </div>
    </div>
  );
}

const styles = {
  container: { display: 'flex', height: '100vh' },
  sidebar: {
    width: '280px',
    backgroundColor: '#1f2937',
    color: 'white',
    padding: '0',
    display: 'flex',
    flexDirection: 'column'
  },
  header: { padding: '20px 16px', borderBottom: '1px solid rgba(255,255,255,0.1)' },
  title: { margin: '0', fontSize: '18px' },
  userProfile: {
    padding: '16px',
    display: 'flex',
    gap: '12px',
    borderBottom: '1px solid rgba(255,255,255,0.1)'
  },
  userAvatar: {
    width: '44px',
    height: '44px',
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '700'
  },
  userName: { margin: 0, fontWeight: '600', fontSize: '14px' },
  userRole: { margin: '2px 0 0 0', fontSize: '12px', color: '#d1d5db' },
  nav: { flex: 1, padding: '16px 0', display: 'flex', flexDirection: 'column' },
  navItem: {
    padding: '12px 16px',
    backgroundColor: 'transparent',
    border: 'none',
    color: '#d1d5db',
    textAlign: 'left',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center'
  },
  navItemActive: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    color: '#60a5fa',
    borderLeft: '3px solid #60a5fa'
  },
  statsWidget: {
    margin: '16px',
    padding: '16px',
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: '8px'
  },
  logoutBtn: {
    margin: '16px',
    padding: '12px 16px',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  main: { flex: 1, overflowY: 'auto', backgroundColor: '#f3f4f6' }
};
```

---

## 4. Environment Configuration

Ensure your `.env` file (or `.env.development`) includes:

```env
VITE_BACKEND_URL=http://localhost:5001
VITE_API_BASE=/api
```

The components use this in initialization:
```javascript
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
```

---

## 5. App.jsx Integration

Update your main App component to route to the dashboard:

```javascript
// src/App.jsx
import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LoginForm from './LoginForm';
import NurseDashboard from './Dashboard/NurseDashboard';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is already logged in
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
    }
    setLoading(false);
  }, []);

  const handleLogin = async (email, password) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      
      if (data.success) {
        setToken(data.token);
        setUser(data.user);
        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));
      } else {
        alert(data.message || 'Login failed');
      }
    } catch (err) {
      alert('Error: ' + err.message);
    }
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={!token ? <LoginForm onLogin={handleLogin} /> : <Navigate to="/dashboard" />}
        />
        <Route
          path="/dashboard"
          element={token ? <NurseDashboard user={user} token={token} onLogout={handleLogout} /> : <Navigate to="/login" />}
        />
        <Route path="/" element={<Navigate to={token ? "/dashboard" : "/login"} />} />
      </Routes>
    </BrowserRouter>
  );
}
```

---

## 6. Backend API Requirements

Ensure your backend implements these endpoints:

### Clock Management
```
POST   /api/nurses/clock-in
POST   /api/nurses/clock-out
GET    /api/nurses/today-record
POST   /api/nurses/break-start
POST   /api/nurses/break-end
```

### History & Scheduling
```
GET    /api/nurses/clockhistory
GET    /api/nurses/schedule
GET    /api/nurses/stats
GET    /api/nurses/statistics
```

### Approval System (Manager)
```
GET    /api/managers/pending-approvals
POST   /api/managers/approve-request
```

### Admin Settings
```
GET    /api/admin/settings
POST   /api/admin/settings
```

---

## 7. Testing the Integration

### Test Checklist:

1. **Clock In/Out Board**
   - [ ] Clock in button works
   - [ ] Clock out button appears when clocked in
   - [ ] Break buttons work correctly
   - [ ] Live time updates every second
   - [ ] Today's summary shows correct values

2. **Clock History**
   - [ ] Date filter works
   - [ ] Records display correctly
   - [ ] CSV export downloads file
   - [ ] Approval status badges display correctly

3. **Manager Approvals**
   - [ ] Only appears for MANAGER/ADMIN role
   - [ ] Tab switching works
   - [ ] Filter and search work
   - [ ] Review & Decide button opens action panel
   - [ ] Approve/Reject buttons submit

4. **Admin Settings**
   - [ ] Only appears for ADMIN role
   - [ ] Settings load from backend
   - [ ] Validation prevents invalid values
   - [ ] Save button updates settings
   - [ ] Validation errors display

5. **Schedule Board**
   - [ ] Week selector works
   - [ ] Shifts display in correct day columns
   - [ ] Summary statistics calculate correctly
   - [ ] Status colors show correctly

6. **Statistics**
   - [ ] Time range selector works
   - [ ] All metrics display
   - [ ] Progress bars animate
   - [ ] Comparison section shows

---

## 8. Styling Customization

All components use inline styles. To customize:

1. **Update color palette** - Search for color values like `#3b82f6`, `#10b981`, etc.
2. **Adjust spacing** - Modify `padding`, `gap`, `margin` values
3. **Change fonts** - Add CSS file or update inline `fontSize`, `fontWeight`

Example customization:
```javascript
const styles = {
  primaryBtn: {
    backgroundColor: '#your-color', // Change here
    padding: '16px', // Or here
    fontSize: '16px' // Or spacing
  }
};
```

---

## 9. Common Issues & Solutions

### Issue: "Cannot find module 'react-icons'"
**Solution:** Run `npm install react-icons`

### Issue: "VITE_BACKEND_URL is undefined"
**Solution:** Add to `.env`:
```
VITE_BACKEND_URL=http://localhost:5001
```

### Issue: "API returns 401 Unauthorized"
**Solution:** Ensure token is valid and passed correctly:
```javascript
headers: { Authorization: `Bearer ${token}` }
```

### Issue: CSV download fails
**Solution:** Ensure browser allows downloads and file isn't blocked

### Issue: Live clock doesn't update
**Solution:** Check that useEffect interval is set:
```javascript
useEffect(() => {
  const interval = setInterval(() => setLiveTime(new Date()), 500);
  return () => clearInterval(interval);
}, []);
```

---

## 10. Performance Optimization

### Recommendations:

1. **Lazy Load Components** - Use React.lazy():
```javascript
const ClockInOutBoard = lazy(() => import('./Pages/ClockInOutBoard'));
```

2. **Memoize Components** - Prevent unnecessary re-renders:
```javascript
export default memo(MyClockHistoryPage);
```

3. **Debounce Search** - Use debounce for filter inputs:
```javascript
const [searchTerm, setSearchTerm] = useState('');
const debouncedSearch = useCallback(debounce((term) => {
  handleSearch(term);
}, 300), []);
```

4. **Cache API Responses** - Store frequently accessed data:
```javascript
const [cache, setCache] = useState({});
if (cache[key]) return cache[key];
```

---

## 11. Accessibility Improvements

Add ARIA labels and semantic HTML:
```javascript
<button
  onClick={handleClockIn}
  aria-label="Clock in to start shift"
  title="Start your work shift"
>
  Clock In
</button>
```

---

## 12. Deployment Checklist

- [ ] All components created in correct locations
- [ ] Environment variables configured
- [ ] Backend API endpoints implemented
- [ ] React Icons installed
- [ ] Authentication working
- [ ] Role-based access verified
- [ ] Responsive design tested on mobile
- [ ] Error handling verified
- [ ] Loading states display correctly
- [ ] CSV export works
- [ ] Approval workflow tested end-to-end
- [ ] Settings changes persist

---

## 13. Support & Troubleshooting

For issues with:
- **Component rendering** - Check console for errors
- **API calls** - Verify endpoint URLs in browser DevTools Network tab
- **Styling** - Use browser DevTools Inspector to debug CSS

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Status:** Production Ready
