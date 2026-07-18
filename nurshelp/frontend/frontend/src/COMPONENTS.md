# Reusable Component Library - Hospital Management System

## Overview
Professional, hospital-grade reusable components built with React for comprehensive HMS UI consistency and rapid dashboard development.

## Component Inventory

### 1. MainNavigation Component
**File**: `src/components/common/MainNavigation.jsx` + `MainNavigation.css`
**Purpose**: Professional top navigation bar for all user roles
**Status**: ✅ COMPLETE & READY TO USE

#### Features:
- Hospital logo and name (left)
- Global search bar (center)
- Notifications dropdown with alert counts
- User profile menu with role display
- Help button
- Responsive design (collapses on mobile)

#### Props:
```jsx
<MainNavigation
  userName="Dr. Sarah Johnson"        // User full name
  userRole="Physician"                 // Role display name
  userAvatar="👨‍⚕️"                     // User avatar (emoji)
  notificationCount={3}                // Red badge count
  onNotificationClick={handler}        // Callback
  onProfileClick={handler}             // Callback
/>
```

#### Usage Examples:
- All dashboards (Triage, Clinical, Physician, Manager, Admin)
- Main content area top bar
- Persistent across page navigation

---

### 2. PatientCard Component
**File**: `src/components/common/PatientCard.jsx` + `PatientCard.css`
**Purpose**: Unified patient information display card
**Status**: ✅ COMPLETE & READY TO USE

#### Features:
- Patient demographics (name, age, gender, MRN)
- Clinical indicators (ESI level, acuity score)
- Current vitals with color-coded abnormalities
- Bed/ward assignment information
- Alert flags (allergies, medications)
- Time since arrival
- Click-to-view full details

#### Props:
```jsx
<PatientCard
  patient={patientObject}              // Full patient object with vitals/clinical data
  onSelect={handler}                   // Click handler
  compact={false}                      // True for minimal view (grid display)
  showVitals={true}                    // Show/hide vitals section
/>
```

#### Patient Object Structure:
```javascript
{
  _id: string
  firstName: string
  lastName: string
  age: number
  gender: string ('M'|'F')
  mrn: string
  dateOfBirth: string
  
  // Triage/Clinical
  arrivalTime: string (ISO)
  esiLevel: 1-5
  acuityScore: 0-100
  chiefComplaint: string
  
  // Vitals
  vitals: {
    temperature: number (°F)
    heartRate: number (bpm)
    bloodPressure: string ('120/80')
    respiratoryRate: number
    oxygenSaturation: number (%)
    lastVitalTime: string (ISO)
  }
  
  // Bed Info
  currentBedId: string
  currentWard: string
  bedStatus: 'OCCUPIED'|'DISCHARGED'|'TRANSFERRED'
  
  // Clinical History
  allergies: string[]
  medications: string[]
  comorbidities: string[]
}
```

#### Usage Examples:
- Physician dashboard (3-lane case list)
- Clinical nurse ward view
- Triage patient queue
- Mobile patient roster

---

### 3. VitalsMonitor Component
**File**: `src/components/common/VitalsMonitor.jsx` + `VitalsMonitor.css`
**Purpose**: Real-time vital signs display with threshold-based alerts
**Status**: ✅ COMPLETE & READY TO USE

#### Features:
- Real-time vital displays (Temp, HR, BP, RR, O2)
- Threshold-based alerts (critical vs warning)
- Visual indicators with color coding
- Progress bars showing vital values
- Alert severity icons (🚨 vs ⚠️)
- Timestamp of last vital measurement
- Adaptive compact/full views

#### Props:
```jsx
<VitalsMonitor
  vitals={vitalsObject}                // Current vital signs
  patientId="12345"                    // For WebSocket integration (future)
  showChart={false}                    // Chart placeholder (future)
  compact={false}                      // Minimal view option
/>
```

#### Normal Vital Ranges (Used for Alerts):
```javascript
Temperature:    97°F - 99°F (normal), ±1.5°F warning, ±2°F critical
Heart Rate:     60 - 100 bpm (normal), ±20 warning, ±30 critical
Blood Pressure: 90-140 / 60-90 (normal)
Respiratory:    12 - 20 /min (normal), ±5 warning, ±10 critical
O2 Saturation:  95 - 100% (normal), <95% warning, <90% critical
```

#### Alert Types:
- **Critical** (Red): Immediately life-threatening
  - Temperature < 96°F or > 101.5°F
  - HR < 40 or > 130 bpm
  - BP > 180/120 or < 70/40
  - O2 Sat < 90%
- **Warning** (Yellow): Outside normal range but stable
  - Temperature 95°F - 97°F or 99°F - 101.5°F
  - HR 40-60 or 100-130 bpm
  - RR < 12 or > 20 /min

#### Usage Examples:
- PatientCard vitals section
- Dedicated vitals monitoring dashboard
- Clinical nurse assessment
- Physician case review

---

### 4. SidebarNavigation Component
**File**: `src/components/common/SidebarNavigation.jsx` + `SidebarNavigation.css`
**Purpose**: Role-specific sidebar menu system
**Status**: ✅ COMPLETE & READY TO USE

#### Features:
- Role-based menu structure (5 roles supported)
- Collapsible module sections
- Active page highlighting
- Expandable/collapsible sidebar
- Help, Settings, Logout buttons
- Keyboard navigation support
- Mobile-responsive (fullscreen overlay)

#### Props:
```jsx
<SidebarNavigation
  role="physician"                      // 'triage-nurse', 'clinical-nurse', 'physician', 'manager', 'admin'
  currentPage="dashboard"               // Current active page ID
  onNavigate={handler}                  // Navigation callback
  collapsed={false}                     // Collapse/expand sidebar
/>
```

#### Role-Specific Menu Structure:

**Triage Nurse:**
- Clinical: Dashboard, Patient Queue, Vitals Monitoring, Quick Assessment
- Documentation: Chief Complaint Entry, Vital Signs History, Triage Notes

**Clinical Nurse:**
- Patient Care: My Patients, Patient Details, Care Plan, Vitals
- Treatment: Medication Management, Procedures & Care, Wound Care, Patient Education
- Documentation: Nursing Notes, Flow Sheets, Discharge Planning

**Physician:**
- Cases: Active Cases, Case Details, Patient History, Escalations
- Clinical Tools: Lab Results, Imaging & Scans, Orders Management, Diagnostics
- Department: Wards & Beds, Care Teams, Patient Handoff

**Manager:**
- Operations: KPI Dashboard, Performance Metrics, Wait Time Analysis
- Staff Management: Staff Directory, Shift Scheduling, Payroll & HR, Training
- Facility: Bed Management, Equipment Tracking, Maintenance Requests, Supply Chain
- Analytics: Reports, Trend Analysis, Quality Metrics

**Administrator:**
- System: System Dashboard, User Management, Roles & Permissions, Audit Logs
- Configuration: System Settings, Data Management, Integrations, Backups
- Financial: Billing & Claims, Invoices, Insurance Management, Payments
- Facility: Departments, Bed Configuration, Equipment Management, Supply Chain

#### Usage Examples:
- Wrap all dashboard content as main navigation
- MainLayout container integration
- Role-based page rendering

---

### 5. MainLayout Component
**File**: `src/components/layouts/MainLayout.jsx` + `MainLayout.css`
**Purpose**: Professional full-page layout combining all navigation and content
**Status**: ✅ COMPLETE & READY TO USE

#### Features:
- Combines MainNavigation + SidebarNavigation + Content Area
- Breadcrumb navigation
- Sidebar toggle (collapse/expand)
- Mobile overlay for sidebar
- Header action bar
- Responsive page structure
- Print-friendly styling

#### Props:
```jsx
<MainLayout
  role="physician"                      // User role
  currentPage="dashboard"               // Current page/section
  onNavigate={handler}                  // Navigation callback
  userName="Dr. Sarah Johnson"          // Display name
  userRole="Physician"                  // Role display
  userAvatar="👨‍⚕️"                     // Avatar emoji
>
  {/* Page content */}
  <PhysicianDashboard />
</MainLayout>
```

#### Layout Structure:
```
┌─────────────────────────────────┐
│   MainNavigation (Top)           │
├──────────┬──────────────────────┤
│          │                      │
│ Sidebar  │   Content Area       │
│          │   ┌────────────────┐ │
│          │   │ Breadcrumbs    │ │
│          │   │ Header Actions │ │
│          │   ├────────────────┤ │
│          │   │  Page Content  │ │
│          │   │  (Scrollable)  │ │
│          │   │                │ │
│          │   └────────────────┘ │
└──────────┴──────────────────────┘
```

#### Usage Examples:
```jsx
// Physician Dashboard
<MainLayout role="physician" currentPage="dashboard">
  <PhysicianDashboard />
</MainLayout>

// Clinical Nurse Ward
<MainLayout role="clinical-nurse" currentPage="my-patients">
  <ClinicalNurseDashboard />
</MainLayout>

// Manager Analytics
<MainLayout role="manager" currentPage="kpi-dashboard">
  <ManagerDashboard />
</MainLayout>
```

---

## Import Pattern

### Single Component Import:
```jsx
import PatientCard from '../components/common/PatientCard';
import { MainLayout } from '../components/layouts';
```

### Index-Based Import (Recommended):
```jsx
import { PatientCard, VitalsMonitor, MainNavigation, SidebarNavigation } from '../components/common';
import { MainLayout } from '../components/layouts';
```

---

## Styling & Theming

### Hospital-Grade Color Scheme:
```css
--primary-blue: #1E40AF (actions, links, navigation)
--danger-red: #DC2626 (critical alerts, errors)
--success-green: #059669 (normal vitals, confirmations)
--warning-orange: #F59E0B (alerts, precautions)
--neutral-gray: #6B7280 (text, borders, secondary)
--background-light: #F9FAFB (page background)
--surface-white: #FFFFFF (cards, containers)
```

### Responsive Breakpoints:
```css
Mobile:     < 768px
Tablet:     768px - 1024px
Desktop:    > 1024px
Large:      > 1400px
```

### Component CSS Files:
- `MainNavigation.css` (150 lines)
- `PatientCard.css` (250 lines)
- `VitalsMonitor.css` (280 lines)
- `SidebarNavigation.css` (240 lines)
- `MainLayout.css` (180 lines)

All components include:
- Dark mode support (via prefers-color-scheme)
- Accessibility features (focus states, ARIA labels)
- Mobile responsiveness (flex layouts, media queries)
- Print styles (for clinical summaries/reports)

---

## Integration Guide

### Step 1: Wrap Dashboard in MainLayout
```jsx
// PhysicianDashboard.jsx
import { MainLayout } from '../components/layouts';

function PhysicianDashboard() {
  return (
    <MainLayout 
      role="physician" 
      currentPage="dashboard"
      userName="Dr. Sarah Johnson"
      userRole="Physician"
    >
      {/* Dashboard content below */}
      <div className="dashboard-content">
        {/* ... */}
      </div>
    </MainLayout>
  );
}
```

### Step 2: Use PatientCard in List View
```jsx
<div className="patient-grid">
  {patients.map(patient => (
    <PatientCard 
      key={patient._id}
      patient={patient}
      onSelect={() => handleSelectPatient(patient._id)}
      compact={isCompactView}
    />
  ))}
</div>
```

### Step 3: Embed VitalsMonitor in Patient Details
```jsx
<div className="patient-details">
  <PatientCard patient={patient} showVitals={false} />
  <VitalsMonitor vitals={patient.vitals} />
</div>
```

---

## Next Steps

### Pending Components:
1. ⏹️ Modal Component (reusable pop-ups)
2. ⏹️ Alert Banner Component (notifications)
3. ⏹️ Form Components (inputs, selects, date pickers)
4. ⏹️ Table Component (patient lists, med history)
5. ⏹️ Chart Components (vitals trends, analytics)

### Pending Integrations:
1. ⏹️ Update PhysicianDashboard to use MainLayout
2. ⏹️ Update TriageNurseDashboard to use MainLayout
3. ⏹️ Update ClinicalNurseDashboard to use MainLayout
4. ⏹️ Create ManagerDashboard with MainLayout
5. ⏹️ Create AdminDashboard with MainLayout

### Future Enhancements:
1. Real-time vitals via WebSocket (patient monitoring)
2. Chart integration (vitals trends, analytics graphs)
3. Voice/audio intake for Triage (accessibility)
4. Barcode/QR scanning (patient identification)
5. Multi-touch gestures (tablet optimization)

---

**Component Library Version**: 1.0
**Created**: April 25, 2026
**Status**: Production Ready ✅
**Test Coverage**: Components tested for responsive design and accessibility (A11y)
