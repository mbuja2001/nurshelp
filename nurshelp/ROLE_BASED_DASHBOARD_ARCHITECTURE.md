# Role-Based Dashboard Architecture & Real-Time Triage Quality Dashboard

## Overview
The dashboard system is **context-aware** and renders different views based on:
1. **Nurse's Current Role** (from staff profile + shift assignment)
2. **Task Mode** (determined by shift + department + assigned role)
3. **Current Shift State** (clocked in → clinical care → finalized)

---

## 1. ROLE-BASED DASHBOARD ARCHITECTURE

### 1.1 Dashboard Partitioning by Role

```
Every nurse can have MULTIPLE roles (interchangeable):
┌─────────────────────────────────────────┐
│ Nurse Profile: Sarah Smith              │
├─────────────────────────────────────────┤
│ Primary Role: Triage Nurse              │
│ Secondary Roles:                        │
│  • Clinical Care Provider               │
│  • Bed Coordinator                      │
│  • Training Lead (Senior)               │
│                                         │
│ Today's Shift Assignment:               │
│  • Department: Emergency Department     │
│  • Role: Triage (Assigned: 08:00-14:00)│
│  • Role: Clinical Care (As needed)      │
│                                         │
│ Current Status: CLOCKED IN              │
│ Current Task Mode: TRIAGE               │
└─────────────────────────────────────────┘
```

### 1.2 Task Mode Determination Logic

```javascript
// Location: src/services/TaskModeService.js

class TaskModeService {
  determineTaskMode(nurse, currentShift, currentAssignment) {
    // Read from: Shift, NurseAssignment, current time
    
    const taskMode = {
      primary: null,      // TRIAGE | CLINICAL_CARE | ADMIN
      secondary: null,    // Can run parallel
      department: null,
      ward: null,
      reason: null,       // Why this mode?
    };
    
    // Step 1: Is there a current assignment for triage?
    if (currentAssignment?.role === 'TRIAGE' && isTimeInShift()) {
      taskMode.primary = 'TRIAGE';
      taskMode.department = currentAssignment.department;
      taskMode.ward = currentAssignment.ward;
      taskMode.reason = 'Assigned to triage in ED';
    }
    
    // Step 2: Non-triage assignment?
    else if (currentAssignment?.role === 'CLINICAL_CARE') {
      taskMode.primary = 'CLINICAL_CARE';
      taskMode.department = currentAssignment.department;
      taskMode.ward = currentAssignment.ward;
      taskMode.reason = 'Assigned to patient care';
    }
    
    // Step 3: Both triage AND clinical care in ED?
    // (Can switch between modes depending on queue depth)
    else if (currentAssignment?.flexRoles?.includes('TRIAGE', 'CLINICAL_CARE')) {
      taskMode.primary = 'TRIAGE';        // Default
      taskMode.secondary = 'CLINICAL_CARE'; // Can jump to if urgent
      taskMode.reason = 'Cross-trained in ED';
    }
    
    // Step 4: Administrative or management?
    else if (nurse.roleLevel === 'SENIOR' || nurse.roleLevel === 'MANAGER') {
      taskMode.primary = 'ADMIN_DASHBOARD';
      taskMode.reason = 'Senior staff visibility mode';
    }
    
    return taskMode;
  }
}
```

### 1.3 Role → Dashboard Mapping

```
┌─────────────────────────────────────────────────────────────┐
│ ROLE-BASED DASHBOARD ROUTING                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ TRIAGE MODE:                                               │
│ Role: Junior/Mid Nurse assigned to Triage                  │
│ Shows: Triage Queue → SATS Input → Assign                  │
│ Dashboard: "Triage Quality Dashboard" (Sprint 1)           │
│ Metrics: Avg Time, Data Completeness, Confidence          │
│                                                             │
│ CLINICAL CARE MODE:                                         │
│ Role: Junior/Mid Nurse assigned to Ward                    │
│ Shows: "My Patients" → Vitals → Procedures → Escalate     │
│ Dashboard: "Clinical Care Mode" (from plan)               │
│ Metrics: Treatment time, Response latency, Procedures     │
│                                                             │
│ SENIOR NURSE / CHARGE NURSE MODE:                         │
│ Role: Senior Nurse managing shift                          │
│ Shows: Entire ED overview + both metrics                   │
│ Dashboard: "Ward Operations Dashboard"                     │
│ Metrics: Queue depth, Triage backlog, Staffing gaps       │
│                                                             │
│ PHYSICIAN MODE:                                             │
│ Role: Doctor                                               │
│ Shows: Escalation alerts + patient queue                   │
│ Dashboard: "Physician Alert Tray" + "Patient Queue"       │
│ Metrics: Escalation response times, Case complexity       │
│                                                             │
│ MANAGEMENT MODE:                                            │
│ Role: Hospital Manager / Administrator                     │
│ Shows: Cross-ward metrics + analytics                      │
│ Dashboard: "Hospital Operations Dashboard"                 │
│ Metrics: Hospital Pulse, Waste, Billing, Staff Efficiency │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. TRIAGE QUALITY DASHBOARD (Sprint 1 - Triage Mode)

### 2.1 Who Sees It

```
Roles that see Triage Quality Dashboard:
✓ Triage Nurse (Junior/Mid) - Their own metrics + department
✓ Charge Nurse (Senior) - All triage nurses in the ward
✓ Manager - All wards triage metrics
✗ Physician - Does NOT see this (different dashboard)
✗ Bed Coordinator - Does NOT see this (different dashboard)
```

### 2.2 Real-Time Triage Quality Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│ 🚑 TRIAGE QUALITY DASHBOARD - Emergency Department          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ [Mode: TRIAGE]  [Department: ED]  [Shift: 08:00-14:00]    │
│                                                             │
│ ┌─ REAL-TIME PERFORMANCE ──────────────────────────────┐   │
│ │                                                      │   │
│ │ 📊 TODAY'S TRIAGE SUMMARY (47 triages):             │   │
│ │ ────────────────────────────────────────────────   │   │
│ │ Average Triage Time: 8.5 min  ✓ (target: <10min)  │   │
│ │ Data Completeness: 94%        ✓ (target: >90%)    │   │
│ │ Average Confidence: 4.2/5      ✓ (target: >4.0)    │   │
│ │                                                      │   │
│ │ Patients In Queue: 3                                │   │
│ │ Waiting > 5 min: 1                                  │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─ MISTRIAGE DETECTION ────────────────────────────────┐   │
│ │                                                      │   │
│ │ 🚨 Retriages within 10min: 2 (4.3%)               │   │
│ │    • Patient 4521: GREEN → ORANGE (vitals change) │   │
│ │    • Patient 4542: YELLOW → RED (new symptom)     │   │
│ │                                                      │   │
│ │ Early Correction = GOOD (nurse caught own error)   │   │
│ │ Trend: 2-3 per day (normal)                        │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─ BY SKILL LEVEL (ED Shift) ──────────────────────────┐   │
│ │                                                      │   │
│ │ SENIOR (2 nurses):                                  │   │
│ │ • Avg Time: 7.2 min  | Confidence: 4.8/5          │   │
│ │ • Data Completeness: 97%  ✓                         │   │
│ │ • Triage: 18 patients                               │   │
│ │                                                      │   │
│ │ MID-LEVEL (2 nurses):                               │   │
│ │ • Avg Time: 8.8 min  | Confidence: 4.1/5          │   │
│ │ • Data Completeness: 93%  ✓                         │   │
│ │ • Triaged: 20 patients                              │   │
│ │                                                      │   │
│ │ JUNIOR (1 nurse):                                   │   │
│ │ • Avg Time: 11.3 min | Confidence: 3.6/5  ⚠️       │   │
│ │ • Data Completeness: 88%  ⚠️ (below target)        │   │
│ │ • Triaged: 9 patients                               │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─ TRIAGE TIME BREAKDOWN ──────────────────────────────┐   │
│ │                                                      │   │
│ │ Component Avg Times (within triage duration):      │   │
│ │ • Opening chart: 1.2 min                            │   │
│ │ • Vitals entry: 2.4 min                             │   │
│ │ • Assessment: 3.1 min                               │   │
│ │ • Triage level decision: 1.8 min                    │   │
│ │ ─────────────────                                   │   │
│ │ Total: 8.5 min                                      │   │
│ │                                                      │   │
│ │ Slowest step: Assessment (3.1 min)                  │   │
│ │ Optimization: Less template text? Simpler UI?      │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ [Current Patient Queue: 3] [Refresh: Auto 30s] [More Stats]│
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Triage Quality Metrics Breakdown

```javascript
// What each metric tracks and why it matters

METRIC 1: Average Triage Time
├─ What: Mean time from triageStartTime → triageCompleteTime
├─ Why: Identifies throughput bottlenecks
├─ Target: < 10 minutes per patient (SA standard)
├─ Alert: If > 15 min, investigate
└─ By: Per nurse, per shift, per department

METRIC 2: Data Completeness %
├─ What: (Fields filled / Total required fields) × 100
├─ Required: BP, HR, Temp, SpO2, RR, SATS score, chief complaint
├─ Why: Missing data = poor triage quality
├─ Target: > 90%
├─ Alert: If < 85%, flag for data entry training
└─ By: Per nurse, trends over time

METRIC 3: Average Confidence Score
├─ What: Mean of nurse confidence slider (1-5) per triage
├─ Why: Predicts mistriage risk (low confidence = higher error rate)
├─ Target: > 4.0
├─ Alert: If < 3.5, nurse may need support
├─ Insight: If confident but data incomplete = data entry issue
└─ By: Per nurse, per shift, by experience level

METRIC 4: Mistriage Detection (Retriages within 10 min)
├─ What: Count of patients reassessed within 10 min of triage
├─ Why: Early mistriage catch = patient safety + data quality signal
├─ Breakdown:
│  • "UPGRADE" (nurse chose less serious): Good sign (nurse corrects)
│  • "DOWNGRADE" (nurse chose more serious): Could be over-caution or legitimate
├─ Target: 2-3% (some correction is normal)
├─ Alert: If > 5%, triage process may need review
└─ By: Per nurse, per shift

METRIC 5: Time to Bed Assignment (Secondary)
├─ What: Time from triageCompleteTime → bed assignment
├─ Why: Identifies bottleneck after triage
├─ Not in triage dashboard but important for workflow
└─ Target: < 5 minutes (in Clinical Care Mode dashboard)
```

### 2.4 Data Flow: How Metrics Are Calculated

```
TRIAGE FORM SUBMISSION
       ↓
POST /api/triage/:triageId/complete
       ↓
Backend calculates:
  • triageDuration = triageCompleteTime - triageStartTime
  • dataCompleteness = (fieldsWithValue / requiredFields) × 100
  • Stores in Database: nurseTriage document
       ↓
Real-time aggregation (runs every 30 seconds):
  • Query all triages in last 24 hours
  • Group by: Department, Nurse, Shift
  • Calculate: Avg time, avg completeness, avg confidence
  • Store in: TriageQualityMetrics collection
       ↓
Frontend polls every 30s:
  GET /api/analytics/triage/quality
       ↓
Dashboard updates with:
  • New metrics
  • Performance alerts
  • Comparison charts
```

### 2.5 Alerts & Auto-Actions

```javascript
// Context-aware alerts based on role & task mode

ALERT 1: Data Completeness Below Target
├─ Trigger: dataCompleteness < 85% for nurse
├─ Who sees: Charge Nurse, Manager
├─ Action: Auto-suggest "Data Entry Training"
└─ Auto-escalate if: Persists for 3+ triages

ALERT 2: Triage Time Exceeding Threshold
├─ Trigger: triageDuration > 15 minutes
├─ Who sees: Charge Nurse in real-time
├─ Action: "Check on Nurse - possible blocker?"
└─ Auto-escalate if: Multiple patients backing up

ALERT 3: Low Confidence + Mistriage Pattern
├─ Trigger: avgConfidence < 3.5 AND retriageRate > 5%
├─ Who sees: Senior Nurse + Manager
├─ Action: "Offer clinical support to nurse"
└─ Insight: Junior staff struggling with case complexity?

ALERT 4: Mistriage Detection (Self-Correction)
├─ Trigger: Patient retriaged within 10 minutes
├─ Who sees: Nurse (confirmation), Charge Nurse (oversight)
├─ Action: Log as positive (nurse catching own error)
└─ Analytics: "Self-Correction Event" for quality tracking
```

---

## 3. ROLE-BASED DASHBOARD ROUTING (Context-Aware)

### 3.1 Frontend Component Architecture

```javascript
// Location: src/Dashboard/Pages/DashboardRouter.jsx

import React, { useContext, useEffect, useState } from 'react';
import { StaffStateContext } from '../context/StaffStateProvider';
import TriageQualityDashboard from './TriageQualityDashboard';
import ClinicalCareModeDashboard from './ClinicalCareMode';
import WardOperationsDashboard from './WardOperationsDashboard';
import PhysicianDashboard from './PhysicianDashboard';
import ManagementDashboard from './ManagementDashboard';

export const DashboardRouter = () => {
  const { mode, taskMode, userRole, department } = useContext(StaffStateContext);
  const [dashboardComponent, setDashboardComponent] = useState(null);
  
  useEffect(() => {
    // Determine which dashboard to render based on context
    
    if (mode === 'TRIAGE') {
      if (userRole === 'SENIOR' || userRole === 'MANAGER') {
        // Senior nurses see overview + all triage metrics
        setDashboardComponent(<TriageQualityDashboard scope="DEPARTMENT" />);
      } else {
        // Junior/Mid nurses see personal metrics
        setDashboardComponent(<TriageQualityDashboard scope="PERSONAL" />);
      }
    }
    
    else if (mode === 'CLINICAL_CARE') {
      setDashboardComponent(<ClinicalCareModeDashboard />);
    }
    
    else if (userRole === 'SENIOR' || taskMode.primary === 'ADMIN') {
      setDashboardComponent(<WardOperationsDashboard department={department} />);
    }
    
    else if (userRole === 'PHYSICIAN') {
      setDashboardComponent(<PhysicianDashboard />);
    }
    
    else if (userRole === 'MANAGER') {
      setDashboardComponent(<ManagementDashboard />);
    }
    
  }, [mode, taskMode, userRole]);
  
  return dashboardComponent;
};
```

### 3.2 StaffStateProvider Enhancement (with Task Mode)

```javascript
// Location: src/context/StaffStateProvider.jsx (Enhanced)

export const StaffStateProvider = ({ children }) => {
  const [mode, setMode] = useState('TRIAGE');  // TRIAGE, CLINICAL_CARE, ADMIN
  const [taskMode, setTaskMode] = useState(null);  // ← NEW
  const [userRole, setUserRole] = useState(null);
  const [department, setDepartment] = useState(null);
  const [activePatient, setActivePatient] = useState(null);
  
  // On login or shift change
  useEffect(() => {
    const computedTaskMode = TaskModeService.determineTaskMode(
      currentNurse,
      currentShift,
      currentAssignment
    );
    
    setTaskMode(computedTaskMode);
    setMode(computedTaskMode.primary);
    setDepartment(computedTaskMode.department);
  }, [currentShift, currentAssignment]);
  
  return (
    <StaffStateContext.Provider
      value={{
        mode,
        setMode,
        taskMode,      // ← NEW
        userRole,
        department,
        activePatient,
        setActivePatient,
      }}
    >
      {children}
    </StaffStateContext.Provider>
  );
};
```

---

## 4. TRIAGE QUALITY DASHBOARD COMPONENT (Detailed)

### 4.1 Component File Structure

```
src/Dashboard/Components/TriageQuality/
├── TriageQualityDashboard.jsx           (main container)
├── RealTimePerformance.jsx              (metrics card)
├── MistriageDetectionPanel.jsx          (mistriage alerts)
├── SkillLevelComparison.jsx             (by experience)
├── TriageTimeBreakdown.jsx              (component times)
├── AlertsPanel.jsx                      (auto-alerts)
└── hooks/
    └── useTriageMetrics.js              (data fetching)
```

### 4.2 Main Component Logic

```javascript
// Location: src/Dashboard/Components/TriageQuality/TriageQualityDashboard.jsx

import React, { useState, useEffect, useContext } from 'react';
import { StaffStateContext } from '../../context/StaffStateProvider';
import { useTriageMetrics } from './hooks/useTriageMetrics';

export const TriageQualityDashboard = ({ scope = 'PERSONAL' }) => {
  // scope: 'PERSONAL' (just my metrics) or 'DEPARTMENT' (all staff)
  
  const { userRole, department } = useContext(StaffStateContext);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Custom hook fetches metrics every 30 seconds
  const { 
    realTimeMetrics, 
    mistriages,
    byExperience,
    timeBreakdown,
  } = useTriageMetrics(scope, department);
  
  useEffect(() => {
    setMetrics({
      realTime: realTimeMetrics,
      mistriages,
      byExperience,
      timeBreakdown,
    });
    setLoading(false);
  }, [realTimeMetrics]);
  
  if (loading) return <div>Loading dashboard...</div>;
  
  return (
    <div className="triage-quality-dashboard">
      
      {/* Header */}
      <header className="dashboard-header">
        <h1>🚑 Triage Quality Dashboard</h1>
        <div className="context-info">
          <span>Department: {department}</span>
          <span>Mode: TRIAGE</span>
          <span>Shift: {getCurrentShift()}</span>
          <span>Auto-refresh: 30s</span>
        </div>
      </header>
      
      {/* Section 1: Real-Time Performance */}
      <section className="performance-section">
        <RealTimePerformance metrics={metrics.realTime} />
      </section>
      
      {/* Section 2: Mistriage Detection */}
      <section className="mistriage-section">
        <MistriageDetectionPanel mistriages={metrics.mistriages} />
      </section>
      
      {/* Section 3: By Skill Level (only if DEPARTMENT scope) */}
      {scope === 'DEPARTMENT' && (
        <section className="experience-section">
          <SkillLevelComparison data={metrics.byExperience} />
        </section>
      )}
      
      {/* Section 4: Triage Time Breakdown */}
      <section className="breakdown-section">
        <TriageTimeBreakdown data={metrics.timeBreakdown} />
      </section>
      
      {/* Section 5: Real-time Alerts */}
      <section className="alerts-section">
        <AlertsPanel 
          completenessAlerts={getCompletenessAlerts()}
          timeAlerts={getTimeAlerts()}
          confidenceAlerts={getConfidenceAlerts()}
        />
      </section>
      
    </div>
  );
};
```

### 4.3 Real-Time Performance Component

```javascript
// Location: src/Dashboard/Components/TriageQuality/RealTimePerformance.jsx

export const RealTimePerformance = ({ metrics }) => {
  
  const getStatusColor = (value, target) => {
    if (value >= target) return 'green';
    if (value >= target * 0.95) return 'yellow';
    return 'red';
  };
  
  return (
    <div className="performance-card">
      <h3>📊 Today's Triage Summary</h3>
      
      <div className="metrics-grid">
        
        {/* Metric 1: Avg Triage Time */}
        <div className="metric-box">
          <div className="metric-label">Average Triage Time</div>
          <div className="metric-value" style={{
            color: getStatusColor(metrics.avgTriageTime, 10)
          }}>
            {metrics.avgTriageTime.toFixed(1)} min
          </div>
          <div className="metric-target">Target: &lt;10 min</div>
          <div className="metric-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${Math.min(100, (metrics.avgTriageTime / 10) * 100)}%` }}
              />
            </div>
          </div>
        </div>
        
        {/* Metric 2: Data Completeness */}
        <div className="metric-box">
          <div className="metric-label">Data Completeness</div>
          <div className="metric-value" style={{
            color: getStatusColor(metrics.dataCompleteness, 90)
          }}>
            {metrics.dataCompleteness.toFixed(1)}%
          </div>
          <div className="metric-target">Target: &gt;90%</div>
          <div className="metric-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${metrics.dataCompleteness}%` }}
              />
            </div>
          </div>
        </div>
        
        {/* Metric 3: Average Confidence */}
        <div className="metric-box">
          <div className="metric-label">Average Confidence</div>
          <div className="metric-value" style={{
            color: getStatusColor(metrics.avgConfidence * 20, 80)  // Convert 1-5 to 0-100
          }}>
            {metrics.avgConfidence.toFixed(1)}/5 ⭐
          </div>
          <div className="metric-target">Target: &gt;4.0</div>
          <div className="metric-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${(metrics.avgConfidence / 5) * 100}%` }}
              />
            </div>
          </div>
        </div>
        
        {/* Additional Metrics */}
        <div className="metric-box">
          <div className="metric-label">Patients Triaged Today</div>
          <div className="metric-value">{metrics.patientCount}</div>
        </div>
        
        <div className="metric-box">
          <div className="metric-label">Currently in Queue</div>
          <div className="metric-value" style={{
            color: metrics.queueDepth > 5 ? 'red' : 'green'
          }}>
            {metrics.queueDepth}
          </div>
        </div>
        
      </div>
      
    </div>
  );
};
```

---

## 5. INTEGRATION WITH EXISTING PLANS

### 5.1 How This Fits into Sprint 1

```
SPRINT 1: Perfect Object 2 (Nurse Triage) + Analytics Foundation

Week 1:
  ✓ Enhance nurseTriage.model.js with timestamps
  ✓ Create TriageQualityMetrics collection
  ✓ Build POST /api/triage/complete endpoint
  ✓ Build GET /api/analytics/triage/quality endpoint

Week 2:
  ✓ Build TriageQualityDashboard component (THIS DOCUMENT)
  ✓ Integrate with StaffStateProvider (task mode logic)
  ✓ Build real-time metrics hooks
  ✓ Build mistriage detection panel
  ✓ Deploy dashboard to nurses
  
OUTCOME: Every triage now measured, every nurse sees their metrics
```

### 5.2 How This Integrates with Clinical Care Mode Plan

```
FLOW:
1. Nurse triages patient (Uses: TriageQualityDashboard, Mode=TRIAGE)
2. Nurse assigns patient to bed
3. Nurse switches mode to CLINICAL_CARE
4. Dashboard switches to: ClinicalCareModeDashboard (from CLINICAL_CARE_MODE_PLAN.md)
5. Nurse completes treatment
6. Nurse hits "Finalise & Sync"
7. Objects 1-3 bundled for physician review

Task Mode Switching:
├─ Mode = TRIAGE (User in TriageQualityDashboard)
├─ Mode = CLINICAL_CARE (User in ClinicalCareModeDashboard)
└─ Mode = ADMIN (User in WardOperationsDashboard or ManagementDashboard)
```

### 5.3 How This Integrates with Analytics-First Roadmap

```
SPRINT 1 DELIVERABLE (from ANALYTICS_FIRST_ROADMAP.md):
"By end of week 2, have real-time triage quality metrics"

THIS DOCUMENT provides:
✓ Real-Time Triage Quality Dashboard (the visualization)
✓ Role-based routing (TRIAGE mode → this dashboard)
✓ Data completeness tracking
✓ Mistriage detection
✓ Experience level comparison
✓ Integration with existing shift/assignment system

FLOWS INTO Sprint 2:
Once Object 2 perfected, add Object 1 (AI) comparison
→ TriageAccuracyDashboard (from ANALYTICS_FIRST_ROADMAP.md)
```

---

## 6. DATA COLLECTION & METRICS CALCULATION

### 6.1 Required Database Changes (Sprint 1)

```javascript
// Enhance: src/models/nurseTriage.model.js

Add these fields:
{
  // Timing hooks (for metrics)
  triageStartTime: Date,        // When nurse opens triage form
  vitalsRecordedTime: Date,     // When vitals section completed
  assessmentTime: Date,         // When assessment written
  triageCompleteTime: Date,     // When nurse clicks "Assign"
  
  // Metrics at submission
  triageDuration: Number,       // ms (calculated on submit)
  dataCompleteness: Number,     // 0-100%
  fieldsWithValue: Number,      // Count of filled fields
  requiredFields: Number,       // Count of required fields
  vitalsValidity: Boolean,      // Are ranges valid?
  
  // Confidence flag
  nurseConfidenceScore: Number, // 1-5 (from slider)
  
  // Experience level (from nurse profile at time of triage)
  triageNurseExperience: String, // "JUNIOR" | "MID" | "SENIOR"
}
```

### 6.2 New Collection: TriageQualityMetrics

```javascript
// Location: src/models/triageQualityMetrics.model.js (NEW)

{
  _id: ObjectId,
  hospitalId: ObjectId,
  departmentId: ObjectId | String,  // "ED" | "ICU" | "GENERAL"
  
  // Scope (for granularity)
  scope: "SHIFT" | "DAILY" | "WEEKLY",
  scopeDate: Date,                  // Date of shift/day
  shiftTime: String,                // "08:00-14:00" (if shift scope)
  
  // Real-time metrics
  metrics: {
    total_triages: Number,
    avg_triage_duration: Number,   // minutes
    median_triage_duration: Number,
    p95_triage_duration: Number,
    
    avg_data_completeness: Number,  // %
    vital_invalidity_count: Number,
    
    avg_nurse_confidence: Number,   // 1-5 scale
    
    queue_depth_current: Number,
    patients_waiting_gt_5min: Number,
  },
  
  // Mistriage data
  mistriage: {
    retriages_within_10min: Number,
    retriage_rate: Number,          // %
    upgrades: Number,               // Nurse corrected to more serious
    downgrades: Number,             // Nurse corrected to less serious
  },
  
  // By experience
  by_experience: {
    JUNIOR: {
      avg_duration: Number,
      avg_confidence: Number,
      data_completeness: Number,
      count: Number,
    },
    MID: { ... },
    SENIOR: { ... },
  },
  
  // By staff member (if daily scope)
  by_staff: [
    {
      staffId: ObjectId,
      staffName: String,
      experience: String,
      triages: Number,
      avg_duration: Number,
      avg_confidence: Number,
      data_completeness: Number,
    }
  ],
  
  // Calculated at
  calculatedAt: Date,
  refreshInterval: "30s",  // How often this is updated
}
```

### 6.3 API Endpoints for Sprint 1

```javascript
// Triage Quality Endpoints

POST /api/triage/start
  Body: { encounterId, patientId, wardId, triageNurseId }
  Response: { triageSessionId, startTime }
  
POST /api/triage/:triageId/vitals
  Body: { vitals: {...}, recordedAt }
  Response: { satsScore, validity: Boolean }

POST /api/triage/:triageId/complete
  Body: { 
    triageLevel, 
    triageReasoning,
    confidenceScore,
    assignedBedId,
  }
  Response: {
    triageId,
    triageDuration,
    dataCompleteness,
    metrics: { ... }
  }
  Action: 
    - Calculate all metrics
    - Create nurseTriage document
    - Create/update TriageQualityMetrics
    - Check for mistriage (auto-trigger comparison)

GET /api/analytics/triage/quality
  PARAMS: 
    - department (required, e.g., "ED")
    - scope: "PERSONAL" | "DEPARTMENT" (required)
    - timeRange: "24h" | "7d" | "30d" (optional, default 24h)
    - wardId (optional, for ward-level metrics)
  
  Response (if PERSONAL):
  {
    staff_id: ObjectId,
    metrics: {
      avg_triage_duration: Number,
      data_completeness: Number,
      avg_confidence: Number,
      today_triages: Number,
      current_queue: Number,
    },
    mistriage: { ... },
  }
  
  Response (if DEPARTMENT):
  {
    department: String,
    metrics: { ... },
    by_experience: { ... },
    by_staff: [ ... ],
    alerts: [ ... ],
  }

GET /api/analytics/triage/mistriage
  PARAMS: timeRange, departmentId
  Response: {
    retriages: [
      {
        patientId: ObjectId,
        originalTriageLevel: String,
        newTriageLevel: String,
        reason: String,
        timeToRetriage: Number,  // minutes
      }
    ],
    summary: { ... }
  }
```

---

## 7. SUMMARY: Role-Based Dashboard Architecture

```
┌─────────────────────────────────────────────────────────┐
│ DASHBOARD ROUTING BY ROLE + TASK MODE                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ NURSE (Triage Mode):                                    │
│ ├─ Task Mode Determined By: Shift + Assignment        │
│ ├─ Primary Mode: TRIAGE                                │
│ ├─ Dashboard: TriageQualityDashboard                   │
│ ├─ Scope: PERSONAL (their own metrics)                 │
│ ├─ Metrics: Time, Completeness, Confidence, Mistriages│
│ └─ Updates: Every 30 seconds                           │
│                                                         │
│ NURSE (Clinical Care Mode):                             │
│ ├─ Task Mode Determined By: Shift + Assignment        │
│ ├─ Primary Mode: CLINICAL_CARE                         │
│ ├─ Dashboard: ClinicalCareModeDashboard                │
│ ├─ Scope: PERSONAL ("My Patients")                     │
│ ├─ Metrics: Vitals, Procedures, Escalations, Duration │
│ └─ Updates: Real-time                                  │
│                                                         │
│ CHARGE NURSE / SENIOR:                                  │
│ ├─ Task Mode: Both TRIAGE + ADMIN                      │
│ ├─ Dashboard: TriageQualityDashboard (DEPARTMENT scope)│
│ ├─ Scope: DEPARTMENT (all triage metrics)              │
│ ├─ Metrics: All staff performance, alerts, benchmarks  │
│ └─ Can see: Individual nurse performance + trends      │
│                                                         │
│ MANAGER / DIRECTOR:                                     │
│ ├─ Task Mode: ADMIN_DASHBOARD                          │
│ ├─ Dashboard: ManagementDashboard                      │
│ ├─ Scope: HOSPITAL-WIDE                                │
│ ├─ Metrics: All wards, analytics, billing, waste       │
│ └─ Can see: Cross-ward trends + Hospital Pulse         │
│                                                         │
│ PHYSICIAN:                                              │
│ ├─ Task Mode: ESCALATION_ALERT                         │
│ ├─ Dashboard: PhysicianAlertTray + PatientQueue        │
│ ├─ Scope: ESCALATIONS ASSIGNED TO THEM                 │
│ ├─ Metrics: Alert response time, case complexity       │
│ └─ Can see: Pending reviews + nurse escalations        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

This architecture ensures:
✅ **Role-awareness**: Different people see different metrics
✅ **Task-mode awareness**: Same nurse sees different dashboards based on shift assignment
✅ **Real-time**: Metrics update every 30 seconds for triage dashboard
✅ **Actionable**: Each dashboard has alerts and optimization suggestions
✅ **Scalable**: Can add new roles/modes without breaking existing dashboards
