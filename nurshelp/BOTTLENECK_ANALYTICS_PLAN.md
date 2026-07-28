# Bottleneck Analytics: Bed Turnaround Time (BTT) Implementation Plan

## Overview
The "Bottleneck Analytics" system tracks resource waste by measuring the gap between patient discharge and next triage assignment. This identifies communication lag, inefficient cleaning workflows, and bottlenecks that cause admitted patients to wait despite available beds.

---

## 1. THE KILLER METRIC: Bed Turnaround Time (BTT)

### Definition
**BTT = Time from Physician Feedback (Discharge) → Next Triage Assignment**

### Calculation Formula
```
BTT = Triage Assignment Time - Physician Discharge Feedback Time
```

### Key Timeline Points
1. **T0**: Physician marks patient for discharge (via Feedback/EHR)
2. **T1**: Bed status changes to "CLEANING" (nursing action)
3. **T2**: Bed cleaned and marked "AVAILABLE"
4. **T3**: New patient triaged and assigned to bed
5. **BTT = T3 - T0** (Total turnaround time in minutes/hours)

### BTT Components (Breakdown)
- **Discharge Lag**: T1 - T0 (delay before cleaning starts)
- **Cleaning Duration**: T2 - T1 (actual bed cleaning time)
- **Assignment Lag**: T3 - T2 (delay before new patient assigned despite bed available)

---

## 2. THE PROBLEM: Resource Waste in SA Hospitals

### Current Scenario
- Patients waiting in street/outpatient area
- Beds sit empty and clean
- Physicians completed discharge
- Nobody coordinating the handoff
- **Result**: 45+ minutes of wasted bed capacity while Orange/Red patients wait

### Real-World Impact
```
Emergency Department Example:
- 10 Orange triage patients waiting
- 4 available beds ready for occupancy
- Average BTT: 45-60 minutes per discharge
- Daily waste: 3-4 hours of bed capacity
- Annual patient harm: Delayed critical care for ~200+ patients
```

---

## 3. ANALYTICS DASHBOARD: What to Display

### 3.1 Real-Time BTT Monitoring
```
Ward View:
┌─────────────────────────────────────────┐
│ ED Ward - BTT Analytics                  │
├─────────────────────────────────────────┤
│ Active Discharges: 3                     │
│ Beds in Cleaning: 2                      │
│ Available Beds: 4                        │
│ Patients Waiting: 8                      │
│                                          │
│ ⚠️ BOTTLENECK ALERT:                    │
│    "Bed 4 vacant 45min | 10 waiting"    │
│    "ED has 4 available beds but 8       │
│     patients waiting - BTT avg 52min"   │
└─────────────────────────────────────────┘
```

### 3.2 BTT Distribution Chart
- **X-axis**: Bed ID
- **Y-axis**: BTT (minutes)
- **Color**: Green (<20min), Yellow (20-40min), Red (>40min)
- **Drill-down**: Click bed → see detailed timeline

### 3.3 Bottleneck Severity Index (BSI)
```
BSI = (Waiting Patients × Average BTT) / Available Beds

High BSI = Many patients waiting + Slow turnaround
Low BSI = Smooth flow
```

### 3.4 Historical BTT Trends
- **By Ward**: ED vs ICU vs General (compare efficiency)
- **By Time of Day**: Peak vs off-peak BTT patterns
- **By Discharge Type**: Planned vs emergency discharges
- **By Cleaning Team**: Which team has fastest turnaround?

---

## 4. DATABASE SCHEMA EXTENSIONS

### 4.1 BedStatusHistory (Enhance Existing)
```javascript
{
  _id: ObjectId,
  bedId: ObjectId,
  wardId: ObjectId,
  
  // Timeline tracking
  previousStatus: "OCCUPIED",
  newStatus: "CLEANING",
  statusChangedAt: Date,
  statusChangedBy: ObjectId (nurse_id),
  
  // Discharge reference
  encounterIdDischarged: ObjectId,
  physicianDischargeTime: Date,  // ← CRITICAL: Physician marked discharge
  physicianName: String,
  
  // Cleaning tracking
  cleaningStartedAt: Date,
  cleaningEndedAt: Date,
  cleaningDuration: Number, // minutes
  cleanedByNurseId: ObjectId,
  cleaningNotes: String,
  
  // Triage assignment
  nextEncounterId: ObjectId,
  nextTriageAssignmentTime: Date,  // ← CRITICAL: New patient assigned
  
  // Calculated BTT
  btt_total: Number, // minutes (nextTriageTime - physicianDischargeTime)
  btt_discharge_lag: Number, // minutes (cleaningStartedAt - physicianDischargeTime)
  btt_cleaning: Number, // minutes (cleaningEndedAt - cleaningStartedAt)
  btt_assignment_lag: Number, // minutes (nextTriageTime - cleaningEndedAt)
}
```

### 4.2 New Collection: BottleneckAlert
```javascript
{
  _id: ObjectId,
  wardId: ObjectId,
  hospitalId: ObjectId,
  
  alertType: "HIGH_BTT" | "LONG_WAIT" | "CLEANING_BACKLOG",
  severity: "INFO" | "WARNING" | "CRITICAL",
  
  metrics: {
    btt_average: Number, // last 24 hours
    waiting_patients: Number,
    available_beds: Number,
    bsi_score: Number,
  },
  
  message: String, // Human-readable alert
  triggered_at: Date,
  resolved_at: Date,
  acknowledged_by: ObjectId, // ward manager
  action_taken: String,
}
```

---

## 5. API ENDPOINTS REQUIRED

### 5.1 BTT Analytics Endpoint
```
GET /api/analytics/btt/ward/:wardId
PARAMS:
  - timeRange: "24h" | "7d" | "30d"
  - includeHistorical: true/false

RESPONSE:
{
  wardId: ObjectId,
  wardName: String,
  department: String,
  
  summary: {
    btt_average: 35.2, // minutes
    btt_median: 28,
    btt_p95: 72, // 95th percentile
    
    discharge_lag_avg: 5.2,
    cleaning_avg: 18.5,
    assignment_lag_avg: 11.5,
  },
  
  current_status: {
    active_discharges: 3,
    beds_cleaning: 2,
    available_beds: 4,
    waiting_patients: 8,
    bsi_score: 16.0,
  },
  
  bed_breakdown: [
    {
      bedId: ObjectId,
      bedNumber: "ED-01",
      btt: 42.5,
      status: "AVAILABLE",
      last_discharge: "2026-04-17T10:30:00Z",
      days_since_discharge: 0.25,
    },
    // ... more beds
  ],
  
  alerts: [
    {
      type: "HIGH_BTT",
      severity: "CRITICAL",
      message: "Bed 4 vacant 45min while 10 patients waiting",
      time_idle: 45,
      waiting_count: 10,
    }
  ],
}
```

### 5.2 Discharge Trigger Endpoint
```
POST /api/encounters/:encounterId/physician-discharge
BODY:
{
  physicianId: ObjectId,
  reason: String,
  discharge_notes: String,
}

ACTION:
- Record physicianDischargeTime
- Trigger BedStatusHistory entry
- Queue bed for cleaning
- Check for BTT bottleneck
- Send alert to ward if BSI > threshold
```

### 5.3 Cleaning Status Endpoint
```
POST /api/beds/:bedId/cleaning-status
BODY:
{
  action: "START" | "COMPLETE",
  nurseId: ObjectId,
  cleaningDuration: Number, // if COMPLETE
  notes: String,
}

CALCULATION:
- On COMPLETE: Calculate BTT_discharge_lag + BTT_cleaning
- Check if assignment_lag is growing
- Trigger alert if BTT exceeds threshold
```

### 5.4 Historical BTT Report
```
GET /api/analytics/btt/report
PARAMS:
  - startDate: ISO date
  - endDate: ISO date
  - wardId: ObjectId (optional)
  - metric: "btt_avg" | "btt_worst" | "bsi_score"
  - groupBy: "ward" | "time_of_day" | "discharge_type"

RESPONSE: Aggregated BTT data with trends
```

---

## 6. FRONTEND COMPONENTS REQUIRED

### 6.1 BTT Dashboard Widget (TriageAnalytics.jsx Extension)
```
Location: src/Dashboard/Pages/TriageAnalytics.jsx

New Section:
┌─ Bottleneck Analytics Tab ─────────────────┐
│                                             │
│ [Real-time BTT] [Trends] [By Ward] [Alerts]│
│                                             │
│ ⚠️ CRITICAL ALERTS                         │
│ • ED: 10 waiting, 4 beds, BTT 52min       │
│ • ICU: 2 waiting, 1 bed, BTT 38min        │
│                                             │
│ Ward Comparison Chart                      │
│ (ED vs ICU vs General BTT averages)        │
│                                             │
│ Bed-by-Bed Heatmap (color = BTT duration) │
│                                             │
│ Timeline View (discharge → cleaning time)  │
└─────────────────────────────────────────────┘
```

### 6.2 BTT Heatmap Component
```javascript
// BedTTHeatmap.jsx
- Grid showing all beds colored by BTT
- Green: <20min
- Yellow: 20-40min
- Red: >40min
- Click bed → see detailed timeline
```

### 6.3 Bottleneck Alert Display
```javascript
// BottleneckAlert.jsx
- Real-time alert banner
- Shows: Waiting patients count
- Shows: Available beds count
- Shows: BSI score trend
- Action buttons: "Assign Now" | "Escalate"
```

### 6.4 BTT Timeline Visualization
```javascript
// BTTTimeline.jsx
For each bed discharge:
┌──────────────────────────────────────────┐
│ Bed ED-01 - Discharge Timeline           │
├──────────────────────────────────────────┤
│ Physician Discharge      10:30 AM         │
│ │                                        │
│ ├─ Discharge Lag: 5 min (👎 Slow)       │
│ │                                        │
│ ├─ Cleaning Started       10:35 AM       │
│ ├─ Cleaning Duration: 18 min (👍 Good)  │
│ │                                        │
│ ├─ Cleaning Done          10:53 AM       │
│ ├─ Assignment Lag: 12 min (BTN waiting) │
│ │                                        │
│ └─ Triage Assigned        11:05 AM       │
│                                          │
│ Total BTT: 35 min                        │
└──────────────────────────────────────────┘
```

---

## 7. ALERT RULES & THRESHOLDS

### 7.1 BTT Alert Triggers
```javascript
const BTT_THRESHOLDS = {
  NORMAL: 20,      // < 20 min = good
  WARNING: 40,     // 20-40 min = monitor
  CRITICAL: 60,    // > 60 min = emergency
};

const DISCHARGE_LAG_THRESHOLD = 10;  // min
const ASSIGNMENT_LAG_THRESHOLD = 15; // min

const BSI_CRITICAL = 20;  // (waiting_patients × avg_btt) / available_beds
```

### 7.2 Alert Conditions
```
IF waitingPatients > 5 AND availableBeds > 0 AND avgBTT > 40:
  → ALERT: "High patient wait + Slow turnaround"
  
IF bsi_score > 20:
  → ALERT: "Critical bottleneck detected"
  
IF assignmentLag > 20 AND availableBeds > 3:
  → ALERT: "Slow patient assignment despite available capacity"
  
IF cleaningDuration > 30:
  → ALERT: "Cleaning backlog - consider extra support"
```

### 7.3 Alert Routing
- **Ward Manager**: All alerts
- **Director/Admin**: CRITICAL + BSI > threshold
- **Cleaning Team Lead**: Cleaning duration alerts
- **Triage Coordinator**: Assignment lag alerts

---

## 8. IMPLEMENTATION ROADMAP

### Phase 1: Data Collection (Week 1)
- [ ] Extend BedStatusHistory model with BTT fields
- [ ] Add physicianDischargeTime tracking to encounter discharge
- [ ] Create API endpoint for cleaning status updates
- [ ] Implement BTT calculation logic

### Phase 2: API & Backend (Week 2)
- [ ] Create `/api/analytics/btt/ward/:wardId` endpoint
- [ ] Implement BottleneckAlert collection & creation logic
- [ ] Create alert generation for high BTT scenarios
- [ ] Build historical BTT report endpoint
- [ ] Add alert routing to notification system

### Phase 3: Frontend Dashboard (Week 3)
- [ ] Create BedTTHeatmap component
- [ ] Build BottleneckAlert banner component
- [ ] Create BTT Timeline visualization
- [ ] Add Bottleneck Analytics tab to TriageAnalytics.jsx
- [ ] Implement real-time BTT updates via WebSocket

### Phase 4: Testing & Refinement (Week 4)
- [ ] Unit tests for BTT calculation
- [ ] Integration tests for alert triggering
- [ ] Load test with high volume discharges
- [ ] User acceptance testing with staff
- [ ] Performance optimization

### Phase 5: Advanced Features (Later)
- [ ] Predictive BTT (machine learning)
- [ ] Staff efficiency scoring (which team has fastest cleaning?)
- [ ] Discharge pattern analysis
- [ ] Recommendation engine ("Assign Bed 4 next - closest to patient")
- [ ] Mobile alert notifications

---

## 9. SUCCESS METRICS

### 9.1 Expected Outcomes
- **BTT Reduction**: From 45+ min → Target 25 min average
- **Patient Wait Reduction**: 30-40% fewer patients in waiting area
- **Bed Utilization**: Increase from 65% → 80%+
- **Staff Satisfaction**: Reduced "communication lag" complaints
- **Patient Throughput**: +15-20% more discharges/admissions per day

### 9.2 KPIs to Track
```
1. Average BTT (target: < 25 min)
2. Median BTT (target: < 20 min)
3. BTT P95 (target: < 50 min)
4. BSI Score (target: < 10)
5. Waiting patients (target: < 3)
6. Bed idle time (target: < 15 min)
7. Discharge lag (target: < 5 min)
8. Cleaning lag (target: < 20 min)
9. Assignment lag (target: < 10 min)
```

---

## 10. TECHNICAL IMPLEMENTATION NOTES

### 10.1 Real-Time Updates
- Use WebSocket for live BTT updates
- Subscribe to specific ward or bed
- Broadcast on status changes

### 10.2 Data Aggregation
- Calculate BTT every time status changes
- Store historical data in BedStatusHistory
- Aggregate for reports (hourly, daily, weekly, monthly)

### 10.3 Performance Considerations
- Index BedStatusHistory by `wardId` + `statusChangedAt`
- Cache ward-level aggregations (update every 5 min)
- Use MongoDB aggregation pipeline for reports

### 10.4 Privacy & Logging
- Log all BTT alert triggers for auditing
- Record which alert was acknowledged by whom & when
- HIPAA-compliant: Don't store patient names in alert logs

---

## 11. DEPENDENCIES

### 11.1 Must Be Completed First
- ✅ Bed allocation system (completed)
- ✅ Triage assignment workflow (existing)
- ✅ Physician discharge workflow (needs verification)
- ✅ Bed status transitions (completed)

### 11.2 External Systems
- EHR/Encounter system (discharge source)
- Notification system (for alerts)
- WebSocket infrastructure (for real-time updates)

---

## 12. RISK MITIGATION

### Risks & Mitigation Strategies
```
Risk: False alerts (too many notifications)
Mitigation: Tune thresholds based on hospital baseline

Risk: Staff ignores alerts
Mitigation: Alert severity escalation + manager review

Risk: Data quality (incomplete discharge tracking)
Mitigation: Audit trail + manual correction workflow

Risk: Performance impact (high query load)
Mitigation: Pre-aggregate metrics cache + async calculations
```

---

## 13. FUTURE ENHANCEMENTS

- **AI Prediction**: Predict BTT based on time of day + discharge type
- **Staffing Optimizer**: Recommend cleaning staff allocation
- **Patient Routing**: Route new patients to fastest-turning beds
- **Mobile App**: Alerts to staff phones in real-time
- **Integration**: Link to hospital operating room scheduler for better flow

---

## Summary
The Bottleneck Analytics system transforms bed turnaround data into actionable insights that reduce patient wait times, improve bed utilization, and expose communication gaps in hospital workflows. By focusing on the BTT metric, administrators can identify where resource waste occurs and take targeted corrective action.

---

---

# FEATURE 2: INTEGRATED CONSUMABLES & LAB ORDERS (The "BOM")

## Overview
The Bill of Materials (BOM) system tracks all consumables, medications, and supplies used during a patient encounter. This enables resource waste detection, accurate billing for private medical aid, stock management for public hospitals, and identification of abnormal consumption patterns.

---

## 1. THE COMPONENT: Consumables Logger in Clinical Encounter

### 1.1 What Gets Logged
```
Quick-tap consumables buttons inside Clinical Care Mode:

┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  IV Starter    │    │  Catheter      │    │  Oxygen        │
│   Kit          │    │                │    │  (Hourly)      │
└────────────────┘    └────────────────┘    └────────────────┘

┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  Dressing      │    │  Medications   │    │  Bandages      │
│  Change        │    │  (Custom)      │    │                │
└────────────────┘    └────────────────┘    └────────────────┘

Plus custom free-text entry for unlisted items
```

### 1.2 Pre-defined Consumables Library
```javascript
const CONSUMABLES = {
  IV_STARTER_KIT: {
    name: "IV Starter Kit",
    category: "Vascular Access",
    cost: 450, // ZAR
    reorder_level: 50,
    unit: "kit",
  },
  
  CATHETER: {
    name: "Urinary Catheter (Foley)",
    category: "Catheters",
    cost: 120,
    reorder_level: 100,
    unit: "unit",
  },
  
  OXYGEN_HOURLY: {
    name: "Oxygen Supply (1 hour)",
    category: "Respiratory",
    cost: 85,
    reorder_level: "continuous",
    unit: "hour",
  },
  
  DRESSING: {
    name: "Wound Dressing (Sterile)",
    category: "Wound Care",
    cost: 65,
    reorder_level: 150,
    unit: "unit",
  },
  
  MEDICATIONS: {
    name: "Medications (Custom Entry)",
    category: "Pharmaceuticals",
    cost: "varies",
    unit: "dose",
  },
  
  // ... more consumables
};
```

---

## 2. THE INSIGHT: Resource Waste Detection

### 2.1 Abnormality Flagging
```
RULE: If consumables for "Green" patient exceed normal threshold:
  → Flag as abnormal
  → Log for investigation
  → Alert supervisor

EXAMPLES:
❌ "Green" patient (stable, minor injury):
   - Used: 5 IV Starter Kits (normal: 0-1)
   - Cost: ZAR 2,250 (normal: ZAR 0-450)
   → ALERT: "Excessive consumables for triage level"

✓ "Orange" patient (serious, needs intervention):
   - Used: 2 IV Starter Kits, 1 Catheter, 4 hours O2
   - Cost: ZAR 1,680 (within normal range)
   → OK: Appropriate resource use

❌ "Red" patient (critical):
   - Used: 20 dressings (normal: 3-5)
   - Cost: ZAR 1,300 (unusually high)
   → ALERT: Possible procedural inefficiency
```

### 2.2 Cost Per Patient Analysis
```
Financial Impact Dashboard:
┌──────────────────────────────────────┐
│ Daily Consumables Cost Analysis       │
├──────────────────────────────────────┤
│ Total Patients: 145                   │
│ Total Consumables Cost: ZAR 23,450    │
│ Average Cost/Patient: ZAR 161.72      │
│                                       │
│ By Triage Level:                      │
│ Green:  ZAR 45/patient (avg)         │
│ Orange: ZAR 185/patient (avg)        │
│ Red:    ZAR 520/patient (avg)        │
│                                       │
│ Anomalies Detected: 3                 │
│ • Patient #4521: ZAR 1,240 (7x norm) │
│ • Patient #4589: ZAR 890 (5.5x norm) │
│ • Patient #4612: ZAR 650 (4.2x norm) │
└──────────────────────────────────────┘
```

---

## 3. DATABASE SCHEMA

### 3.1 ConsumablesUsage Collection
```javascript
{
  _id: ObjectId,
  encounterId: ObjectId,  // Link to clinical encounter
  patientId: ObjectId,
  wardId: ObjectId,
  nurseId: ObjectId,      // Who logged the consumable
  
  consumableName: String, // "IV Starter Kit", "Oxygen (Hourly)", etc.
  consumableType: String, // "Vascular Access", "Respiratory", etc.
  quantity: Number,       // How many units used
  unit: String,          // "kit", "unit", "dose", "hour"
  unitCost: Number,      // Cost per unit in ZAR
  totalCost: Number,     // quantity × unitCost
  
  timestamp: Date,       // When consumable was logged
  notes: String,        // Optional notes from nurse
  
  // Abnormality tracking
  isAbnormal: Boolean,  // Flagged by system
  abnormalityReason: String, // "Excessive for triage level"
  abnormalityScore: Number, // 0-100 (100 = most abnormal)
}
```

### 3.2 ConsumablesAnomalyLog Collection
```javascript
{
  _id: ObjectId,
  encounterId: ObjectId,
  patientId: ObjectId,
  triageLevel: String, // GREEN, YELLOW, ORANGE, RED
  
  consumableAnalysis: {
    total_cost: Number,
    expected_cost: Number,  // Based on triage level
    cost_variance_percent: Number,
    
    abnormal_items: [
      {
        itemName: String,
        quantity: Number,
        cost: Number,
        severity: "LOW" | "MEDIUM" | "HIGH",
      }
    ],
  },
  
  possible_causes: [String], // Investigator notes
  resolution: String,        // "Approved" | "Investigated" | "Fraud Alert"
  resolved_by: ObjectId,    // Manager/supervisor
  resolved_at: Date,
}
```

---

## 4. API ENDPOINTS

### 4.1 Log Consumable Usage
```
POST /api/encounters/:encounterId/consumables
BODY:
{
  nurseId: ObjectId,
  consumableName: String,
  quantity: Number,
  notes: String,
}

RESPONSE:
{
  success: true,
  consumableId: ObjectId,
  cost: 450,
  logged_at: ISO timestamp,
  abnormality_flags: [] // If any detected
}
```

### 4.2 Get Encounter BOM
```
GET /api/encounters/:encounterId/bom
RESPONSE:
{
  encounterId: ObjectId,
  patientId: ObjectId,
  triageLevel: String,
  
  consumables: [
    {
      name: "IV Starter Kit",
      quantity: 2,
      unitCost: 450,
      totalCost: 900,
      timestamp: ISO,
    }
  ],
  
  summary: {
    total_cost: 2340,
    expected_cost: 1800,
    variance: 540,
    variance_percent: 30,
    abnormal_items: 1,
  },
}
```

### 4.3 Consumables Analytics
```
GET /api/analytics/consumables
PARAMS:
  - startDate: ISO date
  - endDate: ISO date
  - wardId: ObjectId (optional)
  - triageLevel: String (optional)
  - anomalies_only: Boolean

RESPONSE:
{
  total_patients: 145,
  total_cost: 23450,
  average_cost_per_patient: 161.72,
  
  by_triage_level: {
    GREEN: { avg: 45, total_cost: 2250 },
    ORANGE: { avg: 185, total_cost: 10800 },
    RED: { avg: 520, total_cost: 10400 },
  },
  
  high_cost_outliers: [
    {
      encounterId: ObjectId,
      patientId: ObjectId,
      cost: 1240,
      expected_cost: 185,
      variance_percent: 570,
      flagged_items: ["IV Starter Kit (qty: 4)"],
    }
  ],
  
  stock_alerts: [
    {
      consumableName: "IV Starter Kit",
      used_today: 45,
      remaining_stock: 5,
      reorder_level: 50,
      status: "CRITICAL",
    }
  ],
}
```

---

## 5. FRONTEND COMPONENTS

### 5.1 Consumables Logging Widget (in Clinical Care Mode)
```javascript
// Location: src/Dashboard/Pages/ClinicalCareMode.jsx

New Section: "Consumables & Supplies"

┌─────────────────────────────────────────┐
│ Quick Add Consumables:                  │
├─────────────────────────────────────────┤
│ [+] IV Starter Kit                      │
│ [+] Catheter (Foley)                    │
│ [+] Oxygen (Hourly)                     │
│ [+] Dressing                            │
│ [+] Medications (Custom)                │
│ [+] Custom Item                         │
│                                          │
│ Recent Additions:                       │
│ • IV Starter Kit ×2 @ ZAR 900           │
│ • Oxygen ×3hrs @ ZAR 255                │
│ • Catheter ×1 @ ZAR 120                 │
│ ───────────────────────────             │
│ Current Encounter Total: ZAR 1,275      │
└─────────────────────────────────────────┘
```

### 5.2 BOM (Bill of Materials) Report
```javascript
// Location: src/Dashboard/Pages/EncounterBOM.jsx

Display full list of all consumables + costs:

┌─────────────────────────────────────────────────┐
│ Encounter #512 - Bill of Materials              │
├─────────────────────────────────────────────────┤
│ Patient: John Doe                               │
│ Triage: ORANGE                                  │
│ Duration: 2h 35m                               │
│                                                 │
│ Item                  Qty  Unit Cost  Total    │
│ ─────────────────────────────────────────────  │
│ IV Starter Kit         2    ZAR 450   ZAR 900  │
│ Catheter               1    ZAR 120   ZAR 120  │
│ Oxygen (hours)         2.5  ZAR 85    ZAR 212  │
│ Dressing               3    ZAR 65    ZAR 195  │
│ Medications            4    ZAR 45    ZAR 180  │
│ ─────────────────────────────────────────────  │
│ TOTAL CONSUMABLES:                  ZAR 1,607 │
│                                                 │
│ Expected Cost (Orange): ZAR 1,200              │
│ Variance: + ZAR 407 (33% over)                 │
│ ⚠️ FLAGGED FOR REVIEW                          │
└─────────────────────────────────────────────────┘
```

### 5.3 Consumables Analytics Dashboard
```javascript
// Location: src/Dashboard/Pages/ConsumablesAnalytics.jsx

Daily/Weekly reports showing:
- Total consumables cost
- Cost by triage level
- High-cost anomalies (with drill-down)
- Stock levels (warnings for low inventory)
- Efficiency trends over time
```

---

## 6. USE CASES

### 6.1 Private Medical Aid Billing
```
Patient: John Smith
Insurance: Discovery Health
Encounter: ED Visit 2h 35m
CPT Code: 99283 (ED - Minor)

Bill of Materials:
- IV Starter Kit ×2: ZAR 900
- Catheter ×1: ZAR 120
- Oxygen ×2.5 hrs: ZAR 212
- Dressings: ZAR 195
- Medications: ZAR 180

Consumables Subtotal: ZAR 1,607
Clinical Fee: ZAR 2,500
Staff Allocation: ZAR 800
Total Invoice: ZAR 4,907

Exported to: Billing System → Medical Aid Claim
```

### 6.2 Stock Management (Public Hospital)
```
Daily Consumables Report:
- 45 IV Starter Kits used (stock: 5 remaining)
- 78 Catheters used (stock: 22 remaining)
- Stock Status: CRITICAL ⚠️

Action: Auto-generate purchase order for:
- IV Starter Kits: Order 200 units
- Catheters: Order 150 units
- Resend to procurement department
```

### 6.3 Fraud Detection / Efficiency Review
```
Anomaly Alert:
Patient #4521 (Green triage):
- Used: 5 IV Starter Kits (normal: 0-1)
- Cost: ZAR 2,250
- Severity: HIGH

Review Required:
[ ] Medication waste
[ ] Training issue
[ ] Equipment malfunction
[ ] Billing error

Action Taken: _____________________
Date: ________________  Investigator: ________________
```

---

## 7. ALERTS & THRESHOLDS

```javascript
const CONSUMABLES_THRESHOLDS = {
  GREEN: {
    max_cost: 150,
    alert_percent: 200, // 2x normal
  },
  YELLOW: {
    max_cost: 350,
    alert_percent: 150,
  },
  ORANGE: {
    max_cost: 800,
    alert_percent: 125,
  },
  RED: {
    max_cost: 2000,
    alert_percent: 120,
  },
};

// Stock Management
const LOW_STOCK_THRESHOLD = 20; // units
const CRITICAL_STOCK_THRESHOLD = 5; // units
const AUTO_REORDER_QUANTITY = 200; // units
```

---

## 8. IMPLEMENTATION ROADMAP

### Phase 1: Data Model & API (Week 1)
- [ ] Create ConsumablesUsage collection
- [ ] Create ConsumablesAnomalyLog collection
- [ ] Create consumables library/catalog
- [ ] Implement POST consumables logging endpoint
- [ ] Implement GET BOM endpoint

### Phase 2: Backend Logic (Week 2)
- [ ] Create consumables cost calculation engine
- [ ] Implement abnormality detection algorithm
- [ ] Create analytics aggregation functions
- [ ] Build stock management logic
- [ ] Implement auto-reorder triggers

### Phase 3: Frontend Components (Week 3)
- [ ] Build consumables logging widget in Clinical Care
- [ ] Create BOM report display
- [ ] Build consumables analytics dashboard
- [ ] Add real-time cost tracking display

### Phase 4: Integration & Testing (Week 4)
- [ ] Integrate with billing system
- [ ] Test abnormality detection
- [ ] Load test with high volume usage
- [ ] UAT with clinicians
- [ ] Deploy monitoring & alerts

---

---

# FEATURE 3: PHARMACY/LAB HANDOVER (Inter-Departmental Sync)

## Overview
The Pharmacy/Lab Handover system enables seamless communication between clinical departments by creating digital order vouchers that work even offline. Staff can request lab tests, imaging, or pharmacy services with clinical context, and technicians can access the request via QR code or manual lookup.

---

## 1. THE COMPONENT: Order Request Module

### 1.1 Supported Order Types
```
From Clinical Care Mode, nurses/physicians can request:

Laboratory Orders:
  • Blood Tests (Basic Bloods, Full Blood Count, Chemistry Panel, etc.)
  • Urinalysis
  • Culture & Sensitivity
  • Microbiology tests

Imaging Orders:
  • X-Ray (Chest, Abdomen, Limbs, etc.)
  • Ultrasound
  • CT Scan (if available)

Pharmacy Orders:
  • Medication fulfillment
  • Intravenous preparation

Specialty Consults:
  • Cardiology
  • Surgery
  • Neurology
```

### 1.2 Order Creation Workflow
```
Step 1: Nurse/Physician in Clinical Care → "Request Order"
Step 2: Select order type (Lab, Imaging, Pharmacy)
Step 3: Enter clinical indication (Chief Complaint summary)
Step 4: AI Suggestion attached automatically
Step 5: System generates Digital Voucher
Step 6: Voucher displays QR code + Manual reference code
```

---

## 2. THE "OFFLINE REALITY": Digital Voucher

### 2.1 What is a Digital Voucher?
```
A Digital Voucher is a temporary order record that:
✓ Links patient clinical data to the order
✓ Contains AI-generated diagnostic suggestions
✓ Survives network disconnection
✓ Can be printed or shown on screen
✓ Contains QR code for rapid technician lookup
✓ Works across departments (pharmacy, lab, imaging)
```

### 2.2 Digital Voucher Contents
```
┌─────────────────────────────────────────┐
│ DIGITAL ORDER VOUCHER                   │
├─────────────────────────────────────────┤
│                                         │
│ Reference: ORD-2026-04-17-00512        │
│ QR Code: [██████████████████████]       │
│           [██████████████████████]       │
│           [██████████████████████]       │
│                                         │
│ Patient: John Doe (ID: 4521)            │
│ Age: 45 | Sex: M                        │
│ Chief Complaint: Chest pain, SOB        │
│                                         │
│ Order Type: Laboratory                  │
│ Requested Tests:                        │
│  • Troponin I (Cardiac marker)          │
│  • Full Blood Count                     │
│  • Electrolyte Panel                    │
│                                         │
│ Clinical Indication:                    │
│ Possible acute coronary syndrome        │
│                                         │
│ AI Suggestion:                          │
│ "Given chest pain + SOB, consider also: │
│  - ECG (URGENT)                         │
│  - D-Dimer (rule out PE)                │
│  - Myoglobin (secondary marker)"        │
│                                         │
│ Requested by: Dr. Sarah M.              │
│ Ward: ED | Bed: 04                      │
│ Date/Time: 17 Apr 2026, 14:32:00       │
│ Status: PENDING ⏳                      │
│                                         │
│ Lab: Scan QR → View full details online │
│ Or manually enter ORD-2026-04-17-00512  │
└─────────────────────────────────────────┘
```

---

## 3. DATABASE SCHEMA

### 3.1 OrderRequest Collection
```javascript
{
  _id: ObjectId,
  
  // Order metadata
  orderId: String, // "ORD-2026-04-17-00512"
  orderType: "LABORATORY" | "IMAGING" | "PHARMACY" | "CONSULT",
  departmentRequested: String, // "Lab", "Radiology", "Pharmacy"
  
  // Patient & clinical context
  encounterId: ObjectId,
  patientId: ObjectId,
  patientName: String,
  patientAge: Number,
  patientSex: String,
  
  // Order details
  requestedBy: ObjectId, // Nurse/Physician ID
  requestedByName: String,
  requestDateTime: Date,
  
  orderedTests: [
    {
      testName: String, // "Troponin I", "Full Blood Count"
      testCode: String, // Lab test code
      priority: "ROUTINE" | "URGENT",
    }
  ],
  
  // Clinical context
  chiefComplaint: String,
  aiSuggestion: String,  // Generated by ML model
  additionalNotes: String,
  
  // Digital Voucher
  voucherQRCode: String, // QR code data (encoded)
  voucherManualCode: String, // For manual entry
  voucherGeneratedAt: Date,
  voucherExpiresAt: Date, // After 24 hours or encounter end
  
  // Status tracking
  status: "PENDING" | "RECEIVED" | "IN_PROGRESS" | "COMPLETED" | "CANCELLED",
  statusHistory: [
    {
      status: String,
      changedAt: Date,
      changedBy: ObjectId,
      notes: String,
    }
  ],
  
  // Lab result (if applicable)
  resultReceived: Boolean,
  resultDateTime: Date,
  resultSummary: String,
  resultAttachment: ObjectId, // Link to document
}
```

### 3.2 OfflineOrderQueue (For network issues)
```javascript
{
  _id: ObjectId,
  deviceId: String, // Laptop/tablet ID
  
  unsyncedOrders: [
    {
      tempOrderId: String,
      orderData: Object,
      createdAt: Date,
      synced: Boolean,
      syncedAt: Date,
    }
  ],
}
```

---

## 4. API ENDPOINTS

### 4.1 Create Order Request
```
POST /api/encounters/:encounterId/order-request
BODY:
{
  orderType: "LABORATORY",
  departmentRequested: "Lab",
  orderedTests: [
    { testName: "Troponin I", testCode: "TROPI" },
    { testName: "Full Blood Count", testCode: "FBC" },
  ],
  chiefComplaint: "Chest pain, shortness of breath",
  additionalNotes: Optional notes,
}

RESPONSE:
{
  success: true,
  orderId: "ORD-2026-04-17-00512",
  voucherQRCode: String,
  voucherManualCode: String,
  voucherExpiresAt: ISO timestamp,
  aiSuggestion: "Given chest pain + SOB, consider also...",
}
```

### 4.2 Get Order by QR/Manual Code
```
GET /api/order-request/lookup
PARAMS:
  - qrCode: String (optional)
  - manualCode: String (optional)

RESPONSE:
{
  orderId: String,
  patientName: String,
  chiefComplaint: String,
  orderedTests: Array,
  aiSuggestion: String,
  status: String,
  createdAt: ISO,
}
```

### 4.3 Update Order Status
```
POST /api/order-request/:orderId/status
BODY:
{
  status: "RECEIVED" | "IN_PROGRESS" | "COMPLETED",
  notes: String,
}

RESPONSE: Updated order object
```

### 4.4 Get Results
```
GET /api/order-request/:orderId/results
RESPONSE:
{
  orderId: String,
  status: String,
  resultReceived: Boolean,
  resultSummary: String,
  resultDateTime: ISO,
  resultLink: URL, // To PDF or document
}
```

---

## 5. FRONTEND COMPONENTS

### 5.1 Order Request Widget (in Clinical Care Mode)
```javascript
// Location: src/Dashboard/Pages/ClinicalCareMode.jsx

New Section: "Request Services"

┌──────────────────────────────────────────┐
│ Request Lab/Imaging/Pharmacy             │
├──────────────────────────────────────────┤
│                                          │
│ [+] Laboratory Tests                    │
│ [+] Imaging (X-Ray, Ultrasound)         │
│ [+] Pharmacy Fulfillment                │
│ [+] Specialty Consult                   │
│                                          │
│ Pending Requests:                       │
│ ✓ Lab (Troponin, FBC) - RECEIVED        │
│ ⏳ X-Ray (Chest) - IN_PROGRESS          │
│ 🕐 Pharmacy (Amoxicillin) - PENDING     │
└──────────────────────────────────────────┘
```

### 5.2 Digital Voucher Display & Print
```javascript
// Location: src/Dashboard/Pages/DigitalVoucher.jsx

Display QR code with:
- Print capability
- Option to send to lab/pharmacy via SMS/email
- Manual reference code display
- Real-time status updates
- AI suggestion modal
```

### 5.3 Lab/Pharmacy Scanner Interface
```javascript
// Location: src/Dashboard/Pages/VoucherScanner.jsx

Allow lab/pharmacy staff to:
- Scan QR code with device camera
- Manually enter reference code
- See order details instantly
- Mark order as "Received" in system
- Add results when test complete
```

### 5.4 Order Tracking Dashboard
```javascript
// Location: src/Dashboard/Pages/OrderTracker.jsx

Real-time view of:
- All outstanding orders for current ward
- Status of each order (Pending/In Progress/Completed)
- Time since order created (alert if > 1 hour)
- Quick-link to order details & AI suggestions
- Filter by department (Lab, Imaging, Pharmacy)
```

---

## 6. AI INTEGRATION: Smart Suggestions

### 6.1 How AI Suggestions Work
```
Input:
- Patient chief complaint
- Triage level
- Vital signs
- Past medical history (if available)

Output:
- Recommended tests beyond initial request
- Priority levels
- Supporting evidence (guidelines)

Example:
User Request: "Troponin I, Full Blood Count"
AI Suggestion:
"Given chest pain + SOB + elevated HR,
 also consider:
 - D-Dimer (to rule out pulmonary embolism)
 - ECG (URGENT - not a lab test but critical)
 - Myoglobin (secondary cardiac marker)
 - BNP (heart failure evaluation)"
```

### 6.2 Model Integration
```
Call ML model endpoint:
POST /api/ml/suggest-tests
BODY:
{
  chiefComplaint: String,
  triageLevel: String,
  vitals: { HR, BP, RR, O2, Temp },
  pastHistory: String,
}

RESPONSE:
{
  suggestions: [
    {
      testName: String,
      rationale: String,
      priority: "ROUTINE" | "URGENT",
    }
  ],
  reasoning: String, // Explain why
}
```

---

## 7. OFFLINE CAPABILITY

### 7.1 Offline Order Creation (Mobile)
```
If network lost while creating order:
1. Order saved to device local storage
2. Auto-sync when connection restored
3. User sees "Synced" confirmation
4. No duplicate orders created
```

### 7.2 QR Code Works Offline
```
Lab technician with tablet:
1. Pre-loads pending orders (via WiFi)
2. Network drops
3. Still can scan QR codes stored locally
4. Updates sync when network returns
```

---

## 8. USE CASES

### 8.1 ER Doctor Ordering Labs (Connected Environment)
```
1. Patient: 45M with chest pain
2. Doctor in Clinical Care switches to "Order Request"
3. Selects "Laboratory"
4. Checks: Troponin, FBC, Chemistry panel
5. System attaches AI suggestion (D-Dimer, ECG)
6. Digital voucher generated instantly
7. Lab staff scan QR code → begin processing
8. Results delivered back within 30 mins
9. Results appear in Clinical Care dashboard automatically
```

### 8.2 Rural Hospital (Limited Internet)
```
1. Nurse creates order for X-Ray on laptop
2. Network drops while generating voucher
3. Order saved locally
4. Network restored after 2 hours
5. Order auto-syncs to central system
6. QR code generated and printed
7. Patient handed physical voucher
8. Takes to radiology department (offline)
9. Tech scans code from printed paper
10. Later: X-Ray results sync back online
```

### 8.3 Pharmacy Fulfillment
```
1. Physician orders Amoxicillin via Order Request
2. Digital voucher created with medication details
3. Pharmacy staff receives alert
4. Scans QR code → sees order in pharmacy system
5. Verifies patient allergies (cross-check with EHR)
6. Prepares medication
7. Marks as "Completed" when dispensed
8. Nurse receives alert that med ready for pickup
```

---

## 9. IMPLEMENTATION ROADMAP

### Phase 1: Data & API (Week 1)
- [ ] Create OrderRequest collection
- [ ] Create OfflineOrderQueue collection
- [ ] Build POST order-request endpoint
- [ ] Build GET order lookup endpoint
- [ ] Implement QR code generation

### Phase 2: Backend & ML Integration (Week 2)
- [ ] Build order status tracking system
- [ ] Integrate AI suggestion model
- [ ] Implement offline sync logic
- [ ] Create result tracking system

### Phase 3: Frontend Components (Week 3)
- [ ] Build order request widget in Clinical Care
- [ ] Create digital voucher display
- [ ] Build QR scanner interface (for lab/pharmacy)
- [ ] Create order tracker dashboard

### Phase 4: Testing & Deployment (Week 4)
- [ ] Test offline scenario
- [ ] Test QR code scanning
- [ ] UAT with lab/pharmacy staff
- [ ] Deploy and monitor

---

---

# FEATURE 4: AUTOMATED "OPERATIONAL HEALTH" SCORE (Hospital Pulse)

## Overview
The Hospital Pulse is a real-time operational health score (1-100) that gives hospital executives a single, actionable metric to assess facility safety and efficiency. It combines staffing, patient acuity, bed availability, and triage speed into a holistic score.

---

## 1. THE METRIC: Hospital Pulse Score (1–100)

### 1.1 Definition
```
Hospital Pulse Score = Aggregated measure of facility operational health

Scale:
90-100: OPTIMAL       ✅ Fully staffed, low acuity, fast triage
70-89:  HEALTHY       ✓ Normal operations, minor strains
50-69:  CAUTION       ⚠️ Resource strain, monitoring needed
30-49:  WARNING       🔴 Critical resource shortage
1-29:   CRITICAL      🚨 Immediate intervention required
```

### 1.2 Formula
```
Hospital Pulse = (Staffing Factor × 0.25) 
                + (Bed Availability × 0.25) 
                + (Patient Acuity Balance × 0.25) 
                + (Triage Efficiency × 0.25)

Each component: 0-100 normalized score
```

---

## 2. FORMULA COMPONENTS

### 2.1 Staffing Factor (0-100)
```
Calculation:
  Required Staff = (Current Patient Count × 0.5) + Fixed Overhead
  Available Staff = Currently working nurses + physicians
  
  Staffing Factor = (Available Staff / Required Staff) × 100
  
  Adjustments:
  - If > 100: Cap at 100 (can't have more than optimal)
  - If < 30: Critical shortage flag
  
Example:
  Patients: 50
  Required: (50 × 0.5) + 10 = 35 staff
  Available: 32 staff
  
  Staffing Factor = (32 / 35) × 100 = 91.4
```

### 2.2 Bed Availability (0-100)
```
Calculation:
  Available Beds = Beds in "AVAILABLE" status
  Total Beds = All operational beds
  Expected Occupancy = 75% (optimal for a hospital)
  
  If Occupancy ≤ 75%:
    Bed Availability = 100
  If Occupancy > 75%:
    Bed Availability = 100 - ((Occupancy - 75) × 2)
    
Example 1 (Good):
  Total: 100 beds
  Occupied: 70 beds (70% occupancy)
  Bed Availability = 100 ✅
  
Example 2 (Strain):
  Total: 100 beds
  Occupied: 90 beds (90% occupancy)
  Bed Availability = 100 - ((90 - 75) × 2) = 70 ⚠️
  
Example 3 (Critical):
  Total: 100 beds
  Occupied: 98 beds (98% occupancy)
  Bed Availability = 100 - ((98 - 75) × 2) = 54 🔴
```

### 2.3 Patient Acuity Balance (0-100)
```
Calculation:
  Acuity Composition = Count of patients by level:
  - GREEN (stable): 40%
  - YELLOW (moderate): 35%
  - ORANGE (serious): 20%
  - RED (critical): 5%
  
  Actual Composition = Observed percentages
  
  Deviation = Sum of absolute differences × 10
  
  Acuity Balance = 100 - Deviation
  
Example (Healthy):
  Expected: G=40%, Y=35%, O=20%, R=5%
  Actual:   G=42%, Y=33%, O=19%, R=6%
  Deviations: 2+2+1+1 = 6
  Acuity Balance = 100 - (6×10) = 40 ⚠️ (Too many deviations)
  
Example (Ideal):
  Expected: G=40%, Y=35%, O=20%, R=5%
  Actual:   G=40%, Y=35%, O=20%, R=5%
  Deviations: 0
  Acuity Balance = 100 ✅
  
Example (Crisis):
  Expected: G=40%, Y=35%, O=20%, R=5%
  Actual:   G=20%, Y=25%, O=30%, R=25%
  Deviations: 20+10+10+20 = 60
  Acuity Balance = 100 - (60×10) = -500 (capped at 1) 🚨
```

### 2.4 Triage Efficiency (0-100)
```
Calculation:
  Target Triage Time = 15 minutes
  
  For each waiting patient, measure:
  - Time in waiting room
  - Adjust for triage priority
  
  Average Wait Time = Mean of all waiting patients
  
  If Average Wait ≤ 15 min:
    Triage Efficiency = 100
  Else:
    Triage Efficiency = 100 - ((Average Wait - 15) × 2)
    
Example 1 (Good):
  Waiting patients: [10 min, 8 min, 12 min]
  Average Wait = 10 min
  Triage Efficiency = 100 ✅
  
Example 2 (Backlog):
  Waiting patients: [34 min, 45 min, 28 min, 52 min]
  Average Wait = 39.75 min
  Triage Efficiency = 100 - ((39.75 - 15) × 2) = 50.5 ⚠️
  
Example 3 (Emergency):
  Waiting patients: [2 hours, 3 hours, 1.5 hours avg]
  Average Wait = 120 min
  Triage Efficiency = 100 - ((120 - 15) × 2) = -130 (capped at 1) 🚨
```

---

## 3. DATABASE SCHEMA

### 3.1 HospitalPulse Collection (Real-time)
```javascript
{
  _id: ObjectId,
  hospitalId: ObjectId,
  
  // Current composite score
  pulseScore: Number, // 1-100
  timestamp: Date,
  
  // Component breakdown
  components: {
    staffing_factor: Number,
    bed_availability: Number,
    acuity_balance: Number,
    triage_efficiency: Number,
  },
  
  // Status
  status: "OPTIMAL" | "HEALTHY" | "CAUTION" | "WARNING" | "CRITICAL",
  
  // Details
  staffing: {
    available: Number,
    required: Number,
    deficit: Number,
  },
  
  beds: {
    available: Number,
    total: Number,
    occupancy_rate: Number,
  },
  
  acuity: {
    green_count: Number,
    yellow_count: Number,
    orange_count: Number,
    red_count: Number,
  },
  
  triage: {
    patients_waiting: Number,
    avg_wait_time: Number, // minutes
  },
  
  // Alerts
  alerts: [
    {
      component: "STAFFING" | "BEDS" | "ACUITY" | "TRIAGE",
      severity: "INFO" | "WARNING" | "CRITICAL",
      message: String,
    }
  ],
}
```

### 3.2 PulseHistory Collection (For trends)
```javascript
{
  _id: ObjectId,
  hospitalId: ObjectId,
  
  timestamp: Date,
  pulseScore: Number,
  staffing_factor: Number,
  bed_availability: Number,
  acuity_balance: Number,
  triage_efficiency: Number,
}
```

---

## 4. API ENDPOINTS

### 4.1 Get Current Hospital Pulse
```
GET /api/hospital/:hospitalId/pulse
RESPONSE:
{
  hospitalId: ObjectId,
  hospitalName: String,
  pulseScore: 76,
  status: "HEALTHY",
  timestamp: ISO,
  
  components: {
    staffing_factor: 91,
    bed_availability: 85,
    acuity_balance: 68,
    triage_efficiency: 62,
  },
  
  details: {
    staffing: {
      available: 32,
      required: 35,
      deficit: 3,
    },
    beds: {
      available: 15,
      occupied: 85,
      total: 100,
      occupancy_rate: 85,
    },
    acuity: {
      green: 45,
      yellow: 28,
      orange: 20,
      red: 7,
    },
    triage: {
      patients_waiting: 8,
      avg_wait_time: 18,
    },
  },
  
  alerts: [
    {
      component: "TRIAGE",
      severity: "WARNING",
      message: "Average triage time 3 min above target",
    }
  ],
}
```

### 4.2 Get Pulse History (Trends)
```
GET /api/hospital/:hospitalId/pulse/history
PARAMS:
  - timeRange: "24h" | "7d" | "30d"
  
RESPONSE:
{
  hospitalId: ObjectId,
  timeRange: "24h",
  dataPoints: [
    {
      timestamp: ISO,
      pulseScore: 76,
      staffing_factor: 91,
      bed_availability: 85,
      acuity_balance: 68,
      triage_efficiency: 62,
    },
    // ... more data points (hourly)
  ],
  
  trends: {
    avg_pulse: 74,
    min_pulse: 52,
    max_pulse: 88,
    trend: "STABLE" | "IMPROVING" | "DECLINING",
  },
}
```

### 4.3 Alert on Critical Pulse
```
Trigger: When pulseScore < 40

POST /api/hospital/:hospitalId/pulse/alert
BODY:
{
  pulseScore: 38,
  message: "Hospital operating at critical capacity",
}

ACTION:
- Email CEO/Director immediately
- Send SMS to admin staff
- Display RED banner on all dashboards
- Log incident
```

---

## 5. FRONTEND COMPONENTS

### 5.1 Hospital Pulse Widget (Management Dashboard)
```javascript
// Location: src/Dashboard/Pages/ManagementDashboard.jsx

// Large, prominent display
┌────────────────────────────────────┐
│         🏥 HOSPITAL PULSE           │
├────────────────────────────────────┤
│                                    │
│            SCORE: 76               │
│            STATUS: HEALTHY ✓       │
│                                    │
│  ┌──────────────────────────────┐  │
│  │████████████████░░░░░░░░░░░░░│  │ (76%)
│  └──────────────────────────────┘  │
│                                    │
│ Last Updated: Just now             │
│ Trend: Stable (Past 24h avg: 74)  │
│                                    │
│ [View Details]  [Alert History]   │
└────────────────────────────────────┘
```

### 5.2 Pulse Breakdown Widget
```javascript
// Location: src/Dashboard/Pages/ManagementDashboard.jsx (Details)

┌─────────────────────────────────────┐
│ Hospital Pulse Breakdown            │
├─────────────────────────────────────┤
│                                     │
│ Staffing Factor:      91  ✅        │
│ ████████████████████░ (91%)         │
│ 32 available / 35 required          │
│                                     │
│ Bed Availability:     85  ✓         │
│ ████████████████░░ (85%)            │
│ 15 available / 100 total (85% occ)  │
│                                     │
│ Acuity Balance:       68  ⚠️        │
│ ███████████████░░░░░ (68%)          │
│ Excess: 7 Orange, 2 Red             │
│                                     │
│ Triage Efficiency:    62  ⚠️        │
│ ████████████░░░░░░░░ (62%)          │
│ Avg wait: 18 min (3 min over target)│
│                                     │
└─────────────────────────────────────┘
```

### 5.3 Pulse Trends Chart
```javascript
// Location: src/Dashboard/Pages/PulseTrends.jsx

Display Hospital Pulse over time:
- Line graph showing score evolution (hourly data)
- Color zones: Green (90+), Yellow (70-89), Orange (50-69), Red (<50)
- Hover for details
- Identify patterns (peak times, recurring issues)
```

### 5.4 CEO Mobile View
```javascript
// Location: Mobile app or simplified view

┌────────────────────────┐
│  St. Mary's Hospital   │
│  ─────────────────────  │
│                        │
│   PULSE: 76            │
│                        │
│  Status: HEALTHY ✓     │
│                        │
│  [Tap for Details]     │
│                        │
│  Last 24h: ▓▓░░░░░▓▓▓▓ │
│            LOW → HIGH  │
│                        │
│  ⚠️ 1 Warning Alert    │
└────────────────────────┘
```

---

## 6. USE CASES

### 6.1 CEO Decision-Making
```
Scenario 1: CEO on their phone at 14:30
- Opens hospital app
- Sees Hospital Pulse: 76 (HEALTHY)
- Glances at component breakdown
- Decision: Continue normal operations

Scenario 2: Crisis Alert
- Pulse drops to 35 (CRITICAL)
- CEO receives HTTP notification
- Red banner displays: "Critical Staffing Shortage"
- CEO can drill down to see:
  - 8 staff scheduled but only 4 available (call-outs)
  - 95% bed occupancy
  - Average triage wait: 67 minutes
- Decision: Activate contingency staffing, call in reserves

Scenario 3: Trend Analysis
- CEO reviews past 30 days
- Pulse regularly drops to 50-60 on weekend shifts
- Pattern identified: Weekend understaffing
- Decision: Implement weekend bonus pay to attract staff
```

### 6.2 Department Manager Coordination
```
Ward Manager sees Pulse at 62 (CAUTION):
- Checks breakdown: Triage efficiency is 45 (very low)
- Investigates: 15 patients waiting, avg 35 min wait
- Action: Activates triage room 2, brings in extra nurse
- Re-checks in 10 min: Pulse now 71 (HEALTHY)
```

### 6.3 Operations Director during Surge
```
Evening surge event:
- Pulse: 45 (WARNING) at 18:00
- Components: Acuity 38 (too many critical patients), Staffing 72
- Director contacts:
  - ICU manager: Can you take 3 Orange patients?
  - More staff available for call-in?
  - Any beds available in general ward?
- Actions taken
- Pulse recovers to 58 (CAUTION) within 30 min
- Continues monitoring every 15 min during surge
```

---

## 7. ALERT RULES & ESCALATION

```javascript
const PULSE_THRESHOLDS = {
  OPTIMAL: { min: 90, color: "GREEN", alert: false },
  HEALTHY: { min: 70, color: "BLUE", alert: false },
  CAUTION: { min: 50, color: "YELLOW", alert: false },
  WARNING: { min: 30, color: "ORANGE", alert: true },
  CRITICAL: { min: 0, color: "RED", alert: true },
};

// Alert escalation
IF pulseScore < 40:
  → Send SMS to Hospital Director
  → Email to CEO
  → Escalate to Operations Manager
  → Post RED alert on all dashboards
  → Log as incident

IF pulseScore drops > 20 points in 1 hour:
  → Investigate root cause
  → Alert department head
  → Check for staff call-outs
```

---

## 8. CALCULATION INTERVAL

```
Real-time Calculation:
- Every minute (or on major event: admission/discharge/clock-in/clock-out)
- Update HospitalPulse collection
- Broadcast update to all connected dashboards
- Archive hourly snapshots to PulseHistory
- Email alerts if threshold breached
```

---

## 9. IMPLEMENTATION ROADMAP

### Phase 1: Data Model & Calculation Engine (Week 1)
- [ ] Create HospitalPulse collection
- [ ] Create PulseHistory collection
- [ ] Implement Staffing Factor calculation
- [ ] Implement Bed Availability calculation
- [ ] Implement Acuity Balance calculation
- [ ] Implement Triage Efficiency calculation

### Phase 2: API & Business Logic (Week 2)
- [ ] Build GET /api/hospital/:hospitalId/pulse endpoint
- [ ] Build GET /api/hospital/:hospitalId/pulse/history endpoint
- [ ] Implement real-time calculation engine
- [ ] Create alert triggering logic
- [ ] Implement WebSocket broadcast for real-time updates

### Phase 3: Frontend Components (Week 3)
- [ ] Create Hospital Pulse widget
- [ ] Create Pulse Breakdown widget
- [ ] Create Pulse Trends chart
- [ ] Create CEO mobile view
- [ ] Integrate alerts into dashboard

### Phase 4: Integration & Testing (Week 4)
- [ ] Test calculation accuracy
- [ ] Test with realistic data volumes
- [ ] UAT with management team
- [ ] Load testing for real-time updates
- [ ] Deployment and monitoring

---

## 10. SUCCESS METRICS

```
Implementation Success:
✓ Hospital Pulse calculated in real-time (< 1 min delay)
✓ Accuracy: Formula validated against manual audits
✓ Alert delivery: 100% delivery rate, <5 min latency
✓ Mobile visibility: CEO can check anytime
✓ Actionability: Management can drill down to root causes

Operational Impact:
- Decision response time: Reduced from hours → minutes
- Resource efficiency: Better staffing allocation
- Patient safety: Earlier identification of overload
- Staff satisfaction: Better visibility reduces chaos
```

---

## Summary
The four integrated features—Bottleneck Analytics, Consumables BOM, Pharmacy/Lab Handover, and Hospital Pulse—transform the nurse helpdesk system into a complete **Hospital Operating System (HOS)**. They enable:

1. **Visibility**: Real-time metrics for operations and executives
2. **Efficiency**: Reduced wait times, better resource allocation
3. **Safety**: Early warning of critical conditions
4. **Billing**: Accurate cost tracking for private and public sectors
5. **Integration**: Seamless inter-departmental communication
6. **Offline Capability**: Works even without internet connectivity
7. **Scalability**: Grows with hospital needs

These features move from "Scheduling & Triage" to "Hospital Management Platform."
