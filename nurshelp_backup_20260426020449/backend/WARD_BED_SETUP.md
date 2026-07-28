# Ward & Bed Management System - Setup Guide

## Overview

This document provides setup instructions for the ward and bed management system with live status tracking.

---

## Database Models

### 1. Ward Model
Organizes beds into logical ward groups for hospital operation.

```javascript
{
  wardId: "ICU-Ward-A",                    // Unique identifier
  wardName: "Intensive Care Unit - Ward A",
  department: "ICU",                       // ED, ICU, OR, MATERNITY, NICU, PEDIATRICS, PSYCH, GENERAL, CARDIO, NEURO, ONCOLOGY
  location: {
    floor: 2,
    wing: "East",
    area: "North Wing"
  },
  totalBeds: 6,
  occupiedBeds: 2,
  availableBeds: 3,
  cleaningBeds: 1,
  maintenanceBeds: 0,
  specialization: ["Cardiac", "Respiratory"],
  averageOccupancyRate: 45,                // Auto-calculated
  isActive: true,
  managingNurseId: "ObjectId",             // Optional: nurse in charge
  notes: "Recently upgraded monitoring equipment"
}
```

### 2. BedSchedule Model (Updated)
Individual bed tracking with ward reference.

```javascript
{
  wardId: "ObjectId",                      // Reference to Ward
  bedId: "ICU-Ward-A-B01",                 // Unique bed ID
  roomNumber: "ICU-2A-01",
  department: "ICU",
  bedType: "INTENSIVE",                    // STANDARD, MONITORED, ISOLATION, INTENSIVE, HIGH_DEPENDENCY
  status: "OCCUPIED",                      // AVAILABLE, OCCUPIED, CLEANING, MAINTENANCE
  statusChangedAt: "2026-04-13T10:15:00Z", // When status last changed
  currentEncounterId: "ObjectId",
  assignedNurseId: "nurse-id",
  occupancyStartTime: "2026-04-13T08:30:00Z",
  expectedDischargeTime: "2026-04-14T14:00:00Z",
  acuityLevel: "HIGH",
  hasMonitor: true,
  hasVentilator: true,
  hasDialysis: false,
  canIsolate: false,
  lastCleanedAt: "2026-04-13T08:00:00Z",
  cleaningDurationMinutes: 30,
  estimatedReadyTime: "2026-04-13T10:45:00Z",
  cleanedByStaffId: "staff-id",
  totalOccupancyCount: 24
}
```

### 3. BedStatusHistory Model
Complete audit trail of all bed status changes.

```javascript
{
  bedId: "ObjectId",
  bedIdString: "ICU-Ward-A-B01",
  wardId: "ObjectId",
  previousStatus: "AVAILABLE",
  currentStatus: "OCCUPIED",
  changedBy: "ObjectId",                   // Nurse/staff who made change
  changedByName: "Sarah Johnson",
  timestamp: "2026-04-13T10:15:00Z",
  reason: "PATIENT_ADMITTED",              // PATIENT_ADMITTED, PATIENT_DISCHARGED, CLEANING_STARTED, CLEANING_COMPLETED, MAINTENANCE_STARTED, MAINTENANCE_COMPLETED, UPGRADE, DOWNGRADE
  encounterId: "ObjectId",                 // Optional: which patient
  notes: "Patient admitted from ED with Cardiac event",
  durationMinutes: 125                     // How long in previous status
}
```

---

## API Endpoints

### Ward Operations

**GET /api/beds/wards** - Get all wards with live status
```bash
curl http://localhost:5001/api/beds/wards?department=ICU
```

Response:
```json
{
  "success": true,
  "data": [
    {
      "_id": "ObjectId",
      "wardId": "ICU-Ward-A",
      "wardName": "Intensive Care Unit - Ward A",
      "department": "ICU",
      "totalBeds": 6,
      "currentStatus": {
        "available": 2,
        "occupied": 3,
        "cleaning": 1,
        "maintenance": 0
      },
      "occupancyRate": 50,
      "bedsLayout": [
        {
          "_id": "ObjectId",
          "bedId": "ICU-Ward-A-B01",
          "status": "OCCUPIED",
          "bedType": "INTENSIVE",
          "statusChangedAt": "2026-04-13T10:15:00Z"
        }
      ]
    }
  ],
  "timestamp": "2026-04-13T10:30:00Z"
}
```

**GET /api/beds/ward/:wardId** - Get detailed ward view
```bash
curl http://localhost:5001/api/beds/ward/ObjectId
```

**GET /api/beds/live-status** - Real-time dashboard view
```bash
curl http://localhost:5001/api/beds/live-status?department=ICU
```

### Bed Management (Protected)

**PATCH /api/beds/:bedId/status** - Update bed status
```bash
curl -X PATCH http://localhost:5001/api/beds/ObjectId/status \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "CLEANING",
    "reason": "PATIENT_DISCHARGED",
    "notes": "Deep clean requested"
  }'
```

Status Values: `AVAILABLE`, `OCCUPIED`, `CLEANING`, `MAINTENANCE`

**POST /api/beds/:bedId/assign** - Assign patient to bed
```bash
curl -X POST http://localhost:5001/api/beds/ObjectId/assign \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "encounterId": "ObjectId",
    "nurseName": "Sarah Johnson"
  }'
```

**POST /api/beds/:bedId/discharge** - Discharge patient (moves to CLEANING)
```bash
curl -X POST http://localhost:5001/api/beds/ObjectId/discharge \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**GET /api/beds/:bedId/history** - Get bed status history
```bash
curl http://localhost:5001/api/beds/ObjectId/history?limit=20&days=7 \
  -H "Authorization: Bearer TOKEN"
```

### Analytics (Protected)

**GET /api/beds/analytics/occupancy** - Occupancy analytics
```bash
curl http://localhost:5001/api/beds/analytics/occupancy?wardId=ObjectId&days=7 \
  -H "Authorization: Bearer TOKEN"
```

Response:
```json
{
  "success": true,
  "data": {
    "period": "Last 7 days",
    "totalAdmissions": 45,
    "averageStayHours": "18.5",
    "bedMetrics": [
      {
        "bedId": "ICU-Ward-A-B01",
        "admissions": 8,
        "averageStayHours": "22.3",
        "utilizationScore": "11.4"
      }
    ]
  }
}
```

---

## Frontend Integration

### WardBoard Component

```jsx
import WardBoard from './Dashboard/WardBoard';

function App() {
  return (
    <WardBoard token={authToken} />
  );
}
```

**Features:**
- 🟢 Ward selector with occupancy rates
- 🔴 Live bed grid with color coding
- 🟡 Real-time refresh (10s wards, 5s selected ward)
- 🔵 Status filtering (All, Available, Occupied, Cleaning, Maintenance)
- Quick status update buttons
- Responsive grid layout

**Color Codes:**
- 🟢 Green = AVAILABLE
- 🔴 Red = OCCUPIED
- 🟡 Yellow = CLEANING
- 🔵 Blue = MAINTENANCE

---

## Seeding Sample Data

Create `backend/src/seeds/seedWards.js`:

```javascript
import mongoose from "mongoose";
import { Ward } from "../models/ward.model.js";
import { BedSchedule } from "../models/bedSchedule.model.js";
import dotenv from "dotenv";

dotenv.config();

const seedWards = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    
    // Clear existing data
    await Ward.deleteMany({});
    await BedSchedule.deleteMany({});

    // Create wards
    const wards = await Ward.create([
      {
        wardId: "ICU-Ward-A",
        wardName: "Intensive Care Unit - Ward A",
        department: "ICU",
        location: { floor: 2, wing: "East", area: "North" },
        totalBeds: 6,
        specialization: ["Cardiac", "Respiratory"]
      },
      {
        wardId: "ED-Ward-1",
        wardName: "Emergency Department - Ward 1",
        department: "ED",
        location: { floor: 1, wing: "Main", area: "Triage" },
        totalBeds: 12,
        specialization: ["Trauma", "Cardiac"]
      },
      {
        wardId: "Gen-Ward-B",
        wardName: "General Surgery - Ward B",
        department: "GENERAL",
        location: { floor: 3, wing: "West", area: "Post-Op" },
        totalBeds: 8,
        specialization: ["Post-Operative Care"]
      }
    ]);

    // Create beds for each ward
    for (const ward of wards) {
      const beds = [];
      for (let i = 1; i <= ward.totalBeds; i++) {
        beds.push({
          wardId: ward._id,
          bedId: `${ward.wardId}-B${String(i).padStart(2, '0')}`,
          roomNumber: `Room-${i}`,
          department: ward.department,
          bedType: i % 3 === 0 ? "INTENSIVE" : "MONITORED",
          status: i % 4 === 0 ? "CLEANING" : "AVAILABLE",
          hasMonitor: true,
          hasVentilator: i % 2 === 0,
          acuityLevel: "MODERATE",
          turnaroundTimeMinutes: 30
        });
      }
      await BedSchedule.insertMany(beds);
    }

    console.log("✅ Ward seed complete");
    process.exit(0);
  } catch (err) {
    console.error("❌ Seed error:", err);
    process.exit(1);
  }
};

seedWards();
```

Run seed:
```bash
node backend/src/seeds/seedWards.js
```

---

## Workflow Examples

### Scenario 1: Patient Admission
1. Triage identifies bed available in ICU-Ward-A
2. POST /api/beds/:bedId/assign with encounterId
3. Bed status changes: AVAILABLE → OCCUPIED
4. BedStatusHistory records: reason = "PATIENT_ADMITTED"
5. Ward occupancy rate updates automatically

### Scenario 2: Patient Discharge
1. Nurse clicks "Discharge" on patient
2. POST /api/beds/:bedId/discharge
3. Bed status changes: OCCUPIED → CLEANING
4. estimatedReadyTime calculated (statusChangedAt + cleaningDurationMinutes)
5. Housekeeping sees bed in CLEANING status
6. Housekeeping updates status to AVAILABLE when done
7. BedStatusHistory records: reason = "PATIENT_DISCHARGED" with durationMinutes

### Scenario 3: Maintenance
1. Biomedical staff identifies bed needs servicing
2. PATCH /api/beds/:bedId/status with status = "MAINTENANCE"
3. Bed hidden from assignment UI
4. After repair: PATCH status back to "AVAILABLE"
5. BedStatusHistory tracks: reason = "MAINTENANCE_COMPLETED"

---

## Key Features

✅ **Live Status Updates**
- Real-time bed status (AVAILABLE, OCCUPIED, CLEANING, MAINTENANCE)
- Auto-refresh every 5-10 seconds
- Color-coded visual indicators

✅ **Ward Organization**
- Beds grouped by ward
- Department-based filtering
- Occupancy rate per ward
- Capacity planning

✅ **Audit Trail**
- Complete change history
- Who changed what, when, and why
- Duration tracking for each status
- Patient encounter linkage

✅ **Smart Calculations**
- Auto-calculate estimatedReadyTime
- Track occupancy count per bed
- Calculate ward-level metrics
- Generate utilization reports

✅ **Responsive UI**
- Grid layout adapts to screen size
- Touch-friendly on mobile
- Quick status updates
- Filter by status

---

## Configuration

### Default Cleaning Time
```javascript
// bedSchedule.model.js
cleaningDurationMinutes: { type: Number, default: 30 }
```

Change default in seed or via API request.

### Status Filter Options
```javascript
// Valid statuses
["AVAILABLE", "OCCUPIED", "CLEANING", "MAINTENANCE"]
```

### Department Options
```javascript
// Valid departments
["ED", "ICU", "OR", "MATERNITY", "NICU", "PEDIATRICS", "PSYCH", "GENERAL", "CARDIO", "NEURO", "ONCOLOGY"]
```

### Bed Types
```javascript
// Valid bed types
["STANDARD", "MONITORED", "ISOLATION", "INTENSIVE", "HIGH_DEPENDENCY"]
```

---

## Testing

### Test Live Status Updates
```bash
# Terminal 1: Start server
npm start

# Terminal 2: Watch ward status
watch -n 5 'curl http://localhost:5001/api/beds/wards'

# Terminal 3: Update a bed
curl -X PATCH http://localhost:5001/api/beds/ObjectId/status \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "CLEANING"}'
```

### Monitor Changes
```bash
# View bed history
curl http://localhost:5001/api/beds/ObjectId/history \
  -H "Authorization: Bearer TOKEN"
```

---

## Performance Tuning

- **Query Indexes**: All key fields indexed (status, wardId, department, timestamp)
- **Lean Queries**: Used in read-only endpoints for speed
- **Aggregation**: Use /analytics/occupancy for complex metrics
- **Refresh Rate**: Adjust in WardBoard component based on load

```javascript
// Current refresh rates
const interval = setInterval(fetchWards, 10000);      // Wards list
const interval = setInterval(fetchWardDetails, 5000); // Selected ward
```

---

## Troubleshooting

**Beds not appearing?**
- Check wardId reference (must exist in Ward collection)
- Verify bed status is not MAINTENANCE (hidden by default)

**Status not updating?**
- Verify JWT token is valid
- Check user has protect middleware (req.user exists)
- Look at server logs for errors

**Occupancy rate wrong?**
- Pre-save hook in ward.model.js calculates availableBeds
- Force recalculation: `await ward.save()`

**Missing estimates for cleaning?**
- estimatedReadyTime only set when status = CLEANING
- Check cleaningDurationMinutes value

