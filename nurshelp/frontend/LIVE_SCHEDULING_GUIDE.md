# Live Scheduling Guide

## Backend Setup (✅ COMPLETE)

### Database Models Created:
1. **PhysicianSchedule** - Daily physician availability, specialty, workload
2. **NurseSchedule** - Daily nurse roster, beds, ESI level, certifications
3. **BedSchedule** - Real-time bed status, occupancy, features

### Endpoints Implemented:

#### Physician Schedules
- `GET /api/schedules/physicians?date=YYYY-MM-DD&specialty=Cardiology&status=ON_DUTY`
- `GET /api/schedules/physicians/me?date=YYYY-MM-DD` - Current user's schedule
- `POST /api/schedules/physicians` - Create/Update single schedule
- `POST /api/schedules/physicians/import` - Bulk import from CSV

#### Nurse Schedules
- `GET /api/schedules/nurses?date=YYYY-MM-DD&status=ON_DUTY`
- `GET /api/schedules/nurses/me?date=YYYY-MM-DD` - Current user's schedule
- `POST /api/schedules/nurses` - Create/Update single schedule
- `POST /api/schedules/nurses/import` - Bulk import from CSV

#### Bed Schedules
- `GET /api/schedules/beds?status=AVAILABLE&department=ER`
- `GET /api/schedules/beds/all?department=ER`
- `POST /api/schedules/beds/assign` - Assign bed to patient/nurse
- `POST /api/schedules/beds/release` - Mark bed for cleaning
- `POST /api/schedules/beds/initialize` - Create default 9-bed layout

---

## Frontend Setup (✅ COMPLETE)

### Staff Schedule Page Updated:
- **3 Tabs**: My Schedule | Physicians | Nurses
- **Date Picker**: Select schedule date (default: today)
- **Add/Edit UI**: Modal with discipline-specific fields
- **CSV Import/Export**: Upload or download schedules
- **Real-time Filtering**: By status, specialty (physicians), ESI level (nurses)
- **Live Display**: Shows current workload (patients for physicians, beds for nurses)

---

## CSV Format Specifications

### Physicians Schedule CSV
```csv
physicianId,name,surname,email,specialty,date,status,shiftStart,shiftEnd,maxPatients,notes
PHY001,John,Smith,john@hospital.com,Cardiology,2026-04-10,ON_DUTY,08:00,16:00,20,"Lead cardiologist"
PHY002,Sarah,Johnson,sarah@hospital.com,ER,2026-04-10,ON_CALL,14:00,22:00,25,""
PHY003,Michael,Williams,michael@hospital.com,Neurology,2026-04-10,OFF,,,0,"On vacation"
```

**Required Fields**: physicianId, name, specialty, date
**Optional Fields**: surname, email, status (default: OFF), shiftStart, shiftEnd (default: 08:00-16:00), maxPatients (default: 20), notes

**Status Values**: ON_DUTY | ON_CALL | OFF | VACATION

### Nurses Schedule CSV
```csv
nurseId,name,surname,email,licenseNumber,date,status,shiftStart,shiftEnd,maxBeds,esiLevel,notes
NUR001,Jane,Doe,jane@hospital.com,RN123456,2026-04-10,ON_DUTY,07:00,19:00,5,2,"ACLS certified"
NUR002,Bob,Smith,bob@hospital.com,RN789012,2026-04-10,ON_DUTY,07:00,15:00,4,3,"Prefers pediatrics"
NUR003,Alice,Brown,alice@hospital.com,RN345678,2026-04-10,OFF,,,0,3,"Call-out"
```

**Required Fields**: nurseId, name, date
**Optional Fields**: surname, email, licenseNumber, status (default: OFF), shiftStart, shiftEnd (default: 07:00-19:00), maxBeds (default: 5), esiLevel (default: 3, range 1-3), certifications (pipe-separated: ACLS|BLS|PALS), notes

**Status Values**: ON_DUTY | ON_CALL | OFF | VACATION | CALL_OUT
**ESI Levels**: 1 (Expert) | 2 (Intermediate) | 3 (Trainee)

---

## Quick Start

### 1. Initialize Beds (One-time)
```bash
curl -X POST http://localhost:5001/api/schedules/beds/initialize \
  -H "Content-Type: application/json"
```
Creates 9 default beds: ER (3) + ICU (3) + General (3)

### 2. Import Physician Schedule
Save CSV file and upload via UI or:
```bash
curl -X POST http://localhost:5001/api/schedules/physicians/import \
  -H "Content-Type: application/json" \
  -d '{"csvData": "physicianId,name,specialty,...\nPHY001,John,Cardiology,...\n"}'
```

### 3. Import Nurse Schedule
```bash
curl -X POST http://localhost:5001/api/schedules/nurses/import \
  -H "Content-Type: application/json" \
  -d '{"csvData": "nurseId,name,date,...\nNUR001,Jane,2026-04-10,...\n"}'
```

### 4. Check Availability
```bash
# Get available physicians today
curl "http://localhost:5001/api/schedules/physicians?date=2026-04-10&status=ON_DUTY"

# Get available nurses in ICU
curl "http://localhost:5001/api/schedules/nurses?date=2026-04-10&status=ON_DUTY"

# Get available beds
curl "http://localhost:5001/api/schedules/beds?status=AVAILABLE"
```

### 5. View Personal Schedule
```bash
# Current logged-in user's nurse schedule
curl "http://localhost:5001/api/schedules/nurses/me?date=2026-04-10" \
  -H "Authorization: Bearer <JWT_TOKEN>"

# Or physician schedule
curl "http://localhost:5001/api/schedules/physicians/me?date=2026-04-10" \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

---

## Integration with Patient Report Tabs

When implementing manual triage bypass (Patient Report Tab 2), fetch schedules:

```javascript
// Get available physicians
const physicianRes = await fetch(
  `http://localhost:5001/api/schedules/physicians?date=${todayDate}`,
  { headers: { 'Authorization': `Bearer ${token}` } }
);
const { physicians } = await physicianRes.json();

// Get available nurses
const nurseRes = await fetch(
  `http://localhost:5001/api/schedules/nurses?date=${todayDate}`,
  { headers: { 'Authorization': `Bearer ${token}` } }
);
const { nurses } = await nurseRes.json();

// Get available beds
const bedRes = await fetch(
  `http://localhost:5001/api/schedules/beds?status=AVAILABLE`,
  { headers: { 'Authorization': `Bearer ${token}` } }
);
const { beds } = await bedRes.json();

// Render selection UI from these lists
```

---

## Testing Checklist

- [ ] Backend server starts without errors: `npm start` in backend/
- [ ] MongoDB connection successful (check console logs)
- [ ] `/api/schedules/beds/initialize` creates 9 beds
- [ ] Can export CSV from frontend (Staff Schedule page)
- [ ] Can import CSV from frontend (Staff Schedule page)
- [ ] Physician schedule shows in "Physicians" tab after import
- [ ] Nurse schedule shows in "Nurses" tab after import
- [ ] "My Schedule" tab shows logged-in user's schedule
- [ ] Date picker changes are reflected in displayed data
- [ ] Filters work (specialty for physicians, status for both)
- [ ] Add manual schedule via modal saves to database
- [ ] Import errors are displayed in error message
- [ ] Workload shows correctly (current/max patients for physicians, beds for nurses)

---

## Next Steps

Once live scheduling is perfected:
1. Proceed with Patient Report Tabs (3-tab view + bypass workflow)
2. Implement compute modes settings
3. Add real-time bed polling for Beds Allocation page
4. Add CSV hot-reload scheduling for daily updates
5. End-to-end testing of entire NurseDashboard workflow
