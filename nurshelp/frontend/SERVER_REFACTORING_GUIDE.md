# Server.js Refactoring Guide: Old vs New Triage Architecture

## Current server.js Analysis

### Routes Currently Registered
```javascript
app.use("/api/nurses", nurseRouter);
app.use("/api/encounters", encounterRouter);
app.use("/api/triage", triageRouter);              // OLD: POST /api/triage/run
app.use("/api/physicians", physicianRouter);       // GET /api/physicians/recommendations?encounterId
app.use("/api/reports/nurse-confirmed", nurseConfirmedReportRouter);
app.use("/api/schedules", scheduleRouter);
```

### Current Route Definitions

#### triage.route.js (OLD)
```javascript
import { runTriage } from "../controllers/encounter.controller.js";

const router = express.Router();

// POST /api/triage/run
router.post("/run", runTriage);

export default router;
```

**Endpoint:** `POST /api/triage/run`
- Calls: `encounter.controller.js::runTriage()`
- Purpose: Run full AI triage (Python always called)
- Issue: No manual path, no computeMode support

#### physician.route.js (OLD)
```javascript
import { getRecommendations } from "../controllers/physician.controller.js";

const router = express.Router();

// GET /api/physicians/recommendations?encounterId=<id>
router.get("/recommendations", getRecommendations);

export default router;
```

**Endpoint:** `GET /api/physicians/recommendations?encounterId=<id>`
- Calls: `physician.controller.js::getRecommendations()`
- Purpose: Fetch 5 recommended physicians for encounter
- Issue: Separate from triage, physician.csv loaded conditionally

---

## NEW Refactored Architecture

### New triage.route.js

```javascript
import express from "express";
import {
  submitTriage,
  getTriageRequest,
  getUnattendedTriages,
  updateSpecialistFeedback,
  getStaffSchedule,
  markTriageAttended
} from "../controllers/triage.controller.js";

const router = express.Router();

// ===== TRIAGE SUBMISSION =====
// POST /api/triage
// Body: { triageNurseId, computeMode, symptoms, vitalSigns, ... }
// Response: { triageRequestId, manualReport, aiReport, staffAvailable, processingMetrics }
router.post("/", submitTriage);

// ===== STAFF DATA FOR UI =====
// GET /api/triage/staff?date=2026-04-11
// Response: { date, nurses: [...], physicians: [...] }
// Used by frontend to show available staff during triage entry
router.get("/staff", getStaffSchedule);

// ===== RETRIEVE TRIAGE REQUEST =====
// GET /api/triage/:id
// Response: Full TriageRequest document with manual + AI + specialist reports
router.get("/:id", getTriageRequest);

// ===== LIST UNATTENDED TRIAGES =====
// GET /api/triage?status=TRIAGED&isAttended=false&department=ED
// Response: Array of waiting TriageRequests for dashboard
router.get("/", getUnattendedTriages);

// ===== ASSIGN STAFF TO TRIAGE =====
// PATCH /api/triage/:id/attended
// Body: { nurseId, physicianId, bedId, department }
// Response: Updated TriageRequest with assignments
router.patch("/:id/attended", markTriageAttended);

// ===== SPECIALIST FEEDBACK =====
// PATCH /api/triage/:id/specialist-feedback
// Body: { specialistId, actualESILevel, actualAcuity, finalDiagnosis, feedbackNotes }
// Response: Updated TriageRequest with specialist report
router.patch("/:id/specialist-feedback", updateSpecialistFeedback);

export default router;
```

### Why Route Order Matters

**CRITICAL:** Express matches routes top-to-bottom. Place specific routes BEFORE catch-all params:

```javascript
// ✅ CORRECT ORDER
router.post("/", submitTriage);           // POST /api/triage/
router.get("/staff", getStaffSchedule);   // GET /api/triage/staff (specific)
router.get("/:id", getTriageRequest);     // GET /api/triage/:id (general param)
router.patch("/:id/attended", ...);       // PATCH /api/triage/:id/attended

// ❌ WRONG ORDER (would break /staff and /attended routes)
router.get("/:id", getTriageRequest);     // This would match /staff first!
```

---

## Server.js - Required Changes

### Step 1: Update Imports

**Old:**
```javascript
import triageRouter from "./routes/triage.route.js";
import physicianRouter from "./routes/physician.route.js";
```

**New:**
```javascript
import triageRouter from "./routes/triage.route.js";
import physicianRouter from "./routes/physician.route.js";  // Keep for backward compat
```

(No change needed - keep both)

### Step 2: Register Routes

**Old:**
```javascript
app.use("/api/triage", triageRouter);          // /api/triage/run
app.use("/api/physicians", physicianRouter);   // /api/physicians/recommendations
```

**New:**
```javascript
// NEW PRODUCTION TRIAGE ENDPOINTS
app.use("/api/triage", triageRouter);
  // POST /api/triage                      - Submit triage (manual or AI)
  // GET /api/triage/staff?date=...       - Fetch available staff
  // GET /api/triage/:id                  - Retrieve triage request
  // PATCH /api/triage/:id/attended       - Assign staff
  // PATCH /api/triage/:id/specialist-... - Add specialist review

// LEGACY - Keep for backward compatibility
app.use("/api/physicians", physicianRouter);
  // GET /api/physicians/recommendations?encounterId=... (old endpoint)
```

### Step 3: Complete New server.js

```javascript
import express from "express";
import dotenv from "dotenv";
import mongoose from "mongoose";
import cors from "cors";

import nurseRouter from "./routes/nurse.route.js";
import encounterRouter from "./routes/encounter.route.js";
import triageRouter from "./routes/triage.route.js";              // ← Updated
import physicianRouter from "./routes/physician.route.js";       // ← Keep for legacy
import nurseConfirmedReportRouter from "./routes/nurseConfirmedReport.route.js";
import scheduleRouter from "./routes/schedule.route.js";

dotenv.config();

const app = express();

// ----------------------
// 🔥 MUST BE FIRST
// ----------------------
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// ----------------------
// CORS
// ----------------------
app.use(cors({
  origin: "http://localhost:5173",
  credentials: true,
}));

// ----------------------
// MongoDB
// ----------------------
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
  console.error("❌ MONGO_URI missing");
  process.exit(1);
}

mongoose.connect(MONGO_URI)
  .then(() => console.log("✅ MongoDB connected"))
  .catch(err => {
    console.error("❌ MongoDB error:", err);
    process.exit(1);
  });

// ----------------------
// ROUTES
// ----------------------
app.use("/api/nurses", nurseRouter);
app.use("/api/encounters", encounterRouter);

// NEW: Production triage endpoints (supports manual + AI workflows)
app.use("/api/triage", triageRouter);
  // POST  /api/triage                          - Submit triage (manual or AI)
  // GET   /api/triage/staff?date=...          - Available staff for UI
  // GET   /api/triage/:id                     - Retrieve triage + all reports
  // GET   /api/triage?filters                 - Find unattended triages
  // PATCH /api/triage/:id/attended            - Assign staff
  // PATCH /api/triage/:id/specialist-feedback - Add specialist review

// LEGACY: Keep for backward compatibility with old code
app.use("/api/physicians", physicianRouter);
  // GET /api/physicians/recommendations?encounterId=... (old endpoint)

app.use("/api/reports/nurse-confirmed", nurseConfirmedReportRouter);
app.use("/api/schedules", scheduleRouter);

// ----------------------
// HEALTH CHECK
// ----------------------
app.get("/health", (req, res) => {
  res.json({ 
    ok: true,
    message: "Server running",
    triageEndpoints: [
      "POST /api/triage - Submit triage (manual or AI)",
      "GET /api/triage/staff - Get available staff",
      "GET /api/triage/:id - Retrieve triage request",
      "PATCH /api/triage/:id/attended - Assign staff",
      "PATCH /api/triage/:id/specialist-feedback - Add specialist feedback"
    ]
  });
});

// ----------------------
// GLOBAL ERROR HANDLER
// ----------------------
app.use((err, req, res, next) => {
  console.error("🔥 GLOBAL ERROR:", err);
  res.status(500).json({ message: err.message || "Server error" });
});

// ----------------------
// START
// ----------------------
const PORT = process.env.PORT || 5001;

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📋 Documentation: http://localhost:${PORT}/health`);
});
```

---

## Migration Path

### Phase 1: Deploy New Endpoints (Non-Breaking)

1. **Add new triage.controller.js** with all new functions
2. **Update triage.route.js** with all new routes
3. **Update server.js** to include new routes
4. **Start backend** - Both old and new endpoints work simultaneously
5. **Old `/api/triage/run` still works** (via encounter.controller.js::runTriage)
6. **Old `/api/physicians/recommendations` still works**

### Phase 2: Switch Frontend Gradually

1. **Update UI** to use new `/api/triage` endpoint
2. **Support two triage workflows**: manual + AI-assisted
3. **Store computeMode** in TriageRequest for audit
4. **Keep old code** in case of issues

### Phase 3: Deprecate Old Endpoints (Optional)

```javascript
// Mark old endpoints as deprecated
app.use("/api/triage/run", (req, res) => {
  res.status(410).json({
    error: "DEPRECATED",
    message: "POST /api/triage/run is deprecated. Use POST /api/triage instead",
    migration: "Include computeMode parameter: {'computeMode': 'LLM_ONLY'|'RAG_FULL'|'HYBRID'}"
  });
});
```

---

## Route Mapping Reference

### Old Endpoints → New Endpoints

| Old Endpoint | Old Purpose | New Endpoint | New Purpose | Notes |
|---|---|---|---|---|
| `POST /api/triage/run` | Run AI triage | `POST /api/triage` | Submit triage (manual or AI) | Add `computeMode` param |
| `GET /api/physicians/recommendations?encounterId=X` | Get 5 docs for encounter | `POST /api/triage` response | Included in AI response | Returned in `aiReport.recommendedPhysician` |
| None | N/A | `GET /api/triage/staff` | Fetch available staff | New for UI + AI context |
| None | N/A | `GET /api/triage/:id` | Retrieve stored triage | New for viewing all reports |
| None | N/A | `GET /api/triage?filters` | Find unattended triages | New for dashboard |
| None | N/A | `PATCH /api/triage/:id/attended` | Assign staff | New for workflow |
| None | N/A | `PATCH /api/triage/:id/specialist-feedback` | Add specialist review | New for accuracy tracking |

---

## Why This Structure?

### Benefits of New Approach

✅ **Single Endpoint for Triage** (`POST /api/triage`)
- Handles both manual and AI workflows
- Computemode controls behavior
- Response includes all needed data

✅ **Staff Data Separate** (`GET /api/triage/staff`)
- Fetched independently, used by UI and AI
- Frontend can show available staff immediately
- Reduces latency in triage entry

✅ **Complete Audit Trail**
- `GET /api/triage/:id` returns all 3 reports
- Manual report (what nurse decided)
- AI report (what AI suggested)
- Specialist report (actual diagnosis)

✅ **Backward Compatible**
- Old `/api/physicians/recommendations` still works
- Old `/api/triage/run` still works
- Migration gentle, not forced

✅ **RESTful Design**
- Follows HTTP conventions
- POST for state change
- GET for retrieval
- PATCH for updates
- Clear resource hierarchy

---

## Testing the New Flow

### Test 1: Manual Triage
```bash
curl -X POST http://localhost:5001/api/triage \
  -H "Content-Type: application/json" \
  -d '{
    "triageNurseId": "nurse123",
    "computeMode": "LLM_ONLY",
    "symptoms": ["chest pain"],
    "vitalSigns": {
      "temperature": 98.6,
      "heartRate": 102,
      "systolicBP": 145,
      "diastolicBP": 88,
      "oxygenSaturation": 94
    },
    "manualESILevel": 2,
    "manualAcuity": 75,
    "triageReasoningNotes": "Chest pain with vital abnormalities"
  }'
```

**Expected:** Returns ~500ms with manual report only

### Test 2: AI-Assisted Triage
```bash
curl -X POST http://localhost:5001/api/triage \
  -H "Content-Type: application/json" \
  -d '{
    "triageNurseId": "nurse123",
    "computeMode": "RAG_FULL",
    "symptoms": ["chest pain"],
    "vitalSigns": {...},
    "manualESILevel": 2,
    ...
  }'
```

**Expected:** Returns ~3-5s with both manual + AI reports

### Test 3: Staff Schedule
```bash
curl -X GET "http://localhost:5001/api/triage/staff?date=2026-04-11"
```

**Expected:** Returns available nurses and physicians

### Test 4: Retrieve Triage
```bash
curl -X GET "http://localhost:5001/api/triage/triageRequestId123"
```

**Expected:** Returns all 3 reports (manual, AI, specialist)

---

## Logging & Monitoring

Add logging to track endpoint usage:

```javascript
// In server.js, add logging middleware
app.use((req, res, next) => {
  if (req.path.startsWith('/api/triage')) {
    console.log(`[TRIAGE] ${req.method} ${req.path}`, {
      computeMode: req.body?.computeMode,
      triageNurseId: req.body?.triageNurseId,
      timestamp: new Date().toISOString()
    });
  }
  next();
});
```

Monitor in production:
- Count: Requests to each endpoint
- Latency: By computeMode (LLM_ONLY should be <1s)
- Errors: Failed AI calls, missing parameters
- Accuracy: Manual vs AI vs Specialist ESI levels
