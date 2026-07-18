# Implementation Checklist: Old vs New Triage Architecture

## Phase 1: Backend Infrastructure ✅ (Partially Complete)

### Models
- [x] Create `backend/src/models/triageRequest.model.js`
  - Stores all 3 reports (manual, AI, specialist)
  - Includes computeMode for audit trail
  - Tracks unattended patients

- [ ] OPTIONAL: Keep `Encounter` model for legacy compatibility
  - Old encounters won't have TriageRequest linked
  - New system uses TriageRequest as primary

### Controllers
- [x] Create `backend/src/controllers/triage.controller.js`
  - `submitTriage()` - Main POST endpoint (manual + AI)
  - `getStaffScheduleData()` - Helper to fetch staff
  - `getStaffSchedule()` - GET endpoint for staff UI
  - `getTriageRequest()` - Retrieve by ID
  - `getUnattendedTriages()` - Find waiting patients
  - `markTriageAttended()` - Assign staff
  - `updateSpecialistFeedback()` - Add specialist review

- [x] Update `backend/src/controllers/triage.controller.js` from triage.controller.js we already updated

- [ ] OPTIONAL: Keep `encounter.controller.js::runTriage()` for backward compat
  - Old `/api/triage/run` endpoint still works

### Routes
- [ ] Create `backend/src/routes/triage.route.js` (NEW)
  ```javascript
  router.post("/", submitTriage);                    // POST /api/triage
  router.get("/staff", getStaffSchedule);           // GET /api/triage/staff
  router.get("/:id", getTriageRequest);             // GET /api/triage/:id
  router.get("/", getUnattendedTriages);           // GET /api/triage?filters
  router.patch("/:id/attended", markTriageAttended); // PATCH /api/triage/:id/attended
  router.patch("/:id/specialist-feedback", updateSpecialistFeedback);
  ```

- [ ] Keep `backend/src/routes/physician.route.js` (LEGACY)
  - Backward compatibility

### Server Configuration
- [ ] Update `backend/src/server.js`
  ```javascript
  app.use("/api/triage", triageRouter);      // NEW endpoints
  app.use("/api/physicians", physicianRouter); // LEGACY (optional)
  ```

---

## Phase 2: Python Backend Integration

### ESI_Engine.py Updates
- [ ] Modify `@app.route('/triage', methods=['POST'])` in ESI_Engine.py to:
  1. Parse `computeMode` from JSON payload
  2. If `computeMode == "LLM_ONLY"`:
     - Skip physician.csv loading
     - Skip FAISS index initialization
     - Skip embedding-based search
     - Return basic ESI level + reasoning only
  
  3. If `computeMode == "RAG_FULL"`:
     - Load physician.csv
     - Initialize FAISS (threshold > 0.70)
     - Retrieve 15-20 similar cases
     - Rank physicians by specialty + workload
     - Return full response with recommendations
  
  4. If `computeMode == "HYBRID"`:
     - Load physician.csv (lighter)
     - Initialize FAISS (threshold > 0.50)
     - Retrieve 10 similar cases
     - Rank physicians with confidence weighting
     - Return balanced response

- [ ] Update logging:
  ```python
  print(f'[triage] Compute Mode: {compute_mode}')
  # Log which physician.csv loads, which FAISS threshold used
  ```

- [ ] Ensure response includes:
  - `esiLevel`, `acuityScore`, `confidence`
  - `recommendedPhysician` (if mode != "LLM_ONLY")
  - `recommendedNurse` (if available)
  - `relevantCasesRetrieved` (if mode != "LLM_ONLY")
  - `staffScheduling` context used for ranking

---

## Phase 3: Frontend Implementation

### Triage Interface Component (NEW)
- [ ] Create `frontend/src/Triage/TriageInterface.jsx`
  - Display two workflow options:
    - Option A: "Fast Manual Triage" → computeMode="LLM_ONLY"
    - Option B: "AI-Assisted Triage" → computeMode="RAG_FULL" or "HYBRID"
  - Form fields for manual entry (symptoms, vitals, ESI decision)
  - Staff selection dropdown (populated from `/api/triage/staff`)
  - computeMode radio buttons
  - Submit button

### Staff Schedule Display (NEW)
- [ ] Create `frontend/src/Triage/StaffScheduleDisplay.jsx`
  - Fetch from `GET /api/triage/staff?date=2026-04-11`
  - Display available nurses (name, specialty, workload)
  - Display available physicians (name, specialty, workload)
  - Used during manual triage for staff selection
  - Used to show staff context to AI

### Triage Request View (NEW)
- [ ] Create `frontend/src/Triage/TriageRequestView.jsx`
  - Fetch from `GET /api/triage/:id`
  - Display all 3 reports side-by-side:
    1. **Manual Report** (nurse's decision)
       - ESI level, acuity, reasoning
       - Recommended department
       - Immediate interventions
    
    2. **AI Report** (if available)
       - ESI level, acuity, confidence
       - Recommended physician
       - Similar cases with clickable details
       - "Accept" / "Override" buttons
    
    3. **Specialist Report** (if available)
       - Actual ESI, diagnosis
       - Was mistriage? If yes, type (UNDER/OVER)
       - Feedback notes
  
  - Show accuracy comparison
  - Link to specialist feedback form

### Specialist Feedback Form (NEW)
- [ ] Create `frontend/src/Triage/SpecialistFeedback.jsx`
  - Form for specialist to enter:
    - Actual ESI level
    - Final diagnosis
    - Mistriage assessment
    - Feedback notes
  - PATCH to `/api/triage/:id/specialist-feedback`
  - Shows manual vs AI vs specialist ESI for comparison

### Update PatientReport.jsx
- [ ] Replace old triage flow with new one
  - Old: `/api/triage/run` then `/api/physicians/recommendations`
  - New: `POST /api/triage` in one call
  
- [ ] If computeMode = "LLM_ONLY":
  - Fetch and display staff schedule
  - Show manual triage form only
  - No AI processing shown
  
- [ ] If computeMode = "RAG_FULL/HYBRID":
  - Show loading state during AI processing
  - Display both manual + AI reports
  - Show physician recommendations with confidence
  - Allow nurse to confirm or override

### Update RightColumn.jsx
- [ ] Adapt to display TriageRequest data instead of Encounter.triage
- [ ] Show all 3 reports (manual, AI, specialist)
- [ ] Add specialist feedback callout if available
- [ ] Display computeMode used ("LLM_ONLY", "RAG_FULL", "HYBRID")

---

## Phase 4: Integration & Migration

### Database Migration
- [ ] Create script to migrate old Encounter.triage data to TriageRequest
  ```python
  for encounter in Encounter.find():
    if encounter.triage:
      triageRequest = {
        encounterId: encounter._id,
        triageNurseId: encounter.nurse_id,
        manualTriageReport: { ... },  # Extract from encounter.triage
        aiTriageReport: encounter.triage,  # Store AI as-is
        computeMode: "RAG_FULL",  # Assume old ones used RAG
        status: "TRIAGED",
        isAttended: encounter.status != "pending"
      }
      TriageRequest.create(triageRequest)
  ```

### Backward Compatibility
- [ ] Keep old endpoints working:
  - `POST /api/triage/run` (calls encounter.controller.js::runTriage)
  - `GET /api/physicians/recommendations` (backward compat)
  
- [ ] Old code continues to work while new UI migrated
- [ ] Can run both simultaneously

### Feature Flags (Optional)
- [ ] Add `.env` flag: `USE_NEW_TRIAGE=true|false`
  - Frontend checks flag to show old vs new UI
  - Gradual rollout enabled
  - Can revert quickly if issues

---

## Phase 5: Testing & Validation

### Unit Tests
- [ ] Test triage.controller.js::submitTriage()
  - Verify computeMode="LLM_ONLY" skips Python call
  - Verify computeMode="RAG_FULL" calls Python
  - Verify staff data is fetched and passed correctly

- [ ] Test triage.controller.js::getStaffSchedule()
  - Verify current staff returned
  - Verify date filtering works

- [ ] Test TriageRequest model
  - Verify all 3 reports stored correctly
  - Verify indexes work for common queries

### Integration Tests
- [ ] Test manual workflow end-to-end
  - POST /api/triage with computeMode="LLM_ONLY"
  - Verify response ~500ms
  - Verify only manual report in response

- [ ] Test AI workflow end-to-end
  - POST /api/triage with computeMode="RAG_FULL"
  - Verify response includes AI report
  - Verify physician recommendations present

- [ ] Test specialist feedback
  - PATCH /api/triage/:id/specialist-feedback
  - Verify mistriage detection works
  - Verify accuracy comparison displayed

### Manual Testing Scenarios
- [ ] Triage nurse does manual ESI (no AI)
  - Chooses "Fast Manual"
  - Completes in <1 second
  - Staff list displays
  - Can manually select nurse/physician
  
- [ ] Triage nurse uses AI assistance
  - Chooses "AI-Assisted"
  - ESI_Engine loads physician.csv + FAISS
  - Returns recommendations
  - Nurse sees both manual + AI reports
  - Can accept or override AI suggestion
  
- [ ] Specialist reviews patient later
  - Reviews both manual + AI decision
  - Assesses accuracy
  - Submits feedback
  - Accuracy metrics updated in system

---

## Phase 6: Deployment & Monitoring

### Pre-Deployment
- [ ] Backup MongoDB (TriageRequest collection new)
- [ ] Test both old and new code paths
- [ ] Verify no breaking changes in Encounter model

### Deployment Steps
1. [ ] Deploy new triage.controller.js
2. [ ] Deploy new triage.route.js 
3. [ ] Deploy updated server.js
4. [ ] Deploy ESI_Engine.py updates (Python side)
5. [ ] Verify `/health` endpoint shows new routes
6. [ ] Deploy frontend new TriageInterface components
7. [ ] Enable new UI (feature flag or direct replacement)
8. [ ] Monitor error logs for issues

### Monitoring
- [ ] Track endpoint usage:
  - Count requests per mode (LLM_ONLY vs RAG_FULL vs HYBRID)
  - Count errors per mode
  - Latency percentiles (p50, p95, p99)
  
- [ ] Track accuracy metrics:
  - Manual ESI vs Specialist ESI agreement rate
  - AI ESI vs Specialist ESI agreement rate
  - Mistriage rate (UNDER/OVER)
  - Manual override rate (when nurse ignores AI)
  
- [ ] Alert on:
  - ESI_Engine failures (Python not responding)
  - physician.csv load failures
  - FAISS index initialization failures
  - High error rate on new endpoints

### Rollback Plan
- [ ] If critical issues:
  1. Disable new UI (feature flag)
  2. Keep old endpoints functional
  3. New TriageRequest data remains (safe)
  4. Old Encounter flow still works

---

## File Change Summary

### Backend Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/src/models/triageRequest.model.js` | Create | Store all 3 reports |
| `backend/src/controllers/triage.controller.js` | Create/Extend | New triage logic |
| `backend/src/routes/triage.route.js` | Create | Route all new endpoints |
| `backend/src/server.js` | Modify | Register new routes |
| `backend/ML/ESI_Engine.py` | Modify | Parse computeMode, conditionally load FAISS |
| `backend/src/controllers/schedule.controller.js` | Modify | Add getStaffScheduleData export |

### Frontend Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `frontend/src/Triage/TriageInterface.jsx` | Create | Main triage entry UI |
| `frontend/src/Triage/StaffScheduleDisplay.jsx` | Create | Show available staff |
| `frontend/src/Triage/TriageRequestView.jsx` | Create | View all 3 reports |
| `frontend/src/Triage/SpecialistFeedback.jsx` | Create | Specialist review form |
| `frontend/src/PatientSummary/PatientReport.jsx` | Modify | Use new endpoints |
| `frontend/src/PatientSummary/Components/RightColumn.jsx` | Modify | Display new TriageRequest |

---

## Success Criteria

✅ **Phase 1 Complete When:**
- TriageRequest model exists in MongoDB
- New triage endpoints callable in Postman
- Old endpoints still work (backward compat)
- server.js registers both old + new routes

✅ **Phase 2 Complete When:**
- ESI_Engine.py checks computeMode parameter
- LLM_ONLY mode skips FAISS loading
- RAG_FULL mode loads physician.csv + FAISS
- Response includes computeMode in fields

✅ **Phase 3 Complete When:**
- Frontend shows triage interface
- Two workflow options available
- Staff schedule displays
- Can manual triage in <1s
- Can AI triage in 2-5s

✅ **Phase 4 Complete When:**
- Old encounters migrated to TriageRequest
- Backward compatibility tested
- No breaking changes observed

✅ **Phase 5 Complete When:**
- All test scenarios pass
- Manual workflow fast (<1s)
- AI workflow slower (2-5s) but complete
- Specialist feedback stored correctly

✅ **Phase 6 Complete When:**
- Deployed to production
- Monitoring dashboards show endpoint usage
- No critical errors
- Users transitioning to new workflow
