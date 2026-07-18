# Manual Triage UI Implementation Guide

## Overview

This guide explains how to integrate the new Manual Triage UI components into the existing Patient Dashboard. The new components provide a complete manual triage workflow with two distinct paths: Manual Triage (pure human judgment) and AI-Assisted Triage (optional AI help).

## New Components Created

### 1. **TriageTabHeader.jsx** - Header with Tab Selection
- Displays "Patient Dashboard" title
- Shows Manual Triage and AI-Assisted Triage tabs
- Replaces original header with tab functionality
- Imports: `./TriageTabHeader.jsx`

### 2. **ManualTriageForm.jsx** - Complete Manual Triage Workflow
- Full form for manual triage entry
- Steps:
  1. Display patient info and current vitals
  2. Nurse manually selects ESI level (color-coded, 1-5)
  3. Nurse enters acuity score (0-100 slider)
  4. Nurse enters clinical reasoning
  5. Nurse selects recommended department
  6. Nurse enters immediate interventions
  7. Nurse selects assigned nurse, physician, and bed
  8. Submit (POST /api/triage without computeMode)
- Imports: `./ManualTriageForm.jsx`

### 3. **StaffScheduleDisplay.jsx** - Staff Selection Component
- Shows available nurses, physicians, and beds
- Fetches from `GET /api/triage/staff`
- Used within ManualTriageForm for staff selection
- Color-coded: Available = green, Occupied/Off-shift = red
- Imports: `./StaffScheduleDisplay.jsx`

### 4. **TriageTabContainer.jsx** - Tab Manager
- Manages switching between Manual and AI-Assisted tabs
- Shows ManualTriageForm when Manual is selected
- Shows placeholder for AI-Assisted (coming soon)
- Imports: `./TriageTabContainer.jsx`

## Integration Steps

### Step 1: Update PS.jsx

Replace the Header import and usage with TriageTabContainer:

```javascript
// At the top of PS.jsx, add:
import TriageTabContainer from "./Components/TriageTabContainer";

// Remove or keep the old Header for reference
// import Header from "./Components/Header";

// In the JSX layout, replace the header and old form sections:
export default function PS() {
  const [patientData, setPatientData] = useState(...);
  const [vitalsData, setVitalsData] = useState(...);
  const [triageResult, setTriageResult] = useState(null);
  
  // ... existing state ...
  
  const handleTriageSubmit = (result) => {
    // Handle successful triage submission
    setTriageResult(result);
    console.log("Triage submitted:", result);
    // Optionally: navigate to triage view or dashboard
  };

  return (
    <div style={layoutStyle}>
      {/* Replace old Header with TriageTabHeader + Container */}
      <TriageTabContainer
        patientData={patientData}
        vitalsData={vitalsData}
        onTriageSubmit={handleTriageSubmit}
      />
      
      {/* Keep existing columns */}
      <div style={columnLayoutStyle}>
        <LeftColumn data={patientData} setData={setPatientData} />
        <MiddleColumn data={transcript} setData={setTranscript} />
        <RightColumn 
          triageResult={triageResult}
          vitalsData={vitalsData}
        />
      </div>
    </div>
  );
}
```

### Step 2: Verify API Endpoints

Ensure backend has these endpoints implemented:

```
GET /api/triage/staff?date=YYYY-MM-DD
  └─ Returns: { nurses: [...], physicians: [...], beds: [...] }

POST /api/triage (NO computeMode = Manual path)
  ├─ Input: {
  │   triageNurseId,
  │   symptoms, vitalSigns, medicalHistory,
  │   manualESILevel, manualAcuity, triageReasoningNotes,
  │   assignedNurseId, assignedPhysicianId, bedAssigned
  │ }
  └─ Output: { triageRequestId, manualReport, processingMetrics }
```

### Step 3: Update localStorage Keys

The components use these localStorage keys:
- `currentNurseId` - Current logged-in nurse ID (set during login)
- `token` - Authentication token (already set by auth system)

### Step 4: Styling Integration

The components use inline styles and CSS files. Ensure these CSS files are in place:
- `ManualTriageForm.css`
- `StaffScheduleDisplay.css`
- `TriageTabContainer.css`
- `TriageTabHeader.css`

## User Flow Diagram

```
Patient Dashboard
│
├─ Tab: Manual Triage
│  ├─ Display patient info + vitals
│  ├─ Nurse selects ESI level (1-5)
│  ├─ Nurse enters acuity score
│  ├─ Nurse enters clinical reasoning
│  ├─ Nurse selects department
│  ├─ Nurse enters interventions
│  ├─ GET /api/triage/staff (fetch available staff)
│  ├─ Nurse selects: Nurse + Physician + Bed
│  ├─ Submit (POST /api/triage, NO computeMode)
│  └─ Result: Manual triage report in EHR
│             Unattended status in Nurse Dashboard
│             Can add specialist feedback later
│
└─ Tab: AI-Assisted Triage (Coming Soon)
   └─ Will include same flow + optional AI suggestions
```

## Integration with EHR & Dashboard

### EHR (Patient Record View)
After manual triage submission:
- TriageRequest created with `manualTriageReport` only
- `computeMode: null` (indicates manual path)
- `aiTriageReport: null` (no AI used)
- Patient appears in EHR with manual triage visible
- Shows: ESI level, acuity, reasoning, assigned staff, interventions

### Nurse Dashboard
After manual triage submission:
- Patient appears in "Unattended Triages" list if not yet confirmed
- Shows assigned nurse/physician (can confirm or reassign)
- Priority queue sorted by ESI level
- Can add specialist feedback once reviewed

### Timeline
1. **Triage Submitted**: Status = "TRIAGED", `isAttended: false`
2. **Nurse Confirms**: Status = "ATTENDED", `isAttended: true`
3. **Specialist Reviews**: Add feedback via PATCH /api/triage/:id/specialist-feedback
4. **EHR Complete**: All reports visible (manual + specialist)

## Key Features of Manual Triage Form

### ESI Color Coding
- Level 1 (Red) - Emergent
- Level 2 (Orange) - Urgent
- Level 3 (Yellow) - Semi-Urgent
- Level 4 (Blue) - Less Urgent
- Level 5 (Green) - Minor

### Validation
- All fields required before submit
- ESI level must be selected
- Clinical reasoning required (min 1 char)
- Staff selection required (nurse + physician + bed)
- Button disabled until form valid

### Success/Error Handling
- Shows error alerts on submission failure
- Success message appears for 2 seconds
- Form clears after successful submission
- Calls parent callback `onTriageSubmit(result)`

### Staff Selection
- Collapsible staff schedule section
- Shows available and unavailable staff
- Unavailable staff displayed as disabled
- Selection summary shows confirmation
- Can deselect and reselect

## Testing the Manual Triage Flow

### Test Case 1: Happy Path
1. Navigate to Patient Dashboard
2. Click "Manual Triage" tab
3. Enter ESI level 2
4. Enter acuity 75
5. Enter reasoning: "Chest pain with vital abnormalities"
6. Select department: "ED"
7. Enter interventions: "ECG\nIV Access"
8. Expand staff section
9. Select nurse, physician, bed
10. Click "Submit Manual Triage"
11. Verify: Success message appears, form clears
12. Check EHR: Patient appears with manual triage report

### Test Case 2: Validation
1. Click Submit without filling fields
2. Verify: Error "Please complete all required fields"
3. Select all fields
4. Verify: Submit button becomes enabled

### Test Case 3: Staff Selection
1. Expand staff section
2. Click unavailable staff (off-shift)
3. Verify: Cannot select (card remains disabled)
4. Click available staff
5. Verify: Card highlights in blue
6. Selection summary updates
7. Deselect
8. Verify: Can reselect different staff

## Backend Integration Checklist

- [ ] `GET /api/triage/staff` endpoint returns staff data
- [ ] `POST /api/triage` endpoint accepts manual payload
- [ ] Manual payload without `computeMode` triggers manual path
- [ ] Manual path does NOT call ESI_Engine.py
- [ ] TriageRequest model stores manual reports
- [ ] Database stores `computeMode: null` for manual
- [ ] EHR page displays manual triage report
- [ ] Nurse Dashboard shows unattended tab includes manual triages
- [ ] Specialist feedback endpoint ready for accuracy tracking

## Future: AI-Assisted Tab

The AI-Assisted Triage tab is a placeholder. When implemented:

1. Same initial form as Manual Triage
2. Optional step: Choose AI mode (RAG_FULL or HYBRID)
3. Submit with `computeMode` parameter
4. Backend calls ESI_Engine.py
5. Display both manual AND AI reports side-by-side
6. Nurse can accept AI or confirm manual decision
7. Both reports stored for accuracy comparison

## Troubleshooting

### Issue: "Staff schedule not loading"
- Check: Backend `/api/triage/staff` is running
- Check: Mock data in SSO component (fallback works)
- Check: Network tab for 404 errors

### Issue: "Form won't submit"
- Check: All required fields filled (validation msg shows which ones)
- Check: Browser console for JS errors
- Check: Backend `/api/triage` is running
- Check: Token stored in localStorage

### Issue: "Styling looks off"
- Check: CSS files imported in components
- Check: No conflicts with existing PS.css classes
- Check: Browser dev tools computed styles

## Files Modified/Created

**New Files:**
- `/frontend/src/PatientSummary/Components/TriageTabHeader.jsx`
- `/frontend/src/PatientSummary/Components/TriageTabHeader.css`
- `/frontend/src/PatientSummary/Components/ManualTriageForm.jsx`
- `/frontend/src/PatientSummary/Components/ManualTriageForm.css`
- `/frontend/src/PatientSummary/Components/StaffScheduleDisplay.jsx`
- `/frontend/src/PatientSummary/Components/StaffScheduleDisplay.css`
- `/frontend/src/PatientSummary/Components/TriageTabContainer.jsx`
- `/frontend/src/PatientSummary/Components/TriageTabContainer.css`
- `/frontend/src/PatientSummary/MANUAL_TRIAGE_UI_GUIDE.md` (this file)

**Files to Update:**
- `/frontend/src/PatientSummary/PS.jsx` (integrate TriageTabContainer)
- `/backend/src/routes/triage.route.js` (ensure endpoints exist)
- `/backend/src/controllers/triage.controller.js` (ensure logic)
- `/backend/src/models/triageRequest.model.js` (ensure schema)

## Success Criteria

✅ Manual triage form displays correctly
✅ All fields validate properly
✅ Staff schedule loads and allows selection
✅ Form submits to /api/triage without computeMode
✅ Manual triage appears in EHR
✅ Patient shows as unattended in Nurse Dashboard
✅ Priority queue sorts by ESI level
✅ Specialist can add feedback later
✅ No errors in browser console
✅ UI responsive on mobile

## Next Steps

1. **Now:** Create components (DONE)
2. **Next:** Integrate into PS.jsx
3. **Then:** Test with backend endpoints
4. **Later:** Implement AI-Assisted Triage tab
5. **Finally:** Deploy to production with feature flag
