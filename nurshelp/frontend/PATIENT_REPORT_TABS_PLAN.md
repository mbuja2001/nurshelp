# Patient Report: 3-Tab View Implementation Plan

## Current State

### EHR.jsx (Has 3-Tab Structure)
✅ Already has tabs:
- `ai-suggested` (AI Suggested Report)
- `nurse-approved` (Nurse Confirmed Report)
- `physician-attended` (Physician Feedback Report)

✅ Has mock data for each report type:
```javascript
aiSuggestedReport: { diagnosis, recommendations, confidence }
nurseApprovedReport: { approvedAt, nurseNotes }
physicianReport: { attendedAt, physicianName, labResults, scans, resources, finalDiagnosis, discharged }
```

### PatientReport.jsx (Currently Single View)
❌ Has editable fields but NO TABS
❌ Focuses on rendering AI data with edit capabilities
❌ No "bypass AI" flow for manual triage

---

## Proposed: PatientReport.jsx - 3-Tab View

### Tab 1: "AI Suggested Report" (Read-Only View)
```
┌─────────────────────────────────────────────┐
│ AI Suggested Triage Report                  │
├─────────────────────────────────────────────┤
│                                             │
│ Generated: 2026-04-10 14:32:15             │
│ Confidence: 94%                            │
│                                             │
│ AI Diagnosis: Acute Myocardial Infarction  │
│ Recommendations:                            │
│ ├─ Admit to Cardiac Care Unit              │
│ ├─ 12-Lead ECG                             │
│ └─ Troponin Level Test                     │
│                                             │
│ Suggested Physician:                       │
│ └─ Dr. Smith (Cardiology) - Match: 0.94   │
│                                             │
│ [Bypass AI & Manual Triage] [Confirm]     │
└─────────────────────────────────────────────┘
```

### Tab 2: "Manual Triage (Bypass AI)" (Interactive Edit)
```
When user clicks [Bypass AI & Manual Triage]:

┌─────────────────────────────────────────────┐
│ Manual Triage Confirmation                  │
├─────────────────────────────────────────────┤
│                                             │
│🚨 You're bypassing AI recommendations      │
│                                             │
│ Diagnosis (Manual Entry):                  │
│ └─ [Text field] "_______________"          │
│                                             │
│ AVAILABLE PHYSICIANS (From Schedule):      │
│ ├─ Dr. Smith (Cardiology) [On Duty]        │
│ │  Current Patients: 3/20                  │
│ │  [Select]                                │
│ │                                          │
│ ├─ Dr. Johnson (ER) [On Call]             │
│ │  Current Patients: 1/20                  │
│ │  [Select]                                │
│ │                                          │
│ └─ Dr. Williams (Cardiology) [On Duty]    │
│    Current Patients: 5/20                  │
│    [Select]                                │
│                                             │
│ AVAILABLE NURSES (From Schedule):          │
│ ├─ Jane Doe [Available]                    │
│ │  Assigned Beds: 2/5                      │
│ │  [Select]                                │
│ │                                          │
│ └─ John Smith [Available]                  │
│    Assigned Beds: 1/5                      │
│    [Select]                                │
│                                             │
│ AVAILABLE BEDS:                            │
│ ├─ ER-01 [Available] - Standard            │
│ ├─ ICU-03 [Available] - Monitored         │
│ └─ General-12 [Available] - Standard       │
│                                             │
│ Manual Notes:                              │
│ └─ [Text area] "_______________"          │
│                                             │
│ [Cancel] [Save Triage]                    │
└─────────────────────────────────────────────┘
```

### Tab 3: "Nurse Confirmed Triage" (View Confirmed)
```
┌─────────────────────────────────────────────┐
│ Nurse Confirmed Triage Report               │
├─────────────────────────────────────────────┤
│                                             │
│ Confirmed By: Jane Doe (Nurse)             │
│ Confirmed At: 2026-04-10 14:45:32         │
│                                             │
│ Manual Diagnosis: Acute Chest Pain        │
│ Assigned Physician: Dr. Smith (Cardiology) │
│ Assigned Nurse: Jane Doe                   │
│ Assigned Bed: ER-01 (Standard)            │
│                                             │
│ Nurse Notes:                               │
│ "Patient alert and oriented. Vitals       │
│  stable but experiencing chest tightness" │
│                                             │
│ Status: AWAITING PHYSICIAN REVIEW         │
│ [Edit] [Finalize]                         │
└─────────────────────────────────────────────┘
```

### Tab 4: "Physician Feedback Report" (Read-Only)
```
┌─────────────────────────────────────────────┐
│ Physician Feedback Report                   │
├─────────────────────────────────────────────┤
│                                             │
│ Reviewed By: Dr. Smith (Cardiology)        │
│ Reviewed At: 2026-04-10 15:12:00          │
│                                             │
│ Final Diagnosis: Stable Angina             │
│ Lab Results:                               │
│ ├─ Troponin: 0.02 ng/mL (Normal)          │
│ ├─ BNP: 450 pg/mL (Elevated)              │
│                                             │
│ Scans:                                     │
│ └─ ECG: Mild ST-T wave changes            │
│                                             │
│ Treatment Plan:                            │
│ ├─ IV Line inserted                       │
│ ├─ Aspirin 325mg PO                       │
│ └─ Nitroglycerin PRN                      │
│                                             │
│ Disposition: ADMITTED                     │
│ Bed Assigned: ER-01 → ICU-03 (Transfer)   │
│ Estimated Stay: 2-3 days                  │
│                                             │
│ [Discharge] [Transfer] [Comments]         │
└─────────────────────────────────────────────┘
```

---

## Data Flow: AI Triage → Manual Bypass → Nurse Confirm → Physician Review

```
Step 1: AI Triage Generated
└─ Patient arrives
└─ AI suggests: Diagnosis (94% confidence)
└─ AI recommends: Dr. Smith, Cardiology
└─ User sees Tab 1: "AI Suggested Report"

Step 2: Nurse Bypasses AI (Optional)
└─ Nurse clicks [Bypass AI & Manual Triage]
└─ User moves to Tab 2: "Manual Triage"
└─ API Calls:
   ├─ GET /api/schedules/physicians (available today)
   ├─ GET /api/schedules/nurses (available today)
   └─ GET /api/schedules/beds (available today)
└─ Nurse manually enters diagnosis
└─ Nurse selects from available physicians/nurses/beds
└─ Nurse clicks [Save Triage]
└─ Data saved: creates nurseConfirmedReport

Step 3: Nurse Confirmed Report Created
└─ Tab 2 now shows "Nurse Confirmed Triage"
└─ Displays: Manual diagnosis + selected staff + nurse notes
└─ Status: "AWAITING PHYSICIAN REVIEW"

Step 4: Physician Reviews & Provides Feedback
└─ Physician views Tab 3 & 4
└─ Adds lab results, imaging, final diagnosis
└─ Updates disposition (admitted/discharged/transferred)
└─ Tab 4 shows: "Physician Feedback Report"
```

---

## Backend Endpoints Required

### For Tab 2 (Manual Triage Bypass) - Scheduling Data:
```
GET /api/schedules/physicians?date=2026-04-10
  Returns: { physicians: [
    { id, name, specialty, status, currentPatientCount, maxPatients },
    ...
  ]}

GET /api/schedules/nurses?date=2026-04-10
  Returns: { nurses: [
    { id, name, status, currentPatientCount, maxPatients, assignedBeds: [...] },
    ...
  ]}

GET /api/schedules/beds
  Returns: { beds: [
    { bedId, roomNumber, bedType, status, currentEncounterId, acuityLevel },
    ...
  ]}
```

### For Tab Creation - Report Data:
```
GET /api/encounters/:id/triage
  Returns: { 
    aiSuggestedReport: {...},
    nurseConfirmedReport: {...},  // null if not confirmed yet
    physicianReport: {...}         // null if not reviewed yet
  }

PATCH /api/encounters/:id/triage
  Saves manual triage + staff assignment
  Body: { 
    diagnosis, 
    selectedPhysician, 
    selectedNurse, 
    selectedBed, 
    nurseNotes 
  }
```

---

## Frontend Implementation

### Changes to PatientReport.jsx:
1. Add `activeTab` state: ['ai-suggested', 'manual-triage', 'nurse-confirmed', 'physician-feedback']
2. Add Tab Navigation UI (4 buttons)
3. Render different components based on activeTab:
   - Tab 1: TriageAISuggestedView (read-only)
   - Tab 2: TriageManualBypassView (interactive with schedules)
   - Tab 3: TriageNurseConfirmedView (read-only)
   - Tab 4: TriagePhysicianFeedbackView (read-only)
4. Add [Bypass AI & Manual Triage] button → switches to Tab 2
5. Add useSchedules hook to fetch physician/nurse/bed data

### New Components:
- `TriageAISuggestedView.jsx` - Display AI report
- `TriageManualBypassView.jsx` - Manual entry + schedule selection
- `TriageNurseConfirmedView.jsx` - Show confirmed triage
- `TriagePhysicianFeedbackView.jsx` - Show physician report

---

## Questions Before Implementation

1. ✅ Should Tab 2 only appear if nurse clicks [Bypass AI]? Or always show?
2. ✅ Should nurses be able to edit Tab 3 after saving? Or locked?
3. ✅ Should physicians automatically get notified when Tab 3 is ready?
4. ✅ Should discharge/transfer buttons be on Tab 4 only?
5. ✅ Should we show "AI Confidence Score" prominently to encourage/discourage bypass?

---

## Implementation Checklist

- [ ] Add 3-tab navigation to PatientReport.jsx
- [ ] Create TriageAISuggestedView component
- [ ] Create TriageManualBypassView component (with schedule fetching)
- [ ] Create TriageNurseConfirmedView component
- [ ] Create TriagePhysicianFeedbackView component
- [ ] Add useSchedules hook
- [ ] Update mock data in PatientReport to include all 3 report types
- [ ] Test tab switching
- [ ] Test [Bypass AI] flow
- [ ] Test manual staff selection from schedules
- [ ] Integration test: EHR → PatientReport tabs sync
