# PS.jsx Integration Instructions

## Current State Analysis

The existing `PS.jsx` currently has:
- Simple Header component with Log Out, title, and Intake Status buttons
- Three-column layout (LeftColumn, MiddleColumn, RightColumn)
- Patient data state management (localStorage persistence)
- Vitals state management
- Transcript state management

## Integration Plan

We need to replace the header section and add the tab-based triage workflow.

## Step-by-Step Integration

### 1. Import New Components

Add these imports at the top of PS.jsx (after existing imports):

```javascript
// NEW: Import tab-based triage components
import TriageTabContainer from "./Components/TriageTabContainer";
// Remove or comment out old header if you want
// import Header from "./Components/Header";
```

### 2. Add State for Triage Results

In the `useEffect` or state initialization section, add:

```javascript
const [triageResult, setTriageResult] = useState(null);
const [showTriageForm, setShowTriageForm] = useState(false);

useEffect(() => {
  // Try to restore triage result from localStorage
  const savedTriageResult = localStorage.getItem("triageResult");
  if (savedTriageResult) {
    setTriageResult(JSON.parse(savedTriageResult));
    setShowTriageForm(true);
  }
}, []);
```

### 3. Add Triage Submission Handler

Add this function in the component:

```javascript
const handleTriageSubmit = (triageData) => {
  // Save triage result to state and localStorage
  setTriageResult(triageData);
  localStorage.setItem("triageResult", JSON.stringify(triageData));
  
  // Log for debugging
  console.log("Manual Triage Submitted:", triageData);
  
  // Optional: Close triage form or navigate
  // setShowTriageForm(false); // If using a modal/drawer
  
  // Optional: Show success message
  alert("Manual triage submitted successfully!");
};
```

### 4. Replace Header in JSX

Find the current header section in PS.jsx and replace it:

**OLD (Current PS.jsx):**
```javascript
<Header 
  onLogOut={handleLogOut} 
  onIntakeStatus={handleIntakeStatus}
/>
```

**NEW:**
```javascript
<TriageTabContainer
  patientData={patientData}
  vitalsData={vitalsData}
  onTriageSubmit={handleTriageSubmit}
/>
```

### 5. Update Column Layout (Optional Enhancement)

If the right column needs to display the triage result, update RightColumn:

```javascript
<RightColumn 
  triageResult={triageResult}
  vitalsData={vitalsData}
/>
```

And in RightColumn.jsx, add:

```javascript
export default function RightColumn({ triageResult, vitalsData, ...props }) {
  return (
    <div className="right-column">
      {/* Existing content */}
      
      {triageResult && (
        <div className="triage-result-summary" style={{
          marginTop: "20px",
          padding: "15px",
          backgroundColor: "#f0f9ff",
          borderLeft: "4px solid #2563eb",
          borderRadius: "4px"
        }}>
          <h4>Manual Triage Result</h4>
          <p><strong>ESI Level:</strong> {triageResult.manualESILevel}</p>
          <p><strong>Acuity:</strong> {triageResult.manualAcuity}</p>
          <p><strong>Department:</strong> {triageResult.recommendedDepartment}</p>
        </div>
      )}
    </div>
  );
}
```

## Complete Integration Example

Here's a complete example of the updated PS.jsx structure:

```javascript
import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import TriageTabContainer from "./Components/TriageTabContainer";
import LeftColumn from "./Components/LeftColumn";
import MiddleColumn from "./Components/MiddleColumn";
import RightColumn from "./Components/RightColumn";
import Vitals from "./Components/Vitals";
import "./PS.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function PS() {
  const { patientId } = useParams();
  
  // Patient data state
  const [patientData, setPatientData] = useState(() => {
    const saved = localStorage.getItem("patientData");
    return saved ? JSON.parse(saved) : {};
  });

  // Vitals state
  const [vitalsData, setVitalsData] = useState(() => {
    const saved = localStorage.getItem("vitalsData");
    return saved ? JSON.parse(saved) : {};
  });

  // Transcript state
  const [transcript, setTranscript] = useState(() => {
    const saved = localStorage.getItem("transcript");
    return saved ? JSON.parse(saved) : [];
  });

  // NEW: Triage result state
  const [triageResult, setTriageResult] = useState(() => {
    const saved = localStorage.getItem("triageResult");
    return saved ? JSON.parse(saved) : null;
  });

  // Persist patient data to localStorage
  useEffect(() => {
    localStorage.setItem("patientData", JSON.stringify(patientData));
  }, [patientData]);

  // Persist vitals to localStorage
  useEffect(() => {
    localStorage.setItem("vitalsData", JSON.stringify(vitalsData));
  }, [vitalsData]);

  // Persist transcript to localStorage
  useEffect(() => {
    localStorage.setItem("transcript", JSON.stringify(transcript));
  }, [transcript]);

  // Handle logout
  const handleLogOut = () => {
    localStorage.clear();
    window.location.href = "/login";
  };

  // NEW: Handle triage submission
  const handleTriageSubmit = (triageData) => {
    console.log("Manual Triage Submitted:", triageData);
    setTriageResult(triageData);
    localStorage.setItem("triageResult", JSON.stringify(triageData));
    
    // Optional: Show success notification
    // You can integrate with a toast notification library here
  };

  return (
    <div className="patient-summary-container">
      {/* NEW: Replace old Header with TriageTabContainer */}
      <TriageTabContainer
        patientData={patientData}
        vitalsData={vitalsData}
        onTriageSubmit={handleTriageSubmit}
      />

      {/* Main content area */}
      <div className="content-wrapper">
        {/* Vitals section */}
        <Vitals vitalsData={vitalsData} setVitalsData={setVitalsData} />

        {/* Three-column layout */}
        <div className="three-column-layout">
          <LeftColumn 
            data={patientData} 
            setData={setPatientData} 
          />
          <MiddleColumn 
            data={transcript}
            setData={setTranscript}
          />
          <RightColumn 
            triageResult={triageResult}
            vitalsData={vitalsData}
          />
        </div>
      </div>
    </div>
  );
}
```

## CSS Integration

Ensure your PS.css includes these layout styles:

```css
.patient-summary-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f8fafc;
}

.content-wrapper {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.three-column-layout {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  gap: 20px;
  flex: 1;
}

@media (max-width: 1024px) {
  .three-column-layout {
    grid-template-columns: 1fr;
  }
}
```

## LocalStorage Keys Used

The integrated system uses these localStorage keys:

```
- "currentNurseId"      // Nurse logged in
- "patientData"         // Patient info
- "vitalsData"          // Current vital signs
- "transcript"          // Q&A transcript
- "triageResult"        // Manual triage result
- "token"               // JWT auth token
```

## Testing the Integration

### Test 1: Component Renders
1. Navigate to patient dashboard
2. Verify TriageTabContainer appears at top
3. Verify tabs show "Manual Triage" and "AI-Assisted Triage"
4. Verify existing columns still display below

### Test 2: Tab Switching
1. Click "Manual Triage" tab
2. Verify ManualTriageForm displays
3. Click "AI-Assisted Triage" tab
4. Verify placeholder shows
5. Click "Manual Triage" again
6. Verify form content persists

### Test 3: Form Submission
1. Fill out Manual Triage form completely
2. Click Submit
3. Verify: Form clears
4. Verify: Triage result displays in RightColumn (if implemented)
5. Check localStorage: "triageResult" should contain data

### Test 4: Data Persistence
1. Submit a triage
2. Refresh page
3. Verify: Triage result still displayed
4. Verify: localStorage persisted data

## Debugging

### If components don't display:
1. Check browser console for import errors
2. Verify file paths are correct
3. Ensure CSS files are imported
4. Check that useParams/useLocation work correctly

### If styling looks broken:
1. Check CSS file imports in components
2. Look for CSS class conflicts
3. Check z-index for tab overlays
4. Verify responsive breakpoints

### If forms won't submit:
1. Check browser console for errors
2. Verify BACKEND_URL is set
3. Check network tab for API calls
4. Verify localStorage has currentNurseId set

## Migration Path

### Phase 1: Basic Integration (Now)
- [ ] Import TriageTabContainer
- [ ] Replace Header with TriageTabContainer
- [ ] Add handleTriageSubmit handler
- [ ] Test tab switching
- [ ] Test form validation

### Phase 2: Backend Integration
- [ ] Verify /api/triage endpoint exists
- [ ] Verify /api/triage/staff endpoint exists
- [ ] Test form submission to backend
- [ ] Test staff loading from backend
- [ ] Verify manual triage appears in EHR

### Phase 3: Dashboard Integration
- [ ] Update Nurse Dashboard to show manual triages
- [ ] Update EHR to display manual triage report
- [ ] Implement specialist feedback workflow
- [ ] Implement priority queue sorting

### Phase 4: AI-Assisted Implementation
- [ ] Create AIAssistedTriageForm component
- [ ] Implement computeMode handling
- [ ] Display both manual + AI reports
- [ ] Allow nurse to accept AI or confirm manual

## Expected Behavior After Integration

1. **User Logs In**
   - Navigated to Patient Dashboard
   - Sees patient info and vitals

2. **User Clicks Manual Triage Tab**
   - ManualTriageForm displays
   - Staff schedule section collapsed

3. **User Fills Form**
   - Selects ESI level
   - Enters acuity score
   - Enters clinical reasoning
   - Selects department
   - Enters interventions

4. **User Selects Staff**
   - Expands staff section
   - Sees available nurses, physicians, beds
   - Clicks to select (cards highlight in blue)

5. **User Submits**
   - Form validates (all fields required)
   - POSTs to /api/triage (no computeMode)
   - Shows success message
   - Form clears
   - Triage result appears (if RightColumn updated)
   - Manual triage appears in EHR

## Notes

- The TriageTabContainer manages all state switching
- ManualTriageForm handles form state and submission
- StaffScheduleDisplay handles staff selection UI
- Each component is self-contained and testable
- Mock data works if backend is unavailable
- All state persists to localStorage for resilience
