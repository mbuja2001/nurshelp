# Analytics-First Implementation Roadmap: Triage Perfection → Real-Time Metrics

## Overview
Priority sequence to solve your organization problem and unlock analytics tracking:

1. **Sprint 1 (Week 1-2)**: Perfect Object 2 (Manual Triage) + Analytics hooks
2. **Sprint 2 (Week 3-4)**: Integrate Object 1 (AI Triage) + Start tracking discrepancies
3. **Sprint 3 (Week 5-6)**: Deploy Object 3 (Clinical Encounter) + Full billing & waste visibility

By end of Sprint 1, you'll have **real-time triage quality metrics**.
By end of Sprint 2, you'll have **triage vs AI accuracy tracking**.
By end of Sprint 3, you'll have **complete waste, response, and billing analytics**.

---

## SPRINT 1: TRIAGE PERFECTION + ANALYTICS FOUNDATION (Week 1-2)

### Goal
Perfect the Manual Triage workflow (Object 2), add analytics hooks, and start measuring triage efficiency.

### 1.1 Audit Current Triage Process

First, identify what's working and what's breaking:

```javascript
// Questions to answer:
1. How long does triage take per patient? (Currently tracking?)
2. What's the SATS input variability? (Are nurses filling all fields?)
3. Which patients get retriaged within 10 minutes? (Mistriage rate?)
4. Average time from triage → bed assignment?
5. How many patients wait > 30 min for triage?
```

### 1.2 Object 2 (Nurse Triage Report) - Perfect the Data Capture

**Database Enhancement**:

```javascript
// Location: src/models/nurseTriage.model.js (Enhance existing)

{
  _id: ObjectId,
  encounterId: ObjectId,
  patientId: ObjectId,
  
  // Timing (analytics hooks)
  triageStartTime: Date,        // ← NEW: When nurse opens triage sheet
  vitalsRecordedTime: Date,     // ← NEW: When vitals filled
  assessmentTime: Date,         // ← NEW: When assessment written
  triageCompleteTime: Date,     // ← NEW: When "Assign" button clicked
  triageDuration: Number,       // ms (calculated: completeTime - startTime)
  
  // SATS Score calculation (keep existing)
  vitals: {
    systolicBP: Number,
    diastolicBP: Number,
    heartRate: Number,
    temperature: Number,
    spo2: Number,
    respiratoryRate: Number,
  },
  
  satsScore: {
    calculated: Number,         // 1-3 (most abnormal parameter)
    components: {
      respiratoryRate: { value: Number, score: Number },
      spo2: { value: Number, score: Number },
      systemicSystolic: { value: Number, score: Number },
      temperature: { value: Number, score: Number },
      alertness: { value: String, score: Number },
    },
  },
  
  // Chief Complaint & presentation
  chiefComplaint: String,
  duration: String,             // "2 hours", "1 day"
  
  // Initial Assessment
  triageLevel: String,          // "GREEN" | "YELLOW" | "ORANGE" | "RED"
  triageReasoning: String,      // WHY the nurse chose this level
  
  // Flags (analytics)
  dataCompleteness: Number,     // % of fields filled (0-100)
  vitalsValidity: Boolean,      // Are numbers within expected range?
  highConfidence: Boolean,      // Did nurse feel confident in triage?
  
  // Training flag
  triageNurseExperience: "JUNIOR" | "MID" | "SENIOR", // From staff profile
  
  // Status
  status: "COMPLETED",
  triageNurseId: ObjectId,
  
  // Audit trail
  createdAt: Date,
  updatedAt: Date,
}
```

**Triage Form Enhancements**:

```javascript
// Location: src/Dashboard/Components/TriageForm.jsx

Improvements needed:
1. Real-time SATS score calculation (show scoring breakdown)
2. Auto-validation on vital thresholds
3. Confidence slider (1-5: How sure are you?)
4. Visual cues for abnormal readings
5. Progress bar showing % complete
6. Mandatory field highlighting
7. Time-on-screen counter (show how long triage is taking)
8. AI suggestion preview (bottom-right corner, non-intrusive)
```

### 1.3 Analytics Hooks: Triage Quality Dashboards

Immediately start tracking these metrics:

```javascript
// New Collection: TriageQualityMetrics

{
  _id: ObjectId,
  hospitalId: ObjectId,
  wardId: ObjectId,
  
  // Timestamp
  recordedAt: Date,
  
  // Real-time metrics
  metrics: {
    // Triage Efficiency
    avg_triage_duration: Number,    // minutes
    median_triage_duration: Number,
    p95_triage_duration: Number,    // 95th percentile
    
    // Data Quality
    avg_data_completeness: Number,  // 0-100%
    vital_invalidity_rate: Number,  // % of invalid readings
    
    // Triage Confidence
    avg_nurse_confidence: Number,   // 1-5 scale
    
    // Mistriage Detection (Preliminary)
    retriages_within_10min: Number, // Patients reassessed quickly
    retriage_rate: Number,          // % of patients
    
    // Workflow
    avg_triage_to_assignment: Number,  // Time from triage → bed
    patients_waiting_for_triage: Number,
    
    // By Experience Level
    by_experience: {
      JUNIOR: { avg_duration: Number, confidence: Number, error_rate: Number },
      MID: { ... },
      SENIOR: { ... },
    },
  },
}
```

### 1.4 Real-Time Triage Analytics Dashboard

**Create New Component**:

```javascript
// Location: src/Dashboard/Pages/TriageQualityDashboard.jsx

Display real-time metrics:
┌─ Triage Quality Dashboard ─────────────────┐
│                                             │
│ TODAY'S TRIAGE METRICS:                    │
│                                             │
│ Avg Triage Time: 8.5 min                  │
│ Patients Triaged: 47                      │
│ Data Completeness: 94%                    │
│ Avg Confidence: 4.2/5 ⭐                  │
│                                             │
│ 🚨 ALERTS:                                │
│ • 3 patients retriaged within 10 min       │
│ • 1 value threshold violation              │
│ • 2 junior nurses below confidence avg     │
│                                             │
│ By Experience:                            │
│ Senior:  7.2 min | 4.8/5 confidence       │
│ Mid:     8.8 min | 4.1/5 confidence       │
│ Junior:  11.3 min | 3.6/5 confidence      │
│                                             │
└─────────────────────────────────────────────┘
```

### 1.5 API Endpoints for Sprint 1

```javascript
// Object 2 (Nurse Triage) Endpoints

POST /api/triage/start
  Body: { encounterId, patientId, wardId, triageNurseId }
  Response: { triageSessionId, startTime }
  Action: Record triageStartTime

POST /api/triage/:triageId/vitals
  Body: { vitals: {...}, recordedAt }
  Response: { satsScore, validity }
  Action: Record vitalsRecordedTime, validate ranges

POST /api/triage/:triageId/complete
  Body: { 
    triageLevel, 
    triageReasoning, 
    confidence,
    assignedBedId,
  }
  Response: { 
    triageCompleted: true, 
    triageDuration, 
    dataCompleteness,
  }
  Action: Calculate metrics, create audit entry

GET /api/analytics/triage/quality
  PARAMS: timeRange (24h, 7d, 30d), wardId (optional)
  Response: TriageQualityMetrics for dashboard
```

### 1.6 Success Criteria for Sprint 1

```
✅ Object 2 (Manual Triage) perfectly captures:
   - All vital signs with timestamps
   - Triage level + reasoning
   - Nurse confidence score
   - Time metrics (start/complete)
   
✅ Analytics tracking enabled:
   - Triage duration per patient
   - Data quality % per triage
   - Nurse confidence trends
   
✅ Dashboard shows:
   - Real-time triage queue depth
   - Average triage time
   - Data completeness %
   - Confidence by experience level
   
✅ No more "mystery" triages:
   - Every row in the triage dashboard has timestamps
   - Can trace: Check-in → Triage Start → Triage Complete → Assignment
```

---

## SPRINT 2: AI TRIAGE INTEGRATION + DISCREPANCY TRACKING (Week 3-4)

### Goal
Integrate AI Suggested Triage (Object 1) alongside Manual Triage, measure accuracy, detect mistriages.

### 2.1 Object 1 (AI Suggested Report) - Standardize the Output

**What you've already tested**:
- RAG engine (pulls clinical knowledge)
- LLM-only model
- Stress tested both

**Now standardize the format**:

```javascript
// Location: src/models/aiSuggestedReport.model.js

{
  _id: ObjectId,
  encounterId: ObjectId,
  patientId: ObjectId,
  
  // When was this generated?
  generatedAt: Date,
  generatedBy: String,          // "RAG" | "LLM" | "HYBRID"
  processingTime: Number,       // ms (how long to generate?)
  
  // AI's diagnostic suggestion
  suggestedTriageLevel: String, // "GREEN" | "YELLOW" | "ORANGE" | "RED"
  suggestedDiagnosis: String,   // e.g., "Acute coronary syndrome"
  confidence: Number,           // 0-100%
  
  // Supporting reasoning (for transparency)
  keyFindings: [String],        // List of interesting clinical facts
  riskFactors: [String],        // What made it serious?
  recommendedTests: [String],   // What to order? (links to Object 3 later)
  
  // Alternative diagnoses (differential)
  differentials: [
    {
      diagnosis: String,
      probability: Number,      // 0-100%
    }
  ],
  
  // Model performance tags
  modelVersion: String,         // "RAG-v2.1" | "LLM-gpt-4"
  modelAccuracy: Number,        // Historical accuracy of this model
  
  // Comparison to manual triage (filled later)
  manualTriageLevel: String,    // Filled after Object 2 complete
  comparisonResult: String,     // "MATCH" | "MISMATCH" | "UPGRADE" | "DOWNGRADE"
  discrepancyFlag: Boolean,
  
  status: "GENERATED",
  
  createdAt: Date,
}
```

### 2.2 Comparison & Discrepancy Detection

**When Object 2 is completed, immediately compare**:

```javascript
// Location: src/services/TriageComparisonService.js

class TriageComparisonService {
  
  compareTriages(aiSuggestion, manualTriage) {
    const comparison = {
      ai_level: aiSuggestion.suggestedTriageLevel,
      manual_level: manualTriage.triageLevel,
      match: aiSuggestion.suggestedTriageLevel === manualTriage.triageLevel,
    };
    
    if (!comparison.match) {
      // Determine direction of discrepancy
      const levelMap = { 'GREEN': 1, 'YELLOW': 2, 'ORANGE': 3, 'RED': 4 };
      const aiScore = levelMap[comparison.ai_level];
      const manualScore = levelMap[comparison.manual_level];
      
      if (manualScore > aiScore) {
        comparison.type = "UPGRADE";        // Nurse triaged more serious
        comparison.severity = "POSITIVE";   // Nurse is being cautious
      } else {
        comparison.type = "DOWNGRADE";      // Nurse triaged as less serious
        comparison.severity = "ALERT";      // Potential mistriage!
      }
    } else {
      comparison.type = "MATCH";
      comparison.severity = "GOOD";
    }
    
    comparison.score = this.calculateDiscrepancyScore(aiSuggestion, manualTriage);
    
    return comparison;
  }
  
  calculateDiscrepancyScore(ai, manual) {
    // 0 = Perfect match
    // 100 = Complete opposite
    
    const levelDistance = Math.abs(
      this.triageLevelToNum(ai.suggestedTriageLevel) - 
      this.triageLevelToNum(manual.triageLevel)
    );
    
    const confidencePenalty = Math.abs(
      ai.confidence - manual.confidence
    );
    
    const score = (levelDistance * 25) + (confidencePenalty * 0.25);
    return Math.min(100, score);
  }
}
```

### 2.3 Discrepancy Analytics Collection

```javascript
// New Collection: TriageDiscrepancyLog

{
  _id: ObjectId,
  encounterId: ObjectId,
  patientId: ObjectId,
  
  // Comparison
  aiTriageLevel: String,
  manualTriageLevel: String,
  comparisonType: "MATCH" | "UPGRADE" | "DOWNGRADE",
  discrepancyScore: Number,     // 0-100
  
  // Impact Tracking
  patientOutcome: String,       // Filled later by physician
  wasAIMistaken: Boolean,       // Filled when physician reviews
  wasNurseMistaken: Boolean,    // Filled when physician reviews
  
  // Timing
  triageTime: Date,
  comparisonTime: Date,
  
  // Attribution
  triageNurseId: ObjectId,
  aiModel: String,              // "RAG-v2.1" | "LLM-gpt-4"
  
  // Analytics
  flagForReview: Boolean,       // High discrepancy score?
  trainingValue: Boolean,       // Good case study?
}
```

### 2.4 Real-Time Discrepancy Dashboard

```javascript
// Location: src/Dashboard/Pages/TriageAccuracyDashboard.jsx

Display:
┌─ Triage Accuracy Report ──────────────────┐
│                                           │
│ TODAY'S COMPARISON (47 triages):          │
│                                           │
│ ✅ Perfect Matches: 38/47 (81%)          │
│ ⚠️  Upgrades (Nurse cautious): 7/47 (15%)│
│ 🚨 Downgrades (Potential risk): 2/47 (4%)│
│                                           │
│ AI Model Performance:                    │
│ RAG Model: 84% accuracy                  │
│ LLM Model: 79% accuracy                  │
│                                           │
│ Nurse Performance (by experience):       │
│ Senior:  92% match with AI               │
│ Mid:     85% match with AI               │
│ Junior:  73% match with AI               │
│                                           │
│ 🔍 Cases Flagged for Review: 3           │
│ • Patient 4521: AI=RED, Nurse=ORANGE    │
│ • Patient 4589: AI=YELLOW, Nurse=GREEN  │
│ • Patient 4612: Confidence gap 40%      │
│                                           │
└─────────────────────────────────────────┘
```

### 2.5 APIs for Sprint 2

```javascript
// Comparison Endpoints

POST /api/triage/:triageId/compare-ai
  Body: { aiSuggestionId }
  Response: { 
    comparison_type: "MATCH" | "UPGRADE" | "DOWNGRADE",
    discrepancy_score: Number,
    flagged: Boolean,
  }
  Action: Create TriageDiscrepancyLog entry

GET /api/analytics/triage/accuracy
  PARAMS: timeRange, wardId, modelVersion (optional)
  Response: {
    match_rate: Number,
    upgrade_rate: Number,
    downgrade_rate: Number,
    by_experience: {},
    by_model: {},
    flagged_cases: [],
  }

GET /api/analytics/triage/discrepancies
  PARAMS: timeRange, severity (HIGH | LOW)
  Response: List of flagged triages with patient outcomes

GET /api/analytics/ai-model/performance
  PARAMS: modelVersion
  Response: {
    accuracy: Number,
    false_positive_rate: Number,
    false_negative_rate: Number,
    best_for: String,     // Which acuity level?
    worst_for: String,
  }
```

### 2.6 Success Criteria for Sprint 2

```
✅ Every triage now has AI comparison:
   - Object 2 (Manual) vs Object 1 (AI) tracked
   - Discrepancy score calculated
   - Flagged cases identified
   
✅ Can answer:
   - "Is the AI better than junior nurses?" YES/NO (with data)
   - "Which model (RAG vs LLM) performs better?" [ANSWER]
   - "What % of nurses' triages match AI?" [ANSWER]
   - "Which patients need physician review?" [LIST]
   
✅ Training data captured:
   - Identified high-discrepancy cases for human review
   - Can correlate with physician feedback later (Object 4)
   
✅ Dashboard shows:
   - Real-time accuracy metrics
   - Model performance comparison
   - Nurse experience vs AI performance
```

---

## SPRINT 3: CLINICAL ENCOUNTER + BILLING & WASTE ANALYTICS (Week 5-6)

### Goal
Deploy Object 3 (Clinical Encounter Record), link to Objects 1-2, and unlock billing + waste tracking.

### 3.1 Implement Clinical Encounter Record (Object 3)

As specified in CLINICAL_CARE_MODE_PLAN.md, but now integrated with triage analytics.

```javascript
// Location: src/models/clinicalEncounterRecord.model.js

Key additions for billing & waste:
{
  // Links to Objects 1 & 2
  aiSuggestionId: ObjectId,        // Object 1
  nurseTriage: {
    triageId: ObjectId,            // Object 2
    triageAccuracy: String,        // "MATCH" | "UPGRADE" | "DOWNGRADE"
  },
  
  // Procedures & consumables (billing data)
  procedureLog: [
    {
      procedureName: String,
      procedureType: String,
      cost: Number,                // ZAR
      necessityScore: Number,      // Did AI suggest this? (0-100)
    }
  ],
  
  consumablesUsed: [
    {
      consumableName: String,
      quantity: Number,
      unitCost: Number,
      totalCost: Number,
      aiRecommended: Boolean,      // Was this suggested by AI?
    }
  ],
  
  // Waste tracking
  waste: {
    unnecessary_procedures: [String],
    excessive_consumables: Boolean,
    resource_efficiency_score: Number,  // Did we use resources wisely?
  },
  
  // Billing attachment
  billingBundleId: ObjectId,       // Link to charge capture
}
```

### 3.2 Response Latency Tracking

```javascript
// From escalation requests in Object 3

escalationRequests: [
  {
    requestedAt: Date,             // Nurse clicked "Request Physician"
    acknowledgedAt: Date,          // Physician acknowledged
    responseLatency: Number,       // minutes
    resolvedAt: Date,
    resolutionLatency: Number,     // minutes to actual treatment change
  }
]

// Analytics: Response Latency
GET /api/analytics/escalations/response-time
  Response: {
    avg_response_latency: 3.5,           // minutes (target: < 5)
    p95_response_latency: 12,            // 95th percentile
    by_priority: {
      RED: { avg: 1.2, target: 1 },
      ORANGE: { avg: 4.3, target: 5 },
      YELLOW: { avg: 8.1, target: 15 },
    },
    physicians: [
      { name: "Dr. Sarah", avg_response: 2.1 },
      { name: "Dr. James", avg_response: 5.8 },
    ],
  }
```

### 3.3 Comprehensive Billing Dashboard

```javascript
// Location: src/Dashboard/Pages/BillingAnalytics.jsx

Display:
┌─ Per-Patient Billing Report ──────────────┐
│                                           │
│ Patient: John Doe (ID: 4521)              │
│ Encounter: 17 Apr 2026 - 14:35-16:10     │
│ Duration: 95 minutes                      │
│                                           │
│ BILLING BREAKDOWN:                        │
│ ──────────────────────────────────────    │
│ Clinical Fee (ED Level): ZAR 2,500       │
│                                           │
│ Procedures:                              │
│ • IV Access Started: ZAR 450             │
│ • Blood Draw: ZAR 200                    │
│ • ECG: ZAR 300                           │
│ Procedures Subtotal: ZAR 950             │
│                                           │
│ Consumables:                             │
│ • IV Starter Kit (2): ZAR 900            │
│ • Dressing (3): ZAR 195                  │
│ • Oxygen (2.5 hrs): ZAR 212              │
│ Consumables Subtotal: ZAR 1,307          │
│                                           │
│ Staff Allocation:                        │
│ • Nurse (95 min): ZAR 800                │
│ • Physician (15 min): ZAR 300            │
│ Staff Subtotal: ZAR 1,100                │
│                                           │
│ ────────────────────────────────────────  │
│ TOTAL ENCOUNTER COST: ZAR 5,857          │
│                                           │
│ 📊 ANALYSIS:                             │
│ AI Suggested Cost: ZAR 4,200             │
│ Actual Cost: ZAR 5,857                   │
│ Variance: +40% (Over budget)             │
│ Reason: Extra IV kit usage               │
│                                           │
│ 🏥 FOR MEDICAL AID BILLING:              │
│ To: Discovery Health                     │
│ Claim Code: 99283 (ED - Minor)           │
│ Claimed Amount: ZAR 5,200 (after negotiation)
│                                           │
└─────────────────────────────────────────┘
```

### 3.4 Waste Detection Dashboard

```javascript
// Location: src/Dashboard/Pages/ResourceWasteAnalytics.jsx

Display:
┌─ Resource Waste Analysis ─────────────────┐
│ (Comparing Object 1 AI + Object 2 Triage  │
│  vs Object 3 Actual Resources)            │
│                                           │
│ TODAY'S WASTE SUMMARY:                   │
│ ────────────────────────────────────────  │
│                                           │
│ 🚨 HIGH WASTE ALERTS:                   │
│ • Patient 4521: 5 IV kits for GREEN pt   │
│   Expected: 0-1 | Actual: 5 | Cost: +ZAR 1,800 |
│   Flag: EXCESSIVE                        │
│                                           │
│ • Patient 4589: 6 dressings for YELLOW pt
│   Expected: 2-3 | Actual: 6 | Cost: +ZAR 195  │
│   Flag: EXCESSIVE                        │
│                                           │
│ ⚠️  MODERATE WASTE:                      │
│ • Average oxygen cost (per patient): ZAR 150
│ • Highest oxygen cost: Patient 4542 = ZAR 340 │
│   (2.5x average - investigate duration)  │
│                                           │
│ 📊 DAILY TOTALS:                         │
│ Predicted Waste (vs baseline): ZAR 3,200 │
│ Patients flagged for investigation: 4    │
│ Estimated recovery potential: ZAR 8,900  │
│ (If these 4 procedures optimized)        │
│                                           │
│ 🎯 BY STAFF MEMBER:                      │
│ Nurse "Sarah": Waste per patient ZAR 150 │
│ Nurse "Emily": Waste per patient ZAR 45  │
│ → Sarah's team uses 3x more consumables  │
│    (Training opportunity?)               │
│                                           │
└─────────────────────────────────────────┘
```

### 3.5 Integrated Response & Treatment Latency

```javascript
// New Analytics: Full encounter timeline

{
  patientId: ObjectId,
  encounterTimeline: {
    checkin: ISO timestamp,
    triage_start: ISO timestamp,
    triage_complete: ISO timestamp,
    triage_duration: minutes,
    
    bed_assignment: ISO timestamp,
    time_to_bed: minutes,           // checkin → bed
    
    clinical_care_start: ISO timestamp,
    clinical_care_duration: minutes,
    
    first_escalation: ISO timestamp,
    escalation_response_time: minutes,
    
    treatment_complete: ISO timestamp,
    total_treatment_duration: minutes,  // triage_start → complete
    
    physician_review: ISO timestamp,
    total_encounter_duration: minutes,  // checkin → review
  },
  
  analytics: {
    triage_efficiency: "GOOD" | "AVERAGE" | "SLOW",
    response_efficiency: "GOOD" | "AVERAGE" | "SLOW",
    treatment_efficiency: "GOOD" | "AVERAGE" | "SLOW",
  },
}
```

### 3.6 APIs for Sprint 3

```javascript
// Billing & Waste Endpoints

GET /api/analytics/billing/per-patient
  PARAMS: encounterId
  Response: Complete billing breakdown + waste analysis

GET /api/analytics/waste/by-staff
  PARAMS: timeRange, staffId (optional)
  Response: {
    total_waste_cost: Number,
    waste_per_patient: Number,
    excessive_items: [{ item, quantity, excess, cost }],
    training_recommendation: String,
  }

GET /api/analytics/waste/by-procedure
  PARAMS: timeRange
  Response: {
    highest_waste_procedures: [...],
    most_wasteful_nursing_team: {...},
    conservation_opportunities: [...],
  }

GET /api/analytics/encounters/timeline
  PARAMS: encounterId
  Response: Complete timeline with all key milestones + latencies

GET /api/analytics/response-efficiency
  PARAMS: timeRange, priorityLevel (optional)
  Response: Response latency metrics + by physician ranking
```

### 3.7 Success Criteria for Sprint 3

```
✅ Complete 5-Object Architecture:
   Object 1: AI Suggestion ✓
   Object 2: Nurse Triage ✓
   Object 3: Clinical Encounter ✓
   Object 4: Physician Feedback (ready to integrate)
   Object 5: Audit Bundle (ready to integrate)

✅ Analytics fully operational:
   - Triage quality (duration, confidence, data completeness)
   - Triage accuracy (manual vs AI discrepancies)
   - Response latency (physician acknowledgment times)
   - Resource waste (consumables vs predicted)
   - Billing per patient (with medical aid codes)
   - Staff efficiency (by nurse, by team)

✅ Can answer critical questions:
   - "What's our daily waste in ZAR?" → Dashboard shows
   - "Is Nurse A wasting resources?" → Comparison chart
   - "What was the response time for RED patients?" → Report
   - "How accurate is the AI?" → Trending chart
   - "How much to bill Discovery Health?" → Per encounter
   - "Are junior nurses triaging incorrectly?" → Performance matrix

✅ Stakeholders can see ROI:
   - Finance: Cost per patient, medical aid revenue
   - Operations: Waste alerts, response latency tracking
   - Quality: Triage accuracy, AI performance
   - Admin: Staff efficiency metrics
```

---

## COMBINED VIEW: What Gets Unlocked When

### End of Sprint 1
```
METRICS AVAILABLE:
✓ Triage duration per patient
✓ Triage data completeness %
✓ Nurse confidence scores
✓ SATS validity checks
✓ Triage queue depth
✓ Nurse performance (by experience level)

DASHBOARDS:
✓ Real-time Triage Quality Dashboard
✓ Triage duration trends
```

### End of Sprint 2
```
METRICS AVAILABLE (SPRINT 1 + NEW):
✓ AI vs Manual triage comparison
✓ Mistriage detection rate
✓ Model performance (RAG vs LLM)
✓ Experience level impact on accuracy
✓ Discrepancy flagging
✓ Physician review candidates identified

DASHBOARDS:
✓ Triage Accuracy Report
✓ AI Model Performance Comparison
✓ Quality assurance tracking
```

### End of Sprint 3
```
METRICS AVAILABLE (SPRINT 1 + 2 + NEW):
✓ Per-patient billing breakdown
✓ Resource waste identification
✓ Response latency tracking (escalation)
✓ Treatment duration metrics
✓ Staff efficiency scoring
✓ Medical aid billing codes applied
✓ Waste-by-procedure analysis
✓ Complete encounter timeline

DASHBOARDS:
✓ Per-Patient Billing Report
✓ Resource Waste Analytics
✓ Response Efficiency Report
✓ Treatment Duration Analysis
✓ Staff Performance (by waste/efficiency)
✓ Medical Aid Revenue Tracking
```

---

## IMPLEMENTATION PRIORITIES

### Must-Have First (Sprint 1)
- [ ] Object 2 enhancements (timestamps, confidence, reasoning)
- [ ] SATS calculation with validity checks
- [ ] Real-time Triage Quality Dashboard
- [ ] Analytics foundation APIs

### Should-Have Second (Sprint 2)
- [ ] Object 1 standardization (discrepancy score)
- [ ] Triage comparison engine
- [ ] AI Model performance tracking
- [ ] Triage Accuracy Dashboard

### Nice-to-Have Third (Sprint 3)
- [ ] Full Object 3 with billing
- [ ] Waste detection engine
- [ ] Response latency tracking
- [ ] Comprehensive billing dashboards

---

## Database Schema Checklist

```
Sprint 1 Create:
□ Enhance nurseTriage.model.js (add timestamps)
□ Create TriageQualityMetrics collection
□ Create audit log for triage events

Sprint 2 Create:
□ Create TriageDiscrepancyLog collection
□ Enhance aiSuggestedReport.model.js (standardize format)

Sprint 3 Create:
□ Create clinicalEncounterRecord.model.js (full version)
□ Create BillingRecord collection (for medical aid claims)
□ Create ResourceWasteLog collection
□ Create ResponseLatencyLog collection
```

---

## Summary: The Path to Perfect Analytics

**Week 1-2**: Have object 2 perfect with timestamped triage quality tracking
**Week 3-4**: Compare object 1 vs object 2, start identifying mistriages
**Week 5-6**: Deploy object 3, unlock billing, waste, response time analytics

By **end of week 6**, you'll have:
- 100% visibility into triage quality
- AI accuracy metrics vs manual triage
- Per-patient billing with waste detection
- Response latency tracking for escalations
- Staff efficiency rankings
- Medical aid revenue tracking

All the organization problems you've described will be visible and measurable.
