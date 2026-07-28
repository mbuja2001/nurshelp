# Clinical Handover & Escalation System - Complete Strategic Roadmap

**Status**: Ready for Phase 2 Implementation  
**Date**: 18 April 2026  
**Objective**: Transform from Staff Scheduling → Clinical Decision Support with Legal Compliance

---

## PART 1: THE 5-OBJECT FRAMEWORK (YOUR ARCHITECTURE)

### What You've Perfected ✅

```
STAFF MANAGEMENT PIPELINE (Complete & Verified)
├── Hospital Registration ✅
├── Staff Onboarding with Credentials ✅
├── Role-Based Dashboard Rendering ✅
├── Shift Scheduling & Assignment ✅
├── Clock In/Out with Flags ✅
│   ├── Late arrivals flagged
│   ├── Absences flagged
│   ├── Irregular shifts tracked
│   └── Break request approvals
└── Multi-Hospital Isolation ✅

DATABASE FOUNDATION (Exists but Not Fully Integrated)
├── Object 1: AI Suggested Report (AiSuggestedReport.model.js) ✅
├── Object 2: Nurse SATS Triage (TriageRequest.manualTriageReport) ✅
├── Object 3: Clinical Encounter (Encounter.model.js) ✅ [NEEDS ESCALATION TRACKING]
├── Object 4: Physician Feedback (TriageRequest.specialistFeedbackReport) ✅
└── Object 5: Audit Trail (audit_logger.py) ✅ [NEEDS INTEGRATION INTO MODELS]
```

---

## PART 2: WHAT YOU NEED TO BUILD (THE CLINICAL HANDOVER)

### Phase 2A: Escalation Alert Logic (Live P2P Trigger)

**Purpose**: Nurses can escalate deteriorating patients → Physician gets LIVE notification (no cloud dependency)

#### 1. EscalationService (Backend)
**File**: `backend/src/services/escalation.service.js`

```javascript
// Pseudo-code structure
class EscalationService {
  constructor(wsConnections) {
    this.wsConnections = wsConnections; // Map of userId → WebSocket
  }

  // Nurse clicks "Request Physician" button
  async requestPhysicianEscalation(encounterId, escalationData) {
    const payload = {
      type: "ESCALATION_ALERT",
      priority: escalationData.priority, // RED, ORANGE, YELLOW, GREEN
      location: escalationData.location, // Bed 4, ED Ward
      reason: escalationData.reason,     // Vitals deteriorating / Unresponsive
      timestamp: new Date(),
      requestedBy: escalationData.nurseId,
      encounterId: encounterId,
      patientName: escalationData.patientName,
    };

    // Save to DB for audit trail
    await EscalationAlert.create({
      encounterId,
      nurseId: escalationData.nurseId,
      physicianId: escalationData.physicianId,
      status: "PENDING",
      requestedAt: new Date(),
      payload,
    });

    // Broadcast to all physicians on duty
    const physicians = await Nurse.find({ role: "PHYSICIAN", isDuty: true });
    for (const physician of physicians) {
      const ws = this.wsConnections.get(physician._id.toString());
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
      }
    }
  }

  // Physician clicks "Acknowledge"
  async acknowledgeEscalation(alertId, physicianId) {
    const alert = await EscalationAlert.findByIdAndUpdate(
      alertId,
      {
        status: "ACKNOWLEDGED",
        acknowledgedAt: new Date(),
        acknowledgedBy: physicianId,
        responseLatency: Date.now() - alert.requestedAt.getTime(), // ms
      },
      { new: true }
    );

    return alert;
  }
}
```

#### 2. Frontend: "Request Physician" Button
**Location**: `frontend/src/Dashboard/Components/ClinicalCareMode.jsx`

```javascript
const ClinicalCareMode = ({ encounterId, bedNumber, patientName, currentVitals }) => {
  const [showEscalationForm, setShowEscalationForm] = useState(false);
  const [escalationReason, setEscalationReason] = useState("");

  const handleRequestPhysician = async () => {
    const escalationData = {
      priority: calculatePriority(currentVitals), // RED if O2 <90%, HR >130, etc
      location: `ED Ward - Bed ${bedNumber}`,
      reason: escalationReason,
      vitalsSnapshot: currentVitals,
    };

    // Send via API (which internally uses WebSocket)
    const res = await fetch(`${BACKEND_URL}/api/escalations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({
        encounterId,
        nurseId: userId,
        ...escalationData,
      }),
    });

    if (res.ok) {
      setShowEscalationForm(false);
      toast.success("Physician notified - awaiting acknowledgment");
    }
  };

  return (
    <div style={styles.clinicalCarePanel}>
      <button onClick={() => setShowEscalationForm(true)} style={styles.escalationButton}>
        🚨 REQUEST PHYSICIAN (Critical Escalation)
      </button>
      {showEscalationForm && (
        <EscalationForm
          onSubmit={handleRequestPhysician}
          vitals={currentVitals}
          location={`Bed ${bedNumber}`}
        />
      )}
    </div>
  );
};
```

#### 3. Frontend: Physician Alert Tray (High-Priority Overlay)
**Location**: `frontend/src/Dashboard/Components/PhysicianAlertTray.jsx`

```javascript
import SoundAlert from "./SoundAlert"; // plays beep.mp3

const PhysicianAlertTray = ({ userId, wsConnection }) => {
  const [alerts, setAlerts] = useState([]);
  const [playingSound, setPlayingSound] = useState(false);

  useEffect(() => {
    // Listen for escalation alerts from WebSocket
    wsConnection.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === "ESCALATION_ALERT") {
        // 🔔 NEW ALERT - OVERRIDE ALL VIEWS
        playAlarmSound();
        flashScreen(); // CSS animation

        setAlerts((prev) => [
          {
            ...message,
            id: generateId(),
            receivedAt: new Date(),
            acknowledged: false,
          },
          ...prev,
        ]);
      }
    };
  }, [wsConnection]);

  const handleAcknowledge = async (alertId) => {
    const res = await fetch(`${BACKEND_URL}/api/escalations/${alertId}/acknowledge`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
    });

    if (res.ok) {
      // Remove from tray and update DB with response time
      setAlerts((prev) => prev.filter((a) => a.id !== alertId));
    }
  };

  // RENDER: Persistent tray that overrides all other UI
  return (
    <>
      {/* Sound alert */}
      {playingSound && <SoundAlert />}

      {/* High-Priority Notification Tray */}
      {alerts.length > 0 && (
        <div style={styles.alertTrayOverlay}>
          <div style={styles.alertTrayHeader}>
            🚨 CRITICAL ALERTS ({alerts.length})
          </div>
          {alerts.map((alert) => (
            <div key={alert.id} style={styles.alertItem}>
              <div style={styles.alertPriority}>
                Priority: <strong style={{ color: getPriorityColor(alert.priority) }}>{alert.priority}</strong>
              </div>
              <div style={styles.alertLocation}>{alert.location}</div>
              <div style={styles.alertReason}>{alert.reason}</div>
              <button
                onClick={() => handleAcknowledge(alert.id)}
                style={styles.acknowledgeButton}
              >
                ✓ ACKNOWLEDGE
              </button>
            </div>
          ))}
        </div>
      )}
    </>
  );
};

const styles = {
  alertTrayOverlay: {
    position: "fixed",
    top: 0,
    right: 0,
    width: "400px",
    maxHeight: "100vh",
    backgroundColor: "#fff5f5",
    border: "3px solid #dc2626",
    boxShadow: "0 20px 25px rgba(220, 38, 38, 0.5)",
    zIndex: 9999, // Above all other UI
    overflowY: "auto",
    animation: "slideIn 0.3s ease-out",
  },
  alertTrayHeader: {
    fontWeight: 700,
    fontSize: 16,
    padding: "16px",
    backgroundColor: "#dc2626",
    color: "white",
    borderBottom: "1px solid #991b1b",
  },
  alertItem: {
    padding: "16px",
    borderBottom: "1px solid #fecaca",
  },
  acknowledgeButton: {
    width: "100%",
    padding: "10px",
    marginTop: "8px",
    backgroundColor: "#16a34a",
    color: "white",
    border: "none",
    borderRadius: "6px",
    fontWeight: 600,
    cursor: "pointer",
  },
};
```

---

### Phase 2B: Finalise & Sync (Legal Data Bundling)

**Purpose**: Package all 5 objects into HL7 FHIR-compliant EHR bundle for export

#### 1. EHRBundleFactory (Backend)
**File**: `backend/src/services/ehr.bundle.factory.js`

```javascript
class EHRBundleFactory {
  /**
   * Build complete EHR bundle from the 5 objects
   * This is the "legal" record sent to health systems
   */
  async createBundle(encounterId) {
    // Fetch all 5 objects
    const encounter = await Encounter.findById(encounterId)
      .populate("nurse_id", "firstName lastName licenseNumber")
      .populate("assignedPhysicianId", "firstName lastName specialization");

    const triageRequest = await TriageRequest.findOne({ encounterId });
    const aiReport = await AiSuggestedReport.findOne({ encounter: encounterId });
    const nurseConfirmed = await NurseConfirmedReport.findOne({ encounter: encounterId });
    const escalations = await EscalationAlert.find({ encounterId });

    // INTEGRITY CHECK: Verify all 5 objects are linked
    if (!triageRequest || !encounter) {
      throw new Error("Integrity check failed: Missing Triage or Encounter object");
    }

    // Build HL7 FHIR bundle
    const bundle = {
      // HEADER
      resourceType: "Bundle",
      type: "document",
      timestamp: new Date().toISOString(),
      entry: [
        // SEGMENT 1: PATIENT + TRIAGE (Objects 1 & 2)
        {
          resource: {
            resourceType: "Patient",
            id: encounter.patient.id_number,
            name: [
              {
                given: [encounter.patient.name],
                family: encounter.patient.surname,
              },
            ],
            contact: [
              {
                telecom: [{ system: "phone", value: encounter.patient.phone }],
              },
            ],
          },
        },
        {
          resource: {
            resourceType: "Observation",
            code: { coding: [{ system: "http://loinc.org", code: "73516-2", display: "Triage Priority" }] },
            // Object 2: Nurse SATS assessment
            valueCodeableConcept: {
              coding: [
                {
                  system: "http://example.com/sats",
                  code: triageRequest.manualTriageReport.satsLevel,
                  display: triageRequest.manualTriageReport.satsLevel.toUpperCase(),
                },
              ],
            },
            issued: triageRequest.manualTriageReport.createdAt,
          },
        },
        // Object 1: AI Suggested (if exists)
        ...(aiReport
          ? [
              {
                resource: {
                  resourceType: "Observation",
                  code: { coding: [{ system: "http://example.com/ai", code: "AI_TRIAGE" }] },
                  valueCodeableConcept: {
                    coding: [
                      {
                        code: aiReport.triage.satsLevel || "UNKNOWN",
                        display: `AI Confidence: ${Math.round(aiReport.aiMeta.confidence * 100)}%`,
                      },
                    ],
                  },
                  issued: aiReport.createdAt,
                },
              },
            ]
          : []),

        // SEGMENT 2: CLINICAL ENCOUNTER (Object 3)
        {
          resource: {
            resourceType: "Encounter",
            id: encounter._id,
            status: encounter.status,
            class: { code: "EMER", display: "Emergency" },
            type: [
              {
                coding: [{ code: encounter.patient.symptoms || "GENERAL", display: encounter.patient.symptoms }],
              },
            ],
            participant: [
              {
                individual: {
                  reference: `Practitioner/${encounter.nurse_id._id}`,
                  display: `${encounter.nurse_id.firstName} ${encounter.nurse_id.lastName} (Nurse)`,
                },
              },
              {
                individual: {
                  reference: `Practitioner/${encounter.assignedPhysicianId || "TBD"}`,
                  display: encounter.assignedPhysicianName || "Unassigned",
                },
              },
            ],
            // Vitals as part of encounter
            contained: [
              {
                resourceType: "Observation",
                id: "vitals",
                code: { coding: [{ code: "8480-6", display: "Systolic BP" }] },
                valueQuantity: { value: encounter.vitals.bp_systolic, unit: "mmHg" },
              },
            ],
            // Escalation tracking
            reasonReference: escalations.map((e) => ({
              reference: `Observation/${e._id}`,
              display: `${e.payload.reason} @ ${e.payload.location}`,
            })),
          },
        },

        // SEGMENT 3: PHYSICIAN FEEDBACK (Object 4)
        ...(triageRequest?.specialistFeedbackReport
          ? [
              {
                resource: {
                  resourceType: "Observation",
                  code: { coding: [{ code: "PHYSICIAN_REVIEW", display: "Specialist Feedback" }] },
                  valueCodeableConcept: {
                    coding: [
                      {
                        code: triageRequest.specialistFeedbackReport.actualESILevel,
                        display: triageRequest.specialistFeedbackReport.finalDiagnosis,
                      },
                    ],
                  },
                  issued: triageRequest.specialistFeedbackReport.createdAt,
                  performer: [
                    {
                      reference: `Practitioner/${triageRequest.specialistFeedbackReport.specialistId}`,
                    },
                  ],
                },
              },
            ]
          : []),

        // SEGMENT 4: AUDIT TRAIL (Object 5)
        {
          resource: {
            resourceType: "AuditEvent",
            event: {
              code: "TRIAGE_COMPLETE",
              action: "A", // Add
              outcome: "0", // Success
            },
            recorded: new Date().toISOString(),
            agent: [
              {
                altId: encounter.nurse_id._id,
                name: `${encounter.nurse_id.firstName} ${encounter.nurse_id.lastName}`,
              },
            ],
            entity: [
              {
                whatReference: { reference: `Encounter/${encounter._id}` },
                description: "Triage Completion",
              },
            ],
          },
        },
      ],
    };

    // DISCREPANCY ANALYSIS
    const discrepancies = this.analyzeDiscrepancies(
      aiReport,
      triageRequest.manualTriageReport,
      triageRequest.specialistFeedbackReport
    );

    bundle.meta = {
      lastUpdated: new Date().toISOString(),
      discrepancy_score: discrepancies.score, // 0-1: how different was AI from Nurse?
      discrepancies_found: discrepancies.items,
    };

    return bundle;
  }

  /**
   * Discrepancy Analysis: Did AI predict risk that humans missed?
   */
  analyzeDiscrepancies(aiReport, nurseReport, physicianReport) {
    const items = [];

    // Discrepancy 1: Did Physician override Nurse triage?
    if (physicianReport && aiReport) {
      if (physicianReport.actualESILevel !== nurseReport.satsLevel) {
        items.push({
          type: "NURSE_PHYSICIAN_MISMATCH",
          nurse_assessment: nurseReport.satsLevel,
          physician_assessment: physicianReport.actualESILevel,
          was_upgraded: physicianReport.actualESILevel > nurseReport.satsLevel,
        });
      }
    }

    // Discrepancy 2: Did AI predict something humans missed?
    if (aiReport) {
      const ai_high_confidence = aiReport.aiMeta.confidence > 0.8;
      const ai_different_call = aiReport.triage.satsLevel !== nurseReport.satsLevel;

      if (ai_high_confidence && ai_different_call && !physicianReport) {
        items.push({
          type: "AI_MISSED_BY_NURSE",
          ai_suggestion: aiReport.triage.satsLevel,
          nurse_assessment: nurseReport.satsLevel,
          ai_confidence: aiReport.aiMeta.confidence,
          message: "AI detected risk that nurse did not identify",
        });
      }
    }

    return {
      score: items.length / 10, // Simplified scoring
      items,
    };
  }
}
```

#### 2. Frontend: Finalise & Sync Button
**Location**: `frontend/src/Dashboard/Components/FinaliseEncounter.jsx`

```javascript
const FinaliseEncounter = ({ encounterId }) => {
  const [showIntegrityCheck, setShowIntegrityCheck] = useState(false);
  const [bundleStatus, setBundleStatus] = useState("ready"); // ready, bundling, synced, error

  const handleFinaliseClick = async () => {
    setBundleStatus("bundling");

    try {
      // 1. Send bundle data
      const res = await fetch(`${BACKEND_URL}/api/ehr/bundles`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          encounterId,
          action: "FINALISE_AND_SYNC",
        }),
      });

      if (!res.ok) throw new Error("Bundle creation failed");

      const bundle = await res.json();

      // 2. Show discrepancy analysis to physician
      if (bundle.meta.discrepancies_found.length > 0) {
        showDiscrepancyWarning(bundle.meta.discrepancies_found);
      }

      // 3. Sync to EHR (Healthbridge, Med-e-Mass)
      const syncRes = await fetch(`${BACKEND_URL}/api/ehr/sync`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          bundleId: bundle._id,
          targetEHR: "Med-e-Mass", // Or Healthbridge
        }),
      });

      if (syncRes.ok) {
        // 4. UPDATE BED STATUS
        await updateBedStatus(encounterId, "VACANT_CLEANING");

        // 5. CLEAR STAFF ACTIVE LIST
        await clearStaffActivePatient(encounterId);

        setBundleStatus("synced");
        toast.success("✓ Encounter finalized & synced to EHR");

        // Navigate back to main dashboard
        setTimeout(() => window.location.href = "/", 2000);
      }
    } catch (error) {
      setBundleStatus("error");
      console.error("Finalisation error:", error);
      toast.error(`Error: ${error.message}`);
    }
  };

  return (
    <button onClick={handleFinaliseClick} style={styles.finaliseButton} disabled={bundleStatus !== "ready"}>
      {bundleStatus === "ready" && "✍️ Sign & Finalise"}
      {bundleStatus === "bundling" && "⏳ Creating legal record..."}
      {bundleStatus === "synced" && "✓ Synced to EHR"}
      {bundleStatus === "error" && "❌ Error - Retry"}
    </button>
  );
};
```

---

## PART 3: THE TECHNICAL BLUEPRINT

### New Files to Create

```
backend/
├── src/
│   ├── services/
│   │   ├── escalation.service.js       ← WebSocket escalation logic
│   │   ├── ehr.bundle.factory.js       ← FHIR bundle packing
│   │   └── response.latency.logger.js  ← Metrics collection
│   ├── models/
│   │   ├── escalationAlert.model.js    [ENHANCE] Add response tracking
│   │   └── ehrBundle.model.js          ← New
│   ├── routes/
│   │   ├── escalation.route.js         ← New
│   │   └── ehr.route.js                ← New
│   ├── controllers/
│   │   ├── escalation.controller.js    ← New
│   │   └── ehr.controller.js           ← New
│   ├── middleware/
│   │   └── websocket.handler.js        ← New (handle WS connections)
│   └── config/
│       └── hl7.fhir.config.js          ← HL7 standards mapping

frontend/
├── src/
│   ├── Dashboard/
│   │   ├── Components/
│   │   │   ├── ClinicalCareMode.jsx    ← New (Nurse UI for escalation)
│   │   │   ├── PhysicianAlertTray.jsx  ← New (Physician override UI)
│   │   │   ├── FinaliseEncounter.jsx   ← New (Sign & sync UI)
│   │   │   └── DiscrepancyAlert.jsx    ← New (AI vs Nurse diff display)
│   │   ├── Services/
│   │   │   └── websocket.service.js    ← New (Handle WS connection)
│   │   └── Pages/
│   │       └── ClinicalHandover.jsx    ← New (Full workflow UI)
│   └── hooks/
│       └── useEscalationListener.js    ← New (Hook for WS alerts)
```

---

## PART 4: FEATURE MATRIX (WHAT YOU HAVE vs. WHAT YOU BUILD)

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| **Staff Management** |
| Hospital Registration | ✅ | manager.route.js | Working |
| Staff Onboarding | ✅ | manager.route.js | Working |
| Clock In/Out | ✅ | clock.service.js | Perfected |
| Break Approvals | ✅ | break.request.model.js | Perfected |
| Shift Assignment | ✅ | schedule.model.js | Perfected |
| **Clinical Triage (Phase 1)** |
| Manual SATS Triage | ✅ | triage.route.js | Working |
| AI Triage (LLM/RAG) | ✅ | aiSuggestedReport.model.js | Ready |
| Nurse Confirmation | ✅ | nurseConfirmedReport.model.js | Ready |
| Physician Feedback | ✅ | specialistFeedbackReport schema | Ready |
| **Clinical Handover (Phase 2 - NEW)** |
| Escalation Alert Service | ⏳ | escalation.service.js | TO BUILD |
| Physician Alert Tray | ⏳ | PhysicianAlertTray.jsx | TO BUILD |
| Response Latency Tracking | ⏳ | response.latency.logger.js | TO BUILD |
| EHR Bundle Factory | ⏳ | ehr.bundle.factory.js | TO BUILD |
| Finalise & Sync | ⏳ | FinaliseEncounter.jsx | TO BUILD |
| Discrepancy Analysis | ⏳ | ehr.bundle.factory.js | TO BUILD |
| Bed Auto-Toggle | ⏳ | beds.service.js | TO BUILD |
| Staff Active List Clear | ⏳ | schedule.service.js | TO BUILD |

---

## PART 5: IMPLEMENTATION SEQUENCE (RECOMMENDED ORDER)

### Sprint 1 (Week 1): Foundation
1. **WebSocket Infrastructure**
   - Create `backend/middleware/websocket.handler.js`
   - Register in `server.js`
   - Set up user-to-socket mapping

2. **Escalation Backend**
   - Create `escalation.service.js`
   - Create `escalation.model.js` (if needed)
   - Create `escalation.route.js` with POST `/api/escalations`

3. **Test End-to-End Webhook**
   - Nurse triggers escalation via REST API
   - Physician WebSocket receives message

### Sprint 2 (Week 2): Physician UI
1. **PhysicianAlertTray Component**
   - Persistent overlay that overrides all UI
   - Sound alert on new escalation
   - Acknowledge button with response tracking

2. **ClinicalCareMode Component** (Nurse side)
   - "Request Physician" button in Clinical Care Mode
   - Form to specify reason & vitals snapshot

3. **WebSocket Service Hook**
   - `useEscalationListener()` for both Nurse & Physician
   - Reconnection logic

### Sprint 3 (Week 3): EHR Bundling
1. **EHRBundleFactory**
   - Fetch all 5 objects
   - Integrity check (are they linked?)
   - Build HL7 FHIR bundle

2. **ehr.controller.js**
   - POST `/api/ehr/bundles` - Create bundle
   - POST `/api/ehr/sync` - Send to external EHR

3. **FinaliseEncounter Component**
   - Sign & Sync button
   - Show discrepancy warnings

### Sprint 4 (Week 4): Resource Management
1. **Bed auto-toggle service**
   - Mark bed as "VACANT_CLEANING" on finalise

2. **Staff active list service**
   - Clear staff's current patient assignment

3. **Analytics Dashboard**
   - Response Latency metrics
   - Discrepancy tracking
   - Resource usage (consumables, time)

---

## PART 6: DECISION POINTS (YOU'RE HERE)

### Question 1: Notifications Component
**Current State**: You have `Notifications.jsx` (static page)

**Decision**: Do you want to:
- **A. Enhance** the existing page to be real-time (WebSocket updates)?
- **B. Create new** PhysicianAlertTray as separate HIGH-PRIORITY overlay?
- **C. Both** (Notifications page for history, AlertTray for live)?

**My Recommendation**: **Option C** - They serve different purposes:
- **Notifications page**: History, read/unread toggle
- **PhysicianAlertTray**: LIVE alerts that interrupt the physician

---

### Question 2: Escalation Storage
**Decision**: Where to store escalation records?

- **A. New Collection**: `EscalationAlert` (MongoDB)
- **B. Nested in Encounter**: `encounter.escalationRequests: []`
- **C. Nested in Encounter + Cache in Redis**: Fast lookup

**My Recommendation**: **Option A** - Separate collection for:
- Scaling (millions of escalations over time)
- Query by response latency independently
- Analytics on physician response times

---

### Question 3: WebSocket vs. Polling
**Decision**: How should Physician get alerts?

- **A. WebSocket** (Live, low latency, "push" from server)
- **B. Long-polling** (HTTP, backwards compatible)
- **C. Both** (WebSocket primary, polling fallback)

**My Recommendation**: **Option C** - WebSocket for primary, HTTP long-polling as fallback for:
- Web stability
- Mobile app compatibility
- Network switching

---

## PART 7: API CONTRACTS (REST + WebSocket)

### REST API Endpoints

```
POST /api/escalations
Body: {
  encounterId: string,
  nurseId: string,
  reason: string,
  priority: "RED" | "ORANGE" | "YELLOW" | "GREEN",
  location: string,
  vitalsSnapshot: object
}
Response: { alertId, status: "PENDING", ... }

POST /api/escalations/:alertId/acknowledge
Body: { physicianId: string }
Response: { 
  status: "ACKNOWLEDGED", 
  responseLatency: number (ms),
  ...
}

POST /api/ehr/bundles
Body: { encounterId, action: "FINALISE_AND_SYNC" }
Response: {
  bundleId,
  resourceType: "Bundle",
  meta: { discrepancy_score, discrepancies_found: [] },
  ...
}

POST /api/ehr/sync
Body: { bundleId, targetEHR: string }
Response: { syncStatus: "QUEUED", estimatedTime: number }
```

### WebSocket Messages

```
// Server → Physician (when escalation triggered)
{
  type: "ESCALATION_ALERT",
  priority: "RED",
  location: "Bed 4",
  reason: "Vitals deteriorating",
  patientName: "John Doe",
  timestamp: ISO8601,
  encounterId: string
}

// Physician → Server (when acknowledged)
{
  type: "ESCALATION_ACKNOWLEDGE",
  alertId: string,
  timestamp: ISO8601
}

// Server → Physician (ongoing alerts you haven't acknowledged yet)
{
  type: "ESCALATION_SYNC",
  alerts: [{ ... }, { ... }]
}
```

---

---

# EXPANDED ARCHITECTURE: THE FINAL BOSS

## THE CHALLENGE: Concurrent ER + Ward Management

**Real Hospital Scenario**:
- A Physician on-call sees a "Red" trauma case (ER, needs immediate action)
- **Meanwhile**: A "Yellow" patient has been waiting 6 hours in observation
- **Meanwhile**: A Ward nurse flags a patient's heart rate creeping up slowly
- **Meanwhile**: An ER nurse is triaging a new "Green" ankle sprain

**The Problem**: Most hospital dashboards show ONE list. The physician gets blinded.

**Your Solution**: Multi-lane priority rendering based on CONTEXT (ER vs Ward, Role, Acuity, Time-Waiting).

---

## PART 1: THE VIEW MATRIX SYSTEM

### Concept: Role + Context = Dashboard Template

```javascript
// backend/src/config/viewMatrix.config.js

export const VIEW_MATRIX = {
  // Key format: "{role}|{context}|{acuity}"
  
  "PHYSICIAN|ON_CALL|ALL": {
    name: "Physician Command Centre (Hybrid)",
    lanes: [
      { id: "active_er", title: "🔴 ACTIVE ER", sortBy: "acuity_desc", color: "#dc2626" },
      { id: "waiting_discharge", title: "⏳ Observation (Awaiting Sign-Off)", sortBy: "time_waiting_asc", color: "#ea580c" },
      { id: "ward_rounds", title: "📋 Ward Rounds (In-Patient)", sortBy: "last_vitals_time_desc", color: "#0284c7" },
    ],
    escalationStyle: "AGGRESSIVE", // Pop-ups, beeping, override UI
    responseTimeTarget: 180, // seconds
  },

  "NURSE|TRIAGE|CASUALTY": {
    name: "Triage Nurse - Waiting Room",
    lanes: [
      { id: "queue", title: "👥 Queue (Waiting to be Triaged)", sortBy: "arrival_time_asc", color: "#6b7280" },
      { id: "in_progress", title: "⚕️ Currently Triaging", sortBy: "triage_start_time", color: "#f97316" },
      { id: "awaiting_physician", title: "📞 Awaiting Physician Review", sortBy: "acuity_desc", color: "#ec4899" },
    ],
    escalationStyle: "RAPID", // Audio alerts, rapid handoff
    responseTimeTarget: 60, // seconds per patient
  },

  "NURSE|WARD|GENERAL": {
    name: "Ward Nurse - Bed Management",
    lanes: [
      { id: "bed_map", title: "🛏️ Bed Assignment", renderMode: "GRID", sortBy: "bed_number" },
      { id: "vitals_overdue", title: "⚠️ Vitals Overdue", sortBy: "overdue_time_desc", color: "#fbbf24" },
      { id: "medication_due", title: "💊 Medications Due", sortBy: "due_time_asc", color: "#10b981" },
      { id: "red_flags", title: "🚨 Red Flags (Auto-escalate to Physician)", sortBy: "severity", color: "#dc2626" },
    ],
    escalationStyle: "BATCHED", // Show trend, batch non-critical alerts
    responseTimeTarget: 300, // seconds (more time for ward context)
  },

  "PHYSICIAN|WARD_ROUNDS|SPECIFIC_WARD": {
    name: "Physician - Ward Rounds (Specific Ward)",
    lanes: [
      { id: "assigned_beds", title: "👤 My Assigned Patients", sortBy: "last_review_time_desc", color: "#0284c7" },
      { id: "escalated", title: "⚠️ Escalated to Me", sortBy: "escalation_time_desc", color: "#dc2626" },
      { id: "discharge_ready", title: "✅ Ready for Discharge Review", sortBy: "disposition_date", color: "#16a34a" },
    ],
    escalationStyle: "CONTEXTUAL", // Wards are slower, but red flags still interrupt
    responseTimeTarget: 600, // seconds
  },
};
```

---

## PART 2: PATIENT STATE MACHINE

### Database Schema Update: Track Patient Journey

```javascript
// backend/src/models/patientJourney.model.js

const patientJourneySchema = new mongoose.Schema({
  // Patient reference
  patientId: { type: mongoose.Schema.Types.ObjectId, ref: "Patient", required: true },
  hospitalId: { type: mongoose.Schema.Types.ObjectId, ref: "Hospital", required: true },

  // CURRENT STATE (drives conditional rendering)
  currentState: {
    type: String,
    enum: [
      "WAITING_TRIAGE",      // Not yet seen by anyone - in waiting room
      "TRIAGING",            // Nurse is actively triaging
      "AWAITING_PHYSICIAN",  // Triage complete, waiting for doctor review
      "UNDER_TREATMENT_ER",  // Actively being treated in ER
      "OBSERVATION",         // Waiting in observation bed, awaiting disposition
      "HOSPITALIZED",        // Admitted to ward (state changes on disposition)
      "DISCHARGED",          // Left hospital (final state)
      "TRANSFERRED",         // Moved to another facility
    ],
    default: "WAITING_TRIAGE",
  },

  // LOCATION TRACKING (for which dashboard to show patient)
  location: {
    type: {
      type: String,
      enum: ["WAITING_ROOM", "ER_BED", "OBSERVATION_BED", "WARD_BED", "DISCHARGED"],
      default: "WAITING_ROOM",
    },
    bedNumber: String, // "ED-4" or "WARD-A-12"
    wardName: String, // "Emergency Department" or "Medical Ward A"
    assignedNurseId: mongoose.Schema.Types.ObjectId,
    assignedPhysicianId: mongoose.Schema.Types.ObjectId,
  },

  // TIMING DATA (for escalation and analytics)
  arrivedAt: { type: Date, default: Date.now },
  triageStartedAt: Date,
  triageCompletedAt: Date,
  physicianSeenAt: Date,
  
  // THE CRITICAL FIELD: Disposition
  disposition: {
    status: {
      type: String,
      enum: ["PENDING", "DISCHARGED", "ADMITTED", "TRANSFERRED", "DECEASED"],
      default: "PENDING",
    },
    decidedAt: Date,
    decidedBy: mongoose.Schema.Types.ObjectId, // Physician ID
    wards_admitted_to: [
      {
        wardName: String,
        bedNumber: String,
        admittedAt: Date,
        dischargedAt: Date,
      },
    ],
    reason_if_discharged: String, // "Resolved" | "Against medical advice" | etc
    discharge_referrals: [String], // ["Follow-up with GP", "Refer to Cardiology"]
  },

  // ACUITY TRACKING (for lane prioritization)
  currentAcuity: {
    type: String,
    enum: ["GREEN", "YELLOW", "ORANGE", "RED"],
    default: "GREEN",
  },
  acuityHistory: [
    {
      level: String,
      changedAt: Date,
      reason: String, // "Vitals deteriorating" | "Patient stable"
      changedBy: mongoose.Schema.Types.ObjectId,
    },
  ],

  // ESCALATION STATE (for alert rendering)
  escalations: [
    {
      requestedAt: Date,
      requestedBy: mongoose.Schema.Types.ObjectId,
      reason: String,
      priority: String,
      acknowledgedAt: Date,
      acknowledgedBy: mongoose.Schema.Types.ObjectId,
      responseLatency: Number, // ms
      status: String, // "PENDING", "ACKNOWLEDGED", "RESOLVED"
    },
  ],

  // AUDIT TRAIL (WHO CHANGED WHAT WHEN)
  stateTransitions: [
    {
      fromState: String,
      toState: String,
      changedAt: Date,
      changedBy: mongoose.Schema.Types.ObjectId, // User ID
      reason: String,
    },
  ],

  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
}, { timestamps: true });

// Indexes for rapid query filtering
patientJourneySchema.index({ hospitalId: 1, currentState: 1 });
patientJourneySchema.index({ hospitalId: 1, "location.type": 1 });
patientJourneySchema.index({ "location.assignedNurseId": 1, currentState: 1 });
patientJourneySchema.index({ "location.assignedPhysicianId": 1, currentState: 1 });
patientJourneySchema.index({ currentAcuity: 1, arrivedAt: -1 }); // For lane sorting
patientJourneySchema.index({ "disposition.status": 1, "disposition.decidedAt": -1 });

export const PatientJourney = mongoose.model("PatientJourney", patientJourneySchema);
```

---

## PART 3: THE 4 DASHBOARD VARIANTS

### Variant A: Physician Command Centre (Hybrid ER + Ward)

**Frontend Component**: `PhysicianCommandCentre.jsx`

```javascript
import React, { useState, useEffect } from 'react';
import { useLaneManager } from '../../hooks/useLaneManager';

const PhysicianCommandCentre = ({ physicianId, token }) => {
  const [lanes, setLanes] = useState({
    active_er: [],
    waiting_discharge: [],
    ward_rounds: [],
  });
  
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // 1. Connect to WebSocket for real-time updates
    const ws = new WebSocket(`${BACKEND_WS_URL}/physician/${physicianId}`);
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === "PATIENT_STATE_CHANGE") {
        // Re-fetch lanes (or could update incrementally)
        fetchLanes();
      }
      
      if (message.type === "ESCALATION_ALERT") {
        // 🔴 Auto-promote patient to top of Lane 1
        handleEscalation(message);
      }
    };
    
    setSocket(ws);
    fetchLanes();
    
    return () => ws.close();
  }, [physicianId]);

  const fetchLanes = async () => {
    const res = await fetch(
      `${BACKEND_URL}/api/physician/lanes?physicianId=${physicianId}&context=ON_CALL`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    const data = await res.json();
    setLanes(data);
  };

  const handleEscalation = (escalation) => {
    // Find the patient in any lane
    const patient = Object.values(lanes).flat().find(p => p._id === escalation.patientId);
    
    if (patient) {
      // Remove from current lane
      // Re-add to active_er with GLOWING BORDER
      setLanes(prev => ({
        ...prev,
        active_er: [
          { ...patient, alerting: true, alertSince: Date.now() },
          ...prev.active_er,
        ],
      }));
      
      // Play sound
      playAlertSound();
      flashScreen();
    }
  };

  return (
    <div style={styles.container}>
      {/* Lane 1: Active ER (NOW) */}
      <Lane
        title="🔴 ACTIVE ER - IMMEDIATE ACTION"
        patients={lanes.active_er}
        color="#dc2626"
        onPatientClick={(patient) => openPatientDetail(patient, "ER")}
        renderMode="CARD" // Large cards with vital signs live
      />

      {/* Lane 2: Observation (WAITING FOR ME) */}
      <Lane
        title="⏳ OBSERVATION - AWAITING SIGN-OFF"
        patients={lanes.waiting_discharge}
        color="#ea580c"
        onPatientClick={(patient) => openPatientDetail(patient, "OBSERVATION")}
        renderMode="LIST" // Smaller cards, sorted by time waiting
      />

      {/* Lane 3: Ward Rounds (ROUNDS) */}
      <Lane
        title="📋 WARD ROUNDS - IN-PATIENTS"
        patients={lanes.ward_rounds}
        color="#0284c7"
        onPatientClick={(patient) => openPatientDetail(patient, "WARD")}
        renderMode="GRID" // Bed-oriented view
      />

      {/* MODAL: Patient Detail + Disposition Control */}
      {selectedPatient && (
        <PatientDetailModal
          patient={selectedPatient}
          context={selectedContext}
          onDisposition={handleDisposition}
          onEscalate={handleEscalate}
          onClose={() => setSelectedPatient(null)}
        />
      )}
    </div>
  );
};

// Lane Component (Reusable)
const Lane = ({ title, patients, color, onPatientClick, renderMode }) => {
  return (
    <div style={{ ...styles.lane, borderLeftColor: color }}>
      <h2 style={{ color, marginBottom: 16, fontSize: 18, fontWeight: 700 }}>
        {title} ({patients.length})
      </h2>

      {renderMode === "CARD" && (
        <div style={styles.cardGrid}>
          {patients.map(patient => (
            <PatientCard
              key={patient._id}
              patient={patient}
              onClick={() => onPatientClick(patient)}
              priority="HIGH"
              showVitalsLive={true}
            />
          ))}
        </div>
      )}

      {renderMode === "LIST" && (
        <div style={styles.list}>
          {patients.map(patient => (
            <PatientRow
              key={patient._id}
              patient={patient}
              onClick={() => onPatientClick(patient)}
              showTimeWaiting={true}
            />
          ))}
        </div>
      )}

      {renderMode === "GRID" && (
        <BedGrid patients={patients} onBedClick={onPatientClick} />
      )}
    </div>
  );
};
```

---

### Variant B: Triage Nurse Dashboard (ER Queue)

**Frontend Component**: `TriageNurseDashboard.jsx`

```javascript
const TriageNurseDashboard = ({ nurseId, token }) => {
  const [lanes, setLanes] = useState({
    queue: [],              // Waiting to be triaged
    in_progress: [],        // Currently being triaged
    awaiting_physician: [], // Triage done, waiting for doc
  });

  useEffect(() => {
    // Stream updates for QUEUE movements
    const ws = new WebSocket(`${BACKEND_WS_URL}/nurse/triage/${nurseId}`);
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === "PATIENT_JOINED_QUEUE") {
        setLanes(prev => ({
          ...prev,
          queue: [...prev.queue, message.patient].sort(sortByArrivalTime),
        }));
        
        // Quick beep - new patient
        playQuietAlert();
      }
    };
  }, [nurseId]);

  return (
    <div style={styles.container}>
      {/* LANE 1: Queue */}
      <div style={styles.lane}>
        <h2 style={{ color: "#6b7280" }}>
          👥 WAITING TO BE TRIAGED ({lanes.queue.length})
        </h2>
        <div style={styles.queueList}>
          {lanes.queue.map((patient, idx) => (
            <QueuePatientCard
              key={patient._id}
              patient={patient}
              position={idx + 1}
              waitingMinutes={getWaitingMinutes(patient.arrivedAt)}
              onSelect={() => handleSelectForTriage(patient)}
            />
          ))}
        </div>
      </div>

      {/* LANE 2: In Progress */}
      <div style={styles.lane}>
        <h2 style={{ color: "#f97316" }}>
          ⚕️ CURRENTLY TRIAGING ({lanes.in_progress.length})
        </h2>
        {lanes.in_progress.map(patient => (
          <TriagingCard key={patient._id} patient={patient} />
        ))}
      </div>

      {/* LANE 3: Awaiting Physician */}
      <div style={styles.lane}>
        <h2 style={{ color: "#ec4899" }}>
          📞 AWAITING PHYSICIAN ({lanes.awaiting_physician.length})
        </h2>
        {lanes.awaiting_physician.map(patient => (
          <AwaitingPhysicianCard
            key={patient._id}
            patient={patient}
            showWaitingTime={true}
          />
        ))}
      </div>
    </div>
  );
};
```

---

### Variant C: Ward Nurse Dashboard (In-Patient Management)

**Frontend Component**: `WardNurseDashboard.jsx`

```javascript
const WardNurseDashboard = ({ nurseId, assignedWard, token }) => {
  const [bedMap, setBedMap] = useState({
    occupied: [],
    vacant: [],
    alerts: [],
  });

  useEffect(() => {
    // Stream bed assignments and vital changes
    const ws = new WebSocket(
      `${BACKEND_WS_URL}/nurse/ward/${nurseId}?ward=${assignedWard}`
    );
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === "VITALS_RED_FLAG") {
        // Patient's vitals deteriorating - show trend warning
        showTrendWarning(message);
      }
    };
  }, [nurseId, assignedWard]);

  return (
    <div style={styles.wardDashboard}>
      {/* SECTION 1: Bed Map Grid */}
      <div style={styles.section}>
        <h2>🛏️ BED MAP - {assignedWard}</h2>
        <BedMapGrid
          beds={bedMap.occupied}
          onBedClick={(bed) => openBedDetail(bed)}
        />
      </div>

      {/* SECTION 2: Vitals Overdue */}
      <div style={styles.section}>
        <h2>⚠️ VITALS OVERDUE</h2>
        {bedMap.vitalOverdue.map(patient => (
          <VitalsOverdueCard
            key={patient._id}
            patient={patient}
            overdueMinutes={getOverdueMinutes(patient.lastVitalsTime)}
            onQuickEntry={() => enterVitals(patient)}
          />
        ))}
      </div>

      {/* SECTION 3: Red Flags (Auto-Escalate) */}
      <div style={styles.section}>
        <h2>🚨 RED FLAGS - AUTO-ESCALATE TO PHYSICIAN</h2>
        {bedMap.alerts.map(alert => (
          <RedFlagAlert
            key={alert._id}
            alert={alert}
            trend={alert.trendData} // Heart rate creeping up over 3 hrs
            onAcknowledge={() => acknowledgeAlert(alert._id)}
          />
        ))}
      </div>

      {/* SECTION 4: Medications Due */}
      <div style={styles.section}>
        <h2>💊 MEDICATIONS DUE</h2>
        {bedMap.medicationDue.map(med => (
          <MedicationDueCard
            key={med._id}
            medication={med}
            onAdminister={() => recordMedicationAdministration(med)}
          />
        ))}
      </div>
    </div>
  );
};

// BED MAP GRID (Render beds as visual grid)
const BedMapGrid = ({ beds, onBedClick }) => {
  return (
    <div style={styles.bedGrid}>
      {beds.map(bed => (
        <div
          key={bed.bedNumber}
          onClick={() => onBedClick(bed)}
          style={{
            ...styles.bedCell,
            backgroundColor: getBedColor(bed.patientAcuity),
            borderColor: bed.hasAlert ? "#dc2626" : "#e5e7eb",
            borderWidth: bed.hasAlert ? 3 : 1,
            cursor: "pointer",
          }}
        >
          <div style={styles.bedNumber}>Bed {bed.bedNumber}</div>
          <div style={styles.patientName}>{bed.patientName || "VACANT"}</div>
          {bed.patientAcuity && (
            <div style={styles.acuityIndicator}>{bed.patientAcuity}</div>
          )}
          {bed.hasAlert && <div style={styles.alertIcon}>⚠️</div>}
        </div>
      ))}
    </div>
  );
};
```

---

### Variant D: Physician on Ward Rounds (Specific Ward Focus)

**Frontend Component**: `WardRoundsPhysician.jsx`

```javascript
const WardRoundsPhysician = ({ physicianId, wardId, token }) => {
  const [patients, setPatients] = useState({
    assigned_beds: [],
    escalated: [],
    discharge_ready: [],
  });

  return (
    <div style={styles.wardRounds}>
      {/* Lane 1: My Assigned Patients */}
      <Lane
        title="👤 MY ASSIGNED PATIENTS"
        patients={patients.assigned_beds}
        renderMode="CARD"
        compact={true}
      />

      {/* Lane 2: Escalated (Auto-promoted) */}
      <Lane
        title="⚠️ ESCALATED TO ME"
        patients={patients.escalated}
        renderMode="CARD"
        highlight="RED"
      />

      {/* Lane 3: Discharge Ready */}
      <Lane
        title="✅ READY FOR DISCHARGE REVIEW"
        patients={patients.discharge_ready}
        renderMode="LIST"
      />
    </div>
  );
};
```

---

## PART 4: THE DISPOSITION LOGIC (The Critical Gate)

### Physician Feedback Modal with Disposition Control

```javascript
// frontend/src/Dashboard/Components/DispositionControl.jsx

const DispositionControl = ({ patientId, onDispose, showAsModal = true }) => {
  const [disposition, setDisposition] = useState("PENDING");
  const [selectedWard, setSelectedWard] = useState(null);
  const [dischargeReason, setDischargeReason] = useState("");
  const [referrals, setReferrals] = useState([]);

  const handleDispose = async () => {
    const payload = {
      patientId,
      action: disposition === "DISCHARGED" ? "DISCHARGE" : "ADMIT",
      ...(disposition === "DISCHARGED" && {
        reason: dischargeReason,
        referrals,
      }),
      ...(disposition === "ADMITTED" && {
        wardId: selectedWard,
      }),
    };

    const res = await fetch(`${BACKEND_URL}/api/disposition`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      // THIS IS WHERE THE MAGIC HAPPENS:
      // 
      // 1. Patient state changes from OBSERVATION → HOSPITALIZED or DISCHARGED
      // 2. Patient moves from "Observation Lane" → "Ward Lane" or is removed entirely
      // 3. EHR bundle is created with ENCOUNTER TYPE = "INPATIENT" or "AMBULATORY"
      // 4. If HOSPITALIZED: Bed is assigned, nurse is notified
      // 5. If DISCHARGED: Clock stops, metrics are calculated
      
      toast.success(`Patient ${disposition}`);
      onDispose({ disposition, wardId: selectedWard });
    }
  };

  return (
    <div style={styles.dispositionModal}>
      <h2>📋 DISPOSITION: What happens next?</h2>

      <div style={styles.options}>
        <button
          onClick={() => setDisposition("DISCHARGED")}
          style={{
            ...styles.btn,
            backgroundColor: disposition === "DISCHARGED" ? "#16a34a" : "#e5e7eb",
          }}
        >
          ✅ DISCHARGE TO COMMUNITY
        </button>

        <button
          onClick={() => setDisposition("ADMITTED")}
          style={{
            ...styles.btn,
            backgroundColor: disposition === "ADMITTED" ? "#0284c7" : "#e5e7eb",
          }}
        >
          🛏️ ADMIT TO WARD
        </button>
      </div>

      {disposition === "DISCHARGED" && (
        <div>
          <label>Reason for Discharge</label>
          <select
            value={dischargeReason}
            onChange={(e) => setDischargeReason(e.target.value)}
          >
            <option value="">Select reason...</option>
            <option value="RESOLVED">Condition Resolved</option>
            <option value="STABILIZED">Stabilized for Home Care</option>
            <option value="AMA">Against Medical Advice</option>
            <option value="TRANSFERRED">Transferred to Another Facility</option>
          </select>

          <label>Referrals</label>
          <textarea
            value={referrals.join("\n")}
            onChange={(e) => setReferrals(e.target.value.split("\n"))}
            placeholder="e.g., Follow-up with GP in 2 weeks&#10;Refer to Cardiology"
          />
        </div>
      )}

      {disposition === "ADMITTED" && (
        <div>
          <label>Select Ward</label>
          <select value={selectedWard} onChange={(e) => setSelectedWard(e.target.value)}>
            <option value="">Choose ward...</option>
            <option value="MEDICAL_WARD_A">Medical Ward A</option>
            <option value="SURGERY_WARD">Surgery Ward</option>
            <option value="ICU">ICU</option>
          </select>
        </div>
      )}

      <button onClick={handleDispose} style={styles.submitBtn}>
        SAVE DISPOSITION & CLOSE CASE
      </button>
    </div>
  );
};
```

### Backend: Handle Disposition (State Transition)

```javascript
// backend/src/controllers/disposition.controller.js

export const handleDisposition = async (req, res) => {
  const { patientId, action, reason, ward, referrals } = req.body;
  const physicianId = req.user._id;

  try {
    const patient = await PatientJourney.findById(patientId);

    if (action === "DISCHARGE") {
      // Step 1: Update patient state
      patient.currentState = "DISCHARGED";
      patient.disposition.status = "DISCHARGED";
      patient.disposition.decidedAt = new Date();
      patient.disposition.decidedBy = physicianId;
      patient.disposition.reason_if_discharged = reason;
      patient.disposition.discharge_referrals = referrals;

      // Step 2: Create AMBULATORY EHR bundle
      const bundle = await EHRBundleFactory.createBundle(patient.encounterId, "AMBULATORY");

      // Step 3: Queue for EHR sync
      await EHRSyncQueue.create({
        bundleId: bundle._id,
        bundleType: "AMBULATORY",
        status: "PENDING",
        targetEHR: "Med-e-Mass",
      });

      // Step 4: Record metrics
      const timeInSystem = Date.now() - patient.arrivedAt;
      await ConsultationMetrics.create({
        patientId,
        type: "AMBULATORY",
        totalTime: timeInSystem,
        dispositionTime: Date.now() - patient.physicianSeenAt,
      });

      await patient.save();
      
      return res.json({
        status: "DISCHARGED",
        bundleId: bundle._id,
        metrics: { timeInSystem },
      });
    }

    if (action === "ADMIT") {
      // Step 1: Update patient state
      patient.currentState = "HOSPITALIZED";
      patient.disposition.status = "ADMITTED";
      patient.disposition.decidedAt = new Date();
      patient.disposition.decidedBy = physicianId;

      // Step 2: Assign bed
      const bed = await assignBed(ward);
      patient.location.type = "WARD_BED";
      patient.location.bedNumber = bed.bedNumber;
      patient.location.wardName = ward;

      // Step 3: Create INPATIENT EHR bundle (note: NOT finalized, stays open)
      const bundle = await EHRBundleFactory.createBundle(
        patient.encounterId,
        "INPATIENT",
        { admittingWard: ward, bedNumber: bed.bedNumber }
      );

      // Step 4: Notify ward nurse
      const wardNurses = await Nurse.find({ assignedWard: ward, isOnDuty: true });
      for (const nurse of wardNurses) {
        const ws = wsConnections.get(nurse._id.toString());
        if (ws) {
          ws.send(JSON.stringify({
            type: "NEW_PATIENT_ADMITTED",
            patientId,
            patientName: patient.name,
            bedNumber: bed.bedNumber,
            acuity: patient.currentAcuity,
          }));
        }
      }

      // Step 5: Record metrics
      const timeToDisposition = Date.now() - patient.physicianSeenAt;
      await ConsultationMetrics.create({
        patientId,
        type: "ADMISSION",
        timeToDisposition, // 🎯 **DECISION TO ADMIT LATENCY** - This is your CEO metric!
      });

      await patient.save();

      return res.json({
        status: "ADMITTED",
        bedNumber: bed.bedNumber,
        wardName: ward,
        bundleId: bundle._id,
      });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```

---

## PART 5: ESCALATION RENDERING BY CONTEXT

### Different Alert Styles for Different Worlds

```javascript
// frontend/src/hooks/useEscalationAlert.js

export const useEscalationAlert = (context) => {
  const showAlert = (escalation) => {
    if (context === "ER") {
      // ER: AGGRESSIVE - must respond in seconds
      playLoudAlertSound("alarm_critical.mp3"); // Very loud
      flashScreenRedish(); // Aggressive visual
      renderPopupOverlay(escalation); // Modal overlays everything
      setAlertDuration(10000); // Auto-close after 10s if not ack'd
    } else if (context === "WARD") {
      // WARD: BATCHED + TREND - can respond slower, but show trends
      playQuietAlert("beep.mp3"); // Gentle beep
      showTrendWarningCard(escalation); // Show creeping trend
      renderInLane(escalation); // Add to escalation lane (don't override)
      setAlertDuration(null); // Stays visible until manually dismissed
    }
  };

  return { showAlert };
};
```

---

## PART 6: ANALYTICS IMPACT

### Metrics That Drive CEO Decisions

```javascript
// backend/src/services/hospitalMetrics.service.js

export class HospitalMetricsService {
  // RESOURCE WASTE: How long between "decision to admit" and actual bed assignment?
  async calculateDecisionToAdmitLatency(dateRange) {
    const admissions = await ConsultationMetrics.find({
      type: "ADMISSION",
      createdAt: { $gte: dateRange.start, $lte: dateRange.end },
    });

    return {
      avgLatency: avg(admissions.map(a => a.timeToDisposition)),
      p95Latency: percentile(admissions.map(a => a.timeToDisposition), 95),
      medicationChangesMissed: admissions.length, // If latency > acceptable
    };
  }

  // ESCALATION EFFECTIVENESS: Are physicians responding fast?
  async calculatePhysicianResponseTime(physicianId, context) {
    const escalations = await EscalationAlert.find({
      acknowledgedBy: physicianId,
      payload: { priority: { $in: ["RED", "ORANGE"] } },
    });

    if (context === "ER") {
      return {
        avgResponseTime: avg(escalations.map(e => e.responseLatency)),
        targetResponseTime: 180, // seconds
        compliance: escalations.filter(e => e.responseLatency < 180).length / escalations.length,
      };
    } else if (context === "WARD") {
      return {
        avgResponseTime: avg(escalations.map(e => e.responseLatency)),
        targetResponseTime: 600, // 10 minutes for wards
        compliance: escalations.filter(e => e.responseLatency < 600).length / escalations.length,
      };
    }
  }

  // TRIAGE ACCURACY: Did the nurse get it right?
  async calculateTriageAccuracy() {
    const escalations = await EscalationAlert.aggregate([
      {
        $lookup: {
          from: "triagerequests",
          localField: "encounterId",
          foreignField: "encounterId",
          as: "triage",
        },
      },
      {
        $addFields: {
          wasNurseWrong: {
            $ne: ["$triage.manualTriageReport.satsLevel", "$escalation.priority"],
          },
        },
      },
    ]);

    return {
      accuracy: escalations.filter(e => !e.wasNurseWrong).length / escalations.length,
      misTriages: escalations.filter(e => e.wasNurseWrong),
    };
  }
}
```

---

## PART 7: YOUR DATA MODEL CHECKLIST

### New/Enhanced Models Needed

```
[  ] PatientJourney (NEW) - Tracks patient through ER → Ward → Discharge
     ├─ currentState: enum
     ├─ location: embedded (bed, ward, assigned staff)
     ├─ disposition: embedded (status, decided by whom, referrals)
     └─ escalations: array of escalation records

[  ] ViewMatrix CONFIG (NEW) - Define dashboard variants
     ├─ {role|context} → dashboard template
     └─ lanes, sortBy, escalationStyle

[  ] ConsultationMetrics (NEW or ENHANCE) - Timing analytics
     ├─ patientId
     ├─ timeInSystem
     ├─ timeToDisposition    ← 🎯 **CEO wants this**
     └─ responseLatencies

[  ] Enhance Encounter.model.js
     ├─ Add linked patientJourneyId
     └─ Add disposition tracking

[  ] Enhance EscalationAlert.model.js
     ├─ Add context field (ER vs WARD)
     └─ Add responseLatency tracking
```

---

## NEXT IMMEDIATE ACTIONS

### To Build the Hybrid Dashboard System:

**Phase 1: Data Layer (Sprint 1-2, Week 1-2)**
1. Create `PatientJourney.model.js` with state machine
2. Create `ViewMatrix` configuration system
3. Enhance existing models to reference patientJourney

**Phase 2: Backend API (Sprint 2-3, Week 2-3)**
1. Create `/api/physician/lanes` endpoint (returns 3 lanes based on VIEW_MATRIX)
2. Create `/api/nurse/queue` endpoint (returns triage queue)
3. Create `/api/nurse/bed-map` endpoint (returns ward bed assignments)
4. Create `/api/disposition` endpoint (handles admit/discharge state transitions)

**Phase 3: Frontend Components (Sprint 3-4, Week 3-4)**
1. Build `PhysicianCommandCentre.jsx` (3-lane hybrid view)
2. Build `TriageNurseDashboard.jsx` (queue-focused)
3. Build `WardNurseDashboard.jsx` (bed-map grid)
4. Build `DispositionControl.jsx` (the critical gate)

**Phase 4: Integration (Sprint 4, Week 4)**
1. Hook up WebSocket for state changes
2. Implement escalation re-routing (auto-promote to Lane 1)
3. Add metrics collection to ConsultationMetrics

---

## PART 8: BED REQUIREMENT TRACKING (Critical for CEO Metrics)

The `needsBed` field is a **leading indicator** of patient journey flow and a powerful accuracy metric:

```
TRIAGE NURSE: "Does this patient need a bed?"
                    ↓
              YES (needsBed=true)    OR    NO (needsBed=false)
                    ↓                              ↓
            PREDICTED: ADMIT              PREDICTED: DISCHARGE/OBSERVATION
                    ↓                              ↓
    Goes to AWAITING_PHYSICIAN      Goes to AWAITING_PHYSICIAN
                    ↓                              ↓
    PHYSICIAN confirms or overrides  PHYSICIAN confirms or overrides
                    ↓                              ↓
    [DISCREPANCY CHECK]    ---------→ [ACCURACY METRIC] ← CEO Dashboard metric
            ↓                              ↓
    If triage ≠ physician:           Track as learning signal
      Log in TriageDiscrepancyLog    for model improvement
      Monitor for patterns            (daily trending)
```

### Why This Matters (CEO Perspective)

**Daily Triage Accuracy Report Example**:
```
Metrics from Yesterday:
- 87 patients triaged
- Nurse said "yes bed needed": 45 patients
  └─ Physician admitted 38 → 84% RECALL RATE (excellent)
- Nurse said "no bed needed": 42 patients
  └─ Physician discharged 38 → 90% SPECIFICITY (excellent)
- DISCREPANCIES DETECTED: 4 false negatives
  └─ Cost impact: $3,200 per unexpected readmission
  └─ Action item: Review 4 cases in QA meeting

Recommendation: Train team on early deterioration recognition
```

### Implementation Flow

```javascript
// STEP 1: During triage (Nurse)
const triageAssessment = await TriageAssessment.create({
  patientId,
  triagePriority: 3,
  needsBed: true,                 // ← THE KEY DECISION
  bedType: "WARD",
  specialRequirements: ["cardiac_monitoring"],
  reasoning: "Acute abdomen, stable vitals, needs imaging",
  assessedBy: triageNurseId,
  assessedAt: new Date(),
});

// STEP 2: Waiting for physician (patient state = AWAITING_PHYSICIAN)
// System tracks: How long until physician reviews?
const waitTime = Date.now() - triageAssessment.assessedAt;
if (waitTime > ER_TARGET_LATENCY) {
  // Alert: Physician review overdue
  escalateAlert("PHYSICIAN_REVIEW_OVERDUE", { patientId, waitTime });
}

// STEP 3: During disposition (Physician)
const disposition = await Disposition.create({
  patientId,
  physicianDecision: "ADMITTED",  // Physician's actual decision
  triageRecommendation: triageAssessment.needsBed,  // What nurse predicted
  discrepancy: triageAssessment.needsBed === true ? null : {
    type: "FALSE_NEGATIVE",
    reason: "Condition deteriorated in observation",
    costImpact: 3200,
  },
  decidedAt: new Date(),
  decidedBy: physicianId,
});

// STEP 4: Log discrepancy for QA/ML training
if (disposition.discrepancy) {
  await TriageDiscrepancyLog.create({
    triageAssessmentId: triageAssessment._id,
    dispositionId: disposition._id,
    discrepancy: disposition.discrepancy,
    hospitalId: req.user.hospitalId,
    date: new Date(),
  });
}
```

### CEO Dashboard Component

```javascript
// backend/src/routes/analytics.route.js
export const getDailyTriageAccuracy = async (req, res) => {
  const { hospitalId, date } = req.query;
  
  // Get all dispositions from yesterday
  const yesterday = new Date(date);
  yesterday.setDate(yesterday.getDate() - 1);
  
  const dispositions = await Disposition.find({
    hospitalId,
    decidedAt: { $gte: yesterday, $lt: date },
  }).populate("triageRecommendation");

  // Calculate metrics
  const totalCases = dispositions.length;
  const admitPredicted = dispositions.filter(d => d.triageRecommendation === true).length;
  const admitActual = dispositions.filter(d => d.physicianDecision === "ADMITTED").length;
  const discrepancies = dispositions.filter(d => d.discrepancy !== null).length;
  
  const truePositives = dispositions.filter(d => 
    d.triageRecommendation === true && d.physicianDecision === "ADMITTED"
  ).length;
  
  const falseNegatives = dispositions.filter(d => 
    d.triageRecommendation === false && d.physicianDecision === "ADMITTED"
  ).length;

  const recallRate = admitActual > 0 ? (truePositives / admitActual * 100).toFixed(1) : "N/A";
  const specificityRate = totalCases - admitActual > 0 
    ? ((totalCases - admitActual - falseNegatives) / (totalCases - admitActual) * 100).toFixed(1) 
    : "N/A";

  res.json({
    date: date.toISOString().split('T')[0],
    summary: {
      totalCases,
      discrepancies,
      discrepancyRate: (discrepancies / totalCases * 100).toFixed(1) + "%",
    },
    accuracy: {
      recallRate: recallRate + "%",      // Of admitted patients, % we predicted correctly
      specificityRate: specificityRate + "%", // Of discharged, % we predicted correctly
      overallAccuracy: ((truePositives + (totalCases - admitActual - falseNegatives)) / totalCases * 100).toFixed(1) + "%",
    },
    details: {
      truePositives,
      falseNegatives,
      costImpactFalseNegatives: falseNegatives * 3200,
    },
  });
};

// Usage in frontend:
// GET /api/analytics/triage-accuracy?date=2026-04-19&hospitalId=XXX
```

---

## CLOSING: WHY THIS MATTERS

You're not building a "Patient Management System"—you're building a **Concurrent State Machine** that tracks the same patient through multiple contexts simultaneously:

- **At 2 PM**: Patient is in ER (Nurse sees them in Queue, Physician sees them in Active ER Lane)
- **At 3 PM**: Physician decides to admit → Patient state changes to HOSPITALIZED
- **At 3 PM (now)**: Nurse sees them in Ward Dashboard, not in ER Queue
- **At 7 PM**: Ward nurse escalates deteriorating vitals → Auto-promotes to Physician's Ward Escalation Lane
- **At 8 PM**: Physician orders discharge → Patient state → DISCHARGED, EHR syncs, metrics calculated

**This is the Final Boss because** most hospital systems can't handle this. They show ONE list. Your multi-lane system makes the physician feel in control across all contexts.

**CEO will love this because** they get the "Decision to Admit Latency" metric—the killer waste metric showing how long patients wait for bed assignments after doctors decide to admit them.

---

**Ready to build?** Start with the data layer (PatientJourney model) or the backend API (/api/physician/lanes)?
