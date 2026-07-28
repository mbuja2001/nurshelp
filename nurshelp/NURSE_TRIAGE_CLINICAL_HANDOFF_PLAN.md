# Nurse Triage → Clinical Handoff Implementation Plan
## User Journey-Driven Development

**Status**: Ready to implement  
**Date**: 20 April 2026  
**Goal**: Build dashboard routing based on assigned shift role within a single clock-in session

---

## THE USER JOURNEY (Actual Hospital Day)

### Timeline: 7:00 AM - 3:00 PM Shift

```
7:00 AM
└─ Triage Nurse "Alice" clocks in
   └─ Role: ROLE_TRIAGE_NURSE, Shift: CASUALTY/ED
   └─ Dashboard: "Triage Queue" (waiting room patients)
   └─ ACTION: Assign clinical team
       └─ Picks: Clinical Nurse "Bob", Physician "Dr. Chen" (ER specialty)
       └─ State: Bob + Dr. Chen are now ASSIGNED FOR THIS SHIFT
       
7:15 AM
└─ Patient "John" arrives → Triage Queue
   └─ Alice triages John → YELLOW priority
   └─ ⭐ NEW STEP: Alice SPECIFIES: "NEEDS BED? YES"
   └─ State: John.triageData.needsBed = true
   └─ Hands off to Bob (Clinical Nurse)
   
7:20 AM
└─ Bob is now in "CLINICAL MODE"
   └─ Dashboard: "Clinical Nurse - Managing Non-Hospitalized Patients"
   └─ STATE: John is in OBSERVATION BED, awaiting physician review
   └─ Bob knows: "This patient potentially needs a bed" (from triage data)
   
7:30 AM
└─ Dr. Chen receives notification: "John in Observation - awaiting sign-off"
   └─ Dr. Chen's Dashboard: "Physician Clinical Mode - Active Cases"
   └─ Shows: "Triage Assessment: NEEDS BED"
   └─ ACTION: Review John → Decide based on triage assessment:
       ├─ If DISCHARGE → John doesn't need bed, dismisses to community (AMBULATORY EHR)
       ├─ If ADMIT → John needs bed, assigned to ward (INPATIENT EHR)
       └─ If DISCHARGE (conflicting triage) → Records DISCREPANCY in triage accuracy
   
8:00 AM (Scenario A: John needs bed + stays)
└─ Dr. Chen: ADMIT John to Medical Ward A, Bed 12
   └─ EHR Bundle: INPATIENT admission record
   └─ Bob's dashboard: John removed (transitioned to Ward Nurse)
   └─ Ward Nurse sees John appear in bed map
   
8:00 AM (Scenario B: John discharged despite needs bed)
└─ Dr. Chen: DISCHARGE John (vital signs improved, safe for home)
   └─ EHR Bundle: AMBULATORY record
   └─ System flags: ⚠️ DISCREPANCY: "Triage said Needs Bed, but Physician discharged"
   └─ Analytics: This becomes a triage accuracy metric
```

---

## IMPLEMENTATION LAYERS

### Layer 1: Shift Role System + Triage Assessment

**Status**: You likely have this in `Nurse.model.js` and `schedule.model.js`

**Need to create**:
```javascript
// backend/src/models/shift.model.js

const shiftSchema = new mongoose.Schema({
  hospitalId: mongoose.Schema.Types.ObjectId,
  date: Date,
  
  // SHIFT CONTEXT (determines which dashboard to show)
  shiftContext: {
    type: String,
    enum: [
      "TRIAGE_NURSE",           // Works in ER waiting room
      "CLINICAL_NURSE_ER",      // Manages non-hospitalized (observation)
      "CLINICAL_NURSE_WARD",    // Manages hospitalized (ward beds)
      "PHYSICIAN_ER",           // Manages ER + Observation cases
      "PHYSICIAN_WARD",         // Manages ward rounds
    ],
  },

  // WHO IS ASSIGNED
  assignedPersonnel: {
    // TRIAGE TEAM (picks their own clinical team)
    trageNurseId: mongoose.Schema.Types.ObjectId,
    triageNurseName: String,

    // CLINICAL TEAM (picked by triage nurse)
    clinicalNurseId: mongoose.Schema.Types.ObjectId,
    clinicalNurseName: String,
    clinicalNurseContext: { type: String, enum: ["ER", "WARD"] },

    physicianId: mongoose.Schema.Types.ObjectId,
    physicianName: String,
    physicianSpecialty: String, // "ER", "Surgery", "Family Medicine"
    physicianContext: String,   // "ER" | "WARD_ROUNDS"
  },

  // SHIFT TIMING
  shiftStart: Date,
  shiftEnd: Date,
  breakStart: Date,
  breakEnd: Date,

  // STATE TRACKING
  status: { type: String, enum: ["PENDING", "ACTIVE", "COMPLETED"] },
  teamAssignedAt: Date,
  teamAssignedBy: mongoose.Schema.Types.ObjectId, // Triage nurse

  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

export const Shift = mongoose.model("Shift", shiftSchema);
```

---

### NEW: Triage Assessment Model

**File**: `backend/src/models/triageAssessment.model.js`

```javascript
/**
 * TRIAGE ASSESSMENT
 * Created by Triage Nurse when patient is triaged
 * Tracks: Priority + Whether patient needs hospitalization (bed)
 */

const triageAssessmentSchema = new mongoose.Schema({
  // REFERENCE
  patientId: mongoose.Schema.Types.ObjectId,
  encounterId: mongoose.Schema.Types.ObjectId,
  triageNurseId: { type: mongoose.Schema.Types.ObjectId, ref: "Nurse" },
  hospitalId: mongoose.Schema.Types.ObjectId,

  // TRIAGE CLINICAL DATA (from Triage form)
  chiefComplaint: String,
  symptoms: [String],
  vitalSigns: {
    temperature: Number,
    heartRate: Number,
    respiratoryRate: Number,
    systolicBP: Number,
    diastolicBP: Number,
    oxygenSaturation: Number,
  },
  medicalHistory: String,
  medications: [String],
  allergies: [String],

  // ⭐ TRIAGE PRIORITY (SATS scoring)
  triagePriority: {
    type: String,
    enum: ["GREEN", "YELLOW", "ORANGE", "RED"],
    default: "YELLOW",
  },
  triageReasoning: String, // Why this priority?

  // ⭐ CRITICAL FIELD: DOES PATIENT NEED A BED?
  needsBed: {
    type: Boolean,
    required: true, // Triage nurse MUST specify
    description: "If true, patient likely needs hospitalization after physician approval",
  },

  bedNeedReasoning: String, // Why does/doesn't patient need bed?
  // Examples:
  // "YES: Suspected appendicitis, may need surgery"
  // "NO: Ankle sprain, likely discharged after imaging"
  // "MAYBE: Needs observation for 6 hours, then decided"

  // TRIAGE TIMING
  triageStartedAt: Date,
  triageCompletedAt: Date,

  // STATE AFTER TRIAGE
  status: {
    type: String,
    enum: ["AWAITING_CLINICAL_NURSE", "WITH_CLINICAL_NURSE", "PHYSICIAN_REVIEW"],
    default: "AWAITING_CLINICAL_NURSE",
  },

  // PHYSICIAN DECISION (filled in later by physician)
  physicianDecision: {
    status: { type: String, enum: ["PENDING", "DISCHARGED", "ADMITTED"] },
    decidedAt: Date,
    decidedBy: mongoose.Schema.Types.ObjectId,
    
    // DISCREPANCY TRACKING
    discrepancy: {
      hasDiscrepancy: Boolean, // Did physician decision conflict with triage assessment?
      // Examples:
      // "Triage said needsBed=true, but physician discharged"
      // "Triage said needsBed=false, but physician admitted"
      triagePrediction: String,
      physicianDecision: String,
      reason: String, // Why the discrepancy?
    },
  },

  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
}, { timestamps: true });

triageAssessmentSchema.index({ hospitalId: 1, triageCompletedAt: -1 });
triageAssessmentSchema.index({ patientId: 1 });
triageAssessmentSchema.index({ triageNurseId: 1, triageCompletedAt: -1 });
triageAssessmentSchema.index({ needsBed: 1, triagePriority: 1 }); // For analytics

export const TriageAssessment = mongoose.model("TriageAssessment", triageAssessmentSchema);
```

---

### Layer 2: Dashboard Routing Engine

**File**: `backend/src/config/dashboardRouter.config.js`

```javascript
/**
 * DASHBOARD ROUTING MATRIX
 * Maps: (userId + currentShiftContext) → Dashboard Component + Data Fetch Endpoint
 */

export const DASHBOARD_ROUTES = {
  // ========== TRIAGE NURSE CONTEXT ==========
  TRIAGE_NURSE: {
    component: "TriageNurseDashboard",
    dataEndpoint: "/api/nurse/triage/queue",
    permissions: [
      "view_waiting_patients",
      "assign_clinical_team",
      "perform_triage",
      "create_encounter",
    ],
    controls: {
      canAssignTeam: true,        // Button: "Assign Clinical Team"
      canTriage: true,
      canViewClinicalMode: false, // Can't see patient hospital details
      canViewWards: false,
    },
    escalationStyle: "RAPID",
    responseTimeTarget: 60,
  },

  // ========== CLINICAL NURSE (ER - Non-Hospitalized) ==========
  CLINICAL_NURSE_ER: {
    component: "ClinicalNurseDashboard_ER",
    dataEndpoint: "/api/nurse/clinical/non-hospitalized",
    permissions: [
      "view_observation_patients",
      "log_vitals",
      "request_physician",
      "manage_observations",
    ],
    controls: {
      canAssignTeam: false,
      canTriage: false,
      canViewClinicalMode: true,  // Can see full observation details
      canViewWards: false,
      canRequestPhysician: true,  // Button: "Request Physician Review"
      canDischargeFromObservation: false, // Physician-only
    },
    escalationStyle: "BATCHED",
    responseTimeTarget: 180,
  },

  // ========== CLINICAL NURSE (WARD - Hospitalized) ==========
  CLINICAL_NURSE_WARD: {
    component: "WardNurseDashboard",
    dataEndpoint: "/api/nurse/clinical/hospitalized",
    permissions: [
      "view_bed_map",
      "log_vitals",
      "request_physician",
      "manage_medications",
    ],
    controls: {
      canViewClinicalMode: true,
      canViewBedMap: true,
      canRequestPhysician: true,
      canManageMedications: true,
      canDischargePatient: false,
    },
    escalationStyle: "BATCHED",
    responseTimeTarget: 300,
  },

  // ========== PHYSICIAN (ER - Clinical Mode) ==========
  PHYSICIAN_ER: {
    component: "PhysicianClinicalDashboard_ER",
    dataEndpoint: "/api/physician/clinical/active-cases",
    permissions: [
      "view_active_cases",
      "receive_escalations",
      "assign_disposition",
      "sign_off_encounter",
    ],
    controls: {
      canViewClinicalMode: true,
      canReceiveEscalations: true,
      canAssignDisposition: true,    // DISCHARGE or ADMIT
      canCreateEHRBundle: true,
      canSignOff: true,
    },
    escalationStyle: "AGGRESSIVE",
    responseTimeTarget: 180,
  },

  // ========== PHYSICIAN (WARD - Rounds) ==========
  PHYSICIAN_WARD: {
    component: "WardRoundsPhysician",
    dataEndpoint: "/api/physician/ward-rounds/assigned-patients",
    permissions: [
      "view_assigned_patients",
      "review_vitals",
      "sign_off_encounter",
      "order_discharge",
    ],
    controls: {
      canViewClinicalMode: true,
      canDisposePatients: true,
      canEditTreatmentPlan: true,
    },
    escalationStyle: "CONTEXTUAL",
    responseTimeTarget: 600,
  },
};

/**
 * DETERMINE DASHBOARD FOR USER
 * Input: userId + hospitalId
 * Output: { component, dataEndpoint, controls }
 */
export async function getDashboardForUser(userId, hospitalId) {
  // 1. Find active shift for user on this date
  const todayShift = await Shift.findOne({
    hospitalId,
    $or: [
      { triageNurseId: userId },
      { clinicalNurseId: userId },
      { physicianId: userId },
    ],
    date: {
      $gte: new Date().setHours(0, 0, 0, 0),
      $lt: new Date().setHours(23, 59, 59, 999),
    },
  });

  if (!todayShift) {
    return null; // No active shift
  }

  // 2. Determine user's role in this shift
  let userContext;
  if (todayShift.triageNurseId?.toString() === userId.toString()) {
    userContext = "TRIAGE_NURSE";
  } else if (todayShift.clinicalNurseId?.toString() === userId.toString()) {
    userContext =
      todayShift.clinicalNurseContext === "ER"
        ? "CLINICAL_NURSE_ER"
        : "CLINICAL_NURSE_WARD";
  } else if (todayShift.physicianId?.toString() === userId.toString()) {
    userContext =
      todayShift.physicianContext === "ER"
        ? "PHYSICIAN_ER"
        : "PHYSICIAN_WARD";
  }

  // 3. Look up dashboard config
  const config = DASHBOARD_ROUTES[userContext];

  return {
    context: userContext,
    shift: todayShift,
    ...config,
  };
}
```

---

### Layer 3: Backend APIs

#### 3a. Assign Clinical Team (Triage Nurse Action)

**File**: `backend/src/routes/shift.route.js`

```javascript
/**
 * POST /api/shift/assign-clinical-team
 * 
 * Triage Nurse action: picks Clinical Nurse + Physician for this shift
 * 
 * Input:
 * {
 *   trageNurseId: string,
 *   clinicalNurseId: string,
 *   clinicalNurseContext: "ER" | "WARD",
 *   physicianId: string,
 *   physicianContext: "ER" | "WARD_ROUNDS"
 * }
 */
router.post("/assign-clinical-team", protect, async (req, res) => {
  const {
    shiftId,
    clinicalNurseId,
    clinicalNurseContext,
    physicianId,
    physicianContext,
    physicianSpecialty,
  } = req.body;
  const triageNurseId = req.user._id;

  try {
    // Verify triage nurse is authorized
    const shift = await Shift.findById(shiftId);
    if (shift.triageNurseId.toString() !== triageNurseId.toString()) {
      return res.status(403).json({ error: "Only triage nurse can assign team" });
    }

    // Verify availability
    const clinicalNurse = await Nurse.findById(clinicalNurseId);
    const physician = await Nurse.findById(physicianId);

    if (!clinicalNurse || !physician) {
      return res.status(404).json({ error: "Staff not found" });
    }

    // Update shift with assignments
    shift.assignedPersonnel = {
      trageNurseId,
      triageNurseName: req.user.firstName,
      clinicalNurseId,
      clinicalNurseName: `${clinicalNurse.firstName} ${clinicalNurse.lastName}`,
      clinicalNurseContext,
      physicianId,
      physicianName: `${physician.firstName} ${physician.lastName}`,
      physicianSpecialty,
      physicianContext,
    };

    shift.teamAssignedAt = new Date();
    shift.teamAssignedBy = triageNurseId;
    shift.status = "ACTIVE";

    await shift.save();

    // 🔔 NOTIFY assigned staff via WebSocket
    const io = req.app.get("io");
    io.to(clinicalNurseId.toString()).emit("shift_assigned", {
      context: clinicalNurseContext,
      dashboard: "CLINICAL_NURSE_ER",
      shift,
    });

    io.to(physicianId.toString()).emit("shift_assigned", {
      context: physicianContext,
      dashboard: "PHYSICIAN_ER",
      shift,
    });

    res.json({
      status: "TEAM_ASSIGNED",
      shift,
      message: `${clinicalNurse.firstName} and ${physician.firstName} are now active for this shift`,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

#### 3b. Get Dashboard Config for User

**File**: `backend/src/routes/user.route.js`

```javascript
/**
 * GET /api/user/dashboard-config
 * Returns the dashboard to render based on user's current shift
 */
router.get("/dashboard-config", protect, async (req, res) => {
  const userId = req.user._id;
  const hospitalId = req.user.hospitalId;

  try {
    const dashboard = await getDashboardForUser(userId, hospitalId);

    if (!dashboard) {
      return res.status(404).json({
        error: "NO_ACTIVE_SHIFT",
        message: "No active shift for today. Please clock in first.",
      });
    }

    res.json(dashboard);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

#### 3c. Submit Triage Assessment (Triage Nurse Action)

**File**: `backend/src/routes/triage.route.js`

```javascript
/**
 * POST /api/triage/submit-assessment
 * 
 * Triage Nurse creates triage assessment
 * ⭐ MUST include: needsBed (boolean)
 * 
 * Input:
 * {
 *   patientId: string,
 *   encounterId: string,
 *   chiefComplaint: string,
 *   symptoms: [string],
 *   vitalSigns: { temp, hr, rr, bp, o2 },
 *   medicalHistory: string,
 *   triagePriority: "GREEN" | "YELLOW" | "ORANGE" | "RED",
 *   triageReasoning: string,
 *   needsBed: boolean,  ⭐ THE CRITICAL FIELD
 *   bedNeedReasoning: string
 * }
 */
router.post("/submit-assessment", protect, async (req, res) => {
  const triageNurseId = req.user._id;
  const hospitalId = req.user.hospitalId;

  const {
    patientId,
    encounterId,
    chiefComplaint,
    symptoms,
    vitalSigns,
    medicalHistory,
    medications,
    allergies,
    triagePriority,
    triageReasoning,
    needsBed,        // ⭐
    bedNeedReasoning, // ⭐
  } = req.body;

  // Validate required fields
  if (!needsBed === undefined) {
    return res.status(400).json({
      error: "needsBed field is required. Triage nurse must specify if patient needs bed.",
    });
  }

  try {
    // Create triage assessment
    const assessment = await TriageAssessment.create({
      patientId,
      encounterId,
      triageNurseId,
      hospitalId,
      chiefComplaint,
      symptoms,
      vitalSigns,
      medicalHistory,
      medications,
      allergies,
      triagePriority,
      triageReasoning,
      needsBed,        // ⭐ Store this crucial field
      bedNeedReasoning,
      triageStartedAt: new Date(),
      triageCompletedAt: new Date(),
      status: "AWAITING_CLINICAL_NURSE",
    });

    // Update patient journey state
    const patientJourney = await PatientJourney.findOneAndUpdate(
      { patientId },
      {
        currentState: "TRIAGING",
        "disposition.needsBed": needsBed, // Store for later physician reference
      }
    );

    // 🔔 NOTIFY Clinical Nurse that patient is ready for handoff
    const io = req.app.get("io");
    const shift = await Shift.findOne({
      triageNurseId,
      date: { $gte: new Date().setHours(0, 0, 0, 0) },
    });

    if (shift && shift.clinicalNurseId) {
      io.to(shift.clinicalNurseId.toString()).emit("patient_ready_for_clinical", {
        patientId,
        patientName: `${req.body.firstName} ${req.body.lastName}`,
        triagePriority,
        needsBed,  // ⭐ Tell clinical nurse what triage nurse decided
        bedNeedReasoning,
        assessment: assessment._id,
      });
    }

    res.json({
      status: "TRIAGED",
      assessment,
      message: `Patient triaged as ${triagePriority}, needs bed: ${needsBed}`,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

#### 3d. Get Available Staff for Assignment

**File**: `backend/src/routes/staff.route.js`

```javascript
/**
 * GET /api/staff/available-for-shift
 * Lists nurses and physicians available to be assigned to clinical team
 */
router.get("/available-for-shift", protect, async (req, res) => {
  const { role, context } = req.query; // role: "NURSE" | "PHYSICIAN", context: "ER" | "WARD"
  const hospitalId = req.user.hospitalId;

  try {
    const available = await Nurse.find({
      hospitalId,
      role: role === "PHYSICIAN" ? "PHYSICIAN" : "STAFF",
      isOnDuty: true,
      $expr: {
        // Exclude the user themselves
        $ne: ["$_id", req.user._id],
      },
    })
      .select("firstName lastName role specialty licenseNumber _id")
      .lean();

    res.json({
      available,
      count: available.length,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

---

### Layer 4: Frontend - Dashboard Router

**File**: `frontend/src/Dashboard/DashboardRouter.jsx`

```javascript
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { decodeJwt } from "../utils/jwt";

// Import all dashboard variants
import TriageNurseDashboard from "./Dashboards/TriageNurseDashboard";
import ClinicalNurseDashboard_ER from "./Dashboards/ClinicalNurseDashboard_ER";
import WardNurseDashboard from "./Dashboards/WardNurseDashboard";
import PhysicianClinicalDashboard_ER from "./Dashboards/PhysicianClinicalDashboard_ER";
import WardRoundsPhysician from "./Dashboards/WardRoundsPhysician";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * DashboardRouter: Determines which dashboard to show based on user's shift context
 * This is THE MAIN COMPONENT that determines user's entire experience
 */
const DashboardRouter = ({ token }) => {
  const [dashboardConfig, setDashboardConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchDashboardConfig();
  }, [token]);

  const fetchDashboardConfig = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/user/dashboard-config`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.status === 404) {
        const data = await res.json();
        if (data.error === "NO_ACTIVE_SHIFT") {
          setError("You have not clocked in yet. Please clock in to see your dashboard.");
          setLoading(false);
          return;
        }
      }

      if (!res.ok) throw new Error("Failed to fetch dashboard config");

      const config = await res.json();
      setDashboardConfig(config);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div style={{ textAlign: "center", padding: "20px" }}>Loading dashboard...</div>;
  }

  if (error) {
    return (
      <div style={{ textAlign: "center", padding: "20px", color: "#991b1b" }}>
        ⚠️ {error}
        <br />
        <button onClick={() => navigate("/clock-in-out")}>Go to Clock In</button>
      </div>
    );
  }

  // Render the appropriate dashboard based on context
  const renderDashboard = () => {
    const { context, shift, controls } = dashboardConfig;

    switch (context) {
      case "TRIAGE_NURSE":
        return (
          <TriageNurseDashboard
            token={token}
            shift={shift}
            controls={controls}
            onTeamAssigned={() => fetchDashboardConfig()} // Reload to see updated dashboard
          />
        );

      case "CLINICAL_NURSE_ER":
        return (
          <ClinicalNurseDashboard_ER
            token={token}
            shift={shift}
            controls={controls}
            nurseName={shift.assignedPersonnel.clinicalNurseName}
          />
        );

      case "CLINICAL_NURSE_WARD":
        return (
          <WardNurseDashboard
            token={token}
            shift={shift}
            controls={controls}
            wardName="Medical Ward A" // From shift assignment
          />
        );

      case "PHYSICIAN_ER":
        return (
          <PhysicianClinicalDashboard_ER
            token={token}
            shift={shift}
            controls={controls}
            physicianName={shift.assignedPersonnel.physicianName}
          />
        );

      case "PHYSICIAN_WARD":
        return (
          <WardRoundsPhysician
            token={token}
            shift={shift}
            controls={controls}
          />
        );

      default:
        return <div>Unknown dashboard context: {context}</div>;
    }
  };

  return renderDashboard();
};

export default DashboardRouter;
```

---

### Layer 5: Frontend - Triage Nurse Dashboard with Team Assignment

**File**: `frontend/src/Dashboard/Dashboards/TriageNurseDashboard.jsx`

```javascript
import React, { useState, useEffect } from "react";
import { FiUsers, FiFilter, FiRefreshCw } from "react-icons/fi";
import SidebarLayout from "../SidebarLayout";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const TriageNurseDashboard = ({ token, shift, onTeamAssigned }) => {
  const [queue, setQueue] = useState([]);
  const [availableStaff, setAvailableStaff] = useState({ nurses: [], physicians: [] });
  const [showAssignModal, setShowAssignModal] = useState(!shift.teamAssignedAt); // Show modal if team not yet assigned
  const [selectedNurse, setSelectedNurse] = useState(null);
  const [selectedPhysician, setSelectedPhysician] = useState(null);
  const [assigning, setAssigning] = useState(false);

  useEffect(() => {
    fetchQueue();
    if (showAssignModal) {
      fetchAvailableStaff();
    }
  }, []);

  const fetchQueue = async () => {
    const res = await fetch(`${BACKEND_URL}/api/nurse/triage/queue`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await res.json();
    setQueue(data.patients || []);
  };

  const fetchAvailableStaff = async () => {
    const [nursesRes, physiciansRes] = await Promise.all([
      fetch(`${BACKEND_URL}/api/staff/available-for-shift?role=NURSE&context=ER`, {
        headers: { Authorization: `Bearer ${token}` },
      }),
      fetch(`${BACKEND_URL}/api/staff/available-for-shift?role=PHYSICIAN&context=ER`, {
        headers: { Authorization: `Bearer ${token}` },
      }),
    ]);

    const nurses = await nursesRes.json();
    const physicians = await physiciansRes.json();

    setAvailableStaff({
      nurses: nurses.available,
      physicians: physicians.available,
    });
  };

  const handleAssignTeam = async () => {
    if (!selectedNurse || !selectedPhysician) {
      alert("Please select both a nurse and a physician");
      return;
    }

    setAssigning(true);

    try {
      const res = await fetch(`${BACKEND_URL}/api/shift/assign-clinical-team`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          shiftId: shift._id,
          clinicalNurseId: selectedNurse._id,
          clinicalNurseContext: "ER",
          physicianId: selectedPhysician._id,
          physicianContext: "ER",
          physicianSpecialty: selectedPhysician.specialty,
        }),
      });

      if (!res.ok) throw new Error("Assignment failed");

      setShowAssignModal(false);
      onTeamAssigned(); // Parent will reload dashboard config
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setAssigning(false);
    }
  };

  return (
    <SidebarLayout username={shift.assignedPersonnel?.triageNurseName}>
      {/* TEAM ASSIGNMENT MODAL */}
      {showAssignModal && (
        <div style={styles.modalOverlay}>
          <div style={styles.modal}>
            <h2>👥 Assign Clinical Team for Today</h2>
            <p>Pick a Clinical Nurse and Physician to manage patients in ER/Observation</p>

            <div style={{ marginBottom: 20 }}>
              <label style={styles.label}>Select Clinical Nurse</label>
              <select
                value={selectedNurse?._id || ""}
                onChange={(e) => {
                  const nurse = availableStaff.nurses.find(
                    (n) => n._id === e.target.value
                  );
                  setSelectedNurse(nurse);
                }}
                style={styles.select}
              >
                <option value="">-- Choose a nurse --</option>
                {availableStaff.nurses.map((nurse) => (
                  <option key={nurse._id} value={nurse._id}>
                    {nurse.firstName} {nurse.lastName} ({nurse.licenseNumber})
                  </option>
                ))}
              </select>
            </div>

            <div style={{ marginBottom: 20 }}>
              <label style={styles.label}>Select Physician (ER Specialty)</label>
              <select
                value={selectedPhysician?._id || ""}
                onChange={(e) => {
                  const physician = availableStaff.physicians.find(
                    (p) => p._id === e.target.value
                  );
                  setSelectedPhysician(physician);
                }}
                style={styles.select}
              >
                <option value="">-- Choose a physician --</option>
                {availableStaff.physicians
                  .filter((p) => p.specialty?.includes("ER") || p.specialty?.includes("Emergency"))
                  .map((physician) => (
                    <option key={physician._id} value={physician._id}>
                      Dr. {physician.firstName} {physician.lastName} ({physician.specialty})
                    </option>
                  ))}
              </select>
            </div>

            <button
              onClick={handleAssignTeam}
              disabled={!selectedNurse || !selectedPhysician || assigning}
              style={{
                ...styles.btn,
                backgroundColor: selectedNurse && selectedPhysician ? "#16a34a" : "#d1d5db",
              }}
            >
              {assigning ? "Assigning..." : "✓ Confirm Team Assignment"}
            </button>
          </div>
        </div>
      )}

      {/* MAIN TRIAGE QUEUE DASHBOARD */}
      <div style={styles.container}>
        <div style={styles.header}>
          <h1>🏥 TRIAGE QUEUE - Emergency Department</h1>
          <div style={styles.teamInfo}>
            {shift.assignedPersonnel?.clinicalNurseName && (
              <span style={styles.badge}>
                👩‍⚕️ Clinical: {shift.assignedPersonnel.clinicalNurseName}
              </span>
            )}
            {shift.assignedPersonnel?.physicianName && (
              <span style={styles.badge}>
                🩺 Physician: {shift.assignedPersonnel.physicianName}
              </span>
            )}
            <button onClick={() => setShowAssignModal(true)} style={styles.reassignBtn}>
              Change Team
            </button>
          </div>
        </div>

      <button
        onClick={() => setShowAssignModal(true)}
        style={styles.reassignBtn}
      >
        Change Team
      </button>
    </div>
  </div>

  {/* QUEUE LIST */}
  <div style={styles.queueContainer}>
    <h2 style={{ marginBottom: 12 }}>
      👥 Waiting for Triage ({queue.length})
    </h2>

    {queue.length === 0 ? (
      <div style={styles.emptyState}>No patients waiting</div>
    ) : (
      queue.map((patient, idx) => (
        <div key={patient._id} style={styles.queueItem}>
          <div style={styles.position}>#{idx + 1}</div>
          <div style={styles.patientInfo}>
            <div style={styles.name}>
              {patient.firstName} {patient.lastName}
            </div>
            <div style={styles.complaint}>{patient.chiefComplaint}</div>
            <div style={styles.waitTime}>
              ⏱️ Waiting: {getWaitingTime(patient.arrivedAt)}
            </div>
          </div>
          <button
            onClick={() => handleTriagePatient(patient)}
            style={styles.triageBtn}
          >
            📋 Triage
          </button>
        </div>
      ))
    )}
  </div>
</div>
```

**NEW: Triage Assessment Form**

**File**: `frontend/src/Dashboard/Components/TriageAssessmentForm.jsx`

```javascript
import React, { useState } from "react";
import { FiCheckCircle, FiX } from "react-icons/fi";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * TRIAGE ASSESSMENT FORM
 * Triage nurse fills this out for each patient
 * ⭐ MUST include: needsBed decision
 */
const TriageAssessmentForm = ({ token, patient, shiftId, onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    chiefComplaint: patient.chiefComplaint || "",
    symptoms: [],
    vitalSigns: {
      temperature: "",
      heartRate: "",
      respiratoryRate: "",
      systolicBP: "",
      diastolicBP: "",
      oxygenSaturation: "",
    },
    medicalHistory: "",
    medications: [],
    allergies: [],
    triagePriority: "YELLOW",
    triageReasoning: "",
    
    // ⭐ THE CRITICAL FIELD
    needsBed: null, // Must be set to true or false
    bedNeedReasoning: "",
  });

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleVitalChange = (field, value) => {
    setFormData({
      ...formData,
      vitalSigns: { ...formData.vitalSigns, [field]: value },
    });
  };

  const handleSubmit = async () => {
    // Validate that needsBed is set
    if (formData.needsBed === null) {
      setError("❌ You must specify whether patient NEEDS A BED (Yes/No)");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const res = await fetch(`${BACKEND_URL}/api/triage/submit-assessment`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          patientId: patient._id,
          shiftId,
          ...formData,
        }),
      });

      if (!res.ok) throw new Error("Assessment submission failed");

      const result = await res.json();
      onSubmit(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={styles.formContainer}>
      <h2>📋 TRIAGE ASSESSMENT - {patient.firstName} {patient.lastName}</h2>

      {error && (
        <div style={styles.errorBanner}>
          {error}
        </div>
      )}

      {/* SECTION 1: Vitals */}
      <div style={styles.section}>
        <h3>🌡️ Vital Signs</h3>
        <div style={styles.vitalGrid}>
          <input
            type="number"
            placeholder="Temp (°C)"
            value={formData.vitalSigns.temperature}
            onChange={(e) => handleVitalChange("temperature", e.target.value)}
            style={styles.input}
          />
          <input
            type="number"
            placeholder="HR (bpm)"
            value={formData.vitalSigns.heartRate}
            onChange={(e) => handleVitalChange("heartRate", e.target.value)}
            style={styles.input}
          />
          <input
            type="number"
            placeholder="Resp (breaths/min)"
            value={formData.vitalSigns.respiratoryRate}
            onChange={(e) => handleVitalChange("respiratoryRate", e.target.value)}
            style={styles.input}
          />
          <input
            type="number"
            placeholder="Systolic BP"
            value={formData.vitalSigns.systolicBP}
            onChange={(e) => handleVitalChange("systolicBP", e.target.value)}
            style={styles.input}
          />
          <input
            type="number"
            placeholder="Diastolic BP"
            value={formData.vitalSigns.diastolicBP}
            onChange={(e) => handleVitalChange("diastolicBP", e.target.value)}
            style={styles.input}
          />
          <input
            type="number"
            placeholder="O2 Sat (%)"
            value={formData.vitalSigns.oxygenSaturation}
            onChange={(e) => handleVitalChange("oxygenSaturation", e.target.value)}
            style={styles.input}
          />
        </div>
      </div>

      {/* SECTION 2: Triage Priority */}
      <div style={styles.section}>
        <h3>🎯 Triage Priority (SATS)</h3>
        <div style={styles.priorityButtons}>
          {["GREEN", "YELLOW", "ORANGE", "RED"].map((priority) => (
            <button
              key={priority}
              onClick={() => setFormData({ ...formData, triagePriority: priority })}
              style={{
                ...styles.priorityBtn,
                backgroundColor:
                  formData.triagePriority === priority
                    ? getPriorityColor(priority)
                    : "#e5e7eb",
                color: formData.triagePriority === priority ? "white" : "#6b7280",
              }}
            >
              {priority}
            </button>
          ))}
        </div>
        <textarea
          placeholder="Why this priority?"
          value={formData.triageReasoning}
          onChange={(e) => setFormData({ ...formData, triageReasoning: e.target.value })}
          style={styles.textarea}
        />
      </div>

      {/* SECTION 3: ⭐ NEEDS BED? (CRITICAL) */}
      <div style={{ ...styles.section, border: "3px solid #dc2626", backgroundColor: "#fff5f5" }}>
        <h3 style={{ color: "#dc2626" }}>⭐ CRITICAL DECISION: DOES THIS PATIENT NEED A BED?</h3>
        <p style={{ fontSize: 14, color: "#6b7280", marginBottom: 12 }}>
          This determines if patient will be admitted to hospital or discharged to community
        </p>

        <div style={styles.bedDecisionButtons}>
          <button
            onClick={() => setFormData({ ...formData, needsBed: true })}
            style={{
              ...styles.bedBtn,
              backgroundColor: formData.needsBed === true ? "#16a34a" : "#e5e7eb",
              color: formData.needsBed === true ? "white" : "#6b7280",
              border: formData.needsBed === true ? "3px solid #15803d" : "1px solid #d1d5db",
            }}
          >
            ✓ YES - Patient NEEDS BED (likely hospital admission)
          </button>

          <button
            onClick={() => setFormData({ ...formData, needsBed: false })}
            style={{
              ...styles.bedBtn,
              backgroundColor: formData.needsBed === false ? "#0284c7" : "#e5e7eb",
              color: formData.needsBed === false ? "white" : "#6b7280",
              border: formData.needsBed === false ? "3px solid #0c63e4" : "1px solid #d1d5db",
            }}
          >
            ✗ NO - Patient does NOT need bed (likely discharge to community)
          </button>
        </div>

        {formData.needsBed !== null && (
          <textarea
            placeholder="Why does/doesn't patient need a bed?"
            value={formData.bedNeedReasoning}
            onChange={(e) => setFormData({ ...formData, bedNeedReasoning: e.target.value })}
            style={styles.textarea}
          />
        )}
      </div>

      {/* SECTION 4: Medical History */}
      <div style={styles.section}>
        <h3>🏥 Medical History & Medications</h3>
        <textarea
          placeholder="Relevant medical history"
          value={formData.medicalHistory}
          onChange={(e) => setFormData({ ...formData, medicalHistory: e.target.value })}
          style={styles.textarea}
        />
        <textarea
          placeholder="Current medications (one per line)"
          value={formData.medications.join("\n")}
          onChange={(e) => setFormData({ ...formData, medications: e.target.value.split("\n") })}
          style={styles.textarea}
        />
        <textarea
          placeholder="Known allergies"
          value={formData.allergies.join("\n")}
          onChange={(e) => setFormData({ ...formData, allergies: e.target.value.split("\n") })}
          style={styles.textarea}
        />
      </div>

      {/* BUTTONS */}
      <div style={styles.actions}>
        <button onClick={onCancel} style={styles.cancelBtn}>
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={submitting || formData.needsBed === null}
          style={{
            ...styles.submitBtn,
            opacity: formData.needsBed === null || submitting ? 0.6 : 1,
          }}
        >
          {submitting ? "Submitting..." : "✓ Complete Triage & Handoff to Clinical Nurse"}
        </button>
      </div>
    </div>
  );
};

const styles = {
  formContainer: { padding: 24, maxWidth: 800, backgroundColor: "white", borderRadius: 8, border: "1px solid #e5e7eb" },
  section: { marginBottom: 24, paddingBottom: 16, borderBottom: "1px solid #e5e7eb" },
  vitalGrid: { display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12, marginBottom: 12 },
  input: { padding: 10, border: "1px solid #d1d5db", borderRadius: 6, fontSize: 14 },
  textarea: { width: "100%", minHeight: 100, padding: 10, border: "1px solid #d1d5db", borderRadius: 6, fontSize: 14, marginBottom: 12, fontFamily: "inherit" },
  priorityButtons: { display: "flex", gap: 8, marginBottom: 12 },
  priorityBtn: { flex: 1, padding: 12, border: "1px solid #d1d5db", borderRadius: 6, fontWeight: 600, cursor: "pointer" },
  bedDecisionButtons: { display: "flex", gap: 12, marginBottom: 12, flexDirection: "column" },
  bedBtn: { padding: 16, border: "2px solid #d1d5db", borderRadius: 6, fontWeight: 600, cursor: "pointer", fontSize: 14 },
  actions: { display: "flex", gap: 12, justifyContent: "flex-end" },
  submitBtn: { padding: 12, backgroundColor: "#16a34a", color: "white", border: "none", borderRadius: 6, fontWeight: 600, cursor: "pointer" },
  cancelBtn: { padding: 12, backgroundColor: "#e5e7eb", color: "#6b7280", border: "none", borderRadius: 6, fontWeight: 600, cursor: "pointer" },
  errorBanner: { padding: 12, backgroundColor: "#fee2e2", color: "#991b1b", borderRadius: 6, marginBottom: 12, border: "1px solid #fecaca" },
};

function getPriorityColor(priority) {
  switch (priority) {
    case "GREEN":
      return "#16a34a";
    case "YELLOW":
      return "#eab308";
    case "ORANGE":
      return "#f97316";
    case "RED":
      return "#dc2626";
    default:
      return "#6b7280";
  }
}

export default TriageAssessmentForm;

function getWaitingTime(arrivedAt) {
  const minutes = Math.floor((Date.now() - new Date(arrivedAt)) / 60000);
  return `${minutes}m`;
}

export default TriageNurseDashboard;
```

---

### Layer 6: Clinical Nurse Dashboard (ER) Stub

**File**: `frontend/src/Dashboard/Dashboards/ClinicalNurseDashboard_ER.jsx`

```javascript
import React, { useState, useEffect } from "react";
import SidebarLayout from "../SidebarLayout";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * CLINICAL NURSE DASHBOARD (ER)
 * Nurse assigned in clinical mode manages:
 * - Non-hospitalized patients in observation
 * - Vitals monitoring
 * - Escalations to physician
 */
const ClinicalNurseDashboard_ER = ({ token, shift, controls, nurseName }) => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchObservationPatients();
  }, []);

  const fetchObservationPatients = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/nurse/clinical/non-hospitalized`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setPatients(data.patients || []);
    } catch (error) {
      console.error("Error fetching patients:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SidebarLayout username={nurseName}>
      <div style={styles.container}>
        <h1>⚕️ CLINICAL MODE - Non-Hospitalized Patients</h1>
        <p style={styles.subtitle}>Managing observation & awaiting disposition</p>

        {loading ? (
          <p>Loading...</p>
        ) : (
          <div style={styles.patientGrid}>
            {patients.map((patient) => (
              <div key={patient._id} style={styles.patientCard}>
                <h3>{patient.firstName} {patient.lastName}</h3>
                <p>Bed: {patient.bedNumber}</p>
                <p>Status: {patient.currentState}</p>
                <button
                  style={styles.btn}
                  disabled={!controls.canRequestPhysician}
                >
                  Request Physician Review
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </SidebarLayout>
  );
};

const styles = {
  container: { padding: 24 },
  subtitle: { color: "#6b7280", marginBottom: 20 },
  patientGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16 },
  patientCard: { padding: 16, border: "1px solid #d1d5db", borderRadius: 8, backgroundColor: "#f9fafb" },
  btn: { width: "100%", marginTop: 12, padding: 10, backgroundColor: "#0284c7", color: "white", border: "none", borderRadius: 6, cursor: "pointer" },
};

export default ClinicalNurseDashboard_ER;
```

---

## PHYSICIAN DISPOSITION LOGIC (Updated with needsBed field)

**File**: `backend/src/controllers/disposition.controller.js`

```javascript
/**
 * POST /api/disposition
 * 
 * Physician decides patient outcome based on:
 * 1. Triage assessment (needsBed)
 * 2. Clinical observations (vitals, response to treatment)
 * 3. Medical decision
 */
export const handleDisposition = async (req, res) => {
  const { patientId, decision, wardIfAdmit, notes } = req.body;
  const physicianId = req.user._id;

  try {
    // Get triage assessment to check what nurse recommended
    const triageAssessment = await TriageAssessment.findOne({ patientId });
    const patientJourney = await PatientJourney.findOne({ patientId });

    if (!triageAssessment) {
      return res.status(404).json({ error: "Triage assessment not found" });
    }

    // Check for DISCREPANCY
    let discrepancy = null;
    if (decision === "DISCHARGED" && triageAssessment.needsBed === true) {
      // Nurse said needs bed, but physician discharged
      discrepancy = {
        hasDiscrepancy: true,
        triagePrediction: "NEEDS_BED",
        physicianDecision: "DISCHARGED",
        reason: notes || "Physician overrode triage assessment",
      };
    } else if (decision === "ADMITTED" && triageAssessment.needsBed === false) {
      // Nurse said doesn't need bed, but physician admitted
      discrepancy = {
        hasDiscrepancy: true,
        triagePrediction: "NO_BED_NEEDED",
        physicianDecision: "ADMITTED",
        reason: notes || "Condition deteriorated after triage",
      };
    }

    // Update triage assessment with physician decision
    triageAssessment.physicianDecision = {
      status: decision,
      decidedAt: new Date(),
      decidedBy: physicianId,
      discrepancy,
    };
    await triageAssessment.save();

    // Update patient journey state
    if (decision === "DISCHARGED") {
      patientJourney.currentState = "DISCHARGED";
      patientJourney.disposition.status = "DISCHARGED";
      
      // Create AMBULATORY EHR bundle
      const bundle = await EHRBundleFactory.createBundle(
        patientJourney.encounterId,
        "AMBULATORY"
      );

      // Queue for EHR sync
      await EHRSyncQueue.create({
        bundleId: bundle._id,
        bundleType: "AMBULATORY",
        targetEHR: "Med-e-Mass",
      });

      // Record discrepancy if exists
      if (discrepancy?.hasDiscrepancy) {
        await TriageDiscrepancyLog.create({
          triageAssessmentId: triageAssessment._id,
          discrepancy,
          resolutionType: "OVERRIDE", // Physician override noted
        });
      }

    } else if (decision === "ADMITTED") {
      patientJourney.currentState = "HOSPITALIZED";
      patientJourney.disposition.status = "ADMITTED";
      patientJourney.location.type = "WARD_BED";
      patientJourney.location.bedNumber = wardIfAdmit.bedNumber;
      patientJourney.location.wardName = wardIfAdmit.wardName;

      // Create INPATIENT EHR bundle (NOT finalized, stays open for ongoing care)
      const bundle = await EHRBundleFactory.createBundle(
        patientJourney.encounterId,
        "INPATIENT",
        {
          admittingWard: wardIfAdmit.wardName,
          bedNumber: wardIfAdmit.bedNumber,
          admissionReason: triageAssessment.triageReasoning,
          triageNeedsAssessment: triageAssessment.needsBed,
        }
      );

      // Notify ward nurse
      const wardNurses = await Nurse.find({
        hospitalId: req.user.hospitalId,
        assignedWard: wardIfAdmit.wardName,
        isOnDuty: true,
      });

      const io = req.app.get("io");
      for (const nurse of wardNurses) {
        io.to(nurse._id.toString()).emit("new_patient_admitted", {
          patientId,
          bedNumber: wardIfAdmit.bedNumber,
          triageAssessment: {
            priority: triageAssessment.triagePriority,
            reasoning: triageAssessment.triageReasoning,
            needsBed: triageAssessment.needsBed,
          },
        });
      }

      // Record discrepancy if exists
      if (discrepancy?.hasDiscrepancy) {
        await TriageDiscrepancyLog.create({
          triageAssessmentId: triageAssessment._id,
          discrepancy,
          resolutionType: "DETERIORATION", // Condition worsened
        });
      }
    }

    await patientJourney.save();

    res.json({
      status: "DISPOSITION_RECORDED",
      decision,
      discrepancy: discrepancy || null,
      message: discrepancy
        ? `⚠️ Discrepancy noted: Triage said ${discrepancy.triagePrediction}, but physician decided ${decision}`
        : `✓ Patient ${decision.toLowerCase()}`,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```

---

## IMPLEMENTATION ROADMAP (UPDATED)

### Sprint 1: Foundation (This Week)
- [ ] Create/enhance `Shift.model.js` with team assignment
- [ ] **[NEW] Create `TriageAssessment.model.js` with needsBed field**
- [ ] Build `dashboardRouter.config.js`
- [ ] Create `POST /api/shift/assign-clinical-team`
- [ ] Create `GET /api/user/dashboard-config`
- [ ] Build `DashboardRouter.jsx` component

### Sprint 2: Nurse Dashboard (Next Week)
- [ ] Build `TriageNurseDashboard.jsx` (team assignment modal)
- [ ] **[NEW] Build `TriageAssessmentForm.jsx` (WITH needsBed decision UI)**
- [ ] Create `POST /api/triage/submit-assessment` endpoint
- [ ] Build `ClinicalNurseDashboard_ER.jsx` (observation patients)
- [ ] Build `WardNurseDashboard.jsx` (bed grid)
- [ ] Create corresponding backend endpoints

### Sprint 3: Physician Dashboard (Week 3)
- [ ] Build `PhysicianClinicalDashboard_ER.jsx` (active cases)
- [ ] **[ENHANCED] Add disposition control with needsBed tracking**
- [ ] Create `POST /api/disposition` endpoint with discrepancy logic
- [ ] **[NEW] Create `TriageDiscrepancyLog.model.js` for analytics**

### Sprint 4: EHR Integration (Week 4)
- [ ] Integrate `EHRBundleFactory` with disposition workflow
- [ ] Create EHR sync on disposition decision
- [ ] **[NEW] Dashboard showing triage discrepancies for QA review**

---

## TESTING SCENARIOS

### Scenario 1: Triage Nurse Assigns Team
1. Alice (Triage) clocks in → sees modal
2. Picks Bob (Clinical Nurse) + Dr. Chen (Physician)
3. ✅ Bob and Dr. Chen's dashboards change to CLINICAL MODE

### Scenario 2: Patient Handoff
1. Alice triages John → YELLOW → hands to Bob
2. Bob enters CLINICAL MODE dashboard
3. Bob logs vitals, requests Dr. Chen
4. John appears in Dr. Chen's physician dashboard

### Scenario 3: Disposition
1. Dr. Chen reviews John → decides DISCHARGE
2. EHR bundle created (AMBULATORY)
3. John removed from Bob's dashboard
4. Metrics recorded

---

**Which model/API should we code first?** I recommend starting with:
1. Enhance `Shift.model.js` 
2. Build `dashboardRouter.config.js`
3. Then the backend APIs

This unlocks the **entire routing system**. Ready?
