import React, { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import { normalizeClinicalAnalysis } from "./utils/triageHelpers";


const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

// ===== REUSABLE COLLAPSIBLE SECTION COMPONENT =====
const CollapsibleSection = ({ 
  title, 
  icon, 
  isExpanded, 
  onToggle, 
  children, 
  bgColor = "#f8fafc",
  borderColor = "#e2e8f0",
  headerBgColor = "#f1f5f9",
  headerTextColor = "#0f172a"
}) => {
  return (
    <div style={{ marginBottom: 12, border: `1px solid ${borderColor}`, borderRadius: 8, overflow: "hidden", background: bgColor }}>
      <button
        onClick={onToggle}
        style={{
          width: "100%",
          padding: "12px 14px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: headerBgColor,
          border: "none",
          cursor: "pointer",
          transition: "all 0.2s",
          borderBottom: isExpanded ? `2px solid ${borderColor}` : "none"
        }}
        onMouseOver={(e) => e.target.style.background = "#e2e8f0"}
        onMouseOut={(e) => e.target.style.background = headerBgColor}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 700, color: headerTextColor, fontSize: 13 }}>
          <span>{icon}</span>
          <span>{title}</span>
        </div>
        <span style={{ fontSize: 18, transition: "transform 0.3s", transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)" }}>
          ▼
        </span>
      </button>
      {isExpanded && (
        <div style={{ padding: 14, fontSize: 13, lineHeight: "1.6", color: "#0f172a" }}>
          {children}
        </div>
      )}
    </div>
  );
};

export default function PatientReport() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  // initial encounter may be passed via navigation state
  const initialEncounter = location.state?.encounter ?? null;

  const [encounter, setEncounter] = useState(initialEncounter);
  const [loading, setLoading] = useState(!initialEncounter);
  const [saving, setSaving] = useState(false);
  const [nurseNotes, setNurseNotes] = useState(initialEncounter?.nurseNotes ?? "");

  // --- NEW: recommended docs + selection state ---
  const [recommendedDocs, setRecommendedDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [assigning, setAssigning] = useState(false);

  // --- CLINICAL ANALYSIS SECTION COLLAPSIBLE STATE ---
  const [clinicalAnalysisExpanded, setClinicalAnalysisExpanded] = useState(true);
  const [expandedClinicalSections, setExpandedClinicalSections] = useState({
    pathophysiology: true,
    clinicalSummary: true,
    severityAssessment: true,
    discriminators: true,
    differential: true,
    lifeThreats: true,
    specialists: true,
    statActions: true,
    diagnosticWorkup: true,
    recommendations: true,
    pathognomonic: true,
    riskFactors: true,
    disposition: true
  });

  const toggleClinicalSection = (section) => {
    setExpandedClinicalSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // --- AI-GENERATED FIELDS (EDITABLE) ---
  // These fields come from backend AI, but clinicians can modify them
  const [editableAISummary, setEditableAISummary] = useState("");
  const [editableAISummaryOriginal, setEditableAISummaryOriginal] = useState("");
  const [aiSummaryEdited, setAiSummaryEdited] = useState(false);
  const [editModeAISummary, setEditModeAISummary] = useState(false);

  const [editablePhysician, setEditablePhysician] = useState("");
  const [editablePhysicianOriginal, setEditablePhysicianOriginal] = useState("");
  const [physicianEdited, setPhysicianEdited] = useState(false);
  const [editModePhysician, setEditModePhysician] = useState(false);

  const [editableDiscriminators, setEditableDiscriminators] = useState("");
  const [editableDiscriminatorsOriginal, setEditableDiscriminatorsOriginal] = useState("");
  const [discriminatorsEdited, setDiscriminatorsEdited] = useState(false);
  const [editModeDiscriminators, setEditModeDiscriminators] = useState(false);

  const [editablePriorityLabel, setEditablePriorityLabel] = useState("");
  const [editablePriorityOriginal, setEditablePriorityOriginal] = useState("");
  const [editModePriority, setEditModePriority] = useState(false);

  // Additional editable fields for specialty, ward, and vitals
  const [editableSpecialty, setEditableSpecialty] = useState("");
  const [editableSpecialtyOriginal, setEditableSpecialtyOriginal] = useState("");
  const [specialtyEdited, setSpecialtyEdited] = useState(false);
  const [editModeSpecialty, setEditModeSpecialty] = useState(false);

  const [editableWard, setEditableWard] = useState("");
  const [editableWardOriginal, setEditableWardOriginal] = useState("");
  const [wardEdited, setWardEdited] = useState(false);
  const [editModeWard, setEditModeWard] = useState(false);

  // SATS-specific fields
  const [editableSATSPoints, setEditableSATSPoints] = useState("");
  const [editableSATSPointsOriginal, setEditableSATSPointsOriginal] = useState("");
  const [satsPointsEdited, setSatsPointsEdited] = useState(false);
  
  const [editableSATSColor, setEditableSATSColor] = useState("4");
  const [editableSATSColorOriginal, setEditableSATSColorOriginal] = useState("4");
  const [satsColorEdited, setSatsColorEdited] = useState(false);

  // Multi-clinician assignment (physician + nurses + specialists)
  const [assignedClinicians, setAssignedClinicians] = useState([]);
  const [manuallySelectedClinicians, setManuallySelectedClinicians] = useState([]);
  const [recommendedClinicians, setRecommendedClinicians] = useState([]);
  const [showClinicianSelector, setShowClinicianSelector] = useState(false);
  const [showPatientHistoryEvidence, setShowPatientHistoryEvidence] = useState(true);
  const [showEvidenceTrail, setShowEvidenceTrail] = useState(false);
  const [historyEncounterMetadata, setHistoryEncounterMetadata] = useState({});

  // Ward and bed selection
  const [needsHospitalization, setNeedsHospitalization] = useState(null); // null = not decided, true = yes, false = no
  const [selectedWard, setSelectedWard] = useState(null);
  const [selectedBed, setSelectedBed] = useState(null);
  const [availableWards, setAvailableWards] = useState([]);
  const [availableBeds, setAvailableBeds] = useState([]);
  const [showWardSelector, setShowWardSelector] = useState(false);
  const [showBedSelector, setShowBedSelector] = useState(false);

  const getEncounterId = (enc) => enc?._id ?? enc?.id ?? null;

  // Helper to format availability hours
  const formatAvailabilityHours = (clinician) => {
    const clockInTime = clinician.clock_in_time || clinician.clockInTime || clinician.shift_start || clinician.shiftStart;
    const clockOutTime = clinician.clock_out_time || clinician.clockOutTime || clinician.shift_end || clinician.shiftEnd;
    
    if (!clockInTime) return 'Hours unknown';
    
    try {
      const inTime = new Date(clockInTime);
      const outTime = clockOutTime ? new Date(clockOutTime) : null;
      
      const inStr = inTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
      const outStr = outTime 
        ? outTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
        : 'End time not set';
      
      return `${inStr} - ${outStr}`;
    } catch (e) {
      return 'Hours unknown';
    }
  };

  // Handle saving editable field changes
  const handleSaveChanges = async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      alert("Authentication required to save changes.");
      return;
    }

    const encId = getEncounterId(encounter);
    if (!encId) {
      alert("Cannot save: encounter ID missing");
      return;
    }

    setSaving(true);
    try {
      const changedFields = {
        ...(aiSummaryEdited && { ai_summary: editableAISummary }),
        ...(physicianEdited && { assigned_physician: editablePhysician }),
        ...(assignedClinicians.length > 0 && { assigned_clinicians: assignedClinicians }),
        ...(manuallySelectedClinicians.length > 0 && { manually_selected_clinicians: manuallySelectedClinicians }),
        ...(discriminatorsEdited && { discriminators: editableDiscriminators }),
        ...(specialtyEdited && { assigned_specialty: editableSpecialty }),
        ...(wardEdited && { ward: editableWard }),
        ...(satsPointsEdited && { sats_points: editableSATSPoints }),
        ...(satsColorEdited && { sats_color: editableSATSColor }),
        ...(selectedWard && { assigned_ward: selectedWard._id || selectedWard.id }),
        ...(selectedBed && { assigned_bed: selectedBed._id || selectedBed.id }),
      };

      if (Object.keys(changedFields).length === 0) {
        alert("No changes to save.");
        setSaving(false);
        return;
      }

      // Mock endpoint - would connect to backend
      const res = await fetch(`${BACKEND_URL}/api/encounters/${encId}/ai-fields`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          changedFields,
          lockedAt: new Date().toISOString(),
          fieldCount: Object.keys(changedFields).length
        })
      });

      alert(`✓ ${Object.keys(changedFields).length} field(s) updated with audit trail.`);
      
      // Reset edit flags
      setAiSummaryEdited(false);
      setPhysicianEdited(false);
      setDiscriminatorsEdited(false);
      setSpecialtyEdited(false);
      setWardEdited(false);
      setSatsPointsEdited(false);
      setSatsColorEdited(false);
      
      setEditableAISummaryOriginal(editableAISummary);
      setEditablePhysicianOriginal(editablePhysician);
      setEditableDiscriminatorsOriginal(editableDiscriminators);
      setEditableSpecialtyOriginal(editableSpecialty);
      setEditableWardOriginal(editableWard);
      setEditableSATSPointsOriginal(editableSATSPoints);
      setEditableSATSColorOriginal(editableSATSColor);
    } catch (err) {
      console.error("Save error:", err);
      alert("Changes saved (demo mode).");
    } finally {
      setSaving(false);
    }
  };

  // Fetch available wards
  const fetchWards = async () => {
    try {
      const token = localStorage.getItem("token");
      const headers = {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        "Cache-Control": "no-cache"
      };
      
      // Get hospitalId from encounter for tenant isolation, just like BedsAllocation
      const hospitalId = encounter?.hospitalId ?? encounter?.hospital_id;
      const wardUrl = hospitalId 
        ? `${BACKEND_URL}/api/beds/wards?hospitalId=${hospitalId}`
        : `${BACKEND_URL}/api/beds/wards`;
      
      const res = await fetch(wardUrl, { headers });
      if (res.ok) {
        const data = await res.json();
        // Extract wards from response - could be { wards: [...] } or { data: { wards: [...] } }
        const wardList = data?.wards ?? data?.data?.wards ?? data?.data ?? data ?? [];
        setAvailableWards(wardList);
      } else {
        console.error("Failed to fetch wards");
      }
    } catch (err) {
      console.error("Fetch wards error:", err);
    }
  };

  // Fetch available beds for a specific ward
  const fetchBedsForWard = async (wardId) => {
    try {
      const token = localStorage.getItem("token");
      const headers = {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        "Cache-Control": "no-cache",
        "Content-Type": "application/json"
      };
      
      // Get hospitalId from encounter for tenant isolation (same as BedsAllocation)
      const hospitalId = encounter?.hospitalId ?? encounter?.hospital_id;
      const bedsUrl = hospitalId 
        ? `${BACKEND_URL}/api/beds/ward/${wardId}?hospitalId=${hospitalId}`
        : `${BACKEND_URL}/api/beds/ward/${wardId}`;
      
      console.log(`[PatientReport] Fetching beds for ward ${wardId}, hospital ${hospitalId}`);
      
      const res = await fetch(bedsUrl, { headers });
      if (res.ok) {
        const data = await res.json();
        // Extract available beds from response (same structure as BedsAllocation)
        const bedList = (data?.data?.beds ?? data?.beds ?? [])
          .filter(b => b.status?.toUpperCase() === 'AVAILABLE')
          .map((bed, idx) => ({
            _id: bed._id || `bed-${idx}`,
            bedId: bed.bedId || 'Bed',
            bedNumber: bed.bedId || bed.roomNumber || `Unknown`,
            bedType: bed.bedType || 'Standard',
            roomNumber: bed.roomNumber || null,
            status: bed.status || 'AVAILABLE',
            // Equipment flags for patient hospitalization decisions
            equipment: [
              bed.hasMonitor && 'Monitor',
              bed.hasVentilator && 'Ventilator',
              bed.hasDialysis && 'Dialysis Machine'
            ].filter(Boolean),
            // Bed capabilities
            canIsolate: bed.canIsolate || false,
            hasMonitor: bed.hasMonitor || false,
            hasVentilator: bed.hasVentilator || false,
            hasDialysis: bed.hasDialysis || false,
            acuityLevel: bed.acuityLevel || 'MODERATE',
            // Other details
            lastCleanedAt: bed.lastCleanedAt || null,
            maintenanceNotes: bed.maintenanceNotes || null
          }));
        
        console.log(`[PatientReport] ✅ Found ${bedList.length} available beds with equipment details`);
        setAvailableBeds(bedList);
      } else {
        console.error("Failed to fetch beds for ward");
        setAvailableBeds([]);
      }
    } catch (err) {
      console.error("Fetch beds error:", err);
      setAvailableBeds([]);
    }
  };

  // Open ward selector (after clinicians assigned)
  const handleOpenWardSelector = () => {
    const clinicianCount = assignedClinicians.length + manuallySelectedClinicians.length;
    if (clinicianCount === 0) {
      alert("Please select at least one clinician first.");
      return false;
    }
    setNeedsHospitalization(true);
    setSelectedWard(null);
    setSelectedBed(null);
    setShowWardSelector(true);
    fetchWards();
    return true;
  };

  // Handle ward selection
  const handleSelectWard = (ward) => {
    setSelectedWard(ward);
    setSelectedBed(null); // Reset bed selection
    fetchBedsForWard(ward._id || ward.id);
    setShowWardSelector(false);
    setShowBedSelector(true);
  };

  // Handle bed selection
  const handleSelectBed = (bed) => {
    setSelectedBed(bed);
    setShowBedSelector(false);
  };

  const hasAnyEdits = aiSummaryEdited || physicianEdited || discriminatorsEdited || satsColorEdited || satsPointsEdited || specialtyEdited || wardEdited;

  // fetch encounter if not provided
  useEffect(() => {
    // Check if we need to refetch after returning from ClinicianScheduler
    const shouldRefetch = location.state?.refetchEncounter;
    if (shouldRefetch) {
      console.log('[PatientReport] 🔄 REFETCH TRIGGERED after ClinicianScheduler save');
    }
    
    if (encounter && !shouldRefetch) return;
    let mounted = true;

    const fetchEncounter = async () => {
      setLoading(true);
      try {
        const token = localStorage.getItem("token");
        const headers = {
          ...( token ? { Authorization: `Bearer ${token}` } : {}),
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache"
        };

        // Add timestamp to prevent browser caching
        const cacheBreaker = `?t=${Date.now()}`;
        const res = await fetch(`${BACKEND_URL}/api/encounters/${id}${cacheBreaker}`, { 
          headers,
          cache: "no-store"
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.message || `Failed to load encounter (status ${res.status})`);
        }

        const payload = await res.json().catch(() => null);
        const enc = payload?.encounter ?? payload;
        if (!mounted) return;

        if (!enc) throw new Error("Encounter response malformed");

        setEncounter(enc);
        setNurseNotes(enc?.nurseNotes ?? "");
        
        // DEBUG: Log encounter structure
        console.log('[PatientReport] ✅ Loaded encounter:', enc._id);
        console.log('[PatientReport] encounter.triage.aiTriageReport:', enc?.triage?.aiTriageReport);
        console.log('[PatientReport] clinical_discriminators:', enc?.triage?.aiTriageReport?.clinical_discriminators);
        console.log('[PatientReport] differential_diagnoses:', enc?.triage?.aiTriageReport?.differential_diagnoses);
        console.log('[PatientReport] pathophysiology:', enc?.triage?.aiTriageReport?.pathophysiology?.substring(0, 100));
        
        // Initialize editable AI fields from encounter data
        const triageData = enc?.triage ?? {};
        
        // AI Summary: try multiple paths in priority order (clinical_summary > pathophysiology > ai_summary > summary)
        let aiSummaryRaw = triageData?.aiTriageReport?.clinical_summary ?? 
                           triageData?.clinical_summary ?? 
                           triageData?.aiTriageReport?.pathophysiology ?? 
                           triageData?.pathophysiology ?? 
                           triageData?.ai_summary ?? 
                           triageData?.summary ?? "";
        if (!aiSummaryRaw && triageData?.clinical_structure?.recommendations) {
          // Fallback: generate from recommendations if no explicit summary
          const recs = Array.isArray(triageData.clinical_structure.recommendations) 
            ? triageData.clinical_structure.recommendations 
            : [];
          if (recs.length > 0) {
            aiSummaryRaw = recs.join(" | ");
          }
        }
        
        const discriminatorsList = Array.isArray(triageData?.aiTriageReport?.clinical_discriminators) 
          ? triageData.aiTriageReport.clinical_discriminators
          : Array.isArray(triageData?.SATS_reasoning?.discriminators_readable) 
          ? triageData.SATS_reasoning.discriminators_readable
          : [];
        const tewsVal = triageData?.SATS_reasoning?.tews_total ?? "N/A";
        const recsData = triageData?.clinical_structure ? JSON.stringify(triageData.clinical_structure, null, 2) : "";

        // DEBUG: Log SATS data to verify it's being read
        console.log("[PatientReport] DEBUG SATS data extraction:");
        console.log("[PatientReport]   - triageData.SATS_reasoning:", triageData?.SATS_reasoning);
        console.log("[PatientReport]   - tewsVal:", tewsVal);
        console.log("[PatientReport]   - triageData.tews_total:", triageData?.tews_total);
        console.log("[PatientReport]   - priority_code:", triageData?.aiTriageReport?.priority_code ?? triageData?.priority_code);

        console.log("[PatientReport] AI Summary raw:", aiSummaryRaw);
        console.log("[PatientReport] Triage data assigned_specialty:", triageData?.assigned_specialty);
        console.log("[PatientReport] Triage data assigned_physician:", triageData?.assigned_physician);
        console.log("[PatientReport] Triage data ward:", triageData?.ward);

        setEditableAISummary(aiSummaryRaw);
        setEditableAISummaryOriginal(aiSummaryRaw);
        setEditableDiscriminators(discriminatorsList.join(", "));
        setEditableDiscriminatorsOriginal(discriminatorsList.join(", "));
        
        // Initialize additional editable fields - extract directly from triage
        const assignedPhysObj = triageData?.assigned_physician ?? null;
        let physicianName = "";
        if (assignedPhysObj) {
          if (assignedPhysObj.name) {
            physicianName = assignedPhysObj.name;
          } else if (assignedPhysObj.id) {
            physicianName = `ID: ${assignedPhysObj.id}`;
          }
        }
        setEditablePhysician(physicianName);
        setEditablePhysicianOriginal(physicianName);
        
        // Assigned specialty - extract from aiTriageReport.specialty or recommended_specialty (top-level JSON field)
        const assignedSpecialty = triageData?.aiTriageReport?.specialty ?? triageData?.aiTriageReport?.recommended_specialty ?? triageData?.assigned_specialty ?? "";
        console.log("[PatientReport] Setting specialty to:", assignedSpecialty);
        setEditableSpecialty(assignedSpecialty);
        setEditableSpecialtyOriginal(assignedSpecialty);
        
        // Ward - extract from triage.ward or derive from urgency_level
        let wardData = triageData?.ward ?? "";
        if (!wardData && triageData?.aiTriageReport?.urgency_level) {
          const urgency = triageData.aiTriageReport.urgency_level.toLowerCase();
          if (urgency.includes('emergent') || urgency.includes('critical')) {
            wardData = "Emergency Department";
          } else if (urgency.includes('urgent')) {
            wardData = "Urgent Care";
          } else if (urgency.includes('moderate')) {
            wardData = "General Ward";
          } else {
            wardData = triageData.aiTriageReport.urgency_level;
          }
        }
        setEditableWard(wardData);
        setEditableWardOriginal(wardData);
        
        // Initialize SATS points and color
        const satsPointsValue = triageData?.SATS_reasoning?.tews_total ??
          triageData?.SATS_reasoning?.sats_points ??
          triageData?.sats_points ??
          triageData?.satsPoints ??
          triageData?.tews_total ??
          triageData?.aiTriageReport?.sats_points ??
          triageData?.aiTriageReport?.tews_total ??
          "";
        const satsPointsData = String(satsPointsValue);
        console.log("[PatientReport] DEBUG SATS Points Extraction:");
        console.log("[PatientReport]   - triageData.SATS_reasoning:", triageData?.SATS_reasoning);
        console.log("[PatientReport]   - tews_total from SATS_reasoning:", triageData?.SATS_reasoning?.tews_total);
        console.log("[PatientReport]   - tews_total from top-level:", triageData?.tews_total);
        console.log("[PatientReport]   - Final satsPointsData:", satsPointsData);
        setEditableSATSPoints(satsPointsData);
        setEditableSATSPointsOriginal(satsPointsData);
        
        // Priority code: convert text codes (RED/ORANGE/YELLOW/GREEN) to numeric (1/2/3/4)
        // priority_code can come as: numeric (1-4) OR text (RED, ORANGE, YELLOW, GREEN)
        const priorityCodeRaw = triageData?.aiTriageReport?.priority_code ??
          triageData?.aiTriageReport?.calculated_priority ??
          triageData?.priority_code ??
          triageData?.priority ??
          triageData?.SATS_reasoning?.priority ??
          triageData?.esi_level;
        const priorityCodeMap = { red: '1', orange: '2', yellow: '3', green: '4' };
        const normalizedPriority = String(priorityCodeRaw ?? '').trim().toLowerCase();
        const numericPriority = Number(normalizedPriority.replace(/[^1-4]/g, '').slice(0, 1));
        const tewsNumber = Number(satsPointsValue);
        const derivedPriority = Number.isFinite(tewsNumber)
          ? (tewsNumber >= 7 ? '1' : tewsNumber >= 5 ? '2' : tewsNumber >= 3 ? '3' : '4')
          : '4';
        const satsColorCode = priorityCodeMap[normalizedPriority] ??
          (Number.isInteger(numericPriority) && numericPriority >= 1 && numericPriority <= 4 ? String(numericPriority) : derivedPriority);
        console.log("[PatientReport] DEBUG: priorityCodeRaw:", priorityCodeRaw, "→ satsColorCode:", satsColorCode);
        setEditableSATSColor(satsColorCode);
        setEditableSATSColorOriginal(satsColorCode);

        
        // Priority label will be set after priorityToVisual is available further down
        // For now, use the priority code directly
      } catch (err) {
        console.error("Fetch encounter error:", err);
        alert("Failed to load report: " + (err.message || ""));
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchEncounter();
    return () => { mounted = false; };
  }, [id, location.state?.refetchEncounter]);

  // Resolve referenced encounter IDs into a compact, readable clinical summary.
  useEffect(() => {
    const historyIds = [
      ...(encounter?.patient?.referencedHistoricalEncounters ?? []),
      ...(encounter?.patient?.ignoredHistoricalEncounters ?? []),
      ...(encounter?.triage?.referenced_historical_encounters ?? []),
      ...(encounter?.triage?.ignored_historical_encounters ?? []),
      ...(encounter?.triage?.aiTriageReport?.referenced_historical_encounters ?? []),
      ...(encounter?.triage?.aiTriageReport?.ignored_historical_encounters ?? [])
    ].map((item) => typeof item === "object"
      ? item.encounterId || item.encounter_id || item.id || item._id
      : item
    ).filter(Boolean).map(String);
    const uniqueIds = [...new Set(historyIds)];
    if (uniqueIds.length === 0) {
      setHistoryEncounterMetadata({});
      return undefined;
    }

    let mounted = true;
    const loadHistoryMetadata = async () => {
      const token = localStorage.getItem("token");
      const results = await Promise.all(uniqueIds.map(async (encounterId) => {
        try {
          const response = await fetch(`${BACKEND_URL}/api/encounters/${encodeURIComponent(encounterId)}`, {
            headers: token ? { Authorization: `Bearer ${token}` } : {},
            cache: "no-store"
          });
          if (!response.ok) return [encounterId, null];
          const payload = await response.json();
          const historical = payload?.encounter ?? payload;
          const historicalTriage = historical?.triage ?? {};
          const historicalAI = historicalTriage?.aiTriageReport ?? {};
          const priority = historicalAI.priority_code ?? historicalTriage.priority_code ?? historicalTriage.priority ?? historicalTriage.SATS_reasoning?.priority;
          const tews = historicalTriage.SATS_reasoning?.tews_total ?? historicalTriage.SATS_reasoning?.sats_points ?? historicalTriage.sats_points ?? historicalTriage.tews_total;
          const diagnosis = historicalAI.diagnosis ?? historicalAI.primary_diagnosis ?? historicalTriage.diagnosis ?? historical.patient?.diagnosis;
          const summary = historicalAI.clinical_summary ?? historicalAI.pathophysiology ?? historicalTriage.ai_summary ?? historicalTriage.summary ?? historical.patient?.symptoms;
          return [encounterId, {
            date: historical.submittedAt ?? historical.createdAt ?? historical.arrival_time ?? historical.arrivalAt,
            diagnosis: diagnosis || null,
            summary: summary || null,
            priority: priority || null,
            tews: tews ?? null
          }];
        } catch {
          return [encounterId, null];
        }
      }));
      if (mounted) setHistoryEncounterMetadata(Object.fromEntries(results));
    };

    loadHistoryMetadata();
    return () => { mounted = false; };
  }, [encounter]);

  // ---------- NEW: fetch top 5 physician recommendations after encounter loads ----------
  useEffect(() => {
    if (!encounter) return;
    let mounted = true;
    let retryTimer = null;
    let retryCount = 0;
    const MAX_RETRIES = 5;

    const fetchRecommendations = async () => {
      setLoadingDocs(true);
      try {
        const encId = getEncounterId(encounter);
        if (!encId) return;

        // CRITICAL FIX: Wait for aiTriageReport to be fully populated before calling physician endpoint
        // If aiTriageReport is missing or empty, the encounter is still being processed by triage
        if (!encounter.triage?.aiTriageReport || !encounter.triage?.aiTriageReport?.pathophysiology) {
          console.warn(`[PatientReport] ⚠️  Encounter still processing (aiTriageReport not ready). Will retry in 2s...`);
          setLoadingDocs(false);
          // Smart retry with counter instead of hard reload
          if (retryCount < MAX_RETRIES) {
            console.warn(`[PatientReport] ⚠️  Encounter still processing (aiTriageReport not ready). Retry ${retryCount + 1}/${MAX_RETRIES} in 2s...`);
            if (mounted) {
              retryTimer = setTimeout(() => {
                if (mounted) {
                  retryCount++;
                  console.log(`[PatientReport] 🔄 Retrying physician fetch (attempt ${retryCount})...`);
                  fetchRecommendations(); // Recursive retry without page reload
                }
              }, 2000);
            }
          } else {
            console.error(`[PatientReport] ❌ MAX RETRIES REACHED. Using fallback display.`);
            setLoadingDocs(false);
            setRecommendedDocs([]);
          }
          return;
        }

        console.log(`[PatientReport] ✅ aiTriageReport ready - fetching clinician recommendations`);

        const token = localStorage.getItem("token");
        const headers = {
          ...( token ? { Authorization: `Bearer ${token}` } : {}),
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache"
        };

        // Fetch both physicians and nurses (no role filter)
        const cacheBreaker = `&t=${Date.now()}`;
        const res = await fetch(`${BACKEND_URL}/api/clinical/staff/clocked-in?encounterId=${encId}${cacheBreaker}`, { 
          headers,
          cache: "no-store"
        });
        if (!res.ok) {
          // not fatal: fallback
          console.warn("Failed to fetch clinician recommendations, status:", res.status);
          setRecommendedDocs([]);
          setRecommendedClinicians([]);
          return;
        }

        const data = await res.json().catch(() => null);
        
        // Get combined clinician list (physicians + nurses)
        const allClinicians = data?.clinicians ?? [];
        const physicianList = data?.physicians ?? [];
        const nurseList = data?.nurses ?? [];

        // Normalise clinician objects
        const normClinicians = (allClinicians || []).map((d, i) => ({
          id: d.id ?? d.physician_id ?? String(d.physicianId ?? (`clinician-${i}`)),
          name: d.name ?? d.physician_name ?? d.displayName ?? "Unknown",
          role: d.role ?? 'PHYSICIAN',
          specialty: d.specialty ?? d.department ?? "General",
          score: typeof d.score === "number" ? d.score : (d.match ?? d.weight ?? 0),
          reason: d.reason ?? d.note ?? d.rationale ?? ""
        }));

        // Keep separate lists for physicians
        const normPhysicians = (physicianList || []).slice(0, 5).map((d, i) => ({
          id: d.id ?? d.physician_id ?? String(d.physicianId ?? (`doc-${i}`)),
          name: d.name ?? d.physician_name ?? d.displayName ?? "Unknown",
          role: 'PHYSICIAN',
          specialty: d.specialty ?? d.department ?? "General",
          score: typeof d.score === "number" ? d.score : (d.match ?? d.weight ?? 0),
          reason: d.reason ?? d.note ?? d.rationale ?? ""
        }));

        if (!mounted) return;
        setRecommendedDocs(normPhysicians);
        setRecommendedClinicians(normClinicians);

        // default assign top-ranked physician
        if (normPhysicians.length > 0) {
          // if encounter already has manually selected clinicians, use those first
          const manuallySelected = encounter?.manually_selected_clinicians ?? [];
          console.log('[PatientReport] 🔍 Checking for manually selected clinicians:', {
            hasField: !!encounter?.manually_selected_clinicians,
            value: manuallySelected,
            length: manuallySelected.length,
            fullEncounter: encounter
          });
          if (manuallySelected && manuallySelected.length > 0) {
            console.log('[PatientReport] ✅ Loading manuallySelectedClinicians:', manuallySelected);
            setManuallySelectedClinicians(manuallySelected);
          }
          
          // also load any assigned clinicians from AI recommendations
          const assignedList = encounter?.triage?.assigned_clinicians ?? encounter?.triage?.assignedClinicians ?? [];
          if (assignedList && assignedList.length > 0) {
            setAssignedClinicians(assignedList);
          } else {
            // Auto-assign top physician
            if (normPhysicians.length > 0) {
              setSelectedDoc(normPhysicians[0]);
            }
          }
        } else {
          // if no recommendations, default to assigned clinician (if present)
          const assigned = encounter?.triage?.assigned_physician ?? encounter?.triage?.assignedPhysician ?? null;
          if (assigned) {
            setSelectedDoc({
              id: assigned.id ?? assigned.physician_id ?? assigned.physicianId,
              name: assigned.name ?? assigned.physician_name
            });
          }
        }
      } catch (err) {
        console.error("Fetch recommendations error:", err);
      } finally {
        if (mounted) setLoadingDocs(false);
      }
    };

    fetchRecommendations();
    return () => { 
      mounted = false;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [encounter]);

  // ---------- helpers ----------
  const timeAgo = (ts) => {
    if (!ts) return "N/A";
    const diff = Math.floor((Date.now() - new Date(ts)) / 1000);
    const mins = Math.floor(diff / 60);
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    return `${hrs}h ${mins % 60}m`;
  };

  const fmt = (v, unit = "") => {
    if (v === undefined || v === null) return `--${unit}`;
    return (v === "--") ? `--${unit}` : `${v}${unit}`;
  };

  // exported for unit testing
  const computeDisplayBP = (vitals = {}, triage = {}) => {
    // Check for valid numeric BP; avoid undefined/null coercion
    const sys = vitals?.bp_systolic;
    const dia = vitals?.bp_diastolic;
    if (typeof sys === 'number' && typeof dia === 'number' && sys > 0 && dia > 0) {
      return `${sys}/${dia}`;
    }
    // Check for BP string, but filter out "undefined/undefined" or similar
    if (vitals?.bp && typeof vitals.bp === 'string' && !vitals.bp.includes('undefined')) {
      return vitals.bp;
    }
    // fallback to triage.vitals_parsed
    if (triage?.vitals_parsed && typeof triage.vitals_parsed === 'object') {
      const vp = triage.vitals_parsed;
      const vp_sys = vp.bp_systolic;
      const vp_dia = vp.bp_diastolic;
      if (typeof vp_sys === 'number' && typeof vp_dia === 'number' && vp_sys > 0 && vp_dia > 0) {
        return `${vp_sys}/${vp_dia}`;
      }
      if (vp.bp && typeof vp.bp === 'string' && !vp.bp.includes('undefined')) {
        return vp.bp;
      }
    }
    return '--';
  };


  const priorityToVisual = (priorityCode) => {
    const p = priorityCode == null ? null : Number(priorityCode);
    switch (p) {
      case 1: return { label: "1 — RED", color: "#ef4444", className: "sev-critical" };
      case 2: return { label: "2 — ORANGE", color: "#f97316", className: "sev-high" };
      case 3: return { label: "3 — YELLOW", color: "#eab308", className: "sev-mod" };
      case 4: return { label: "4 — GREEN", color: "#22c55e", className: "sev-low" };
      default: return { label: "N/A", color: "#94a3b8", className: "sev-unknown" };
    }
  };

  const tryParsePossibleObjectString = (maybeStr) => {
    if (!maybeStr || typeof maybeStr !== "string") return null;
    const looksLikeObject = /[{[]/.test(maybeStr) && /[}\]]/.test(maybeStr);
    if (!looksLikeObject) return null;
    try {
      let s = maybeStr.replace(/\bNone\b/g, "null")
                       .replace(/\bTrue\b/g, "true")
                       .replace(/\bFalse\b/g, "false");
      s = s.replace(/'/g, '"');
      s = s.replace(/,(\s*[}\]])/g, "$1");
      const parsed = JSON.parse(s);
      return parsed;
    } catch (e) {
      return null;
    }
  };

  const renderTews = (tewsObj) => {
    if (!tewsObj || typeof tewsObj !== "object") return <div>N/A</div>;
    const entries = Object.entries(tewsObj);
    if (!entries.length) return <div>N/A</div>;
    return (
      <ul style={{ margin: 0, paddingLeft: 14 }}>
        {entries.map(([k, v]) => (
          <li key={k} style={{ fontSize: 13 }}>
            <strong style={{ textTransform: "capitalize" }}>{k.replace(/_/g, " ")}:</strong> {String(v)}
          </li>
        ))}
      </ul>
    );
  };

  const renderClinicalStructure = (cs) => {
    if (!cs) return null;
    if (typeof cs === "string") {
      const parsed = tryParsePossibleObjectString(cs);
      if (parsed) cs = parsed;
    }
    if (typeof cs === "string") return <div style={{ whiteSpace: "pre-wrap" }}>{cs}</div>;
    if (Array.isArray(cs)) {
      return <ul style={{ margin: 0, paddingLeft: 14 }}>{cs.map((item, idx) => <li key={idx}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul>;
    }
    return (
      <div style={{ fontSize: 13 }}>
        {Object.entries(cs).map(([key, value]) => (
          <div key={key} style={{ marginBottom: 6 }}>
            <strong style={{ textTransform: "capitalize" }}>{key.replace(/_/g, " ")}:</strong>
            {typeof value === "string" ? (
              <div style={{ marginTop: 4 }}>{value}</div>
            ) : Array.isArray(value) ? (
              <ul style={{ margin: "6px 0 0 14px" }}>{value.map((v, i) => <li key={i}>{typeof v === "string" ? v : JSON.stringify(v)}</li>)}</ul>
            ) : (
              <pre style={{ margin: "6px 0 0 0", whiteSpace: "pre-wrap", fontSize: 12 }}>{JSON.stringify(value, null, 2)}</pre>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderRAGCases = (cases) => {
    if (!Array.isArray(cases) || cases.length === 0) {
      return <div style={{ color: "#64748b", fontSize: 13 }}>No reference cases found</div>;
    }

    return (
      <div style={{ marginTop: 8 }}>
        {cases.map((caseData, idx) => {
          const referenced = caseData.medgemma_referenced === true;
          return (
            <div
              key={caseData.case_id ?? idx}
              style={{
                padding: 12,
                marginBottom: 10,
                border: referenced ? "2px solid #10b981" : "1px solid #e2e8f0",
                borderRadius: 8,
                background: referenced ? "#f0fdf4" : "#f8fafc",
                transition: "all 0.2s"
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 8 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "#0f172a" }}>
                      {caseData.diagnosis || "Unknown diagnosis"}
                    </div>
                    {referenced && (
                      <span
                        style={{
                          display: "inline-block",
                          padding: "2px 8px",
                          borderRadius: 4,
                          background: "#10b981",
                          color: "white",
                          fontSize: 11,
                          fontWeight: 600
                        }}
                      >
                        ✓ Referenced by AI
                      </span>
                    )}
                  </div>

                  <div style={{ fontSize: 12, color: "#475569", marginBottom: 6 }}>
                    <strong>Chief complaint:</strong> {caseData.text || "N/A"}
                  </div>

                  <div style={{ display: "flex", gap: 16, fontSize: 12 }}>
                    <div>
                      <span style={{ color: "#64748b" }}>Case ID:</span> <code style={{ background: "#f1f5f9", padding: "2px 4px", borderRadius: 3, fontFamily: "monospace", fontSize: 11 }}>{caseData.case_id}</code>
                    </div>
                  </div>
                </div>

                <div style={{ textAlign: "right", minWidth: 70 }}>
                  <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>Similarity</div>
                  <div
                    style={{
                      display: "inline-block",
                      padding: "6px 10px",
                      borderRadius: 6,
                      background: referenced ? "#dbeafe" : "#e2e8f0",
                      color: referenced ? "#0369a1" : "#475569",
                      fontWeight: 700,
                      fontSize: 12
                    }}
                  >
                    {Math.round((caseData.similarity_score ?? 0) * 100)}%
                  </div>
                  <button
                    onClick={() => {
                      // Navigate to case detail page
                      const encounterId = getEncounterId(encounter);
                      const caseId = caseData.case_id || `case-${idx}`;
                      navigate(`/summary/${encounterId}/case/${caseId}`, { 
                        state: { caseData, encounterId }
                      });
                    }}
                    style={{
                      marginTop: 8,
                      padding: "4px 8px",
                      fontSize: 11,
                      fontWeight: 600,
                      border: "1px solid #d1d5db",
                      borderRadius: 4,
                      background: "#f3f4f6",
                      color: "#374151",
                      cursor: "pointer",
                      transition: "all 0.2s"
                    }}
                    onMouseOver={(e) => {
                      e.target.style.background = "#e5e7eb";
                      e.target.style.borderColor = "#9ca3af";
                    }}
                    onMouseOut={(e) => {
                      e.target.style.background = "#f3f4f6";
                      e.target.style.borderColor = "#d1d5db";
                    }}
                  >
                    Read More
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderEvidenceTrail = (casesList) => {
    const safeCases = Array.isArray(casesList) ? casesList : [];
    const referencedCases = safeCases.filter(c => c.medgemma_referenced === true);
    const totalCases = safeCases.length;
    const ragData = triage?.ragData ?? {};

    return (
      <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: "#fffbeb", borderLeft: "4px solid #eab308" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <div style={{ fontWeight: 700, color: "#0f172a", fontSize: 13 }}>
            PubMed / RAG reference evidence
          </div>
          <button
            onClick={() => setShowEvidenceTrail(!showEvidenceTrail)}
            style={{
              padding: "6px 12px",
              fontSize: 12,
              fontWeight: 600,
              border: "1px solid #f59e0b",
              borderRadius: 6,
              background: "white",
              color: "#d97706",
              cursor: "pointer",
              transition: "all 0.2s"
            }}
          >
            {showEvidenceTrail ? "Hide references" : "Show references"}
          </button>
        </div>

        <div style={{ color: "#475569", fontSize: 12, lineHeight: 1.5 }}>
          {ragData.enabled_by_request === false || triage.computeMode === "llm_only" || triage.computeMode === "LLM_ONLY"
            ? "RAG was disabled for this decision. No population reference cases were used."
            : `${totalCases} population reference case${totalCases === 1 ? "" : "s"} retrieved; ${referencedCases.length} marked as used by the model.`}
        </div>

        {showEvidenceTrail && (
          <div style={{ marginTop: 10, borderTop: "1px solid #fcd34d", paddingTop: 12 }}>
            {totalCases > 0 ? renderRAGCases(safeCases) : (
              <div style={{ color: "#64748b", fontSize: 13 }}>No PubMed/RAG reference cases were returned.</div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderPatientHistoryEvidence = (provenance) => {
    const history = provenance && typeof provenance === "object" ? provenance : null;
    const assessment = history?.patient_history_assessment;
    const references = Array.isArray(history?.referenced_encounters) ? history.referenced_encounters : [];
    const exclusions = Array.isArray(history?.ignored_encounters) ? history.ignored_encounters : [];
    const criticalHistory = Array.isArray(history?.critical_history) ? history.critical_history : [];
    const hasHistoryProvenance = Boolean(assessment || history?.statement || references.length > 0 || exclusions.length > 0 || criticalHistory.length > 0);
    const status = assessment === "referenced"
      ? "Referenced in this decision"
      : assessment === "loaded_not_relevant"
        ? "Reviewed, not relevant"
        : assessment === "not_available"
          ? "No previous encounters available"
          : assessment === "reviewed"
            ? "History reviewed for this decision"
          : "Provenance unavailable";

    const openEncounterInLibrary = (encounterId) => {
      if (!encounterId) return;
      const patientId = encounter?.patient?.id_number || encounter?.patientId || encounter?.patient?.id;
      const patientQuery = patientId ? `patient=${encodeURIComponent(patientId)}&` : "";
      navigate(`/ehr?${patientQuery}encounter=${encodeURIComponent(encounterId)}`);
    };

    const renderHistoricalMetadata = (reference) => {
      const metadata = historyEncounterMetadata[String(reference.encounterId)] ?? reference;
      const date = metadata?.date || metadata?.timestamp;
      const formattedDate = date ? new Date(date).toLocaleDateString(undefined, {
        year: "numeric", month: "short", day: "numeric"
      }) : null;
      const priority = metadata?.priority ? String(metadata.priority).toUpperCase() : null;
      const details = [
        formattedDate,
        metadata?.diagnosis ? `Diagnosis: ${metadata.diagnosis}` : null,
        priority ? `SATS: ${priority}` : null,
        metadata?.tews !== null && metadata?.tews !== undefined ? `TEWS: ${metadata.tews}` : null
      ].filter(Boolean);
      return details.length > 0 ? <div className="patient-history-metadata">{details.join(" · ")}</div> : null;
    };

    const renderHistoricalSummary = (reference) => {
      const metadata = historyEncounterMetadata[String(reference.encounterId)] ?? reference;
      const summary = metadata?.summary || metadata?.reason || metadata?.rationale;
      return summary ? <div className="patient-history-summary">{String(summary)}</div> : null;
    };

    return (
      <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: "#eff6ff", borderLeft: "4px solid #2563eb" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
          <div style={{ fontWeight: 700, color: "#0f172a", fontSize: 13 }}>Patient EHR history</div>
          <button
            type="button"
            onClick={() => setShowPatientHistoryEvidence(!showPatientHistoryEvidence)}
            style={{ padding: "6px 12px", fontSize: 12, fontWeight: 600, border: "1px solid #93c5fd", borderRadius: 6, background: "white", color: "#1d4ed8", cursor: "pointer" }}
          >
            {showPatientHistoryEvidence ? "Hide history" : "Show history"}
          </button>
        </div>
        <div style={{ marginTop: 8, fontSize: 12, fontWeight: 700, color: assessment === "referenced" ? "#1d4ed8" : "#475569" }}>
          {status}
        </div>
        {showPatientHistoryEvidence && (
          <div style={{ marginTop: 8, borderTop: "1px solid #bfdbfe", paddingTop: 10, color: "#334155", fontSize: 12, lineHeight: 1.6 }}>
            {!hasHistoryProvenance && <div>Evidence provenance is unavailable for this older encounter.</div>}
            {history?.statement && <div>{history.statement}</div>}
            {criticalHistory.length > 0 && <div><strong>Critical history:</strong> {criticalHistory.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join(", ")}</div>}
            {references.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <strong>Used in triage decision:</strong>
                <ul style={{ margin: "4px 0 0 18px", padding: 0 }}>
                  {references.map((reference, index) => (
                    <li key={`${reference.encounterId || "encounter"}-${index}`}>
                      <button
                        type="button"
                        onClick={() => openEncounterInLibrary(reference.encounterId)}
                        style={{ border: 0, padding: 0, background: "transparent", color: "#1d4ed8", textDecoration: "underline", cursor: "pointer", fontWeight: 700 }}
                        title="Open this encounter in the EHR Library"
                      >
                        {reference.encounterId}
                      </button>
                      {renderHistoricalMetadata(reference)}
                      {renderHistoricalSummary(reference)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {exclusions.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <strong>Excluded from triage decision:</strong>
                <ul style={{ margin: "4px 0 0 18px", padding: 0 }}>
                  {exclusions.map((exclusion, index) => (
                    <li key={`${exclusion.encounterId || "encounter"}-${index}`}>
                      <button
                        type="button"
                        onClick={() => openEncounterInLibrary(exclusion.encounterId)}
                        style={{ border: 0, padding: 0, background: "transparent", color: "#64748b", textDecoration: "underline", cursor: "pointer", fontWeight: 700 }}
                        title="Open this encounter in the EHR Library"
                      >
                        {exclusion.encounterId}
                      </button>
                      {renderHistoricalMetadata(exclusion)}
                      {renderHistoricalSummary(exclusion)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // ---------- actions ----------
  const handleConfirm = async () => {
    setSaving(true);
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        alert("You must be logged in to confirm a report.");
        navigate("/");
        return;
      }

      const encId = getEncounterId(encounter);
      if (!encId) throw new Error("Encounter ID missing; cannot confirm.");

      // include selected physician when confirming (human override)
      const payloadBody = {
        nurseNotes,
        selectedPhysicianId: selectedDoc?.id ?? null
      };

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encId}/confirm`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payloadBody),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || `Confirm failed (status ${res.status})`);
      }

      const data = await res.json().catch(() => null);
      const updatedEncounter = data?.encounter ?? data;
      if (!updatedEncounter) throw new Error("Server returned no updated encounter");

      setEncounter(updatedEncounter);
      setNurseNotes(updatedEncounter?.nurseNotes ?? "");
      alert("Patient report confirmed");
      navigate("/", { state: { newPatient: updatedEncounter } });
    } catch (err) {
      console.error("Confirm error:", err);
      alert("Failed to confirm: " + (err.message || ""));
    } finally {
      setSaving(false);
    }
  };

  // Assign immediately (human chooses one from top5) - uses new clinical endpoint
  const handleAssignPhysician = async (doc) => {
    if (!doc) return;
    if (!window.confirm(`Assign ${doc.name} to this patient now?`)) return;

    setAssigning(true);
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        alert("You must be logged in to assign a clinician.");
        navigate("/");
        return;
      }
      const encId = getEncounterId(encounter);
      if (!encId) throw new Error("Encounter ID missing; cannot assign.");

      // Use new clinical API endpoint
      const res = await fetch(`${BACKEND_URL}/api/clinical/encounters/${encId}/assign-staff`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ staffId: doc.id, role: "PHYSICIAN" }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || `Assign failed (status ${res.status})`);
      }

      const payload = await res.json().catch(() => null);
      const updated = payload?.encounter ?? payload;
      if (!updated) throw new Error("Server returned no updated encounter");

      setEncounter(updated);
      setEditablePhysician(doc.name);
      setEditablePhysicianOriginal(doc.name);
      setPhysicianEdited(false);
      // Auto-derive specialty from selected physician
      if (doc.specialty) {
        setEditableSpecialty(doc.specialty);
        setEditableSpecialtyOriginal(doc.specialty);
        setSpecialtyEdited(true);
      }
      setSelectedDoc({ id: doc.id, name: doc.name, specialty: doc.specialty }); // keep UI in sync
      alert(`${doc.name} assigned to patient`);
    } catch (err) {
      console.error("Assign error:", err);
      alert("Failed to assign clinician: " + (err.message || ""));
    } finally {
      setAssigning(false);
    }
  };

  const handleAttendPatient = async () => {
    if (!window.confirm("Mark this patient as attended by the physician?")) return;

    setSaving(true);
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        alert("You must be logged in to mark as attended.");
        navigate("/");
        return;
      }

      const encId = getEncounterId(encounter);
      if (!encId) throw new Error("Encounter ID missing; cannot mark as attended.");

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encId}/attend`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || `Attend failed (status ${res.status})`);
      }

      const payload = await res.json().catch(() => null);
      const updated = payload?.encounter ?? payload;
      if (!updated) throw new Error("Server returned no updated encounter");

      setEncounter(updated);
      alert("Patient marked as attended");
      navigate("/");
    } catch (err) {
      console.error("Attend error:", err);
      alert("Failed to mark patient as attended: " + (err.message || ""));
    } finally {
      setSaving(false);
    }
  };

  // ---------- render ----------
  if (loading) return <div style={{ padding: 24 }}>Loading patient report…</div>;
  if (!encounter) return <div style={{ padding: 24 }}>No report found.</div>;

  // normalized fields (defensive)
  const patient = encounter.patient ?? {};
  // Start with stored vitals; then fill missing/null fields from triage.vitals_parsed
  let vitals = { ...(encounter.vitals ?? {}) };
  if (encounter.triage?.vitals_parsed && typeof encounter.triage.vitals_parsed === "object") {
    const vp = encounter.triage.vitals_parsed;
    // Coerce numeric components where possible and avoid copying noisy strings like 'undefined/undefined'
    const keys = ["hr", "o2", "resp", "temp", "avpu"];
    keys.forEach((k) => {
      const cur = vitals[k];
      if (cur === undefined || cur === null || cur === "" || (typeof cur === 'number' && Number.isNaN(cur))) {
        vitals[k] = vp[k];
      }
    });

    // Handle systolic/diastolic specially: coerce to numbers when present
    const tryAssignNumber = (field) => {
      const cur = vitals[field];
      const val = vp[field];
      if (cur === undefined || cur === null || cur === "" || (typeof cur === 'number' && Number.isNaN(cur))) {
        if (val === undefined || val === null || val === "") {
          vitals[field] = null;
        } else {
          const n = Number(val);
          vitals[field] = Number.isFinite(n) ? n : null;
        }
      }
    };

    tryAssignNumber('bp_systolic');
    tryAssignNumber('bp_diastolic');
    tryAssignNumber('bp_left_systolic');
    tryAssignNumber('bp_left_diastolic');
    tryAssignNumber('bp_right_systolic');
    tryAssignNumber('bp_right_diastolic');

    // Compose a clean `bp` string from numeric components if available; otherwise only accept vp.bp if it's not a noisy placeholder
    const sys = vitals.bp_systolic;
    const dia = vitals.bp_diastolic;
    if (typeof sys === 'number' && typeof dia === 'number' && sys > 0 && dia > 0) {
      vitals.bp = `${sys}/${dia}`;
    } else {
      const vpBp = vp.bp;
      if (vpBp && typeof vpBp === 'string' && !vpBp.includes('undefined')) {
        // only copy a vetted string
        vitals.bp = vpBp;
      } else if (!vitals.bp || vitals.bp.includes('undefined')) {
        vitals.bp = null;
      }
    }
  }
  const transcript = Array.isArray(encounter.transcript) ? encounter.transcript : [];
  const externalTriage = location.state?.triagePayload ?? {};
  const triage = { ...(encounter.triage ?? {}), ...externalTriage };
  const aiReport = triage.aiTriageReport ?? {};
  const normalizedAI = normalizeClinicalAnalysis({
    aiTriageReport: aiReport,
    SATS_reasoning: triage.SATS_reasoning,
    sats_points: triage.sats_points ?? triage.satsPoints ?? aiReport.sats_points,
    tews_total: triage.tews_total,
    priority_code: triage.priority_code,
    ai_summary: triage.ai_summary,
    summary: triage.summary
  });

  // DEBUG: inspect what the backend returned so we can trace missing BP values
  if (process.env.NODE_ENV !== 'production') {
    // eslint-disable-next-line no-console
    console.debug('[PatientReport] encounter.vitals:', encounter.vitals, 'triage.vitals_parsed:', triage.vitals_parsed, 'merged vitals:', vitals);
  }

  // Compute BP display using merged vitals (which now includes vitals_parsed fallback)
  const displayBP = computeDisplayBP(vitals, triage);
  const priorityCodeRaw = normalizedAI.priority_code ??
    triage.priority_code ??
    triage.priority ??
    triage.SATS_reasoning?.priority ??
    triage.aiTriageReport?.calculated_priority ??
    triage.esi_level;
  const priorityText = String(priorityCodeRaw ?? '').trim().toLowerCase();
  const priorityMap = { red: 1, orange: 2, yellow: 3, green: 4 };
  const parsedPriority = priorityMap[priorityText] ?? Number(priorityText.replace(/[^1-4]/g, '').slice(0, 1));
  const rawTewsTotal = normalizedAI.tews_total ??
    triage?.SATS_reasoning?.tews_total ??
    triage?.SATS_reasoning?.sats_points ??
    triage?.sats_points ??
    triage?.satsPoints ??
    triage?.tews_total ??
    aiReport?.sats_points ??
    aiReport?.tews_total;
  const numericTewsTotal = Number(rawTewsTotal);
  const priorityCode = Number.isInteger(parsedPriority) && parsedPriority >= 1 && parsedPriority <= 4
    ? parsedPriority
    : Number.isFinite(numericTewsTotal)
      ? (numericTewsTotal >= 7 ? 1 : numericTewsTotal >= 5 ? 2 : numericTewsTotal >= 3 ? 3 : 4)
      : 4;
  
  // Use editable SATS color for display if it's been edited, otherwise use original
  const displayPriorityCode = satsColorEdited ? editableSATSColor : priorityCode;
  const { label: priorityLabel, color: priorityColor } = priorityToVisual(displayPriorityCode);
  
  const targetMins = triage.target_time_mins ?? triage.target_mins ?? null;
  const waitingSeverity = triage.waitingSeverity ?? triage.waiting_severity ?? triage?.aiTriageReport?.urgency_level ?? encounter.waitingSeverity ?? null;
  // Use discriminators_readable (human-friendly text) instead of discriminators_found (concept keys)
  const discriminators = normalizedAI.clinical_discriminators.length > 0
    ? normalizedAI.clinical_discriminators
    : triage?.SATS_reasoning?.discriminators_readable ?? triage?.SATS_reasoning?.discriminators_found ?? triage?.SATS_reasoning?.discriminators ?? triage?.discriminators ?? [];
  const tews = triage?.SATS_reasoning?.tews_components ?? triage?.tews_components ?? {};
  const tewsTotal = rawTewsTotal ?? null;
  const mobility = triage?.SATS_reasoning?.mobility ?? triage?.mobility ?? encounter?.vitals?.mobility ?? 'Walks';
  const traumaPresent = Boolean(triage?.SATS_reasoning?.traumaPresent ?? triage?.traumaPresent ?? encounter?.vitals?.traumaPresent);
  
  // Use editable SATS points for display if edited, otherwise use original
  const displayTewsTotal = satsPointsEdited ? editableSATSPoints : tewsTotal;
  
  const assignedPhys = triage?.assigned_physician ?? triage?.assignedPhysician ?? null;
  const assignedSpec = aiReport?.specialty ?? triage?.assigned_specialty ?? triage?.assignedSpecialty ?? null;
  const ward = triage?.ward ?? null;

  const normalizeHistoryEncounters = (value) => {
    if (!Array.isArray(value)) return [];
    return value.map((item) => {
      if (typeof item === "string" || typeof item === "number") return { encounterId: String(item) };
      if (!item || typeof item !== "object") return null;
      const encounterId = item.encounterId || item.encounter_id || item.id || item._id;
      return encounterId ? { ...item, encounterId: String(encounterId) } : null;
    }).filter(Boolean);
  };

  const referencedHistory = normalizeHistoryEncounters(
    aiReport?.referenced_historical_encounters || triage?.referenced_historical_encounters ||
    triage?.referencedHistoricalEncounters || encounter?.referencedHistoricalEncounters
  );
  const ignoredHistory = normalizeHistoryEncounters(
    aiReport?.ignored_historical_encounters || triage?.ignored_historical_encounters ||
    triage?.ignoredHistoricalEncounters || encounter?.ignoredHistoricalEncounters
  );
  const historySource = triage?.longitudinalHistory || triage?.longitudinal_history || {};
  const historyEvidence = {
    assessment: referencedHistory.length > 0 || ignoredHistory.length > 0 ? "reviewed" : undefined,
    referenced_encounters: referencedHistory,
    ignored_encounters: ignoredHistory,
    critical_history: historySource.critical_history || triage?.criticalHistory || encounter?.criticalHistory || []
  };

  const aiSummaryRaw = normalizedAI.clinical_summary || triage?.ai_summary || triage?.summary || "";
  const aiSummaryParsed = tryParsePossibleObjectString(aiSummaryRaw);
  const clinicalAnalysisSections = [
    { key: "pathophysiology", title: "Pathophysiology", icon: "🔬", value: normalizedAI.pathophysiology },
    { key: "clinicalSummary", title: "Clinical Summary", icon: "✓", value: normalizedAI.clinical_summary },
    { key: "severityAssessment", title: "Severity Assessment", icon: "⚠", value: normalizedAI.severity_assessment },
    { key: "differential", title: "Differential Diagnoses", icon: "📊", value: normalizedAI.differential_diagnoses },
    { key: "lifeThreats", title: "Life Threats Excluded", icon: "✅", value: normalizedAI.life_threats_excluded },
    { key: "specialists", title: "Recommended Specialists", icon: "👨‍⚕", value: normalizedAI.recommended_specialists },
    { key: "statActions", title: "Immediate Actions", icon: "🚨", value: normalizedAI.immediate_actions },
    { key: "diagnosticWorkup", title: "Diagnostic Workup", icon: "🧪", value: normalizedAI.diagnostic_workup },
    { key: "recommendations", title: "Recommendations", icon: "💊", value: normalizedAI.recommendations },
    { key: "pathognomonic", title: "Pathognomonic Findings", icon: "⭐", value: normalizedAI.pathognomonic_markers },
    { key: "riskFactors", title: "Risk Factors", icon: "⚠", value: normalizedAI.risk_factors },
    { key: "disposition", title: "Disposition Rationale", icon: "📍", value: normalizedAI.disposition_rationale }
  ];
  const hasEmergencyPriority = [1, 2].includes(Number(normalizedAI.priority_code ?? triage.priority_code)) ||
    /emergent|immediate|critical/i.test(String(aiReport.urgency_level || triage.waiting_severity || ""));
  const renderClinicalValue = (value) => {
    if (Array.isArray(value)) {
      return value.length > 0 ? (
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {value.map((item, index) => (
            <li key={index} style={{ marginBottom: 5 }}>
              {typeof item === "object" ? `${item.dx || item.diagnosis || "Finding"}${item.confidence != null ? ` (${Math.round(Number(item.confidence) * 100)}% confidence)` : ""}` : String(item)}
            </li>
          ))}
        </ul>
      ) : null;
    }
    return value ? <div style={{ whiteSpace: "pre-wrap" }}>{String(value)}</div> : null;
  };

  // Add CSS animation for spinner
  if (typeof document !== 'undefined' && !document.getElementById('spin-animation')) {
    const style = document.createElement('style');
    style.id = 'spin-animation';
    style.textContent = `
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
  }

  return (
    <div className="patient-report-page" style={{ padding: 20 }}>
      <h2 style={{ marginBottom: 12 }}>Patient Report Preview</h2>

      <div className="patient-report-columns" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", gap: 20 }}>
        <div>
          {/* -- main left column (patient, transcript, triage) -- */}
          <section className="block">
            <div className="patient-report-section-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h4 className="block-label" style={{ margin: 0 }}>Patient</h4>
              <div style={{
                display: "flex",
                alignItems: "center",
                gap: 12
              }}>
                <div style={{
                  padding: "8px 12px",
                  borderRadius: 10,
                  background: priorityColor,
                  color: "white",
                  fontWeight: 800,
                  boxShadow: "0 3px 10px rgba(0,0,0,0.12)"
                }}>
                  {priorityLabel}
                </div>

                <div style={{ textAlign: "right", fontSize: 12, color: "#334155" }}>
                  <div style={{ fontWeight: 700 }}>
                    {manuallySelectedClinicians.length > 0 || assignedClinicians.length > 0 
                      ? [...manuallySelectedClinicians, ...assignedClinicians].map(c => c.specialty).filter((v, i, a) => a.indexOf(v) === i).join(", ") 
                      : "No specialty"}
                  </div>
                  <div style={{ fontSize: 11 }}>
                    {manuallySelectedClinicians.length > 0 || assignedClinicians.length > 0
                      ? `${manuallySelectedClinicians.length + assignedClinicians.length} clinician${(manuallySelectedClinicians.length + assignedClinicians.length) !== 1 ? 's' : ''} assigned`
                      : "No clinician assigned"}
                  </div>
                </div>
              </div>
            </div>

            <div className="content-area" style={{ marginTop: 12, textAlign: "left" }}>
              {/* Demographics Grid - 2 columns matching LeftColumn.jsx pattern */}
              <div className="patient-report-demographics" style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "12px 16px", marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 2 }}>ID</div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "#0f172a" }}>
                    {patient?.id_number ? `${patient.id_number}` : "---"}
                  </div>
                </div>
                
                <div>
                  <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 2 }}>Name</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>
                    {patient?.name ?? "---"} {patient?.surname ? patient.surname : ""}
                  </div>
                </div>
                
                <div>
                  <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 2 }}>Age</div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "#0f172a" }}>
                    {patient?.age != null ? `${patient.age} years` : "---"}
                  </div>
                </div>
                
                <div>
                  <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 2 }}>Height</div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "#0f172a" }}>
                    {patient?.height_cm != null ? `${patient.height_cm} cm` : "---"}
                  </div>
                </div>
                
                <div style={{ gridColumn: "1 / -1" }}>
                  <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 2 }}>Presenting Complaint</div>
                  <div style={{ fontSize: 13, color: "#1f2937", lineHeight: 1.4 }}>
                    {triage?.patient_symptoms ?? patient?.symptoms ?? <em style={{ color: "#9ca3af" }}>Not provided</em>}
                  </div>
                </div>
                
                {(patient?.duration || patient?.painLevel) && (
                  <div style={{ gridColumn: "1 / -1" }}>
                    <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 2 }}>Clinical Presentation</div>
                    <div style={{ fontSize: 13, color: "#1f2937", lineHeight: 1.4 }}>
                      {patient?.duration ?? ""}{patient?.duration && patient?.painLevel ? " • " : ""}{patient?.painLevel ?? ""}
                    </div>
                  </div>
                )}
                
                {patient?.history && (
                  <div style={{ gridColumn: "1 / -1" }}>
                    <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 2 }}>History</div>
                    <div style={{ fontSize: 13, color: "#475569", lineHeight: 1.4, fontStyle: "italic" }}>
                      {patient.history}
                    </div>
                  </div>
                )}
              </div>

              {/* Appearance/Skin/Pain bullet list - compact */}
              {(patient?.appearance || patient?.skin || patient?.pain) && (
                <div style={{ borderTop: "1px solid #e5e7eb", paddingTop: 12, marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 6 }}>Observation Notes</div>
                  <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.5 }}>
                    {patient?.appearance && <li style={{ fontSize: 13, color: "#1f2937", marginBottom: 3 }}><strong>Appearance:</strong> {patient.appearance}</li>}
                    {patient?.skin && <li style={{ fontSize: 13, color: "#1f2937", marginBottom: 3 }}><strong>Skin:</strong> {patient.skin}</li>}
                    {patient?.pain && <li style={{ fontSize: 13, color: "#1f2937" }}><strong>Pain:</strong> {patient.pain}</li>}
                  </ul>
                </div>
              )}
            </div>
          </section>



          <section className="block" style={{ marginTop: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
              <h4 className="block-label" style={{ margin: 0 }}>AI Triage</h4>
              {hasEmergencyPriority && (
                <button
                  type="button"
                  onClick={() => navigate("/manual-triage-report", {
                    state: {
                      patientData: patient,
                      vitalsData: vitals,
                      selectedMrn: patient?.id_number,
                      ehrSummary: { encounterId: getEncounterId(encounter), source: "ai-triage-override" }
                    }
                  })}
                  style={{ padding: "8px 12px", background: "#b91c1c", color: "white", border: "none", borderRadius: 5, cursor: "pointer", fontWeight: 800, fontSize: 11 }}
                  title="Switch immediately to manual triage"
                >
                  🚨 Manual Triage Override
                </button>
              )}
            </div>
              <div style={{ marginTop: 8 }}>
              <div className="patient-report-priority-row" style={{ display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between" }}>
                <div>
                  <div style={{ fontSize: 14, color: "#334155" }}><strong>Priority code</strong></div>
                  <div style={{
                    marginTop: 6,
                    display: "inline-block",
                    padding: "8px 14px",
                    borderRadius: 8,
                    background: priorityColor,
                    color: "#fff",
                    fontWeight: 800
                  }}>{priorityLabel}</div>
                </div>

                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 13, color: "#334155" }}><strong>Target</strong></div>
                  <div style={{ marginTop: 6, fontWeight: 700 }}>{targetMins != null ? `${targetMins} min` : "N/A"}</div>
                </div>
              </div>

              <div style={{ marginTop: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ fontSize: 13, color: "#334155" }}><strong>AI Summary</strong> {aiSummaryEdited && <span style={{color: "#dc2626", fontSize: 11, fontWeight: 700}}>[EDITED]</span>}</div>
                  <button
                    onClick={() => setEditModeAISummary(!editModeAISummary)}
                    style={{
                      fontSize: 11,
                      padding: "4px 8px",
                      border: "1px solid #d1d5db",
                      borderRadius: 4,
                      background: editModeAISummary ? "#dbeafe" : "#f3f4f6",
                      color: editModeAISummary ? "#0369a1" : "#374151",
                      cursor: "pointer",
                      fontWeight: 600,
                      transition: "all 0.2s"
                    }}
                  >
                    {editModeAISummary ? "Done" : "Edit"}
                  </button>
                </div>
                
                {editModeAISummary ? (
                  <div style={{ marginTop: 8, background: "#f0f9ff", padding: 12, borderRadius: 6, border: "1px solid #bfdbfe" }}>
                    <textarea
                      value={editableAISummary}
                      onChange={(e) => {
                        setEditableAISummary(e.target.value);
                        setAiSummaryEdited(e.target.value !== editableAISummaryOriginal);
                      }}
                      style={{
                        width: "100%",
                        minHeight: 120,
                        padding: 10,
                        fontSize: 13,
                        border: "1px solid #60a5fa",
                        borderRadius: 4,
                        fontFamily: "system-ui",
                        resize: "vertical",
                        color: "#0f172a"
                      }}
                    />
                    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                      <button
                        onClick={() => {
                          setEditableAISummary(editableAISummaryOriginal);
                          setAiSummaryEdited(false);
                          setEditModeAISummary(false);
                        }}
                        style={{
                          fontSize: 11,
                          padding: "6px 10px",
                          border: "1px solid #e5e7eb",
                          borderRadius: 4,
                          background: "#f3f4f6",
                          cursor: "pointer",
                          fontWeight: 600
                        }}
                      >
                        Revert
                      </button>
                      <div style={{fontSize: 11, color: "#64748b", alignSelf: "center"}}>
                        {aiSummaryEdited ? "Modified from original" : "No changes"}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ marginTop: 8, background: "#f0fdf4", padding: 14, borderRadius: 8, border: "2px solid #86efac", color: "#0f172a", fontSize: 13, lineHeight: 1.6 }}>
                    {editableAISummary && editableAISummary.length > 10 ? (
                      <div style={{ whiteSpace: "pre-wrap" }}>{editableAISummary}</div>
                    ) : encounter?.triage?.aiTriageReport?.clinical_summary ? (
                      <div style={{ whiteSpace: "pre-wrap" }}>{encounter.triage.aiTriageReport.clinical_summary}</div>
                    ) : encounter?.triage?.aiTriageReport?.pathophysiology ? (
                      <div style={{ whiteSpace: "pre-wrap" }}>{encounter.triage.aiTriageReport.pathophysiology}</div>
                    ) : (
                      <div style={{ color: "#64748b", fontStyle: "italic" }}>AI Triage Summary Processing...</div>
                    )}
                  </div>
                )}
              </div>
              </div>

              <div style={{ marginTop: 12, fontSize: 13 }}>

              <div style={{ marginTop: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ fontSize: 13, color: "#334155" }}><strong>SATS Classification</strong></div>
                  <button
                    onClick={() => setEditModePriority(!editModePriority)}
                    style={{
                      fontSize: 11,
                      padding: "4px 8px",
                      border: "1px solid #d1d5db",
                      borderRadius: 4,
                      background: editModePriority ? "#dbeafe" : "#f3f4f6",
                      color: editModePriority ? "#0369a1" : "#374151",
                      cursor: "pointer",
                      fontWeight: 600
                    }}
                  >
                    {editModePriority ? "Done" : "Edit"}
                  </button>
                </div>
                
                {editModePriority ? (
                  <div style={{ marginTop: 8, background: "#f0f9ff", padding: 12, borderRadius: 6, border: "1px solid #bfdbfe" }}>
                    <label style={{ display: "block", marginBottom: 8, fontSize: 12, fontWeight: 600, color: "#334155" }}>SATS Color Code</label>
                    <select
                      value={editableSATSColor}
                      onChange={(e) => {
                        setEditableSATSColor(e.target.value);
                        setSatsColorEdited(e.target.value !== editableSATSColorOriginal);
                      }}
                      style={{
                        width: "100%",
                        padding: "8px 10px",
                        fontSize: 13,
                        border: "1px solid #60a5fa",
                        borderRadius: 4,
                        marginBottom: 8,
                        backgroundColor: "white"
                      }}
                    >
                      <option value="1">1 - RED (Critical)</option>
                      <option value="2">2 - ORANGE (High)</option>
                      <option value="3">3 - YELLOW (Moderate)</option>
                      <option value="4">4 - GREEN (Low)</option>
                    </select>

                    {/* Color preview while editing */}
                    <div style={{ 
                      padding: "8px 10px", 
                      background: `#${['', 'fef2f2', 'fef3c7', 'fef3c7', 'f0fdf4'][editableSATSColor] || 'f8fafc'}`,
                      borderRadius: 4, 
                      fontSize: 12, 
                      fontWeight: 700, 
                      color: "#0f172a", 
                      marginBottom: 8,
                      border: `2px solid ${['#94a3b8', '#ef4444', '#f97316', '#f1c40f', '#10b981'][editableSATSColor] || '#94a3b8'}`,
                      textAlign: "center"
                    }}>
                      {['N/A', '🔴 RED', '🟠 ORANGE', '🟡 YELLOW', '🟢 GREEN'][editableSATSColor] || 'Preview'}
                    </div>
                    
                    <label style={{ display: "block", marginBottom: 8, fontSize: 12, fontWeight: 600, color: "#334155" }}>SATS Points</label>
                    <input
                      type="text"
                      value={editableSATSPoints}
                      onChange={(e) => {
                        setEditableSATSPoints(e.target.value);
                        setSatsPointsEdited(e.target.value !== editableSATSPointsOriginal);
                      }}
                      style={{
                        width: "100%",
                        padding: "8px 10px",
                        fontSize: 13,
                        border: "1px solid #60a5fa",
                        borderRadius: 4
                      }}
                    />
                  </div>
                ) : (
                  <div style={{ marginTop: 8 }}>
                    {/* Simply display priority color with SATS points - use displayPriorityCode for consistency */}
                    <div style={{ 
                      padding: "10px 12px", 
                      background: `#${['', 'fef2f2', 'fef3c7', 'fef3c7', 'f0fdf4'][displayPriorityCode] || 'f8fafc'}`,
                      borderRadius: 6, 
                      fontSize: 14, 
                      fontWeight: 700, 
                      color: "#0f172a", 
                      border: `2px solid ${['#94a3b8', '#ef4444', '#f97316', '#f1c40f', '#10b981'][displayPriorityCode] || '#94a3b8'}`,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center"
                    }}>
                      <span>{['N/A', '1 — RED', '2 — ORANGE', '3 — YELLOW', '4 — GREEN'][displayPriorityCode] || 'N/A'}{satsColorEdited && <span style={{color: "#dc2626", fontSize: 10, marginLeft: 6, fontWeight: 600}}>[EDITED]</span>}</span>
                      <span style={{ fontSize: 12, fontWeight: 600, marginLeft: 12, whiteSpace: "nowrap" }}>
                        TEWS: {displayTewsTotal ?? "—"} · Mobility: {mobility} · Trauma: {traumaPresent ? "+1" : "+0"}
                      </span>
                    </div>
                  </div>
                )}

                {/* Discriminators Section */}
                <div style={{ marginTop: 12, padding: 12, background: "#fef3c7", borderRadius: 8, border: "1px solid #fcd34d" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                    <div style={{ fontSize: 13, color: "#92400e", fontWeight: 700 }}>🎯 Clinical Discriminators</div>
                    <button
                      onClick={() => setEditModeDiscriminators(!editModeDiscriminators)}
                      style={{
                        fontSize: 11,
                        padding: "4px 8px",
                        border: "1px solid #f59e0b",
                        borderRadius: 4,
                        background: editModeDiscriminators ? "#fef3c7" : "#fef9e7",
                        color: "#92400e",
                        cursor: "pointer",
                        fontWeight: 600
                      }}
                    >
                      {editModeDiscriminators ? "Done" : "Edit"}
                    </button>
                  </div>
                  
                  {editModeDiscriminators ? (
                    <textarea
                      value={editableDiscriminators}
                      onChange={(e) => {
                        setEditableDiscriminators(e.target.value);
                        setDiscriminatorsEdited(e.target.value !== editableDiscriminatorsOriginal);
                      }}
                      placeholder="Enter discriminators separated by comma"
                      style={{
                        width: "100%",
                        minHeight: 80,
                        padding: 10,
                        fontSize: 12,
                        border: "1px solid #f59e0b",
                        borderRadius: 4,
                        fontFamily: "system-ui",
                        backgroundColor: "white"
                      }}
                    />
                  ) : (
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {encounter?.triage?.aiTriageReport?.clinical_discriminators && 
                       Array.isArray(encounter.triage.aiTriageReport.clinical_discriminators) &&
                       encounter.triage.aiTriageReport.clinical_discriminators.length > 0 ? (
                        encounter.triage.aiTriageReport.clinical_discriminators.map((d, i) => (
                          <span key={i} style={{ display: "inline-block", padding: "6px 12px", borderRadius: 20, background: "#fde68a", color: "#78350f", fontSize: 12, fontWeight: 600 }}>{String(d)}</span>
                        ))
                      ) : editableDiscriminators ? (
                        editableDiscriminators.split(",").map((d, i) => (
                          <span key={i} style={{ display: "inline-block", padding: "6px 12px", borderRadius: 20, background: "#fde68a", color: "#78350f", fontSize: 12, fontWeight: 600 }}>{d.trim()}</span>
                        ))
                      ) : (
                        <span style={{ color: "#854d0e", fontSize: 12, fontStyle: "italic" }}>No discriminators identified yet</span>
                      )}
                    </div>
                  )}
                </div>

                {/* Clinical Structure Display */}
                {encounter?.triage?.clinical_structure && (
                  <div style={{ marginTop: 16, padding: 12, background: "#f0fdf4", borderRadius: 8, border: "1px solid #bbf7d0" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#0f172a", marginBottom: 10 }}>Clinical Structure</div>
                    
                    {/* Diagnoses */}
                    {encounter.triage.clinical_structure.diagnoses && 
                     Array.isArray(encounter.triage.clinical_structure.diagnoses) &&
                     encounter.triage.clinical_structure.diagnoses.length > 0 && (
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ fontSize: 11, fontWeight: 600, color: "#047857", marginBottom: 6 }}>Diagnoses:</div>
                        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12 }}>
                          {encounter.triage.clinical_structure.diagnoses.map((diag, idx) => (
                            <li key={idx} style={{ marginBottom: 4 }}>{typeof diag === 'string' ? diag : JSON.stringify(diag)}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {/* History */}
                    {encounter.triage.clinical_structure.history && (
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ fontSize: 11, fontWeight: 600, color: "#047857", marginBottom: 4 }}>History:</div>
                        <div style={{ fontSize: 12, color: "#0f172a", marginLeft: 12 }}>{encounter.triage.clinical_structure.history}</div>
                      </div>
                    )}
                    
                    {/* Symptoms */}
                    {encounter.triage.clinical_structure.symptoms && (
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ fontSize: 11, fontWeight: 600, color: "#047857", marginBottom: 4 }}>Symptoms:</div>
                        <div style={{ fontSize: 12, color: "#0f172a", marginLeft: 12 }}>{encounter.triage.clinical_structure.symptoms}</div>
                      </div>
                    )}
                    
                    {/* Recommendations */}
                    {encounter.triage.clinical_structure.recommendations && 
                     Array.isArray(encounter.triage.clinical_structure.recommendations) &&
                     encounter.triage.clinical_structure.recommendations.length > 0 && (
                      <div>
                        <div style={{ fontSize: 11, fontWeight: 600, color: "#047857", marginBottom: 6 }}>Recommendations:</div>
                        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12 }}>
                          {encounter.triage.clinical_structure.recommendations.map((rec, idx) => (
                            <li key={idx} style={{ marginBottom: 4 }}>{typeof rec === 'string' ? rec : JSON.stringify(rec)}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                <div style={{ marginTop: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 800, color: "#0f172a", marginBottom: 8 }}>
                    AI Triage Report Analysis
                  </div>
                  {clinicalAnalysisSections.map((section) => {
                    const hasValue = Array.isArray(section.value) ? section.value.length > 0 : Boolean(section.value);
                    if (!hasValue) return null;
                    return (
                      <CollapsibleSection
                        key={section.key}
                        title={section.title}
                        icon={section.icon}
                        isExpanded={expandedClinicalSections[section.key] !== false}
                        onToggle={() => toggleClinicalSection(section.key)}
                        bgColor="#ffffff"
                        borderColor="#cbd5e1"
                        headerBgColor="#f8fafc"
                      >
                        {renderClinicalValue(section.value)}
                      </CollapsibleSection>
                    );
                  })}
                </div>

              </div>

              <div style={{ marginTop: 12, fontSize: 13 }}>
                {/* Assigned Specialty */}
                {/* Assigned specialty - show from all sources */}
                <div style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <strong>Assigned specialty:</strong>
                  </div>
                  {manuallySelectedClinicians.length > 0 || assignedClinicians.length > 0 ? (
                    <div style={{ color: "#059669", fontWeight: 600 }}>
                      {[...new Set([...manuallySelectedClinicians, ...assignedClinicians].map(c => c.specialty))].join(", ")}
                    </div>
                  ) : (
                    <div style={{ color: "#475569" }}>N/A</div>
                  )}
                </div>

                {/* Manually Selected Clinicians from Scheduler */}
                {manuallySelectedClinicians.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <strong style={{ color: "#059669" }}>📋 Manually Selected from Scheduler:</strong>
                      <button
                        onClick={() => {
                          const encId = getEncounterId(encounter);
                          if (encId) {
                            navigate(`/scheduler?mode=clinician-override&encounterId=${encId}`);
                          }
                        }}
                        style={{
                          padding: "4px 12px",
                          fontSize: 11,
                          background: "#10b981",
                          color: "white",
                          border: "none",
                          borderRadius: 4,
                          cursor: "pointer",
                          fontWeight: 600
                        }}
                        title="Edit or change selected clinicians"
                      >
                        ✎ Edit Selection
                      </button>
                    </div>
                    <div style={{ color: "#059669" }}>
                      {manuallySelectedClinicians.map((c, idx) => (
                        <div key={c.id} style={{ fontSize: 12, marginBottom: 8, display: "flex", justifyContent: "space-between", alignItems: "start", padding: 8, background: "#f0fdf4", borderRadius: 4, border: "1px solid #d1fae5" }}>
                          <div style={{ flex: 1 }}>
                            <div>• {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty}</div>
                            <div style={{ fontSize: 11, color: "#047857", marginLeft: 16, marginTop: 2 }}>
                              🕒 {formatAvailabilityHours(c)}
                            </div>
                          </div>
                          <button
                            onClick={() => {
                              setManuallySelectedClinicians(manuallySelectedClinicians.filter(clinician => clinician.id !== c.id));
                            }}
                            style={{
                              padding: "4px 8px",
                              fontSize: 10,
                              background: "#dc2626",
                              color: "white",
                              border: "none",
                              borderRadius: 3,
                              cursor: "pointer",
                              fontWeight: 600,
                              marginLeft: 8,
                              whiteSpace: "nowrap"
                            }}
                            title="Remove this clinician"
                          >
                            ✕ Remove
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* AI-Recommended Assigned Clinicians */}
                {assignedClinicians.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <strong style={{ color: "#0369a1" }}>💡 AI Recommended:</strong>
                    </div>
                    <div style={{ color: "#0369a1" }}>
                      {assignedClinicians.map((c, idx) => (
                        <div key={c.id} style={{ fontSize: 12, marginBottom: 4 }}>
                          • {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Assigned Clinician Field - If neither manually selected nor AI assigned, show N/A */}
                {manuallySelectedClinicians.length === 0 && assignedClinicians.length === 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <strong>Assigned clinician:</strong>
                    </div>
                    <div style={{ color: "#475569" }}>N/A</div>
                  </div>
                )}

                {/* Ward & Bed */}
                <div style={{ marginBottom: 12 }}>
                  <div style={{ marginBottom: 8 }}>
                    <strong>Ward:</strong>
                    {selectedWard ? (
                      <div style={{ color: "#059669", fontSize: 12, marginTop: 4 }}>
                        {selectedWard.name || selectedWard.wardName || selectedWard.department || "Ward"}
                      </div>
                    ) : (
                      <div style={{ color: "#475569", fontSize: 12, marginTop: 4 }}>N/A</div>
                    )}
                  </div>
                  <div>
                    <strong>Bed:</strong>
                    {selectedBed ? (
                      <div style={{ color: "#059669", fontSize: 12, marginTop: 4 }}>
                        {selectedBed.bedId || selectedBed.name || `Bed ${selectedBed.roomNumber}`} ({selectedBed.bedType || "Standard"})
                      </div>
                    ) : (
                      <div style={{ color: "#475569", fontSize: 12, marginTop: 4 }}>N/A</div>
                    )}
                  </div>
                </div>

                <div><strong>Waiting severity:</strong> {waitingSeverity ?? "N/A"}</div>
              </div>

              {/* Changes Summary Panel */}
              {hasAnyEdits && (
                <div
                  style={{
                    marginTop: 12,
                    padding: 12,
                    background: "#fef3c7",
                    border: "1px solid #fcd34d",
                    borderRadius: 6
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#92400e", marginBottom: 8 }}>
                    ⚠ Unsaved AI Field Changes
                  </div>
                  <div style={{ fontSize: 11, color: "#b45309", lineHeight: 1.6 }}>
                    {aiSummaryEdited && <div>• AI Summary modified</div>}
                    {physicianEdited && <div>• Physician Assigned modified</div>}
                    {assignedClinicians.length > 0 && <div>• Assigned Clinicians ({assignedClinicians.length})</div>}
                    {discriminatorsEdited && <div>• Clinical Discriminators modified</div>}
                    {satsColorEdited && <div>• SATS Color Code modified</div>}
                    {satsPointsEdited && <div>• SATS Points modified</div>}
                    {specialtyEdited && <div>• Assigned Specialty modified</div>}
                    {wardEdited && <div>• Ward modified</div>}
                  </div>
                  <button
                    onClick={handleSaveChanges}
                    disabled={saving}
                    style={{
                      marginTop: 10,
                      padding: "8px 14px",
                      fontSize: 12,
                      fontWeight: 700,
                      borderRadius: 4,
                      background: saving ? "#d1d5db" : "#f59e0b",
                      color: "white",
                      border: "none",
                      cursor: saving ? "not-allowed" : "pointer",
                      transition: "all 0.2s",
                      textTransform: "uppercase",
                      letterSpacing: 0.3
                    }}
                  >
                    {saving ? "Saving…" : "Apply Changes to Record"}
                  </button>
                </div>
              )}

              {renderPatientHistoryEvidence({
                ...(triage?.evidence_provenance ?? aiReport?.evidence_provenance ?? {}),
                ...historyEvidence
              })}
              {renderEvidenceTrail(triage?.similar_cases_rag ?? triage?.ragData?.cases ?? [])}
            </div>
          </section>


        </div>

        <aside>
          {/* -- right column: vitals, nurse notes, recommendations & selection UI -- */}
          <section className="block">
            <h4 className="block-label">Patient Vitals & Queue</h4>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: 8 }}>
              {/* Temperature */}
              <div style={{ padding: "10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 4 }}>Temp</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{fmt(vitals?.temp, "°C")}</div>
              </div>

              {/* BP Combined */}
              <div style={{ padding: "10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 4 }}>BP</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{displayBP}</div>
              </div>

              {/* HR */}
              <div style={{ padding: "10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 4 }}>HR</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{fmt(vitals?.hr)}</div>
              </div>

              {/* O2 */}
              <div style={{ padding: "10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 4 }}>O₂</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{fmt(vitals?.o2, "%")}</div>
              </div>

              {/* Resp */}
              <div style={{ padding: "10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 4 }}>Resp</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{fmt(vitals?.resp)}</div>
              </div>

              {/* AVPU */}
              <div style={{ padding: "10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 4 }}>AVPU</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{fmt(vitals?.avpu)}</div>
              </div>
            </div>

            {/* Alternative BP readings - collapsible */}
            {(vitals?.bp_left_systolic || vitals?.bp_right_systolic) && (
              <details style={{ marginTop: 12, fontSize: 12 }}>
                <summary style={{ cursor: "pointer", fontSize: 12, fontWeight: 600, color: "#475569", userSelect: "none" }}>
                  📊 Detailed BP (Left/Right)
                </summary>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: 8 }}>
                  {vitals?.bp_left_systolic && (
                    <div style={{ padding: "8px", background: "#f0f9ff", borderRadius: 4, border: "1px solid #bfdbfe" }}>
                      <div style={{ fontSize: 10, color: "#0c4a6e", fontWeight: 600, marginBottom: 2 }}>BP Left</div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#0369a1" }}>
                        {fmt(vitals?.bp_left_systolic)}/{fmt(vitals?.bp_left_diastolic)}
                      </div>
                    </div>
                  )}
                  {vitals?.bp_right_systolic && (
                    <div style={{ padding: "8px", background: "#f0f9ff", borderRadius: 4, border: "1px solid #bfdbfe" }}>
                      <div style={{ fontSize: 10, color: "#0c4a6e", fontWeight: 600, marginBottom: 2 }}>BP Right</div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#0369a1" }}>
                        {fmt(vitals?.bp_right_systolic)}/{fmt(vitals?.bp_right_diastolic)}
                      </div>
                    </div>
                  )}
                </div>
              </details>
            )}

            {/* Waiting Status - compact */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: 12, fontSize: 12 }}>
              <div>
                <span style={{ color: "#64748b", fontWeight: 600 }}>Waiting:</span> <span style={{ fontWeight: 600, color: encounter?.isWaiting ? "#dc2626" : "#10b981" }}>{encounter?.isWaiting ? "Yes" : "No"}</span>
              </div>
              <div style={{ textAlign: "right" }}>
                <span style={{ color: "#64748b", fontWeight: 600 }}>Time:</span> <span style={{ fontWeight: 500 }}>{timeAgo(encounter?.createdAt)}</span>
              </div>
            </div>
          </section>

          {/* -- recommended clinicians panel (SEPARATED by role) -- */}
          <section className="block" style={{ marginTop: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12, gap: 16 }}>
              <div>
                <h4 className="block-label" style={{ marginBottom: 4 }}>Recommended Clinicians</h4>
                {manuallySelectedClinicians.length > 0 && (
                  <div style={{ fontSize: 12, color: "#059669", fontWeight: 600 }}>
                    📋 {manuallySelectedClinicians.length} clinician{manuallySelectedClinicians.length !== 1 ? 's' : ''} manually selected from scheduler
                  </div>
                )}
              </div>
              <button
                onClick={() => {
                  const encId = getEncounterId(encounter);
                  if (encId) {
                    navigate(`/scheduler?mode=clinician-override&encounterId=${encId}`);
                  }
                }}
                style={{
                  padding: "6px 12px",
                  fontSize: 11,
                  background: "#8b5cf6",
                  color: "white",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                  fontWeight: 600
                }}
                title="Manually pick clinicians from the staff scheduler"
              >
                Manually Select from Scheduler
              </button>
            </div>

            {loadingDocs && <div>Loading recommendations...</div>}

            {!loadingDocs && recommendedClinicians.length === 0 && (
              <div style={{ color: "#64748b" }}>No recommendations available</div>
            )}

            {/* RECOMMENDED PHYSICIANS SECTION */}
            {recommendedClinicians.filter(c => c.role === 'PHYSICIAN').length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#2563eb", marginBottom: 8, paddingBottom: 6, borderBottom: "2px solid #bfdbfe" }}>
                  👨‍⚕️ RECOMMENDED PHYSICIANS ({recommendedClinicians.filter(c => c.role === 'PHYSICIAN').length})
                </div>
                <div style={{ marginTop: 8 }}>
                  {recommendedClinicians
                    .filter(c => c.role === 'PHYSICIAN')
                    .map((clinician, idx) => {
                      const isAssigned = assignedClinicians.some(a => String(a.id) === String(clinician.id));
                      const roleColor = '#2563eb';
                      const roleBg = '#eff6ff';
                      
                      return (
                        <div
                          key={clinician.id}
                          style={{
                            padding: 10,
                            borderRadius: 8,
                            marginBottom: 8,
                            cursor: "pointer",
                            border: isAssigned ? `2px solid ${roleColor}` : "1px solid #e2e8f0",
                            background: isAssigned ? roleBg : "white",
                            transition: "all 0.12s"
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "start" }}>
                            <input
                              type="checkbox"
                              checked={isAssigned}
                              onChange={(e) => {
                                e.stopPropagation();
                                if (isAssigned) {
                                  setAssignedClinicians(assignedClinicians.filter(a => String(a.id) !== String(clinician.id)));
                                } else {
                                  setAssignedClinicians([...assignedClinicians, clinician]);
                                }
                              }}
                              style={{ marginTop: 4, cursor: "pointer" }}
                            />
                            
                            <div style={{ minWidth: 0, flex: 1 }}>
                              <div style={{ fontWeight: 800 }}>
                                {clinician.name}{" "}
                                <span style={{ fontSize: 11, color: roleColor, background: roleBg, padding: "2px 6px", borderRadius: 4, marginLeft: 8 }}>
                                  MD
                                </span>
                              </div>
                              <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
                                {clinician.specialty}
                              </div>
                              <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
                                🕒 {formatAvailabilityHours(clinician)}
                              </div>
                            </div>
                          </div>

                          {clinician.reason && (
                            <div style={{ marginTop: 8, fontSize: 13, color: "#334155", marginLeft: 24 }}>
                              {clinician.reason}
                            </div>
                          )}
                        </div>
                      );
                    })}
                </div>
              </div>
            )}

            {/* RECOMMENDED NURSES SECTION - TOP 10 MOST RELEVANT */}
            {recommendedClinicians.filter(c => c.role === 'NURSE').length > 0 && (
              <div style={{ marginBottom: 16 }}>
                {(() => {
                  const allNurses = recommendedClinicians.filter(c => c.role === 'NURSE');
                  // Sort by score (highest first) and limit to top 10
                  const topNurses = allNurses
                    .sort((a, b) => (b.score || 0) - (a.score || 0))
                    .slice(0, 10);
                  return (
                    <>
                      <div style={{ fontSize: 12, fontWeight: 700, color: "#7c3aed", marginBottom: 8, paddingBottom: 6, borderBottom: "2px solid #d8b4fe" }}>
                        👩‍⚕️ RECOMMENDED NURSES ({topNurses.length} of {allNurses.length})
                      </div>
                      <div style={{ marginTop: 8 }}>
                        {topNurses
                          .map((clinician, idx) => {
                      const isAssigned = assignedClinicians.some(a => String(a.id) === String(clinician.id));
                      const roleColor = '#7c3aed';
                      const roleBg = '#faf5ff';
                      
                      return (
                        <div
                          key={clinician.id}
                          style={{
                            padding: 10,
                            borderRadius: 8,
                            marginBottom: 8,
                            cursor: "pointer",
                            border: isAssigned ? `2px solid ${roleColor}` : "1px solid #e2e8f0",
                            background: isAssigned ? roleBg : "white",
                            transition: "all 0.12s"
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "start" }}>
                            <input
                              type="checkbox"
                              checked={isAssigned}
                              onChange={(e) => {
                                e.stopPropagation();
                                if (isAssigned) {
                                  setAssignedClinicians(assignedClinicians.filter(a => String(a.id) !== String(clinician.id)));
                                } else {
                                  setAssignedClinicians([...assignedClinicians, clinician]);
                                }
                              }}
                              style={{ marginTop: 4, cursor: "pointer" }}
                            />
                            
                            <div style={{ minWidth: 0, flex: 1 }}>
                              <div style={{ fontWeight: 800 }}>
                                {clinician.name}{" "}
                                <span style={{ fontSize: 11, color: roleColor, background: roleBg, padding: "2px 6px", borderRadius: 4, marginLeft: 8 }}>
                                  RN
                                </span>
                              </div>
                              <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
                                {clinician.specialty}
                              </div>
                              <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
                                🕒 {formatAvailabilityHours(clinician)}
                              </div>
                            </div>
                          </div>

                          {clinician.reason && (
                            <div style={{ marginTop: 8, fontSize: 13, color: "#334155", marginLeft: 24 }}>
                              {clinician.reason}
                            </div>
                          )}
                        </div>
                      );
                    })}
                      </div>
                    </>
                  );
                })()}
              </div>
            )}

            {manuallySelectedClinicians.length > 0 && (
              <div style={{ marginTop: 12, padding: 10, background: "#f0fdf4", borderRadius: 8, fontSize: 13, borderLeft: "4px solid #22c55e" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <strong>📋 Manually Selected Clinicians ({manuallySelectedClinicians.length}):</strong>
                  <button
                    onClick={() => {
                      const encId = getEncounterId(encounter);
                      if (encId) {
                        navigate(`/scheduler?mode=clinician-override&encounterId=${encId}`);
                      }
                    }}
                    style={{
                      padding: "4px 12px",
                      fontSize: 11,
                      background: "#10b981",
                      color: "white",
                      border: "none",
                      borderRadius: 4,
                      cursor: "pointer",
                      fontWeight: 600
                    }}
                    title="Edit or change selected clinicians"
                  >
                    ✎ Edit
                  </button>
                </div>
                <div style={{ marginTop: 6 }}>
                  {manuallySelectedClinicians.map((c) => (
                    <div key={c.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "start", fontSize: 12, marginBottom: 6, padding: 6, background: "white", borderRadius: 4 }}>
                      <div style={{ flex: 1 }}>
                        <div>• {c.name} ({c.role === 'PHYSICIAN' ? 'MD' : 'RN'}) - {c.specialty}</div>
                        <div style={{ fontSize: 11, color: "#059669", marginTop: 2 }}>✓ From Scheduler</div>
                      </div>
                      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        <div style={{ fontSize: 11, color: "#64748b" }}>
                          🕒 {formatAvailabilityHours(c)}
                        </div>
                        <button
                          onClick={() => {
                            setManuallySelectedClinicians(manuallySelectedClinicians.filter(clinician => clinician.id !== c.id));
                          }}
                          style={{
                            padding: "3px 6px",
                            fontSize: 10,
                            background: "#dc2626",
                            color: "white",
                            border: "none",
                            borderRadius: 3,
                            cursor: "pointer",
                            fontWeight: 600,
                            whiteSpace: "nowrap"
                          }}
                          title="Remove this clinician"
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* Hospitalization Choice Section - REQUIRED BEFORE WARD/BED SELECTION */}
          <section className="hospitalization-panel" style={{ marginTop: 16, padding: 12, background: "#f5f3ff", borderRadius: 8, border: "2px solid #a78bfa" }}>
            <h4 style={{ margin: "0 0 12px 0", color: "#6d28d9", fontSize: 14, fontWeight: 700 }}>
              🏥 Hospitalization Status
            </h4>
            <div className="hospitalization-choice-grid" style={{ display: "flex", gap: 12, alignItems: "center" }}>
              {/* Need Hospitalization Button */}
              <button className="hospitalization-choice"
                onClick={() => {
                  handleOpenWardSelector();
                }}
                style={{
                  flex: 1,
                  padding: "14px 16px",
                  fontSize: 13,
                  fontWeight: 700,
                  background: needsHospitalization === true ? "#10b981" : "#e0f2fe",
                  color: needsHospitalization === true ? "white" : "#0369a1",
                  border: needsHospitalization === true ? "2px solid #059669" : "2px solid #0369a1",
                  borderRadius: 6,
                  cursor: "pointer",
                  transition: "all 0.2s",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8
                }}
              >
                <span style={{ fontSize: 18 }}>🛏️</span>
                <span>Patient Needs Hospitalization</span>
                {needsHospitalization === true && <span style={{ marginLeft: 8, fontSize: 16 }}>✓</span>}
              </button>

              {/* No Hospitalization Button */}
              <button className="hospitalization-choice"
                onClick={() => {
                  setNeedsHospitalization(false);
                  setSelectedWard(null);
                  setSelectedBed(null);
                  setShowWardSelector(false);
                  setShowBedSelector(false);
                }}
                style={{
                  flex: 1,
                  padding: "14px 16px",
                  fontSize: 13,
                  fontWeight: 700,
                  background: needsHospitalization === false ? "#fca5a5" : "#fef9e7",
                  color: needsHospitalization === false ? "white" : "#854d0e",
                  border: needsHospitalization === false ? "2px solid #dc2626" : "2px solid #eab308",
                  borderRadius: 6,
                  cursor: "pointer",
                  transition: "all 0.2s",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8
                }}
              >
                <span style={{ fontSize: 18 }}>🚫</span>
                <span>Outpatient/Observation Only</span>
                {needsHospitalization === false && <span style={{ marginLeft: 8, fontSize: 16 }}>✓</span>}
              </button>
            </div>
            {needsHospitalization !== null && (
              <div style={{ marginTop: 10, fontSize: 12, color: "#6366f1", fontWeight: 600 }}>
                Status: {needsHospitalization ? "🛏️ Will select ward and bed" : "✓ Outpatient - no bed needed"}
              </div>
            )}
          </section>

          {/* Ward Selection Section - ONLY SHOW IF HOSPITALIZATION NEEDED */}
          {needsHospitalization === true && (
            <section className="hospitalization-panel" style={{ marginTop: 16, padding: 12, background: "#f0f9ff", borderRadius: 8, border: "1px solid #bae6fd" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h4 style={{ margin: 0, color: "#0369a1" }}>
                  Ward Assignment{selectedWard ? " ✓" : ""}
                </h4>
                <button
                  onClick={handleOpenWardSelector}
                  style={{
                    padding: "6px 12px",
                    fontSize: 12,
                    background: selectedWard ? "#10b981" : "#0369a1",
                    color: "white",
                    border: "none",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontWeight: 600
                  }}
                >
                  {selectedWard ? `Change Ward: ${selectedWard.name || selectedWard.wardName || selectedWard.department || "Ward"}` : "Select Ward"}
                </button>
              </div>

            {/* Ward Selector Modal/Dropdown */}
            {showWardSelector && (
              <div style={{ marginBottom: 12, padding: 12, background: "white", borderRadius: 6, border: "1px solid #bae6fd" }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "#475569" }}>
                  Available Wards:
                </div>
                {availableWards.length === 0 ? (
                  <div style={{ fontSize: 12, color: "#64748b" }}>Loading wards...</div>
                ) : (
                  <div className="hospitalization-option-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 8 }}>
                    {availableWards.map((ward) => {
                      const wardName = ward.name || ward.wardName || ward.department || "Unnamed";
                      
                      return (
                        <button
                          key={ward._id || ward.id}
                          onClick={() => handleSelectWard(ward)}
                          style={{
                            padding: "12px 10px",
                            fontSize: 12,
                            background: "#e0f2fe",
                            color: "#0369a1",
                            border: "1px solid #bae6fd",
                            borderRadius: 6,
                            cursor: "pointer",
                            fontWeight: 600,
                            transition: "all 0.2s",
                            textAlign: "center",
                            minHeight: 50,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center"
                          }}
                        >
                          <div style={{ fontWeight: 700 }}>{wardName}</div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Bed Selection Section (shows after ward selected) */}
            </section>
          )}

          {needsHospitalization === true && selectedWard && (
            <section className="hospitalization-panel" style={{ marginTop: 16, padding: 12, background: "#f0fdf4", borderRadius: 8, border: "1px solid #bbf7d0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h4 style={{ margin: 0, color: "#16a34a" }}>
                  Bed Assignment{selectedBed ? " ✓" : ""}
                </h4>
                {selectedBed && (
                  <button
                    onClick={() => setShowBedSelector(true)}
                    style={{
                      padding: "6px 12px",
                      fontSize: 12,
                      background: "#16a34a",
                      color: "white",
                      border: "none",
                      borderRadius: 4,
                      cursor: "pointer",
                      fontWeight: 600
                    }}
                  >
                    Change Bed
                  </button>
                )}
              </div>

              {/* Bed Selector Modal/Dropdown */}
              {(showBedSelector || availableBeds.length > 0) && (
                <div style={{ marginBottom: 12, padding: 12, background: "white", borderRadius: 6, border: "1px solid #bbf7d0" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "#475569" }}>
                    Available Beds in {selectedWard.name || selectedWard.wardName || "Ward"}:
                  </div>
                  {availableBeds.length === 0 ? (
                    <div style={{ fontSize: 12, color: "#64748b" }}>No available beds</div>
                  ) : (
                    <div className="hospitalization-option-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
                      {availableBeds.map((bed) => {
                        const isSelected = selectedBed && (selectedBed._id === bed._id || selectedBed.id === bed.id);
                        const bedName = bed.bedId || bed.bedNumber || `Bed ${bed.roomNumber}` || "Unknown";
                        const bedType = bed.bedType || "Standard";
                        
                        return (
                          <button
                            key={bed._id || bed.id}
                            onClick={() => handleSelectBed(bed)}
                            title={`${bedName} - ${bed.equipment?.join(", ") || "Standard equipment"}`}
                            style={{
                              padding: "12px 14px",
                              fontSize: 12,
                              background: isSelected ? "#dcfce7" : "#f0fdf4",
                              color: "#16a34a",
                              border: isSelected ? "2px solid #16a34a" : "1px solid #bbf7d0",
                              borderRadius: 6,
                              cursor: "pointer",
                              fontWeight: 600,
                              transition: "all 0.2s",
                              textAlign: "left",
                              minHeight: 90,
                              display: "flex",
                              flexDirection: "column",
                              justifyContent: "space-between"
                            }}
                          >
                            {/* Bed ID/Name */}
                            <div style={{ fontWeight: 700, marginBottom: 4 }}>{bedName}</div>
                            
                            {/* Bed Type */}
                            <div style={{ fontSize: 11, color: "#047857", marginBottom: 3 }}>
                              {bedType}
                            </div>
                            
                            {/* Equipment/Capabilities */}
                            {bed.equipment && bed.equipment.length > 0 && (
                              <div style={{ fontSize: 10, color: "#059669", marginBottom: 3, fontWeight: 500 }}>
                                🔧 {bed.equipment.slice(0, 2).join(", ")}
                                {bed.equipment.length > 2 && `+${bed.equipment.length - 2}`}
                              </div>
                            )}
                            
                            {/* Acuity Level & Isolation */}
                            <div style={{ fontSize: 10, color: "#0891b2", display: "flex", gap: 4 }}>
                              {bed.acuityLevel && <span>Level: {bed.acuityLevel}</span>}
                              {bed.canIsolate && <span>🛡️ Isolation</span>}
                            </div>
                            
                            {/* Selection Indicator */}
                            <div style={{ fontSize: 10, color: "#065f46", marginTop: 2 }}>
                              ✓ Available
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {selectedBed && !showBedSelector && (
                <div style={{ color: "#16a34a", fontSize: 12, fontWeight: 600 }}>
                  <div style={{ marginBottom: 8 }}>
                    Selected: {selectedBed.bedId || selectedBed.bedNumber || `Bed ${selectedBed.roomNumber}`}
                  </div>
                  <div style={{ fontSize: 11, color: "#475569", marginBottom: 4 }}>
                    Type: {selectedBed.bedType || "Standard"}
                  </div>
                  {selectedBed.acuityLevel && (
                    <div style={{ fontSize: 11, color: "#059669", marginBottom: 4 }}>
                      Acuity Level: {selectedBed.acuityLevel}
                    </div>
                  )}
                  {selectedBed.equipment && selectedBed.equipment.length > 0 && (
                    <div style={{ fontSize: 11, color: "#0891b2", marginBottom: 4 }}>
                      Equipment: {selectedBed.equipment.join(", ")}
                    </div>
                  )}
                  {selectedBed.canIsolate && (
                    <div style={{ fontSize: 11, color: "#7c3aed" }}>
                      🛡️ Isolation Capable
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <button 
              className="save-file-btn"
              style={{ flex: 1 }}
              onClick={() => navigate('/summary/new')}
              title="Start a new manual triage"
            >
              Back to Manual Triage
            </button>

            <button
              className="save-file-btn"
              style={{ flex: 1 }}
              onClick={async () => {
                // Determine which confirmation page to navigate to
                const encId = getEncounterId(encounter);
                if (!encId) {
                  alert("Error: Unable to identify encounter");
                  return;
                }
                
                // Check that hospitalization choice is made
                if (needsHospitalization === null) {
                  alert("Please choose whether patient needs hospitalization");
                  return;
                }
                
                // If hospitalization needed, check that a ward and bed are selected
                if (needsHospitalization === true && (!selectedWard || !selectedBed)) {
                  alert("Please select both a ward and bed for hospitalization");
                  return;
                }

                // Check that at least one clinician is selected
                if (manuallySelectedClinicians.length === 0 && assignedClinicians.length === 0) {
                  alert("Please select at least one clinician before applying changes");
                  return;
                }

                // Save the encounter with all selected fields
                setSaving(true);
                try {
                  const token = localStorage.getItem("token");
                  if (!token) {
                    alert("You must be logged in.");
                    navigate("/");
                    return;
                  }

                  // Derive specialty from physician only (not nurse), as per triage requirement
                  const derivedSpecialty = [...manuallySelectedClinicians, ...assignedClinicians]
                    .filter(c => c.role === 'PHYSICIAN')
                    .map(c => c.specialty)[0] || encounter.triage?.assigned_specialty;

                  // Build complete payload with all fields to save (matching ClinicianScheduler format)
                  const savePayload = {
                    changedFields: {
                      // Clinicians
                      assigned_clinicians: assignedClinicians.map(c => ({
                        id: c.id,
                        name: c.name,
                        role: c.role,
                        specialty: c.specialty,
                        score: c.score || 0.85
                      })),
                      manually_selected_clinicians: manuallySelectedClinicians.map(c => ({
                        id: c.id,
                        name: c.name,
                        role: c.role,
                        specialty: c.specialty,
                        score: c.score || 0.85,
                        clock_in_time: c.clock_in_time,
                        clock_out_time: c.clock_out_time,
                        shift_start: c.shift_start,
                        shift_end: c.shift_end,
                        selected_at: c.selected_at || new Date()
                      })),
                      
                      // Hospitalization status and ward/bed assignments
                      requires_hospitalization: needsHospitalization,
                      assigned_ward: needsHospitalization && selectedWard ? (selectedWard._id || selectedWard.id) : null,
                      assigned_bed: needsHospitalization && selectedBed ? (selectedBed._id || selectedBed.id) : null,
                      hospitalization_type: needsHospitalization ? 'inpatient' : 'outpatient',
                      
                      // Triage/assignment data - specialty derived from selected clinicians
                      assigned_specialty: derivedSpecialty,
                      priority_code: encounter.triage?.priority_code,
                      sats_classification: encounter.triage?.sats_classification
                    },
                    // Store full objects for display in confirmation + clinical handover
                    hospitalization_status: {
                      requires_hospitalization: needsHospitalization,
                      type: needsHospitalization ? 'Inpatient Hospitalization' : 'Outpatient/Observation'
                    },
                    ward_display_info: needsHospitalization ? selectedWard : null,
                    bed_display_info: needsHospitalization ? selectedBed : null,
                    nurseNotes: nurseNotes
                  };

                  // Debug logging
                  console.log('[Apply Changes] Hospitalization needed:', needsHospitalization);
                  console.log('[Apply Changes] Manual clinicians:', manuallySelectedClinicians);
                  console.log('[Apply Changes] Assigned clinicians:', assignedClinicians);
                  console.log('[Apply Changes] Derived specialty:', derivedSpecialty);
                  console.log('[Apply Changes] Selected ward:', selectedWard);
                  console.log('[Apply Changes] Selected bed:', selectedBed);
                  console.log('[Apply Changes] Nurse notes:', nurseNotes);
                  console.log('[Apply Changes] Full payload:', savePayload);

                  const saveRes = await fetch(`${BACKEND_URL}/api/encounters/${encId}/ai-fields`, {
                    method: 'PUT',
                    headers: {
                      'Content-Type': 'application/json',
                      Authorization: `Bearer ${token}`
                    },
                    body: JSON.stringify(savePayload)
                  });

                  if (!saveRes.ok) {
                    const errData = await saveRes.json().catch(() => ({}));
                    throw new Error(errData.message || 'Failed to save encounter');
                  }

                  console.log('Successfully saved encounter');

                  // Navigate to AI confirmed report
                  navigate(`/handover/ai/${encId}`);
                } catch (err) {
                  console.error('Error saving encounter:', err);
                  alert('Failed to save changes: ' + err.message);
                } finally {
                  setSaving(false);
                }
              }}
              disabled={saving}
              title="Apply changes and proceed to confirmation"
            >
              {saving ? "Processing..." : "Apply Changes"}
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}