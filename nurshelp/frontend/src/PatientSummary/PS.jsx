// src/PatientSummary/PS.jsx
import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import Footer from "./Components/footer";
import LeftColumn from "./Components/LeftColumn";
import RightColumn from "./Components/RightColumn";
import Vitals from "./Components/Vitals";
import TriageTabContainer from "./Components/TriageTabContainer";
import SidebarLayout from "../Dashboard/SidebarLayout";

import "./PS.css";
import "./PS_index.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/** Normalize temperature into a safe numeric value (or null).
 *  - Accepts numeric-like input, returns number with 1 decimal place.
 *  - Returns null for invalid or clinically impossible values.
 *  - Clinical guard: accept 30.0 - 45.0 °C (anything outside likely parse bug).
 */
const normalizeTemp = (val) => {
  if (val === "" || val === null || val === undefined) return null;
  const num = Number(val);
  if (Number.isNaN(num)) return null;
  if (num < 30 || num > 45) return null;
  return Number(num.toFixed(1));
};

/** Parse BP string "120/80" or "120-80" into components. */
const parseBP = (bpStr) => {
  if (!bpStr || typeof bpStr !== "string") {
    return { bp: null, systolic: null, diastolic: null };
  }
  const m = bpStr.match(/(\d{2,3})\s*[/\-]\s*(\d{2,3})/);
  if (m) {
    return {
      bp: `${m[1]}/${m[2]}`,
      systolic: Number(m[1]),
      diastolic: Number(m[2]),
    };
  }
  return { bp: bpStr, systolic: null, diastolic: null };
}

const formatDate = (isoString) => {
  if (!isoString) return 'Unknown';
  try {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short', day: 'numeric', hour: 'numeric', minute: 'numeric'
    }).format(new Date(isoString));
  } catch (e) {
    return isoString;
  }
};

const isSearchable = (query) => query && query.trim().length >= 2;

export default function PS() {
  const navigate = useNavigate();
  const { patientId } = useParams();

  // ---------- State ----------
  const [patientData, setPatientData] = useState(() => {
    const saved = localStorage.getItem("ps_patient");
    return saved
      ? JSON.parse(saved)
      : { id_number: "", name: "Patient A", surname: "", age: null, height_cm: null, symptoms: "", duration: "", painLevel: "", history: "" };
  });

  const [transcript, setTranscript] = useState([]);

  const [vitalsData, setVitalsData] = useState({
    temp: "--",
    bp: "--",
    hr: "--",
    o2: "--",
    resp: "--",
    bp_systolic: null,
    bp_diastolic: null,
    bp_left_systolic: null,
    bp_left_diastolic: null,
    bp_right_systolic: null,
    bp_right_diastolic: null,
    avpu: null
  });

  const [triageResult, setTriageResult] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loadingEncounter, setLoadingEncounter] = useState(false);
  const [activeTriageTab, setActiveTriageTab] = useState("manual");
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [ehrSummary, setEhrSummary] = useState(null);
  const [selectedMrn, setSelectedMrn] = useState(null);
  const [loadingEHR, setLoadingEHR] = useState(false);

  // ---------- Persistence ----------
  useEffect(() => { localStorage.setItem("ps_patient", JSON.stringify(patientData)); }, [patientData]);
  useEffect(() => { localStorage.setItem("ps_transcript", JSON.stringify(transcript)); }, [transcript]);
  useEffect(() => { localStorage.setItem("ps_vitals", JSON.stringify(vitalsData)); }, [vitalsData]);

  // ---------- Restore from localStorage for crash recovery (existing encounters only) ----------
  useEffect(() => {
    if (!patientId || patientId === "new") return; // Don't restore for new encounters
    const savedTranscript = localStorage.getItem("ps_transcript");
    const savedVitals = localStorage.getItem("ps_vitals");
    
    // Only restore if we don't have data yet (first load)
    if (transcript.length === 0 && savedTranscript) {
      try {
        setTranscript(JSON.parse(savedTranscript));
      } catch (e) {
        console.warn("Failed to restore transcript from localStorage");
      }
    }
    if ((!vitalsData || !vitalsData.bp || vitalsData.bp === "--") && savedVitals) {
      try {
        setVitalsData(JSON.parse(savedVitals));
      } catch (e) {
        console.warn("Failed to restore vitals from localStorage");
      }
    }
  }, [patientId]);

  const loadPatientEHR = async (mrn) => {
    if (!mrn) return;
    setLoadingEHR(true);
    setEhrSummary(null);
    setSearchError(null);

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${BACKEND_URL}/api/patients/ehr/${encodeURIComponent(mrn)}`, {
        headers,
        cache: 'no-store'
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Unable to load EHR (${res.status})`);
      }

      const data = await res.json();
      const encounters = data.encounters || [];
      setEhrSummary({
        hospital: data.hospital || null,
        totalEncounters: data.totalEncounters ?? encounters.length,
        encounters,
        lastEncounter: encounters[0] || null
      });
      setSelectedMrn(mrn);

      if (encounters.length > 0) {
        const patient = encounters[0].patient || {};
        setPatientData((prev) => ({
          ...prev,
          id_number: mrn,
          name: patient.name || prev.name || '',
          surname: patient.surname || prev.surname || '',
          gender: patient.gender || prev.gender || '',
          age: patient.age ?? prev.age ?? null,
          ageYears: patient.ageYears ?? prev.ageYears ?? null,
          ageMonths: patient.ageMonths ?? prev.ageMonths ?? null,
          height_cm: patient.height_cm ?? prev.height_cm ?? null,
          symptoms: patient.symptoms ?? prev.symptoms ?? '',
          history: patient.history ?? prev.history ?? '',
          duration: patient.duration ?? prev.duration ?? '',
          painLevel: patient.painLevel ?? prev.painLevel ?? ''
        }));
      }
    } catch (err) {
      setSearchError(err.message || 'Failed to load patient EHR');
    } finally {
      setLoadingEHR(false);
    }
  };

  const handleSearchSubmit = async () => {
    if (!isSearchable(searchQuery)) {
      setSearchError('Enter at least 2 characters to search');
      setSearchResults([]);
      return;
    }

    setSearchLoading(true);
    setSearchError(null);
    setSearchResults([]);

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${BACKEND_URL}/api/patients/search?search=${encodeURIComponent(searchQuery.trim())}`, {
        headers,
        cache: 'no-store'
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Search failed (${res.status})`);
      }

      const data = await res.json();
      setSearchResults(data.encounters || []);
      if ((data.encounters || []).length === 0) {
        setSearchError('No matching patients found. Continue with a new intake.');
      }
    } catch (err) {
      setSearchError(err.message || 'Patient search failed');
    } finally {
      setSearchLoading(false);
    }
  };

  const handlePatientSelect = async (encounter) => {
    const patient = encounter.patient || {};
    const mrn = patient.id_number || encounter.patient?.id_number;
    if (!mrn) {
      setSearchError('Selected record is missing MRN');
      return;
    }

    setSearchError(null);
    setSelectedMrn(mrn);
    setSearchQuery(mrn);
    setPatientData((prev) => ({
      ...prev,
      id_number: mrn,
      name: patient.name || prev.name || '',
      surname: patient.surname || prev.surname || '',
      gender: patient.gender || prev.gender || '',
      age: patient.age ?? prev.age ?? null,
      ageYears: patient.ageYears ?? prev.ageYears ?? null,
      ageMonths: patient.ageMonths ?? prev.ageMonths ?? null,
      height_cm: patient.height_cm ?? prev.height_cm ?? null,
      symptoms: patient.symptoms ?? prev.symptoms ?? '',
      history: patient.history ?? prev.history ?? '',
      duration: patient.duration ?? prev.duration ?? '',
      painLevel: patient.painLevel ?? prev.painLevel ?? ''
    }));

    await loadPatientEHR(mrn);
  };

  const clearPatientSelection = () => {
    setSelectedMrn(null);
    setEhrSummary(null);
    setSearchResults([]);
    setSearchError(null);
  };

  const startNewIntake = () => {
    clearPatientSelection();
    setSearchQuery('');
    setPatientData({
      id_number: '',
      name: '',
      surname: '',
      gender: '',
      age: null,
      ageYears: null,
      ageMonths: null,
      height_cm: null,
      symptoms: '',
      duration: '',
      painLevel: '',
      history: ''
    });
    setTranscript([]);
    setVitalsData({
      temp: '--',
      bp: '--',
      hr: '--',
      o2: '--',
      resp: '--',
      avpu: 'A',
      bp_systolic: null,
      bp_diastolic: null,
      bp_left_systolic: null,
      bp_left_diastolic: null,
      bp_right_systolic: null,
      bp_right_diastolic: null
    });
  };

  const getPatientAgeLabel = () => {
    if (patientData?.ageYears != null) return `${patientData.ageYears} yr${patientData.ageYears === 1 ? '' : 's'}`;
    if (patientData?.age != null) return `${patientData.age} yr${patientData.age === 1 ? '' : 's'}`;
    return 'Unknown age';
  };

  const getPatientGenderLabel = () => {
    return patientData?.gender ? String(patientData.gender).toUpperCase() : 'Unknown gender';
  };

  const getPatientSearchStatus = () => {
    if (searchLoading) return 'Searching hospital records…';
    if (searchError) return searchError;
    if (searchResults.length > 0) return `${searchResults.length} patient${searchResults.length === 1 ? '' : 's'} found`;
    return 'Search MRN, first name, or surname to preload patient data from the hospital EHR.';
  };

  // Auto open vitals when route is /summary/new + CLEAR localStorage for fresh encounter
  useEffect(() => {
    if (patientId === "new") {
      // CRITICAL: Clear all previous patient data when starting a NEW encounter
      localStorage.removeItem("ps_patient");
      localStorage.removeItem("ps_transcript");
      localStorage.removeItem("ps_vitals");
      setPatientData({});
      setTranscript([]);
      setVitalsData({ temp: "--", bp: "", hr: "", o2: "", resp: "", avpu: "A", bp_systolic: null, bp_diastolic: null, bp_left_systolic: null, bp_left_diastolic: null, bp_right_systolic: null, bp_right_diastolic: null });
      setTriageResult(null);
      setShowForm(true);
    }
  }, [patientId]);

  // ---------- Load existing encounter ----------
  useEffect(() => {
    let mounted = true;
    const loadEncounter = async () => {
      if (!patientId || patientId === "new") return;
      setLoadingEncounter(true);
      try {
        // CRITICAL: Clear localStorage before loading to prevent ghost data from previous encounters
        if (patientId !== "new") {
          localStorage.removeItem("ps_patient");
          localStorage.removeItem("ps_transcript");
          localStorage.removeItem("ps_vitals");
        }

        const res = await fetch(`${BACKEND_URL}/api/encounters/${patientId}`, {
          headers: {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache"
          },
          cache: "no-store"
        });
        if (!res.ok) throw new Error("Failed to fetch encounter");
        const body = await res.json();
        const encounter = body?.encounter ?? body;
        if (!mounted || !encounter) return;

        setPatientData(encounter.patient || {});
        setTranscript(encounter.transcript || []);

        // sanitize / normalize vitals loaded from backend
        setVitalsData({
          temp: normalizeTemp(encounter.vitals?.temp) ?? "--",
          bp: encounter.vitals?.bp ?? (encounter.triage?.vitals_parsed?.bp ?? "--"),
          hr: encounter.vitals?.hr ?? "--",
          o2: encounter.vitals?.o2 ?? "--",
          resp: encounter.vitals?.resp ?? "--",
          bp_systolic: (typeof encounter.vitals?.bp_systolic === "number") ? encounter.vitals.bp_systolic : (encounter.triage?.vitals_parsed?.bp_systolic ?? null),
          bp_diastolic: (typeof encounter.vitals?.bp_diastolic === "number") ? encounter.vitals.bp_diastolic : (encounter.triage?.vitals_parsed?.bp_diastolic ?? null),
          bp_left_systolic: encounter.vitals?.bp_left_systolic ?? null,
          bp_left_diastolic: encounter.vitals?.bp_left_diastolic ?? null,
          bp_right_systolic: encounter.vitals?.bp_right_systolic ?? null,
          bp_right_diastolic: encounter.vitals?.bp_right_diastolic ?? null,
          avpu: encounter.vitals?.avpu ?? encounter.triage?.vitals_parsed?.avpu ?? null
        });

        setTriageResult(encounter.triage || null);
        setShowForm(false);
      } catch (err) {
        console.warn("Unable to load encounter:", err);
      } finally {
        if (mounted) setLoadingEncounter(false);
      }
    };
    loadEncounter();
    return () => { mounted = false; };
  }, [patientId]);

  // ---------- Light validation before sending ----------
  const validateVitals = (v) => {
    if (!v?.bp || v.bp === "--") return { ok: false, message: "BP is required" };
    const tempNorm = normalizeTemp(v.temp);
    if (v.temp && tempNorm === null) return { ok: false, message: "Invalid temperature (expect 30–45 °C)" };
    return { ok: true };
  };

  // ---------- Extract clinical handover context from JWT (Dynamic from Login) ----------
  const getTriageContext = () => {
    // Decode JWT token to get hospitalId and triageNurseId
    let hospitalId = null;
    let triageNurseId = null;
    let hospitalName = null;
    
    try {
      const token = localStorage.getItem('token');
      if (token) {
        // Decode JWT payload (middle part)
        const payload = token.split('.')[1];
        const decoded = JSON.parse(atob(payload));
        
        hospitalId = decoded.hospitalId;      // ObjectId of hospital
        hospitalName = decoded.hospitalName;  // Hospital name
        triageNurseId = decoded.id;           // Staff member's ID (who is triaging)
        
        console.log(`[JWT DECODED] Hospital: ${hospitalName} (${hospitalId})`);
        console.log(`[JWT DECODED] Triage Nurse: ${triageNurseId}`);
      }
    } catch (e) {
      console.error("[JWT DECODE ERROR]", e);
    }
    
    // Fallbacks if JWT missing
    if (!hospitalId) {
      hospitalId = localStorage.getItem('currentHospital') || 'UNKNOWN_HOSPITAL';
    }
    if (!triageNurseId) {
      triageNurseId = localStorage.getItem('userId') || 'UNKNOWN_NURSE';
    }
    
    // Get patientId from form
    const patientId = patientData?.id_number || patientData?._id || 'NEW_PATIENT';
    
    console.log(`[TRIAGE CONTEXT] Hospital: ${hospitalId}`);
    console.log(`[TRIAGE CONTEXT] Triage Nurse: ${triageNurseId}`);
    console.log(`[TRIAGE CONTEXT] Patient: ${patientId}`);
    
    return { hospitalId, triageNurseId, patientId, hospitalName };
  };

  // ---------- Build strict payload for backend ----------
  const buildPayload = () => {
    const safeTranscript = Array.isArray(transcript) ? transcript : [{ id: 1, type: "note", text: String(transcript || "") }];
    const bpParsed = parseBP(String(vitalsData?.bp ?? ""));

    const bpFinal = (bpParsed.systolic && bpParsed.diastolic) ? `${bpParsed.systolic}/${bpParsed.diastolic}` : null;

    // Get clinical handover context (from JWT, not hardcoded)
    const { hospitalId, triageNurseId, patientId, hospitalName } = getTriageContext();

    console.log(`\n📤 [AI TRIAGE] Clinical Handover Payload:`);
    console.log(`  - Hospital: ${hospitalName || hospitalId}`);
    console.log(`  - Patient: ${patientId}`);
    console.log(`  - Triage Nurse: ${triageNurseId}`);
    console.log(`  - Endpoint: POST /api/triage/hospital-request`);

    return {
      // ===== CLINICAL HANDOVER CONTEXT (from JWT) =====
      hospitalId,           // ObjectId - retrieved from JWT
      patientId,            // Patient identifier
      triageNurseId,        // Staff ID from JWT (who is triaging)
      
      // ===== PATIENT & VITALS DATA =====
      patient: {
        id_number: patientData?.id_number ?? "",
        name: patientData?.name ?? "",
        surname: patientData?.surname ?? "",
        gender: patientData?.gender ?? "",
        // Support both new (ageYears/ageMonths) and legacy (age) formats
        age: patientData?.ageYears ?? patientData?.age ?? null,
        ageMonths: patientData?.ageMonths ?? null,
        height_cm: patientData?.height_cm ?? null,
        symptoms: patientData?.symptoms ?? "",
        history: patientData?.history ?? "",
        duration: patientData?.duration ?? "",
        painLevel: patientData?.painLevel ?? ""
      },
      vitals: {
        temp: normalizeTemp(vitalsData?.temp),
        bp: bpFinal,
        bp_left_systolic: vitalsData?.bp_left_systolic ?? null,
        bp_left_diastolic: vitalsData?.bp_left_diastolic ?? null,
        bp_right_systolic: vitalsData?.bp_right_systolic ?? null,
        bp_right_diastolic: vitalsData?.bp_right_diastolic ?? null,
        hr: vitalsData?.hr === "--" ? null : (vitalsData?.hr != null ? Number(vitalsData.hr) : null),
        o2: vitalsData?.o2 === "--" ? null : (vitalsData?.o2 != null ? Number(vitalsData.o2) : null),
        resp: vitalsData?.resp === "--" ? null : (vitalsData?.resp != null ? Number(vitalsData.resp) : null),
        bp_systolic: bpParsed.systolic ?? null,
        bp_diastolic: bpParsed.diastolic ?? null,
        avpu: vitalsData?.avpu ?? null
      },
      transcript: safeTranscript,
      arrival_time: new Date().toISOString(),
      
      // CRITICAL: Use LLM_ONLY mode to prevent redundant Flask calls
      // This skips the initial RAG/ESI_Engine call and goes directly to LLM synthesis
      // Reduces duplicate GGUF inferences from 2 calls to 1 call
      computeMode: 'LLM_ONLY'
    };
  };


  // ---------- Submit AI triage (with clinical handover) ----------
  const handleSavePatient = async () => {
    const vCheck = validateVitals(vitalsData);
    if (!vCheck.ok) return alert(vCheck.message);

    // CRITICAL: Prevent double-submit during async operation
    if (saving) {
      console.warn(`[AI TRIAGE] Request already in progress - ignoring duplicate submit`);
      return;
    }

    setSaving(true);
    try {
      const payload = buildPayload();
      const { hospitalId, triageNurseId, patientId, hospitalName } = getTriageContext();
      
      console.log(`\n✅ [AI TRIAGE] Clinical handover context extracted:`);
      console.log(`  - Hospital: ${hospitalName || hospitalId}`);
      console.log(`  - Triage Nurse: ${triageNurseId}`);
      console.log(`  - Patient: ${patientId}`);
      console.log(`  - Endpoint: POST /api/triage/hospital-request`);
      console.log(`  - Vital Signs: ${payload.vitals.temp}°C, BP ${payload.vitals.bp}, HR ${payload.vitals.hr}, O2 ${payload.vitals.o2}`);
      console.log(`  - Chief Complaint: "${payload.patient.symptoms.substring(0, 60)}..."`);
      
      // DEBUG: Show exact payload structure before sending
      console.log(`\n[AI TRIAGE] 🔍 DEBUG: Payload structure before sending:`);
      console.log(`  - payload.patient.symptoms type: ${typeof payload.patient.symptoms}`);
      console.log(`  - payload.patient.symptoms value: ${JSON.stringify(payload.patient.symptoms)}`);
      console.log(`  - payload.patient.symptoms length: ${payload.patient.symptoms?.length || 'N/A'}`);
      console.log(`  - payload.patient object keys: ${Object.keys(payload.patient).join(', ')}`);
      console.log(`  - Full payload size: ${JSON.stringify(payload).length} bytes`);
      
      console.log(`[AI TRIAGE] 📮 Sending request to backend...`);
      
      // Call AI triage endpoint with clinical handover context
      // Use strict timeout and abort signal to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 360000); // 6 minute max timeout for GGUF inference
      
      try {
        const payloadStr = JSON.stringify(payload);
        console.log(`[AI TRIAGE] Raw JSON payload size: ${payloadStr.length} bytes`);
        
        const res = await fetch(`${BACKEND_URL}/api/triage/hospital-request`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payloadStr,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          console.error(`❌ [AI TRIAGE] HTTP ${res.status}:`, errBody);
          throw new Error(errBody.message || `Triage failed (${res.status}) - ${errBody.error || 'Unknown error'}`);
        }
        
        const data = await res.json();
        
        console.log(`[AI TRIAGE] 📮 Response received from backend - Status: ${res.status}`);
        console.log(`[AI TRIAGE]   - Response keys: ${Object.keys(data).join(', ')}`);
        console.log(`[AI TRIAGE]   - Has encounter: ${!!data.encounter}`);
        console.log(`[AI TRIAGE]   - Has triage: ${!!data.triage}`);
        console.log(`[AI TRIAGE]   - Has data: ${!!data.data}`);
        
        // Backend returns response structure:
        // { success: true, encounter: {...}, triage: {...}, data: {...} }
        const enc = data.encounter;
        
        if (!enc) {
          console.error("❌ [AI TRIAGE] No encounter in response. Full response:", JSON.stringify(data, null, 2));
          throw new Error("Backend did not return encounter object - check logs for response structure");
        }

        console.log(`[AI TRIAGE] ✅ Response received successfully`);
        console.log(`[AI TRIAGE]   - Encounter ID: ${enc._id}`);
        console.log(`[AI TRIAGE]   - Encounter keys: ${Object.keys(enc).join(', ')}`);
        console.log(`[AI TRIAGE]   - Has triage object: ${!!enc.triage}`);
        console.log(`[AI TRIAGE]   - Has aiTriageReport: ${!!enc.triage?.aiTriageReport}`);
        
        if (enc.triage?.aiTriageReport) {
          console.log(`[AI TRIAGE]   - aiTriageReport keys: ${Object.keys(enc.triage.aiTriageReport).join(', ')}`);
          console.log(`[AI TRIAGE]   - ESI Level: ${enc.triage.aiTriageReport.esi_level || 'N/A'}`);
          console.log(`[AI TRIAGE]   - Specialty: ${enc.triage.aiTriageReport.specialty || 'N/A'}`);
          console.log(`[AI TRIAGE]   - Clinical Summary: ${enc.triage.aiTriageReport.clinical_summary ? 'Present (' + enc.triage.aiTriageReport.clinical_summary.length + ' chars)' : 'EMPTY'}`);
          console.log(`[AI TRIAGE]   - Clinical Discriminators: ${(enc.triage.aiTriageReport.clinical_discriminators || []).length} items - ${JSON.stringify((enc.triage.aiTriageReport.clinical_discriminators || []).slice(0, 2))}`);
          console.log(`[AI TRIAGE]   - Pathophysiology: ${enc.triage.aiTriageReport.pathophysiology ? 'Present (' + enc.triage.aiTriageReport.pathophysiology.length + ' chars)' : 'EMPTY'}`);
          console.log(`[AI TRIAGE]   - Differential Diagnoses: ${(enc.triage.aiTriageReport.differential_diagnoses || []).length} items`);
        }

        // CRITICAL FIX: Use encounter._id directly (not patientId from context)
        // Navigate with full encounter object to avoid re-fetching
        console.log(`[AI TRIAGE] 🧭 Navigating to PatientReport view...`);
        navigate(`/summary/${enc._id}`, { 
          state: { 
            encounter: enc,
            triagePayload: data.data || data.triage || {},
            fromTriage: true
          },
          replace: false
        });
      } catch (fetchErr) {
        clearTimeout(timeoutId);
        if (fetchErr.name === 'AbortError') {
          throw new Error('Request timeout - GGUF inference took too long (6+ minutes)');
        }
        throw fetchErr;
      }
    } catch (err) {
      console.error(`\n❌ [AI TRIAGE] Error:`, err.message);
      alert("❌ AI Triage failed:\n" + (err.message || "Unknown error"));
    } finally {
      setSaving(false);
    }
  };

  // ---------- Render ----------
  const storedUser = JSON.parse(localStorage.getItem("user") || "{}");
  const displayName = storedUser?.name || storedUser?.username || "Staff";

  return (
    <SidebarLayout username={displayName} handleLogout={() => {}}>
      <div className="ps-main-page-wrapper">
        <div className="app-wrapper" style={{ filter: saving ? "blur(4px)" : "none", transition: "filter 0.3s ease", pointerEvents: saving ? "none" : "auto" }}>
            <div className="ps-page-header">
            <div className="ps-page-headline">
              <span className="eyebrow-label">Triage Intake</span>
              <h1>Triage with Health Record Library Search </h1>
            </div>
          
          </div>

          <TriageTabContainer
            activeTab={activeTriageTab}
            onTabChange={(tab) => setActiveTriageTab(tab)}
          />

          <main className="main-container">
          <div className="dashboard-grid" style={{ display: "flex", gap: 20, padding: 20 }}>
            <LeftColumn
              data={patientData}
              setData={setPatientData}
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              searchResults={searchResults}
              searchLoading={searchLoading}
              searchError={searchError}
              selectedMrn={selectedMrn}
              handleSearchSubmit={handleSearchSubmit}
              handlePatientSelect={handlePatientSelect}
              loadingEHR={loadingEHR}
              clearPatientSelection={clearPatientSelection}
            />
            <RightColumn
              vitalsData={vitalsData}
              triage={triageResult}
              activeTab={activeTriageTab}
              onOpenForm={() => setShowForm(true)}
              onManualTriage={() => {
                const state = {
                  patientData,
                  vitalsData,
                  ehrSummary,
                  selectedMrn
                };
                localStorage.setItem("manual_triage_report_state", JSON.stringify(state));
                navigate("/manual-triage-report", { state });
              }}
              onAiTriage={handleSavePatient}
              saving={saving}
            />
          </div>
        </main>

          <Footer />
        </div>

        {/* AI Loading Overlay */}
        {saving && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "rgba(0, 0, 0, 0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 9999,
          backdropFilter: "blur(2px)"
        }}>
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 24,
            background: "white",
            padding: "48px 32px",
            borderRadius: "12px",
            boxShadow: "0 10px 40px rgba(0, 0, 0, 0.2)"
          }}>
            {/* Spinner */}
            <div style={{
              width: 48,
              height: 48,
              border: "4px solid #e5e7eb",
              borderTop: "4px solid #3b82f6",
              borderRadius: "50%",
              animation: "spin 1s linear infinite"
            }} />
            {/* Text */}
            <div style={{
              fontSize: "18px",
              fontWeight: 600,
              color: "#1f2937",
              textAlign: "center"
            }}>
              ⏳ Waiting for AI reasoning
            </div>
            <div style={{
              fontSize: "13px",
              color: "#6b7280",
              textAlign: "center",
              maxWidth: 300
            }}>
              Processing vitals and clinical data...
            </div>
          </div>
          <style>{`
            @keyframes spin {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      )}

        {/* Vitals modal */}
        {showForm && (
          <Vitals
            vitalsData={vitalsData}
            onComplete={() => setShowForm(false)}
            onCancel={() => setShowForm(false)}
            setVitalsData={(v) => {
              // merge parsed bp components if present and normalize temperature
              const parsed = parseBP(String(v?.bp ?? ""));
              setVitalsData((prev) => ({
                ...prev,
                ...v,
                temp: normalizeTemp(v?.temp),
                bp: parsed.bp ?? v.bp ?? prev.bp,
                bp_systolic: parsed.systolic ?? (v.bp_systolic ?? prev.bp_systolic),
                bp_diastolic: parsed.diastolic ?? (v.bp_diastolic ?? prev.bp_diastolic)
              }));
            }}
          />
        )}
      </div>
    </SidebarLayout>
  );
}