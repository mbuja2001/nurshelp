// src/PatientSummary/PS.jsx
import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useRef } from "react";

import Footer from "./Components/footer";
import LeftColumn from "./Components/LeftColumn";
import RightColumn from "./Components/RightColumn";
import Vitals from "./Components/Vitals";
import TriageTabContainer from "./Components/TriageTabContainer";
import SidebarLayout from "../Dashboard/SidebarLayout";

import "./PS.css";
import "./PS_index.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const getCleanPatientData = () => ({
  id_number: null,
  name: null,
  surname: null,
  gender: null,
  age: null,
  ageYears: null,
  ageMonths: null,
  height_cm: null,
  symptoms: null,
  duration: null,
  painLevel: null,
  history: null,
  criticalHistoryStatus: '',
  criticalHistory: {
    allergies: [],
    adverseReactions: [],
    medications: [],
    conditions: [],
    implantsOrProcedures: []
  },
});

const readCachedValue = (key, fallback) => {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch (error) {
    return fallback;
  }
};

const getCleanVitalsData = () => ({
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
  avpu: null,
  mobility: "Walks",
  traumaPresent: false
});

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

const formatClinicalTimestamp = (value) => {
  if (!value) return 'No date';
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return 'No date';
    return new Intl.DateTimeFormat('en-US', {
      month: 'short', day: 'numeric', year: 'numeric'
    }).format(date);
  } catch (e) {
    return 'No date';
  }
};

const isSearchable = (query) => query && query.trim().length >= 2;

const asList = (value) => Array.isArray(value)
  ? value.filter((item) => item !== null && item !== undefined && String(item).trim() !== '')
  : (value ? String(value).split(',').map((item) => item.trim()).filter(Boolean) : []);

const normalizeCriticalHistory = (source = {}) => {
  const history = source?.criticalHistory || source?.critical_history || {};
  const fallback = source?.criticalHistory || source?.critical_history || source || {};

  return {
    allergies: asList(history?.allergies ?? fallback?.allergies ?? []),
    adverseReactions: asList(history?.adverseReactions ?? history?.adverse_reactions ?? fallback?.adverseReactions ?? fallback?.adverse_reactions ?? []),
    medications: asList(history?.medications ?? fallback?.medications ?? []),
    conditions: asList(history?.conditions ?? fallback?.conditions ?? []),
    implantsOrProcedures: asList(history?.implantsOrProcedures ?? history?.implants_or_procedures ?? fallback?.implantsOrProcedures ?? fallback?.implants_or_procedures ?? [])
  };
};

const extractCriticalHistoryFromPatient = (patient = {}) => {
  const history = normalizeCriticalHistory(patient);
  const explicitStatus = patient?.criticalHistoryStatus || patient?.critical_history_status || patient?.criticalHistory?.status || '';
  const normalizedStatus = explicitStatus === 'reported' || explicitStatus === 'none_established' ? explicitStatus : '';

  return {
    criticalHistoryStatus: normalizedStatus,
    criticalHistory: history
  };
};

const extractCriticalHistoryFromEncounters = (encounters = []) => {
  const sources = encounters.flatMap((encounter) => [
    encounter?.patient || {},
    encounter || {}
  ]);
  const criticalHistory = sources.reduce((merged, source) => {
    const current = extractCriticalHistoryFromPatient(source).criticalHistory;
    Object.keys(merged).forEach((field) => {
      merged[field] = [...new Set([...merged[field], ...current[field]])];
    });
    return merged;
  }, {
    allergies: [],
    adverseReactions: [],
    medications: [],
    conditions: [],
    implantsOrProcedures: []
  });
  const hasExplicitReportedHistory = sources.some((source) => (
    source?.criticalHistoryStatus === 'reported' || source?.critical_history_status === 'reported'
  ));
  const hasCriticalValues = Object.values(criticalHistory).some((items) => items.length > 0);

  return {
    criticalHistoryStatus: hasExplicitReportedHistory || hasCriticalValues ? 'reported' : '',
    criticalHistory
  };
};

export default function PS() {
  const navigate = useNavigate();
  const { patientId } = useParams();

  // ---------- State ----------
  const [patientData, setPatientData] = useState(() => readCachedValue('ps_patient', getCleanPatientData()));

  const [transcript, setTranscript] = useState(() => readCachedValue('ps_transcript', []));

  const [vitalsData, setVitalsData] = useState(() => ({ ...getCleanVitalsData(), ...readCachedValue('ps_vitals', {}) }));

  const [triageResult, setTriageResult] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loadingEncounter, setLoadingEncounter] = useState(false);
  const [activeTriageTab, setActiveTriageTab] = useState("manual");
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [ehrSummary, setEhrSummary] = useState(() => readCachedValue('ps_ehr_summary', null));
  const [selectedMrn, setSelectedMrn] = useState(() => readCachedValue('ps_selected_mrn', null));
  const [loadingEHR, setLoadingEHR] = useState(false);
  const [showAIPreview, setShowAIPreview] = useState(false);
  const [aiPreviewApproved, setAiPreviewApproved] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);
  const searchRequestRef = useRef(null);
  const ehrRequestRef = useRef(null);

  const resetPatientForm = () => {
    setPatientData(getCleanPatientData());
    setTranscript([]);
    setVitalsData({
      temp: '--',
      bp: '--',
      hr: '--',
      o2: '--',
      resp: '--',
      bp_systolic: null,
      bp_diastolic: null,
      bp_left_systolic: null,
      bp_left_diastolic: null,
      bp_right_systolic: null,
      bp_right_diastolic: null,
      avpu: null,
      mobility: 'Walks',
      traumaPresent: false
    });
    setEhrSummary(null);
    setSelectedMrn(null);
    localStorage.removeItem('ps_patient');
    localStorage.removeItem('ps_transcript');
    localStorage.removeItem('ps_vitals');
    localStorage.removeItem('ps_ehr_summary');
    localStorage.removeItem('ps_selected_mrn');
  };

  const clearVitals = () => {
    if (!window.confirm("Are you sure you want to clear patient vitals?")) return;
    setVitalsData(getCleanVitalsData());
  };

  // ---------- Persistence ----------
  useEffect(() => {
    localStorage.setItem('ps_patient', JSON.stringify(patientData));
  }, [patientData]);

  useEffect(() => {
    localStorage.setItem('ps_transcript', JSON.stringify(transcript));
  }, [transcript]);

  useEffect(() => {
    localStorage.setItem('ps_vitals', JSON.stringify(vitalsData));
  }, [vitalsData]);

  useEffect(() => {
    if (ehrSummary) localStorage.setItem('ps_ehr_summary', JSON.stringify(ehrSummary));
    else localStorage.removeItem('ps_ehr_summary');
  }, [ehrSummary]);

  useEffect(() => {
    if (selectedMrn) localStorage.setItem('ps_selected_mrn', JSON.stringify(selectedMrn));
    else localStorage.removeItem('ps_selected_mrn');
  }, [selectedMrn]);

  const loadPatientEHR = async (mrn) => {
    if (!mrn) return;
    ehrRequestRef.current?.abort();
    const controller = new AbortController();
    ehrRequestRef.current = controller;
    setLoadingEHR(true);
    setEhrSummary(null);
    setSearchError(null);

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${BACKEND_URL}/api/patients/ehr/${encodeURIComponent(mrn)}`, {
        headers,
        cache: 'no-store',
        signal: controller.signal
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
        const folderCriticalHistory = extractCriticalHistoryFromEncounters(encounters);
        setPatientData({
          ...getCleanPatientData(),
          id_number: patient.id_number || '',
          name: patient.name || '',
          surname: patient.surname || '',
          gender: patient.gender || '',
          age: patient.age ?? null,
          ageYears: patient.ageYears ?? null,
          ageMonths: patient.ageMonths ?? null,
          height_cm: patient.height_cm ?? null,
          criticalHistoryStatus: folderCriticalHistory.criticalHistoryStatus,
          criticalHistory: folderCriticalHistory.criticalHistory
        });
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      setSearchError(err.message || 'Failed to load patient EHR');
    } finally {
      if (ehrRequestRef.current === controller) {
        ehrRequestRef.current = null;
        setLoadingEHR(false);
      }
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
    searchRequestRef.current?.abort();
    const controller = new AbortController();
    searchRequestRef.current = controller;

    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${BACKEND_URL}/api/patients/search?search=${encodeURIComponent(searchQuery.trim())}`, {
        headers,
        cache: 'no-store',
        signal: controller.signal
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
      if (err.name === 'AbortError') return;
      setSearchError(err.message || 'Patient search failed');
    } finally {
      if (searchRequestRef.current === controller) {
        searchRequestRef.current = null;
        setSearchLoading(false);
      }
    }
  };

  const handlePatientSelect = async (encounter) => {
    const patient = encounter.patient || {};
    const identifier = patient.id_number || encounter._id || encounter.id;
    if (!identifier) {
      setSearchError('Selected record is missing a patient identifier');
      return;
    }

    setSearchError(null);
    setSelectedMrn(identifier);
    setSearchQuery(patient.id_number || identifier);

    const criticalHistoryFields = extractCriticalHistoryFromPatient(patient);
    setPatientData({
      ...getCleanPatientData(),
      id_number: patient.id_number || '',
      name: patient.name || '',
      surname: patient.surname || '',
      gender: patient.gender || '',
      age: patient.age ?? null,
      ageYears: patient.ageYears ?? null,
      ageMonths: patient.ageMonths ?? null,
      height_cm: patient.height_cm ?? null,
      symptoms: '',
      history: '',
      duration: '',
      painLevel: '',
      criticalHistoryStatus: criticalHistoryFields.criticalHistoryStatus,
      criticalHistory: criticalHistoryFields.criticalHistory
    });

    await loadPatientEHR(identifier);
  };

  const clearPatientSelection = () => {
    ehrRequestRef.current?.abort();
    searchRequestRef.current?.abort();
    ehrRequestRef.current = null;
    searchRequestRef.current = null;
    setLoadingEHR(false);
    setSearchLoading(false);
    setSelectedMrn(null);
    setSearchResults([]);
    setSearchError(null);
    resetPatientForm();
  };

  const cancelSearch = () => {
    searchRequestRef.current?.abort();
    searchRequestRef.current = null;
    setSearchLoading(false);
  };

  const cancelEHRLoad = () => {
    clearPatientSelection();
  };

  const startNewIntake = () => {
    clearPatientSelection();
    setSearchQuery('');
    resetPatientForm();
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
          avpu: encounter.vitals?.avpu ?? encounter.triage?.vitals_parsed?.avpu ?? null,
          mobility: encounter.vitals?.mobility ?? "Walks",
          traumaPresent: Boolean(encounter.vitals?.traumaPresent)
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
      hospitalId,
      patientId,
      triageNurseId,
      
      // ===== PATIENT & VITALS DATA =====
      patient: {
        id_number: patientData?.id_number ?? "",
        name: patientData?.name ?? "",
        surname: patientData?.surname ?? "",
        gender: patientData?.gender ?? "",
        age: patientData?.ageYears ?? patientData?.age ?? null,
        ageMonths: patientData?.ageMonths ?? null,
        height_cm: patientData?.height_cm ?? null,
        symptoms: patientData?.symptoms ?? "",
        history: patientData?.history ?? "",
        duration: patientData?.duration ?? "",
        painLevel: patientData?.painLevel ?? "",
        criticalHistoryStatus: patientData?.criticalHistoryStatus || "",
        criticalHistory: patientData?.criticalHistory || {}
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
      computeMode: ragEnabled ? 'HYBRID' : 'LLM_ONLY',
      ragEnabled
    };
  };

  const buildAIPreviewPayload = () => {
    const payload = buildPayload();
    const historicalEncounters = ehrSummary?.encounters || [];
    const renderedEncounters = historicalEncounters.slice(0, 6).map((enc, idx) => {
      const patient = enc?.patient || {};
      const date = enc?.date || enc?.submittedAt || enc?.createdAt || null;
      const shortDate = formatClinicalTimestamp(date) || `Encounter ${idx + 1}`;
      const symptomSummary = patient?.symptoms || enc?.symptoms || 'No symptom summary';
      const priorHistory = patient?.history || enc?.history || 'No prior history';
      return {
        encounterIndex: idx + 1,
        timestamp: shortDate,
        symptoms: symptomSummary,
        priorHistory,
        status: enc?.status || 'Unknown status'
      };
    });

    const historyText = renderedEncounters.length
      ? renderedEncounters.map((item) => (
          `${item.timestamp} • ${item.status} • ${item.symptoms} • ${item.priorHistory}`
        )).join("\n\n")
      : "No prior encounter history loaded for this patient.";

    return {
      ...payload,
      aiContextPreview: {
        source: selectedMrn ? `Loaded EHR record: ${selectedMrn}` : 'Current intake form',
        historicalEncounterCount: historicalEncounters.length,
        loadedRecordCount: historicalEncounters.length,
        criticalHistory: patientData?.criticalHistory || {},
        historyContextPreview: historyText,
        renderedEncounters,
        transcriptPreview: Array.isArray(payload.transcript)
          ? payload.transcript.slice(0, 5)
          : [payload.transcript],
        computeMode: payload.computeMode
      }
    };
  };

  const openAIPreview = () => {
    setAiPreviewApproved(false);
    setShowAIPreview(true);
  };

  const confirmAIPreview = async () => {
    setAiPreviewApproved(true);
    setShowAIPreview(false);
    await handleSavePatient();
  };

  // ---------- Submit AI triage (with clinical handover) ----------
  const handleSavePatient = async () => {
    const vCheck = validateVitals(vitalsData);
    if (!vCheck.ok) return alert(vCheck.message);
    if (!['reported', 'none_established'].includes(patientData?.criticalHistoryStatus)) {
      return alert('Complete Critical History: record the known facts or select "None established".');
    }

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
              <h1>Triage with Patient History </h1>
              <p className = "sho">Load vitals, symptoms and patient history</p>
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
              cancelSearch={cancelSearch}
              cancelEHRLoad={cancelEHRLoad}
            />
            <RightColumn
              vitalsData={vitalsData}
              triage={triageResult}
              activeTab={activeTriageTab}
              onOpenForm={() => setShowForm(true)}
              onClearVitals={clearVitals}
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
              onAiTriage={openAIPreview}
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

      {showAIPreview && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(15, 23, 42, 0.55)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 2000,
          padding: 20
        }} onClick={() => setShowAIPreview(false)}>
          <div
            style={{
              width: 'min(1000px, 92vw)',
              maxHeight: '88vh',
              overflow: 'auto',
              background: '#fff',
              borderRadius: 18,
              boxShadow: '0 24px 80px rgba(15, 23, 42, 0.3)',
              padding: 24,
              color: '#111827'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, gap: 16 }}>
              <div>
                <div style={{ fontSize: 12, letterSpacing: '0.14em', textTransform: 'uppercase', color: '#2563eb', fontWeight: 700 }}>
                  AI triage review
                </div>
                <h3 style={{ margin: '8px 0 0', fontSize: 28 }}>Clinical context before auto-triage</h3>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {selectedMrn && (
                  <button
                    type="button"
                    onClick={() => navigate(`/ehr?patient=${encodeURIComponent(selectedMrn)}`)}
                    style={{ border: '1px solid #bfdbfe', background: '#eff6ff', color: '#1d4ed8', borderRadius: 8, padding: '10px 14px', cursor: 'pointer', fontWeight: 600 }}
                  >
                    Open EHR record
                  </button>
                )}
                <button type="button" onClick={() => setShowAIPreview(false)} style={{ border: '1px solid #d1d5db', background: '#fff', borderRadius: 8, padding: '10px 14px', cursor: 'pointer', fontWeight: 600 }}>
                  Close
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 18 }}>
              <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 12, padding: 14 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#1d4ed8', fontWeight: 700 }}>Patient</div>
                <div style={{ fontWeight: 700, marginTop: 8 }}>{patientData?.name || 'Unknown'} {patientData?.surname || ''}</div>
                <div style={{ color: '#374151', marginTop: 4 }}>{patientData?.id_number || 'No MRN'}</div>
              </div>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 12, padding: 14 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#475569', fontWeight: 700 }}>Historical context</div>
                <div style={{ fontWeight: 700, marginTop: 8 }}>{ehrSummary?.totalEncounters ?? 0} prior encounter(s)</div>
                <div style={{ color: '#374151', marginTop: 4 }}>{selectedMrn ? `Loaded from ${selectedMrn}` : 'New intake'}</div>
              </div>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 12, padding: 14 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#475569', fontWeight: 700 }}>Compute mode</div>
                <div style={{ fontWeight: 700, marginTop: 8 }}>{buildAIPreviewPayload().computeMode}</div>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#374151', marginTop: 8, cursor: 'pointer' }}>
                  <input type="checkbox" checked={ragEnabled} onChange={(event) => setRagEnabled(event.target.checked)} />
                  Use PubMed reference cases
                </label>
                <div style={{ color: '#64748b', marginTop: 4, fontSize: 12 }}>
                  Patient EHR history remains available in both modes.
                </div>
              </div>
            </div>

            <div style={{ marginBottom: 18, padding: 16, background: '#f8fafc', borderRadius: 12, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 700, marginBottom: 12 }}>Encounter records included in this AI input</div>
              {buildAIPreviewPayload().aiContextPreview.renderedEncounters.length > 0 ? (
                <div style={{ display: 'grid', gap: 10 }}>
                  {buildAIPreviewPayload().aiContextPreview.renderedEncounters.map((item) => (
                    <div key={`${item.timestamp}-${item.encounterIndex}`} style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 6, flexWrap: 'wrap' }}>
                        <strong style={{ color: '#111827' }}>Encounter {item.encounterIndex}</strong>
                        <span style={{ color: '#475569', fontSize: 12 }}>{item.timestamp}</span>
                      </div>
                      <div style={{ color: '#374151', fontSize: 12, lineHeight: 1.6 }}>
                        <div><strong>Status:</strong> {item.status}</div>
                        <div><strong>Symptoms:</strong> {item.symptoms}</div>
                        <div><strong>History:</strong> {item.priorHistory}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: '#475569', fontSize: 12 }}>No prior encounter history loaded for this patient.</div>
              )}
            </div>

            <div style={{ marginBottom: 18, padding: 16, background: '#0f172a', color: '#e2e8f0', borderRadius: 12 }}>
              <div style={{ fontWeight: 700, marginBottom: 10 }}>Clinical summary for AI triage</div>
              <div style={{ fontSize: 12, lineHeight: 1.7, color: '#dbeafe' }}>
                <div><strong>Current concern:</strong> {patientData?.symptoms || 'Not recorded'}</div>
                <div><strong>Vitals:</strong> Temp {normalizeTemp(vitalsData?.temp) ?? '--'}°C • BP {vitalsData?.bp || '--'} • HR {vitalsData?.hr ?? '--'} • O₂ {vitalsData?.o2 ?? '--'}%</div>
                <div><strong>Critical history:</strong> {patientData?.criticalHistoryStatus === 'reported' ? 'Known risk factors documented' : 'None established'} </div>
                <div><strong>Relevant details:</strong> {patientData?.criticalHistory ? Object.values(patientData.criticalHistory).flat().slice(0, 5).join(', ') || 'No additional risk factors recorded' : 'No additional risk factors recorded'}</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, borderTop: '1px solid #e5e7eb', paddingTop: 16, flexWrap: 'wrap' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontWeight: 600, color: '#374151', cursor: 'pointer', maxWidth: '65%' }}>
                <input type="checkbox" checked={aiPreviewApproved} onChange={(e) => setAiPreviewApproved(e.target.checked)} />
                I have reviewed the patient history and approve this context for AI triage.
              </label>

              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                {selectedMrn && (
                  <button
                    type="button"
                    onClick={() => navigate(`/ehr?patient=${encodeURIComponent(selectedMrn)}`)}
                    style={{ background: '#fff', color: '#1f2937', border: '1px solid #cbd5e1', borderRadius: 10, padding: '12px 18px', fontWeight: 700, cursor: 'pointer' }}
                  >
                    View EHR
                  </button>
                )}
                <button
                  type="button"
                  disabled={!aiPreviewApproved}
                  style={{
                    background: aiPreviewApproved ? '#2563eb' : '#cbd5e1',
                    color: '#fff',
                    border: 'none',
                    borderRadius: 10,
                    padding: '12px 22px',
                    fontWeight: 700,
                    cursor: aiPreviewApproved ? 'pointer' : 'not-allowed'
                  }}
                  onClick={confirmAIPreview}
                >
                  Approve & Submit Auto-Triage
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </SidebarLayout>
  );
}