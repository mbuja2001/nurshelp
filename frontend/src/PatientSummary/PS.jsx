// src/PatientSummary/PS.jsx
import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import Header from "./Components/Header";
import Footer from "./Components/footer";
import LeftColumn from "./Components/LeftColumn";
import RightColumn from "./Components/RightColumn";
import Vitals from "./Components/Vitals";

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

  // ---------- Build strict payload for backend ----------
  const buildPayload = () => {
    const safeTranscript = Array.isArray(transcript) ? transcript : [{ id: 1, type: "note", text: String(transcript || "") }];
    const bpParsed = parseBP(String(vitalsData?.bp ?? ""));

    const bpFinal = (bpParsed.systolic && bpParsed.diastolic) ? `${bpParsed.systolic}/${bpParsed.diastolic}` : null;

    return {
      patient: {
        id_number: patientData?.id_number ?? "",
        name: patientData?.name ?? "",
        surname: patientData?.surname ?? "",
        age: patientData?.age ?? null,
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
      arrival_time: new Date().toISOString()
    };
  };

  // ---------- Submit triage ----------
  const handleSavePatient = async () => {
    const vCheck = validateVitals(vitalsData);
    if (!vCheck.ok) return alert(vCheck.message);

    setSaving(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/triage/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload())
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || `Triage failed (${res.status})`);
      }
      const data = await res.json();
      const enc = data.encounter;
      if (!enc) throw new Error("No encounter returned");

      // navigate to patient report
      navigate(`/summary/${enc._id || enc.id}`, { state: { encounter: enc } });
    } catch (err) {
      console.error("Save patient error:", err);
      alert("Save failed: " + (err.message || "unknown error"));
    } finally {
      setSaving(false);
    }
  };

  // ---------- Render ----------
  return (
    <div className="ps-main-page-wrapper">
      <div className="app-wrapper">
        <Header
          onBackClick={() => navigate("/")}
          onLogout={() => { localStorage.removeItem("token"); navigate("/"); }}
        />

        <main className="main-container">
          <div className="dashboard-grid" style={{ display: "flex", gap: 20, padding: 20 }}>
            <LeftColumn data={patientData} setData={setPatientData} />
            <RightColumn
              vitalsData={vitalsData}
              triage={triageResult}
              onOpenForm={() => setShowForm(true)}
              onSavePatient={handleSavePatient}
              saving={saving}
            />
          </div>
        </main>

        <Footer />
      </div>

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
  );
}