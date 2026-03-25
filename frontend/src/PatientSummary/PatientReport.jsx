import React, { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

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

  // --- RAG cases evidence trail ---
  const [showEvidenceTrail, setShowEvidenceTrail] = useState(false);
  const [expandedCase, setExpandedCase] = useState(null);

  const getEncounterId = (enc) => enc?._id ?? enc?.id ?? null;

  // fetch encounter if not provided
  useEffect(() => {
    if (encounter) return;
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
      } catch (err) {
        console.error("Fetch encounter error:", err);
        alert("Failed to load report: " + (err.message || ""));
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchEncounter();
    return () => { mounted = false; };
  }, [id, encounter]);

  // ---------- NEW: fetch top 5 physician recommendations after encounter loads ----------
  useEffect(() => {
    if (!encounter) return;
    let mounted = true;

    const fetchRecommendations = async () => {
      setLoadingDocs(true);
      try {
        const encId = getEncounterId(encounter);
        if (!encId) return;

        const token = localStorage.getItem("token");
        const headers = {
          ...( token ? { Authorization: `Bearer ${token}` } : {}),
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache"
        };

        // Expected endpoint: returns { physicians: [ { id, name, specialty, score, reason }, ... ] }
        const cacheBreaker = `&t=${Date.now()}`;
        const res = await fetch(`${BACKEND_URL}/api/physicians/recommendations?encounterId=${encId}${cacheBreaker}`, { 
          headers,
          cache: "no-store"
        });
        if (!res.ok) {
          // not fatal: fallback
          console.warn("Failed to fetch physician recommendations, status:", res.status);
          setRecommendedDocs([]);
          return;
        }

        const data = await res.json().catch(() => null);
        const docs = data?.physicians ?? data?.results ?? [];

        // Normalise doc objects (id vs physician_id)
        const norm = (docs || []).slice(0, 5).map((d, i) => ({
          id: d.id ?? d.physician_id ?? String(d.physicianId ?? (`doc-${i}`)),
          name: d.name ?? d.physician_name ?? d.displayName ?? "Unknown",
          specialty: d.specialty ?? d.department ?? "General",
          score: typeof d.score === "number" ? d.score : (d.match ?? d.weight ?? 0),
          reason: d.reason ?? d.note ?? d.rationale ?? ""
        }));

        if (!mounted) return;
        setRecommendedDocs(norm);

        // default select top ranked (if not already assigned)
        if (norm.length > 0) {
          // if encounter already has assigned clinician, prefer that as selected
          const assigned = encounter?.triage?.assigned_physician ?? encounter?.triage?.assignedPhysician ?? null;
          if (assigned && norm.some(d => String(d.id) === String(assigned.id ?? assigned.physician_id ?? assigned.physicianId))) {
            const match = norm.find(d => String(d.id) === String(assigned.id ?? assigned.physician_id ?? assigned.physicianId));
            setSelectedDoc(match);
          } else {
            setSelectedDoc(norm[0]);
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
    return () => { mounted = false; };
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
      case 3: return { label: "3 — YELLOW", color: "#f1c40f", className: "sev-mod" };
      case 4: return { label: "4 — GREEN", color: "#10b981", className: "sev-low" };
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
    if (!Array.isArray(casesList) || casesList.length === 0) return null;

    const referencedCases = casesList.filter(c => c.medgemma_referenced === true);
    const totalCases = casesList.length;

    return (
      <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: "#fffbeb", borderLeft: "4px solid #eab308" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <div style={{ fontWeight: 700, color: "#0f172a", fontSize: 13 }}>
            Evidence Trail
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
            {showEvidenceTrail ? "Hide Evidence" : "Show Evidence"}
          </button>
        </div>

        {showEvidenceTrail && (
          <div style={{ marginTop: 10, borderTop: "1px solid #fcd34d", paddingTop: 12 }}>
            {renderRAGCases(casesList)}
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

  // Assign immediately (human chooses one from top5)
  const handleAssignPhysician = async (doc) => {
    if (!doc) return;
    if (!window.confirm(`Assign ${doc.name} to this patient now?`)) return;

    setAssigning(true);
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        alert("You must be logged in to assign a physician.");
        navigate("/");
        return;
      }
      const encId = getEncounterId(encounter);
      if (!encId) throw new Error("Encounter ID missing; cannot assign.");

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encId}/assign`, {
        method: "PATCH", // or POST depending on your API
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ physicianId: doc.id }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || `Assign failed (status ${res.status})`);
      }

      const payload = await res.json().catch(() => null);
      const updated = payload?.encounter ?? payload;
      if (!updated) throw new Error("Server returned no updated encounter");

      setEncounter(updated);
      setSelectedDoc({ id: doc.id, name: doc.name, specialty: doc.specialty }); // keep UI in sync
      alert(`${doc.name} assigned to patient`);
    } catch (err) {
      console.error("Assign error:", err);
      alert("Failed to assign physician: " + (err.message || ""));
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
  const triage = encounter.triage ?? {};

  // DEBUG: inspect what the backend returned so we can trace missing BP values
  if (process.env.NODE_ENV !== 'production') {
    // eslint-disable-next-line no-console
    console.debug('[PatientReport] encounter.vitals:', encounter.vitals, 'triage.vitals_parsed:', triage.vitals_parsed, 'merged vitals:', vitals);
  }

  // Compute BP display using merged vitals (which now includes vitals_parsed fallback)
  const displayBP = computeDisplayBP(vitals, triage);
  const priorityCode = triage.priority_code ?? triage.priority ?? null;
  const { label: priorityLabel, color: priorityColor } = priorityToVisual(priorityCode);
  const targetMins = triage.target_time_mins ?? triage.target_mins ?? null;
  const waitingSeverity = triage.waitingSeverity ?? triage.waiting_severity ?? encounter.waitingSeverity ?? null;
  // Use discriminators_readable (human-friendly text) instead of discriminators_found (concept keys)
  const discriminators = triage?.SATS_reasoning?.discriminators_readable ?? triage?.SATS_reasoning?.discriminators_found ?? triage?.SATS_reasoning?.discriminators ?? triage?.discriminators ?? [];
  const tews = triage?.SATS_reasoning?.tews_components ?? triage?.tews_components ?? {};
  const tewsTotal = triage?.SATS_reasoning?.tews_total ?? triage?.tews_total ?? null;
  const assignedPhys = triage?.assigned_physician ?? triage?.assignedPhysician ?? null;
  const assignedSpec = triage?.assigned_specialty ?? triage?.assignedSpecialty ?? null;
  const ward = triage?.ward ?? null;

  const aiSummaryRaw = triage?.ai_summary ?? triage?.summary ?? "";
  const aiSummaryParsed = tryParsePossibleObjectString(aiSummaryRaw);

  return (
    <div className="patient-report-page" style={{ padding: 20 }}>
      <h2 style={{ marginBottom: 12 }}>Patient Report Preview</h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 420px", gap: 20 }}>
        <div>
          {/* -- main left column (patient, transcript, triage) -- */}
          <section className="block">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
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
                  <div style={{ fontWeight: 700 }}>{assignedSpec ?? "No specialty"}</div>
                  <div style={{ fontSize: 11 }}>{assignedPhys ? (assignedPhys.name ?? `ID:${assignedPhys.id}`) : "No clinician assigned"}</div>
                </div>
              </div>
            </div>

            <div className="content-area" style={{ marginTop: 12, textAlign: "left" }}>
              <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>
                {patient?.id_number ? `ID: ${patient.id_number}` : "ID: ---"}
              </div>
              <div style={{ fontSize: 20, fontWeight: 800 }}>
                {patient?.name ?? "Unnamed"} {patient?.surname ? patient.surname : ""}
              </div>
              <div style={{ marginTop: 6, fontSize: 13, color: "#475569" }}>
                {patient?.age != null && <span><strong>Age:</strong> {patient.age}y</span>}
                {patient?.age != null && patient?.height_cm && <span> • </span>}
                {patient?.height_cm != null && <span><strong>Height:</strong> {patient.height_cm}cm</span>}
              </div>
              <div style={{ marginTop: 8, color: "#475569" }}>
                <strong>Presenting complaint:</strong> {triage?.patient_symptoms ?? patient?.symptoms ?? <em>Not provided</em>}
              </div>
              <div style={{ marginTop: 6, color: "#475569" }}>
                {patient?.duration ?? ""} {patient?.painLevel ? ` • ${patient?.painLevel}` : ""}
              </div>
              <div style={{ marginTop: 8, color: "#64748b" }}>
                <em>{patient?.history ?? ""}</em>
              </div>
            </div>
          </section>

          <section className="block" style={{ marginTop: 12 }}>
            <h4 className="block-label">Transcript</h4>
            <div style={{ marginTop: 8 }}>
              {(transcript || []).length === 0 && <div style={{ color: "#666" }}>No transcript recorded.</div>}
              {(transcript || []).map((t, i) => (
                <div key={t.id ?? i} style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 12, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.6 }}>{(t.type || "").toUpperCase()}</div>
                  <div style={{ marginTop: 4, fontSize: 15 }}>{t.text}</div>
                </div>
              ))}
            </div>
          </section>

          <section className="block" style={{ marginTop: 12 }}>
            <h4 className="block-label">AI Triage</h4>
            <div style={{ marginTop: 8 }}>
              <div style={{ display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between" }}>
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
                <div style={{ fontSize: 13, color: "#334155" }}><strong>AI summary</strong></div>
                <div style={{ marginTop: 8, background: "#f8fafc", padding: 12, borderRadius: 6, color: "#0f172a" }}>
                  {aiSummaryParsed ? (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {Object.entries(aiSummaryParsed).map(([k, v]) => (
                        <div key={k} style={{ fontSize: 13 }}>
                          <div style={{ color: "#475569", fontSize: 12, textTransform: "capitalize" }}>{k.replace(/_/g, " ")}</div>
                          <div style={{ marginTop: 4, fontWeight: 700 }}>{String(v)}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ whiteSpace: "pre-wrap", fontSize: 14, color: "#0f172a" }}>
                      {String(aiSummaryRaw).replace(/[{}`]/g, "")}
                    </div>
                  )}
                </div>
              </div>

              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 13, color: "#334155", marginBottom: 8 }}><strong>SATS reasoning</strong></div>

                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
                  {(discriminators && discriminators.length > 0) ? discriminators.map((d, i) => (
                    <span key={i} className="chip">{d}</span>
                  )) : <span style={{ color: "#64748b" }}>No discriminators detected</span>}
                </div>

                <div style={{ display: "flex", gap: 18 }}>
                  <div style={{ minWidth: 160 }}>
                    <div style={{ fontSize: 12, color: "#475569" }}><strong>TEWS total</strong></div>
                    <div style={{ marginTop: 6, fontWeight: 800 }}>{tewsTotal != null ? tewsTotal : "N/A"}</div>
                    <div style={{ marginTop: 8 }}>{renderTews(tews)}</div>
                  </div>

                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, color: "#475569" }}><strong>Clinical recommendations</strong></div>
                    <div style={{ marginTop: 6 }}>
                      {triage?.clinical_structure ? renderClinicalStructure(triage.clinical_structure) : <div style={{ color: "#64748b" }}>No structured recommendations</div>}
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 12, fontSize: 13 }}>
                <div><strong>Assigned specialty:</strong> {assignedSpec ?? "N/A"}</div>
                <div style={{ marginTop: 6 }}><strong>Assigned clinician:</strong> {assignedPhys ? (assignedPhys.name ?? `ID:${assignedPhys.id}`) : "N/A"}</div>
                <div style={{ marginTop: 6 }}><strong>Ward:</strong> {ward ?? "N/A"}</div>
                <div style={{ marginTop: 6 }}><strong>Waiting severity:</strong> {waitingSeverity ?? "N/A"}</div>
              </div>

              {renderEvidenceTrail(triage?.similar_cases_rag)}
            </div>
          </section>
        </div>

        <aside>
          {/* -- right column: vitals, nurse notes, recommendations & selection UI -- */}
          <section className="block">
            <h4 className="block-label">Patient Vitals & Queue</h4>

            <div className="vitals-display-list" style={{ marginTop: 8 }}>
              <div className="vital-item"><span>Temp:</span> <strong>{fmt(vitals?.temp, "°C")}</strong></div>
              <div className="vital-item"><span>BP:</span> <strong>{displayBP}</strong></div>
              <div className="vital-item"><span>BP Sys:</span> <strong>{fmt(vitals?.bp_systolic)}</strong></div>
              <div className="vital-item"><span>BP Dia:</span> <strong>{fmt(vitals?.bp_diastolic)}</strong></div>
              
              {/* Left Arm BP */}
              <div className="vital-item" style={{ borderTop: '1px solid #e5e7eb', paddingTop: '6px', marginTop: '6px' }}>
                <span style={{ fontWeight: '600', color: '#4b5563' }}>BP Left</span>
              </div>
              <div className="vital-item"><span>Sys:</span> <strong>{fmt(vitals?.bp_left_systolic)}</strong></div>
              <div className="vital-item"><span>Dia:</span> <strong>{fmt(vitals?.bp_left_diastolic)}</strong></div>
              
              {/* Right Arm BP */}
              <div className="vital-item">
                <span style={{ fontWeight: '600', color: '#4b5563' }}>BP Right</span>
              </div>
              <div className="vital-item"><span>Sys:</span> <strong>{fmt(vitals?.bp_right_systolic)}</strong></div>
              <div className="vital-item"><span>Dia:</span> <strong>{fmt(vitals?.bp_right_diastolic)}</strong></div>
              
              <div className="vital-item"><span>HR:</span> <strong>{fmt(vitals?.hr)}</strong></div>
              <div className="vital-item"><span>O₂:</span> <strong>{fmt(vitals?.o2, "%")}</strong></div>
              <div className="vital-item"><span>Resp:</span> <strong>{fmt(vitals?.resp)}</strong></div>
              <div className="vital-item"><span>AVPU:</span> <strong>{fmt(vitals?.avpu)}</strong></div>
            </div>

            <div style={{ marginTop: 12, color: "#334155" }}>
              <div><strong>Waiting:</strong> {encounter?.isWaiting ? "Yes" : "No"}</div>
              <div style={{ marginTop: 6 }}><strong>Waiting time:</strong> {timeAgo(encounter?.createdAt)}</div>
            </div>
          </section>

          <section className="block" style={{ marginTop: 12 }}>
            <h4 className="block-label">Nurse Notes</h4>
            <textarea
              className="edit-textarea"
              value={nurseNotes}
              onChange={(e) => setNurseNotes(e.target.value)}
              style={{ minHeight: 120 }}
            />
          </section>

          {/* -- recommended physicians panel -- */}
          <section className="block" style={{ marginTop: 12 }}>
            <h4 className="block-label">Recommended Physicians (Top 5)</h4>

            {loadingDocs && <div>Loading recommendations...</div>}

            {!loadingDocs && recommendedDocs.length === 0 && (
              <div style={{ color: "#64748b" }}>No recommendations available</div>
            )}

            <div style={{ marginTop: 8 }}>
              {recommendedDocs.map((doc, idx) => {
                const isSelected = String(selectedDoc?.id) === String(doc.id);
                return (
                  <div
                    key={doc.id}
                    onClick={() => setSelectedDoc(doc)}
                    style={{
                      padding: 10,
                      borderRadius: 8,
                      marginBottom: 8,
                      cursor: "pointer",
                      border: isSelected ? "2px solid #2563eb" : "1px solid #e2e8f0",
                      background: isSelected ? "#eff6ff" : "white",
                      transition: "all 0.12s"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 800 }}>
                          {idx + 1}. {doc.name}
                        </div>
                        <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
                          {doc.specialty}
                        </div>
                      </div>

                      <div style={{ textAlign: "right", minWidth: 72 }}>
                        <div style={{ fontWeight: 800 }}>
                          {typeof doc.score === "number" ? `${Math.round(doc.score * 100)}%` : "--"}
                        </div>
                        <div style={{ fontSize: 11, color: "#64748b" }}>
                          {doc.reason || doc.source || 'match'}
                        </div>
                      </div>
                    </div>

                    {doc.reason && (
                      <div style={{ marginTop: 8, fontSize: 13, color: "#334155" }}>
                        {doc.reason}
                      </div>
                    )}

                    <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
                      <button
                        onClick={(e) => { e.stopPropagation(); setSelectedDoc(doc); }}
                        className="edit-action-btn"
                        style={{ padding: "6px 8px" }}
                      >
                        Select
                      </button>

                      <button
                        onClick={async (e) => { e.stopPropagation(); await handleAssignPhysician(doc); }}
                        className="save-file-btn"
                        style={{ padding: "6px 8px" }}
                        disabled={assigning}
                        title="Assign this physician now"
                      >
                        {assigning ? "Assigning..." : "Assign"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {selectedDoc && (
              <div style={{ marginTop: 10, padding: 10, background: "#f1f5f9", borderRadius: 8, fontSize: 13 }}>
                <strong>Selected:</strong> {selectedDoc.name}{" "}
                {selectedDoc.specialty ? <em style={{ color: "#475569" }}>• {selectedDoc.specialty}</em> : null}
              </div>
            )}
          </section>

          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <button className="edit-action-btn" onClick={() => navigate(-1)}>Edit</button>

            <button
              className="save-file-btn"
              onClick={handleConfirm}
              disabled={saving}
              title="Confirm & Save report (selected physician will be included)"
            >
              {saving ? "Confirming..." : "Confirm & Save Report"}
            </button>

            {encounter?.isWaiting && (
              <button
                className="save-file-btn"
                onClick={handleAttendPatient}
                disabled={saving}
                title="Mark as attended"
              >
                Mark as Attended
              </button>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}