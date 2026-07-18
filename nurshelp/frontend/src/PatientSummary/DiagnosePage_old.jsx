import React, { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const getSeverityConfig = (severity) => {
  switch (Number(severity)) {
    case 1:
      return { label: "Critical", color: "#dc2626", bg: "#fee2e2" };
    case 2:
      return { label: "Very Urgent", color: "#f97316", bg: "#ffedd5" };
    case 3:
      return { label: "Urgent", color: "#eab308", bg: "#fef9c3" };
    case 4:
      return { label: "Standard", color: "#0f766e", bg: "#ccfbf1" };
    default:
      return { label: "Unknown", color: "#475569", bg: "#f1f5f9" };
  }
};

const DiagnosePage = () => {
  const { encounterId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const initialEncounter = location.state?.encounter ?? null;
  const [encounter, setEncounter] = useState(initialEncounter);
  const [loading, setLoading] = useState(!initialEncounter);
  const [error, setError] = useState(null);
  const [diagnosisText, setDiagnosisText] = useState(initialEncounter?.diagnosis || "");
  const [applyingDisposition, setApplyingDisposition] = useState(false);
  const [message, setMessage] = useState(null);

  const fetchEncounter = async () => {
    if (!encounterId) return;
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.message || `Unable to load encounter ${encounterId}`);
      }

      const data = await res.json();
      const loadedEncounter = data.encounter || data.case || data;
      setEncounter(loadedEncounter);
      setDiagnosisText(loadedEncounter?.diagnosis || "");
    } catch (err) {
      console.error("[DIAGNOSE PAGE] Fetch encounter failed:", err);
      setError(err.message || "Unable to load encounter data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!encounter) {
      fetchEncounter();
    }
  }, [encounterId]);

  const getEncounterId = () => encounterId || encounter?._id || encounter?.id;

  const persistDiagnosis = async (overrideDiagnosis = diagnosisText) => {
    const id = getEncounterId();
    if (!id) {
      throw new Error("Missing encounter identifier.");
    }

    const token = localStorage.getItem("token");
    const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/diagnosis`, {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        diagnosis: overrideDiagnosis?.trim() || "Diagnosis recorded by physician",
        notes: "Diagnosis captured during physician disposition workflow"
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.message || "Failed to record diagnosis");
    }

    const data = await res.json();
    const updatedEncounter = data.encounter || data;
    setEncounter(updatedEncounter);
    setDiagnosisText(updatedEncounter?.diagnosis || overrideDiagnosis || "");
    return updatedEncounter;
  };

  const handleDisposition = async (disposition) => {
    const id = getEncounterId();
    if (!id) {
      setError("Missing encounter identifier.");
      return;
    }

    if (disposition === "DISCHARGED") {
      const confirmed = window.confirm("Discharge this patient from inpatient care?");
      if (!confirmed) return;
    }

    setApplyingDisposition(true);
    setError(null);
    setMessage(null);

    try {
      const savedEncounter = diagnosisText?.trim()
        ? await persistDiagnosis(diagnosisText)
        : encounter;

      if (disposition === "HOSPITALIZED") {
        navigate(`/assign-ward-bed/${id}`, { state: { encounter: savedEncounter || encounter } });
        return;
      }

      const token = localStorage.getItem("token");
      const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/disposition`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          disposition,
          disposition_notes: disposition === "DISCHARGED"
            ? "Discharged from diagnosis workflow"
            : "Admitted from diagnosis workflow"
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.message || `Failed to ${disposition.toLowerCase()} patient`);
      }

      const data = await res.json();
      const updatedEncounter = data.encounter || data;
      setEncounter(updatedEncounter);
      setMessage(`Patient ${disposition === "HOSPITALIZED" ? "admitted" : "discharged"} successfully. Diagnosis has been captured in the clinical record.`);
    } catch (err) {
      console.error(`[DIAGNOSE PAGE] Disposition ${disposition} failed:`, err);
      setError(err.message || "Unable to update disposition");
    } finally {
      setApplyingDisposition(false);
    }
  };

  const handleViewTriageReport = () => {
    const id = getEncounterId();
    if (!id) return;
    navigate(`/report/doctor/${id}`);
  };

  const specialty = encounter?.triage?.assigned_specialty || encounter?.assigned_specialty || "General Medicine";
  const severity = encounter?.triage?.aiTriageReport?.esi_level || encounter?.severity || encounter?.priority_code || 0;
  const severityConfig = getSeverityConfig(severity);

  return (
    <div style={{ padding: "24px", maxWidth: 1240, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <div>
          <p style={{ margin: 0, fontSize: 12, letterSpacing: "0.18em", textTransform: "uppercase", color: "#64748b" }}>Diagnosis workflow</p>
          <h1 style={{ margin: "8px 0 0 0", fontSize: 32, color: "#111827", lineHeight: 1.1 }}>Clinical diagnosis & disposition</h1>
          <p style={{ margin: "10px 0 0 0", color: "#475569", maxWidth: 680 }}>Review the patient encounter, record the diagnosis, and place the patient into the correct inpatient pathway.</p>
        </div>
        <button
          onClick={() => navigate(-1)}
          style={{ padding: "12px 18px", borderRadius: 14, border: "1px solid #cbd5e1", background: "white", color: "#475569", fontWeight: 700, cursor: "pointer" }}
        >
          ← Back to dashboard
        </button>
      </div>

      {loading ? (
        <div style={{ padding: 32, borderRadius: 20, background: "#f8fafc", textAlign: "center", color: "#64748b" }}>Loading encounter details…</div>
      ) : error ? (
        <div style={{ padding: 28, borderRadius: 18, background: "#fde8e8", color: "#991b1b" }}>⚠️ {error}</div>
      ) : !encounter ? (
        <div style={{ padding: 28, borderRadius: 18, background: "#f8fafc", color: "#475569" }}>Encounter not found.</div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "1.7fr 1fr", gap: 24 }}>
          <div style={{ background: "white", borderRadius: 24, padding: 28, boxShadow: "0 18px 45px rgba(15,23,42,0.08)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
              <div>
                <p style={{ margin: 0, fontSize: 12, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.14em" }}>Patient</p>
                <h2 style={{ margin: "8px 0 0 0", fontSize: 24, color: "#0f172a" }}>{encounter.patientName || encounter.firstName || "Patient"}</h2>
                <p style={{ margin: "8px 0 0 0", color: "#475569", fontSize: 14 }}>ID: {encounter._id?.substring(0, 10) || encounter.id || "Unknown"}</p>
              </div>
              <div style={{ display: "grid", gap: 10, textAlign: "right" }}>
                <span style={{ padding: "10px 16px", borderRadius: 999, background: severityConfig.bg, color: severityConfig.color, fontWeight: 700, fontSize: 12 }}>{severityConfig.label}</span>
                <span style={{ padding: "10px 16px", borderRadius: 999, background: "#eef2ff", color: "#4338ca", fontSize: 12, fontWeight: 700 }}>{specialty}</span>
              </div>
            </div>

            <section style={{ marginBottom: 24 }}>
              <h3 style={{ margin: 0, color: "#111827", fontSize: 16, fontWeight: 700 }}>Triage summary</h3>
              <p style={{ margin: "12px 0 0 0", color: "#475569", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
                {encounter.triage?.aiTriageReport?.ai_summary || encounter.triage?.aiTriageReport?.chief_complaint || encounter.triage?.summary || "No triage summary available."}
              </p>
            </section>

            <section style={{ display: "grid", gap: 16, marginBottom: 24 }}>
              <div style={{ padding: 18, borderRadius: 20, background: "#f8fafc", border: "1px solid #e2e8f0" }}>
                <p style={{ margin: 0, color: "#64748b", fontSize: 13, letterSpacing: "0.12em", textTransform: "uppercase" }}>Current status</p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 14 }}>
                  <span style={{ fontWeight: 700, color: "#0f172a" }}>{encounter.disposition || encounter.status || "Pending"}</span>
                  <span style={{ padding: "8px 12px", background: "#eef2ff", color: "#4338ca", borderRadius: 14, fontSize: 12, fontWeight: 700 }}>{encounter.handover?.handover_status || "Not acknowledged"}</span>
                </div>
              </div>
              <div style={{ padding: 18, borderRadius: 20, background: "#f8fafc", border: "1px solid #e2e8f0" }}>
                <p style={{ margin: 0, color: "#64748b", fontSize: 13, letterSpacing: "0.12em", textTransform: "uppercase" }}>Admission guidance</p>
                <p style={{ margin: "12px 0 0 0", color: "#475569", lineHeight: 1.7 }}>
                  {encounter.triage?.aiTriageReport?.recommended_treatment || encounter.triage?.recommendation || "Review the triage recommendation and mark the patient for hospital admission or safe discharge."}
                </p>
              </div>
            </section>

            <section style={{ marginBottom: 24 }}>
              <label htmlFor="diagnosis" style={{ display: "block", marginBottom: 10, fontWeight: 700, color: "#111827" }}>Final diagnosis</label>
              <textarea
                id="diagnosis"
                value={diagnosisText}
                onChange={(e) => setDiagnosisText(e.target.value)}
                placeholder="Enter diagnosis here..."
                style={{ width: "100%", minHeight: 180, padding: 18, borderRadius: 18, border: "1px solid #d1d5db", fontSize: 14, color: "#0f172a", resize: "vertical" }}
              />
              <p style={{ margin: "10px 0 0 0", color: "#64748b", fontSize: 13 }}>
                The diagnosis is recorded automatically when you admit or discharge the patient.
              </p>
            </section>

            {message && (
              <div style={{ padding: "14px 18px", marginBottom: 18, borderRadius: 18, background: "#ecfdf5", color: "#166534" }}>
                {message}
                {encounter?.status === 'DISCHARGED' && (
                  <div style={{ marginTop: 10 }}>
                    Final EHR record complete. <button
                      onClick={handleViewTriageReport}
                      style={{ background: 'transparent', border: 'none', color: '#1d4ed8', fontWeight: 700, cursor: 'pointer', textDecoration: 'underline' }}
                    >View physician report</button>
                  </div>
                )}
              </div>
            )}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
              <button
                onClick={handleViewTriageReport}
                style={{ padding: "14px 20px", borderRadius: 16, border: "1px solid #c7d2fe", background: "white", color: "#3730a3", fontWeight: 700, cursor: "pointer", minWidth: 180 }}
              >
                View physician triage report
              </button>
            </div>
          </div>

          <aside style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ padding: 28, borderRadius: 24, background: "linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.96))", color: "white", minHeight: 320, boxShadow: "0 18px 45px rgba(15,23,42,0.14)" }}>
              <p style={{ margin: 0, textTransform: "uppercase", letterSpacing: "0.18em", fontSize: 12, color: "#93c5fd" }}>Actions</p>
              <h2 style={{ margin: "16px 0 0 0", fontSize: 22, lineHeight: 1.2 }}>Inpatient disposition</h2>
              <p style={{ margin: "16px 0 0 0", color: "#cbd5e1", fontSize: 14, lineHeight: 1.7 }}>
                Admit or discharge the patient once the clinical diagnosis is confirmed. Use this workflow to keep the decision in the same patient context.
              </p>
              <div style={{ display: "grid", gap: 12, marginTop: 24 }}>
                <button
                  onClick={() => handleDisposition("HOSPITALIZED")}
                  disabled={applyingDisposition}
                  style={{ padding: "14px 18px", borderRadius: 18, border: "none", background: "#10b981", color: "white", fontWeight: 700, cursor: applyingDisposition ? "wait" : "pointer" }}
                >
                  {applyingDisposition ? "Processing…" : "🛏️ Admit to hospital"}
                </button>
                <button
                  onClick={() => handleDisposition("DISCHARGED")}
                  disabled={applyingDisposition}
                  style={{ padding: "14px 18px", borderRadius: 18, border: "1px solid rgba(255,255,255,0.24)", background: "rgba(255,255,255,0.08)", color: "white", fontWeight: 700, cursor: applyingDisposition ? "wait" : "pointer" }}
                >
                  {applyingDisposition ? "Processing…" : "🚪 Discharge patient"}
                </button>
              </div>
            </div>

            <div style={{ padding: 24, borderRadius: 24, background: "#f8fafc", border: "1px solid #e2e8f0" }}>
              <p style={{ margin: 0, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Encounter details</p>
              <div style={{ marginTop: 18, display: "grid", gap: 14, color: "#0f172a" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Ward</span>
                  <strong>{encounter.ward || encounter.assigned_ward || "Unassigned"}</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Assigned physician</span>
                  <strong>
                    {typeof encounter.assigned_physician === 'object' && encounter.assigned_physician?.name
                      ? encounter.assigned_physician.name
                      : encounter.assigned_physician || encounter.physician || "Unassigned"}
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Specialty</span>
                  <strong>{specialty}</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Disposition</span>
                  <strong>{encounter.disposition || encounter.status || "Pending"}</strong>
                </div>
              </div>
            </div>

            <div style={{ padding: 22, borderRadius: 24, background: "#ffffff", border: "1px solid #e2e8f0" }}>
              <p style={{ margin: 0, fontSize: 12, letterSpacing: "0.14em", textTransform: "uppercase", color: "#64748b" }}>Clinical snapshot</p>
              <div style={{ marginTop: 18, display: "grid", gap: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", color: "#475569" }}>
                  <span>Primary diagnosis</span>
                  <strong>{encounter.diagnosis || "TBD"}</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", color: "#475569" }}>
                  <span>Presenting complaint</span>
                  <strong>{encounter.chiefComplaint || encounter.triage?.aiTriageReport?.chief_complaint || "Unknown"}</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", color: "#475569" }}>
                  <span>Last update</span>
                  <strong>{encounter.updatedAt ? new Date(encounter.updatedAt).toLocaleString() : "Unknown"}</strong>
                </div>
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
};

export default DiagnosePage;
