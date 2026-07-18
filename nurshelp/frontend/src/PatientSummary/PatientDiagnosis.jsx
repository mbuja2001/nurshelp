import React, { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const PatientDiagnosis = () => {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const initialEncounter = location.state?.encounter ?? null;
  const [encounter, setEncounter] = useState(initialEncounter);
  const [loading, setLoading] = useState(!initialEncounter);
  const [saving, setSaving] = useState(false);
  const [dispositionLoading, setDispositionLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState(initialEncounter?.diagnosis || "");
  const [treatmentPlan, setTreatmentPlan] = useState(initialEncounter?.treatment_plan || "");
  const [notes, setNotes] = useState(initialEncounter?.physician_notes || "");
  const [message, setMessage] = useState(null);

  const fetchEncounter = async (encounterId) => {
    const token = localStorage.getItem("token");
    if (!token) {
      setMessage("Authentication required to load patient data.");
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.message || `Unable to load encounter ${encounterId}`);
      }

      const data = await res.json();
      const encounterData = data.encounter || data.case || data;
      setEncounter(encounterData);
      setDiagnosis(encounterData.diagnosis || "");
      setTreatmentPlan(encounterData.treatment_plan || "");
      setNotes(encounterData.physician_notes || "");
      setMessage(null);
    } catch (err) {
      console.error("[PatientDiagnosis] Fetch error:", err);
      setMessage(err.message || "Failed to load encounter details.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!encounter && id) {
      fetchEncounter(id);
    }
  }, [id]);

  const handleSubmitDiagnosis = async () => {
    if (!id) {
      alert("Missing encounter identifier.");
      return;
    }

    setSaving(true);
    const token = localStorage.getItem("token");
    try {
      const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/diagnosis`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          diagnosis,
          treatment_plan: treatmentPlan,
          notes
        })
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.message || "Failed to save diagnosis");
      }

      const data = await res.json();
      setEncounter(data.encounter || encounter);
      setMessage("Diagnosis saved successfully.");
    } catch (err) {
      console.error("[PatientDiagnosis] Save error:", err);
      alert(err.message || "Unable to save diagnosis.");
    } finally {
      setSaving(false);
    }
  };

  const handleDisposition = async (disposition) => {
    if (!id) {
      alert("Missing encounter identifier.");
      return;
    }

    if (disposition === "DISCHARGED") {
      const confirm = window.confirm(`Confirm discharge for ${encounter?.patientName || "this patient"}?`);
      if (!confirm) return;
    }

    setDispositionLoading(true);
    const token = localStorage.getItem("token");
    try {
      const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/disposition`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          disposition,
          disposition_notes: `${disposition === "HOSPITALIZED" ? "Admitted" : "Discharged"} by physician via diagnosis workflow`
        })
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.message || `Failed to update disposition to ${disposition}`);
      }

      const data = await res.json();
      setEncounter(data.encounter || encounter);
      setMessage(`Patient ${disposition === "HOSPITALIZED" ? "admitted" : "discharged"} successfully.`);
    } catch (err) {
      console.error("[PatientDiagnosis] Disposition error:", err);
      alert(err.message || "Unable to update disposition.");
    } finally {
      setDispositionLoading(false);
    }
  };

  const renderPatientInfo = () => {
    if (!encounter) return null;
    const patientName = encounter.patientName || encounter.firstName || "Unknown Patient";
    const triageReport = encounter.triage?.aiTriageReport || encounter.triage || {};
    const severity = triageReport.esi_level || encounter.severity || "N/A";
    const specialty = triageReport.assigned_specialty || encounter.assigned_specialty || "General";

    return (
      <div style={{ display: "grid", gap: "18px", marginTop: "18px" }}>
        <div style={{ borderRadius: "24px", background: "#ffffff", padding: "22px", boxShadow: "0 18px 45px rgba(15, 23, 42, 0.07)", border: "1px solid #e2e8f0" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "16px", flexWrap: "wrap" }}>
            <div>
              <p style={{ margin: 0, color: "#475569", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Patient</p>
              <h1 style={{ margin: "8px 0 0", fontSize: "24px", color: "#0f172a" }}>{patientName}</h1>
              <p style={{ margin: "6px 0 0", color: "#64748b", fontSize: "14px" }}>Encounter ID: {id}</p>
            </div>
            <div style={{ display: "grid", gap: "8px", minWidth: "200px" }}>
              <div style={{ padding: "14px 18px", borderRadius: "18px", background: "#f8fafc", border: "1px solid #e2e8f0" }}>
                <p style={{ margin: 0, color: "#475569", fontSize: "12px" }}>Severity</p>
                <p style={{ margin: "8px 0 0", color: "#0f172a", fontWeight: 700 }}>{severity}</p>
              </div>
              <div style={{ padding: "14px 18px", borderRadius: "18px", background: "#eff6ff", border: "1px solid #bfdbfe" }}>
                <p style={{ margin: 0, color: "#1e40af", fontSize: "12px" }}>Assigned Specialty</p>
                <p style={{ margin: "8px 0 0", color: "#1e3a8a", fontWeight: 700 }}>{specialty}</p>
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "18px" }}>
          <section style={{ borderRadius: "24px", background: "#0f172a", color: "#e2e8f0", padding: "24px", minHeight: "280px", boxShadow: "0 18px 45px rgba(15, 23, 42, 0.12)" }}>
            <p style={{ textTransform: "uppercase", letterSpacing: "0.12em", fontSize: "12px", color: "#94a3b8", margin: 0 }}>Triage Summary</p>
            <h2 style={{ margin: "16px 0 8px", fontSize: "20px", color: "#f8fafc" }}>Clinical Context</h2>
            <p style={{ margin: 0, lineHeight: 1.8, color: "#cbd5e1", fontSize: "14px" }}>
              {triageReport.ai_summary || triageReport.summary || encounter.clinical_summary || "No triage summary available."}
            </p>
            <div style={{ marginTop: "24px", display: "grid", gap: "14px" }}>
              <div style={{ display: "grid", gap: "6px" }}>
                <span style={{ color: "#94a3b8", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Primary diagnosis suggestion</span>
                <strong style={{ color: "#f8fafc" }}>{triageReport.primary_diagnosis || "Pending clinician review"}</strong>
              </div>
              <div style={{ display: "grid", gap: "6px" }}>
                <span style={{ color: "#94a3b8", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Recommended treatment</span>
                <strong style={{ color: "#f8fafc" }}>{triageReport.recommended_treatment || "No recommendation yet"}</strong>
              </div>
            </div>
          </section>

          <aside style={{ display: "grid", gap: "18px" }}>
            <div style={{ borderRadius: "24px", background: "#ffffff", padding: "22px", boxShadow: "0 18px 45px rgba(15, 23, 42, 0.06)", border: "1px solid #e2e8f0" }}>
              <p style={{ margin: 0, color: "#475569", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Status</p>
              <p style={{ margin: "10px 0 0", fontSize: "18px", fontWeight: 700, color: "#0f172a" }}>{encounter.status || "ADMITTED"}</p>
              <p style={{ margin: "10px 0 0", color: "#64748b", fontSize: "13px" }}>{encounter.disposition_notes || "Diagnosis workflow in progress."}</p>
            </div>
            <div style={{ borderRadius: "24px", background: "rgba(59, 130, 246, 0.1)", padding: "22px", border: "1px solid rgba(59, 130, 246, 0.2)" }}>
              <p style={{ margin: 0, color: "#1e3a8a", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Live encounter details</p>
              <div style={{ marginTop: "16px", display: "grid", gap: "12px", color: "#0f172a" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Patient age</span><span>{encounter.age || encounter.patient?.age || "N/A"}</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Gender</span><span>{encounter.gender || encounter.patient?.gender || "N/A"}</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Bed</span><span>{encounter.assigned_bed?.bedId || encounter.assigned_bed || "Unassigned"}</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Ward</span><span>{encounter.assigned_ward || encounter.ward || "Unassigned"}</span></div>
              </div>
            </div>
          </aside>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "18px" }}>
          <section style={{ borderRadius: "24px", background: "#ffffff", padding: "24px", border: "1px solid #e2e8f0", boxShadow: "0 18px 45px rgba(15, 23, 42, 0.05)" }}>
            <p style={{ margin: 0, color: "#475569", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Physician diagnosis</p>
            <h2 style={{ margin: "16px 0 20px", fontSize: "20px", color: "#0f172a" }}>Clinical decision</h2>
            <div style={{ display: "grid", gap: "16px" }}>
              <div style={{ display: "grid", gap: "8px" }}>
                <label style={{ fontSize: "13px", color: "#475569" }}>Diagnosis</label>
                <textarea
                  value={diagnosis}
                  onChange={(e) => setDiagnosis(e.target.value)}
                  placeholder="Enter final diagnosis"
                  rows={4}
                  style={{ width: "100%", minHeight: "120px", borderRadius: "16px", border: "1px solid #d1d5db", padding: "14px", resize: "vertical", fontSize: "14px", color: "#0f172a" }}
                />
              </div>
              <div style={{ display: "grid", gap: "8px" }}>
                <label style={{ fontSize: "13px", color: "#475569" }}>Treatment plan</label>
                <textarea
                  value={treatmentPlan}
                  onChange={(e) => setTreatmentPlan(e.target.value)}
                  placeholder="Describe recommended treatment or admission plan"
                  rows={4}
                  style={{ width: "100%", minHeight: "120px", borderRadius: "16px", border: "1px solid #d1d5db", padding: "14px", resize: "vertical", fontSize: "14px", color: "#0f172a" }}
                />
              </div>
              <div style={{ display: "grid", gap: "8px" }}>
                <label style={{ fontSize: "13px", color: "#475569" }}>Physician notes</label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add any clinical notes, admission rationale, or instructions"
                  rows={4}
                  style={{ width: "100%", minHeight: "120px", borderRadius: "16px", border: "1px solid #d1d5db", padding: "14px", resize: "vertical", fontSize: "14px", color: "#0f172a" }}
                />
              </div>
            </div>
          </section>

          <aside style={{ display: "grid", gap: "18px" }}>
            <div style={{ borderRadius: "24px", background: "#f8fafc", padding: "20px", border: "1px solid #e2e8f0" }}>
              <p style={{ margin: 0, color: "#475569", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Inpatient disposition</p>
              <div style={{ marginTop: "18px", display: "grid", gap: "12px" }}>
                <button
                  onClick={() => handleDisposition("HOSPITALIZED")}
                  disabled={dispositionLoading}
                  style={{ width: "100%", padding: "14px 16px", background: "#10b981", color: "white", border: "none", borderRadius: "16px", cursor: dispositionLoading ? "wait" : "pointer", fontWeight: 700, fontSize: "14px" }}
                >
                  {dispositionLoading ? "Processing..." : "Admit to Hospital"}
                </button>
                <button
                  onClick={() => handleDisposition("DISCHARGED")}
                  disabled={dispositionLoading}
                  style={{ width: "100%", padding: "14px 16px", background: "#ef4444", color: "white", border: "none", borderRadius: "16px", cursor: dispositionLoading ? "wait" : "pointer", fontWeight: 700, fontSize: "14px" }}
                >
                  {dispositionLoading ? "Processing..." : "Discharge Patient"}
                </button>
              </div>
            </div>

            <div style={{ borderRadius: "24px", background: "#ffffff", padding: "22px", border: "1px solid #e2e8f0", boxShadow: "0 18px 35px rgba(15, 23, 42, 0.04)" }}>
              <p style={{ margin: 0, color: "#475569", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Quick actions</p>
              <div style={{ marginTop: "16px", display: "grid", gap: "12px" }}>
                <button
                  onClick={handleSubmitDiagnosis}
                  disabled={saving}
                  style={{ width: "100%", padding: "14px 16px", background: "#6366f1", color: "white", border: "none", borderRadius: "16px", cursor: saving ? "wait" : "pointer", fontWeight: 700, fontSize: "14px" }}
                >
                  {saving ? "Saving diagnosis…" : "Save diagnosis"}
                </button>
                <button
                  onClick={() => navigate(-1)}
                  style={{ width: "100%", padding: "14px 16px", background: "#f8fafc", color: "#0f172a", border: "1px solid #cbd5e1", borderRadius: "16px", cursor: "pointer", fontWeight: 700, fontSize: "14px" }}
                >
                  Close and return
                </button>
              </div>
            </div>

            {message && (
              <div style={{ borderRadius: "20px", padding: "16px", background: "#e0f2fe", color: "#0c4a6e", border: "1px solid #7dd3fc" }}>
                {message}
              </div>
            )}
          </aside>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: "24px", minHeight: "100vh", background: "#f8fafc" }}>
      <div style={{ maxWidth: 1280, margin: "0 auto", display: "grid", gap: "18px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "14px" }}>
          <div>
            <p style={{ margin: 0, color: "#64748b", fontSize: "13px", textTransform: "uppercase", letterSpacing: "0.12em" }}>Diagnosis workflow</p>
            <h1 style={{ margin: "10px 0 0", fontSize: "28px", color: "#0f172a" }}>Clinical diagnosis & disposition</h1>
            <p style={{ margin: "8px 0 0", color: "#475569", fontSize: "15px" }}>
              Review the patient context, record the diagnosis, then admit or discharge using the inpatient workflow.
            </p>
          </div>
          <button
            onClick={() => navigate("/dashboard")}
            style={{ padding: "14px 18px", borderRadius: "16px", border: "1px solid #cbd5e1", background: "white", color: "#0f172a", cursor: "pointer", fontWeight: 700 }}
          >
            Back to Dashboard
          </button>
        </div>

        {loading ? (
          <div style={{ padding: "60px", borderRadius: "24px", background: "white", textAlign: "center", border: "1px solid #e2e8f0" }}>
            Loading patient details…
          </div>
        ) : message && !encounter ? (
          <div style={{ padding: "36px", borderRadius: "24px", background: "#fee2e2", color: "#991b1b", border: "1px solid #fecaca" }}>
            {message}
          </div>
        ) : (
          renderPatientInfo()
        )}
      </div>
    </div>
  );
};

export default PatientDiagnosis;
