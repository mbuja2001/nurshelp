import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

const SurgeryPage = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const [form, setForm] = useState({
    procedure_name: "",
    surgery_type: "ELECTIVE",
    priority: "MEDIUM",
    rationale: "",
    scheduled_at: "",
    surgeon: "",
    surgical_team: "",
    notes: "",
    status: "PENDING",
    outcome: "",
    findings: "",
    complications: ""
  });

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem("token");
  const user = JSON.parse(localStorage.getItem("user") || "{}");

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const headers = { Authorization: `Bearer ${token}` };
        const res = await fetch(`${BACKEND_URL}/api/encounters/${id}?reportType=SURGERY`, { headers });
        if (!res.ok) throw new Error("Failed to load encounter");

        const data = await res.json();
        const enc = data.encounter;
        setEncounter(enc);
        setForm((prev) => ({
          ...prev,
          surgeon: enc?.assignedPhysicianName || user?.name || prev.surgeon || ""
        }));
      } catch (err) {
        console.error("Fetch error:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (id) fetchData();
  }, [id, token, BACKEND_URL, user?.name]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setMessage(null);

    try {
      const headers = {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json"
      };

      const payload = {
        procedure_name: form.procedure_name.trim(),
        surgery_type: form.surgery_type,
        priority: form.priority,
        rationale: form.rationale.trim(),
        scheduled_at: form.scheduled_at || null,
        surgeon: form.surgeon.trim(),
        surgical_team: form.surgical_team.split(",").map((item) => item.trim()).filter(Boolean),
        notes: form.notes.trim()
      };

      const createRes = await fetch(`${BACKEND_URL}/api/encounters/${id}/surgery`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload)
      });

      if (!createRes.ok) {
        const errData = await createRes.json().catch(() => ({}));
        throw new Error(errData.message || "Failed to create surgery record");
      }

      const createData = await createRes.json();
      const surgeryRecord = createData?.surgeryRecord;
      let savedRecord = surgeryRecord;

      if (surgeryRecord && (form.status || form.outcome || form.findings || form.complications)) {
        const updateRes = await fetch(`${BACKEND_URL}/api/encounters/${id}/surgery/${surgeryRecord._id || surgeryRecord.surgery_id}`, {
          method: "PATCH",
          headers,
          body: JSON.stringify({
            status: form.status,
            outcome: form.outcome.trim(),
            findings: form.findings.trim(),
            complications: form.complications.trim(),
            notes: form.notes.trim()
          })
        });

        if (!updateRes.ok) {
          const errData = await updateRes.json().catch(() => ({}));
          throw new Error(errData.message || "Failed to record surgery outcome");
        }

        const updateData = await updateRes.json();
        savedRecord = updateData?.surgeryRecord || surgeryRecord;
      }

      setEncounter((prev) => prev ? {
        ...prev,
        surgery_records: [...(prev.surgery_records || []), savedRecord]
      } : prev);
      setMessage("Surgery workflow saved successfully.");
    } catch (err) {
      console.error("Surgery save error:", err);
      setError(err.message || "Unable to save surgery workflow");
    } finally {
      setSubmitting(false);
    }
  };

  const severity = encounter?.triage?.aiTriageReport?.esi_level || encounter?.severity || 4;
  const severityConfig = {
    1: { label: "CRITICAL", color: "#dc2626", bg: "#fee2e2" },
    2: { label: "HIGH", color: "#ea580c", bg: "#ffedd5" },
    3: { label: "MODERATE", color: "#b45309", bg: "#fef3c7" },
    4: { label: "STABLE", color: "#059669", bg: "#d1fae5" },
    5: { label: "MINIMAL", color: "#0891b2", bg: "#cffafe" }
  }[severity] || { label: "UNKNOWN", color: "#475569", bg: "#f1f5f9" };

  const specialty = encounter?.triage?.assigned_specialty || encounter?.assigned_specialty || "General Medicine";
  const patientName = encounter?.patientName || encounter?.patient?.name || encounter?.firstName || "Patient";

  if (loading) return <div style={{ padding: 32, textAlign: "center", color: "#64748b" }}>Loading surgery workflow…</div>;
  if (error) return <div style={{ padding: 28, borderRadius: 18, background: "#fde8e8", color: "#991b1b" }}>⚠️ {error}</div>;
  if (!encounter) return <div style={{ padding: 28, borderRadius: 18, background: "#f8fafc", color: "#475569" }}>Encounter not found.</div>;

  return (
    <div style={{ padding: "24px", maxWidth: 1400, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32, gap: 16 }}>
        <div>
          <p style={{ margin: 0, fontSize: 12, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.14em" }}>Clinical Decision</p>
          <h1 style={{ margin: "8px 0 0 0", fontSize: 32, color: "#0f172a", fontWeight: 700 }}>Surgery Workflow</h1>
        </div>
        <button
          onClick={() => navigate(-1)}
          style={{ padding: "12px 18px", borderRadius: 14, border: "1px solid #cbd5e1", background: "white", color: "#475569", fontWeight: 700, cursor: "pointer" }}
        >
          ← Back
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2.2fr 1fr", gap: 28, marginBottom: 32 }}>
        <div style={{ background: "white", borderRadius: 24, padding: 28, boxShadow: "0 4px 12px rgba(0,0,0,0.06)", border: "1px solid #e5e7eb" }}>
          <div style={{ marginBottom: 28, paddingBottom: 20, borderBottom: "2px solid #f3f4f6" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 16 }}>
              <div>
                <h2 style={{ margin: "0 0 4px 0", fontSize: 24, color: "#111827", fontWeight: 700 }}>{patientName}</h2>
                <p style={{ margin: "4px 0 0 0", color: "#6b7280", fontSize: 14 }}>ID: {encounter._id?.substring(0, 10)}</p>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <span style={{ padding: "8px 14px", borderRadius: 999, background: severityConfig.bg, color: severityConfig.color, fontWeight: 700, fontSize: 12 }}>{severityConfig.label}</span>
                <span style={{ padding: "8px 14px", borderRadius: 999, background: "#eff6ff", color: "#2563eb", fontWeight: 700, fontSize: 12 }}>{specialty}</span>
              </div>
            </div>
          </div>

          <section style={{ marginBottom: 28 }}>
            <h3 style={{ margin: "0 0 12px 0", fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>Clinical context</h3>
            <p style={{ margin: 0, color: "#475569", lineHeight: 1.7, fontSize: 14 }}>
              {encounter.triage?.aiTriageReport?.ai_summary || encounter.triage?.summary || "No summary available"}
            </p>
          </section>

          <section style={{ marginBottom: 28 }}>
            <h3 style={{ margin: "0 0 16px 0", fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>Surgery request details</h3>
            <div style={{ padding: 16, background: "#f9fafb", borderRadius: 12, border: "1px solid #e5e7eb" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <select
                  value={form.surgery_type}
                  onChange={(e) => setForm({ ...form, surgery_type: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option value="ELECTIVE">ELECTIVE</option>
                  <option value="EMERGENCY">EMERGENCY</option>
                  <option value="URGENT">URGENT</option>
                </select>
                <select
                  value={form.priority}
                  onChange={(e) => setForm({ ...form, priority: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option value="URGENT">🔴 URGENT</option>
                  <option value="HIGH">🟠 HIGH</option>
                  <option value="MEDIUM">🟡 MEDIUM</option>
                  <option value="LOW">🟢 LOW</option>
                </select>
              </div>
              <div style={{ marginBottom: 12 }}>
                <input
                  type="text"
                  placeholder="Procedure name"
                  value={form.procedure_name}
                  onChange={(e) => setForm({ ...form, procedure_name: e.target.value })}
                  style={{ width: "100%", padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
              </div>
              <div style={{ marginBottom: 12 }}>
                <textarea
                  placeholder="Clinical rationale for the procedure"
                  value={form.rationale}
                  onChange={(e) => setForm({ ...form, rationale: e.target.value })}
                  style={{ width: "100%", minHeight: 80, padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, fontFamily: "inherit" }}
                />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <input
                  type="datetime-local"
                  value={form.scheduled_at}
                  onChange={(e) => setForm({ ...form, scheduled_at: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
                <input
                  type="text"
                  placeholder="Surgeon"
                  value={form.surgeon}
                  onChange={(e) => setForm({ ...form, surgeon: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
              </div>
              <div style={{ marginBottom: 12 }}>
                <input
                  type="text"
                  placeholder="Surgical team (comma separated)"
                  value={form.surgical_team}
                  onChange={(e) => setForm({ ...form, surgical_team: e.target.value })}
                  style={{ width: "100%", padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
              </div>
              <div style={{ marginBottom: 12 }}>
                <textarea
                  placeholder="Perioperative notes"
                  value={form.notes}
                  onChange={(e) => setForm({ ...form, notes: e.target.value })}
                  style={{ width: "100%", minHeight: 90, padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, fontFamily: "inherit" }}
                />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <select
                  value={form.status}
                  onChange={(e) => setForm({ ...form, status: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option value="PENDING">PENDING</option>
                  <option value="SCHEDULED">SCHEDULED</option>
                  <option value="IN_PROGRESS">IN PROGRESS</option>
                  <option value="COMPLETED">COMPLETED</option>
                  <option value="POST_OP_REVIEW">POST-OP REVIEW</option>
                  <option value="COMPLICATION">COMPLICATION</option>
                </select>
                <input
                  type="text"
                  placeholder="Outcome"
                  value={form.outcome}
                  onChange={(e) => setForm({ ...form, outcome: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <textarea
                  placeholder="Operative findings"
                  value={form.findings}
                  onChange={(e) => setForm({ ...form, findings: e.target.value })}
                  style={{ width: "100%", minHeight: 70, padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, fontFamily: "inherit" }}
                />
                <textarea
                  placeholder="Complications"
                  value={form.complications}
                  onChange={(e) => setForm({ ...form, complications: e.target.value })}
                  style={{ width: "100%", minHeight: 70, padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, fontFamily: "inherit" }}
                />
              </div>
              <button
                onClick={handleSubmit}
                disabled={submitting || !form.procedure_name.trim()}
                style={{ width: "100%", padding: 12, borderRadius: 10, border: "none", background: submitting ? "#64748b" : "#0f766e", color: "white", fontWeight: 700, cursor: submitting ? "wait" : "pointer" }}
              >
                {submitting ? "Saving surgery workflow…" : "Save surgery workflow"}
              </button>
            </div>
          </section>

          {message && <div style={{ marginTop: 12, padding: 12, borderRadius: 10, background: "#dcfce7", color: "#166534" }}>✓ {message}</div>}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 20, height: "fit-content" }}>
          <div style={{ padding: 24, background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)", borderRadius: 16, color: "white" }}>
            <p style={{ margin: 0, fontSize: 11, color: "#93c5fd", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em" }}>Surgery actions</p>
            <h3 style={{ margin: "12px 0 0 0", fontSize: 18, fontWeight: 700, lineHeight: 1.3 }}>Perioperative workflow</h3>
            <p style={{ margin: "12px 0 0 0", color: "#cbd5e1", fontSize: 13, lineHeight: 1.6 }}>
              Document the planned procedure, operative team, and outcomes directly against the patient encounter.
            </p>
            <div style={{ display: "grid", gap: 10, marginTop: 20 }}>
              <button
                onClick={handleSubmit}
                disabled={submitting || !form.procedure_name.trim()}
                style={{ padding: "14px 16px", borderRadius: 10, border: "none", background: "#14b8a6", color: "white", fontWeight: 700, cursor: submitting ? "wait" : "pointer", fontSize: 13 }}
              >
                {submitting ? "Saving…" : "📝 Save surgery record"}
              </button>
              <button
                onClick={() => navigate(-1)}
                style={{ padding: "14px 16px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.3)", background: "rgba(255,255,255,0.1)", color: "white", fontWeight: 700, cursor: "pointer", fontSize: 13 }}
              >
                ← Return to dashboard
              </button>
            </div>
          </div>

          <div style={{ padding: 20, background: "white", borderRadius: 16, border: "1px solid #e5e7eb" }}>
            <p style={{ margin: 0, fontSize: 11, color: "#6b7280", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em" }}>Recorded surgery history</p>
            <div style={{ marginTop: 16, display: "grid", gap: 12, fontSize: 13 }}>
              {(encounter.surgery_records || []).length > 0 ? encounter.surgery_records.map((record, idx) => (
                <div key={record._id || idx} style={{ padding: 12, background: "#f8fafc", borderRadius: 10, border: "1px solid #e2e8f0" }}>
                  <strong>{record.procedure_name || "Surgery record"}</strong>
                  <div style={{ marginTop: 4, color: "#475569" }}>Status: {record.status || "PENDING"}</div>
                  {record.outcome ? <div style={{ marginTop: 4, color: "#047857" }}>Outcome: {record.outcome}</div> : null}
                </div>
              )) : <p style={{ margin: 0, color: "#64748b" }}>No surgery records yet.</p>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SurgeryPage;
