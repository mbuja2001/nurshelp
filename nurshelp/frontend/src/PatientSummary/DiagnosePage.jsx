import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";

const DiagnosePage = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [encounter, setEncounter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [applyingDisposition, setApplyingDisposition] = useState(false);

  // Form state
  const [diagnosisText, setDiagnosisText] = useState("");
  const [treatments, setTreatments] = useState([]);
  const [newTreatment, setNewTreatment] = useState({ name: "", intervention_type: "MEDICATION", dosage: "", frequency: "", route: "Oral", indication: "" });
  const [recommendations, setRecommendations] = useState([]);
  const [newRecommendation, setNewRecommendation] = useState({ description: "", recommendation_type: "ADMISSION", priority: "MEDIUM" });
  const [progressionNote, setProgressionNote] = useState("");
  const [medicationRequests, setMedicationRequests] = useState([]);
  const [newMedicationRequest, setNewMedicationRequest] = useState({ medication_name: "", dosage: "", frequency: "", route: "Oral", indication: "" });

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem("token");

  // Fetch encounter
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const headers = { Authorization: `Bearer ${token}` };
        const res = await fetch(`${BACKEND_URL}/api/encounters/${id}`, { headers });
        if (!res.ok) throw new Error("Failed to load encounter");
        
        const data = await res.json();
        const enc = data.encounter;
        setEncounter(enc);
        setDiagnosisText(enc.diagnosis || "");
        setTreatments(enc.treatments || []);
        setRecommendations(enc.recommendations || []);
        setMedicationRequests(enc.medication_requests || []);
      } catch (err) {
        console.error("Fetch error:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (id) fetchData();
  }, [id, token, BACKEND_URL]);

  // Persist diagnosis & clinical data
  const persistDiagnosis = async (overrideDiagnosis = diagnosisText, requestList = medicationRequests) => {
    const headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json"
    };

    const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/diagnosis`, {
      method: "PUT",
      headers,
      body: JSON.stringify({
        diagnosis: overrideDiagnosis?.trim() || "Diagnosis recorded by physician",
        treatment_plan: "", // Will be built from treatments array
        notes: progressionNote || "Clinical diagnosis and treatment plan documented",
        treatments: treatments,
        recommendations: recommendations,
        progression_notes: progressionNote ? [{ content: progressionNote, note_type: "ASSESSMENT", recorded_at: new Date() }] : [],
        medication_requests: requestList
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.message || "Failed to record diagnosis");
    }

    const data = await res.json();
    return data.encounter || data;
  };

  const persistInpatientProgressReport = async (overrideDiagnosis = diagnosisText, requestList = medicationRequests) => {
    const headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json"
    };

    const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/inpatient-report`, {
      method: "PUT",
      headers,
      body: JSON.stringify({
        diagnosis: overrideDiagnosis?.trim() || "Inpatient progression documented by physician",
        treatment_plan: "",
        notes: progressionNote || "Inpatient progression report documented",
        treatments: treatments,
        recommendations: recommendations,
        progression_notes: progressionNote ? [{ content: progressionNote, note_type: "PROGRESS", recorded_at: new Date() }] : [],
        medication_requests: requestList
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.message || "Failed to create inpatient report");
    }

    const data = await res.json();
    return data.encounter || data;
  };

  const handleAddTreatment = () => {
    if (newTreatment.name.trim()) {
      setTreatments([...treatments, { ...newTreatment, prescribed_at: new Date(), status: "PRESCRIBED" }]);
      setNewTreatment({ name: "", intervention_type: "MEDICATION", dosage: "", frequency: "", route: "Oral", indication: "" });
    }
  };

  const handleRemoveTreatment = (idx) => {
    setTreatments(treatments.filter((_, i) => i !== idx));
  };

  const handleAddRecommendation = () => {
    if (newRecommendation.description.trim()) {
      setRecommendations([...recommendations, { ...newRecommendation, recommended_at: new Date(), status: "PENDING" }]);
      setNewRecommendation({ description: "", recommendation_type: "ADMISSION", priority: "MEDIUM" });
    }
  };

  const handleRemoveRecommendation = (idx) => {
    setRecommendations(recommendations.filter((_, i) => i !== idx));
  };

  const handleAddMedicationRequest = async () => {
    if (!newMedicationRequest.medication_name.trim()) return;

    const nextMedicationRequests = [...medicationRequests, {
      ...newMedicationRequest,
      status: 'PENDING',
      requested_at: new Date()
    }];

    setMedicationRequests(nextMedicationRequests);
    setNewMedicationRequest({ medication_name: '', dosage: '', frequency: '', route: 'Oral', indication: '' });

    try {
      const persistedEncounter = isAlreadyHospitalized
        ? await persistInpatientProgressReport(diagnosisText, nextMedicationRequests)
        : await persistDiagnosis(diagnosisText, nextMedicationRequests);

      if (persistedEncounter) {
        setEncounter(persistedEncounter);
        setMessage('Medication request queued for pharmacy review.');
      }
    } catch (err) {
      console.error('Medication request save error:', err);
      setError(err.message || 'Failed to queue medication request for pharmacy review');
    }
  };

  const handleRemoveMedicationRequest = (idx) => {
    setMedicationRequests(medicationRequests.filter((_, i) => i !== idx));
  };

  const handleDisposition = async (disposition) => {
    if (disposition === "DISCHARGED") {
      const confirmed = window.confirm("Discharge this patient from inpatient care?");
      if (!confirmed) return;
    }

    setApplyingDisposition(true);
    setError(null);
    setMessage(null);

    try {
      const isAlreadyHospitalized = Boolean(
        encounter?.isHospitalized ||
        encounter?.currentBedId ||
        encounter?.assigned_bed ||
        encounter?.triage?.assigned_bed?.id ||
        encounter?.status === "HOSPITALIZED"
      );

      if (disposition === "HOSPITALIZED" && isAlreadyHospitalized) {
        const savedEncounter = diagnosisText?.trim() ? await persistInpatientProgressReport(diagnosisText) : encounter;
        setEncounter(savedEncounter || encounter);
        setMessage("Patient is already hospitalized. Saved this as a new inpatient progression report for the current stay.");
        return;
      }

      const savedEncounter = diagnosisText?.trim() ? await persistDiagnosis(diagnosisText) : encounter;

      if (disposition === "HOSPITALIZED") {
        navigate(`/assign-ward-bed/${id}`, { state: { encounter: savedEncounter || encounter } });
        return;
      }

      const headers = {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json"
      };

      const res = await fetch(`${BACKEND_URL}/api/encounters/${id}/disposition`, {
        method: "PUT",
        headers,
        body: JSON.stringify({
          disposition,
          disposition_notes: disposition === "DISCHARGED" ? "Discharged from diagnosis workflow" : "Admitted from diagnosis workflow"
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.message || `Failed to ${disposition.toLowerCase()} patient`);
      }

      const data = await res.json();
      setEncounter(data.encounter || data);
      setMessage(`Patient ${disposition === "HOSPITALIZED" ? "admitted" : "discharged"} successfully with clinical documentation.`);
    } catch (err) {
      console.error("Disposition error:", err);
      setError(err.message || "Unable to update disposition");
    } finally {
      setApplyingDisposition(false);
    }
  };

  const handleCreateInpatientProgressReport = async () => {
    if (!window.confirm("Save this as a new inpatient progression report for the current stay?")) {
      return;
    }

    setApplyingDisposition(true);
    setError(null);
    setMessage(null);

    try {
      const savedEncounter = diagnosisText?.trim() ? await persistInpatientProgressReport(diagnosisText) : encounter;
      setEncounter(savedEncounter || encounter);
      setMessage("New inpatient report saved to the active stay.");
      navigate(`/report/inpatient/${id}`);
    } catch (err) {
      console.error("Inpatient report error:", err);
      setError(err.message || "Unable to create inpatient progress report");
    } finally {
      setApplyingDisposition(false);
    }
  };

  const handleViewTriageReport = () => navigate(`/report/doctor/${id}`);

  const severity = encounter?.triage?.aiTriageReport?.esi_level || encounter?.severity || 4;
  const isAlreadyHospitalized = Boolean(
    encounter?.isHospitalized ||
    encounter?.currentBedId ||
    encounter?.assigned_bed ||
    encounter?.triage?.assigned_bed?.id ||
    encounter?.status === "HOSPITALIZED"
  );
  const severityConfig = {
    1: { label: 'CRITICAL', color: '#dc2626', bg: '#fee2e2' },
    2: { label: 'HIGH', color: '#ea580c', bg: '#ffedd5' },
    3: { label: 'MODERATE', color: '#b45309', bg: '#fef3c7' },
    4: { label: 'STABLE', color: '#059669', bg: '#d1fae5' },
    5: { label: 'MINIMAL', color: '#0891b2', bg: '#cffafe' }
  }[severity] || { label: 'UNKNOWN', color: '#475569', bg: '#f1f5f9' };

  const specialty = encounter?.triage?.assigned_specialty || encounter?.assigned_specialty || "General Medicine";
  const patientName = encounter?.patientName || encounter?.firstName || "Patient";

  if (loading) return <div style={{ padding: 32, textAlign: "center", color: "#64748b" }}>Loading encounter…</div>;
  if (error) return <div style={{ padding: 28, borderRadius: 18, background: "#fde8e8", color: "#991b1b" }}>⚠️ {error}</div>;
  if (!encounter) return <div style={{ padding: 28, borderRadius: 18, background: "#f8fafc", color: "#475569" }}>Encounter not found.</div>;

  return (
    <div style={{ padding: "24px", maxWidth: 1400, margin: "0 auto" }}>
      {/* HEADER */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32, gap: 16 }}>
        <div>
          <p style={{ margin: 0, fontSize: 12, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.14em" }}>Clinical Decision</p>
          <h1 style={{ margin: "8px 0 0 0", fontSize: 32, color: "#0f172a", fontWeight: 700 }}>Diagnosis & Treatment Plan</h1>
        </div>
        <button
          onClick={() => navigate(-1)}
          style={{ padding: "12px 18px", borderRadius: 14, border: "1px solid #cbd5e1", background: "white", color: "#475569", fontWeight: 700, cursor: "pointer" }}
        >
          ← Back
        </button>
      </div>

      {/* MAIN GRID: Primary Clinical Info + Disposition Actions */}
      <div style={{ display: "grid", gridTemplateColumns: "2.2fr 1fr", gap: 28, marginBottom: 32 }}>
        
        {/* LEFT: Patient Context & Clinical Overview */}
        <div style={{ background: "white", borderRadius: 24, padding: 28, boxShadow: "0 4px 12px rgba(0,0,0,0.06)", border: "1px solid #e5e7eb" }}>
          
          {/* Patient Header */}
          <div style={{ marginBottom: 28, paddingBottom: 20, borderBottom: "2px solid #f3f4f6" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 16 }}>
              <div>
                <h2 style={{ margin: "0 0 4px 0", fontSize: 24, color: "#111827", fontWeight: 700 }}>{patientName}</h2>
                <p style={{ margin: "4px 0 0 0", color: "#6b7280", fontSize: 14 }}>ID: {encounter._id?.substring(0, 10)}</p>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <span style={{ padding: "8px 14px", borderRadius: 999, background: severityConfig.bg, color: severityConfig.color, fontWeight: 700, fontSize: 12 }}>{severityConfig.label}</span>
                <span style={{ padding: "8px 14px", borderRadius: 999, background: "#f0f9ff", color: "#0369a1", fontWeight: 700, fontSize: 12 }}>{specialty}</span>
              </div>
            </div>
          </div>

          {/* Clinical Context */}
          <section style={{ marginBottom: 28 }}>
            <h3 style={{ margin: "0 0 12px 0", fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>Clinical context</h3>
            <p style={{ margin: 0, color: "#475569", lineHeight: 1.7, fontSize: 14 }}>
              {encounter.triage?.aiTriageReport?.ai_summary || encounter.triage?.summary || "No summary available"}
            </p>
          </section>

          {/* PRIMARY DIAGNOSIS */}
          <section style={{ marginBottom: 28 }}>
            <label style={{ display: "block", marginBottom: 12, fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Primary Diagnosis
            </label>
            <textarea
              value={diagnosisText}
              onChange={(e) => setDiagnosisText(e.target.value)}
              placeholder="Document the physician's clinical diagnosis..."
              style={{
                width: "100%",
                minHeight: 120,
                padding: 14,
                borderRadius: 12,
                border: "1px solid #d1d5db",
                fontSize: 14,
                color: "#0f172a",
                fontFamily: "inherit",
                resize: "vertical"
              }}
            />
            <p style={{ margin: "8px 0 0 0", color: "#6b7280", fontSize: 12 }}>Clinical diagnosis is recorded when admission/discharge is processed.</p>
          </section>

          {/* TREATMENTS */}
          <section style={{ marginBottom: 28 }}>
            <h3 style={{ margin: "0 0 16px 0", fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>Treatment Plan</h3>

            {/* Add Treatment */}
            <div style={{ padding: 16, background: "#f9fafb", borderRadius: 12, border: "1px solid #e5e7eb", marginBottom: 16 }}>
              <p style={{ margin: "0 0 12px 0", fontSize: 12, color: "#6b7280", fontWeight: 600 }}>ADD INTERVENTION</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <select
                  value={newTreatment.intervention_type}
                  onChange={(e) => setNewTreatment({ ...newTreatment, intervention_type: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option>MEDICATION</option>
                  <option>PROCEDURE</option>
                  <option>THERAPY</option>
                  <option>MONITORING</option>
                  <option>IMAGING</option>
                  <option>LAB</option>
                </select>
                <input
                  type="text"
                  placeholder="Intervention name (e.g., Ibuprofen, Chest X-ray)"
                  value={newTreatment.name}
                  onChange={(e) => setNewTreatment({ ...newTreatment, name: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 12 }}>
                <input
                  type="text"
                  placeholder="Dosage (e.g., 500mg)"
                  value={newTreatment.dosage}
                  onChange={(e) => setNewTreatment({ ...newTreatment, dosage: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
                <input
                  type="text"
                  placeholder="Frequency (e.g., Q4-6h)"
                  value={newTreatment.frequency}
                  onChange={(e) => setNewTreatment({ ...newTreatment, frequency: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
                <select
                  value={newTreatment.route}
                  onChange={(e) => setNewTreatment({ ...newTreatment, route: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option>Oral</option>
                  <option>IV</option>
                  <option>IM</option>
                  <option>SC</option>
                  <option>Topical</option>
                </select>
              </div>
              <input
                type="text"
                placeholder="Indication (why this treatment?)"
                value={newTreatment.indication}
                onChange={(e) => setNewTreatment({ ...newTreatment, indication: e.target.value })}
                style={{ width: "100%", padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, marginBottom: 12 }}
              />
              <button
                onClick={handleAddTreatment}
                style={{ width: "100%", padding: 10, background: "#3b82f6", color: "white", border: "none", borderRadius: 8, fontWeight: 600, cursor: "pointer", fontSize: 13 }}
              >
                + Add treatment
              </button>
            </div>

            {/* Treatment List */}
            {treatments.length > 0 && (
              <div style={{ display: "grid", gap: 10 }}>
                {treatments.map((t, idx) => (
                  <div key={idx} style={{ padding: 12, background: "#f0f9ff", borderRadius: 10, border: "1px solid #bfdbfe", display: "flex", justifyContent: "space-between", alignItems: "start" }}>
                    <div style={{ fontSize: 13, color: "#0c4a6e" }}>
                      <strong>{t.name}</strong> ({t.intervention_type}) <br/>
                      {t.dosage && <span>{t.dosage}</span>} {t.frequency && <span>• {t.frequency}</span>} {t.route && <span>• {t.route}</span>}
                      {t.indication && <div style={{ marginTop: 4, color: "#0369a1", fontSize: 12 }}>→ {t.indication}</div>}
                    </div>
                    <button
                      onClick={() => handleRemoveTreatment(idx)}
                      style={{ background: "transparent", border: "none", color: "#dc2626", cursor: "pointer", fontWeight: 700, fontSize: 16 }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* RECOMMENDATIONS */}
          <section style={{ marginBottom: 28 }}>
            <h3 style={{ margin: "0 0 16px 0", fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>Clinical Recommendations</h3>

            {/* Add Recommendation */}
            <div style={{ padding: 16, background: "#f9fafb", borderRadius: 12, border: "1px solid #e5e7eb", marginBottom: 16 }}>
              <p style={{ margin: "0 0 12px 0", fontSize: 12, color: "#6b7280", fontWeight: 600 }}>ADD RECOMMENDATION</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <select
                  value={newRecommendation.recommendation_type}
                  onChange={(e) => setNewRecommendation({ ...newRecommendation, recommendation_type: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option>ADMISSION</option>
                  <option>TRANSFER</option>
                  <option>SPECIALTY_CONSULT</option>
                  <option>FOLLOW_UP</option>
                  <option>DISCHARGE_INSTRUCTION</option>
                  <option>DIAGNOSTIC_TEST</option>
                </select>
                <select
                  value={newRecommendation.priority}
                  onChange={(e) => setNewRecommendation({ ...newRecommendation, priority: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option value="URGENT">🔴 URGENT</option>
                  <option value="HIGH">🟠 HIGH</option>
                  <option value="MEDIUM">🟡 MEDIUM</option>
                  <option value="LOW">🟢 LOW</option>
                </select>
              </div>
              <textarea
                placeholder="Recommendation details..."
                value={newRecommendation.description}
                onChange={(e) => setNewRecommendation({ ...newRecommendation, description: e.target.value })}
                style={{ width: "100%", minHeight: 70, padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13, marginBottom: 12, fontFamily: "inherit" }}
              />
              <button
                onClick={handleAddRecommendation}
                style={{ width: "100%", padding: 10, background: "#10b981", color: "white", border: "none", borderRadius: 8, fontWeight: 600, cursor: "pointer", fontSize: 13 }}
              >
                + Add recommendation
              </button>
            </div>

            {/* Recommendation List */}
            {recommendations.length > 0 && (
              <div style={{ display: "grid", gap: 10 }}>
                {recommendations.map((r, idx) => (
                  <div key={idx} style={{ padding: 12, background: "#f0fdf4", borderRadius: 10, border: "1px solid #dcfce7", display: "flex", justifyContent: "space-between", alignItems: "start" }}>
                    <div style={{ fontSize: 13, color: "#14532d" }}>
                      <strong>{r.recommendation_type}</strong> • {r.priority} <br/>
                      {r.description}
                    </div>
                    <button
                      onClick={() => handleRemoveRecommendation(idx)}
                      style={{ background: "transparent", border: "none", color: "#dc2626", cursor: "pointer", fontWeight: 700, fontSize: 16 }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* MEDICATION REQUESTS */}
          <section style={{ marginBottom: 28 }}>
            <h3 style={{ margin: "0 0 16px 0", fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>Medication Requests for Pharmacy Review</h3>
            <p style={{ margin: "0 0 12px 0", color: "#64748b", fontSize: 12 }}>These requests are routed to the pharmacy review queue for pharmacist authorization and follow-up.</p>
            <div style={{ padding: 16, background: "#fef3c7", borderRadius: 12, border: "1px solid #fde68a", marginBottom: 12 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.7fr 0.7fr", gap: 12, marginBottom: 12 }}>
                <input
                  type="text"
                  placeholder="Medication name"
                  value={newMedicationRequest.medication_name}
                  onChange={(e) => setNewMedicationRequest({ ...newMedicationRequest, medication_name: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
                <input
                  type="text"
                  placeholder="Dose"
                  value={newMedicationRequest.dosage}
                  onChange={(e) => setNewMedicationRequest({ ...newMedicationRequest, dosage: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
                <input
                  type="text"
                  placeholder="Frequency"
                  value={newMedicationRequest.frequency}
                  onChange={(e) => setNewMedicationRequest({ ...newMedicationRequest, frequency: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "0.7fr 1.2fr", gap: 12, marginBottom: 12 }}>
                <select
                  value={newMedicationRequest.route}
                  onChange={(e) => setNewMedicationRequest({ ...newMedicationRequest, route: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                >
                  <option>Oral</option>
                  <option>IV</option>
                  <option>IM</option>
                  <option>SC</option>
                  <option>Topical</option>
                </select>
                <input
                  type="text"
                  placeholder="Clinical indication"
                  value={newMedicationRequest.indication}
                  onChange={(e) => setNewMedicationRequest({ ...newMedicationRequest, indication: e.target.value })}
                  style={{ padding: 10, borderRadius: 8, border: "1px solid #d1d5db", fontSize: 13 }}
                />
              </div>
              <button
                onClick={handleAddMedicationRequest}
                style={{ width: "100%", padding: 10, background: "#f59e0b", color: "white", border: "none", borderRadius: 8, fontWeight: 600, cursor: "pointer", fontSize: 13 }}
              >
                + Send to pharmacy
              </button>
            </div>
            {medicationRequests.length > 0 && (
              <div style={{ display: "grid", gap: 10 }}>
                {medicationRequests.map((request, idx) => (
                  <div key={idx} style={{ padding: 12, background: "#fffbeb", borderRadius: 10, border: "1px solid #fde68a", display: "flex", justifyContent: "space-between", alignItems: "start" }}>
                    <div style={{ fontSize: 13, color: "#92400e" }}>
                      <strong>{request.medication_name || request.name}</strong> • {request.dosage || '—'} • {request.frequency || '—'} • {request.route || 'Oral'}
                      {request.indication ? <div style={{ marginTop: 4, color: "#a16207" }}>→ {request.indication}</div> : null}
                      <div style={{ marginTop: 4, fontSize: 12, color: "#b45309" }}>Status: {request.status || 'PENDING'}</div>
                    </div>
                    <button
                      onClick={() => handleRemoveMedicationRequest(idx)}
                      style={{ background: "transparent", border: "none", color: "#dc2626", cursor: "pointer", fontWeight: 700, fontSize: 16 }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* PROGRESSION NOTES */}
          <section>
            <label style={{ display: "block", marginBottom: 12, fontSize: 14, fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Clinical Progression Notes
            </label>
            <textarea
              value={progressionNote}
              onChange={(e) => setProgressionNote(e.target.value)}
              placeholder="Document patient progress, response to treatment, clinical assessment..."
              style={{
                width: "100%",
                minHeight: 100,
                padding: 14,
                borderRadius: 12,
                border: "1px solid #d1d5db",
                fontSize: 14,
                color: "#0f172a",
                fontFamily: "inherit",
                resize: "vertical"
              }}
            />
          </section>

          {/* Messages */}
          {message && (
            <div style={{ marginTop: 20, padding: 14, borderRadius: 12, background: "#d1fae5", color: "#166534", fontSize: 14 }}>
              ✓ {message}
            </div>
          )}

          {/* Actions */}
          <div style={{ marginTop: 28, display: "flex", gap: 12 }}>
            <button
              onClick={handleViewTriageReport}
              style={{ padding: "12px 20px", borderRadius: 10, border: "1px solid #d1d5db", background: "white", color: "#3b82f6", fontWeight: 600, cursor: "pointer", fontSize: 13 }}
            >
              📋 View triage report
            </button>
          </div>
        </div>

        {/* RIGHT: Disposition Actions */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20, height: "fit-content" }}>
          
          {/* Inpatient Disposition */}
          <div style={{ padding: 24, background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)", borderRadius: 16, color: "white" }}>
            <p style={{ margin: 0, fontSize: 11, color: "#93c5fd", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em" }}>Disposition</p>
            <h3 style={{ margin: "12px 0 0 0", fontSize: 18, fontWeight: 700, lineHeight: 1.3 }}>Admission Status</h3>
            <p style={{ margin: "12px 0 0 0", color: "#cbd5e1", fontSize: 13, lineHeight: 1.6 }}>
              Admit patient to inpatient ward for continued monitoring and treatment, or discharge to home with instructions.
            </p>
            <div style={{ display: "grid", gap: 10, marginTop: 20 }}>
              <button
                onClick={() => handleDisposition("HOSPITALIZED")}
                disabled={applyingDisposition || isAlreadyHospitalized}
                style={{
                  padding: "14px 16px",
                  borderRadius: 10,
                  border: "none",
                  background: isAlreadyHospitalized ? "#64748b" : "#10b981",
                  color: "white",
                  fontWeight: 700,
                  cursor: applyingDisposition || isAlreadyHospitalized ? "not-allowed" : "pointer",
                  fontSize: 13,
                  opacity: applyingDisposition || isAlreadyHospitalized ? 0.7 : 1
                }}
              >
                {applyingDisposition ? "Processing…" : isAlreadyHospitalized ? "🛑 Already Hospitalized" : "🏥 Admit to Hospital"}
              </button>
              <button
                onClick={handleCreateInpatientProgressReport}
                disabled={applyingDisposition}
                style={{
                  padding: "14px 16px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.3)",
                  background: "#2563eb",
                  color: "white",
                  fontWeight: 700,
                  cursor: applyingDisposition ? "wait" : "pointer",
                  fontSize: 13,
                  opacity: applyingDisposition ? 0.6 : 1
                }}
              >
                {applyingDisposition ? "Processing…" : "📝 Create new inpatient report"}
              </button>
              <button
                onClick={() => handleDisposition("DISCHARGED")}
                disabled={applyingDisposition}
                style={{
                  padding: "14px 16px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.3)",
                  background: "rgba(255,255,255,0.1)",
                  color: "white",
                  fontWeight: 700,
                  cursor: applyingDisposition ? "wait" : "pointer",
                  fontSize: 13,
                  opacity: applyingDisposition ? 0.6 : 1
                }}
              >
                {applyingDisposition ? "Processing…" : "🚪 Discharge Patient"}
              </button>
            </div>
          </div>

          {/* Clinical Summary */}
          <div style={{ padding: 20, background: "white", borderRadius: 16, border: "1px solid #e5e7eb" }}>
            <p style={{ margin: 0, fontSize: 11, color: "#6b7280", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em" }}>Summary</p>
            <div style={{ marginTop: 16, display: "grid", gap: 12, fontSize: 13 }}>
              <div>
                <p style={{ margin: 0, color: "#6b7280", fontSize: 11 }}>ASSIGNED PHYSICIAN</p>
                <p style={{ margin: "4px 0 0 0", fontWeight: 600, color: "#111827" }}>
                  {typeof encounter.assigned_physician === 'object' && encounter.assigned_physician?.name
                    ? encounter.assigned_physician.name
                    : encounter.assigned_physician || encounter.physician || "Unassigned"}
                </p>
              </div>
              <div>
                <p style={{ margin: 0, color: "#6b7280", fontSize: 11 }}>SPECIALTY</p>
                <p style={{ margin: "4px 0 0 0", fontWeight: 600, color: "#111827" }}>{specialty}</p>
              </div>
              <div>
                <p style={{ margin: 0, color: "#6b7280", fontSize: 11 }}>STATUS</p>
                <p style={{ margin: "4px 0 0 0", fontWeight: 600, color: "#111827" }}>{encounter.status || "Pending"}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiagnosePage;
