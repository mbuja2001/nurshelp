import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * NurseConfirmedReportView
 * Read-only display of the final nurse-confirmed patient report.
 * This is the authoritative version saved by the nurse after editing the AI-suggested fields.
 * No editable fields; no action buttons (just View and Back).
 */
export default function NurseConfirmedReportView() {
  const params = useParams();
  // Support both /confirmed/:id (report ID) and /confirmed/encounter/:encounterId
  const reportId = params.id;
  const encounterId = params.encounterId;
  
  const navigate = useNavigate();

  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch the confirmed report
  useEffect(() => {
    if (!reportId && !encounterId) {
      setError("No report or encounter ID provided");
      setLoading(false);
      return;
    }

    let mounted = true;

    const fetchReport = async () => {
      setLoading(true);
      setError(null);
      try {
        const token = localStorage.getItem("token");
        const headers = {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache"
        };

        const cacheBreaker = `?t=${Date.now()}`;
        let url;
        
        // Fetch by report ID or encounter ID
        if (reportId) {
          url = `${BACKEND_URL}/api/reports/nurse-confirmed/${reportId}${cacheBreaker}`;
        } else {
          url = `${BACKEND_URL}/api/reports/nurse-confirmed/encounter/${encounterId}${cacheBreaker}`;
        }
        
        const res = await fetch(url, { headers, cache: "no-store" });

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(
            errBody.message || `Failed to load confirmed report (status ${res.status})`
          );
        }

        const data = await res.json().catch(() => null);
        const confirmedReport = data?.report ?? data;
        if (!confirmedReport) throw new Error("Confirmed report response malformed");

        if (mounted) setReport(confirmedReport);
      } catch (err) {
        console.error("Fetch confirmed report error:", err);
        if (mounted) setError(err.message || "Failed to load confirmed report");
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchReport();
    return () => { mounted = false; };
  }, [reportId, encounterId]);

  // Helper: time ago
  const timeAgo = (ts) => {
    if (!ts) return "N/A";
    const diff = Math.floor((Date.now() - new Date(ts)) / 1000);
    const mins = Math.floor(diff / 60);
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    return `${hrs}h ${mins % 60}m`;
  };

  // Helper: format utility
  const fmt = (v, unit = "") => {
    if (v === undefined || v === null) return `--${unit}`;
    return v === "--" ? `--${unit}` : `${v}${unit}`;
  };

  // Helper: severity color
  const severityColor = (sev) => {
    const s = Number(sev);
    if (s === 3) return "#ef4444"; // High - red
    if (s === 2) return "#f59e0b"; // Medium - orange
    return "#10b981"; // Low - green
  };

  // Render views
  if (loading) return <div style={{ padding: 24 }}>Loading confirmed report…</div>;
  if (error)
    return (
      <div style={{ padding: 24, color: "#ef4444" }}>
        Error: {error}
        <div style={{ marginTop: 12 }}>
          <button onClick={() => navigate("/")} style={{ padding: "8px 16px" }}>
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  if (!report) return <div style={{ padding: 24 }}>No confirmed report found.</div>;

  // Normalize fields
  const patient = report.patient ?? {};
  const vitals = report.vitals ?? {};
  const triage = report.triage ?? {};
  const assignedPhysician = triage.assigned_physician ?? null;

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1 style={{ margin: 0 }}>Final Nurse Confirmed Report</h1>
          <button onClick={() => navigate("/")} style={{ padding: "8px 16px" }}>
            Back
          </button>
        </div>
        <div
          style={{
            fontSize: 13,
            color: "#475569",
            marginTop: 8
          }}
        >
          <strong>Status:</strong> Confirmed •{" "}
          <strong>Confirmed at:</strong>{" "}
          {report.confirmedAt ? new Date(report.confirmedAt).toLocaleString() : "N/A"}
        </div>
      </div>

      {/* READ-ONLY Content Display */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 320px",
          gap: 24
        }}
      >
        {/* Main Content */}
        <main style={{ minWidth: 0 }}>
          {/* Patient Info Section */}
          <section
            style={{
              padding: 16,
              marginBottom: 20,
              borderRadius: 8,
              background: "#f8fafc",
              border: "1px solid #e2e8f0"
            }}
          >
            <h3 style={{ margin: "0 0 12px 0" }}>Patient Information</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>Name</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>
                  {patient.name ?? "N/A"}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>Age</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>
                  {patient.age ?? "N/A"}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>Gender</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>
                  {patient.gender ?? "N/A"}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>ID</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>
                  {patient.medicalRecordNumber ?? patient.id ?? "N/A"}
                </div>
              </div>
            </div>
          </section>

          {/* Vitals Section */}
          <section
            style={{
              padding: 16,
              marginBottom: 20,
              borderRadius: 8,
              background: "#f8fafc",
              border: "1px solid #e2e8f0"
            }}
          >
            <h3 style={{ margin: "0 0 12px 0" }}>Vitals</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>HR</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{fmt(vitals.hr, " bpm")}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>O₂</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{fmt(vitals.o2, "%")}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>Resp</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{fmt(vitals.resp, " /min")}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>Temp</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{fmt(vitals.temp, "°C")}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>BP</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>
                  {vitals.bp || vitals.bp_systolic
                    ? vitals.bp ||
                      `${fmt(vitals.bp_systolic, "")}/${fmt(vitals.bp_diastolic, "")}`
                    : "N/A"}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#64748b" }}>AVPU</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{fmt(vitals.avpu)}</div>
              </div>
            </div>
          </section>

          {/* Chief Complaint */}
          {(triage?.chief_complaint || patient.chief_complaint) && (
            <section
              style={{
                padding: 16,
                marginBottom: 20,
                borderRadius: 8,
                background: "#f8fafc",
                border: "1px solid #e2e8f0"
              }}
            >
              <h3 style={{ margin: "0 0 12px 0" }}>Chief Complaint</h3>
              <div style={{ fontSize: 14, color: "#1e293b", whiteSpace: "pre-wrap" }}>
                {triage?.chief_complaint || patient.chief_complaint || "N/A"}
              </div>
            </section>
          )}

          {/* History of Present Illness */}
          {(triage?.history_of_present_illness || patient.history) && (
            <section
              style={{
                padding: 16,
                marginBottom: 20,
                borderRadius: 8,
                background: "#f8fafc",
                border: "1px solid #e2e8f0"
              }}
            >
              <h3 style={{ margin: "0 0 12px 0" }}>History of Present Illness</h3>
              <div style={{ fontSize: 14, color: "#1e293b", whiteSpace: "pre-wrap" }}>
                {triage?.history_of_present_illness || patient.history || "N/A"}
              </div>
            </section>
          )}

          {/* Assigned Physician */}
          {assignedPhysician && (
            <section
              style={{
                padding: 16,
                marginBottom: 20,
                borderRadius: 8,
                background: "#dbeafe",
                border: "1px solid #0284c7"
              }}
            >
              <h3 style={{ margin: "0 0 12px 0" }}>Assigned Attending Physician</h3>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#0c4a6e" }}>
                {assignedPhysician.name || "N/A"}
              </div>
              {assignedPhysician.specialty && (
                <div style={{ fontSize: 12, color: "#0c4a6e", marginTop: 4 }}>
                  {assignedPhysician.specialty}
                </div>
              )}
            </section>
          )}

          {/* Nurse Notes */}
          {report.nurseNotes && (
            <section
              style={{
                padding: 16,
                marginBottom: 20,
                borderRadius: 8,
                background: "#fef3c7",
                border: "1px solid #f59e0b"
              }}
            >
              <h3 style={{ margin: "0 0 12px 0" }}>Nurse Notes</h3>
              <div style={{ fontSize: 14, color: "#78350f", whiteSpace: "pre-wrap" }}>
                {report.nurseNotes}
              </div>
            </section>
          )}

          {/* Recommendations / AI Analysis */}
          {Array.isArray(report.recommendations) && report.recommendations.length > 0 && (
            <section
              style={{
                padding: 16,
                marginBottom: 20,
                borderRadius: 8,
                background: "#ecfdf5",
                border: "1px solid #10b981"
              }}
            >
              <h3 style={{ margin: "0 0 12px 0" }}>Recommendations</h3>
              <ul style={{ margin: 0, paddingLeft: 16 }}>
                {report.recommendations.map((rec, idx) => (
                  <li key={idx} style={{ fontSize: 14, color: "#065f46", marginBottom: 6 }}>
                    {typeof rec === "string" ? rec : JSON.stringify(rec)}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </main>

        {/* Right Sidebar */}
        <aside style={{ minWidth: 0 }}>
          {/* Summary Card */}
          <div
            style={{
              padding: 12,
              marginBottom: 16,
              borderRadius: 8,
              background: "#f8fafc",
              border: "1px solid #e2e8f0"
            }}
          >
            <h4 style={{ margin: "0 0 10px 0" }}>Report Summary</h4>
            <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.8 }}>
              <div>
                <strong>Status:</strong> <span style={{ color: "#10b981", fontWeight: 600 }}>Confirmed</span>
              </div>
              <div style={{ marginTop: 8 }}>
                <strong>Severity:</strong>{" "}
                <span
                  style={{
                    color: severityColor(report.severity),
                    fontWeight: 600
                  }}
                >
                  {report.severity === 3
                    ? "High"
                    : report.severity === 2
                    ? "Medium"
                    : "Low"}
                </span>
              </div>
              <div style={{ marginTop: 8 }}>
                <strong>Confirmed by:</strong>{" "}
                <div style={{ fontSize: 11, marginTop: 2 }}>
                  {report.confirmedByNurse || "Staff Nurse"}
                </div>
              </div>
              <div style={{ marginTop: 8 }}>
                <strong>Confirmed at:</strong>{" "}
                <div style={{ fontSize: 11, marginTop: 2 }}>
                  {report.confirmedAt
                    ? new Date(report.confirmedAt).toLocaleString()
                    : "N/A"}
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <button
              onClick={() => navigate("/")}
              style={{
                padding: "10px 12px",
                borderRadius: 6,
                border: "none",
                background: "#3b82f6",
                color: "#fff",
                fontWeight: 600,
                cursor: "pointer",
                fontSize: 13
              }}
            >
              Back to Dashboard
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}
