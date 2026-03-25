// src/PatientSummary/Components/RightColumn.jsx
import React, { useState, useEffect } from "react";

export default function RightColumn({ vitalsData = {}, triage = null, onOpenForm, onSavePatient, saving }) {
  const [bedAssigned, setBedAssigned] = useState(false);
  const [doctorNotified, setDoctorNotified] = useState(false);
  const [timeLeft, setTimeLeft] = useState(null);

  // -----------------------------
  // Extract AI fields
  // -----------------------------
  const severity = triage?.priority_code ?? null;
  const summary = triage?.ai_summary ?? "";
  const reasoning = triage?.SATS_reasoning ?? null;

  // -----------------------------
  // Severity mapping (SATS aligned)
  // -----------------------------
  const severityMap = {
    1: { label: "RED", color: "#ef4444", time: 0 },
    2: { label: "ORANGE", color: "#f97316", time: 10 },
    3: { label: "YELLOW", color: "#eab308", time: 60 },
    4: { label: "GREEN", color: "#22c55e", time: 240 }
  };

  const sev = severityMap[severity] || null;

  // -----------------------------
  // Countdown Timer (Time-to-Treat KPI)
  // -----------------------------
  useEffect(() => {
    if (!sev?.time) return;

    const totalSeconds = sev.time * 60;
    setTimeLeft(totalSeconds);

    const interval = setInterval(() => {
      setTimeLeft(prev => (prev > 0 ? prev - 1 : 0));
    }, 1000);

    return () => clearInterval(interval);
  }, [severity]);

  const formatTime = (sec) => {
    if (sec === null) return "--";
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}m ${s}s`;
  };

  // -----------------------------
  // PDF download
  // -----------------------------
  const handleDownload = async () => {
    try {
      const res = await fetch(`/api/pdf/generate-pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vitals: vitalsData, triage })
      });

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "patient-summary.pdf";
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert("PDF generation failed");
    }
  };

  return (
    <div className="right-column-layout">

      {/* ------------------ */}
      {/* 🔴 CASE SEVERITY */}
      {/* ------------------ */}
      <div className="block severity-block">
        <h4 className="block-label">Triage Category</h4>

        {!sev ? (
          <div style={{ color: "#777", margin: "16px 0" }}>— Not Assessed —</div>
        ) : (
          <>
            <div
              style={{
                background: sev.color,
                color: "white",
                borderRadius: "12px",
                padding: "20px",
                textAlign: "center",
                fontWeight: "bold",
                fontSize: "20px"
              }}
            >
              {sev.label}
            </div>

            {sev.time > 0 && (
              <p style={{ marginTop: 10, fontWeight: "bold" }}>
                ⏱ Target: {formatTime(timeLeft)}
              </p>
            )}
          </>
        )}
      </div>

      {/* ------------------ */}
      {/* 🧠 AI SUMMARY */}
      {/* ------------------ */}
      {summary && (
        <div className="block">
          <h4 className="block-label">AI Clinical Summary</h4>
          <div className="content-area">{summary}</div>
        </div>
      )}

      {/* ------------------ */}
      {/* 🔍 EXPLAINABILITY */}
      {/* ------------------ */}
      {reasoning && (
        <div className="block">
          <details>
            <summary style={{ cursor: "pointer", fontWeight: "bold" }}>
              Show Clinical Logic
            </summary>
            <pre style={{ fontSize: "12px", marginTop: "10px" }}>
              {JSON.stringify(reasoning, null, 2)}
            </pre>
          </details>
        </div>
      )}

      {/* ------------------ */}
      {/* VITALS */}
      {/* ------------------ */}
      <div className="block">
        <div className="block-header">
          <h4 className="block-label">Patient Vitals & Queue</h4>
          <button className="edit-action-btn" onClick={onOpenForm}>✎</button>
        </div>

        <div className="vitals-display-list">
          <div className="vital-item"><span>Temp:</span> <strong>{vitalsData?.temp ?? '--'}°C</strong></div>
          <div className="vital-item"><span>BP:</span> <strong>{vitalsData?.bp ?? '--'}</strong></div>
          <div className="vital-item"><span>BP Sys:</span> <strong>{vitalsData?.bp_systolic ?? '--'}</strong></div>
          <div className="vital-item"><span>BP Dia:</span> <strong>{vitalsData?.bp_diastolic ?? '--'}</strong></div>
          
          {/* Left Arm BP */}
          <div className="vital-item" style={{ borderTop: '1px solid #e5e7eb', paddingTop: '8px', marginTop: '8px' }}>
            <span style={{ fontWeight: '600', color: '#4b5563' }}>BP Left</span>
          </div>
          <div className="vital-item"><span>Sys:</span> <strong>{vitalsData?.bp_left_systolic ?? '--'}</strong></div>
          <div className="vital-item"><span>Dia:</span> <strong>{vitalsData?.bp_left_diastolic ?? '--'}</strong></div>
          
          {/* Right Arm BP */}
          <div className="vital-item">
            <span style={{ fontWeight: '600', color: '#4b5563' }}>BP Right</span>
          </div>
          <div className="vital-item"><span>Sys:</span> <strong>{vitalsData?.bp_right_systolic ?? '--'}</strong></div>
          <div className="vital-item"><span>Dia:</span> <strong>{vitalsData?.bp_right_diastolic ?? '--'}</strong></div>
          
          <div className="vital-item"><span>HR:</span> <strong>{vitalsData?.hr ?? '--'}</strong></div>
          <div className="vital-item"><span>O₂:</span> <strong>{vitalsData?.o2 ?? '--'}%</strong></div>
          <div className="vital-item"><span>Resp:</span> <strong>{vitalsData?.resp ?? '--'}</strong></div>
          <div className="vital-item"><span>AVPU:</span> <strong>{vitalsData?.avpu ?? '--'}</strong></div>
        </div>
      </div>

      {/* ------------------ */}
      {/* CHECKLIST */}
      {/* ------------------ */}
      <div className="block">
        <div className="action-checklist">
          <label className={`checklist-item ${bedAssigned ? 'completed' : ''}`}>
            <input type="checkbox" checked={bedAssigned} onChange={() => setBedAssigned(!bedAssigned)} />
            <span>Assign Bed</span>
          </label>

          <label className={`checklist-item ${doctorNotified ? 'completed' : ''}`}>
            <input type="checkbox" checked={doctorNotified} onChange={() => setDoctorNotified(!doctorNotified)} />
            <span>Notify Doctor</span>
          </label>
        </div>
      </div>

      {/* ------------------ */}
      {/* ACTIONS */}
      {/* ------------------ */}
      <div className="bottom-action" style={{ display: "flex", gap: 8, flexDirection: "column" }}>
        <button className="save-file-btn" onClick={onSavePatient} disabled={saving}>
          {saving ? "Saving..." : "💾 SAVE PATIENT FILE"}
        </button>

        <button className="edit-action-btn" onClick={handleDownload}>
          📄 DOWNLOAD PDF
        </button>
      </div>
    </div>
  );
}