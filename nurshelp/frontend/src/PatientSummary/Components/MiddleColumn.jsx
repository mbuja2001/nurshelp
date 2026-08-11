// src/PatientSummary/Components/MiddleColumn.jsx
import React, { useMemo } from "react";

export default function MiddleColumn({
  activeTab = 'manual',
  patientData = {},
  vitalsData = {},
  onTriageSubmit = () => {}
}) {
  const summary = useMemo(() => {
    const name = [patientData?.name, patientData?.surname].filter(Boolean).join(" ").trim() || "Patient";
    const complaint = patientData?.symptoms || "No presenting complaint captured";
    const history = patientData?.history || "No clinical history captured";
    const age = patientData?.ageYears != null
      ? `${patientData.ageYears}y${patientData.ageMonths ? ` ${patientData.ageMonths}m` : ""}`
      : patientData?.age ? `${patientData.age}y` : "Not stated";

    return {
      name,
      age,
      complaint,
      history,
      vitals: {
        temp: vitalsData?.temp ?? "--",
        bp: vitalsData?.bp ?? "--",
        hr: vitalsData?.hr ?? "--",
        o2: vitalsData?.o2 ?? "--",
        resp: vitalsData?.resp ?? "--",
        avpu: vitalsData?.avpu ?? "--"
      }
    };
  }, [patientData, vitalsData]);

  if (activeTab === 'manual') {
    return (
      <div className="middle-column-layout">
        <div className="block" style={{ gap: 16 }}>
          <div className="block-header">
            <div>
              <div className="eyebrow-label" style={{ marginBottom: 8 }}>Manual triage</div>
              <h4 className="block-label">Clinical confirmation report</h4>
            </div>
          </div>

          <div className="block" style={{ padding: 18, background: "rgba(37, 99, 235, 0.05)", borderColor: "rgba(37, 99, 235, 0.16)" }}>
            <div style={{ display: "grid", gap: 12 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Patient</div>
                  <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.name}</div>
                </div>
                <div>
                  <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Age</div>
                  <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.age}</div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Presenting complaint</div>
                <div style={{ fontWeight: 600, color: "#0f172a", marginTop: 4 }}>{summary.complaint}</div>
              </div>

              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Clinical history</div>
                <div style={{ fontWeight: 600, color: "#0f172a", marginTop: 4 }}>{summary.history}</div>
              </div>
            </div>
          </div>

          <div className="block" style={{ padding: 18 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Temp</div>
                <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.vitals.temp}°C</div>
              </div>
              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>BP</div>
                <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.vitals.bp}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>HR</div>
                <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.vitals.hr}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>O₂</div>
                <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.vitals.o2}%</div>
              </div>
              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Resp</div>
                <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.vitals.resp}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>AVPU</div>
                <div style={{ fontWeight: 700, color: "#0f172a", marginTop: 4 }}>{summary.vitals.avpu}</div>
              </div>
            </div>
          </div>

          <div className="block" style={{ padding: 18 }}>
            <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>Manual triage workflow</div>
            <p style={{ margin: "8px 0 0", lineHeight: 1.7, color: "#334155" }}>
              This workspace now functions as the manual confirmation report. The nurse review step is the dedicated place for documenting the clinical decision before locking the encounter.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (activeTab === 'ai') {
    return (
      <div className="middle-column-layout">
        <div className="block" style={{ textAlign: "center", padding: 40 }}>
          <div className="eyebrow-label">AI-assisted triage</div>
          <h4 className="block-label">AI output will render here</h4>
          <p style={{ color: "#64748b", lineHeight: 1.7 }}>
            When the server returns the AI triage output, it will be displayed in this column in the same report-driven style as the manual confirmation view.
          </p>
        </div>
      </div>
    );
  }

  return null;
}