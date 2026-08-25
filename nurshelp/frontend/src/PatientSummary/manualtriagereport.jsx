import React, { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ManualTriageForm from "./Components/ManualTriageForm";
import { getSatsAcuityFromTews } from "./utils/triageHelpers";

const getEmptyPatientState = () => ({
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
  history: null
});

const parseSafeJson = (value) => {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch (err) {
    console.warn("Failed to parse persisted state", err);
    return null;
  }
};

export default function ManualTriageReport() {
  const navigate = useNavigate();
  const location = useLocation();
  const [formReady, setFormReady] = useState(false);
  const [submitRequestCount, setSubmitRequestCount] = useState(0);

  const reportData = useMemo(() => {
    const state = location.state || {};
    const storedState = parseSafeJson(localStorage.getItem("manual_triage_report_state")) || {};
    const fallbackPatient = parseSafeJson(localStorage.getItem("ps_patient")) || getEmptyPatientState();
    const fallbackVitals = parseSafeJson(localStorage.getItem("ps_vitals")) || {};

    const patient = state.patientData || storedState.patientData || fallbackPatient || getEmptyPatientState();
    const vitals = state.vitalsData || storedState.vitalsData || fallbackVitals || {};
    const ehrSummary = state.ehrSummary || storedState.ehrSummary || null;
    const selectedMrn = state.selectedMrn || storedState.selectedMrn || null;

    return { patient, vitals, ehrSummary, selectedMrn };
  }, [location.state]);

  const patientName = [reportData.patient?.name, reportData.patient?.surname].filter(Boolean).join(" ").trim() || "Patient";
  const age = reportData.patient?.ageYears != null
    ? `${reportData.patient.ageYears}y${reportData.patient.ageMonths ? ` ${reportData.patient.ageMonths}m` : ""}`
    : reportData.patient?.age ? `${reportData.patient.age}y` : "Not stated";
  const [analysisData, setAnalysisData] = React.useState({
    pathophysiology: "",
    clinicalSummary: "",
    clinicalDiscriminators: "",
    differentialDiagnoses: "",
    immediateActions: "",
    diagnosticWorkup: "",
    dispositionRationale: ""
  });
  const [tewsData, setTewsData] = React.useState({
    profile: "adult",
    height: "",
    total: 0,
    priority: "green",
    components: {}
  });
  const displayPriority = tewsData.priority || getSatsAcuityFromTews(tewsData.total) || "green";
  const [assignmentData, setAssignmentData] = React.useState({ nurse: null, physician: null, specialist: null, bed: null });
  const [formPayload, setFormPayload] = React.useState(null);
  const [recommendedDepartment, setRecommendedDepartment] = React.useState("ED");
  const [acuityScore, setAcuityScore] = React.useState("50");

  React.useEffect(() => {
    const physician = assignmentData.physician;
    if (!physician) {
      setRecommendedDepartment("Not assigned");
      return;
    }

    setRecommendedDepartment(physician.specialty || "Not assigned");
  }, [assignmentData.physician]);

  const handleManualSubmit = (result) => {
    localStorage.removeItem("manual_triage_report_state");
    if (result?.triageRequestId) {
      navigate(`/handover/manual/${result.triageRequestId}`, { state: { encounter: result } });
    } else {
      navigate("/dashboard");
    }
  };

  const requestManualSubmit = () => {
    setSubmitRequestCount((current) => current + 1);
  };

  return (
    <>
      <style>{`
        .manual-triage-report-page {
          min-height: 100vh;
          background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
          padding: 24px;
          box-sizing: border-box;
        }

        .manual-triage-report-shell {
          width: min(100%, 1440px);
          margin: 0 auto;
          display: grid;
          gap: 18px;
        }

        .manual-triage-report-header {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(220px, 330px);
          align-items: start;
          gap: 16px;
          margin-bottom: 0;
        }

        .manual-triage-report-header-copy {
          min-width: 0;
        }

        .manual-triage-report-header-copy h1 {
          line-height: 1.08;
        }

        .manual-triage-report-header-copy p {
          line-height: 1.45;
          margin-top: 8px;
        }

        .manual-triage-report-actions {
          display: grid;
          gap: 10px;
          align-items: end;
          justify-items: end;
          width: 100%;
          min-width: 0;
          max-width: 330px;
          justify-self: end;
        }

        .manual-triage-report-main {
          display: grid;
          grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.9fr);
          gap: 20px;
          align-items: start;
        }

        .manual-triage-report-main > div {
          min-width: 0;
        }

        .manual-triage-summary-column {
          display: grid;
          gap: 20px;
        }

        @media (max-width: 1100px) {
          .manual-triage-report-main {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 900px) {
          .manual-triage-report-header {
            grid-template-columns: 1fr;
            align-items: stretch;
          }

          .manual-triage-report-actions {
            width: 100%;
            max-width: none;
            justify-items: stretch;
            justify-self: stretch;
          }
        }

        @media (max-width: 640px) {
          .manual-triage-report-page {
            padding: 12px;
          }

          .manual-triage-report-shell {
            gap: 14px;
          }

          .manual-triage-report-header {
            gap: 12px;
            margin-bottom: 0;
            align-items: stretch;
          }

          .manual-triage-report-header-copy {
            flex-basis: 100%;
          }

          .manual-triage-report-header h1 {
            font-size: 26px;
            line-height: 1.12;
            margin-top: 2px;
          }

          .manual-triage-report-header-copy p {
            font-size: 13px;
            line-height: 1.4;
            margin-top: 6px;
          }

          .manual-triage-report-actions {
            width: 100%;
            gap: 8px;
          }

          .manual-triage-report-actions button {
            width: 100%;
          }
        }
      `}</style>

      <div className="manual-triage-report-page">
        <div className="manual-triage-report-shell">
          <div className="manual-triage-report-header">
            <div className="manual-triage-report-header-copy">
              <div style={{ fontSize: 12, letterSpacing: "0.2em", textTransform: "uppercase", color: "#64748b", fontWeight: 700 }}>Manual triage</div>
              <h1 style={{ margin: "4px 0 0", fontSize: 32, color: "#0f172a" }}>Dedicated nurse triage workflow</h1>
              <p style={{ margin: "8px 0 0", color: "#475569", maxWidth: 760 }}>
                Review the patient intake data and complete the formal manual triage assessment. When submitted, the patient is created as a manual triage encounter and routed to clinician handover.
              </p>
            </div>
            <div className="manual-triage-report-actions">
              <button onClick={() => navigate(-1)} style={secondaryButton}>Back to intake</button>
              <button
                onClick={requestManualSubmit}
                disabled={!formReady}
                style={{
                  ...submitButton,
                  ...(formReady ? {} : disabledButton),
                  width: '100%'
                }}
              >
                {formReady ? 'Submit manual report' : 'Complete required fields to submit'}
              </button>
            </div>
          </div>

          <div className="manual-triage-report-main">
            <div style={cardStyle}>
              <ManualTriageForm
                patientData={reportData.patient}
                vitalsData={reportData.vitals}
                ehrSummary={reportData.ehrSummary}
                selectedMrn={reportData.selectedMrn}
                onSubmit={handleManualSubmit}
                onPayloadChange={setFormPayload}
                onSubmitPayload={(payload) => {
                  setFormPayload(payload);
                  localStorage.setItem('manual_triage_report_payload', JSON.stringify(payload));
                }}
                onAnalysisChange={(field, value) => setAnalysisData((current) => ({ ...current, [field]: value }))}
                onTewsChange={setTewsData}
                onAssignmentChange={setAssignmentData}
                onDepartmentChange={setRecommendedDepartment}
                onAcuityChange={setAcuityScore}
                onFormReadyChange={setFormReady}
                submitRequest={submitRequestCount}
              />
            </div>

            <div className="manual-triage-summary-column">
              <div style={cardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 12 }}>
                  <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b", fontWeight: 700 }}>Patient snapshot</div>
                  <div style={{ background: triageColor[tewsData.priority] || "#f8fafc", border: `1px solid ${triageBorder[tewsData.priority] || "#cbd5e1"}`, borderRadius: 10, padding: "8px 10px", textAlign: "right", minWidth: 92 }}>
                    <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700 }}>SATS / TEWS</div>
                    <div style={{ marginTop: 3, color: triageText[displayPriority] || "#0f172a", fontSize: 14, fontWeight: 800 }}>{String(displayPriority).toUpperCase()} · {tewsData.total}</div>
                  </div>
                </div>
                <div style={{ display: "grid", gap: 14 }}>
                  <InfoBlock label="Name" value={patientName} />
                  <InfoBlock label="Age" value={age} />
                  <InfoBlock label="Gender" value={reportData.patient?.gender || "Not stated"} />
                  <InfoBlock label="MRN" value={reportData.patient?.id_number || reportData.selectedMrn || "Not stated"} />
                  <InfoBlock label="Complaint" value={reportData.patient?.symptoms || "Not stated"} />
                  <InfoBlock label="History" value={reportData.patient?.history || "Not stated"} />
                </div>
                <div style={{ marginTop: 18, display: "grid", gap: 10 }}>
                  <div style={triageHeadingStyle}>SATS triage category</div>
                  <div style={{ background: triageColor[displayPriority] || "#f8fafc", border: `1px solid ${triageBorder[displayPriority] || "#cbd5e1"}`, borderRadius: 12, padding: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                      <div>
                        <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>Priority</div>
                        <div style={{ marginTop: 4, fontSize: 20, fontWeight: 800, color: triageText[displayPriority] || "#0f172a" }}>{String(displayPriority).toUpperCase()}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>TEWS</div>
                        <div style={{ marginTop: 4, fontSize: 20, fontWeight: 800, color: "#0f172a" }}>{tewsData.total} <span style={{ fontSize: 11 }}>points</span></div>
                      </div>
                    </div>
                    <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid rgba(100, 116, 139, 0.2)", color: "#475569", fontSize: 12 }}>
                      <strong>Acuity:</strong> {acuityScore}
                    </div>
                    <div style={{ marginTop: 10, color: "#475569", fontSize: 12 }}>Profile: {formatTewsProfile(tewsData.profile)}{tewsData.height ? ` • ${tewsData.height} cm` : ""}</div>
                  </div>
                  <AnalysisBlock label="Pathophysiology" value={analysisData.pathophysiology} />
                  <AnalysisBlock label="Clinical Summary" value={analysisData.clinicalSummary} />
                  <AnalysisBlock label="Clinical Discriminators" value={analysisData.clinicalDiscriminators} />
                  <AnalysisBlock label="Differential Diagnoses" value={analysisData.differentialDiagnoses} />
                  <AnalysisBlock label="Immediate Actions" value={analysisData.immediateActions} />
                  <AnalysisBlock label="Diagnostic Workup" value={analysisData.diagnosticWorkup} />
                  <AnalysisBlock label="Disposition Rationale" value={analysisData.dispositionRationale} />
                  <div style={assignmentHeadingStyle}>Assigned clinicians & wards</div>
                  <AssignmentBlock label="Assigned Nurse" value={assignmentData.nurse?.name || assignmentData.nurse?.id} />
                  <AssignmentBlock label="Assigned Physician" value={assignmentData.physician?.name || assignmentData.physician?.id} />
                  <AssignmentBlock label="Assigned Ward" value={assignmentData.bed?.wardName || assignmentData.bed?.ward || assignmentData.bed?.department} />
                  <AssignmentBlock label="Assigned Bed" value={assignmentData.bed?.bedId || assignmentData.bed?.id || assignmentData.bed?.bedNumber} />
                  <AssignmentBlock label="Recommended Department" value={recommendedDepartment} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function InfoBlock({ label, value }) {
  return (
    <div style={{ background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 12, padding: 12 }}>
      <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", color: "#64748b" }}>{label}</div>
      <div style={{ marginTop: 6, fontWeight: 700, color: "#0f172a", lineHeight: 1.4 }}>{value}</div>
    </div>
  );
}

function AnalysisBlock({ label, value }) {
  return (
    <div style={{ background: "#f8fafc", border: "1px solid #dbeafe", borderRadius: 12, padding: 12 }}>
      <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.1em", color: "#64748b", fontWeight: 700 }}>{label}</div>
      <div style={{ marginTop: 6, color: value ? "#1e293b" : "#94a3b8", lineHeight: 1.55, whiteSpace: "pre-wrap", fontSize: 13 }}>
        {value || "Not documented"}
      </div>
    </div>
  );
}

function AssignmentBlock({ label, value }) {
  return (
    <div style={{ background: "#f8fafc", border: "1px solid #dbeafe", borderRadius: 12, padding: 12 }}>
      <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.1em", color: "#64748b", fontWeight: 700 }}>{label}</div>
      <div style={{ marginTop: 6, color: value ? "#1e293b" : "#94a3b8", lineHeight: 1.55, fontSize: 13 }}>{value || "Not assigned"}</div>
    </div>
  );
}

const analysisHeadingStyle = {
  color: "#0c4a6e",
  fontSize: 13,
  fontWeight: 800,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  padding: "4px 2px"
};

const triageHeadingStyle = {
  color: "#0c4a6e",
  fontSize: 13,
  fontWeight: 800,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  padding: "4px 2px"
};

const assignmentHeadingStyle = {
  ...analysisHeadingStyle,
  marginTop: 8
};

const triageColor = { red: "#fef2f2", orange: "#fff7ed", yellow: "#fefce8", green: "#f0fdf4" };
const triageBorder = { red: "#fca5a5", orange: "#fdba74", yellow: "#fde68a", green: "#86efac" };
const triageText = { red: "#b91c1c", orange: "#c2410c", yellow: "#a16207", green: "#15803d" };
function formatTewsProfile(profile) {
  return { adult: "Adult", "child-younger": "Child 50-95 cm", "child-older": "Child 96-150 cm" }[profile] || "Not selected";
}

const cardStyle = {
  background: "white",
  borderRadius: 20,
  padding: 24,
  boxShadow: "0 18px 40px rgba(15, 23, 42, 0.08)",
  border: "1px solid rgba(148, 163, 184, 0.22)"
};

const secondaryButton = {
  border: "1px solid #cbd5e1",
  background: "white",
  color: "#0f172a",
  borderRadius: 999,
  padding: "10px 16px",
  fontWeight: 700,
  cursor: "pointer"
};

const submitButton = {
  border: "none",
  backgroundColor: "#2563eb",
  color: "white",
  borderRadius: 999,
  padding: "10px 16px",
  fontWeight: 700,
  cursor: "pointer",
  transition: "all 0.2s ease",
  minWidth: 220
};

const disabledButton = {
  backgroundColor: "#cbd5e1",
  color: "#475569",
  cursor: "not-allowed"
};
