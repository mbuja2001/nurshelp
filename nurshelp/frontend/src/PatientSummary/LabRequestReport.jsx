import React, { useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

const LabRequestReport = () => {
  const navigate = useNavigate();
  const { encounterId } = useParams();
  const { state } = useLocation();
  const [downloading, setDownloading] = useState(false);
  const request = state?.request || {};
  const form = state?.form || {};
  const context = state?.hospitalContext || {};
  const specialty = String(state?.specialty || request.specialty || "RADIOLOGY").toUpperCase();
  const title = specialty === "PATHOLOGY" ? "Pathology Request" : "Radiology Request";
  const requestedTest = request.lab_type || form.lab_type || "Not recorded";
  const clinician = request.requested_by || form.clinicianName || "Clinician";
  const recordedAt = request.createdAt || request.requested_at || new Date().toISOString();

  const formatTimestamp = (value) => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "Timestamp not recorded" : date.toLocaleString();
  };

  const downloadPdf = async () => {
    const element = document.getElementById("official-lab-report");
    if (!element) return;
    setDownloading(true);
    try {
      const canvas = await html2canvas(element, { scale: 2, backgroundColor: "#fff", useCORS: true });
      const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
      const margin = 10;
      const width = pdf.internal.pageSize.getWidth() - margin * 2;
      const height = (canvas.height * width) / canvas.width;
      pdf.addImage(canvas.toDataURL("image/png"), "PNG", margin, margin, width, Math.min(height, 277));
      pdf.save(`Official_${specialty}_Request_${String(context.patientName || "Patient").replace(/\s+/g, "_")}.pdf`);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="lab-report-page">
      <style>{`.lab-report-page{min-height:100vh;background:#f4f7fb;padding:32px;color:#10233f}.lab-report-shell{max-width:980px;margin:0 auto}.lab-report-actions{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:18px}.lab-report-button{border:1px solid #9db4cd;background:#fff;color:#0b3b70;padding:11px 16px;font-weight:800;cursor:pointer}.lab-report-button.primary{border-color:#0b3b70;background:#0b3b70;color:#fff}.lab-report-paper{background:#fff;border:1px solid #d9e2ed;box-shadow:0 18px 40px #203b5b14;padding:42px}.lab-report-kicker{color:#2563a6;font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase}.lab-report-paper h1{margin:8px 0 6px;font-size:34px;color:#10233f}.lab-report-subtitle{margin:0;color:#63748a}.lab-report-rule{border:0;border-top:2px solid #0b3b70;margin:26px 0}.lab-report-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;border-bottom:1px solid #d9e2ed;padding-bottom:22px}.lab-report-field{min-width:0}.lab-report-field label{display:block;color:#718198;font-size:10px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;margin-bottom:5px}.lab-report-field div{font-size:15px;overflow-wrap:anywhere}.lab-report-section{border-bottom:1px solid #d9e2ed;padding:22px 0}.lab-report-section h2{font-size:14px;text-transform:uppercase;letter-spacing:.1em;color:#2563a6;margin:0 0 12px}.lab-report-section p{margin:6px 0;line-height:1.55}.lab-report-signature{margin-top:34px;display:grid;grid-template-columns:1fr 1fr;gap:30px}.lab-report-signature div{border-top:1px solid #10233f;padding-top:8px;font-size:13px}@media(max-width:700px){.lab-report-page{padding:16px}.lab-report-paper{padding:24px}.lab-report-grid,.lab-report-signature{grid-template-columns:1fr}.lab-report-actions{align-items:stretch;flex-direction:column}.lab-report-actions button{width:100%}}`}</style>
      <div className="lab-report-shell">
        <div className="lab-report-actions">
          <button className="lab-report-button" type="button" onClick={() => navigate(`/request-lab/${encounterId}/${specialty.toLowerCase()}`)}>Back to request</button>
          <button className="lab-report-button primary" type="button" onClick={downloadPdf} disabled={downloading}>{downloading ? "Preparing PDF..." : "Download official PDF"}</button>
        </div>
        <article id="official-lab-report" className="lab-report-paper">
          <div className="lab-report-kicker">Hospital clinical services · official request</div>
          <h1>{title}</h1>
          <p className="lab-report-subtitle">A timestamped specialist request recorded for the patient EHR.</p>
          <hr className="lab-report-rule" />
          <div className="lab-report-grid">
            <div className="lab-report-field"><label>Patient full name</label><div>{context.patientName || "Not recorded"}</div></div>
            <div className="lab-report-field"><label>MRN</label><div>{context.mrn || "Not recorded"}</div></div>
            <div className="lab-report-field"><label>Date of birth</label><div>{context.dob || "Not recorded"}</div></div>
            <div className="lab-report-field"><label>Location</label><div>{context.location || "Not recorded"}</div></div>
            <div className="lab-report-field"><label>Request timestamp</label><div>{formatTimestamp(recordedAt)}</div></div>
            <div className="lab-report-field"><label>Priority</label><div>{request.priority || form.priority || "MEDIUM"}</div></div>
          </div>
          <section className="lab-report-section">
            <h2>Clinical request</h2>
            <p><strong>Requested {specialty === "PATHOLOGY" ? "pathology test" : "imaging study"}:</strong> {requestedTest}</p>
            <p><strong>Clinical question:</strong> {request.description || form.description || "Not recorded"}</p>
            {specialty === "RADIOLOGY" ? <>
              <p><strong>Modality:</strong> {form.modality || "Not recorded"}</p>
              <p><strong>Body part / protocol:</strong> {form.bodyPart || "Not recorded"}</p>
              <p><strong>Laterality:</strong> {form.anatomicalTarget || "Not recorded"}</p>
              <p><strong>Safety screening:</strong> Pregnancy {form.patientPregnancyStatus || "Not recorded"}; allergies {form.allergies || "None recorded"}; eGFR {form.eGFR || "Not recorded"}; creatinine {form.serumCreatinine || "Not recorded"}; implants {form.implants || "None recorded"}</p>
            </> : <p><strong>Specimen / sample type:</strong> {form.specimenType || form.bodyPart || "Not recorded"}</p>}
          </section>
          <section className="lab-report-section">
            <h2>Clinical provenance</h2>
            <p>This request was entered from the clinician workflow and is available for specialist acknowledgement and EHR chronology.</p>
            <div className="lab-report-signature"><div>Requesting clinician<br /><strong>{clinician}</strong></div><div>Clinician signature<br /><strong>{form.clinicianSignature || clinician}</strong></div></div>
          </section>
        </article>
      </div>
    </div>
  );
};

export default LabRequestReport;
