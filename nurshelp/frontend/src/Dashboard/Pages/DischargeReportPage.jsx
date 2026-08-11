import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

export default function DischargeReportPage() {
  const { encounterId } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/discharge-report`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.message || 'Unable to load discharge report');
        setReport(data.dischargeReport || null);
      } catch (err) {
        setError(err.message || 'Unable to load discharge report');
      } finally {
        setLoading(false);
      }
    };

    if (encounterId) load();
  }, [encounterId]);

  if (loading) return <div style={{ padding: 24 }}>Loading discharge report...</div>;
  if (error) return <div style={{ padding: 24, color: '#b91c1c' }}>{error}</div>;
  if (!report) return <div style={{ padding: 24 }}>No discharge report is available yet.</div>;

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 24, background: '#fff', color: '#111827' }}>
      <h1 style={{ marginBottom: 8 }}>Discharge Report</h1>
      <p style={{ color: '#64748b', marginBottom: 24 }}>Generated from the discharged encounter workflow.</p>
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={{ padding: 16, border: '1px solid #e2e8f0', borderRadius: 12 }}>
          <div><strong>Report ID:</strong> {report.reportId}</div>
          <div><strong>Patient ID:</strong> {report.patientId}</div>
          <div><strong>Discharge Date:</strong> {report.dischargeDate ? new Date(report.dischargeDate).toLocaleString() : 'N/A'}</div>
          <div><strong>Status:</strong> {report.status}</div>
        </section>
        <section style={{ padding: 16, border: '1px solid #e2e8f0', borderRadius: 12 }}>
          <h2 style={{ marginTop: 0 }}>Primary diagnosis</h2>
          <p>{report.primaryDiagnosis || 'Not recorded'}</p>
        </section>
        <section style={{ padding: 16, border: '1px solid #e2e8f0', borderRadius: 12 }}>
          <h2 style={{ marginTop: 0 }}>Procedures</h2>
          {report.procedures?.length ? report.procedures.map((item, index) => <div key={index}>{item.procedureName}</div>) : <div>No procedures recorded</div>}
        </section>
        <section style={{ padding: 16, border: '1px solid #e2e8f0', borderRadius: 12 }}>
          <h2 style={{ marginTop: 0 }}>Medications</h2>
          {report.medications?.length ? report.medications.map((item, index) => <div key={index}>{item.name} · {item.dose}</div>) : <div>No medications recorded</div>}
        </section>
        <section style={{ padding: 16, border: '1px solid #e2e8f0', borderRadius: 12 }}>
          <h2 style={{ marginTop: 0 }}>Follow-up</h2>
          <p>{report.followUpInstructions || 'No follow-up instructions recorded'}</p>
        </section>
      </div>
    </div>
  );
}
