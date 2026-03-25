import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

export default function ReferenceCase() {
  const { encounterId, caseId } = useParams();
  const navigate = useNavigate();

  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch the full case details
  useEffect(() => {
    let mounted = true;

    const fetchCaseDetails = async () => {
      setLoading(true);
      setError(null);
      try {
        const token = localStorage.getItem("token");
        const headers = {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache"
        };

        // Query endpoint to retrieve full case by ID
        const cacheBreaker = `&t=${Date.now()}`;
        const res = await fetch(
          `${BACKEND_URL}/api/cases/${caseId}?encounterId=${encounterId}${cacheBreaker}`,
          { headers, cache: "no-store" }
        );

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(
            errBody.message || `Failed to load case (status ${res.status})`
          );
        }

        const data = await res.json();
        const caseObj = data?.case ?? data;
        if (!mounted) return;

        if (!caseObj) throw new Error("Case response malformed");
        setCaseData(caseObj);
      } catch (err) {
        console.error("Fetch case error:", err);
        if (mounted) setError(err.message || "Failed to load case details");
      } finally {
        if (mounted) setLoading(false);
      }
    };

    if (caseId && encounterId) {
      fetchCaseDetails();
    } else {
      setError("Missing case or encounter ID");
      setLoading(false);
    }

    return () => {
      mounted = false;
    };
  }, [caseId, encounterId]);

  const handleBack = () => {
    navigate(`/summary/${encounterId}`);
  };

  if (loading) {
    return (
      <div style={{ padding: "20px", textAlign: "center" }}>
        <div>Loading case details...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: "20px" }}>
        <button
          onClick={handleBack}
          style={{
            padding: "8px 16px",
            marginBottom: "16px",
            fontSize: 14,
            fontWeight: 600,
            border: "1px solid #cbd5e1",
            borderRadius: 6,
            background: "#f1f5f9",
            color: "#0f172a",
            cursor: "pointer",
            transition: "all 0.2s"
          }}
        >
          ← Back to Report
        </button>
        <div
          style={{
            padding: 16,
            borderRadius: 8,
            background: "#fee2e2",
            color: "#991b1b",
            border: "1px solid #fecaca"
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  if (!caseData) {
    return (
      <div style={{ padding: "20px" }}>
        <button
          onClick={handleBack}
          style={{
            padding: "8px 16px",
            marginBottom: "16px",
            fontSize: 14,
            fontWeight: 600,
            border: "1px solid #cbd5e1",
            borderRadius: 6,
            background: "#f1f5f9",
            color: "#0f172a",
            cursor: "pointer",
            transition: "all 0.2s"
          }}
        >
          ← Back to Report
        </button>
        <div style={{ color: "#64748b" }}>No case data available</div>
      </div>
    );
  }

  // Determine if this case was referenced by MedGemma
  const referenced = caseData.medgemma_referenced === true;

  return (
    <div style={{ padding: "20px", maxWidth: 900, margin: "0 auto" }}>
      {/* Back button */}
      <button
        onClick={handleBack}
        style={{
          padding: "8px 16px",
          marginBottom: "20px",
          fontSize: 14,
          fontWeight: 600,
          border: "1px solid #cbd5e1",
          borderRadius: 6,
          background: "#f1f5f9",
          color: "#0f172a",
          cursor: "pointer",
          transition: "all 0.2s"
        }}
        onMouseOver={(e) => {
          e.target.style.background = "#e2e8f0";
          e.target.style.borderColor = "#94a3b8";
        }}
        onMouseOut={(e) => {
          e.target.style.background = "#f1f5f9";
          e.target.style.borderColor = "#cbd5e1";
        }}
      >
        ← Back to Report
      </button>

      {/* Case header */}
      <div
        style={{
          padding: 16,
          marginBottom: 20,
          borderRadius: 8,
          border: referenced ? "2px solid #10b981" : "1px solid #e2e8f0",
          background: referenced ? "#f0fdf4" : "#f8fafc"
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "start",
            gap: 16
          }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 20, fontWeight: 700, color: "#0f172a", marginBottom: 8 }}>
              {caseData.diagnosis || "Unknown Diagnosis"}
            </div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
              {referenced && (
                <span
                  style={{
                    display: "inline-block",
                    padding: "4px 10px",
                    borderRadius: 6,
                    background: "#dcfce7",
                    color: "#166534",
                    fontSize: 12,
                    fontWeight: 600,
                    border: "1px solid #86efac"
                  }}
                >
                  ✓ Referenced
                </span>
              )}
              {caseData.case_id && (
                <div style={{ fontSize: 13, color: "#64748b" }}>
                  <strong>Case ID:</strong> {caseData.case_id}
                </div>
              )}
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                gap: 12
              }}
            >
              {caseData.similarity_score !== undefined && (
                <div>
                  <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                    Similarity
                  </div>
                  <div
                    style={{
                      fontSize: 16,
                      fontWeight: 700,
                      color: "#0f172a"
                    }}
                  >
                    {Math.round((caseData.similarity_score ?? 0) * 100)}%
                  </div>
                </div>
              )}
              {caseData.source && (
                <div>
                  <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                    Source
                  </div>
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: "#0f172a"
                    }}
                  >
                    {caseData.source}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Case details sections */}
      {caseData.patient_details && (
        <div
          style={{
            padding: 16,
            marginBottom: 16,
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            background: "#f8fafc"
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 12 }}>
            Patient Details
          </div>
          <div style={{ fontSize: 13, color: "#1e293b", lineHeight: 1.6 }}>
            {typeof caseData.patient_details === "string" ? (
              <div style={{ whiteSpace: "pre-wrap" }}>{caseData.patient_details}</div>
            ) : (
              <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                {JSON.stringify(caseData.patient_details, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {caseData.clinical_presentation && (
        <div
          style={{
            padding: 16,
            marginBottom: 16,
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            background: "#f8fafc"
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 12 }}>
            Clinical Presentation
          </div>
          <div style={{ fontSize: 13, color: "#1e293b", lineHeight: 1.6 }}>
            {typeof caseData.clinical_presentation === "string" ? (
              <div style={{ whiteSpace: "pre-wrap" }}>{caseData.clinical_presentation}</div>
            ) : (
              <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                {JSON.stringify(caseData.clinical_presentation, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {caseData.clinical_reasoning && (
        <div
          style={{
            padding: 16,
            marginBottom: 16,
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            background: "#f8fafc"
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 12 }}>
            Clinical Reasoning
          </div>
          <div style={{ fontSize: 13, color: "#1e293b", lineHeight: 1.6 }}>
            {typeof caseData.clinical_reasoning === "string" ? (
              <div style={{ whiteSpace: "pre-wrap" }}>{caseData.clinical_reasoning}</div>
            ) : (
              <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                {JSON.stringify(caseData.clinical_reasoning, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {caseData.differential_diagnosis && (
        <div
          style={{
            padding: 16,
            marginBottom: 16,
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            background: "#f8fafc"
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 12 }}>
            Differential Diagnosis
          </div>
          <div style={{ fontSize: 13, color: "#1e293b", lineHeight: 1.6 }}>
            {typeof caseData.differential_diagnosis === "string" ? (
              <div style={{ whiteSpace: "pre-wrap" }}>{caseData.differential_diagnosis}</div>
            ) : Array.isArray(caseData.differential_diagnosis) ? (
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {caseData.differential_diagnosis.map((item, i) => (
                  <li key={i}>{typeof item === "string" ? item : JSON.stringify(item)}</li>
                ))}
              </ul>
            ) : (
              <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                {JSON.stringify(caseData.differential_diagnosis, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {caseData.management_plan && (
        <div
          style={{
            padding: 16,
            marginBottom: 16,
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            background: "#f8fafc"
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 12 }}>
            Management Plan
          </div>
          <div style={{ fontSize: 13, color: "#1e293b", lineHeight: 1.6 }}>
            {typeof caseData.management_plan === "string" ? (
              <div style={{ whiteSpace: "pre-wrap" }}>{caseData.management_plan}</div>
            ) : (
              <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                {JSON.stringify(caseData.management_plan, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {caseData.key_findings && (
        <div
          style={{
            padding: 16,
            marginBottom: 16,
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            background: "#f8fafc"
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 12 }}>
            Key Findings
          </div>
          <div style={{ fontSize: 13, color: "#1e293b", lineHeight: 1.6 }}>
            {typeof caseData.key_findings === "string" ? (
              <div style={{ whiteSpace: "pre-wrap" }}>{caseData.key_findings}</div>
            ) : Array.isArray(caseData.key_findings) ? (
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {caseData.key_findings.map((item, i) => (
                  <li key={i}>{typeof item === "string" ? item : JSON.stringify(item)}</li>
                ))}
              </ul>
            ) : (
              <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                {JSON.stringify(caseData.key_findings, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {/* Raw data (expandable) */}
      {Object.keys(caseData).length > 0 && (
        <details style={{ marginTop: 20 }}>
          <summary
            style={{
              padding: "12px 16px",
              borderRadius: 8,
              background: "#f1f5f9",
              border: "1px solid #cbd5e1",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: 13,
              color: "#0f172a"
            }}
          >
            Raw Case Data
          </summary>
          <pre
            style={{
              marginTop: 12,
              padding: 16,
              borderRadius: 8,
              background: "#f8fafc",
              border: "1px solid #e2e8f0",
              overflow: "auto",
              fontSize: 12,
              color: "#475569",
              fontFamily: "monospace",
              whiteSpace: "pre-wrap",
              wordWrap: "break-word"
            }}
          >
            {JSON.stringify(caseData, null, 2)}
          </pre>
        </details>
      )}

      {/* Back button at bottom */}
      <button
        onClick={handleBack}
        style={{
          marginTop: 20,
          padding: "8px 16px",
          fontSize: 14,
          fontWeight: 600,
          border: "1px solid #cbd5e1",
          borderRadius: 6,
          background: "#f1f5f9",
          color: "#0f172a",
          cursor: "pointer",
          transition: "all 0.2s"
        }}
        onMouseOver={(e) => {
          e.target.style.background = "#e2e8f0";
          e.target.style.borderColor = "#94a3b8";
        }}
        onMouseOut={(e) => {
          e.target.style.background = "#f1f5f9";
          e.target.style.borderColor = "#cbd5e1";
        }}
      >
        ← Back to Report
      </button>
    </div>
  );
}
