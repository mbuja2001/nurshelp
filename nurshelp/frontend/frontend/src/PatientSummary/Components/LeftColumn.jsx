// src/PatientSummary/Components/LeftColumn.jsx
import React, { useState } from "react";

export default function LeftColumn({ data = {}, setData }) {
  // local UI editing flags
  const [editingPatient, setEditingPatient] = useState(false);
  const [editing, setEditing] = useState({
    symptoms: false, duration: false, painLevel: false, history: false
  });

  // helper to update top-level patient fields (guard setData)
  const updateField = (field, value) => {
    if (typeof setData !== "function") return;
    setData(prev => ({ ...prev, [field]: value }));
  };

  // clear field with confirmation
  const handleClear = (field) => {
    if (!window.confirm("Are you sure you want to clear this field?")) return;
    updateField(field, "");
  };

  return (
    <div className="left-column-layout">
      {/* Consolidated Patient Information Box */}
      <div className="block">
        <div className="block-header">
          <h4 className="block-label">Patient Information</h4>
          <button className="edit-action-btn" onClick={() => setEditingPatient(!editingPatient)}>
            {editingPatient ? "✅" : "✎"}
          </button>
        </div>

        {editingPatient ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", padding: "12px" }}>
            <div className="form-group">
              <label style={{ fontSize: '0.9rem', fontWeight: '600', color: '#1f2937', marginBottom: '4px', display: 'block' }}>ID Number</label>
              <input
                className="edit-input"
                value={data?.id_number || ""}
                onChange={(e) => updateField("id_number", e.target.value)}
                placeholder="e.g. 9201015234089"
                style={{ padding: '8px', fontSize: '0.95rem' }}
              />
            </div>
            <div className="form-group">
              <label style={{ fontSize: '0.9rem', fontWeight: '600', color: '#1f2937', marginBottom: '4px', display: 'block' }}>First Name</label>
              <input
                className="edit-input"
                value={data?.name || ""}
                onChange={(e) => updateField("name", e.target.value)}
                placeholder="e.g. John"
                style={{ padding: '8px', fontSize: '0.95rem' }}
              />
            </div>
            <div className="form-group">
              <label style={{ fontSize: '0.9rem', fontWeight: '600', color: '#1f2937', marginBottom: '4px', display: 'block' }}>Surname</label>
              <input
                className="edit-input"
                value={data?.surname || ""}
                onChange={(e) => updateField("surname", e.target.value)}
                placeholder="e.g. Smith"
                style={{ padding: '8px', fontSize: '0.95rem' }}
              />
            </div>
            <div className="form-group">
              <label style={{ fontSize: '0.9rem', fontWeight: '600', color: '#1f2937', marginBottom: '4px', display: 'block' }}>Gender</label>
              <select
                className="edit-input"
                value={data?.gender || ""}
                onChange={(e) => updateField("gender", e.target.value)}
                style={{ padding: '8px', fontSize: '0.95rem' }}
              >
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div className="form-group">
              <label style={{ fontSize: '0.9rem', fontWeight: '600', color: '#1f2937', marginBottom: '4px', display: 'block' }}>Age</label>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <input
                  type="number"
                  className="edit-input"
                  value={data?.ageYears !== null && data?.ageYears !== undefined ? data.ageYears : ""}
                  onChange={(e) => updateField("ageYears", e.target.value ? Number(e.target.value) : null)}
                  placeholder="Years (optional)"
                  min="0"
                  style={{ padding: '8px', fontSize: '0.95rem', flex: 1 }}
                />
                <span style={{ color: '#6b7280', fontSize: '0.9rem' }}>y</span>
                <input
                  type="number"
                  className="edit-input"
                  value={data?.ageMonths !== null && data?.ageMonths !== undefined ? data.ageMonths : ""}
                  onChange={(e) => {
                    const val = e.target.value ? Number(e.target.value) : null;
                    // Allow months 0-11 for partial years, or any months for age calculation
                    if (val !== null && val < 0) {
                      alert("Months cannot be negative");
                      return;
                    }
                    // If years is 0 or empty and months >= 12, allow it (infant < 1 year)
                    if (val !== null && val > 12 && (!data?.ageYears || data.ageYears === 0)) {
                      alert("For infants less than 1 year, please enter total months (e.g., 5 for 5 months old)");
                      return;
                    }
                    // If years >= 1, months should be 0-11
                    if (val !== null && val > 11 && data?.ageYears && data.ageYears > 0) {
                      alert("Months must be between 0 and 11 for ages 1 year and older");
                      return;
                    }
                    updateField("ageMonths", val);
                  }}
                  placeholder="Months (0-12)"
                  min="0"
                  style={{ padding: '8px', fontSize: '0.95rem', flex: 1 }}
                />
                <span style={{ color: '#6b7280', fontSize: '0.9rem' }}>m</span>
              </div>
              <p style={{ fontSize: '0.8rem', color: '#6b7280', marginTop: '4px', margin: '4px 0 0 0' }}>
                💡 For infants &lt; 1 year: Leave years empty and enter total months (e.g., 5 for 5-month-old)
              </p>
            </div>
            <div className="form-group" style={{ gridColumn: "1 / -1" }}>
              <label style={{ fontSize: '0.9rem', fontWeight: '600', color: '#1f2937', marginBottom: '4px', display: 'block' }}>Height (cm)</label>
              <input
                type="number"
                className="edit-input"
                value={data?.height_cm || ""}
                onChange={(e) => updateField("height_cm", e.target.value ? Number(e.target.value) : null)}
                placeholder="e.g. 170"
                style={{ padding: '8px', fontSize: '0.95rem' }}
              />
            </div>
          </div>
        ) : (
          <div style={{ padding: "12px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", fontSize: "0.95rem" }}>
            <div><strong>ID:</strong> {data?.id_number || "---"}</div>
            <div><strong>Name:</strong> {data?.name || "---"}</div>
            <div><strong>Surname:</strong> {data?.surname || "---"}</div>
            <div><strong>Gender:</strong> {data?.gender || "---"}</div>
            <div>
              <strong>Age:</strong>{" "}
              {data?.ageYears !== null && data?.ageYears !== undefined ? (
                // Has years - show years and months if present
                <span>{data.ageYears}y{data?.ageMonths ? ` ${data.ageMonths}m` : ""}</span>
              ) : data?.ageMonths !== null && data?.ageMonths !== undefined ? (
                // No years, but has months - infant under 1 year
                <span>{data.ageMonths} month{data.ageMonths !== 1 ? "s" : ""}</span>
              ) : data?.age ? (
                // Fallback to legacy age field
                <span>{data.age}y</span>
              ) : (
                "---"
              )}
            </div>
            <div style={{ gridColumn: "1 / -1" }}><strong>Height:</strong> {data?.height_cm ? `${data.height_cm} cm` : "---"}</div>
          </div>
        )}
      </div>

      {[
        { key: "symptoms", label: "Symptoms" },
        { key: "duration", label: "Duration" },
        { key: "painLevel", label: "Pain Level" },
        { key: "history", label: "History" }
      ].map((sec) => (
        <div className="block" key={sec.key}>
          <div className="block-header">
            <h4 className="block-label">{sec.label}</h4>
            <div className="action-group">
              <button
                className="edit-action-btn"
                onClick={() => setEditing(prev => ({ ...prev, [sec.key]: !prev[sec.key] }))}
              >
                {editing[sec.key] ? "✅" : "✎"}
              </button>
              <button className="delete-btn" onClick={() => handleClear(sec.key)}>×</button>
            </div>
          </div>

          {editing[sec.key] ? (
            <textarea
              className="edit-textarea"
              value={data?.[sec.key] || ""}
              onChange={(e) => updateField(sec.key, e.target.value)}
            />
          ) : (
            <div className="content-area">{data?.[sec.key] || <em>Not provided</em>}</div>
          )}
        </div>
      ))}
    </div>
  );
}