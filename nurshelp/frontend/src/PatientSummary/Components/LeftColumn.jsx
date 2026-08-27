// src/PatientSummary/Components/LeftColumn.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FaSearch, FaStethoscope, FaTimes } from "react-icons/fa";

const criticalHistoryFields = [
  ['allergies', 'Allergies'],
  ['adverseReactions', 'Severe adverse reactions'],
  ['medications', 'High-risk or current medications'],
  ['conditions', 'Major conditions'],
  ['implantsOrProcedures', 'Critical implants or procedures']
];

const asList = (value) => Array.isArray(value) ? value : (value ? String(value).split(',').map((item) => item.trim()).filter(Boolean) : []);

export default function LeftColumn({
  data = {},
  setData,
  searchQuery,
  setSearchQuery,
  searchResults = [],
  searchLoading = false,
  searchError = null,
  selectedMrn = null,
  handleSearchSubmit,
  handlePatientSelect,
  loadingEHR = false,
  clearPatientSelection,
  cancelSearch,
  cancelEHRLoad
}) {
  const navigate = useNavigate();
  // local UI editing flags
  const [editingPatient, setEditingPatient] = useState(false);
  const [editing, setEditing] = useState({
    symptoms: false, duration: false, painLevel: false, history: false
  });
  const [searchModalOpen, setSearchModalOpen] = useState(false);

  // helper to update top-level patient fields (guard setData)
  const updateField = (field, value) => {
    if (typeof setData !== "function") return;
    setData(prev => ({ ...prev, [field]: value }));
  };

  const updateCriticalHistory = (field, value) => {
    if (typeof setData !== "function") return;
    setData((prev) => ({
      ...prev,
      criticalHistory: { ...(prev.criticalHistory || {}), [field]: asList(value) }
    }));
  };

  // clear field with confirmation
  const handleClear = (field) => {
    if (!window.confirm("Are you sure you want to clear this field?")) return;
    updateField(field, "");
  };

  const openSearchModal = () => {
    setSearchModalOpen(true);
  };

  const closeSearchModal = () => {
    setSearchModalOpen(false);
  };

  const submitSearch = async () => {
    if (typeof handleSearchSubmit === "function") {
      await handleSearchSubmit();
    }
  };

  const selectEncounter = async (encounter) => {
    if (typeof handlePatientSelect === "function") {
      await handlePatientSelect(encounter);
    }
    setSearchModalOpen(false);
  };

  return (
    <div className="left-column-layout">
      <div
        className="triage-shell-card"
        onClick={openSearchModal}
        style={{ cursor: "pointer" }}
        title="Open patient records search"
      >
        <div className="triage-shell-top">
          <div className="triage-shell-title">
            <div className="triage-shell-icon"><FaStethoscope /></div>
            <div>
              <div className="triage-shell-eyebrow">Load Patient History</div>
              <div className="triage-shell-heading">Electronic Health Database Search</div>
            </div>
          </div>
          <div className="triage-shell-actions">
            <button className="triage-shell-icon-btn" type="button" onClick={(event) => { event.stopPropagation(); openSearchModal(); }} title="Search EHR">
              <FaSearch />
            </button>
            {selectedMrn && (
              <>
                <span className="triage-shell-status-chip">Loaded • {selectedMrn}</span>
                <button
                  
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    clearPatientSelection?.();
                    openSearchModal();
                  }}
                >
                  Choose another patient
                </button>
              </>
            )}
          </div>
        </div>

        <div className="triage-shell-prompt">
          Search by ID number, MRN, name, or surname to preload the patient's clinical story before moving to the dedicated manual triage report.
        </div>

        {searchModalOpen && (
          <div className="triage-search-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <div className="triage-search-modal-card">
              <div className="triage-search-modal-header">
                <div>
                  <div className="triage-shell-eyebrow">Quick EHR lookup</div>
                  <h4>Search hospital records</h4>
                </div>
                <button className="triage-shell-icon-btn" type="button" onClick={closeSearchModal} title="Close search">
                  <FaTimes />
                </button>
              </div>

              <div className="triage-search-input-row">
                <input
                  value={searchQuery || ""}
                  onChange={(e) => setSearchQuery?.(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") submitSearch(); }}
                  placeholder="Type MRN, first name, or surname"
                  aria-label="Search patient records"
                />
                {searchLoading ? (
                  <button type="button" className="triage-search-button" onClick={cancelSearch}>
                    Cancel search
                  </button>
                ) : (
                  <button type="button" className="triage-search-button" onClick={submitSearch}>
                    Search
                  </button>
                )}
              </div>

              {searchError && <div className="triage-search-feedback error">{searchError}</div>}
              {loadingEHR && (
                <div className="triage-search-feedback notice">
                  Loading EHR details…
                  <button type="button" className="triage-search-button" onClick={cancelEHRLoad}>Cancel load</button>
                </div>
              )}

              {searchResults.length > 0 && (
                <div className="triage-search-results">
                  {searchResults.map((encounter) => {
                    const patient = encounter.patient || {};
                    const mrn = patient.id_number || "Encounter record";
                    const status = encounter.status || "Unknown status";
                    return (
                      <button key={`${mrn}-${encounter._id || encounter.id}`} type="button" className="triage-search-result" onClick={() => selectEncounter(encounter)}>
                        <div>
                          <div className="triage-search-result-title">{patient.name || "Unknown"} {patient.surname || ""}</div>
                          <div className="triage-search-result-meta">{mrn} • {status}</div>
                        </div>
                        <span className="triage-search-result-action">Load</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

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
            <div><strong>ID:</strong> {data?.id_number ?? "---"}</div>
            <div><strong>Name:</strong> {data?.name ?? "---"}</div>
            <div><strong>Surname:</strong> {data?.surname ?? "---"}</div>
            <div><strong>Gender:</strong> {data?.gender ?? "---"}</div>
            <div>
              <strong>Age:</strong>{" "}
              {data?.ageYears != null ? (
                <span>{data.ageYears}y{data?.ageMonths ? ` ${data.ageMonths}m` : ""}</span>
              ) : data?.ageMonths != null ? (
                <span>{data.ageMonths} month{data.ageMonths !== 1 ? "s" : ""}</span>
              ) : data?.age != null ? (
                <span>{data.age}y</span>
              ) : (
                "---"
              )}
            </div>
            <div style={{ gridColumn: "1 / -1" }}><strong>Height:</strong> {data?.height_cm != null ? `${data.height_cm} cm` : "---"}</div>
          </div>
        )}
      </div>

      {[
        { key: "symptoms", label: "Symptoms" },
        { key: "duration", label: "Duration" },
        { key: "painLevel", label: "Pain Level" }
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

      {selectedMrn ? <div className="block">
        <div className="block-header">
          <h4 className="block-label">History</h4>
        </div>
        <div
          className="content-area"
        >
          <div>
            Open the Patient Records Library to review the complete clinical history.
          </div>
          <button
            type="button"
            className="triage-search-button"
            style={{ display: "block", margin: "12px auto 0" }}
            onClick={() => navigate(`/ehr?patient=${encodeURIComponent(selectedMrn)}`)}
          >
            Open patient history
          </button>
        </div>
      </div> : <div className="block">
        <div className="block-header">
          <h4 className="block-label">History</h4>
          <div className="action-group">
            <button
              className="edit-action-btn"
              onClick={() => setEditing(prev => ({ ...prev, history: !prev.history }))}
            >
              {editing.history ? "✅" : "✎"}
            </button>
            <button className="delete-btn" onClick={() => handleClear("history")}>×</button>
          </div>
        </div>
        {editing.history ? (
          <textarea
            className="edit-textarea"
            value={data?.history || ""}
            onChange={(e) => updateField("history", e.target.value)}
          />
        ) : (
          <div className="content-area">{data?.history || <em>Not provided</em>}</div>
        )}
      </div>}

      <div className="block critical-history-block">
        <div className="block-header">
          <h4 className="block-label">Critical History <span aria-hidden="true">*</span></h4>
        </div>
        <p className="content-area" style={{ margin: 0 }}>
          Every encounter must record these safety facts, or explicitly state that none have been established.
        </p>
        <div style={{ padding: "12px", display: "grid", gap: "12px" }}>
          <label className="form-group">
            <span style={{ fontWeight: 600 }}>Critical-history status</span>
            <select
              className="edit-input"
              value={data?.criticalHistoryStatus || ""}
              onChange={(event) => updateField("criticalHistoryStatus", event.target.value)}
              required
            >
              <option value="">Select one...</option>
              <option value="reported">Known critical history present</option>
              <option value="none_established">None established</option>
            </select>
          </label>
          {data?.criticalHistoryStatus === "reported" && criticalHistoryFields.map(([field, label]) => (
            <label className="form-group" key={field}>
              <span style={{ fontWeight: 600 }}>{label}</span>
              <textarea
                className="edit-textarea"
                value={asList(data?.criticalHistory?.[field]).join(', ')}
                onChange={(event) => updateCriticalHistory(field, event.target.value)}
                placeholder="Enter items separated by commas"
              />
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}