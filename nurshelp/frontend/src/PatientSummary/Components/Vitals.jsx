// src/PatientSummary/Components/Vitals.jsx
import React, { useEffect, useState } from "react";

/**
 * Vitals modal — full form
 * - Normalizes temperature (clinical guard)
 * - Parses BP strings into systolic/diastolic
 * - Outputs numeric values or nulls
 */

/** Same normalize used in PS.jsx */
const normalizeTemp = (val) => {
  if (val === "" || val === null || val === undefined) return null;
  const num = Number(val);
  if (Number.isNaN(num)) return null;
  if (num < 30 || num > 45) return null;
  return Number(num.toFixed(1));
};

export default function Vitals({ onComplete, onCancel, setVitalsData, vitalsData }) {
  const [formData, setFormData] = useState({
    temp: vitalsData?.temp === '--' ? '' : (vitalsData?.temp ?? ""),
    bp: vitalsData?.bp === '--' ? '' : (vitalsData?.bp ?? ""),
    bp_systolic: vitalsData?.bp_systolic ?? "",
    bp_diastolic: vitalsData?.bp_diastolic ?? "",
    bp_left_systolic: vitalsData?.bp_left_systolic ?? "",
    bp_left_diastolic: vitalsData?.bp_left_diastolic ?? "",
    bp_right_systolic: vitalsData?.bp_right_systolic ?? "",
    bp_right_diastolic: vitalsData?.bp_right_diastolic ?? "",
    hr: vitalsData?.hr === '--' ? '' : (vitalsData?.hr ?? ""),
    o2: vitalsData?.o2 === '--' ? '' : (vitalsData?.o2 ?? ""),
    resp: vitalsData?.resp === '--' ? '' : (vitalsData?.resp ?? ""),
    avpu: vitalsData?.avpu ?? ""
  });

  useEffect(() => {
    setFormData({
      temp: vitalsData?.temp === '--' ? '' : (vitalsData?.temp ?? ""),
      bp: vitalsData?.bp === '--' ? '' : (vitalsData?.bp ?? ""),
      bp_systolic: vitalsData?.bp_systolic ?? "",
      bp_diastolic: vitalsData?.bp_diastolic ?? "",
      bp_left_systolic: vitalsData?.bp_left_systolic ?? "",
      bp_left_diastolic: vitalsData?.bp_left_diastolic ?? "",
      bp_right_systolic: vitalsData?.bp_right_systolic ?? "",
      bp_right_diastolic: vitalsData?.bp_right_diastolic ?? "",
      hr: vitalsData?.hr === '--' ? '' : (vitalsData?.hr ?? ""),
      o2: vitalsData?.o2 === '--' ? '' : (vitalsData?.o2 ?? ""),
      resp: vitalsData?.resp === '--' ? '' : (vitalsData?.resp ?? ""),
      avpu: vitalsData?.avpu ?? ""
    });
  }, [vitalsData]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => {
      const next = { ...prev, [name]: value };

      // parse BP string into components
      if (name === "bp") {
        const m = value.match(/(\d{2,3})\s*(?:\/|over|-)\s*(\d{2,3})/i);
        if (m) {
          next.bp_systolic = m[1];
          next.bp_diastolic = m[2];
        }
      }

      // when editing numeric components, sync consolidated BP string
      if (name === "bp_systolic" || name === "bp_diastolic") {
        if (next.bp_systolic && next.bp_diastolic) {
          next.bp = `${next.bp_systolic}/${next.bp_diastolic}`;
        }
      }

      return next;
    });
  };

  const validate = () => {
    // require either bp string or both systolic and diastolic
    if (!formData.bp && (!formData.bp_systolic || !formData.bp_diastolic)) {
      return { ok: false, message: "BP (or both systolic & diastolic) required" };
    }

    // temperature must be normalizable (or empty)
    if (formData.temp !== "" && normalizeTemp(formData.temp) === null) {
      return { ok: false, message: "Temp must be a number between 30 and 45 °C" };
    }

    // basic numeric checks (optional further guards)
    if (formData.hr && (Number(formData.hr) < 20 || Number(formData.hr) > 220)) {
      return { ok: false, message: "HR out of range (20–220)" };
    }
    if (formData.o2 && (Number(formData.o2) < 30 || Number(formData.o2) > 100)) {
      return { ok: false, message: "O₂ must be between 30 and 100" };
    }
    if (formData.resp && (Number(formData.resp) < 5 || Number(formData.resp) > 80)) {
      return { ok: false, message: "Resp rate out of range (5–80)" };
    }

    return { ok: true };
  };

  const handleSave = () => {
    const v = validate();
    if (!v.ok) return alert(v.message);

    const parsedSbp = formData.bp_systolic ? Number(formData.bp_systolic) : null;
    const parsedDbp = formData.bp_diastolic ? Number(formData.bp_diastolic) : null;
    const consolidatedBp = (parsedSbp && parsedDbp) ? `${parsedSbp}/${parsedDbp}` : (formData.bp || null);

    const normalized = {
      temp: normalizeTemp(formData.temp),
      bp: consolidatedBp,
      bp_systolic: parsedSbp,
      bp_diastolic: parsedDbp,
      bp_left_systolic: formData.bp_left_systolic ? Number(formData.bp_left_systolic) : null,
      bp_left_diastolic: formData.bp_left_diastolic ? Number(formData.bp_left_diastolic) : null,
      bp_right_systolic: formData.bp_right_systolic ? Number(formData.bp_right_systolic) : null,
      bp_right_diastolic: formData.bp_right_diastolic ? Number(formData.bp_right_diastolic) : null,
      hr: formData.hr ? Number(formData.hr) : null,
      o2: formData.o2 ? Number(formData.o2) : null,
      resp: formData.resp ? Number(formData.resp) : null,
      avpu: formData.avpu ? String(formData.avpu).trim().toUpperCase() : null
    };

    // pass normalized to parent (PS.jsx) which will merge/save
    setVitalsData(normalized);
    onComplete();
  };

  // Helper: Check if value is in clinical range
  const isInRange = (field, value) => {
    if (!value || value === "") return null; // empty = not checked
    const num = Number(value);
    const ranges = {
      temp: [30, 45],
      hr: [20, 220],
      o2: [30, 100],
      resp: [5, 80],
      bp_systolic: [60, 250],
      bp_diastolic: [30, 150]
    };
    if (!ranges[field]) return null;
    const [min, max] = ranges[field];
    return num >= min && num <= max;
  };

  return (
    <div className="modal-overlay" style={{ zIndex: 2000 }}>
      <style>{`
        /* Base modal styling - fixed positioning */
        .modal-overlay {
          position: fixed;
          inset: 0;
          background-color: rgba(15, 23, 42, 0.55);
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          padding: 16px;
          z-index: 9999;
        }
        
        .modal-content {
          display: flex;
          flex-direction: column;
          background-color: var(--card, white);
          border-radius: 10px;
          box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
          width: 100%;
          max-width: 550px;
          max-height: 85vh;
          padding: 0;
          overflow: hidden;
        }
        
        .modal-title {
          font-size: 1.3rem;
          font-weight: 700;
          color: var(--text-primary, #000);
          padding: 20px 24px 0;
          margin: 0;
          flex-shrink: 0;
        }
        
        /* Scrollable content area */
        .vitals-form-container {
          display: flex;
          flex-direction: column;
          gap: 20px;
          padding: 16px 24px;
          overflow-y: auto;
          overflow-x: hidden;
          flex: 1;
          /* Custom scrollbar styling */
          scrollbar-width: thin;
          scrollbar-color: rgba(100, 108, 255, 0.3) transparent;
        }
        
        .vitals-form-container::-webkit-scrollbar {
          width: 6px;
        }
        
        .vitals-form-container::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .vitals-form-container::-webkit-scrollbar-thumb {
          background-color: rgba(100, 108, 255, 0.3);
          border-radius: 3px;
        }
        
        .vitals-section {
          padding: 14px;
          background: rgba(0, 0, 0, 0.02);
          border-radius: 8px;
          border-left: 3px solid var(--accent, #646cff);
        }
        
        .vitals-section-title {
          font-size: 0.85rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: var(--text-secondary, #666);
          margin-bottom: 12px;
        }
        
        .form-group {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        
        .form-group label {
          font-size: 0.9rem;
          font-weight: 600;
          color: var(--text-primary, #000);
        }
        
        .form-group .vitals-input {
          font-size: 16px;
          padding: 12px;
          min-height: 44px;
          border: 1.5px solid #ddd;
          border-radius: 6px;
          box-sizing: border-box;
          transition: all 0.2s;
        }
        
        .form-group .vitals-input:focus {
          outline: none;
          border-color: var(--accent, #646cff);
          box-shadow: 0 0 0 3px rgba(100, 108, 255, 0.1);
        }
        
        .form-group .vitals-input.out-of-range {
          border-color: #f97316;
          background-color: rgba(249, 115, 22, 0.05);
        }
        
        .form-group small {
          font-size: 0.75rem;
          color: #666;
          margin-top: 2px;
        }
        
        .bp-section-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        
        .form-group-full {
          grid-column: 1 / -1;
        }
        
        .vitals-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        
        /* Fixed action buttons at bottom */
        .modal-actions {
          display: flex;
          gap: 8px;
          padding: 16px 24px 20px;
          border-top: 1px solid rgba(0, 0, 0, 0.08);
          background-color: var(--card, white);
          flex-shrink: 0;
        }
        
        .modal-actions button {
          flex: 1;
          padding: 12px 16px;
          font-size: 0.95rem;
          font-weight: 600;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 44px;
        }
        
        .modal-actions .save-file-btn {
          background-color: var(--accent, #646cff);
          color: white;
        }
        
        .modal-actions .save-file-btn:hover {
          opacity: 0.9;
          transform: translateY(-1px);
        }
        
        .modal-actions .save-file-btn:active {
          transform: translateY(0);
        }
        
        .modal-actions .edit-action-btn {
          background-color: #f0f0f0;
          color: #333;
        }
        
        .modal-actions .edit-action-btn:hover {
          background-color: #e0e0e0;
        }
        
        /* Tablet responsiveness */
        @media (max-width: 900px) {
          .modal-content {
            width: 90%;
            max-width: 600px;
            max-height: 85vh;
          }
          .vitals-grid {
            grid-template-columns: 1fr 1fr;
          }
          .bp-section-grid {
            grid-template-columns: 1fr 1fr;
          }
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
          .modal-overlay {
            padding: 8px;
            align-items: flex-end;
          }
          
          .modal-content {
            width: 100%;
            max-width: calc(100vw - 16px);
            max-height: 90vh;
            border-radius: 16px 16px 0 0;
          }
          
          .modal-title {
            padding: 16px 16px 0;
            font-size: 1.2rem;
          }
          
          .vitals-form-container {
            padding: 14px 16px;
          }
          
          .modal-actions {
            padding: 12px 16px 16px;
          }
          
          .vitals-grid {
            grid-template-columns: 1fr;
            gap: 12px;
          }
          
          .bp-section-grid {
            grid-template-columns: 1fr;
            gap: 10px;
          }
          
          .form-group label {
            font-size: 0.9rem;
          }
          
          .vitals-input {
            font-size: 16px;
            min-height: 48px;
            padding: 12px;
          }
          
          .modal-actions button {
            min-height: 48px;
          }
          
          .vitals-section {
            padding: 12px;
          }
          
          .vitals-section-title {
            font-size: 0.8rem;
          }
        }
        
        /* Small mobile */
        @media (max-width: 480px) {
          .modal-overlay {
            padding: 4px;
            align-items: flex-end;
          }
          
          .modal-content {
            max-width: calc(100vw - 8px);
            max-height: 92vh;
          }
          
          .modal-title {
            padding: 12px 12px 0;
            font-size: 1.1rem;
            margin-bottom: 4px;
          }
          
          .vitals-form-container {
            padding: 12px;
          }
          
          .modal-actions {
            padding: 10px 12px 12px;
            gap: 6px;
          }
          
          .form-group label {
            font-size: 0.85rem;
          }
          
          .vitals-input {
            font-size: 16px;
            padding: 10px;
            min-height: 44px;
          }
          
          .vitals-grid {
            gap: 8px;
          }
          
          .bp-section-grid {
            gap: 8px;
          }
        }
      `}</style>
      <div className="modal-content">
        <h2 className="modal-title">Clinical Vitals Intake</h2>

        <div className="vitals-form-container">
          {/* Required Vitals Section */}
          <div className="vitals-section">
            <div className="vitals-section-title">📋 Essential Vitals (Required)</div>
            <div className="vitals-grid">
              <div className="form-group">
                <label>Temperature (°C) <span style={{color: 'red'}}>*</span></label>
                <input 
                  name="temp" 
                  type="number" 
                  step="0.1" 
                  min="30"
                  max="45"
                  value={formData.temp} 
                  onChange={handleChange} 
                  className={`vitals-input ${isInRange('temp', formData.temp) === false ? 'out-of-range' : ''}`}
                  placeholder="37.2"
                />
                <small>Valid: 30–45°C (e.g., 37.2)</small>
              </div>

              <div className="form-group">
                <label>BP (mmHg) <span style={{color: 'red'}}>*</span></label>
                <input 
                  name="bp" 
                  type="text" 
                  value={formData.bp} 
                  onChange={handleChange} 
                  className="vitals-input"
                  placeholder="120/80"
                />
                <small>Format: systolic/diastolic</small>
              </div>

              <div className="form-group">
                <label>Heart Rate (bpm)</label>
                <input 
                  name="hr" 
                  type="number" 
                  min="20"
                  max="220"
                  value={formData.hr} 
                  onChange={handleChange} 
                  className={`vitals-input ${isInRange('hr', formData.hr) === false ? 'out-of-range' : ''}`}
                  placeholder="72"
                />
                <small>Valid: 20–220 bpm</small>
              </div>

              <div className="form-group">
                <label>O₂ Saturation (%)</label>
                <input 
                  name="o2" 
                  type="number" 
                  min="30"
                  max="100"
                  value={formData.o2} 
                  onChange={handleChange} 
                  className={`vitals-input ${isInRange('o2', formData.o2) === false ? 'out-of-range' : ''}`}
                  placeholder="98"
                />
                <small>Valid: 30–100%</small>
              </div>

              <div className="form-group">
                <label>Respiratory Rate (breaths/min)</label>
                <input 
                  name="resp" 
                  type="number" 
                  min="5"
                  max="80"
                  value={formData.resp} 
                  onChange={handleChange} 
                  className={`vitals-input ${isInRange('resp', formData.resp) === false ? 'out-of-range' : ''}`}
                  placeholder="16"
                />
                <small>Valid: 5–80 breaths/min</small>
              </div>

              <div className="form-group">
                <label>AVPU Status</label>
                <input 
                  name="avpu" 
                  type="text" 
                  value={formData.avpu} 
                  onChange={handleChange} 
                  className="vitals-input"
                  placeholder="A (Alert)"
                />
                <small>Alert, Verbal, Pain, Unresponsive</small>
              </div>
            </div>
          </div>

          {/* BP Components Section (Optional) */}
          <div className="vitals-section">
            <div className="vitals-section-title">📊 Blood Pressure - Detailed</div>
            
            <div className="form-group form-group-full" style={{marginBottom: 8}}>
              <label style={{fontSize: '0.8rem', color: '#666'}}>Consolidated BP (or enter systolic/diastolic separately below)</label>
            </div>

            <div className="bp-section-grid">
              <div className="form-group">
                <label>Systolic (mmHg)</label>
                <input 
                  name="bp_systolic" 
                  type="number" 
                  min="60"
                  max="250"
                  value={formData.bp_systolic} 
                  onChange={handleChange} 
                  className={`vitals-input ${isInRange('bp_systolic', formData.bp_systolic) === false ? 'out-of-range' : ''}`}
                  placeholder="120"
                />
                <small>60–250 mmHg</small>
              </div>

              <div className="form-group">
                <label>Diastolic (mmHg)</label>
                <input 
                  name="bp_diastolic" 
                  type="number" 
                  min="30"
                  max="150"
                  value={formData.bp_diastolic} 
                  onChange={handleChange} 
                  className={`vitals-input ${isInRange('bp_diastolic', formData.bp_diastolic) === false ? 'out-of-range' : ''}`}
                  placeholder="80"
                />
                <small>30–150 mmHg</small>
              </div>

              <div className="form-group">
                <label>BP Left Systolic</label>
                <input 
                  name="bp_left_systolic" 
                  type="number" 
                  min="60"
                  max="250"
                  value={formData.bp_left_systolic} 
                  onChange={handleChange} 
                  className="vitals-input"
                  placeholder="120"
                />
              </div>

              <div className="form-group">
                <label>BP Left Diastolic</label>
                <input 
                  name="bp_left_diastolic" 
                  type="number" 
                  min="30"
                  max="150"
                  value={formData.bp_left_diastolic} 
                  onChange={handleChange} 
                  className="vitals-input"
                  placeholder="80"
                />
              </div>

              <div className="form-group">
                <label>BP Right Systolic</label>
                <input 
                  name="bp_right_systolic" 
                  type="number" 
                  min="60"
                  max="250"
                  value={formData.bp_right_systolic} 
                  onChange={handleChange} 
                  className="vitals-input"
                  placeholder="120"
                />
              </div>

              <div className="form-group">
                <label>BP Right Diastolic</label>
                <input 
                  name="bp_right_diastolic" 
                  type="number" 
                  min="30"
                  max="150"
                  value={formData.bp_right_diastolic} 
                  onChange={handleChange} 
                  className="vitals-input"
                  placeholder="80"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="modal-actions">
          <button onClick={handleSave} className="save-file-btn">✓ Save & Continue</button>
          <button onClick={onCancel} className="edit-action-btn">✕ Cancel</button>
        </div>
      </div>
    </div>
  );
}