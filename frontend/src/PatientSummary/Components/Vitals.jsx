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

  return (
    <div className="modal-overlay" style={{ zIndex: 2000 }}>
      <div className="modal-content">
        <h2 className="modal-title">Clinical Vitals Intake</h2>

        <div className="vitals-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div className="form-group">
            <label>Temp (°C)</label>
            <input name="temp" type="number" step="0.1" value={formData.temp} onChange={handleChange} className="vitals-input" />
            <small style={{ color: "#6b7280" }}>e.g. 37.2 (valid range 30–45)</small>
          </div>

          <div className="form-group">
            <label>BP (mmHg)</label>
            <input name="bp" type="text" value={formData.bp} onChange={handleChange} placeholder="e.g. 120/80" className="vitals-input" />
            <small style={{ color: "#6b7280" }}>You can also enter systolic/diastolic separately</small>
          </div>

          <div className="form-group">
            <label>BP Sys</label>
            <input name="bp_systolic" type="number" value={formData.bp_systolic} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>BP Dia</label>
            <input name="bp_diastolic" type="number" value={formData.bp_diastolic} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>BP Left Sys</label>
            <input name="bp_left_systolic" type="number" value={formData.bp_left_systolic} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>BP Left Dia</label>
            <input name="bp_left_diastolic" type="number" value={formData.bp_left_diastolic} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>BP Right Sys</label>
            <input name="bp_right_systolic" type="number" value={formData.bp_right_systolic} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>BP Right Dia</label>
            <input name="bp_right_diastolic" type="number" value={formData.bp_right_diastolic} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>HR (BPM)</label>
            <input name="hr" type="number" value={formData.hr} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>O₂ Sat (%)</label>
            <input name="o2" type="number" value={formData.o2} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>Resp Rate</label>
            <input name="resp" type="number" value={formData.resp} onChange={handleChange} className="vitals-input" />
          </div>

          <div className="form-group">
            <label>AVPU</label>
            <input name="avpu" type="text" value={formData.avpu} onChange={handleChange} placeholder="A, V, P, U" className="vitals-input" />
          </div>
        </div>

        <div className="modal-actions" style={{ marginTop: 12, display: "flex", gap: 8 }}>
          <button onClick={handleSave} className="save-file-btn">Save & Continue</button>
          <button onClick={onCancel} className="edit-action-btn">Cancel</button>
        </div>
      </div>
    </div>
  );
}