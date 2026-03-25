import { Encounter } from "../models/encounter.model.js";
import axios from "axios";

const PYTHON_URL = process.env.PYTHON_URL || "http://localhost:5000/triage";

function parsePhysiciansCsv(csvText) {
  const lines = csvText.split(/\r?\n/).filter(l => l.trim());
  const rows = [];
  // assume header present
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    // match: id,name,specialty,"[...]",workload
    const m = line.match(/^\s*(\d+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*("\[.*\]")\s*,\s*([0-9\.eE+-]+)\s*$/);
    if (!m) continue;
    try {
      const id = String(m[1]);
      const name = m[2];
      const specialty = m[3];
      const maskStr = m[4];
      const workload = parseFloat(m[5]);
      let mask = null;
      try {
        mask = JSON.parse(maskStr.replace(/"/g, ''));
      } catch (e) {
        // fallback: extract numbers
        const nums = maskStr.match(/-?\d+/g) || [];
        mask = nums.map(n => Number(n));
      }
      rows.push({ physician_id: id, name, specialty, availability_mask: mask, workload_score: workload });
    } catch (e) {
      continue;
    }
  }
  return rows;
}

export const getRecommendations = async (req, res) => {
  try {
    const encounterId = req.query.encounterId;
    if (!encounterId) return res.status(200).json({ physicians: [] });

    const enc = await import("../models/encounter.model.js").then(m => m.Encounter.findById(encounterId));
    if (!enc) return res.status(404).json({ message: "Encounter not found" });

    const triage = enc.triage || {};
    const specialty = (triage.assigned_specialty || triage.assignedSpecialty || triage?.specialty || "").toString();

    // Prefer triage-provided ranked lists if available
    const lists = [];
    if (Array.isArray(triage.top_5_composite_physicians)) lists.push({ source: 'triage_composite', list: triage.top_5_composite_physicians });
    if (Array.isArray(triage.top_5_embedding_physicians)) lists.push({ source: 'triage_embedding', list: triage.top_5_embedding_physicians });

    const normalize = (p, sourceTag) => {
      const pid = String(p.id ?? p.physician_id ?? p.physicianId ?? p.physicianIdRaw ?? p);
      const name = p.name ?? p.displayName ?? p.physician_name ?? `ID:${pid}`;
      const specialtyField = p.specialty ?? p.department ?? specialty ?? "General Medicine";
      const score = Number(p.composite_score ?? p.embedding_sim ?? p.score ?? p.match ?? 0) || 0;
      const available_now = (p.available_now === true || p.available === true) ? true : (p.available_now === false || p.available === false ? false : null);
      const reason = p.reason ?? (sourceTag ? String(sourceTag) : 'triage_rank');
      return { id: pid, name, specialty: String(specialtyField), score, available_now, source: sourceTag, reason };
    };

    let candidates = [];
    const seen = new Set();

    for (const entry of lists) {
      const src = entry.source || 'triage';
      for (const p of (entry.list || [])) {
        const n = normalize(p, src);
        if (seen.has(n.id)) continue;
        seen.add(n.id);
        candidates.push(n);
      }
    }

    // If no candidates from stored triage lists, call the Python triage service for authoritative ranking
    if (candidates.length === 0) {
      try {
        const payload = {
          patient: enc.patient || {},
          vitals: enc.vitals || {},
          transcript: (Array.isArray(enc.transcript) ? enc.transcript.map(t => t.text).join(" ") : (enc.transcript || ""))
        };
        const pyRes = await axios.post(PYTHON_URL, payload, { timeout: 300000 });
        const pyTriage = pyRes?.data ?? {};
        // adopt assigned specialty from Python if available (authoritative)
        if (pyTriage.assigned_specialty) {
          specialty = (pyTriage.assigned_specialty || specialty).toString();
        }
        if (Array.isArray(pyTriage.top_5_composite_physicians)) {
          for (const p of pyTriage.top_5_composite_physicians) {
            const n = normalize(p, 'py_composite');
            if (!seen.has(n.id)) { seen.add(n.id); candidates.push(n); }
          }
        }
        if (Array.isArray(pyTriage.top_5_embedding_physicians)) {
          for (const p of pyTriage.top_5_embedding_physicians) {
            const n = normalize(p, 'py_embedding');
            if (!seen.has(n.id)) { seen.add(n.id); candidates.push(n); }
          }
        }
      } catch (e) {
        console.warn("[physician.controller] Python triage call failed:", e.message || e);
      }
    }

    // Reorder candidates to prioritize: 1) specialty match & available, 2) specialty match, 3) available, 4) others
    const lowerSpec = (specialty || '').toString().toLowerCase();
    const buckets = { spec_avail: [], spec_notavail: [], avail_notspec: [], others: [] };
    for (const c of candidates) {
      const isSpecMatch = c.specialty && c.specialty.toString().toLowerCase().includes(lowerSpec) && lowerSpec !== '';
      if (isSpecMatch && c.available_now === true) buckets.spec_avail.push(c);
      else if (isSpecMatch) buckets.spec_notavail.push(c);
      else if (c.available_now === true) buckets.avail_notspec.push(c);
      else buckets.others.push(c);
    }

    const sortByScoreDesc = (arr) => arr.sort((a,b) => (b.score || 0) - (a.score || 0));
    sortByScoreDesc(buckets.spec_avail);
    sortByScoreDesc(buckets.spec_notavail);
    sortByScoreDesc(buckets.avail_notspec);
    sortByScoreDesc(buckets.others);

    let final = [...buckets.spec_avail, ...buckets.spec_notavail, ...buckets.avail_notspec, ...buckets.others].slice(0,5);

    // Ensure triage-assigned physician (if present) is preferred and shown first
    try {
      const assigned = triage.assigned_physician || triage.assignedPhysician || triage.assigned || null;
      const assignedId = assigned && (assigned.id || assigned.physician_id || assigned.physicianId) ? String(assigned.id || assigned.physician_id || assigned.physicianId) : null;
      const assignedName = assigned && (assigned.name || assigned.displayName) ? (assigned.name || assigned.displayName) : null;
      if (assignedId) {
        // if assigned already appears in final, move to front; otherwise prepend a best-effort entry
        const idx = final.findIndex(c => String(c.id) === assignedId);
        if (idx > 0) {
          const entry = final.splice(idx, 1)[0];
          final = [entry, ...final].slice(0,5);
        } else if (idx === -1) {
          // CRITICAL: Score MUST be capped at 1.0 (displays as 100%) to prevent 99900% overflow
          const assignedEntry = { id: assignedId, name: assignedName || `ID:${assignedId}`, specialty: specialty || (assigned.specialty || ''), score: 1.0, available_now: true, reason: 'triage_assigned' };
          // dedupe and prepend
          final = [assignedEntry, ...final.filter(c => String(c.id) !== assignedId)].slice(0,5);
        }
      }
    } catch (e) {
      console.warn('[physician.controller] assigned physician promotion failed:', e.message || e);
    }

    // DISABLED: Python specialty search fallback - ESI_Engine now handles specialty-specific physician ranking
    // and database includes proper OBGYN physicians, so fallback search is no longer needed

    return res.status(200).json({ physicians: final });
  } catch (err) {
    console.error("[physician.controller] getRecommendations error:", err);
    return res.status(500).json({ message: "Failed to fetch physician recommendations" });
  }
};
