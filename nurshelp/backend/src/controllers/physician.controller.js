import { Encounter } from "../models/encounter.model.js";
import { TriageAssessment } from "../models/triageAssessment.model.js";
import { Disposition } from "../models/disposition.model.js";
import { PatientJourney } from "../models/patientJourney.model.js";
import PatientEHRRecord from "../models/patientEHRRecord.model.js";
import { Ward } from "../models/ward.model.js";
import { ClinicalStaff } from "../models/clinicalstaff.model.js";
import { getAccessiblePatients, getFieldProjection, auditPatientAccess } from "../utils/rebac.js";
import axios from "axios";

const PYTHON_URL = process.env.PYTHON_URL || "http://localhost:5000/llm-triage";  // Updated to use sophisticated endpoint

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
    // CRITICAL: Only call Flask if we actually need physician ranking AND:
    // 1. Don't already have triage physician data, AND
    // 2. Don't already have complete clinical analysis from GGUF (which means Flask already ran)
    const hasCompleteTriageData = triage.assigned_physician || triage.assignedPhysician;
    const hasGGUFAlreadyRun = triage.aiTriageReport?.pathophysiology && triage.aiTriageReport?.pathophysiology.length > 50;
    
    if (candidates.length === 0 && !hasCompleteTriageData && !hasGGUFAlreadyRun) {
      try {
        console.log(`[physician.controller] 🔍 DIAGNOSTIC CHECK:`);
        console.log(`[physician.controller]   - hasCompleteTriageData: ${hasCompleteTriageData}`);
        console.log(`[physician.controller]   - hasGGUFAlreadyRun: ${hasGGUFAlreadyRun}`);
        console.log(`[physician.controller]   - triage.aiTriageReport exists: ${!!triage.aiTriageReport}`);
        console.log(`[physician.controller]   - pathophysiology length: ${(triage.aiTriageReport?.pathophysiology || '').length}`);
        console.log(`[physician.controller] ⚠️  Calling Python triage - THIS MAY BE DUPLICATE! Check triage logs for parallel requests.`);
        const payload = {
          patient: enc.patient || {},
          vitals: enc.vitals || {},
          transcript: (Array.isArray(enc.transcript) ? enc.transcript.map(t => t.text).join(" ") : (enc.transcript || ""))
        };
        console.log(`[physician.controller] 📨 Payload sent to Flask (no chiefComplaint): ${JSON.stringify(payload).length} bytes`);
        const pyRes = await axios.post(PYTHON_URL, payload, { timeout: 0 });  // No timeout for GGUF inference
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
    } else if (candidates.length === 0 && hasGGUFAlreadyRun) {
      console.log(`[physician.controller] GGUF already ran and clinical data present - skipping Python to prevent duplicate GGUF inference. Using existing triage data.`);
    } else if (candidates.length === 0 && hasCompleteTriageData) {
      console.log(`[physician.controller] Using existing triage-assigned physician instead of calling Python`);
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

/**
 * GET /api/physician/cases?context=ER
 * 
 * Fetch all patients in 3-lane view:
 * - activeCases: RED priority, awaiting physician decision
 * - observationCases: YELLOW/GREEN, awaiting disposition
 * - escalations: Escalated from clinical nurse
 */
export const getPhysicianCases = async (req, res) => {
  try {
    console.log('[GET PHYSICIAN CASES] Called');
    const token = req.headers.authorization?.split(' ')[1];
    const staffId = req.user?._id;
    const hospitalId = req.user?.hospitalId; // From JWT

    if (!staffId || !hospitalId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    // Fetch physician's profile with assignment
    const physician = await ClinicalStaff.findById(staffId);
    if (!physician) {
      return res.status(404).json({ message: "Physician not found" });
    }

    console.log(`[ReBAC] Physician ${physician.firstName} ${physician.lastName} in ${physician.assignment?.department}`);

    // Get patients accessible to this physician (Circle of Care)
    const accessiblePatients = await getAccessiblePatients(staffId, physician);
    console.log(`[ReBAC] ${accessiblePatients.length} patients accessible`);

    // Categorize by lane (ESI / acuity)
    const activeCases = [];
    const observationCases = [];
    const escalations = [];

    const projection = getFieldProjection(physician.systemRole);

    for (const patient of accessiblePatients) {
      // Audit access
      await auditPatientAccess(
        staffId,
        patient._id,
        'PATIENT_VIEW',
        hospitalId,
        'SAME_DEPARTMENT'
      );

      const caseData = {
        _id: patient._id,
        id: patient._id.toString(),
        firstName: patient.name,
        lastName: patient.surname,
        age: patient.age,
        gender: patient.gender,
        mrn: patient.mrn,
        priority: patient.esiLevel ? `ESI-${patient.esiLevel}` : 'UNKNOWN',
        vitals: patient.latestVitals || {},
        bedNumber: patient.currentBedId,
        acuityScore: patient.acuityScore,
        admissionTime: patient.admissionTime,
        department: patient.currentDepartment,
        status: patient.status,
      };

      // Categorize by acuity
      if (patient.esiLevel === 1 || patient.acuityScore >= 85) {
        activeCases.push(caseData);
      } else if (patient.esiLevel === 2 || patient.acuityScore >= 70) {
        activeCases.push(caseData);
      } else {
        observationCases.push(caseData);
      }
    }

    // Sort by acuity (highest first)
    activeCases.sort((a, b) => (b.acuityScore || 0) - (a.acuityScore || 0));
    observationCases.sort((a, b) => (b.acuityScore || 0) - (a.acuityScore || 0));

    const metrics = {
      casesReviewed: accessiblePatients.length,
      dispositionsMade: 0,
      avgResponseTime: '--',
      complianceRate: '--',
      responseTargetMet: 0,
      department: physician.assignment?.department,
      physician: `${physician.firstName} ${physician.lastName}`,
    };

    console.log(`[ReBAC] Returning: ${activeCases.length} active, ${observationCases.length} observation`);

    return res.status(200).json({
      activeCases,
      observationCases,
      escalations,
      metrics,
    });
  } catch (err) {
    console.error("[physician.controller] getPhysicianCases error:", err);
    return res.status(500).json({ message: "Failed to fetch physician cases", error: err.message });
  }
};

/**
 * GET /api/physician/beds?context=ER
 * 
 * Fetch bed occupancy for physician's department (Circle of Care)
 */
export const getWardBeds = async (req, res) => {
  try {
    console.log('[GET WARD BEDS] Called');
    const staffId = req.user?._id;
    const hospitalId = req.user?.hospitalId;

    if (!staffId || !hospitalId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    // Fetch physician's profile
    const physician = await ClinicalStaff.findById(staffId);
    if (!physician) {
      return res.status(404).json({ message: "Physician not found" });
    }

    console.log(`[ReBAC] Fetching beds for physician's department: ${physician.assignment?.department}`);

    const { BedSchedule } = await import('../models/bedSchedule.model.js');

    // Get beds in physician's department only (Circle of Care)
    const departmentBeds = [
      'ED-Ward-A-02', 'ED-Ward-A-03', 'ED-Ward-A-04', 'ED-Ward-A-05',
      'ED-Ward-A-06', 'ED-Ward-A-07', 'ICU-Ward-A-01', 'ICU-Ward-A-02'
    ];

    const bedDocs = await BedSchedule.find({
      bedId: { $in: departmentBeds }
    }).populate('currentEncounterId');

    console.log(`[ReBAC] Found ${bedDocs.length} beds in physician's department`);

    const beds = bedDocs.map(bed => ({
      bedId: bed.bedId,
      roomNumber: bed.roomNumber,
      bedType: bed.bedType || 'STANDARD',
      status: bed.status,
      department: bed.department,
      occupant: bed.currentEncounterId ? bed.currentEncounterId.toString() : null,
      acuityLevel: bed.acuityLevel,
    }));

    const occupancy = {
      total: beds.length,
      occupied: beds.filter(b => b.status === 'OCCUPIED').length,
      available: beds.filter(b => b.status === 'AVAILABLE').length,
      cleaning: beds.filter(b => b.status === 'CLEANING').length,
      maintenance: beds.filter(b => b.status === 'MAINTENANCE').length,
    };

    console.log(`[ReBAC] Bed occupancy: ${occupancy.occupied}/${occupancy.total} occupied`);

    return res.status(200).json({ beds, occupancy });
  } catch (err) {
    console.error("[physician.controller] getWardBeds error:", err);
    return res.status(500).json({ message: "Failed to fetch ward beds", error: err.message });
  }
};

/**
 * POST /api/disposition
 * 
 * Physician submits disposition decision (ADMIT, DISCHARGE, TRANSFER)
 * Body: { patientId, decision, wardIfAdmit, bedNumber, notes }
 */
export const submitDisposition = async (req, res) => {
  try {
    console.log('[SUBMIT DISPOSITION] Called');
    const { patientId, decision, wardIfAdmit, bedNumber, notes } = req.body;
    console.log('[SUBMIT DISPOSITION] PatientId:', patientId, 'Decision:', decision, 'Ward:', wardIfAdmit, 'Bed:', bedNumber);
    const hospitalId = req.hospitalId;
    const physicianId = req.user?._id;
    console.log('[SUBMIT DISPOSITION] HospitalId:', hospitalId, 'PhysicianId:', physicianId);

    if (!hospitalId || !physicianId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    // Get triage assessment
    const triage = await TriageAssessment.findOne({ patientId, hospitalId });
    if (!triage) {
      return res.status(404).json({ message: "Triage assessment not found" });
    }

    // Determine if discrepancy exists
    const predictedAdmission = triage.needsBed; // true = triage predicts admission
    const actualAdmission = decision === 'ADMITTED'; // true = physician admits

    let discrepancy = null;
    if (predictedAdmission && actualAdmission) {
      discrepancy = { type: 'TRUE_POSITIVE', hasDiscrepancy: false };
    } else if (!predictedAdmission && !actualAdmission) {
      discrepancy = { type: 'TRUE_NEGATIVE', hasDiscrepancy: false };
    } else if (predictedAdmission && !actualAdmission) {
      discrepancy = {
        type: 'FALSE_POSITIVE',
        hasDiscrepancy: true,
        reason: 'Triage predicted admission, but physician discharged',
        severity: 'MEDIUM',
      };
    } else {
      discrepancy = {
        type: 'FALSE_NEGATIVE',
        hasDiscrepancy: true,
        reason: 'Triage predicted discharge, but physician admitted',
        severity: 'HIGH',
      };
    }

    // Create disposition record
    const disposition = new Disposition({
      hospitalId,
      patientId,
      triageAssessmentId: triage._id,
      physicianId,
      decision,
      admissionDetails: decision === 'ADMITTED' ? { wardName: wardIfAdmit, bedNumber } : null,
      notes,
      triageRecommendation: triage.needsBed,
      discrepancy,
      triageCompletedAt: triage.triageCompletedAt,
      decisionMadeAt: new Date(),
      responseTimeMinutes: Math.round((new Date() - triage.triageCompletedAt) / 60000),
      status: 'COMPLETED',
    });

    await disposition.save();

    // Update triage record
    triage.dispositionId = disposition._id;
    triage.status = 'DISPOSED';
    await triage.save();

    // Update patient journey
    const journey = await PatientJourney.findOne({ hospitalId, patientId });
    if (journey) {
      if (decision === 'ADMITTED') {
        journey.currentState = 'ADMITTED';
        journey.currentLocation = {
          locationType: 'WARD_BED',
          wardName: wardIfAdmit,
          bedNumber,
        };
        journey.admittedAt = new Date();
      } else if (decision === 'DISCHARGED') {
        journey.currentState = 'DISCHARGED';
        journey.currentLocation = { locationType: 'DISCHARGED' };
        journey.dischargedAt = new Date();
      } else if (decision === 'TRANSFERRED') {
        journey.currentState = 'TRANSFERRED';
        journey.currentLocation = { locationType: 'TRANSFERRED' };
        journey.transferredAt = new Date();
      }
      journey.dispositionId = disposition._id;
      await journey.save();
    }

    return res.status(200).json({
      status: 'DISPOSITION_RECORDED',
      decision,
      discrepancy: discrepancy.hasDiscrepancy ? discrepancy : null,
      message: `✓ Patient ${decision.toLowerCase()}`,
    });
  } catch (err) {
    console.error("[physician.controller] submitDisposition error:", err);
    return res.status(500).json({ message: "Failed to submit disposition" });
  }
};
