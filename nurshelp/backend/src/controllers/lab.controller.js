import mongoose from "mongoose";
import { Encounter } from "../models/encounter.model.js";
import { EquipmentUsage } from "../models/equipmentUsage.model.js";
import { buildDiagnosticEhrContext } from "../utils/diagnosticEhrContext.js";

export const canAccessLabEncounter = (encounter, req) => {
  if (!encounter || !req?.user) return false;

  const encounterHospitalId = encounter.hospitalId?.toString();
  const userHospitalId = req.user.hospitalId || req.hospitalId;
  const userId = req.user.id || req.user._id;

  if (!encounterHospitalId || !userHospitalId || encounterHospitalId !== userHospitalId.toString()) {
    return false;
  }

  if (!userId) return false;

  const normalizedRole = (req.user.role || req.userRole || '').toUpperCase();
  const isPhysicianRole = ['PHYSICIAN', 'DOCTOR', 'SPECIALIST'].includes(normalizedRole);
  const isSpecialistRole = ['RADIOLOGIST', 'PATHOLOGIST', 'PHARMACIST'].includes(normalizedRole);

  if (isSpecialistRole) {
    return true;
  }

  if (encounter.assignedPhysicianId?.toString() === userId.toString()) {
    return true;
  }

  if (encounter.assigned_clinicians?.some((clinician) => clinician.id?.toString() === userId.toString())) {
    return true;
  }

  if (encounter.clinical_handovers?.some((handover) => handover.clinician_id?.toString() === userId.toString())) {
    return true;
  }

  const activeClinicalStatuses = ['HOSPITALIZED', 'DIAGNOSED', 'SEEN', 'AWAITING_PHYSICIAN', 'OFFICIAL_REPORT', 'DISCHARGED'];
  if (isPhysicianRole && activeClinicalStatuses.includes(encounter.status)) {
    return true;
  }

  return false;
};

export const canManageSpecialtyRequest = (req, specialty) => {
  const expectedRole = specialty === 'RADIOLOGY' ? 'RADIOLOGIST' : 'PATHOLOGIST';
  return String(req.user?.role || req.userRole || '').toUpperCase() === expectedRole;
};

/**
 * Lab Request Controller
 * Handles complete lab request workflow for Radiology & Pathology departments
 * 
 * Workflow:
 * 1. Physician requests lab → POST /lab-request
 * 2. Radiologist/Pathologist reviews queue → GET /lab-requests?specialty=
 * 3. Specialist acknowledges request → PATCH /lab-request/:requestId/acknowledge
 * 4. Specialist submits report → PATCH /lab-request/:requestId/submit-report
 * 5. Physician reviews report → PATCH /lab-request/:requestId/review
 */

// ============ PHYSICIAN ENDPOINTS ============

/**
 * POST /api/encounters/:id/lab-request
 * Physician creates a new lab request
 */
export const createLabRequest = async (req, res) => {
  try {
    const encounterId = req.params.encounterId || req.params.id;
    const { specialty, lab_type, description, priority, requested_by } = req.body;

    // Find encounter first so we can validate access against the actual encounter record
    let encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      // Fallbacks: sometimes frontend passes a bed id or patient id — try common alternative lookups
      console.warn('[LAB CONTROLLER] createLabRequest: primary findById failed for', encounterId, '— trying fallbacks');
      const orQueries = [
        { assigned_bed: encounterId },
        { 'bed_display_info._id': encounterId },
        { 'bed_display_info.bedId': encounterId },
        { 'patient.id_number': encounterId },
        { 'patient._id': encounterId }
      ];

      encounter = await Encounter.findOne({ $or: orQueries });
      if (!encounter) {
        return res.status(404).json({ message: 'Encounter not found' });
      }
    }

    const canAccess = canAccessLabEncounter(encounter, req);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    // Validate required fields
    if (!specialty || !lab_type || !['RADIOLOGY', 'PATHOLOGY'].includes(specialty)) {
      return res.status(400).json({ message: 'Invalid specialty or lab_type' });
    }

    // Initialize lab_requests if needed
    if (!encounter.lab_requests) {
      encounter.lab_requests = [];
    }

    const capturedAt = new Date();
    const patientId = encounter.patient?.id_number;
    const historicalEncounters = encounter.hospitalId && patientId
      ? await Encounter.find({
        hospitalId: encounter.hospitalId,
        _id: { $ne: encounter._id },
        'patient.id_number': patientId,
        status: 'DISCHARGED',
        date: { $lt: encounter.date || capturedAt }
      })
        .select('_id date status diagnosis patient.id_number patient.name patient.surname patient.age patient.gender patient.symptoms patient.history patient.criticalHistoryStatus patient.criticalHistory vitals triage SATS_reasoning')
        .sort({ date: -1 })
        .lean()
      : [];
    const ehrContext = buildDiagnosticEhrContext({
      encounter,
      historicalEncounters,
      capturedAt
    });

    // Fetch full clinician record with credentials for signature
    let signingClinician = null;
    if (req.user?._id) {
      try {
        const { Practitioner } = await import('../models/clinicalstaff.model.js');
        signingClinician = await Practitioner.findById(req.user._id).lean();
      } catch (err) {
        console.warn('[Lab Controller] Failed to fetch clinician for request signature:', err.message);
      }
    }

    // Import signature utility
    const { generateClinicianSignature } = await import('../utils/clinicianSignature.js');

    // Create new lab request
    const newLabRequest = {
      request_id: new mongoose.Types.ObjectId(),
      specialty,
      lab_type,
      description: description || '',
      priority: priority || 'MEDIUM',
      requested_by: requested_by || req.user.name || 'Unknown',
      ehrContext,
      requested_at: new Date(),
      status: 'PENDING',
      created_at: new Date(),
      updated_at: new Date()
    };

    // Attach clinician signature metadata if clinician provided
    if (signingClinician) {
      try {
        newLabRequest.clinicianSignature = generateClinicianSignature({
          clinician: signingClinician,
          documentData: newLabRequest,
          documentType: 'LAB_REQUEST'
        });
      } catch (err) {
        console.warn('[Lab Controller] Failed to generate signature for lab request:', err.message);
      }
    }

    encounter.lab_requests.push(newLabRequest);

    // Update status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: `LAB_REQUEST_CREATED_${specialty}`,
      changed_by: req.user.id,
      changed_at: new Date(),
      notes: `${specialty} lab requested: ${lab_type}`
    });

    await encounter.save();

    return res.status(201).json({
      message: 'Lab request created successfully',
      request_id: newLabRequest.request_id,
      status: newLabRequest.status,
      created_at: newLabRequest.created_at
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] createLabRequest error:', err);
    return res.status(500).json({ message: 'Error creating lab request', error: err.message });
  }
};

/**
 * GET /api/encounters/:id/lab-requests
 * Get all lab requests for a specific encounter
 */
export const getEncounterLabRequests = async (req, res) => {
  try {
    const encounterId = req.params.encounterId || req.params.id;

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    const canAccess = canAccessLabEncounter(encounter, req);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    return res.status(200).json({
      lab_requests: encounter.lab_requests || []
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] getEncounterLabRequests error:', err);
    return res.status(500).json({ message: 'Error fetching lab requests', error: err.message });
  }
};

// ============ RADIOLOGIST/PATHOLOGIST ENDPOINTS ============

/**
 * GET /api/lab-requests?specialty=RADIOLOGY&status=PENDING
 * Specialty dashboard fetches their pending queue
 */
export const getLabRequestsForSpecialty = async (req, res) => {
  try {
    const { specialty, status } = req.query;

    if (!specialty || !['RADIOLOGY', 'PATHOLOGY'].includes(specialty)) {
      return res.status(400).json({ message: 'Invalid or missing specialty parameter' });
    }

    // Find all encounters with pending lab requests for this specialty
    const query = {
      hospitalId: req.user.hospitalId,
      'lab_requests.specialty': specialty
    };

    // Add status filter if provided
    if (status) {
      query['lab_requests.status'] = status;
    }

    const encounters = await Encounter.find(query).lean();

    // Format response: flatten lab_requests with encounter context
    const labRequests = [];
    encounters.forEach(encounter => {
      const filteredRequests = encounter.lab_requests.filter(req => {
        if (req.specialty !== specialty) return false;
        if (status && req.status !== status) return false;
        return true;
      });

      filteredRequests.forEach(labReq => {
        labRequests.push({
          encounterId: encounter._id,
          request_id: labReq.request_id,
          specialty: labReq.specialty,
          lab_type: labReq.lab_type,
          description: labReq.description,
          priority: labReq.priority,
          status: labReq.status,
          requested_by: labReq.requested_by,
          requested_at: labReq.requested_at,
          assigned_technician: labReq.assigned_technician,
          assigned_at: labReq.assigned_at,
          started_at: labReq.started_at,
          started_by: labReq.started_by,
          report_content: labReq.report_content,
          clinical_impression: labReq.clinical_impression,
          recommendations: labReq.recommendations || [],
          submitted_at: labReq.submitted_at,
          submitted_by: labReq.submitted_by,
          reviewed_at: labReq.reviewed_at,
          reviewed_by: labReq.reviewed_by,
          review_notes: labReq.review_notes,
          clinicianSignature: labReq.clinicianSignature || null,
          
          // Encounter context for specialist
          patientName: encounter.patient?.name || encounter.patientName || 'Unknown',
          patientAge: encounter.patient?.age || 'N/A',
          patientId: encounter.patient?.id_number || 'N/A',
          vitals: encounter.vitals || {},
          diagnosis: encounter.diagnosis || '',
          assignedPhysicianName: encounter.assignedPhysicianName || encounter.assigned_physician?.name || 'N/A',
          ehrContext: labReq.ehrContext || null
        });
      });
    });

    // Sort by priority (URGENT first) and then by requested_at (oldest first)
    labRequests.sort((a, b) => {
      const priorityOrder = { URGENT: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
      const aPriority = priorityOrder[a.priority] || 2;
      const bPriority = priorityOrder[b.priority] || 2;
      
      if (aPriority !== bPriority) return aPriority - bPriority;
      return new Date(a.requested_at) - new Date(b.requested_at);
    });

    return res.status(200).json({
      specialty,
      status: status || 'ALL',
      count: labRequests.length,
      lab_requests: labRequests
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] getLabRequestsForSpecialty error:', err);
    return res.status(500).json({ message: 'Error fetching lab requests', error: err.message });
  }
};

/**
 * PATCH /api/encounters/:id/lab-request/:requestId/acknowledge
 * Specialist acknowledges and accepts a lab request
 */
export const acknowledgeLabRequest = async (req, res) => {
  try {
    const encounterId = req.params.encounterId || req.params.id;
    const { requestId } = req.params;

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    const canAccess = canAccessLabEncounter(encounter, req);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }
    if (!canManageSpecialtyRequest(req, labRequest.specialty)) {
      return res.status(403).json({ message: 'Only the assigned specialty can accept this request' });
    }
    if (labRequest.status !== 'PENDING') {
      return res.status(409).json({ message: `Request cannot be accepted from status ${labRequest.status}` });
    }

    // Update lab request status
    labRequest.status = 'ACKNOWLEDGED';
    labRequest.assigned_technician = req.user.name;
    labRequest.assigned_technician_id = req.user._id || req.user.id;
    labRequest.assigned_at = new Date();
    labRequest.updated_at = new Date();

    // Update status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: `LAB_REQUEST_ACKNOWLEDGED_${labRequest.specialty}`,
      changed_by: req.user.id,
      changed_at: new Date(),
      notes: `${labRequest.lab_type} lab acknowledged by ${labRequest.assigned_technician}`
    });

    await encounter.save();

    return res.status(200).json({
      message: 'Lab request acknowledged',
      status: labRequest.status,
      assigned_technician: labRequest.assigned_technician,
      assigned_at: labRequest.assigned_at
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] acknowledgeLabRequest error:', err);
    return res.status(500).json({ message: 'Error acknowledging lab request', error: err.message });
  }
};

export const startLabRequest = async (req, res) => {
  try {
    const { encounterId, requestId } = req.params;
    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    if (!canAccessLabEncounter(encounter, req)) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    const labRequest = encounter.lab_requests?.find((request) => request.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }
    if (!canManageSpecialtyRequest(req, labRequest.specialty)) {
      return res.status(403).json({ message: 'Only the assigned specialty can start this request' });
    }
    if (labRequest.status !== 'ACKNOWLEDGED') {
      return res.status(409).json({ message: `Request cannot be started from status ${labRequest.status}` });
    }

    const startedAt = new Date();
    labRequest.status = 'IN_PROGRESS';
    labRequest.started_at = startedAt;
    labRequest.started_by = req.user.name;
    labRequest.started_by_id = req.user._id || req.user.id;
    labRequest.updated_at = startedAt;
    encounter.status_history ||= [];
    encounter.status_history.push({
      status: `LAB_REQUEST_STARTED_${labRequest.specialty}`,
      changed_by: req.user.id,
      changed_at: startedAt,
      notes: `${labRequest.lab_type} started by ${req.user.name}`
    });

    await encounter.save();
    return res.status(200).json({
      message: 'Request work started',
      status: labRequest.status,
      started_at: labRequest.started_at,
      started_by: labRequest.started_by
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] startLabRequest error:', err);
    return res.status(500).json({ message: 'Error starting lab request', error: err.message });
  }
};

/**
 * PATCH /api/encounters/:id/lab-request/:requestId/submit-report
 * Specialist submits completed lab report
 */
export const submitLabReport = async (req, res) => {
  try {
    const encounterId = req.params.encounterId || req.params.id;
    const { requestId } = req.params;
    const {
      report_content,
      report_images,
      clinical_impression,
      recommendations
    } = req.body;

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    const canAccess = canAccessLabEncounter(encounter, req);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }
    if (!canManageSpecialtyRequest(req, labRequest.specialty)) {
      return res.status(403).json({ message: 'Only the assigned specialty can submit this report' });
    }
    if (labRequest.status !== 'IN_PROGRESS') {
      return res.status(409).json({ message: `Report cannot be submitted from status ${labRequest.status}` });
    }
    if (!String(report_content || '').trim() || !String(clinical_impression || '').trim()) {
      return res.status(400).json({ message: 'Findings and clinical impression are required' });
    }
    if (labRequest.specialty === 'RADIOLOGY' && Array.isArray(report_images) && report_images.length > 0) {
      return res.status(400).json({ message: 'Radiology image data must remain in the configured imaging archive' });
    }

    // Fetch full clinician record with credentials for signature
    let signingClinician = null;
    if (req.user?._id) {
      try {
        const { Practitioner } = await import('../models/clinicalstaff.model.js');
        signingClinician = await Practitioner.findById(req.user._id).lean();
      } catch (err) {
        console.warn('[Lab Controller] Failed to fetch clinician for report signature:', err.message);
      }
    }

    // Import signature utility
    const { generateClinicianSignature } = await import('../utils/clinicianSignature.js');

    // Update lab request with report
    labRequest.report_content = String(report_content).trim();
    labRequest.report_images = labRequest.specialty === 'RADIOLOGY' ? [] : report_images || [];
    labRequest.clinical_impression = String(clinical_impression).trim();
    labRequest.recommendations = Array.isArray(recommendations)
      ? recommendations.filter((recommendation) => typeof recommendation === 'string').map((recommendation) => recommendation.trim()).filter(Boolean)
      : [];
    labRequest.status = 'COMPLETED';
    labRequest.submitted_at = new Date();
    labRequest.submitted_by = req.user.name;
    labRequest.submitted_by_id = req.user._id || req.user.id;
    labRequest.updated_at = new Date();

    // Attach clinician signature metadata if clinician provided
    if (signingClinician) {
      try {
        labRequest.clinicianSignature = generateClinicianSignature({
          clinician: signingClinician,
          documentData: labRequest,
          documentType: 'LAB_REPORT'
        });
      } catch (err) {
        console.warn('[Lab Controller] Failed to generate signature:', err.message);
      }
    }

    // Update status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: `LAB_REPORT_SUBMITTED_${labRequest.specialty}`,
      changed_by: req.user.id,
      changed_at: new Date(),
      notes: `${labRequest.lab_type} report submitted by ${labRequest.submitted_by}`
    });

    await encounter.save();

    // If specialist submitted equipmentUsed, increment usage counter
    try {
      const equipmentUsed = req.body.equipmentUsed;
      if (equipmentUsed) {
        const updated = await EquipmentUsage.findOneAndUpdate(
          { equipmentName: equipmentUsed },
          { $inc: { count: 1 }, $set: { lastUsedAt: new Date() } },
          { upsert: true, new: true }
        );
        console.log('[LAB] Incremented equipment usage for', equipmentUsed, '->', updated.count);
      }
    } catch (err) {
      console.warn('[LAB] Failed to increment equipment usage:', err.message);
    }

    return res.status(200).json({
      message: 'Lab report submitted successfully',
      status: labRequest.status,
      submitted_at: labRequest.submitted_at,
      submitted_by: labRequest.submitted_by
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] submitLabReport error:', err);
    return res.status(500).json({ message: 'Error submitting lab report', error: err.message });
  }
};

/**
 * PATCH /api/encounters/:id/lab-request/:requestId/decline
 * Specialist declines a lab request with reason
 */
export const declineLabRequest = async (req, res) => {
  try {
    const encounterId = req.params.encounterId || req.params.id;
    const { requestId } = req.params;
    const { reason } = req.body;

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    const canAccess = canAccessLabEncounter(encounter, req);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }
    if (!canManageSpecialtyRequest(req, labRequest.specialty)) {
      return res.status(403).json({ message: 'Only the assigned specialty can decline this request' });
    }
    if (!['PENDING', 'ACKNOWLEDGED'].includes(labRequest.status)) {
      return res.status(409).json({ message: `Request cannot be declined from status ${labRequest.status}` });
    }
    if (!String(reason || '').trim()) {
      return res.status(400).json({ message: 'A reason is required to decline a request' });
    }

    // Update lab request
    labRequest.status = 'DECLINED';
    labRequest.review_notes = String(reason).trim();
    labRequest.reviewed_at = new Date();
    labRequest.reviewed_by = req.user.name;
    labRequest.reviewed_by_id = req.user._id || req.user.id;
    labRequest.updated_at = new Date();

    // Update status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: `LAB_REQUEST_DECLINED_${labRequest.specialty}`,
      changed_by: req.user.id,
      changed_at: new Date(),
      notes: `${labRequest.lab_type} lab declined. Reason: ${reason || 'Not specified'}`
    });

    await encounter.save();

    return res.status(200).json({
      message: 'Lab request declined',
      status: labRequest.status,
      review_notes: labRequest.review_notes
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] declineLabRequest error:', err);
    return res.status(500).json({ message: 'Error declining lab request', error: err.message });
  }
};

// ============ PHYSICIAN REVIEW ENDPOINTS ============

/**
 * PATCH /api/encounters/:id/lab-request/:requestId/review
 * Physician reviews and acknowledges completed lab report
 */
export const reviewLabReport = async (req, res) => {
  try {
    const encounterId = req.params.encounterId || req.params.id;
    const { requestId } = req.params;
    const { review_notes } = req.body;

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    const canAccess = canAccessLabEncounter(encounter, req);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }
    if (!['PHYSICIAN', 'DOCTOR'].includes(String(req.user?.role || req.userRole || '').toUpperCase())) {
      return res.status(403).json({ message: 'Only the ordering physician may acknowledge a report' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }
    if (labRequest.status !== 'COMPLETED') {
      return res.status(409).json({ message: `Report cannot be acknowledged from status ${labRequest.status}` });
    }

    // Update with physician review
    labRequest.status = 'REVIEWED';
    labRequest.reviewed_at = new Date();
    labRequest.reviewed_by = req.user.name;
    labRequest.reviewed_by_id = req.user._id || req.user.id;
    labRequest.review_notes = review_notes || '';
    labRequest.updated_at = new Date();

    // Update status history
    if (!encounter.status_history) {
      encounter.status_history = [];
    }
    encounter.status_history.push({
      status: `LAB_REPORT_REVIEWED_${labRequest.specialty}`,
      changed_by: req.user.id,
      changed_at: new Date(),
      notes: `${labRequest.lab_type} report reviewed by ${labRequest.reviewed_by}`
    });

    await encounter.save();

    return res.status(200).json({
      message: 'Lab report reviewed successfully',
      status: labRequest.status,
      reviewed_at: labRequest.reviewed_at,
      reviewed_by: labRequest.reviewed_by
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] reviewLabReport error:', err);
    return res.status(500).json({ message: 'Error reviewing lab report', error: err.message });
  }
};

/**
 * GET /api/lab-reports?status=COMPLETED
 * Physician dashboard fetches completed lab reports awaiting review
 */
export const getLabReportsForPhysician = async (req, res) => {
  try {
    const { status } = req.query;
    const targetStatus = status || 'COMPLETED';

    // Find all encounters with completed lab reports for this hospital
    const encounters = await Encounter.find({
      hospitalId: req.user.hospitalId,
      'lab_requests.status': targetStatus
    }).lean();

    // Flatten lab reports with encounter context
    const labReports = [];
    encounters.forEach(encounter => {
      const relevantReports = encounter.lab_requests.filter(req => req.status === targetStatus);

      relevantReports.forEach(labReq => {
        labReports.push({
          encounterId: encounter._id,
          request_id: labReq.request_id,
          specialty: labReq.specialty,
          lab_type: labReq.lab_type,
          description: labReq.description,
          priority: labReq.priority,
          status: labReq.status,
          requested_at: labReq.requested_at,
          submitted_at: labReq.submitted_at,
          submitted_by: labReq.submitted_by,
          report_content: labReq.report_content,
          report_images: labReq.report_images,
          clinical_impression: labReq.clinical_impression,
          recommendations: labReq.recommendations,
          
          // Encounter context
          patientName: encounter.patient?.name || encounter.patientName || 'Unknown',
          patientAge: encounter.patient?.age || 'N/A',
          patientId: encounter.patient?.id_number || 'N/A',
          diagnosis: encounter.diagnosis || '',
          vitals: encounter.vitals || {}
        });
      });
    });

    // Sort by submitted_at (newest first)
    labReports.sort((a, b) => new Date(b.submitted_at) - new Date(a.submitted_at));

    return res.status(200).json({
      count: labReports.length,
      lab_reports: labReports
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] getLabReportsForPhysician error:', err);
    return res.status(500).json({ message: 'Error fetching lab reports', error: err.message });
  }
};

/**
 * Get all lab requests for a physician (aggregated across encounters)
 * Supports filtering by status and specialty
 */
export const getLabRequestsForPhysician = async (req, res) => {
  try {
    const { status, specialty } = req.query;
    const physicianId = req.user?.id || req.user?._id;
    const hospitalId = req.user?.hospitalId || req.hospitalId;

    if (!physicianId) {
      return res.status(401).json({ message: 'Not authorized' });
    }

    // Build query to find encounters with lab requests
    const query = {
      hospitalId: hospitalId,
      'lab_requests.0': { $exists: true } // has at least one lab request
    };

    // Add status filter if provided
    if (status) {
      query['lab_requests.status'] = status;
    }

    // Add specialty filter if provided
    if (specialty) {
      query['lab_requests.specialty'] = specialty;
    }

    // Find encounters with lab requests for this hospital
    const encounters = await Encounter.find(query).lean();

    // Flatten lab requests with encounter context
    const labRequests = [];
    encounters.forEach(encounter => {
      let relevantRequests = encounter.lab_requests || [];

      // Apply status filter
      if (status) {
        relevantRequests = relevantRequests.filter(req => req.status === status);
      }

      // Apply specialty filter
      if (specialty) {
        relevantRequests = relevantRequests.filter(req => req.specialty === specialty);
      }

      relevantRequests.forEach(labReq => {
        labRequests.push({
          _id: labReq._id || labReq.request_id,
          encounterId: encounter._id,
          request_id: labReq.request_id,
          specialty: labReq.specialty,
          lab_type: labReq.lab_type,
          description: labReq.description,
          priority: labReq.priority,
          status: labReq.status,
          requested_at: labReq.requested_at,
          requested_by: labReq.requested_by,
          acknowledged_at: labReq.acknowledged_at,
          submitted_at: labReq.submitted_at,
          submitted_by: labReq.submitted_by,
          report_content: labReq.report_content,
          clinical_impression: labReq.clinical_impression,
          recommendations: labReq.recommendations,

          // Encounter context
          patientName: encounter.patient?.name || encounter.patientName || 'Unknown',
          patientAge: encounter.patient?.age || 'N/A',
          patientId: encounter.patient?.id_number || 'N/A',
          diagnosis: encounter.diagnosis || '',
          ward: encounter.assigned_ward?.name || 'N/A'
        });
      });
    });

    // Sort by requested_at (newest first)
    labRequests.sort((a, b) => new Date(b.requested_at) - new Date(a.requested_at));

    return res.status(200).json({
      count: labRequests.length,
      labRequests: labRequests
    });
  } catch (err) {
    console.error('[LAB CONTROLLER] getLabRequestsForPhysician error:', err);
    return res.status(500).json({ message: 'Error fetching lab requests', error: err.message });
  }
};
