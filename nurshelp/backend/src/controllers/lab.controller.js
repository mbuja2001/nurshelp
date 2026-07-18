import mongoose from "mongoose";
import { Encounter } from "../models/encounter.model.js";
import { canAccessEncounter } from "./encounterAccess.js";
import { EquipmentUsage } from "../models/equipmentUsage.model.js";

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
    const { id: encounterId } = req.params;
    const { specialty, lab_type, description, priority, requested_by } = req.body;

    // Validate access
    const canAccess = await canAccessEncounter(req.user.hospitalId, encounterId, req.user.id);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    // Validate required fields
    if (!specialty || !lab_type || !['RADIOLOGY', 'PATHOLOGY'].includes(specialty)) {
      return res.status(400).json({ message: 'Invalid specialty or lab_type' });
    }

    // Find encounter
    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    // Initialize lab_requests if needed
    if (!encounter.lab_requests) {
      encounter.lab_requests = [];
    }

    // Create new lab request
    const newLabRequest = {
      request_id: new mongoose.Types.ObjectId(),
      specialty,
      lab_type,
      description: description || '',
      priority: priority || 'MEDIUM',
      requested_by: requested_by || req.user.name || 'Unknown',
      requested_at: new Date(),
      status: 'PENDING',
      created_at: new Date(),
      updated_at: new Date()
    };

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
    const { id: encounterId } = req.params;

    // Validate access
    const canAccess = await canAccessEncounter(req.user.hospitalId, encounterId, req.user.id);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
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
          
          // Encounter context for specialist
          patientName: encounter.patient?.name || encounter.patientName || 'Unknown',
          patientAge: encounter.patient?.age || 'N/A',
          patientId: encounter.patient?.id_number || 'N/A',
          vitals: encounter.vitals || {},
          diagnosis: encounter.diagnosis || '',
          assignedPhysicianName: encounter.assignedPhysicianName || encounter.assigned_physician?.name || 'N/A'
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
    const { id: encounterId, requestId } = req.params;
    const { assigned_technician } = req.body;

    // Validate access
    const canAccess = await canAccessEncounter(req.user.hospitalId, encounterId, req.user.id);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }

    // Update lab request status
    labRequest.status = 'ACKNOWLEDGED';
    labRequest.assigned_technician = assigned_technician || req.user.name;
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

/**
 * PATCH /api/encounters/:id/lab-request/:requestId/submit-report
 * Specialist submits completed lab report
 */
export const submitLabReport = async (req, res) => {
  try {
    const { id: encounterId, requestId } = req.params;
    const {
      report_content,
      report_images,
      clinical_impression,
      recommendations,
      submitted_by
    } = req.body;

    // Validate access
    const canAccess = await canAccessEncounter(req.user.hospitalId, encounterId, req.user.id);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }

    // Update lab request with report
    labRequest.report_content = report_content || '';
    labRequest.report_images = report_images || [];
    labRequest.clinical_impression = clinical_impression || '';
    labRequest.recommendations = recommendations || [];
    labRequest.status = 'COMPLETED';
    labRequest.submitted_at = new Date();
    labRequest.submitted_by = submitted_by || req.user.name;
    labRequest.updated_at = new Date();

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
    const { id: encounterId, requestId } = req.params;
    const { reason } = req.body;

    // Validate access
    const canAccess = await canAccessEncounter(req.user.hospitalId, encounterId, req.user.id);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }

    // Update lab request
    labRequest.status = 'DECLINED';
    labRequest.review_notes = reason || 'Declined by specialist';
    labRequest.reviewed_at = new Date();
    labRequest.reviewed_by = req.user.name;
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
    const { id: encounterId, requestId } = req.params;
    const { review_notes } = req.body;

    // Validate access
    const canAccess = await canAccessEncounter(req.user.hospitalId, encounterId, req.user.id);
    if (!canAccess) {
      return res.status(403).json({ message: 'Access denied to this encounter' });
    }

    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ message: 'Encounter not found' });
    }

    // Find the specific lab request
    const labRequest = encounter.lab_requests?.find(req => req.request_id.toString() === requestId);
    if (!labRequest) {
      return res.status(404).json({ message: 'Lab request not found' });
    }

    // Update with physician review
    labRequest.status = 'REVIEWED';
    labRequest.reviewed_at = new Date();
    labRequest.reviewed_by = req.user.name;
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
