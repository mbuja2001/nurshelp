import { Encounter } from '../models/encounter.model.js';
import { RadiologyOrder } from '../models/radiology.model.js';
import { LabOrder } from '../models/lab.model.js';
import { Prescription } from '../models/pharmacy.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';

/**
 * SPECIALIST WORKFLOW CONTROLLER
 * 
 * Manages specialist queues, request workflows, and result submissions
 * - Radiologist dashboard: incoming imaging requests
 * - Pathologist dashboard: incoming lab requests  
 * - Pharmacist dashboard: medication orders/dispensing
 */

// ============================================================
// RADIOLOGIST ENDPOINTS
// ============================================================

/**
 * GET /api/specialist/radiology/queue
 * Get all radiology requests for assigned radiologist
 * Role: RADIOLOGIST only
 */
export const getRadiologyQueue = async (req, res) => {
  try {
    const radiologistId = req.user?._id;
    
    if (!radiologistId) {
      return res.status(401).json({ error: 'Unauthorized - not authenticated' });
    }

    // Verify user is radiologist
    const staff = await ClinicalStaff.findById(radiologistId);
    if (!staff || (staff.role !== 'RADIOLOGY_TECHNICIAN' && staff.role !== 'RADIOLOGIST')) {
      return res.status(403).json({ error: 'Forbidden - only radiologists can access this' });
    }

    // Get all pending/in-progress radiology orders for this hospital
    const orders = await RadiologyOrder.find({
      hospitalId: staff.hospitalId,
      status: { $in: ['pending', 'scheduled', 'in-progress'] }
    })
      .populate('patientId', 'firstName lastName dateOfBirth gender mrn')
      .populate('orderingPhysicianId', 'firstName lastName specialty')
      .sort({ orderDate: -1 })
      .lean();

    // Map to response format
    const queue = orders.map(order => ({
      requestId: order._id,
      orderNumber: order.orderNumber,
      patientName: `${order.patientId?.firstName} ${order.patientId?.lastName}`,
      patientId: order.patientId?._id,
      mrn: order.patientId?.mrn,
      age: calculateAge(order.patientId?.dateOfBirth),
      gender: order.patientId?.gender,
      studyType: order.studyType,
      bodyPart: order.bodyPart,
      clinicalIndication: order.clinicalIndication,
      urgency: order.urgency,
      orderingPhysician: `Dr. ${order.orderingPhysicianId?.firstName} ${order.orderingPhysicianId?.lastName}`,
      orderingPhysicianId: order.orderingPhysicianId?._id,
      orderDate: order.orderDate,
      status: order.status,
      estimatedCompletionTime: calculateEstimatedTime(order.urgency),
      notes: order.notes
    }));

    res.json({
      success: true,
      count: queue.length,
      queue
    });
  } catch (error) {
    console.error('[SPECIALIST] Error fetching radiology queue:', error);
    res.status(500).json({ error: error.message });
  }
};

/**
 * POST /api/specialist/radiology/:requestId/status
 * Update radiology order status (received, in-progress, completed)
 */
export const updateRadiologyStatus = async (req, res) => {
  try {
    const { requestId } = req.params;
    const { status, notes } = req.body;
    const radiologistId = req.user?._id;

    const validStatuses = ['received', 'in-progress', 'completed'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({ error: 'Invalid status' });
    }

    const order = await RadiologyOrder.findByIdAndUpdate(
      requestId,
      {
        status,
        ...(status === 'in-progress' && { performedDate: new Date(), performedBy: radiologistId }),
        notes
      },
      { new: true }
    );

    if (!order) {
      return res.status(404).json({ error: 'Radiology order not found' });
    }

    res.json({
      success: true,
      message: `Radiology order status updated to ${status}`,
      order
    });
  } catch (error) {
    console.error('[SPECIALIST] Error updating radiology status:', error);
    res.status(500).json({ error: error.message });
  }
};

/**
 * POST /api/specialist/radiology/:requestId/results
 * Submit radiology report results
 */
export const submitRadiologyResults = async (req, res) => {
  try {
    const { requestId } = req.params;
    const { findings, impression, recommendations, criticalFindings } = req.body;
    const radiologistId = req.user?._id;

    const order = await RadiologyOrder.findById(requestId);
    if (!order) {
      return res.status(404).json({ error: 'Radiology order not found' });
    }

    // Create report
    const RadiologyReport = require('../models/radiology.model.js').RadiologyReport;
    const report = await RadiologyReport.create({
      radiologyOrderId: requestId,
      encounterId: order.encounterId,
      patientId: order.patientId,
      radiologistId,
      findings,
      impression,
      recommendations: recommendations || [],
      criticalFindings: criticalFindings || [],
      reportDate: new Date(),
      status: 'approved'
    });

    // Update order status
    await RadiologyOrder.findByIdAndUpdate(requestId, {
      status: 'reported'
    });

    // Get ordering physician to notify
    const physician = await ClinicalStaff.findById(order.orderingPhysicianId);
    
    res.json({
      success: true,
      message: 'Radiology results submitted successfully',
      report,
      notifyPhysician: {
        physicianId: physician?._id,
        physicianName: `Dr. ${physician?.firstName} ${physician?.lastName}`,
        message: `Radiology report ready for ${order.studyType} (${order.bodyPart})`
      }
    });
  } catch (error) {
    console.error('[SPECIALIST] Error submitting radiology results:', error);
    res.status(500).json({ error: error.message });
  }
};

// ============================================================
// PATHOLOGIST ENDPOINTS
// ============================================================

/**
 * GET /api/specialist/pathology/queue
 * Get all lab requests for assigned pathologist
 * Role: PATHOLOGIST only
 */
export const getPathologyQueue = async (req, res) => {
  try {
    const pathologistId = req.user?._id;
    
    if (!pathologistId) {
      return res.status(401).json({ error: 'Unauthorized - not authenticated' });
    }

    // Verify user is pathologist
    const staff = await ClinicalStaff.findById(pathologistId);
    if (!staff || (staff.role !== 'PATHOLOGY_TECHNICIAN' && staff.role !== 'PATHOLOGIST')) {
      return res.status(403).json({ error: 'Forbidden - only pathologists can access this' });
    }

    // Get all pending/in-progress lab orders
    const orders = await LabOrder.find({
      hospitalId: staff.hospitalId,
      status: { $in: ['pending', 'collected', 'in-progress'] }
    })
      .populate('patientId', 'firstName lastName dateOfBirth gender mrn')
      .populate('orderingPhysicianId', 'firstName lastName specialty')
      .sort({ orderDate: -1 })
      .lean();

    // Map to response format
    const queue = orders.map(order => ({
      requestId: order._id,
      orderNumber: order.orderNumber,
      patientName: `${order.patientId?.firstName} ${order.patientId?.lastName}`,
      patientId: order.patientId?._id,
      mrn: order.patientId?.mrn,
      age: calculateAge(order.patientId?.dateOfBirth),
      gender: order.patientId?.gender,
      testName: order.testName,
      testCode: order.testCode,
      sampleType: order.sampleType,
      priority: order.priority,
      orderingPhysician: `Dr. ${order.orderingPhysicianId?.firstName} ${order.orderingPhysicianId?.lastName}`,
      orderingPhysicianId: order.orderingPhysicianId?._id,
      orderDate: order.orderDate,
      collectionTime: order.collectionTime,
      status: order.status,
      estimatedCompletionTime: calculateEstimatedTime(order.priority),
      notes: order.clinicalInfo
    }));

    res.json({
      success: true,
      count: queue.length,
      queue
    });
  } catch (error) {
    console.error('[SPECIALIST] Error fetching pathology queue:', error);
    res.status(500).json({ error: error.message });
  }
};

/**
 * POST /api/specialist/pathology/:requestId/status
 * Update lab order status
 */
export const updatePathologyStatus = async (req, res) => {
  try {
    const { requestId } = req.params;
    const { status, notes, processingNotes } = req.body;
    const pathologistId = req.user?._id;

    const validStatuses = ['received', 'in-progress', 'processing', 'completed'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({ error: 'Invalid status' });
    }

    const order = await LabOrder.findByIdAndUpdate(
      requestId,
      {
        status,
        ...(status === 'processing' && { processingStartTime: new Date() } ),
        ...(status === 'completed' && { completionTime: new Date() }),
        notes: processingNotes || notes
      },
      { new: true }
    );

    if (!order) {
      return res.status(404).json({ error: 'Lab order not found' });
    }

    res.json({
      success: true,
      message: `Lab order status updated to ${status}`,
      order
    });
  } catch (error) {
    console.error('[SPECIALIST] Error updating pathology status:', error);
    res.status(500).json({ error: error.message });
  }
};

/**
 * POST /api/specialist/pathology/:requestId/results
 * Submit lab test results
 */
export const submitPathologyResults = async (req, res) => {
  try {
    const { requestId } = req.params;
    const { resultValue, resultUnit, referenceRange, interpretation, isAbnormal, flagSeverity } = req.body;
    const pathologistId = req.user?._id;

    const order = await LabOrder.findById(requestId);
    if (!order) {
      return res.status(404).json({ error: 'Lab order not found' });
    }

    // Create result record
    const LabResult = require('../models/lab.model.js').LabResult;
    const result = await LabResult.create({
      labOrderId: requestId,
      encounterId: order.encounterId,
      patientId: order.patientId,
      testPerformedBy: pathologistId,
      resultDate: new Date(),
      resultValue,
      resultUnit,
      referenceRange,
      isAbnormal,
      flagSeverity: isAbnormal ? (flagSeverity || 'normal') : null,
      interpretation,
      approvalStatus: 'approved'
    });

    // Update order status
    await LabOrder.findByIdAndUpdate(requestId, {
      status: 'completed',
      completionTime: new Date()
    });

    // Get ordering physician to notify
    const physician = await ClinicalStaff.findById(order.orderingPhysicianId);
    
    res.json({
      success: true,
      message: 'Lab results submitted successfully',
      result,
      notifyPhysician: {
        physicianId: physician?._id,
        physicianName: `Dr. ${physician?.firstName} ${physician?.lastName}`,
        message: `Lab results ready for ${order.testName}`,
        isAbnormal,
        flagSeverity
      }
    });
  } catch (error) {
    console.error('[SPECIALIST] Error submitting pathology results:', error);
    res.status(500).json({ error: error.message });
  }
};

// ============================================================
// PHARMACIST ENDPOINTS
// ============================================================

/**
 * GET /api/specialist/pharmacy/queue
 * Get all prescription orders for pharmacist
 * Role: PHARMACIST only
 */
export const getPharmacyQueue = async (req, res) => {
  try {
    const pharmacistId = req.user?._id;
    
    if (!pharmacistId) {
      return res.status(401).json({ error: 'Unauthorized - not authenticated' });
    }

    // Verify user is pharmacist
    const staff = await ClinicalStaff.findById(pharmacistId);
    if (!staff || (staff.role !== 'PHARMACIST' && staff.role !== 'PHARMACY_TECHNICIAN')) {
      return res.status(403).json({ error: 'Forbidden - only pharmacists can access this' });
    }

    // Get all pending prescriptions for this hospital
    const orders = await Prescription.find({
      hospitalId: staff.hospitalId,
      status: { $in: ['pending', 'approved'] }
    })
      .populate('patientId', 'firstName lastName dateOfBirth gender')
      .populate('prescribedBy', 'firstName lastName specialty')
      .sort({ dateIssued: -1 })
      .lean();

    // Map to response format
    const queue = orders.map(order => ({
      requestId: order._id,
      medicationId: order.medicationId,
      medicationName: order.medicationName,
      dosage: order.dosage,
      frequency: order.frequency,
      quantity: order.quantity,
      route: order.route,
      patientName: `${order.patientId?.firstName} ${order.patientId?.lastName}`,
      patientId: order.patientId?._id,
      prescribedBy: `Dr. ${order.prescribedBy?.firstName} ${order.prescribedBy?.lastName}`,
      prescribedById: order.prescribedBy?._id,
      dateIssued: order.dateIssued,
      status: order.status,
      refillsRemaining: order.refillsRemaining,
      duration: order.duration,
      indications: order.indications
    }));

    res.json({
      success: true,
      count: queue.length,
      queue
    });
  } catch (error) {
    console.error('[SPECIALIST] Error fetching pharmacy queue:', error);
    res.status(500).json({ error: error.message });
  }
};

/**
 * POST /api/specialist/pharmacy/:requestId/dispense
 * Dispense medication to assigned nurse/clinician
 */
export const dispenseMedication = async (req, res) => {
  try {
    const { requestId } = req.params;
    const { dispensedQuantity, notes, dispensedTo } = req.body;
    const pharmacistId = req.user?._id;

    const prescription = await Prescription.findByIdAndUpdate(
      requestId,
      {
        status: 'dispensed',
        dispensedDate: new Date(),
        dispensedBy: pharmacistId,
        dispensedQuantity,
        notes
      },
      { new: true }
    ).populate('prescribedBy');

    if (!prescription) {
      return res.status(404).json({ error: 'Prescription not found' });
    }

    res.json({
      success: true,
      message: 'Medication dispensed successfully',
      prescription,
      notifyNurse: {
        nurseId: dispensedTo,
        medication: prescription.medicationName,
        quantity: dispensedQuantity,
        pickupNote: `Ready for pickup at pharmacy`
      }
    });
  } catch (error) {
    console.error('[SPECIALIST] Error dispensing medication:', error);
    res.status(500).json({ error: error.message });
  }
};

// ============================================================
// PHYSICIAN ENDPOINTS (View Results)
// ============================================================

/**
 * GET /api/specialist/physician/results
 * Get all results (radiology + pathology) for patient
 */
export const getPhysicianResults = async (req, res) => {
  try {
    const { patientId, encounterId } = req.query;
    const physicianId = req.user?._id;

    if (!physicianId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    // Verify physician is requesting for their patient or has access
    const encounter = encounterId 
      ? await Encounter.findById(encounterId)
      : null;

    if (encounterId && encounter && encounter.assignedPhysicianId.toString() !== physicianId.toString()) {
      return res.status(403).json({ error: 'Forbidden - not assigned to this patient' });
    }

    // Get radiology reports
    const RadiologyReport = require('../models/radiology.model.js').RadiologyReport;
    const radiologyReports = await RadiologyReport.find(
      patientId ? { patientId } : { encounterId }
    )
      .populate('radiologyOrderId')
      .sort({ reportDate: -1 })
      .lean();

    // Get lab results
    const LabResult = require('../models/lab.model.js').LabResult;
    const labResults = await LabResult.find(
      patientId ? { patientId } : { encounterId }
    )
      .populate('labOrderId')
      .sort({ resultDate: -1 })
      .lean();

    res.json({
      success: true,
      results: {
        radiology: radiologyReports.map(r => ({
          type: 'RADIOLOGY',
          reportId: r._id,
          orderNumber: r.radiologyOrderId?.orderNumber,
          studyType: r.radiologyOrderId?.studyType,
          findings: r.findings,
          impression: r.impression,
          recommendations: r.recommendations,
          criticalFindings: r.criticalFindings,
          reportDate: r.reportDate,
          status: r.status
        })),
        pathology: labResults.map(l => ({
          type: 'PATHOLOGY',
          resultId: l._id,
          orderNumber: l.labOrderId?.orderNumber,
          testName: l.labOrderId?.testName,
          resultValue: l.resultValue,
          resultUnit: l.resultUnit,
          referenceRange: l.referenceRange,
          isAbnormal: l.isAbnormal,
          flagSeverity: l.flagSeverity,
          interpretation: l.interpretation,
          resultDate: l.resultDate,
          status: l.approvalStatus
        }))
      }
    });
  } catch (error) {
    console.error('[SPECIALIST] Error fetching physician results:', error);
    res.status(500).json({ error: error.message });
  }
};

// ============================================================
// HELPER FUNCTIONS
// ============================================================

function calculateAge(dateOfBirth) {
  if (!dateOfBirth) return 'Unknown';
  const today = new Date();
  const birthDate = new Date(dateOfBirth);
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  
  return age;
}

function calculateEstimatedTime(urgency) {
  if (!urgency) return 'N/A';
  
  const times = {
    'stat': '30 minutes',
    'urgent': '2 hours',
    'routine': '24 hours'
  };
  
  return times[urgency.toLowerCase()] || 'N/A';
}
