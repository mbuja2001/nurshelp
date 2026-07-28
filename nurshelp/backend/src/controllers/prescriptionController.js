/**
 * Prescription Controller
 * Handles formal prescription creation, retrieval, and status tracking
 * Complies with South African Medicines and Related Substances Act
 */

import { Prescription } from '../models/pharmacy.model.js';
import { Encounter } from '../models/encounter.model.js';

// ============ CREATE FORMAL PRESCRIPTION ============
export const createPrescription = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const user = req.user;
    const { encounterId, items, special_instructions, contraindications } = req.body;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    if (!encounterId || !items || !Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ success: false, message: 'encounterId and items array are required' });
    }

    // Fetch encounter and patient data
    const encounter = await Encounter.findOne({ _id: encounterId, hospitalId });
    if (!encounter) {
      return res.status(404).json({ success: false, message: 'Encounter not found' });
    }

    const prescriberData = req.body.prescriber || {};
    const patientDetailsFromBody = req.body.patient_details || {};
    const patientId = req.body.patientId || encounter.patient?.id || encounter.patient?._id || encounter._id;
    const physicianId = req.body.physicianId || user?._id || user?.id || 'unknown';

    // Calculate valid until date (30 days for Schedule 3-4, varies by schedule)
    const dateIssued = req.body.date_issued ? new Date(req.body.date_issued) : new Date();
    const validUntil = req.body.valid_until ? new Date(req.body.valid_until) : new Date(dateIssued);
    if (!req.body.valid_until) {
      validUntil.setDate(validUntil.getDate() + 30);
    }

    // Create prescription
    const prescription = new Prescription({
      hospitalId,
      encounterId,
      patientId,
      physicianId,
      medicationId: req.body.medicationId || null,
      medicationName: req.body.medicationName || prescriberData?.medication_name || '',

      // Prescriber details
      prescriber: {
        id: user._id,
        name: prescriberData?.name || user.name,
        email: prescriberData?.email || user.email,
        role: prescriberData?.role || user.role || 'PHYSICIAN',
        specialty: prescriberData?.specialty || user.specialty || 'General',
        hpcsa_number: user.hpcsa_number || 'N/A',
        practice_number: user.practice_number || 'N/A',
        practice_address: user.practice_address || 'Hospital Address',
        phone: user.phone || 'N/A'
      },

      // Patient details
      patient_details: {
        name: patientDetailsFromBody?.name || encounter.patient?.name || 'Patient',
        id_number: patientDetailsFromBody?.id_number || encounter.patient?.id_number || 'N/A',
        date_of_birth: patientDetailsFromBody?.date_of_birth || encounter.patient?.date_of_birth || null,
        age: patientDetailsFromBody?.age || encounter.patient?.age || null,
        address: patientDetailsFromBody?.address || encounter.patient?.address || 'N/A',
        medical_aid: patientDetailsFromBody?.medical_aid || encounter.patient?.medical_aid || 'N/A',
        medical_aid_number: patientDetailsFromBody?.medical_aid_number || encounter.patient?.medical_aid_number || 'N/A'
      },

      // Prescription items
      items: items.map(item => ({
        medicationId: item.medicationId || null,
        medication_name: item.medication_name,
        strength: item.dosage || item.strength || '',
        form: item.form || 'Tablets',
        schedule: item.schedule || 'OTC',
        quantity: item.quantity || 1,
        unit: item.unit || 'Tablets',
        dosage: item.dosage || '',
        frequency: item.frequency || '',
        route: item.route || 'Oral',
        duration: item.duration || '7 days',
        directions: `Take ${item.dosage || '1'} ${item.route || 'orally'} ${item.frequency || 'as needed'} for ${item.duration || '7 days'}`,
        refills: item.refills || 0,
        indication: item.indication || ''
      })),

      // Configuration
      substitution_permitted: req.body.substitution_permitted !== false,
      special_instructions: special_instructions || '',
      contraindications: contraindications || '',
      date_issued: dateIssued,
      valid_until: validUntil,
      status: 'ISSUED',
      submitted_to_pharmacy: false
    });

    await prescription.save();

    encounter.medication_requests = [
      ...(encounter.medication_requests || []),
      {
        medication_name: prescription.medicationName || prescription.items?.[0]?.medication_name || 'Medication',
        dosage: prescription.items?.[0]?.dosage || prescription.dosage || '',
        frequency: prescription.items?.[0]?.frequency || prescription.frequency || '',
        route: prescription.items?.[0]?.route || prescription.route || '',
        indication: prescription.items?.[0]?.indication || prescription.special_instructions || '',
        requested_by: prescriberData?.name || user.name || 'Physician',
        status: 'PENDING',
        requested_at: new Date(),
        prescriptionId: prescription._id
      }
    ];
    await encounter.save();

    console.log(`[PRESCRIPTION] Created prescription ${prescription.prescription_number} for encounter ${encounterId}`);

    res.status(201).json({
      success: true,
      prescription,
      prescription_number: prescription.prescription_number
    });
  } catch (err) {
    console.error('[PRESCRIPTION CREATE ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to create prescription', error: err.message });
  }
};

// ============ GET PRESCRIPTION BY ID ============
export const getPrescription = async (req, res) => {
  console.log('[PRESCRIPTION GET] Handler invoked', { path: req.path, params: req.params, query: req.query });
  try {
    const hospitalId = req.hospitalId;
    const { prescriptionId } = req.params;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    const prescription = await Prescription.findOne({ _id: prescriptionId, hospitalId });
    if (!prescription) {
      return res.status(404).json({ success: false, message: 'Prescription not found' });
    }

    res.status(200).json({ success: true, prescription });
  } catch (err) {
    console.error('[PRESCRIPTION GET ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to fetch prescription', error: err.message });
  }
};

// ============ GET PRESCRIPTIONS BY ENCOUNTER ============
export const getPrescriptionsByEncounter = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { encounterId } = req.params;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    const prescriptions = await Prescription.find({ hospitalId, encounterId }).sort({ createdAt: -1 });

    res.status(200).json({ success: true, prescriptions });
  } catch (err) {
    console.error('[PRESCRIPTION GET ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to fetch prescriptions', error: err.message });
  }
};

// ============ GET ORDERED MEDS FOR PHYSICIAN ============
export const getOrderedMedsForPhysician = async (req, res) => {
  console.log('[PRESCRIPTION ORDERED MEDS] Handler invoked', { path: req.path, params: req.params, query: req.query });
  try {
    console.log('[PRESCRIPTION ORDERED MEDS] Query:', req.query);
    console.log('[PRESCRIPTION ORDERED MEDS] User:', req.user?._id, req.user?.email, req.user?.role);
    const hospitalId = req.hospitalId;
    const userId = req.user?._id || req.user?.id;
    const { patientId, startDate, endDate, countOnly } = req.query;
    const countOnlyMode = String(countOnly).toLowerCase() === 'true';

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    if (!userId) {
      return res.status(401).json({ success: false, message: 'Unauthorized: physician identity missing' });
    }

    const filter = { hospitalId, physicianId: String(userId) };

    if (patientId) {
      filter.$or = [
        { 'patient_details.id_number': patientId },
        { patientId }
      ];
    }

    if (startDate || endDate) {
      filter.date_issued = {};
      if (startDate) {
        const start = new Date(startDate);
        start.setHours(0, 0, 0, 0);
        filter.date_issued.$gte = start;
      }
      if (endDate) {
        const end = new Date(endDate);
        end.setHours(23, 59, 59, 999);
        filter.date_issued.$lte = end;
      }
    }

    if (countOnlyMode) {
      const count = await Prescription.countDocuments(filter);
      return res.status(200).json({ success: true, count });
    }

    const prescriptions = await Prescription.find(filter)
      .select('patient_details patientId encounterId date_issued dateIssued valid_until expiryDate status items prescription_number')
      .sort({ date_issued: -1, createdAt: -1 })
      .lean();

    const formatted = prescriptions.map((p) => ({
      _id: p._id,
      patientId: p.patient_details?.id_number || p.patientId || null,
      patient_name: p.patient_details?.name || '',
      encounterId: p.encounterId,
      status: p.status,
      date_issued: p.date_issued || p.dateIssued,
      valid_until: p.valid_until || p.expiryDate,
      items: p.items || [],
      prescription_number: p.prescription_number || null
    }));

    res.status(200).json({ success: true, count: formatted.length, prescriptions: formatted });
  } catch (err) {
    console.error('[PRESCRIPTION ORDERED MEDS ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to fetch ordered medications', error: err.message });
  }
};

// ============ SUBMIT PRESCRIPTION TO PHARMACY ============
export const submitPrescriptionToPharmacy = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { prescriptionId } = req.params;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    const prescription = await Prescription.findOne({ _id: prescriptionId, hospitalId });
    if (!prescription) {
      return res.status(404).json({ success: false, message: 'Prescription not found' });
    }

    prescription.status = 'SUBMITTED';
    prescription.submitted_at = new Date();
    prescription.submitted_to_pharmacy = true;

    await prescription.save();

    console.log(`[PRESCRIPTION SUBMIT] Prescription ${prescription.prescription_number} submitted to pharmacy`);

    res.status(200).json({ success: true, prescription });
  } catch (err) {
    console.error('[PRESCRIPTION SUBMIT ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to submit prescription', error: err.message });
  }
};

// ============ UPDATE PRESCRIPTION ITEM STATUS ============
export const updatePrescriptionItemStatus = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { prescriptionId, itemId } = req.params;
    const { status, accepted_at, estimated_received } = req.body;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    if (!['PENDING', 'ACCEPTED', 'DISPENSED', 'NOT_RECEIVED'].includes(status)) {
      return res.status(400).json({ success: false, message: 'Invalid status' });
    }

    const prescription = await Prescription.findOne({ _id: prescriptionId, hospitalId });
    if (!prescription) {
      return res.status(404).json({ success: false, message: 'Prescription not found' });
    }

    const item = prescription.items.id(itemId);
    if (!item) {
      return res.status(404).json({ success: false, message: 'Prescription item not found' });
    }

    item.status = status;
    if (accepted_at) item.accepted_at = new Date(accepted_at);
    if (estimated_received) item.estimated_received = new Date(estimated_received);

    // Check if all items are accepted
    const allAccepted = prescription.items.every(i => i.status === 'ACCEPTED' || i.status === 'DISPENSED');
    prescription.all_accepted = allAccepted;

    // Check if all items are dispensed
    const allDispensed = prescription.items.every(i => i.status === 'DISPENSED');
    prescription.all_dispensed = allDispensed;

    if (allAccepted) {
      prescription.status = 'ACCEPTED';
    }
    if (allDispensed) {
      prescription.status = 'DISPENSED';
    }

    await prescription.save();

    console.log(`[PRESCRIPTION ITEM UPDATE] Item ${itemId} status updated to ${status}`);

    res.status(200).json({ success: true, prescription });
  } catch (err) {
    console.error('[PRESCRIPTION ITEM UPDATE ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to update prescription item', error: err.message });
  }
};

// ============ CANCEL PRESCRIPTION ============
export const cancelPrescription = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { prescriptionId } = req.params;
    const { cancellation_reason } = req.body;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    const prescription = await Prescription.findOne({ _id: prescriptionId, hospitalId });
    if (!prescription) {
      return res.status(404).json({ success: false, message: 'Prescription not found' });
    }

    prescription.status = 'CANCELLED';
    prescription.cancelled_by = req.user._id;
    prescription.cancelled_at = new Date();
    prescription.cancellation_reason = cancellation_reason || 'No reason provided';

    await prescription.save();

    console.log(`[PRESCRIPTION CANCEL] Prescription ${prescription.prescription_number} cancelled`);

    res.status(200).json({ success: true, prescription });
  } catch (err) {
    console.error('[PRESCRIPTION CANCEL ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to cancel prescription', error: err.message });
  }
};
