import { Encounter } from '../models/encounter.model.js';
import { Notification } from '../models/notification.model.js';

/**
 * Search for patients in hospital (current user's hospital only)
 * Hospital isolation enforced via JWT hospitalId
 */
export const searchHospitalPatients = async (req, res) => {
  try {
    const { hospitalId } = req.user;  // From JWT token
    const { search, limit = 50, skip = 0 } = req.query;

    if (!hospitalId) {
      return res.status(401).json({ error: 'Hospital ID not found in token' });
    }

    let query = { hospitalId };

    // If search provided, search by MRN or name
    if (search && search.trim()) {
      query.$or = [
        { 'patient.id_number': { $regex: search, $options: 'i' } },
        { 'patient.name': { $regex: search, $options: 'i' } },
        { 'patient.surname': { $regex: search, $options: 'i' } }
      ];
      console.log('[SEARCH] Query term:', search);
    }

    // Fetch encounters for this hospital
    const encounters = await Encounter.find(query)
      .populate('hospitalId', 'name code')
      .populate('original_triage_nurse', 'name email role')
      .populate('assigned_clinicians', 'name role specialty')
      .select({
        'patient': 1,
        'status': 1,
        'validated_effective_priority': 1,
        'original_triage_nurse': 1,
        'assigned_clinicians': 1,
        'clinical_handovers': 1,
        'createdAt': 1,
        'updatedAt': 1
      })
      .sort({ createdAt: -1 })
      .skip(parseInt(skip))
      .limit(parseInt(limit))
      .lean();

    const total = await Encounter.countDocuments(query);

    console.log('[SEARCH] Hospital:', hospitalId, '| Found:', encounters.length, 'of', total);

    res.json({
      success: true,
      total,
      count: encounters.length,
      skip: parseInt(skip),
      limit: parseInt(limit),
      hospitalId,
      encounters
    });

  } catch (err) {
    console.error('[SEARCH] Error:', err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * Get complete EHR for patient by MRN
 * Hospital-isolated: only shows encounters from requesting user's hospital
 */
export const getPatientEHR = async (req, res) => {
  try {
    const { hospitalId } = req.user;  // From JWT
    const { mrn } = req.params;

    if (!hospitalId) {
      return res.status(401).json({ error: 'Hospital ID not found in token' });
    }

    if (!mrn) {
      return res.status(400).json({ error: 'MRN parameter required' });
    }

    console.log('[EHR] Retrieving for MRN:', mrn, 'from hospital:', hospitalId);

    // Query MUST include hospitalId for isolation
    const patientEncounters = await Encounter.find({
      hospitalId,
      'patient.id_number': mrn
    })
      .populate('hospitalId', 'name code country')
      .populate('original_triage_nurse', 'name email role')
      .populate('assigned_clinicians', 'name role specialty')
      .populate('assigned_ward', 'wardName department')
      .populate('assigned_bed', 'bedId bedNumber roomNumber')
      .populate('clinical_handovers.clinician_id', 'name role specialty')
      .sort({ createdAt: -1 })
      .lean();

    if (!patientEncounters || patientEncounters.length === 0) {
      console.log('[EHR] No encounters found for MRN:', mrn, 'in hospital:', hospitalId);
      return res.status(404).json({ error: 'Patient not found in your hospital' });
    }

    console.log('[EHR] Found', patientEncounters.length, 'encounters for patient');

    res.json({
      success: true,
      mrn,
      hospital: patientEncounters[0].hospitalId,
      totalEncounters: patientEncounters.length,
      encounters: patientEncounters
    });

  } catch (err) {
    console.error('[EHR] Error:', err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * Get all AI_TRIAGE_REPORT encounters for a patient
 * Shows triage report history with priorities and outcomes
 */
export const getPatientTriageReports = async (req, res) => {
  try {
    const { hospitalId } = req.user;  // From JWT
    const { mrn } = req.params;

    if (!hospitalId) {
      return res.status(401).json({ error: 'Hospital ID not found in token' });
    }

    if (!mrn) {
      return res.status(400).json({ error: 'MRN parameter required' });
    }

    console.log('[TRIAGE-REPORTS] Retrieving for MRN:', mrn);

    // Find all encounters with OFFICIAL_REPORT status for this patient
    const triageReports = await Encounter.find({
      hospitalId,
      'patient.id_number': mrn,
      status: 'OFFICIAL_REPORT'
    })
      .select({
        'patient': 1,
        'original_triage_nurse': 1,
        'validated_effective_priority': 1,
        'ai_triage_report': 1,
        'manual_triage_report': 1,
        'clinical_handovers': 1,
        'vitals': 1,
        'status': 1,
        'createdAt': 1,
        'updatedAt': 1
      })
      .populate('original_triage_nurse', 'name email role')
      .populate('clinical_handovers.clinician_id', 'name role specialty')
      .sort({ createdAt: -1 })
      .lean();

    console.log('[TRIAGE-REPORTS] Found', triageReports.length, 'reports');

    res.json({
      success: true,
      mrn,
      triageReportCount: triageReports.length,
      reports: triageReports
    });

  } catch (err) {
    console.error('[TRIAGE-REPORTS] Error:', err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * Get all clinical handover records for a patient
 * Shows clinician notifications and acknowledgments timeline
 */
export const getPatientHandoverTimeline = async (req, res) => {
  try {
    const { hospitalId } = req.user;  // From JWT
    const { mrn } = req.params;

    if (!hospitalId) {
      return res.status(401).json({ error: 'Hospital ID not found in token' });
    }

    console.log('[HANDOVER-TIMELINE] Retrieving for MRN:', mrn);

    // Aggregate all handovers for patient across all encounters
    const timeline = await Encounter.aggregate([
      {
        $match: {
          hospitalId,
          'patient.id_number': mrn
        }
      },
      {
        $unwind: '$clinical_handovers'
      },
      {
        $sort: { 'clinical_handovers.handover_at': -1 }
      },
      {
        $group: {
          _id: null,
          total: { $sum: 1 },
          handovers: { $push: '$clinical_handovers' }
        }
      }
    ]);

    const total = timeline.length > 0 ? timeline[0].total : 0;
    const handovers = timeline.length > 0 ? timeline[0].handovers : [];

    console.log('[HANDOVER-TIMELINE] Found', total, 'handover records');

    res.json({
      success: true,
      mrn,
      totalHandovers: total,
      handovers
    });

  } catch (err) {
    console.error('[HANDOVER-TIMELINE] Error:', err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * Get all notifications for a patient's encounters
 * Useful for following up on clinician responses
 */
export const getPatientNotifications = async (req, res) => {
  try {
    const { hospitalId } = req.user;  // From JWT
    const { mrn } = req.params;

    if (!hospitalId) {
      return res.status(401).json({ error: 'Hospital ID not found in token' });
    }

    console.log('[PATIENT-NOTIFICATIONS] Retrieving for MRN:', mrn);

    // First get all encounters for this patient
    const encounters = await Encounter.find({
      hospitalId,
      'patient.id_number': mrn
    }).select('_id').lean();

    const encounterIds = encounters.map(e => e._id);

    // Then get all notifications for those encounters
    const notifications = await Notification.find({
      encounter_id: { $in: encounterIds },
      hospital_id: hospitalId
    })
      .populate('clinician_id', 'name role specialty email')
      .populate('triage_nurse_id', 'name email')
      .sort({ created_at: -1 })
      .lean();

    console.log('[PATIENT-NOTIFICATIONS] Found', notifications.length, 'notifications');

    res.json({
      success: true,
      mrn,
      notificationCount: notifications.length,
      notifications
    });

  } catch (err) {
    console.error('[PATIENT-NOTIFICATIONS] Error:', err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * Get summary stats for a patient
 * Quick overview of all encounters and status
 */
export const getPatientSummary = async (req, res) => {
  try {
    const { hospitalId } = req.user;  // From JWT
    const { mrn } = req.params;

    if (!hospitalId) {
      return res.status(401).json({ error: 'Hospital ID not found in token' });
    }

    console.log('[PATIENT-SUMMARY] Retrieving for MRN:', mrn);

    // Get basic info
    const encounters = await Encounter.find({
      hospitalId,
      'patient.id_number': mrn
    })
      .select('status validated_effective_priority original_triage_nurse clinical_handovers')
      .lean();

    const totalEncounters = encounters.length;
    const officialReports = encounters.filter(e => e.status === 'OFFICIAL_REPORT').length;
    const discharged = encounters.filter(e => e.status === 'DISCHARGED').length;
    const hospitalized = encounters.filter(e => e.status === 'HOSPITALIZED').length;

    let highestPriority = null;
    encounters.forEach(e => {
      if (e.validated_effective_priority) {
        if (!highestPriority || e.validated_effective_priority > highestPriority) {
          highestPriority = e.validated_effective_priority;
        }
      }
    });

    let totalHandovers = 0;
    let acknowledgedHandovers = 0;
    encounters.forEach(e => {
      if (e.clinical_handovers) {
        totalHandovers += e.clinical_handovers.length;
        acknowledgedHandovers += e.clinical_handovers.filter(h => h.handover_status === 'ACKNOWLEDGED').length;
      }
    });

    const summary = {
      mrn,
      totalEncounters,
      officialReports,
      discharged,
      hospitalized,
      highestPriority,
      totalHandovers,
      acknowledgedHandovers: acknowledgedHandovers,
      acknowledgmentRate: totalHandovers > 0 ? Math.round((acknowledgedHandovers / totalHandovers) * 100) : 0
    };

    console.log('[PATIENT-SUMMARY] Stats:', summary);

    res.json({
      success: true,
      summary
    });

  } catch (err) {
    console.error('[PATIENT-SUMMARY] Error:', err);
    res.status(500).json({ error: err.message });
  }
};
