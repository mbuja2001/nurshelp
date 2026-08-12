import { Encounter } from '../models/encounter.model.js';
import { Notification } from '../models/notification.model.js';

/**
 * POST /api/encounters/:id/handover
 * Create AI_TRIAGE_REPORT or MANUAL_TRIAGE_REPORT with notifications
 * This is the CRITICAL endpoint for creating the official triage report
 */
export const createHandover = async (req, res) => {
  try {
    console.log('\n\n====== HANDOVER ENDPOINT CALLED ======');
    console.log('Timestamp:', new Date().toISOString());
    console.log('Request Body keys:', Object.keys(req.body));
    
    const { id: encounterId } = req.params;
    const { 
      reportType = 'AI_TRIAGE_REPORT',  // 'AI_TRIAGE_REPORT' or 'MANUAL_TRIAGE_REPORT'
      clinicians = [],
      manuallySelected = [],
      triageNurse
    } = req.body;

    const { hospitalId: requestingHospitalId } = req.user;  // From JWT

    console.log('\n[HANDOVER] ========== INITIATING HANDOVER ==========');
    console.log('[HANDOVER] Encounter ID:', encounterId);
    console.log('[HANDOVER] Report Type:', reportType);
    console.log('[HANDOVER] Requesting Hospital:', requestingHospitalId);
    console.log('[HANDOVER] FULL REQUEST BODY TYPE:', typeof req.body);
    console.log('[HANDOVER] FULL REQUEST BODY KEYS:', Object.keys(req.body));
    console.log('[HANDOVER] FULL REQUEST BODY:', JSON.stringify(req.body, null, 2));
    console.log('[HANDOVER] === DESTRUCTURED VALUES ===');
    console.log('[HANDOVER] clinicians (from destructure):', JSON.stringify(clinicians, null, 2));
    console.log('[HANDOVER] manuallySelected (from destructure):', JSON.stringify(manuallySelected, null, 2));
    console.log('[HANDOVER] Body received:', { reportType, cliniciansCount: clinicians?.length, manuallySelectedCount: manuallySelected?.length, hasTriageNurse: !!triageNurse });
    console.log('[HANDOVER] Clinicians (AI recommended):', clinicians?.map(c => ({ id: c.id || c._id, name: c.name, role: c.role })) || []);
    console.log('[HANDOVER] Manually selected:', manuallySelected?.map(c => ({ id: c.id || c._id, name: c.name, role: c.role })) || []);
    console.log('[HANDOVER] Clinicians count:', clinicians?.length || 0);
    console.log('[HANDOVER] Manually selected count:', manuallySelected?.length || 0);
    console.log('[HANDOVER] ==========================================\n');

    // Step 1: Fetch encounter
    const encounter = await Encounter.findById(encounterId);
    if (!encounter) {
      return res.status(404).json({ error: 'Encounter not found' });
    }

    // Step 2: Verify hospital isolation (multi-tenant safety)
    // Handle both old encounters (without hospitalId) and new ones (with hospitalId)
    if (encounter.hospitalId) {
      const encounterHospitalId = encounter.hospitalId.toString();
      const requestingHospitalIdStr = String(requestingHospitalId);
      
      console.log('[HANDOVER] Comparing hospitals:');
      console.log('[HANDOVER]   Encounter hospital:', encounterHospitalId);
      console.log('[HANDOVER]   Requesting hospital:', requestingHospitalIdStr);
      
      if (encounterHospitalId !== requestingHospitalIdStr) {
        console.error('[HANDOVER] ❌ Hospital mismatch - Access Denied');
        return res.status(403).json({ 
          error: 'Access denied - encounter belongs to different hospital' 
        });
      }

      console.log('[HANDOVER] ✅ Hospital validated:', requestingHospitalId);
    } else {
      // Legacy data without hospitalId: assign it from requesting hospital
      console.log('[HANDOVER] ⚠️ Legacy encounter without hospitalId - assigning from request');
      encounter.hospitalId = requestingHospitalId;
    }

    // Step 3: Calculate validated priority
    let validated_priority = 'YELLOW';  // Default fallback
    let reportTypeUsed = 'AI';  // Default: short enum value for schema
    let fullReportType = reportType;  // Keep track of full name for audit

    // Determine which report was used and get its priority
    if (encounter.manual_triage_report?.selected_priority) {
      validated_priority = encounter.manual_triage_report.selected_priority;
      reportTypeUsed = 'MANUAL';  // Schema enum: 'MANUAL' or 'AI'
      fullReportType = 'MANUAL_TRIAGE_REPORT';
      console.log('[HANDOVER] Using manual override priority:', validated_priority);
    } else if (encounter.ai_triage_report?.calculated_priority) {
      validated_priority = encounter.ai_triage_report.calculated_priority;
      reportTypeUsed = 'AI';  // Schema enum: 'MANUAL' or 'AI'
      fullReportType = 'AI_TRIAGE_REPORT';
      console.log('[HANDOVER] Using AI calculated priority:', validated_priority);
    } else if (encounter.triage?.priority_code) {
      let priority_num = Number(encounter.triage.priority_code);
      // Map numeric priority to enum value
      const priorityMap = { 1: 'RED', 2: 'ORANGE', 3: 'YELLOW', 4: 'GREEN' };
      validated_priority = priorityMap[priority_num] || 'YELLOW';
      console.log('[HANDOVER] Using triage priority:', encounter.triage.priority_code, '→', validated_priority);
    }

    // Step 4: Mark encounter as awaiting physician review after handover
    encounter.status = 'AWAITING_PHYSICIAN';
    encounter.validated_effective_priority = validated_priority;
    encounter.handover_completed = false;

    // Step 5: Record triage nurse from JWT extraction (frontend passed it)
    if (triageNurse) {
      encounter.original_triage_nurse = {
        nurse_id: triageNurse.id,
        nurse_name: triageNurse.name,
        nurse_email: triageNurse.email,
        nurse_role: triageNurse.role,
        triaged_at: new Date()
      };
      console.log('[HANDOVER] Triage nurse recorded:', triageNurse.name, `<${triageNurse.email}>`);
    }

    // Step 6: Deduplicate clinicians (combine AI recommended + manually selected)
    const allClinicians = [...clinicians, ...manuallySelected];
    const uniqueClinicians = [];
    const seenIds = new Set();

    console.log('[HANDOVER] Merging clinicians:');
    console.log('[HANDOVER]   AI Recommended (' + clinicians.length + '):', clinicians.map(c => c.name).join(', ') || 'none');
    console.log('[HANDOVER]   Manually Selected (' + manuallySelected.length + '):', manuallySelected.map(c => c.name).join(', ') || 'none');

    for (const clinician of allClinicians) {
      const cId = clinician.id?.toString() || clinician._id?.toString();
      if (!seenIds.has(cId)) {
        uniqueClinicians.push(clinician);
        seenIds.add(cId);
      }
    }

    console.log('[HANDOVER] After deduplication:', uniqueClinicians.length, 'unique clinicians');
    console.log('[HANDOVER] Unique clinicians:', uniqueClinicians.map(c => ({ id: c.id || c._id, name: c.name, role: c.role })));

    // Step 7: Create Notification records for EACH clinician
    const notificationsSent = [];

    console.log('[HANDOVER] Starting notification creation loop for', uniqueClinicians.length, 'clinicians');

    for (const clinician of uniqueClinicians) {
      try {
        console.log('[HANDOVER] Creating notification for clinician:', clinician.name, 'ID:', clinician.id || clinician._id);
        
        // Create notification document
        const notification = await Notification.create({
          encounter_id: encounterId,
          hospital_id: encounter.hospitalId,

          clinician_id: clinician.id || clinician._id,
          clinician_name: clinician.name,
          clinician_role: clinician.role || 'PHYSICIAN',
          clinician_specialty: clinician.specialty,

          triage_nurse_id: triageNurse?.id,
          triage_nurse_name: triageNurse?.name,
          triage_nurse_email: triageNurse?.email,

          report_type: fullReportType,  // Use full report type (AI_TRIAGE_REPORT or MANUAL_TRIAGE_REPORT)
          priority_code: validated_priority,

          patient_mrn: encounter.patient?.id_number,
          patient_name: `${encounter.patient?.name || ''} ${encounter.patient?.surname || ''}`.trim(),
          chief_complaint: encounter.patient?.symptoms,

          status: 'UNREAD',
          created_at: new Date(),
          audit_log: [{
            action: 'CREATED',
            timestamp: new Date(),
            notes: `${fullReportType} from ${triageNurse?.name || 'Unknown nurse'}`
          }]
        });

        console.log('[HANDOVER] ✅ Notification created:', {
          clinician: clinician.name,
          notificationId: notification._id
        });

        notificationsSent.push({
          clinician_id: clinician.id || clinician._id,
          clinician_name: clinician.name,
          notification_id: notification._id,
          sent_at: new Date(),
          status: 'NOTIFIED'
        });

        // Add to clinical_handovers tracking array
        encounter.clinical_handovers.push({
          clinician_id: clinician.id || clinician._id,
          clinician_name: clinician.name,
          clinician_role: clinician.role || 'PHYSICIAN',
          clinician_specialty: clinician.specialty,
          handover_type: 'INITIAL',
          report_type: reportTypeUsed,  // Use short form for encounter schema (AI or MANUAL)
          handover_status: 'NOTIFIED',
          handover_at: new Date(),
          notification_id: notification._id,
          is_active: true
        });

      } catch (notifErr) {
        console.error('[HANDOVER] ❌ Error creating notification for', clinician.name);
        console.error('[HANDOVER] Error details:', notifErr.message);
        console.error('[HANDOVER] Full error:', notifErr);
        // Continue with next clinician, don't fail entire request
      }
    }

    console.log('[HANDOVER] Notification creation loop complete. Total notifications:', notificationsSent.length);

    // Step 8: Save updated encounter
    await encounter.save();

    console.log('[HANDOVER] ========== HANDOVER COMPLETE ==========');
    console.log('[HANDOVER] ✅ AI_TRIAGE_REPORT created successfully');
    console.log('[HANDOVER] Status: AWAITING_PHYSICIAN');
    console.log('[HANDOVER] Report Type:', reportTypeUsed);
    console.log('[HANDOVER] Priority:', validated_priority);
    console.log('[HANDOVER] Notifications sent:', notificationsSent.length);

    res.json({
      success: true,
      encounterId,
      reportType: reportTypeUsed,
      notificationsCreated: notificationsSent.length,
      priority: validated_priority,
      triageNurse: triageNurse?.name,
      clinicians: notificationsSent,
      timestamp: new Date()
    });

  } catch (err) {
    console.error('[HANDOVER] ❌ Error:', err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * PUT /api/encounters/:id/ai-fields
 * Save AI fields and triage nurse info before confirmation
 */
export const saveAIFields = async (req, res) => {
  try {
    const { id: encounterId } = req.params;
    const reqBody = req.body;

    // Handle both top-level and nested (changedFields) payload structures
    const changedFields = reqBody.changedFields || {};
    const assigned_clinicians = reqBody.assigned_clinicians || changedFields.assigned_clinicians || [];
    const manually_selected_clinicians = reqBody.manually_selected_clinicians || changedFields.manually_selected_clinicians || [];
    const triage = reqBody.triage || changedFields.triage;
    const nurseNotes = reqBody.nurseNotes || changedFields.nurseNotes;
    const triage_nurse = reqBody.triage_nurse;
    const manual_triage_report = reqBody.manual_triage_report || changedFields.manual_triage_report;
    
    // Extract display info for bed/ward (for UI display in confirmation page)
    const ward_display_info = reqBody.ward_display_info || changedFields.ward_display_info;
    const bed_display_info = reqBody.bed_display_info || changedFields.bed_display_info;
    const assigned_ward = reqBody.assigned_ward ?? changedFields.assigned_ward;
    const assigned_bed = reqBody.assigned_bed ?? changedFields.assigned_bed;

    const { hospitalId: requestingHospitalId } = req.user;

    console.log('[AI-FIELDS] Saving for encounter:', encounterId);
    console.log('[AI-FIELDS] Received payload structure:', Object.keys(reqBody));
    console.log('[AI-FIELDS] Assigned clinicians:', assigned_clinicians?.length || 0);
    console.log('[AI-FIELDS] Manually selected clinicians:', manually_selected_clinicians?.length || 0);

    const encounter = await Encounter.findByIdAndUpdate(
      encounterId,
      {
        $set: {
          ...(triage && { triage }),
          assigned_clinicians,
          manually_selected_clinicians,
          ...(nurseNotes && { nurseNotes }),
          ...(manual_triage_report && { manual_triage_report }),
          ...(ward_display_info && { ward_display_info }),
          ...(bed_display_info && { bed_display_info }),
          ...(assigned_ward !== undefined && { assigned_ward }),
          ...(assigned_bed !== undefined && { assigned_bed }),
          // Save triage nurse if provided
          ...(triage_nurse && {
            original_triage_nurse: {
              nurse_id: triage_nurse.nurse_id,
              nurse_name: triage_nurse.nurse_name,
              nurse_email: triage_nurse.nurse_email,
              nurse_role: triage_nurse.nurse_role,
              triaged_at: new Date(triage_nurse.triaged_at)
            }
          })
        }
      },
      { new: true, runValidators: false }
    );

    if (!encounter) {
      return res.status(404).json({ error: 'Encounter not found' });
    }

    console.log('[AI-FIELDS] ✅ Fields saved');
    console.log('[AI-FIELDS] Triage nurse:', triage_nurse?.nurse_name);
    console.log('[AI-FIELDS] Clinicians assigned:', assigned_clinicians?.length || 0);
    console.log('[AI-FIELDS] Manually selected:', manually_selected_clinicians?.length || 0);
    console.log('[AI-FIELDS] Manual report saved:', !!manual_triage_report);
    console.log('[AI-FIELDS] Ward display info saved:', ward_display_info?.name || 'none');
    console.log('[AI-FIELDS] Bed display info saved:', bed_display_info?.bedNumber || 'none');

    res.json({
      success: true,
      encounterId,
      encounter
    });

  } catch (err) {
    console.error('[AI-FIELDS] Error:', err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * GET /api/encounters/:id/handover-status
 * Get the current status and history of handover
 */
export const getHandoverStatus = async (req, res) => {
  try {
    const { id: encounterId } = req.params;

    const encounter = await Encounter.findById(encounterId)
      .select({
        'status': 1,
        'validated_effective_priority': 1,
        'original_triage_nurse': 1,
        'clinical_handovers': 1,
        'createdAt': 1,
        'updatedAt': 1
      })
      .populate('clinical_handovers.clinician_id', 'name role specialty')
      .lean();

    if (!encounter) {
      return res.status(404).json({ error: 'Encounter not found' });
    }

    console.log('[HANDOVER-STATUS] Encounter:', encounterId, '| Status:', encounter.status);

    res.json({
      success: true,
      encounterId,
      status: encounter.status,
      priority: encounter.validated_effective_priority,
      triageNurse: encounter.original_triage_nurse,
      clinicalHandovers: encounter.clinical_handovers,
      createdAt: encounter.createdAt,
      updatedAt: encounter.updatedAt
    });

  } catch (err) {
    console.error('[HANDOVER-STATUS] Error:', err);
    res.status(500).json({ error: err.message });
  }
};
