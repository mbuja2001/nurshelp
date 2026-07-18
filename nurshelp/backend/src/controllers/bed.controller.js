import { BedSchedule } from "../models/bedSchedule.model.js";
import { Ward } from "../models/ward.model.js";
import { BedStatusHistory } from "../models/bedStatusHistory.model.js";
import { Bed } from "../models/bed.model.js";
import { LabOrder, LabEquipment } from "../models/lab.model.js";
import { buildOccupantSummary } from "../utils/bedOccupantSummary.js";

// =====================================================
// BATCH BED CREATION (Hospital Registration)
// =====================================================

/**
 * POST /api/admin/beds/batch
 * Create multiple beds during hospital registration
 * Protected: Manager/Admin only
 */
export const createBedsBatch = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { beds } = req.body;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found in request'
      });
    }

    if (!beds || !Array.isArray(beds) || beds.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Beds array is required and must not be empty'
      });
    }

    // Convert hospitalId from string to ObjectId for database queries
    const { ObjectId } = await import('mongodb');
    let hospitalIdForQuery;
    try {
      hospitalIdForQuery = new ObjectId(hospitalId);
      console.log(`[BED BATCH CREATE] Converted hospitalId to ObjectId: ${hospitalIdForQuery}`);
    } catch (err) {
      console.error(`[BED BATCH CREATE] Invalid hospitalId format: ${hospitalId}`);
      return res.status(400).json({
        success: false,
        message: 'Invalid hospital context. Please log out and log in again.'
      });
    }

    console.log(`[BED BATCH CREATE] Creating ${beds.length} beds for hospital ${hospitalId}`);

    // Add hospitalId to each bed
    const bedsWithHospitalId = beds.map((bed, idx) => ({
      bed_id: bed.bed_id || `BED-${idx + 1}`,
      ward_unit: bed.ward_unit,
      sub_unit: bed.sub_unit,
      status: 'Available',
      infection_control_tier: bed.infection_control_tier || 'Standard',
      equipment_baseline: bed.equipment_baseline || [],
      special_characteristics: bed.special_characteristics || [],
      compatibility_tags: bed.compatibility_tags || {
        age_group: 'All',
        acuity_level: 'All',
        isolation_capable: false,
        bariatric_capable: false,
      },
      hospitalId,
      createdBy: req.user._id,
    }));

    // Check for duplicates in request
    const bedIds = bedsWithHospitalId.map(b => b.bed_id);
    const duplicates = bedIds.filter((id, idx) => bedIds.indexOf(id) !== idx);
    if (duplicates.length > 0) {
      return res.status(400).json({
        success: false,
        message: `Duplicate bed IDs in request: ${[...new Set(duplicates)].join(', ')}`
      });
    }

    // Check for existing bed IDs in this hospital
    const existingBeds = await Bed.find({
      hospitalId: hospitalIdForQuery,
      bed_id: { $in: bedIds }
    });

    if (existingBeds.length > 0) {
      return res.status(400).json({
        success: false,
        message: `Bed IDs already exist: ${existingBeds.map(b => b.bed_id).join(', ')}`
      });
    }

    // Create beds
    const createdBeds = await Bed.insertMany(bedsWithHospitalId);

    console.log(`[BED BATCH CREATE] Successfully created ${createdBeds.length} beds`);

    res.status(201).json({
      success: true,
      message: `${createdBeds.length} beds created successfully`,
      count: createdBeds.length,
      beds: createdBeds
    });
  } catch (err) {
    console.error('[BED BATCH CREATE ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to create beds',
      error: err.message
    });
  }
};

// =====================================================
// GET WARD DETAILS WITH BED STATUS
// =====================================================

/**
 * GET /api/beds/wards
 * Get all wards with current bed status summary
 */
export const getWardsWithBedStatus = async (req, res) => {
  try {
    const { department } = req.query;

    const query = department ? { department, isActive: true } : { isActive: true };

    const wards = await Ward.find(query)
      .sort({ department: 1, wardId: 1 })
      .lean();

    // Enrich with real-time bed status
    const wardsWithStatus = await Promise.all(
      wards.map(async (ward) => {
        const beds = await BedSchedule.find({ wardId: ward._id }).lean();

        const statusCounts = {
          available: beds.filter(b => b.status === "AVAILABLE").length,
          occupied: beds.filter(b => b.status === "OCCUPIED").length,
          cleaning: beds.filter(b => b.status === "CLEANING").length,
          maintenance: beds.filter(b => b.status === "MAINTENANCE").length
        };

        const occupancyRate = ward.totalBeds > 0 
          ? Math.round((statusCounts.occupied / ward.totalBeds) * 100) 
          : 0;

        return {
          ...ward,
          currentStatus: statusCounts,
          occupancyRate,
          totalBeds: ward.totalBeds,
          bedsLayout: beds.map(b => ({
            _id: b._id,
            bedId: b.bedId,
            status: b.status,
            bedType: b.bedType,
            acuityLevel: b.acuityLevel,
            occupancyStartTime: b.occupancyStartTime,
            expectedDischargeTime: b.expectedDischargeTime,
            statusChangedAt: b.statusChangedAt,
            estimatedReadyTime: b.estimatedReadyTime
          }))
        };
      })
    );

    res.json({
      success: true,
      data: wardsWithStatus,
      timestamp: new Date()
    });

  } catch (error) {
    console.error("[BEDS] Error fetching wards:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch ward data",
      message: error.message
    });
  }
};

/**
 * GET /api/beds/ward/:wardId
 * Get detailed view of a specific ward with all beds
 */
export const getWardDetail = async (req, res) => {
  try {
    const { wardId } = req.params;

    // Find ward
    const ward = await Ward.findById(wardId);
    if (!ward) {
      return res.status(404).json({
        success: false,
        error: "Ward not found"
      });
    }

    // Get all beds in ward with full details
    const beds = await BedSchedule.find({ wardId })
      .populate("currentEncounterId", "patient triage diagnosis severity status assignedPhysicianName assignedNurseName")
      .sort({ bedId: 1 })
      .lean();

    // Normalize bed data - convert statuses to titlecase and add proper mapping
    const normalizedBeds = beds.map(bed => {
      const occupantSummary = buildOccupantSummary(bed, bed.currentEncounterId);

      return {
        ...bed,
        status: bed.status ? bed.status.charAt(0) + bed.status.slice(1).toLowerCase() : 'Available',
        bedId: bed.bedId || bed._id?.toString(),
        bedType: bed.bedType || 'Standard',
        currentEncounterId: occupantSummary,
        lastCleanedAt: bed.lastCleanedAt || new Date(),
        acuityLevel: bed.acuityLevel || 'MODERATE',
        occupancyStartTime: bed.occupancyStartTime || null,
        expectedDischargeTime: bed.expectedDischargeTime || null,
        maintenanceNotes: bed.maintenanceNotes || '',
        canIsolate: bed.canIsolate || false,
        hasMonitor: bed.hasMonitor || false,
        hasVentilator: bed.hasVentilator || false,
        hasDialysis: bed.hasDialysis || false
      };
    });

    // Calculate metrics
    const statusCounts = {
      available: normalizedBeds.filter(b => b.status === "Available").length,
      occupied: normalizedBeds.filter(b => b.status === "Occupied").length,
      cleaning: normalizedBeds.filter(b => b.status === "Cleaning").length,
      maintenance: normalizedBeds.filter(b => b.status === "Maintenance").length
    };

    res.json({
      success: true,
      data: {
        ward,
        beds: normalizedBeds,
        statusCounts,
        occupancyRate: Math.round((statusCounts.occupied / (ward.totalBeds || normalizedBeds.length || 1)) * 100),
        timestamp: new Date()
      }
    });

  } catch (error) {
    console.error("[BEDS] Error fetching ward detail:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch ward detail",
      message: error.message
    });
  }
};

/**
 * GET /api/beds/hospital-stats
 * Get comprehensive bed statistics for entire hospital (for dashboard)
 */
export const getHospitalBedStats = async (req, res) => {
  try {
    const { ObjectId } = await import('mongodb');
    const hospitalIdParam = req.query.hospitalId || req.hospitalId;
    
    if (!hospitalIdParam) {
      return res.status(400).json({ success: false, message: 'Hospital ID required' });
    }

    // Convert to ObjectId
    let hospitalId;
    try {
      hospitalId = new ObjectId(hospitalIdParam);
    } catch (err) {
      console.error('[BEDS] Invalid hospitalId format:', hospitalIdParam, err);
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid hospital ID format',
        hospitalId: hospitalIdParam
      });
    }

    console.log('[BEDS STATS] Fetching stats for hospital:', hospitalId);

    // Get all wards for this hospital
    const wards = await Ward.find({ hospitalId }).lean();
    console.log('[BEDS STATS] Found wards:', wards.length);
    
    if (wards.length === 0) {
      console.warn('[BEDS STATS] No wards found for hospital:', hospitalId);
      return res.json({
        success: true,
        data: {
          total: 0,
          available: 0,
          occupied: 0,
          cleaning: 0,
          maintenance: 0,
          occupancyRate: 0,
          bedsByWard: []
        },
        timestamp: new Date()
      });
    }

    const wardIds = wards.map(w => w._id);

    // Get all beds
    const allBeds = await BedSchedule.find({ wardId: { $in: wardIds } })
      .populate('wardId', 'wardName')
      .lean();
    
    console.log('[BEDS STATS] Found beds:', allBeds.length);

    // Normalize and calculate stats
    const normalizedBeds = allBeds.map(bed => ({
      ...bed,
      status: bed.status ? bed.status.charAt(0) + bed.status.slice(1).toLowerCase() : 'Available'
    }));

    const stats = {
      total: allBeds.length,
      available: normalizedBeds.filter(b => b.status === 'Available').length,
      occupied: normalizedBeds.filter(b => b.status === 'Occupied').length,
      cleaning: normalizedBeds.filter(b => b.status === 'Cleaning').length,
      maintenance: normalizedBeds.filter(b => b.status === 'Maintenance').length,
      occupancyRate: Math.round((normalizedBeds.filter(b => b.status === 'Occupied').length / (allBeds.length || 1)) * 100),
      bedsByWard: wards.map(ward => ({
        _id: ward._id,
        wardName: ward.wardName,
        totalBeds: ward.totalBeds || 0,
        beds: normalizedBeds.filter(b => b.wardId?._id?.toString() === ward._id.toString() || b.wardId === ward._id)
      }))
    };

    console.log('[BEDS STATS] Calculated stats:', stats);

    res.json({
      success: true,
      data: stats,
      timestamp: new Date()
    });
  } catch (error) {
    console.error('[BEDS STATS] Error fetching hospital stats:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch bed statistics',
      error: error.message
    });
  }
};

/**
 * GET /api/beds/live-status
 * Real-time live status of all beds (for dashboard)
 */
export const getLiveBedStatus = async (req, res) => {
  try {
    const { department, wardId, bedId } = req.query;

    const query = {};
    if (department) query.department = department;
    if (wardId) query.wardId = wardId;

    const beds = await BedSchedule.find(query)
      .populate("wardId", "wardName wardId department")
      .populate("currentEncounterId", "patient_id patientName symptoms")
      .lean();

    // Compute equipment usage counts from LabOrders (best-effort)
    let equipmentUsageMap = {};
    try {
      const usageAgg = await LabOrder.aggregate([
        { $match: { equipmentUsed: { $exists: true, $ne: null, $ne: "" } } },
        { $group: { _id: '$equipmentUsed', count: { $sum: 1 } } }
      ]);
      equipmentUsageMap = usageAgg.reduce((acc, item) => {
        acc[item._id] = item.count;
        return acc;
      }, {});
    } catch (err) {
      console.warn('[BEDS] Failed to compute equipment usage aggregation:', err.message);
      equipmentUsageMap = {};
    }

    const buildEquipmentForBed = async (bed) => {
      try {
        const bedRecord = await Bed.findOne({
          $or: [
            { bed_id: bed.bedId },
            { bed_id: bed.bedId?.toString?.() },
            { bed_id: bed.bedId?.trim?.() }
          ]
        }).lean().catch(() => null);

        const baseline = Array.isArray(bedRecord?.equipment_baseline) ? bedRecord.equipment_baseline : [];
        const equipmentObjects = [];

        const addEquipment = (name, status = 'configured', usageCount = 0) => {
          if (!name) return;
          const normalizedName = name.toLowerCase();
          const existing = equipmentObjects.find((item) => item.name.toLowerCase() === normalizedName);
          if (existing) {
            existing.status = existing.status === 'unknown' ? status : existing.status;
            existing.usageCount = existing.usageCount || usageCount;
            return;
          }
          equipmentObjects.push({ name, usageCount, status });
        };

        if (bed.hasMonitor) addEquipment('Cardiac Monitor', 'active', equipmentUsageMap['Cardiac Monitor'] || 0);
        if (bed.hasVentilator) addEquipment('Mechanical Ventilator', 'active', equipmentUsageMap['Mechanical Ventilator'] || 0);
        if (bed.hasDialysis) addEquipment('Dialysis Machine', 'active', equipmentUsageMap['Dialysis Machine'] || 0);
        if (bed.canIsolate) addEquipment('Isolation Setup', 'ready', 0);

        const baselineEntries = await Promise.all(baseline.map(async (eq) => {
          const eqStatus = await LabEquipment.findOne({ equipmentName: eq }).lean().catch(() => null);
          return {
            name: eq,
            usageCount: equipmentUsageMap[eq] || 0,
            status: eqStatus?.status || 'configured'
          };
        }));

        baselineEntries.forEach((entry) => addEquipment(entry.name, entry.status, entry.usageCount));
        return equipmentObjects;
      } catch (err) {
        console.warn('[BEDS] Error enriching equipment for bed', bed.bedId, err.message);
        return [];
      }
    };

    // Group by ward
    const bedsByWard = {};
    beds.forEach(bed => {
      const wName = bed.wardId?.wardName || "Unknown Ward";
      if (!bedsByWard[wName]) {
        bedsByWard[wName] = [];
      }
      bedsByWard[wName].push({
        _id: bed._id,
        bedId: bed.bedId,
        status: bed.status,
        bedType: bed.bedType,
        acuityLevel: bed.acuityLevel,
        hasMonitor: bed.hasMonitor,
        hasVentilator: bed.hasVentilator,
        hasDialysis: bed.hasDialysis,
        statusChangedAt: bed.statusChangedAt,
        estimatedReadyTime: bed.estimatedReadyTime,
        occupancyStartTime: bed.occupancyStartTime,
        expectedDischargeTime: bed.expectedDischargeTime,
        currentPatient: bed.currentEncounterId ? {
          name: bed.currentEncounterId.patientName,
          id: bed.currentEncounterId.patient_id
        } : null
      });
    });

    // Enrich each bed with equipment details and usage counts (non-blocking)
    await Promise.all(Object.keys(bedsByWard).map(async (wardName) => {
      bedsByWard[wardName] = await Promise.all(bedsByWard[wardName].map(async (b) => {
        const equipment = await buildEquipmentForBed({
          bedId: b.bedId,
          hasMonitor: b.hasMonitor,
          hasVentilator: b.hasVentilator,
          hasDialysis: b.hasDialysis,
          canIsolate: b.canIsolate
        });

        return {
          ...b,
          equipment
        };
      }));
    }));

    res.json({
      success: true,
      data: bedsByWard,
      totalBeds: beds.length,
      summary: {
        available: beds.filter(b => b.status === "AVAILABLE").length,
        occupied: beds.filter(b => b.status === "OCCUPIED").length,
        cleaning: beds.filter(b => b.status === "CLEANING").length,
        maintenance: beds.filter(b => b.status === "MAINTENANCE").length
      },
      timestamp: new Date()
    });

  } catch (error) {
    console.error("[BEDS] Error fetching live status:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch live bed status",
      message: error.message
    });
  }
};

// =====================================================
// UPDATE BED STATUS
// =====================================================

/**
 * PATCH /api/beds/:bedId/status
 * Update bed status (AVAILABLE, OCCUPIED, CLEANING, MAINTENANCE)
 */
export const updateBedStatus = async (req, res) => {
  try {
    const { bedId } = req.params;
    const { status, reason, notes } = req.body;
    const changedByName = req.user?.name || "System";
    const changedBy = req.user?._id || null;

    // Validate status
    const validStatuses = ["AVAILABLE", "OCCUPIED", "CLEANING", "MAINTENANCE"];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({
        success: false,
        error: `Invalid status. Must be one of: ${validStatuses.join(", ")}`
      });
    }

    // Find bed
    const bed = await BedSchedule.findById(bedId);
    if (!bed) {
      return res.status(404).json({
        success: false,
        error: "Bed not found"
      });
    }

    const previousStatus = bed.status;

    // Update bed status
    bed.status = status;
    bed.statusChangedAt = new Date();

    if (status === "OCCUPIED") {
      bed.occupancyStartTime = new Date();
      bed.expectedDischargeTime = null;
    } else if (status === "AVAILABLE") {
      bed.currentEncounterId = null;
      bed.assignedNurseId = null;
      bed.occupancyStartTime = null;
      bed.lastCleanedAt = new Date();
    } else if (status === "CLEANING") {
      bed.cleanedByStaffId = changedBy;
    }

    await bed.save();

    // Record status change in history
    const historyEntry = new BedStatusHistory({
      bedId: bed._id,
      bedIdString: bed.bedId,
      wardId: bed.wardId,
      previousStatus,
      currentStatus: status,
      changedBy,
      changedByName,
      reason: reason || "MANUAL_UPDATE",
      notes,
      timestamp: new Date()
    });
    await historyEntry.save();

    // Update ward occupancy counts
    const ward = await Ward.findById(bed.wardId);
    if (ward) {
      const beds = await BedSchedule.find({ wardId: ward._id });
      ward.occupiedBeds = beds.filter(b => b.status === "OCCUPIED").length;
      ward.cleaningBeds = beds.filter(b => b.status === "CLEANING").length;
      ward.maintenanceBeds = beds.filter(b => b.status === "MAINTENANCE").length;
      ward.availableBeds = beds.filter(b => b.status === "AVAILABLE").length;
      await ward.save();
    }

    console.log(`[BEDS] Bed ${bed.bedId} status changed: ${previousStatus} → ${status}`);

    res.json({
      success: true,
      message: `Bed status updated to ${status}`,
      data: {
        bedId: bed.bedId,
        previousStatus,
        newStatus: status,
        statusChangedAt: bed.statusChangedAt,
        estimatedReadyTime: bed.estimatedReadyTime
      }
    });

  } catch (error) {
    console.error("[BEDS] Error updating bed status:", error);
    res.status(500).json({
      success: false,
      error: "Failed to update bed status",
      message: error.message
    });
  }
};

/**
 * POST /api/beds/:bedId/assign
 * Assign patient/encounter to bed
 */
export const assignPatientToBed = async (req, res) => {
  try {
    const { bedId } = req.params;
    const { encounterId, nurseName } = req.body;

    const bed = await BedSchedule.findById(bedId);
    if (!bed) {
      return res.status(404).json({
        success: false,
        error: "Bed not found"
      });
    }

    bed.currentEncounterId = encounterId;
    bed.assignedNurseId = nurseName;
    bed.status = "OCCUPIED";
    bed.occupancyStartTime = new Date();
    await bed.save();

    // Record in history
    const historyEntry = new BedStatusHistory({
      bedId: bed._id,
      bedIdString: bed.bedId,
      wardId: bed.wardId,
      previousStatus: "AVAILABLE",
      currentStatus: "OCCUPIED",
      changedBy: req.user?._id,
      changedByName: req.user?.name || "System",
      reason: "PATIENT_ADMITTED",
      encounterId,
      timestamp: new Date()
    });
    await historyEntry.save();

    console.log(`[BEDS] Patient assigned to bed ${bed.bedId}`);

    res.json({
      success: true,
      message: "Patient assigned to bed",
      data: {
        bedId: bed.bedId,
        status: bed.status,
        occupancyStartTime: bed.occupancyStartTime
      }
    });

  } catch (error) {
    console.error("[BEDS] Error assigning patient:", error);
    res.status(500).json({
      success: false,
      error: "Failed to assign patient to bed",
      message: error.message
    });
  }
};

/**
 * POST /api/beds/:bedId/discharge
 * Discharge patient from bed
 */
export const dischargePatientFromBed = async (req, res) => {
  try {
    const { bedId } = req.params;

    const bed = await BedSchedule.findById(bedId);
    if (!bed) {
      return res.status(404).json({
        success: false,
        error: "Bed not found"
      });
    }

    const previousStatus = bed.status;
    const occupancyDuration = bed.occupancyStartTime
      ? (Date.now() - new Date(bed.occupancyStartTime).getTime()) / (1000 * 60)
      : 0;

    // Move bed to CLEANING status
    bed.status = "CLEANING";
    bed.statusChangedAt = new Date();
    bed.currentEncounterId = null;
    bed.assignedNurseId = null;
    bed.totalOccupancyCount += 1;

    await bed.save();

    // Record in history
    const historyEntry = new BedStatusHistory({
      bedId: bed._id,
      bedIdString: bed.bedId,
      wardId: bed.wardId,
      previousStatus,
      currentStatus: "CLEANING",
      changedBy: req.user?._id,
      changedByName: req.user?.name || "System",
      reason: "PATIENT_DISCHARGED",
      durationMinutes: Math.round(occupancyDuration),
      timestamp: new Date()
    });
    await historyEntry.save();

    console.log(`[BEDS] Patient discharged from bed ${bed.bedId}, moving to cleaning`);

    res.json({
      success: true,
      message: "Patient discharged, bed moving to cleaning",
      data: {
        bedId: bed.bedId,
        status: bed.status,
        occupancyDuration: Math.round(occupancyDuration),
        estimatedReadyTime: bed.estimatedReadyTime
      }
    });

  } catch (error) {
    console.error("[BEDS] Error discharging patient:", error);
    res.status(500).json({
      success: false,
      error: "Failed to discharge patient",
      message: error.message
    });
  }
};

// =====================================================
// BED HISTORY & ANALYTICS
// =====================================================

/**
 * GET /api/beds/:bedId/history
 * Get status change history for a bed
 */
export const getBedHistory = async (req, res) => {
  try {
    const { bedId } = req.params;
    const { limit = 20, days = 7 } = req.query;

    const bed = await BedSchedule.findById(bedId);
    if (!bed) {
      return res.status(404).json({
        success: false,
        error: "Bed not found"
      });
    }

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));

    const history = await BedStatusHistory.find({
      bedId,
      timestamp: { $gte: startDate }
    })
      .sort({ timestamp: -1 })
      .limit(parseInt(limit))
      .lean();

    res.json({
      success: true,
      data: {
        bedId: bed.bedId,
        history,
        totalRecords: history.length,
        period: `Last ${days} days`
      }
    });

  } catch (error) {
    console.error("[BEDS] Error fetching bed history:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch bed history",
      message: error.message
    });
  }
};

/**
 * GET /api/beds/:bedId/equipment
 * Returns equipment baseline for a bed enriched with usage counts and status
 */
export const getBedEquipment = async (req, res) => {
  try {
    const { bedId } = req.params;
    if (!bedId) return res.status(400).json({ success: false, message: 'bedId required' });

    // Find bed schedule entry
    const bed = await BedSchedule.findOne({ bedId }).lean();
    if (!bed) return res.status(404).json({ success: false, message: 'Bed not found' });

    const bedRecord = await Bed.findOne({
      $or: [
        { bed_id: bed.bedId },
        { bed_id: bed.bedId?.toString?.() },
        { bed_id: bed.bedId?.trim?.() }
      ]
    }).lean().catch(() => null);

    const baseline = Array.isArray(bedRecord?.equipment_baseline) ? bedRecord.equipment_baseline : [];

    const usageAgg = await (async () => {
      try {
        const { LabOrder } = await import('../models/lab.model.js');
        const agg = await LabOrder.aggregate([
          { $match: { equipmentUsed: { $exists: true, $ne: null, $ne: '' } } },
          { $group: { _id: '$equipmentUsed', count: { $sum: 1 } } }
        ]);
        return agg.reduce((acc, it) => { acc[it._id] = it.count; return acc; }, {});
      } catch (err) {
        return {};
      }
    })();

    const { LabEquipment } = await import('../models/lab.model.js');
    const equipmentObjects = [];

    const addEquipment = (name, status = 'configured', usageCount = 0) => {
      if (!name) return;
      const normalizedName = name.toLowerCase();
      const existing = equipmentObjects.find((item) => item.name.toLowerCase() === normalizedName);
      if (existing) {
        existing.status = existing.status === 'unknown' ? status : existing.status;
        existing.usageCount = existing.usageCount || usageCount;
        return;
      }
      equipmentObjects.push({ name, status, usageCount });
    };

    if (bed.hasMonitor) addEquipment('Cardiac Monitor', 'active', usageAgg['Cardiac Monitor'] || 0);
    if (bed.hasVentilator) addEquipment('Mechanical Ventilator', 'active', usageAgg['Mechanical Ventilator'] || 0);
    if (bed.hasDialysis) addEquipment('Dialysis Machine', 'active', usageAgg['Dialysis Machine'] || 0);
    if (bed.canIsolate) addEquipment('Isolation Setup', 'ready', 0);

    const baselineEquipment = await Promise.all(baseline.map(async (eq) => {
      const eqStatus = await LabEquipment.findOne({ equipmentName: eq }).lean().catch(() => null);
      return {
        name: eq,
        status: eqStatus?.status || 'configured',
        usageCount: usageAgg[eq] || 0
      };
    }));

    baselineEquipment.forEach((entry) => addEquipment(entry.name, entry.status, entry.usageCount));

    return res.json({ success: true, bedId: bed.bedId, equipment: equipmentObjects });
  } catch (err) {
    console.error('[BEDS] getBedEquipment error:', err);
    return res.status(500).json({ success: false, message: 'Failed to fetch bed equipment', error: err.message });
  }
};

/**
 * GET /api/beds/analytics/occupancy
 * Get occupancy analytics
 */
export const getOccupancyAnalytics = async (req, res) => {
  try {
    const { wardId, days = 7 } = req.query;

    const query = { reason: "PATIENT_DISCHARGED" };
    if (wardId) {
      query.wardId = wardId;
    }

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));
    query.timestamp = { $gte: startDate };

    const history = await BedStatusHistory.find(query).lean();

    // Calculate metrics
    const totalAdmissions = history.length;
    const totalDuration = history.reduce((sum, h) => sum + (h.durationMinutes || 0), 0);
    const averageStay = totalAdmissions > 0 ? totalDuration / totalAdmissions : 0;

    const bedMetrics = {};
    history.forEach(h => {
      if (!bedMetrics[h.bedIdString]) {
        bedMetrics[h.bedIdString] = { count: 0, totalMinutes: 0 };
      }
      bedMetrics[h.bedIdString].count += 1;
      bedMetrics[h.bedIdString].totalMinutes += h.durationMinutes || 0;
    });

    const bedsWithMetrics = Object.entries(bedMetrics).map(([bedId, data]) => ({
      bedId,
      admissions: data.count,
      averageStayHours: (data.totalMinutes / 60 / data.count).toFixed(1),
      utilizationScore: ((data.count / parseInt(days)) * 10).toFixed(1)
    }));

    res.json({
      success: true,
      data: {
        period: `Last ${days} days`,
        totalAdmissions,
        averageStayHours: (averageStay / 60).toFixed(1),
        bedMetrics: bedsWithMetrics.sort((a, b) => b.admissions - a.admissions)
      }
    });

  } catch (error) {
    console.error("[BEDS] Error fetching analytics:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch analytics",
      message: error.message
    });
  }
};
