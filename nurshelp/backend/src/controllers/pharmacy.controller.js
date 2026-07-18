/**
 * Pharmacy Controller
 * Handles pharmaceutical inventory management with hospital isolation
 */

import { Pharmaceutical } from '../models/pharmacy.model.js';
import { Encounter } from '../models/encounter.model.js';

// ============ BATCH CREATE PHARMACEUTICALS ============
export const createPharmaceuticalsBatch = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { pharmaceuticals } = req.body;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found in request'
      });
    }

    if (!pharmaceuticals || !Array.isArray(pharmaceuticals) || pharmaceuticals.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Pharmaceuticals array is required and must not be empty'
      });
    }

    // Convert hospitalId from string to ObjectId for database queries
    const { ObjectId } = await import('mongodb');
    let hospitalIdForQuery;
    try {
      hospitalIdForQuery = new ObjectId(hospitalId);
      console.log(`[PHARMACY BATCH] Converted hospitalId to ObjectId: ${hospitalIdForQuery}`);
    } catch (err) {
      console.error(`[PHARMACY BATCH] Invalid hospitalId format: ${hospitalId}`);
      return res.status(400).json({
        success: false,
        message: 'Invalid hospital context. Please log out and log in again.'
      });
    }

    console.log(`[PHARMACY BATCH] Creating ${pharmaceuticals.length} pharmaceutical records for hospital ${hospitalId}`);

    // Add hospitalId to each pharmaceutical
    const pharmsWithHospitalId = pharmaceuticals.map(pharm => ({
      hospitalId,
      name: pharm.name,
      strength: pharm.strength,
      form: pharm.form || 'tablet',
      route: pharm.route || 'oral',
      stock: pharm.stock || 0,
      minStockLevel: pharm.minStockLevel || 10,
      unitPrice: pharm.unitPrice || 0,
      totalValue: (pharm.stock || 0) * (pharm.unitPrice || 0),
      genericName: pharm.genericName,
      classification: pharm.classification,
      requiresPrescription: pharm.requiresPrescription !== false,
      warnings: pharm.warnings || [],
      contraindications: pharm.contraindications || [],
      createdBy: req.user._id,
    }));

    // Check for duplicates in request
    const names = pharmsWithHospitalId.map(p => p.name.toLowerCase());
    const duplicates = names.filter((name, idx) => names.indexOf(name) !== idx);
    if (duplicates.length > 0) {
      return res.status(400).json({
        success: false,
        message: `Duplicate pharmaceutical names in request: ${[...new Set(duplicates)].join(', ')}`
      });
    }

    // Check for existing drugs in this hospital
    const existingDrugs = await Pharmaceutical.find({
      hospitalId: hospitalIdForQuery,
      name: { $in: pharmsWithHospitalId.map(p => p.name) }
    });

    if (existingDrugs.length > 0) {
      return res.status(400).json({
        success: false,
        message: `Pharmaceuticals already exist: ${existingDrugs.map(d => d.name).join(', ')}`
      });
    }

    // Create pharmaceuticals
    const createdPharms = await Pharmaceutical.insertMany(pharmsWithHospitalId);

    console.log(`[PHARMACY BATCH] Successfully created ${createdPharms.length} pharmaceutical records`);

    res.status(201).json({
      success: true,
      message: `${createdPharms.length} pharmaceutical records created successfully`,
      count: createdPharms.length,
      pharmaceuticals: createdPharms
    });
  } catch (err) {
    console.error('[PHARMACY BATCH ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to create pharmaceuticals',
      error: err.message
    });
  }
};

// ============ GET PHARMACEUTICALS (for hospital) ============
export const getPharmaceuticals = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const pharmaceuticals = await Pharmaceutical.find({ hospitalId })
      .sort({ name: 1 });

    // Calculate summary metrics
    const totalValue = pharmaceuticals.reduce((sum, p) => sum + (p.totalValue || 0), 0);
    const lowStock = pharmaceuticals.filter(p => p.stock < p.minStockLevel).length;

    res.status(200).json({
      success: true,
      count: pharmaceuticals.count,
      pharmaceuticals,
      summary: {
        totalDrugs: pharmaceuticals.length,
        totalInventoryValue: totalValue.toFixed(2),
        lowStockItems: lowStock
      }
    });
  } catch (err) {
    console.error('[PHARMACY GET ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch pharmaceuticals',
      error: err.message
    });
  }
};

// ============ GET SINGLE PHARMACEUTICAL ============
export const getPharmaceutical = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { pharmaceuticalId } = req.params;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const pharm = await Pharmaceutical.findOne({
      _id: pharmaceuticalId,
      hospitalId
    });

    if (!pharm) {
      return res.status(404).json({
        success: false,
        message: 'Pharmaceutical not found'
      });
    }

    res.status(200).json({
      success: true,
      pharmaceutical: pharm
    });
  } catch (err) {
    console.error('[PHARMACY GET ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch pharmaceutical',
      error: err.message
    });
  }
};

// ============ UPDATE PHARMACEUTICAL ============
export const updatePharmaceutical = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { pharmaceuticalId } = req.params;
    const updates = req.body;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    // Don't allow changing hospitalId
    delete updates.hospitalId;

    const pharm = await Pharmaceutical.findOneAndUpdate(
      { _id: pharmaceuticalId, hospitalId },
      { ...updates, updatedAt: new Date() },
      { new: true, runValidators: true }
    );

    if (!pharm) {
      return res.status(404).json({
        success: false,
        message: 'Pharmaceutical not found'
      });
    }

    console.log(`[PHARMACY UPDATE] Pharmaceutical ${pharm.name} updated for hospital ${hospitalId}`);

    res.status(200).json({
      success: true,
      message: 'Pharmaceutical updated successfully',
      pharmaceutical: pharm
    });
  } catch (err) {
    console.error('[PHARMACY UPDATE ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to update pharmaceutical',
      error: err.message
    });
  }
};

// ============ UPDATE STOCK ============
export const updateStock = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { pharmaceuticalId } = req.params;
    const { stock, reason } = req.body;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    if (stock === undefined || stock === null) {
      return res.status(400).json({
        success: false,
        message: 'Stock quantity is required'
      });
    }

    const pharm = await Pharmaceutical.findOne({
      _id: pharmaceuticalId,
      hospitalId
    });

    if (!pharm) {
      return res.status(404).json({
        success: false,
        message: 'Pharmaceutical not found'
      });
    }

    const oldStock = pharm.stock;
    pharm.stock = stock;
    pharm.totalValue = stock * pharm.unitPrice;
    await pharm.save();

    console.log(`[PHARMACY STOCK] ${pharm.name} stock updated: ${oldStock} → ${stock} (Reason: ${reason || 'restock'})`);

    res.status(200).json({
      success: true,
      message: 'Stock updated successfully',
      pharmaceutical: pharm,
      change: stock - oldStock
    });
  } catch (err) {
    console.error('[PHARMACY STOCK ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to update stock',
      error: err.message
    });
  }
};

// ============ GET LOW STOCK ITEMS ============
export const getLowStockItems = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const lowStockItems = await Pharmaceutical.find({
      hospitalId,
      $expr: { $lt: ['$stock', '$minStockLevel'] }
    }).sort({ stock: 1 });

    res.status(200).json({
      success: true,
      count: lowStockItems.length,
      items: lowStockItems
    });
  } catch (err) {
    console.error('[PHARMACY LOW STOCK ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch low stock items',
      error: err.message
    });
  }
};

export const getPendingMedicationRequests = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    const encounters = await Encounter.find({
      hospitalId,
      medication_requests: { $exists: true, $ne: [] }
    })
      .select('patient _id medication_requests status')
      .lean();

    const requests = encounters.flatMap((encounter) => {
      const pendingRequests = (encounter.medication_requests || [])
        .filter((request) => (request.status || 'PENDING') === 'PENDING')
        .map((request) => ({
          _id: request._id || `${encounter._id}:${request.medication_name}`,
          encounterId: encounter._id,
          patientName: encounter.patient?.name || 'Patient',
          patientId: encounter.patient?.id_number || encounter._id,
          medicationName: request.medication_name,
          dosage: request.dosage,
          frequency: request.frequency,
          route: request.route,
          indication: request.indication,
          requestedBy: request.requested_by,
          requestedAt: request.requested_at,
          status: request.status || 'PENDING'
        }));

      return pendingRequests;
    });

    res.status(200).json({ success: true, requests });
  } catch (err) {
    console.error('[PHARMACY REQUESTS ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to fetch pending medication requests', error: err.message });
  }
};

export const updateMedicationRequestStatus = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { encounterId, requestId } = req.params;
    const { status, review_notes } = req.body;

    if (!hospitalId) {
      return res.status(400).json({ success: false, message: 'Hospital ID not found' });
    }

    const encounter = await Encounter.findOne({ _id: encounterId, hospitalId });
    if (!encounter) {
      return res.status(404).json({ success: false, message: 'Encounter not found' });
    }

    const request = encounter.medication_requests?.id(requestId);
    if (!request) {
      return res.status(404).json({ success: false, message: 'Medication request not found' });
    }

    request.status = status || 'APPROVED';
    request.review_notes = review_notes || '';
    request.reviewed_at = new Date();
    request.pharmacy_reference = req.user?.name || req.user?.email || 'Pharmacy';

    await encounter.save();

    res.status(200).json({ success: true, request: request.toObject() });
  } catch (err) {
    console.error('[PHARMACY REQUEST UPDATE ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to update medication request', error: err.message });
  }
};

// ============ DELETE PHARMACEUTICAL ============
export const deletePharmaceutical = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { pharmaceuticalId } = req.params;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const pharm = await Pharmaceutical.findOneAndDelete({
      _id: pharmaceuticalId,
      hospitalId
    });

    if (!pharm) {
      return res.status(404).json({
        success: false,
        message: 'Pharmaceutical not found'
      });
    }

    console.log(`[PHARMACY DELETE] Pharmaceutical ${pharm.name} deleted for hospital ${hospitalId}`);

    res.status(200).json({
      success: true,
      message: 'Pharmaceutical deleted successfully'
    });
  } catch (err) {
    console.error('[PHARMACY DELETE ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to delete pharmaceutical',
      error: err.message
    });
  }
};

// ============ GET PHARMACY INVENTORY (for clinicians) ============
export const getPharmacyInventory = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { PharmacyInventory } = await import('../models/pharmacy.model.js');

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    console.log(`[PHARMACY INVENTORY] Fetching inventory for hospital ${hospitalId}`);

    const inventory = await PharmacyInventory.find({ hospitalId })
      .populate('medicationId')
      .sort({ name: 1 });

    if (!inventory || inventory.length === 0) {
      return res.status(200).json({
        success: true,
        data: [],
        message: 'No inventory items found for this hospital'
      });
    }

    // Transform data for frontend
    const transformedInventory = inventory.map(inv => ({
      _id: inv._id,
      medicationId: inv.medicationId?._id || inv.medicationId,
      medicationName: inv.medicationId?.name || 'Unknown Medication',
      name: inv.medicationId?.name || 'Unknown Medication',
      quantity: inv.quantity,
      minStockLevel: inv.minStockLevel,
      maxStockLevel: inv.maxStockLevel,
      unitPrice: inv.unitPrice,
      status: inv.status,
      facilityId: inv.facilityId,
      category: inv.medicationId?.therapeuticClass || 'General',
      unit: 'units',
      expiryDate: inv.batchNumber?.[0]?.expiryDate || new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
      location: inv.facilityId || 'Main Pharmacy',
      manufacturer: inv.medicationId?.manufacturer || 'Unknown',
      batchNumber: inv.batchNumber?.[0]?.batchId || 'N/A'
    }));

    res.status(200).json({
      success: true,
      data: transformedInventory,
      summary: {
        totalItems: transformedInventory.length,
        lowStock: transformedInventory.filter(i => i.quantity < i.minStockLevel).length,
        outOfStock: transformedInventory.filter(i => i.quantity === 0).length
      }
    });
  } catch (err) {
    console.error('[PHARMACY INVENTORY ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch pharmacy inventory',
      error: err.message
    });
  }
};

// ============ DISPENSE & HANDOVER MEDICATION ============
export const dispenseMedication = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const currentStaffId = req.user._id;
    const { PharmacyDispenseLog, PharmacyInventory } = await import('../models/pharmacy.model.js');

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const {
      prescriptionId,
      medicationId,
      patientId,
      quantity,
      batchNumber,
      dosageGiven,
      routeUsed,
      
      // Pharmacist authorization
      pharmacistId,
      pharmacistNotes,
      
      // Medication schedule
      frequency,
      scheduledTimes,
      duration,
      dosagePerTime,
      instructions,
      
      // Who requested it (prescriber info)
      prescribedBy,
      prescribedByName,
      prescribedByRole, // PHYSICIAN, NURSE, etc.
      
      // Patient condition notes
      patientConditionBefore,
      notes
    } = req.body;

    // Validate required fields
    if (!prescriptionId || !patientId || !quantity || !pharmacistId) {
      return res.status(400).json({
        success: false,
        message: 'Missing required fields: prescriptionId, patientId, quantity, pharmacistId'
      });
    }

    console.log(`[DISPENSE] Processing medication handover for patient ${patientId} by staff ${currentStaffId}`);

    // Create dispense log with full audit trail
    const dispenseLog = new PharmacyDispenseLog({
      hospitalId,
      prescriptionId,
      medicationId,
      patientId,
      
      // Dispensing details
      quantity,
      batchNumber,
      dispensedBy: currentStaffId, // The nurse/clinician giving the med
      facilityId: 'MAIN_PHARMACY',
      dispensedDate: new Date(),
      
      // PHARMACIST AUTHORIZATION
      pharmacistAuthorization: {
        pharmacistId,
        pharmacistName: pharmacistNotes?.split('|')[0] || 'Pharmacist', // Extract if included
        authorizedAt: new Date(),
        pharmacistNotes,
        verifiedStock: quantity,
        status: 'approved'
      },
      
      // MEDICATION SCHEDULE - Dosing Schedule for Patient
      medicationSchedule: {
        frequency,
        scheduledTimes: scheduledTimes || [],
        duration,
        dosagePerTime,
        instructions
      },
      
      // INITIAL ADMINISTRATION
      administrations: [
        {
          administeredAt: new Date(),
          administeredBy: currentStaffId, // Staff giving the medication
          administeredByName: req.user.name || 'Clinical Staff',
          administeredByRole: req.user.role,
          
          dosageGiven,
          routeUsed,
          batchUsed: batchNumber,
          
          status: 'administered',
          patientConditionBefore,
          notes
        }
      ],
      
      totalAdministered: 1,
      totalScheduled: scheduledTimes?.length || 1,
      complianceRate: 100, // First dose administered
      
      // Audit trail: who requested this medication
      prescribedBy,
      prescribedByName,
      prescribedByRole,
      
      notes: `Medication handed over. Prescribed by: ${prescribedByName || 'Unknown'} (${prescribedByRole || 'Unknown'})`
    });

    await dispenseLog.save();

    console.log(`[DISPENSE] ✓ Created dispense log ${dispenseLog._id} for medication ${medicationId}`);

    // ============ REDUCE STOCK IN PHARMACEUTICAL INVENTORY ============
    // Import Pharmaceutical model for stock update
    const { Pharmaceutical } = await import('../models/pharmacy.model.js');
    
    // Find the medication and reduce stock by quantity dispensed
    const updatedPharmaceutical = await Pharmaceutical.findByIdAndUpdate(
      medicationId,
      {
        $inc: { stock: -quantity }, // Decrease stock
        $push: {
          dispensingHistory: {
            dispensedAt: new Date(),
            quantityDispensed: quantity,
            dispensedBy: currentStaffId,
            dispensedToPatient: patientId,
            dispenseLogId: dispenseLog._id,
            prescribedBy: prescribedByName,
            batchUsed: batchNumber
          }
        },
        lastDispensedAt: new Date(),
        lastDispensedBy: currentStaffId
      },
      { new: true }
    );

    if (!updatedPharmaceutical) {
      console.error(`[DISPENSE WARNING] Medication ${medicationId} not found for stock update`);
    } else {
      console.log(`[DISPENSE] ✓ Stock reduced by ${quantity}. New stock: ${updatedPharmaceutical.stock}`);
      
      // Check if stock is now below minimum threshold
      if (updatedPharmaceutical.stock < updatedPharmaceutical.minStockLevel) {
        console.warn(`[DISPENSE] ⚠️ LOW STOCK ALERT: ${updatedPharmaceutical.name} stock (${updatedPharmaceutical.stock}) is below minimum (${updatedPharmaceutical.minStockLevel})`);
      }
    }

    console.log(`[DISPENSE] Pharmacist: ${pharmacistId}, Administered by: ${currentStaffId}, Prescribed by: ${prescribedByName}`);

    res.status(201).json({
      success: true,
      message: 'Medication dispensed and handed over successfully',
      data: {
        dispenseLogId: dispenseLog._id,
        patientId,
        medicationId,
        medicationName: updatedPharmaceutical?.name,
        scheduleLevel: updatedPharmaceutical?.scheduleLevel || 'OTC', // DEA Schedule Level
        quantity,
        dispensedBy: currentStaffId,
        newStock: updatedPharmaceutical?.stock || 'Unknown',
        pharmacistAuthorization: dispenseLog.pharmacistAuthorization,
        medicationSchedule: dispenseLog.medicationSchedule,
        administrationRecord: dispenseLog.administrations[0],
        auditTrail: {
          prescribedBy: prescribedByName,
          prescribedByRole,
          authorizedBy: pharmacistNotes?.split('|')[0] || 'Pharmacist',
          handedOverBy: req.user.name || 'Clinical Staff',
          handedOverAt: new Date(),
          stockUpdated: true,
          previousStock: (updatedPharmaceutical?.stock || 0) + quantity,
          currentStock: updatedPharmaceutical?.stock || 0,
          scheduleCompliance: `Schedule Level ${updatedPharmaceutical?.scheduleLevel || 'OTC'} - Special handling may be required for higher schedules`
        }
      }
    });
  } catch (err) {
    console.error('[DISPENSE ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to dispense medication',
      error: err.message
    });
  }
};

// ============ RECORD MEDICATION ADMINISTRATION ============
export const recordAdministration = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const currentStaffId = req.user._id;
    const { PharmacyDispenseLog } = await import('../models/pharmacy.model.js');

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const {
      dispenseLogId,
      dosageGiven,
      routeUsed,
      batchUsed,
      patientConditionBefore,
      patientConditionAfter,
      sideEffectsObserved,
      status,
      reasonIfSkipped,
      notes,
      witnessingStaffId
    } = req.body;

    if (!dispenseLogId) {
      return res.status(400).json({
        success: false,
        message: 'Dispense log ID is required'
      });
    }

    const dispenseLog = await PharmacyDispenseLog.findOneAndUpdate(
      { _id: dispenseLogId, hospitalId },
      {
        $push: {
          administrations: {
            administeredAt: new Date(),
            administeredBy: currentStaffId,
            administeredByName: req.user.name || 'Clinical Staff',
            administeredByRole: req.user.role,
            dosageGiven,
            routeUsed,
            batchUsed,
            status: status || 'administered',
            patientConditionBefore,
            patientConditionAfter,
            sideEffectsObserved,
            reasonIfSkipped,
            notes,
            witnessingStaff: witnessingStaffId
          }
        },
        $inc: { totalAdministered: status === 'administered' ? 1 : 0 }
      },
      { new: true }
    );

    if (!dispenseLog) {
      return res.status(404).json({
        success: false,
        message: 'Dispense log not found'
      });
    }

    // Calculate compliance rate
    const complianceRate = dispenseLog.totalScheduled > 0 
      ? ((dispenseLog.totalAdministered / dispenseLog.totalScheduled) * 100).toFixed(1)
      : 0;

    await PharmacyDispenseLog.updateOne(
      { _id: dispenseLogId },
      { complianceRate }
    );

    console.log(`[ADMINISTRATION] ✓ Recorded administration for dispense log ${dispenseLogId}`);
    console.log(`[ADMINISTRATION] Status: ${status}, Compliance: ${complianceRate}%`);

    res.status(200).json({
      success: true,
      message: 'Administration recorded successfully',
      data: {
        dispenseLogId,
        totalAdministered: dispenseLog.totalAdministered + (status === 'administered' ? 1 : 0),
        totalScheduled: dispenseLog.totalScheduled,
        complianceRate,
        latestAdministration: dispenseLog.administrations[dispenseLog.administrations.length]
      }
    });
  } catch (err) {
    console.error('[ADMINISTRATION ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to record administration',
      error: err.message
    });
  }
};

// ============ REFILL MEDICATION ============
export const refillMedication = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const currentStaffId = req.user._id;
    const { PharmacyInventory, Pharmaceutical } = await import('../models/pharmacy.model.js');

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const {
      medicationId,
      quantityToAdd,
      batchNumber,
      expiryDate,
      manufacturer,
      dateReceived,
      reason, // e.g., "routine restock", "emergency restock", "replacement"
      unitPrice,
      notes
    } = req.body;

    if (!medicationId || !quantityToAdd || quantityToAdd < 1) {
      return res.status(400).json({
        success: false,
        message: 'Medication ID and quantity (>=1) are required'
      });
    }

    console.log(`[REFILL] Processing refill for medication ${medicationId}: +${quantityToAdd} units`);

    // Find existing inventory
    let inventory = await PharmacyInventory.findOne({ 
      hospitalId, 
      medicationId 
    });

    if (!inventory) {
      // Create new inventory record if doesn't exist
      inventory = new PharmacyInventory({
        hospitalId,
        medicationId,
        facilityId: 'MAIN_PHARMACY',
        quantity: quantityToAdd,
        unitPrice: unitPrice || 0,
        minStockLevel: Math.ceil(quantityToAdd * 0.25),
        maxStockLevel: quantityToAdd * 2,
        status: 'in-stock'
      });
    } else {
      // Update existing inventory
      inventory.quantity += quantityToAdd;
      inventory.unitPrice = unitPrice || inventory.unitPrice;
      
      // Update status based on new quantity
      if (inventory.quantity === 0) {
        inventory.status = 'out-of-stock';
      } else if (inventory.quantity < inventory.minStockLevel) {
        inventory.status = 'low-stock';
      } else {
        inventory.status = 'in-stock';
      }
    }

    // Add batch to batchNumber array
    if (batchNumber) {
      if (!inventory.batchNumber) inventory.batchNumber = [];
      inventory.batchNumber.push({
        batchId: batchNumber,
        quantity: quantityToAdd,
        expiryDate: expiryDate || new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
        dateReceived: dateReceived || new Date(),
        manufacturer: manufacturer || 'Unknown'
      });
    }

    inventory.lastRestockDate = new Date();
    await inventory.save();

    console.log(`[REFILL] ✓ Refilled medication ${medicationId}: new quantity = ${inventory.quantity}`);

    res.status(200).json({
      success: true,
      message: `Medication refilled: +${quantityToAdd} units`,
      data: {
        medicationId,
        newQuantity: inventory.quantity,
        status: inventory.status,
        lastRestockDate: inventory.lastRestockDate,
        batchNumber: batchNumber,
        refillReason: reason,
        auditTrail: {
          refillBy: currentStaffId,
          refillAt: new Date(),
          notes
        }
      }
    });
  } catch (err) {
    console.error('[REFILL ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to refill medication',
      error: err.message
    });
  }
};

// ============ ADD NEW MEDICATION TO STOCK ============
export const addMedicationToStock = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const currentStaffId = req.user._id;
    const { Medication, PharmacyInventory } = await import('../models/pharmacy.model.js');

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    const {
      medicationName,
      strength,
      dosageForm,
      route,
      genericName,
      therapeuticClass,
      manufacturer,
      quantity,
      unitPrice,
      minStockLevel,
      maxStockLevel,
      expiryDate,
      batchNumber,
      warnings,
      contraindications,
      notes
    } = req.body;

    if (!medicationName || !quantity || !unitPrice) {
      return res.status(400).json({
        success: false,
        message: 'Medication name, quantity, and unit price are required'
      });
    }

    console.log(`[ADD MEDICATION] Adding new medication: ${medicationName} (${strength}) to hospital ${hospitalId}`);

    // Check if medication already exists in catalog
    let medication = await Medication.findOne({ name: medicationName });
    
    if (!medication) {
      // Create new medication in catalog
      medication = new Medication({
        name: medicationName,
        strength,
        dosageForm: dosageForm || 'tablet',
        route: route || 'oral',
        genericName,
        therapeuticClass: therapeuticClass || 'General',
        manufacturer,
        rxRequired: true,
        warning: warnings || [],
        contraindications: contraindications || [],
        sideEffects: [],
        dosageInstructions: `See label`,
        maxDailyDose: null
      });

      await medication.save();
      console.log(`[ADD MEDICATION] Created new medication in catalog: ${medication._id}`);
    }

    // Create inventory record for this hospital
    const inventory = new PharmacyInventory({
      hospitalId,
      medicationId: medication._id,
      facilityId: 'MAIN_PHARMACY',
      quantity,
      unitPrice,
      totalValue: quantity * unitPrice,
      minStockLevel: minStockLevel || Math.ceil(quantity * 0.25),
      maxStockLevel: maxStockLevel || quantity * 2,
      reorderPoint: Math.ceil(quantity * 0.25),
      reorderQuantity: Math.ceil(quantity * 0.5),
      status: quantity === 0 ? 'out-of-stock' : 'in-stock',
      lastRestockDate: new Date()
    });

    // Add batch information
    if (batchNumber) {
      inventory.batchNumber.push({
        batchId: batchNumber,
        quantity,
        expiryDate: expiryDate || new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
        dateReceived: new Date(),
        manufacturer: manufacturer || 'Unknown'
      });
    }

    await inventory.save();

    console.log(`[ADD MEDICATION] ✓ Added ${medicationName} to hospital pharmacy: ${quantity} units @ R${unitPrice}`);

    res.status(201).json({
      success: true,
      message: `New medication added to stock: ${medicationName}`,
      data: {
        medicationId: medication._id,
        medicationName,
        strength,
        quantity,
        unitPrice,
        status: inventory.status,
        inventoryId: inventory._id,
        auditTrail: {
          addedBy: currentStaffId,
          addedAt: new Date(),
          notes
        }
      }
    });
  } catch (err) {
    console.error('[ADD MEDICATION ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to add medication to stock',
      error: err.message
    });
  }
};

// ============ GET DISPENSING HISTORY FOR MEDICATION ============
export const getMedicationDispensingHistory = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { medicationId } = req.params;
    const { limit = 50, skip = 0 } = req.query;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    // Get medication with dispensing history
    const medication = await Pharmaceutical.findOne({
      _id: medicationId,
      hospitalId
    }).populate('dispensingHistory.dispensedBy', 'name email role');

    if (!medication) {
      return res.status(404).json({
        success: false,
        message: 'Medication not found'
      });
    }

    // Get dispensing history with pagination
    const dispensingHistory = medication.dispensingHistory
      .sort((a, b) => new Date(b.dispensedAt) - new Date(a.dispensedAt))
      .slice(skip, skip + parseInt(limit));

    // Calculate statistics
    const totalDispensed = medication.dispensingHistory.reduce((sum, record) => sum + (record.quantityDispensed || 0), 0);
    const averageQuantity = medication.dispensingHistory.length > 0 ? (totalDispensed / medication.dispensingHistory.length).toFixed(2) : 0;

    console.log(`[DISPENSE HISTORY] Retrieved ${dispensingHistory.length} dispensing records for ${medication.name}`);

    res.status(200).json({
      success: true,
      data: {
        medication: {
          id: medication._id,
          name: medication.name,
          strength: medication.strength,
          form: medication.form,
          currentStock: medication.stock,
          minStockLevel: medication.minStockLevel,
          lastDispensedAt: medication.lastDispensedAt,
          dispenseCount: medication.dispenseCount || medication.dispensingHistory.length
        },
        dispensingHistory,
        statistics: {
          totalRecords: medication.dispensingHistory.length,
          totalDispensed,
          averageQuantityPerDispense: parseFloat(averageQuantity),
          lastDispenseDate: medication.lastDispensedAt,
          dispenseFrequency: `${medication.dispensingHistory.length} dispenses recorded`
        },
        pagination: {
          skip: parseInt(skip),
          limit: parseInt(limit),
          total: medication.dispensingHistory.length,
          hasMore: (skip + parseInt(limit)) < medication.dispensingHistory.length
        }
      }
    });
  } catch (err) {
    console.error('[DISPENSE HISTORY ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch dispensing history',
      error: err.message
    });
  }
};

// ============ GET INVENTORY WITH DISPENSING STATS ============
export const getInventoryWithStats = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    // Get all medications with dispensing stats
    const medications = await Pharmaceutical.find({ hospitalId })
      .sort({ name: 1 })
      .select('name strength form stock minStockLevel unitPrice totalValue lastDispensedAt dispenseCount dispensingHistory createdAt updatedAt');

    // Calculate comprehensive stats
    const stats = {
      totalMedications: medications.length,
      totalInventoryValue: medications.reduce((sum, m) => sum + (m.totalValue || 0), 0).toFixed(2),
      lowStockItems: medications.filter(m => m.stock < m.minStockLevel).length,
      outOfStockItems: medications.filter(m => m.stock === 0).length,
      totalDispelled: medications.reduce((sum, m) => sum + (m.dispensingHistory?.length || 0), 0),
      recentlyDispensed: medications
        .filter(m => m.lastDispensedAt && new Date(m.lastDispensedAt) > new Date(Date.now() - 24 * 60 * 60 * 1000))
        .length
    };

    // Enhance each medication with dispensing stats
    const medicationsWithStats = medications.map(med => ({
      ...med.toObject(),
      dispensing: {
        dispenseCount: med.dispensingHistory?.length || 0,
        lastDispensedAt: med.lastDispensedAt,
        totalQuantityDispensed: med.dispensingHistory?.reduce((sum, d) => sum + (d.quantityDispensed || 0), 0) || 0,
        defaultUsagePattern: med.dispensingHistory?.length > 0 
          ? `~${(med.dispensingHistory.reduce((sum, d) => sum + (d.quantityDispensed || 0), 0) / med.dispensingHistory.length).toFixed(1)} units per dispense`
          : 'No dispense data'
      },
      stockStatus: med.stock === 0 ? 'Out of Stock' : med.stock < med.minStockLevel ? 'Low Stock' : 'In Stock'
    }));

    console.log(`[INVENTORY STATS] Retrieved ${medications.length} medications with dispensing statistics`);

    res.status(200).json({
      success: true,
      data: {
        medications: medicationsWithStats,
        statistics: stats,
        summary: `Hospital has ${stats.totalMedications} medications in stock. ${stats.lowStockItems} items are low stock, ${stats.outOfStockItems} are out of stock. Total dispensing records: ${stats.totalDispelled}`
      }
    });
  } catch (err) {
    console.error('[INVENTORY STATS ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch inventory statistics',
      error: err.message
    });
  }
};

// ============ GET DISPENSING RECORDS BY DATE RANGE ============
export const getDispensingRecordsByDateRange = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    const { startDate, endDate } = req.query;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    if (!startDate || !endDate) {
      return res.status(400).json({
        success: false,
        message: 'startDate and endDate are required'
      });
    }

    const start = new Date(startDate);
    const end = new Date(endDate);

    // Get all medications and their dispensing records within date range
    const medications = await Pharmaceutical.find({ hospitalId });

    const dispensingRecords = [];
    let totalQuantityDispensed = 0;

    medications.forEach(med => {
      const recordsInRange = med.dispensingHistory?.filter(record => {
        const recordDate = new Date(record.dispensedAt);
        return recordDate >= start && recordDate <= end;
      }) || [];

      recordsInRange.forEach(record => {
        dispensingRecords.push({
          medicationId: med._id,
          medicationName: med.name,
          strength: med.strength,
          form: med.form,
          ...record.toObject?.() || record,
          dispensedAtFormatted: new Date(record.dispensedAt).toLocaleString()
        });
        totalQuantityDispensed += record.quantityDispensed || 0;
      });
    });

    // Sort by date descending
    dispensingRecords.sort((a, b) => new Date(b.dispensedAt) - new Date(a.dispensedAt));

    console.log(`[DISPENSING RECORDS] Retrieved ${dispensingRecords.length} records between ${startDate} and ${endDate}`);

    res.status(200).json({
      success: true,
      data: {
        dateRange: {
          startDate,
          endDate
        },
        records: dispensingRecords,
        summary: {
          totalRecords: dispensingRecords.length,
          totalQuantityDispensed,
          averageQuantityPerDispense: dispensingRecords.length > 0 ? (totalQuantityDispensed / dispensingRecords.length).toFixed(2) : 0,
          uniqueMedicationsDispensed: [...new Set(dispensingRecords.map(r => r.medicationName))].length
        }
      }
    });
  } catch (err) {
    console.error('[DISPENSING RECORDS ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch dispensing records',
      error: err.message
    });
  }
};

// ============ CSV BULK UPLOAD: REFILL & ADD STOCK ============
export const uploadCSVRefill = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    
    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No CSV file uploaded'
      });
    }

    const { parse } = await import('csv-parse/sync');
    const fileContent = req.file.buffer.toString('utf-8');
    
    // Parse CSV with headers
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true
    });

    console.log(`[CSV UPLOAD] Processing ${records.length} records for hospital ${hospitalId}`);

    const results = {
      successful: [],
      failed: [],
      summary: {
        totalRecords: records.length,
        successCount: 0,
        failureCount: 0,
        totalStockAdded: 0,
        totalValue: 0
      }
    };

    // Expected CSV columns: medicationName, strength, quantity, batchNumber, expiryDate (optional), manufacturer (optional)
    for (const [idx, record] of records.entries()) {
      try {
        const {
          medicationName,
          strength,
          quantity,
          batchNumber,
          expiryDate,
          manufacturer,
          form,
          route,
          unitPrice = 0,
          minStockLevel = 10
        } = record;

        // Validate required fields
        if (!medicationName || !quantity || !batchNumber) {
          throw new Error(`Missing required fields: medicationName, quantity, batchNumber`);
        }

        const quantityNum = parseInt(quantity);
        if (isNaN(quantityNum) || quantityNum < 0) {
          throw new Error(`Invalid quantity: ${quantity}`);
        }

        const unitPriceNum = parseFloat(unitPrice) || 0;

        // Find medication by name and hospital
        let medication = await Pharmaceutical.findOne({
          hospitalId,
          name: medicationName
        });

        if (medication) {
          // Update existing medication stock
          const previousStock = medication.stock;
          medication.stock += quantityNum;
          medication.unitPrice = unitPriceNum || medication.unitPrice;
          medication.totalValue = medication.stock * medication.unitPrice;
          medication.updatedAt = new Date();
          
          await medication.save();

          results.successful.push({
            medicationName,
            action: 'refill',
            previousStock,
            newStock: medication.stock,
            quantityAdded: quantityNum,
            batchNumber,
            message: `Refilled ${quantityNum} units`
          });

          results.summary.totalStockAdded += quantityNum;
          results.summary.totalValue += (quantityNum * unitPriceNum);
        } else {
          // Create new medication
          const newMedication = new Pharmaceutical({
            hospitalId,
            name: medicationName,
            strength: strength || 'Unknown',
            form: form || 'Tablet',
            route: route || 'PO',
            stock: quantityNum,
            unitPrice: unitPriceNum,
            totalValue: quantityNum * unitPriceNum,
            minStockLevel: parseInt(minStockLevel) || 10,
            batchNumber: batchNumber,
            lastStockUpdate: new Date(),
            createdAt: new Date()
          });

          await newMedication.save();

          results.successful.push({
            medicationName,
            action: 'create',
            quantity: quantityNum,
            batchNumber,
            message: `Created new medication with ${quantityNum} units`
          });

          results.summary.totalStockAdded += quantityNum;
          results.summary.totalValue += (quantityNum * unitPriceNum);
        }

        results.summary.successCount += 1;
      } catch (err) {
        results.failed.push({
          rowNumber: idx + 2, // +2 accounts for header and 0-indexing
          record,
          error: err.message
        });
        results.summary.failureCount += 1;
        console.error(`[CSV ERROR] Row ${idx + 2}:  ${err.message}`);
      }
    }

    console.log(`[CSV UPLOAD] Completed: ${results.summary.successCount} successful, ${results.summary.failureCount} failed`);

    res.status(200).json({
      success: true,
      message: `CSV processed: ${results.summary.successCount} successful, ${results.summary.failureCount} failed`,
      data: results
    });
  } catch (err) {
    console.error('[CSV UPLOAD ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to process CSV file',
      error: err.message
    });
  }
};

// ============ CSV EXPORT: DOWNLOAD FULL INVENTORY ============
export const exportInventoryCSV = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;

    if (!hospitalId) {
      return res.status(400).json({
        success: false,
        message: 'Hospital ID not found'
      });
    }

    // Get all medications for hospital
    const medications = await Pharmaceutical.find({ hospitalId }).sort({ name: 1 });

    if (medications.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'No medications found for this hospital'
      });
    }

    // Build CSV content
    const csvHeaders = [
      'Medication Name',
      'Strength',
      'Form',
      'Route',
      'Current Stock',
      'Min Stock Level',
      'Unit Price',
      'Total Value',
      'Status',
      'Last Dispensed At',
      'Dispense Count',
      'Schedule Level',
      'Created Date',
      'Updated Date'
    ];

    const csvRows = medications.map(med => [
      med.name || '',
      med.strength || '',
      med.form || '',
      med.route || '',
      med.stock || 0,
      med.minStockLevel || 10,
      med.unitPrice || 0,
      (med.stock * (med.unitPrice || 0)).toFixed(2),
      med.stock === 0 ? 'Out of Stock' : med.stock < med.minStockLevel ? 'Low Stock' : 'In Stock',
      med.lastDispensedAt ? new Date(med.lastDispensedAt).toISOString() : '',
      med.dispenseCount || 0,
      med.scheduleLevel || 'OTC',
      new Date(med.createdAt).toISOString(),
      new Date(med.updatedAt).toISOString()
    ]);

    // Calculate summary
    const totalStock = medications.reduce((sum, m) => sum + (m.stock || 0), 0);
    const totalValue = medications.reduce((sum, m) => sum + ((m.stock || 0) * (m.unitPrice || 0)), 0);
    const lowStockCount = medications.filter(m => m.stock < m.minStockLevel && m.stock > 0).length;
    const outOfStockCount = medications.filter(m => m.stock === 0).length;

    // Summary section
    const summaryRows = [
      [],
      ['INVENTORY SUMMARY'],
      ['Total Medications', medications.length],
      ['Total Units in Stock', totalStock],
      ['Total Inventory Value', `$${totalValue.toFixed(2)}`],
      ['Low Stock Items', lowStockCount],
      ['Out of Stock Items', outOfStockCount],
      ['Export Date', new Date().toISOString()],
      ['Hospital ID', hospitalId]
    ];

    // Combine headers, data, and summary
    const csvContent = [
      csvHeaders.join(','),
      ...csvRows.map(row => row.map(cell => {
        // Escape cells with commas or quotes
        const str = String(cell);
        return str.includes(',') || str.includes('"') ? `"${str.replace(/"/g, '""')}"` : str;
      }).join(','))
    ].join('\n');

    const fullCSV = csvContent + '\n\n' + summaryRows.map(row => 
      row.map(cell => {
        const str = String(cell);
        return str.includes(',') || str.includes('"') ? `"${str.replace(/"/g, '""')}"` : str;
      }).join(',')
    ).join('\n');

    console.log(`[CSV EXPORT] Generated inventory export for hospital ${hospitalId}: ${medications.length} medications`);

    // Set response headers for file download
    const timestamp = new Date().toISOString().split('T')[0];
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="inventory-${timestamp}.csv"`);
    res.send(fullCSV);
  } catch (err) {
    console.error('[CSV EXPORT ERROR]:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to export inventory',
      error: err.message
    });
  }
};
