import mongoose from 'mongoose';

/**
 * Lab Management Models
 * Handles lab orders, results, equipment integration
 */

const labOrderSchema = new mongoose.Schema(
  {
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter', required: true, index: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true, index: true },
    orderingPhysicianId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true },

    // Order Details
    orderNumber: { type: String, unique: true, required: true, index: true },
    testCode: String, // LOINC code
    testName: { type: String, required: true },
    sampleType: { type: String, enum: ['blood', 'urine', 'serum', 'plasma', 'saliva', 'tissue', 'other'] },
    collectionTime: Date,
    priority: { type: String, enum: ['routine', 'urgent', 'stat'], default: 'routine' },

    // Clinical Info
    clinicalIndication: String,
    additionalTests: [String],
    fasting: Boolean,
    specialInstructions: String,

    // Status
    status: {
      type: String,
      enum: ['pending', 'collected', 'in-processing', 'completed', 'verified', 'reported', 'cancelled'],
      default: 'pending',
      index: true,
    },

    // Collection
    collectedBy: mongoose.Schema.Types.ObjectId,
    collectionDate: Date,
    sampleId: String,
    sampleLocation: String,

    // Lab Processing
    processedBy: mongoose.Schema.Types.ObjectId,
    processingDate: Date,
    equipmentUsed: String,
    analysisMethod: String,

    orderDate: { type: Date, default: Date.now, index: true },
    completedDate: Date,
  },
  { timestamps: true, indexes: [{ patientId: 1, orderDate: -1 }, { status: 1, priority: 1 }] }
);

const labResultSchema = new mongoose.Schema(
  {
    labOrderId: { type: mongoose.Schema.Types.ObjectId, ref: 'LabOrder', required: true, index: true },
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter' },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true },

    // Result Information
    testName: String,
    testCode: String,
    resultValue: String,
    resultNumericValue: Number,
    unit: String,

    // Reference Ranges
    referenceRangeMin: Number,
    referenceRangeMax: Number,
    referenceRangeText: String,
    abnormalFlag: {
      type: String,
      enum: ['L', 'H', 'LL', 'HH', 'N'], // Low, High, Critical Low/High, Normal
      index: true,
    },

    // Status
    status: { type: String, enum: ['pending', 'verified', 'reported'], default: 'pending', index: true },
    verifiedBy: mongoose.Schema.Types.ObjectId,
    verifiedDate: Date,

    // Delta Check (comparison with previous result)
    previousResultId: mongoose.Schema.Types.ObjectId,
    deltaValue: Number,
    deltaPercent: Number,
    criticalAlert: Boolean,

    methodology: String,
    completedDate: { type: Date, default: Date.now },
    notes: String,
  },
  { timestamps: true, indexes: [{ patientId: 1, testCode: 1 }, { abnormalFlag: 1 }] }
);

const labEquipmentSchema = new mongoose.Schema(
  {
    equipmentName: { type: String, required: true, unique: true },
    equipmentType: {
      type: String,
      enum: ['analyzer', 'centrifuge', 'incubator', 'microscope', 'reader', 'middleware', 'other'],
      required: true,
    },
    manufacturer: String,
    model: String,
    serialNumber: String,
    facilityId: { type: String, required: true, index: true },

    // Status & Maintenance
    status: { type: String, enum: ['operational', 'maintenance', 'offline', 'retired'], default: 'operational' },
    installationDate: Date,
    lastMaintenanceDate: Date,
    nextMaintenanceDate: Date,

    // Capabilities
    testsSupported: [String],
    samplesPerHour: Number,
    accuracy: Number,
    turnaroundTime: String,

    // Connectivity
    isConnected: Boolean,
    ipAddress: String,
    lastSyncTime: Date,
    connectionStatus: { type: String, enum: ['connected', 'disconnected', 'error'] },

    // Quality Control
    qualityControlResults: [
      {
        testDate: Date,
        parameters: String,
        result: String,
        status: String,
      },
    ],

    alerts: [String],
    notes: String,
  },
  { timestamps: true }
);

export const LabOrder = mongoose.model('LabOrder', labOrderSchema);
export const LabResult = mongoose.model('LabResult', labResultSchema);
export const LabEquipment = mongoose.model('LabEquipment', labEquipmentSchema);
