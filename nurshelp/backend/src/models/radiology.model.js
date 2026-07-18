import mongoose from 'mongoose';

/**
 * Radiology Management Models
 * Handles radiology orders, reports, PACS integration
 */

const radiologyOrderSchema = new mongoose.Schema(
  {
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter', required: true, index: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true, index: true },
    orderingPhysicianId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true },

    // Order Details
    orderNumber: { type: String, unique: true, required: true, index: true },
    studyType: {
      type: String,
      enum: ['xray', 'ct', 'mri', 'ultrasound', 'pet', 'mammography', 'fluoroscopy', 'other'],
      required: true,
    },
    bodyPart: String,
    clinicalIndication: String,
    urgency: { type: String, enum: ['routine', 'urgent', 'stat'], default: 'routine' },

    // Scheduling
    scheduledDate: Date,
    scheduledTime: String,
    orderDate: { type: Date, default: Date.now, index: true },

    // Status
    status: {
      type: String,
      enum: ['pending', 'scheduled', 'in-progress', 'completed', 'reported', 'reviewed', 'cancelled'],
      default: 'pending',
      index: true,
    },

    // Technician & Results
    technicianId: mongoose.Schema.Types.ObjectId,
    performedDate: Date,
    performedBy: String,

    // PACS Integration
    pacsAccessionNumber: String,
    imageCount: Number,
    seriesCount: Number,
    dicomFileIds: [String],
    imageStorageLocation: String,

    alerts: [String],
    notes: String,
    contraindications: [String],
  },
  { timestamps: true, indexes: [{ patientId: 1, orderDate: -1 }, { status: 1, urgency: 1 }] }
);

const radiologyReportSchema = new mongoose.Schema(
  {
    radiologyOrderId: { type: mongoose.Schema.Types.ObjectId, ref: 'RadiologyOrder', required: true, index: true },
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter' },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' },

    // Report Details
    reportDate: { type: Date, default: Date.now, index: true },
    radiologistId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true },
    drafted: Boolean,
    dictated: Boolean,
    transcribed: Boolean,

    // Content
    findings: String,
    impression: String,
    recommendations: [String],
    criticalFindings: [
      {
        finding: String,
        severity: { type: String, enum: ['low', 'medium', 'high'], index: true },
        notificationDate: Date,
        notificationTo: [mongoose.Schema.Types.ObjectId],
      },
    ],

    // Status
    status: { type: String, enum: ['draft', 'pending-approval', 'approved', 'reviewed'], default: 'draft', index: true },
    approvedBy: mongoose.Schema.Types.ObjectId,
    approvedDate: Date,

    // Prior Comparisons
    priorReportIds: [mongoose.Schema.Types.ObjectId],
    changesFromPrior: String,

    notes: String,
  },
  { timestamps: true }
);

const pacsImageSchema = new mongoose.Schema(
  {
    radiologyOrderId: { type: mongoose.Schema.Types.ObjectId, ref: 'RadiologyOrder', required: true, index: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' },

    // DICOM Info
    dicomFileId: String,
    seriesNumber: Number,
    instanceNumber: Number,
    modality: String, // CR, CT, MR, US, etc.
    sopClassUid: String,
    sopInstanceUid: String,

    // Storage
    fileSize: Number,
    storagePath: String,
    compressionType: String,
    accessLog: [
      {
        accessedBy: mongoose.Schema.Types.ObjectId,
        accessedDate: Date,
        purpose: String,
      },
    ],

    // Image Metadata
    captureDate: Date,
    deviceId: String,
    windowCenter: Number,
    windowWidth: Number,
    annotations: [
      {
        text: String,
        coordinates: { x: Number, y: Number },
        createdBy: mongoose.Schema.Types.ObjectId,
      },
    ],
  },
  { timestamps: true, indexes: [{ radiologyOrderId: 1 }, { patientId: 1 }] }
);

export const RadiologyOrder = mongoose.model('RadiologyOrder', radiologyOrderSchema);
export const RadiologyReport = mongoose.model('RadiologyReport', radiologyReportSchema);
export const PacsImage = mongoose.model('PacsImage', pacsImageSchema);
