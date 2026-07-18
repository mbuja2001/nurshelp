import mongoose from 'mongoose';

const invoiceSchema = new mongoose.Schema(
  {
    encounterId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Encounter',
      required: true,
      index: true,
    },
    patientId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Patient',
      required: true,
      index: true,
    },

    // Invoice Identification
    invoiceNumber: {
      type: String,
      unique: true,
      required: true,
      index: true,
    },
    facilityId: {
      type: String,
      required: true,
    },

    // Dates
    dateCreated: {
      type: Date,
      default: Date.now,
      index: true,
    },
    dueDate: {
      type: Date,
      required: true,
    },
    dateOfService: {
      type: Date,
      required: true,
    },

    // Amount Breakdown
    subtotal: {
      type: Number,
      default: 0,
      min: 0,
    },
    discounts: {
      type: Number,
      default: 0,
      min: 0,
    },
    adjustments: {
      type: Number,
      default: 0,
    },
    taxes: {
      type: Number,
      default: 0,
      min: 0,
    },
    total: {
      type: Number,
      required: true,
      min: 0,
    },

    // Line Items
    lineItems: [
      {
        _id: false,
        serviceId: mongoose.Schema.Types.ObjectId,
        description: {
          type: String,
          required: true,
        },
        qty: {
          type: Number,
          default: 1,
          min: 0.1,
        },
        unitPrice: {
          type: Number,
          required: true,
          min: 0,
        },
        amount: {
          type: Number,
          required: true,
          min: 0,
        },
        cptCode: String,
        modifier: String,
        status: {
          type: String,
          enum: ['pending', 'billed', 'paid', 'disputed'],
          default: 'pending',
        },
      },
    ],

    // Payment Tracking
    paymentStatus: {
      type: String,
      enum: ['unpaid', 'partial', 'paid', 'overdue'],
      default: 'unpaid',
      index: true,
    },
    amountPaid: {
      type: Number,
      default: 0,
      min: 0,
    },
    amountDue: {
      type: Number,
      required: true,
    },
    payments: [
      {
        _id: false,
        date: Date,
        amount: Number,
        method: String,
        reference: String,
        paymentId: mongoose.Schema.Types.ObjectId,
      },
    ],

    // Insurance Information
    primaryInsuranceClaim: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'InsuranceClaim',
    },
    secondaryInsuranceClaim: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'InsuranceClaim',
    },
    insuredName: String,
    insuredRelationship: String,
    patientResponsibility: {
      type: Number,
      default: 0,
      min: 0,
    },

    // Status & Metadata
    status: {
      type: String,
      enum: ['draft', 'issued', 'cancelled'],
      default: 'draft',
      index: true,
    },
    sentDate: Date,
    sentVia: {
      type: String,
      enum: ['email', 'mail', 'portal', 'manual'],
    },
    notes: String,
    internalNotes: String,

    // Metadata
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'ClinicalStaff',
    },
    updatedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'ClinicalStaff',
    },

    // Batch Processing
    batchId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'InvoiceBatch',
    },
  },
  {
    timestamps: true,
    indexes: [
      { patientId: 1, dateCreated: -1 },
      { paymentStatus: 1, dueDate: 1 },
      { facilityId: 1, dateCreated: -1 },
    ],
  }
);

// Index for A/R aging
invoiceSchema.index({ paymentStatus: 1, dueDate: 1, status: 1 });

export default mongoose.model('Invoice', invoiceSchema);
