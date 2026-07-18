import mongoose from 'mongoose';

/**
 * Insurance Claims & Accounting Models
 * Handles eClaims submission and accounting/GL
 */

const eClaimSchema = new mongoose.Schema(
  {
    invoiceId: { type: mongoose.Schema.Types.ObjectId, ref: 'Invoice', required: true, index: true },
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter' },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' },

    // eClaim Identification
    claimNumber: { type: String, unique: true, required: true, index: true },
    submissionBatchId: String,
    clearinghouseId: String,
    clearinghouseAcceptanceNumber: String,

    // Claim Form Type
    claimFormType: { type: String, enum: ['cms-1500', 'ub-04', 'ub-92'], default: 'cms-1500' },

    // Insurance Details
    insuranceId: mongoose.Schema.Types.ObjectId,
    claimType: { type: String, enum: ['primary', 'secondary', 'tertiary'], default: 'primary' },

    // Dates
    submissionDate: { type: Date, default: Date.now, index: true },
    fromServiceDate: Date,
    toServiceDate: Date,

    // Amount
    claimAmount: Number,
    approvedAmount: Number,
    deniedAmount: Number,
    pendingAmount: Number,

    // Status Tracking
    submissionStatus: {
      type: String,
      enum: ['draft', 'ready-to-submit', 'submitted', 'acknowledged', 'rejected', 'in-process', 'approved', 'denied', 'partially-approved'],
      default: 'draft',
      index: true,
    },

    clearinghouseStatus: String,
    insuranceStatus: String,
    statusUpdatedDate: Date,

    // Denial Reason Codes
    denialReasons: [
      {
        claimLineNumber: Number,
        reasonCode: String,
        reasonDescription: String,
        severity: { type: String, enum: ['info', 'warning', 'error'], default: 'error' },
        corrective: Boolean,
      },
    ],

    // Validation
    validationErrors: [
      {
        fieldName: String,
        errorCode: String,
        errorDescription: String,
      },
    ],

    // Payment Info
    remittanceAdviceId: String,
    eobDate: Date,
    expectedPaymentDate: Date,
    actualPaymentDate: Date,
    paymentAmount: Number,

    // Resubmission
    originatingClaimId: mongoose.Schema.Types.ObjectId,
    resubmissionNumber: { type: Number, default: 0 },
    resubmissionReason: String,

    // Metadata
    submittedBy: mongoose.Schema.Types.ObjectId,
    facility: String,
    notes: String,
  },
  { timestamps: true, indexes: [{ patientId: 1, submissionDate: -1 }, { submissionStatus: 1 }] }
);

const generalLedgerSchema = new mongoose.Schema(
  {
    // Account Info
    accountCode: { type: String, required: true, unique: true, index: true },
    accountName: String,
    accountType: {
      type: String,
      enum: ['asset', 'liability', 'equity', 'revenue', 'expense', 'retained-earnings'],
      required: true,
      index: true,
    },
    accountCategory: String,

    // Hierarchy
    parentAccountCode: String,
    subAccounts: [String],

    // Balance
    debitBalance: { type: Number, default: 0 },
    creditBalance: { type: Number, default: 0 },
    currentBalance: Number,

    // Tracking
    facilityId: String,
    department: String,
    costCenter: String,

    // Status
    isActive: { type: Boolean, default: true },
    isHeader: Boolean,

    // Metadata
    description: String,
    lastModified: Date,
    modifiedBy: mongoose.Schema.Types.ObjectId,
  },
  { timestamps: true }
);

const journalEntrySchema = new mongoose.Schema(
  {
    // Entry Info
    entryNumber: { type: String, unique: true, required: true, index: true },
    refType: String, // Invoice, Payment, Adjustment, etc.
    refId: mongoose.Schema.Types.ObjectId,

    // Date
    entryDate: { type: Date, required: true, index: true },
    postingDate: Date,
    periodCode: String, // e.g., "2026-04"

    // Status
    status: { type: String, enum: ['draft', 'posted', 'reversed', 'voided'], default: 'draft' },
    postedBy: mongoose.Schema.Types.ObjectId,
    postedDate: Date,

    // Line Items
    lineItems: [
      {
        _id: false,
        lineNumber: Number,
        accountCode: { type: String, required: true },
        accountName: String,
        debit: { type: Number, min: 0 },
        credit: { type: Number, min: 0 },
        description: String,
        department: String,
        costCenter: String,
      },
    ],

    // Totals
    totalDebit: Number,
    totalCredit: Number,

    // Description & Notes
    description: String,
    reference: String,
    notes: String,

    // Audit
    createdBy: mongoose.Schema.Types.ObjectId,
    reversalEntryId: mongoose.Schema.Types.ObjectId,
  },
  { timestamps: true, indexes: [{ entryDate: 1 }, { postingDate: 1 }, { periodCode: 1 }] }
);

const financialReportSchema = new mongoose.Schema(
  {
    // Report Info
    reportType: {
      type: String,
      enum: ['income-statement', 'balance-sheet', 'cash-flow', 'trial-balance', 'aging-summary', 'revenue-by-dept'],
      required: true,
    },
    reportName: String,

    // Period
    fromDate: { type: Date, required: true },
    toDate: { type: Date, required: true },
    periodCode: String,

    // Scope
    facilityId: String,
    department: String,

    // Data
    reportData: mongoose.Schema.Types.Mixed, // Flexible structure for different report types
    summary: {
      totalRevenue: Number,
      totalExpense: Number,
      netIncome: Number,
      totalAssets: Number,
      totalLiabilities: Number,
      totalEquity: Number,
    },

    // Generation
    generatedDate: { type: Date, default: Date.now },
    generatedBy: mongoose.Schema.Types.ObjectId,
    approved: Boolean,
    approvedBy: mongoose.Schema.Types.ObjectId,
    approvalDate: Date,

    // File Info
    fileName: String,
    fileFormat: { type: String, enum: ['pdf', 'excel', 'json'], default: 'pdf' },
    storageLocation: String,

    notes: String,
  },
  { timestamps: true, indexes: [{ reportType: 1, periodCode: 1 }, { facilityId: 1, fromDate: 1 }] }
);

export const EClaim = mongoose.model('EClaim', eClaimSchema);
export const GeneralLedger = mongoose.model('GeneralLedger', generalLedgerSchema);
export const JournalEntry = mongoose.model('JournalEntry', journalEntrySchema);
export const FinancialReport = mongoose.model('FinancialReport', financialReportSchema);
