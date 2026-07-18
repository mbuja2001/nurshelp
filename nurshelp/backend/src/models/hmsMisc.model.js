import mongoose from 'mongoose';

/**
 * Additional HMS Management Models
 * Appointment Scheduling, Blood Bank, Dietary, Alerts, Messaging
 */

const appointmentSchema = new mongoose.Schema(
  {
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true, index: true },
    physicianId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true },
    facilityId: String,

    // Appointment Details
    appointmentNumber: { type: String, unique: true, index: true },
    appointmentType: String, // Follow-up, Initial, etc.
    department: String,
    specialization: String,

    // Scheduling
    scheduledDateTime: { type: Date, required: true, index: true },
    duration: Number, // minutes
    status: {
      type: String,
      enum: ['scheduled', 'confirmed', 'checked-in', 'completed', 'cancelled', 'no-show'],
      default: 'scheduled',
      index: true,
    },

    // Reminders
    reminderSent: Boolean,
    reminderMethod: { type: String, enum: ['sms', 'email', 'push', 'call'] },
    reminderTime: Date,

    // Check-in
    checkInTime: Date,
    checkOutTime: Date,
    roomNumber: String,

    // Notes
    reasonForVisit: String,
    notes: String,
    cancellationReason: String,
  },
  { timestamps: true, indexes: [{ patientId: 1, scheduledDateTime: 1 }, { physicianId: 1, scheduledDateTime: 1 }] }
);

const bloodBankSchema = new mongoose.Schema(
  {
    // Blood Unit Info
    unitId: { type: String, unique: true, required: true, index: true },
    bloodType: { type: String, enum: ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'], required: true, index: true },
    rh: String,
    productType: { type: String, enum: ['whole-blood', 'packed-rbc', 'plasma', 'platelets', 'cryoprecipitate'], required: true },

    // Donor Info
    donorId: mongoose.Schema.Types.ObjectId,
    donationDate: Date,
    collectionCenter: String,

    // Storage
    storageLocation: String,
    storageTemperature: Number,
    expiryDate: { type: Date, required: true, index: true },
    collectionDate: Date,

    // Status
    status: { type: String, enum: ['available', 'reserved', 'transfused', 'discarded', 'expired'], default: 'available', index: true },

    // Testing
    bloodTestitng: {
      hiv: { type: String, enum: ['negative', 'positive', 'pending', 'not-tested'] },
      hepatitisB: { type: String, enum: ['negative', 'positive', 'pending', 'not-tested'] },
      hepatitisC: { type: String, enum: ['negative', 'positive', 'pending', 'not-tested'] },
      syphilis: { type: String, enum: ['negative', 'positive', 'pending', 'not-tested'] },
      malaria: { type: String, enum: ['negative', 'positive', 'pending', 'not-tested'] },
      testDate: Date,
      testedBy: String,
    },

    // Transfusion
    transfusionOrderId: mongoose.Schema.Types.ObjectId,
    transfusedDate: Date,
    transfusedTo: mongoose.Schema.Types.ObjectId,
    transfusedBy: mongoose.Schema.Types.ObjectId,

    // Compatibility
    xmatchCompatible: Boolean,

    notes: String,
  },
  { timestamps: true }
);

const dietaryOrderSchema = new mongoose.Schema(
  {
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter', required: true, index: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true },
    nutritionistId: mongoose.Schema.Types.ObjectId,

    // Diet Details
    dietType: {
      type: String,
      enum: ['regular', 'clear-liquid', 'full-liquid', 'soft', 'bland', 'low-sodium', 'diabetic', 'renal', 'cardiac', 'vegetarian', 'gluten-free', 'other'],
      required: true,
    },
    restrictions: [String],
    allergies: [String],
    preferences: String,

    // Scheduling
    startDate: Date,
    endDate: Date,
    mealsPerDay: Number,

    // Special Instructions
    consistencyModification: String, // Puree, minced, etc.
    caloriesPerDay: Number,
    fluidRestriction: Boolean,
    fluidLimit: String,

    // Status
    status: { type: String, enum: ['pending', 'active', 'modified', 'discontinued'], default: 'pending' },
    modifiedDate: Date,
    modifiedBy: mongoose.Schema.Types.ObjectId,

    notes: String,
  },
  { timestamps: true }
);

const clinicalAlertSchema = new mongoose.Schema(
  {
    encounterId: { type: mongoose.Schema.Types.ObjectId, ref: 'Encounter', required: true, index: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true, index: true },

    // Alert Type
    alertType: {
      type: String,
      enum: [
        'critical-value',
        'abnormal-vital',
        'drug-interaction',
        'allergy',
        'deterioration',
        'critical-glucose',
        'critical-potassium',
        'sepsis-risk',
        'fall-risk',
        'pressure-ulcer-risk',
        'custom',
      ],
      required: true,
      index: true,
    },

    // Severity
    severity: { type: String, enum: ['low', 'medium', 'high', 'critical'], default: 'medium', index: true },

    // Content
    title: String,
    description: String,
    triggeredValue: String,
    normalRange: String,

    // Status
    status: { type: String, enum: ['active', 'acknowledged', 'resolved', 'dismissed'], default: 'active', index: true },
    acknowledgedBy: mongoose.Schema.Types.ObjectId,
    acknowledgedTime: Date,
    acknowledgedNotes: String,

    // Recipients
    alertToRoles: [String],
    sentTo: [mongoose.Schema.Types.ObjectId],
    sentTime: Date,
    readBy: [
      {
        userId: mongoose.Schema.Types.ObjectId,
        readTime: Date,
      },
    ],

    createdTime: { type: Date, default: Date.now, index: true },
  },
  { timestamps: true }
);

const securedMessageSchema = new mongoose.Schema(
  {
    conversationId: { type: String, unique: true, index: true },
    fromUserId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
    toUserIds: [mongoose.Schema.Types.ObjectId],

    // Message
    subject: String,
    body: { type: String, required: true },
    messageType: {
      type: String,
      enum: ['doctor-to-patient', 'staff-to-staff', 'staff-to-patient'],
      default: 'staff-to-staff',
    },

    // Attachments
    attachments: [
      {
        fileId: String,
        fileName: String,
        fileSize: Number,
        uploadedTime: Date,
      },
    ],

    // Status
    status: { type: String, enum: ['sent', 'received', 'read'], default: 'sent' },
    deliveryTime: Date,
    readTime: Date,
    readBy: [mongoose.Schema.Types.ObjectId],

    // Encryption
    encrypted: { type: Boolean, default: true },
    encryptionMethod: String,

    // Audit
    auditLog: [
      {
        action: String,
        userId: mongoose.Schema.Types.ObjectId,
        timestamp: Date,
      },
    ],

    relatedEncounterId: mongoose.Schema.Types.ObjectId,
    relatedPatientId: mongoose.Schema.Types.ObjectId,

    priority: { type: String, enum: ['low', 'normal', 'high', 'urgent'], default: 'normal' },
  },
  { timestamps: true, indexes: [{ conversationId: 1 }, { fromUserId: 1, createdAt: -1 }, { toUserIds: 1 }] }
);

export const Appointment = mongoose.model('Appointment', appointmentSchema);
export const BloodBank = mongoose.model('BloodBank', bloodBankSchema);
export const DietaryOrder = mongoose.model('DietaryOrder', dietaryOrderSchema);
export const ClinicalAlert = mongoose.model('ClinicalAlert', clinicalAlertSchema);
export const SecuredMessage = mongoose.model('SecuredMessage', securedMessageSchema);
