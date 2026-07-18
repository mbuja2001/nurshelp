import mongoose from 'mongoose';

/**
 * Additional System Models
 * Biometric, Feedback, Mobile App, Portal Users
 */

const biometricDeviceSchema = new mongoose.Schema(
  {
    // Device Info
    deviceId: { type: String, unique: true, required: true, index: true },
    deviceType: {
      type: String,
      enum: ['fingerprint', 'facial-recognition', 'iris-scan', 'badge-reader', 'rfid'],
      required: true,
    },
    manufacturer: String,
    model: String,
    serialNumber: String,

    // Location
    facilityId: String,
    location: String,
    zone: String,

    // Status
    status: { type: String, enum: ['active', 'maintenance', 'inactive'], default: 'active' },
    lastCalibrationDate: Date,
    nextCalibrationDate: Date,
    lastMaintenanceDate: Date,

    // Connectivity
    ipAddress: String,
    macAddress: String,
    isConnected: Boolean,
    lastSyncTime: Date,

    // Configuration
    accessLevel: Number,
    authorizedUsers: [mongoose.Schema.Types.ObjectId],
    accessGroups: [String],

    notes: String,
  },
  { timestamps: true }
);

const biometricAuthenticationSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },
    biometricType: { type: String, required: true },

    // Template Data (encrypted)
    biometricTemplate: String,
    templateVersion: Number,
    enrollmentDate: Date,
    enrolledBy: mongoose.Schema.Types.ObjectId,

    // Status
    status: { type: String, enum: ['active', 'inactive', 'suspended'], default: 'active' },
    lastUsedDate: Date,

    // Quality Metrics
    qualityScore: Number,
    matchThreshold: Number,

    // Multi-factor
    backupBiometric: Boolean,
    backupBiometricType: String,

    notes: String,
  },
  { timestamps: true }
);

const accessLogSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },
    deviceId: String,
    biometricType: String,

    // Access Details
    accessTime: { type: Date, default: Date.now, index: true },
    accessResult: { type: String, enum: ['success', 'failed', 'denied'], default: 'success' },
    accessType: { type: String, enum: ['entry', 'exit', 'system-login'], default: 'entry' },
    location: String,

    // Matching
    matchScore: Number,
    attempts: Number,

    // Audit
    facilityId: String,
    notes: String,
  },
  { timestamps: true, indexes: [{ userId: 1, accessTime: -1 }, { accessTime: 1 }] }
);

const feedbackSchema = new mongoose.Schema(
  {
    // Subject
    feedbackType: {
      type: String,
      enum: ['patient-satisfaction', 'staff-feedback', 'complaint', 'suggestion', 'compliment'],
      required: true,
      index: true,
    },

    // From/To
    fromUserId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
    fromUserType: { type: String, enum: ['patient', 'staff'] },
    toUserId: mongoose.Schema.Types.ObjectId, // If directed to specific person
    toUserName: String,

    // Content
    title: String,
    description: String,
    rating: { type: Number, min: 1, max: 5 },
    department: String,
    serviceArea: String,

    // Related
    encounterId: mongoose.Schema.Types.ObjectId,
    appointmentId: mongoose.Schema.Types.ObjectId,

    // Status
    status: { type: String, enum: ['submitted', 'acknowledged', 'in-review', 'resolved', 'closed'], default: 'submitted', index: true },
    reviewedBy: mongoose.Schema.Types.ObjectId,
    reviewedDate: Date,
    reviewNotes: String,

    // Resolution
    actionTaken: String,
    resolutionDate: Date,
    resolutionNotes: String,

    // Follow-up
    followUpRequired: Boolean,
    followUpDate: Date,
    followUpNotes: String,

    // Attachments
    attachments: [
      {
        fileName: String,
        fileSize: Number,
        fileType: String,
      },
    ],

    // Metadata
    submissionDate: { type: Date, default: Date.now, index: true },
    facilityId: String,
    anonymous: Boolean,
  },
  { timestamps: true }
);

const mobileDeviceSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },

    // Device Info
    deviceId: { type: String, unique: true, required: true, index: true },
    deviceType: { type: String, enum: ['ios', 'android', 'web'] },
    osVersion: String,
    appVersion: String,
    deviceModel: String,

    // Registration
    registrationDate: { type: Date, default: Date.now },
    lastActiveDate: Date,
    isActive: { type: Boolean, default: true },

    // Push Notifications
    pushNotificationToken: String,
    pushNotificationEnabled: { type: Boolean, default: true },

    // Sync
    lastSyncDate: Date,
    syncFrequency: String, // e.g., "15 minutes"
    offlineMode: Boolean,
    cachedDataSize: Number, // bytes

    // Security
    deviceEncrypted: Boolean,
    biometricEnabled: Boolean,
    pinRequired: Boolean,

    // Location
    lastKnownLocation: {
      latitude: Number,
      longitude: Number,
      timestamp: Date,
    },

    // Permissions
    permissions: [
      {
        permission: String,
        granted: Boolean,
        grantedDate: Date,
      },
    ],

    notes: String,
  },
  { timestamps: true }
);

const patientPortalAccountSchema = new mongoose.Schema(
  {
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true, unique: true, index: true },

    // Account Info
    email: { type: String, required: true, unique: true, index: true },
    phone: String,
    username: String,

    // Authentication
    passwordHash: String,
    passwordChangedDate: Date,
    temporaryPassword: Boolean,
    passwordExpiryDate: Date,

    // 2FA
    twoFactorEnabled: Boolean,
    twoFactorMethod: { type: String, enum: ['sms', 'email', 'authenticator'] },
    backup2FACode: String,

    // Access
    lastLoginDate: Date,
    lastLoginIP: String,
    accountStatus: { type: String, enum: ['active', 'inactive', 'locked', 'suspended'], default: 'inactive' },
    accountActivationToken: String,
    accountActivationExpiry: Date,

    // Permissions
    canViewMedicalRecords: { type: Boolean, default: true },
    canViewBilling: { type: Boolean, default: true },
    canRequestPrescriptionRefill: { type: Boolean, default: true },
    canScheduleAppointments: { type: Boolean, default: true },
    canMessaging: { type: Boolean, default: true },

    // Communication Preferences
    emailNotifications: { type: Boolean, default: true },
    smsNotifications: { type: Boolean, default: true },
    appointmentReminders: { type: Boolean, default: true },
    resultNotifications: { type: Boolean, default: true },

    // Data Access Log
    dataAccessLog: [
      {
        accessType: String,
        accessDate: Date,
        dataAccessed: String,
      },
    ],

    // Close Account
    accountClosureRequested: Boolean,
    accountClosureDate: Date,
    closureReason: String,

    notes: String,
  },
  { timestamps: true }
);

const physicianPortalAccountSchema = new mongoose.Schema(
  {
    staffId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, unique: true, index: true },

    // Account Info
    email: { type: String, required: true, unique: true },
    phone: String,

    // Authentication
    passwordHash: String,
    passwordChangedDate: Date,
    twoFactorEnabled: Boolean,

    // Access
    lastLoginDate: Date,
    accountStatus: { type: String, enum: ['active', 'inactive'], default: 'active' },

    // Permissions
    canViewPatientList: { type: Boolean, default: true },
    canOrderLabs: { type: Boolean, default: true },
    canOrderRadiology: { type: Boolean, default: true },
    canPrescribeMedications: { type: Boolean, default: true },
    canMessaging: { type: Boolean, default: true },
    canViewBilling: { type: Boolean, default: true },

    // Integration
    hl7CertificateId: String,
    ediAccess: Boolean,

    notes: String,
  },
  { timestamps: true }
);

export const BiometricDevice = mongoose.model('BiometricDevice', biometricDeviceSchema);
export const BiometricAuthentication = mongoose.model('BiometricAuthentication', biometricAuthenticationSchema);
export const AccessLog = mongoose.model('AccessLog', accessLogSchema);
export const Feedback = mongoose.model('Feedback', feedbackSchema);
export const MobileDevice = mongoose.model('MobileDevice', mobileDeviceSchema);
export const PatientPortalAccount = mongoose.model('PatientPortalAccount', patientPortalAccountSchema);
export const PhysicianPortalAccount = mongoose.model('PhysicianPortalAccount', physicianPortalAccountSchema);
