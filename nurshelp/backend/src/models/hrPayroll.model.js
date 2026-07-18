import mongoose from 'mongoose';

/**
 * HR & Payroll Management Models
 * Handles staff management, scheduling, payroll, and compliance
 */

const staffCredentialsSchema = new mongoose.Schema(
  {
    staffId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, unique: true },
    facilityId: { type: String, required: true },

    // Licenses & Certifications
    licenses: [
      {
        licenseType: String, // MD, RN, LP, PA, etc.
        licenseNumber: String,
        state: String,
        issueDate: Date,
        expiryDate: Date,
        status: { type: String, enum: ['active', 'expired', 'suspended', 'revoked'] },
        verificationDate: Date,
        verifiedBy: String,
      },
    ],

    certifications: [
      {
        certificationName: String, // BLS, ACLS, PALS, etc.
        certificationNumber: String,
        issuer: String,
        issueDate: Date,
        expiryDate: Date,
        status: { type: String, enum: ['active', 'expired'] },
      },
    ],

    // Background
    backgroundCheckDate: Date,
    backgroundCheckStatus: String,
    backgroundCheckExpiry: Date,

    // DEA (Drug Enforcement Administration)
    deaNumber: String,
    deaExpiryDate: Date,

    // Compliance
    complaintHistory: [
      {
        complaintType: String,
        date: Date,
        status: String,
        resolution: String,
      },
    ],

    notes: String,
  },
  { timestamps: true }
);

const staffShiftScheduleSchema = new mongoose.Schema(
  {
    staffId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },
    facilityId: { type: String, required: true, index: true },
    department: String,

    // Shift
    shiftDate: { type: Date, required: true, index: true },
    shiftType: { type: String, enum: ['morning', 'afternoon', 'night', 'on-call', 'off'], required: true },
    shiftStart: String, // HH:mm format
    shiftEnd: String,
    duration: Number, // hours

    // Status
    status: { type: String, enum: ['scheduled', 'open', 'swapped', 'cancelled'], default: 'scheduled' },
    assignedBy: mongoose.Schema.Types.ObjectId,
    assignmentDate: Date,

    // Attendance
    checkInTime: String,
    checkOutTime: String,
    attended: Boolean,
    lateMinutes: Number,

    notes: String,
  },
  { timestamps: true, indexes: [{ staffId: 1, shiftDate: 1 }, { facilityId: 1, shiftDate: 1 }] }
);

const payrollSchema = new mongoose.Schema(
  {
    staffId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },
    payrollPeriod: { type: String, required: true, index: true }, // e.g., "2026-04-01_to_2026-04-15"
    facilityId: String,

    // Hours
    regularHours: { type: Number, default: 0 },
    overtimeHours: { type: Number, default: 0 },
    onCallHours: { type: Number, default: 0 },
    totalHours: Number,

    // Pay
    hourlyRate: Number,
    regularPay: Number,
    overtimePay: Number,
    bonuses: Number,
    otherCompensation: Number,
    grossPay: Number,

    // Deductions
    federalTax: Number,
    stateTax: Number,
    fica: Number,
    insurance: Number,
    benefits: Number,
    otherDeductions: Number,
    totalDeductions: Number,

    // Net Pay
    netPay: Number,
    paymentMethod: { type: String, enum: ['direct-deposit', 'check', 'paycard'] },
    paymentDate: Date,

    // Status
    status: { type: String, enum: ['draft', 'pending-approval', 'approved', 'paid'], default: 'draft' },
    approvedBy: mongoose.Schema.Types.ObjectId,
    approvalDate: Date,

    notes: String,
  },
  { timestamps: true }
);

const timeAndAttendanceSchema = new mongoose.Schema(
  {
    staffId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },
    date: { type: Date, required: true, index: true },

    // Biometric Check-in/out
    checkinTime: Date,
    checkoutTime: Date,
    checkinLocation: String,
    checkoutLocation: String,

    // Hours
    hoursWorked: Number,
    status: { type: String, enum: ['present', 'absent', 'late', 'early-leave', 'on-leave'] },

    // Leave
    leaveType: String, // Vacation, Sick, Personal, etc.
    leaveHours: Number,

    // Overtime
    overtimeHours: Number,

    notes: String,
  },
  { timestamps: true }
);

const leaveManagementSchema = new mongoose.Schema(
  {
    staffId: { type: mongoose.Schema.Types.ObjectId, ref: 'ClinicalStaff', required: true, index: true },
    facilityId: String,

    // Leave Types & Balances
    vacationDays: { type: Number, default: 0 },
    vacationDaysUsed: { type: Number, default: 0 },
    vacationDaysRemaining: Number,

    sickDays: { type: Number, default: 0 },
    sickDaysUsed: { type: Number, default: 0 },
    sickDaysRemaining: Number,

    personalDays: { type: Number, default: 0 },
    personalDaysUsed: { type: Number, default: 0 },
    personalDaysRemaining: Number,

    floatingHolidays: { type: Number, default: 0 },
    floatingHolidaysUsed: { type: Number, default: 0 },

    // Leave Requests
    leaveRequests: [
      {
        requestId: mongoose.Schema.Types.ObjectId,
        startDate: Date,
        endDate: Date,
        leaveType: String,
        reason: String,
        status: { type: String, enum: ['pending', 'approved', 'denied'], default: 'pending' },
        approvedBy: mongoose.Schema.Types.ObjectId,
        approvalDate: Date,
      },
    ],

    // Accrual
    accrualRate: Number, // days per month
    lastAccrualDate: Date,
    nextAccrualDate: Date,

    yearStartDate: Date,
  },
  { timestamps: true }
);

export const StaffCredentials = mongoose.model('StaffCredentials', staffCredentialsSchema);
export const StaffShiftSchedule = mongoose.model('StaffShiftSchedule', staffShiftScheduleSchema);
export const Payroll = mongoose.model('Payroll', payrollSchema);
export const TimeAndAttendance = mongoose.model('TimeAndAttendance', timeAndAttendanceSchema);
export const LeaveManagement = mongoose.model('LeaveManagement', leaveManagementSchema);
