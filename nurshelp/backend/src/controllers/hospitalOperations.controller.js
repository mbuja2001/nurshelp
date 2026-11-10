import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { Encounter } from '../models/encounter.model.js';
import { Shift } from '../models/shift.model.js';
import { Bed } from '../models/bed.model.js';
import { Pharmaceutical } from '../models/pharmacy.model.js';
import { AuditLog } from '../models/auditLog.model.js';
import { Hospital } from '../models/hospital.model.js';
import { Ward } from '../models/ward.model.js';
import {
  canonicalizeServiceCoverage,
  getConfiguredHospitalServiceSpecialties,
  inferAssignmentServiceCoverage,
  validateHospitalShiftCoverage
} from '../utils/hospitalShiftCoverage.js';
import { createSyntheticHistoricalEncounter, historicalEncounterDate, normalizeHistoricalDays } from '../utils/syntheticHistoricalEncounter.js';
import { getScheduledShiftAutoEndAt, getScheduledShiftEndAt } from '../utils/scheduledShiftEnd.js';
import { departmentCodeFromName, resolveConfiguredDepartment } from '../utils/hospitalDepartment.js';
import { getAllowedShiftDuties } from '../config/hospitalStaffRoleCatalog.js';
import { normalizeShiftAssignmentRole } from '../utils/shiftTeamAssignment.js';
import { HOSPITAL_DEPARTMENTS, HOSPITAL_SPECIALTIES } from '../config/hospitalStructureCatalog.js';
import { findHospitalStaffRole, HOSPITAL_STAFF_ROLES, legacyHospitalStaffRole } from '../config/hospitalStaffRoleCatalog.js';
import bcrypt from 'bcrypt';

const hospitalIdFor = (req) => req.user?.hospitalId || req.hospitalId;
const dayStart = (value = new Date()) => {
  const date = new Date(value);
  date.setHours(0, 0, 0, 0);
  return date;
};
const roleOf = (staff) => String(staff?.systemRole || staff?.role || '').toUpperCase();
const nameOf = (staff) => [staff?.firstName, staff?.lastName].filter(Boolean).join(' ') || staff?.name || 'Staff member';
const durationHours = (start, end) => {
  const [startHour, startMinute] = String(start).split(':').map(Number);
  const [endHour, endMinute] = String(end).split(':').map(Number);
  let minutes = (endHour * 60 + endMinute) - (startHour * 60 + startMinute);
  if (minutes <= 0) minutes += 24 * 60;
  return minutes / 60;
};

export const getHospitalStructure = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const hospital = await Hospital.findById(hospitalId).select('departments').lean();
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    return res.json({
      success: true,
      departments: hospital.departments || [],
      departmentOptions: HOSPITAL_DEPARTMENTS,
      specialties: HOSPITAL_SPECIALTIES,
      staffRoles: HOSPITAL_STAFF_ROLES
    });
  } catch (error) {
    console.error('[HOSPITAL STRUCTURE] Load error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load hospital structure' });
  }
};

export const updateHospitalStructure = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const requested = req.body?.departments;
    if (!Array.isArray(requested) || requested.length < 1 || requested.length > 100) {
      return res.status(400).json({ success: false, message: 'Select between 1 and 100 departments' });
    }
    const hospital = await Hospital.findById(hospitalId).select('departments');
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });

    const allowedSpecialties = new Set(HOSPITAL_SPECIALTIES.map((item) => item.name));
    const seenDepartments = new Set();
    const seenDepartmentIds = new Set();
    const departments = [];
    for (const input of requested) {
      const name = String(input?.name || '').trim().replace(/\s+/g, ' ');
      const normalizedName = name.toLowerCase();
      if (
        name.length < 2
        || name.length > 60
        || !/^[\p{L}\p{N}][\p{L}\p{N} &().'-]*$/u.test(name)
        || seenDepartments.has(normalizedName)
      ) {
        return res.status(400).json({ success: false, message: `Invalid or duplicate department: ${name || 'unknown'}` });
      }
      seenDepartments.add(normalizedName);
      const serviceTypes = [...new Set(Array.isArray(input.serviceTypes) ? input.serviceTypes : [])];
      if (!serviceTypes.length || serviceTypes.some((type) => !['OUTPATIENT', 'INPATIENT', 'CLINICAL_SUPPORT'].includes(type))) {
        return res.status(400).json({ success: false, message: `Select valid service types for ${name}` });
      }
      const specialties = [...new Set((Array.isArray(input.specialties) ? input.specialties : []).map((item) => String(item?.name || item || '').trim()).filter(Boolean))];
      if (specialties.some((specialty) => !allowedSpecialties.has(specialty))) {
        return res.status(400).json({ success: false, message: `Unknown specialty configured for ${name}` });
      }
      if (!specialties.length) return res.status(400).json({ success: false, message: `Select at least one specialty for ${name}` });
      const existingDepartment = (hospital.departments || []).find((item) => (
        item.name?.trim().toLowerCase() === normalizedName
        || (input.departmentId && item.departmentId === input.departmentId)
      ));
      const departmentId = existingDepartment?.departmentId || departmentCodeFromName(name);
      if (seenDepartmentIds.has(departmentId)) {
        return res.status(400).json({ success: false, message: `Department names must produce unique IDs: ${name}` });
      }
      seenDepartmentIds.add(departmentId);
      departments.push({
        name,
        departmentId,
        serviceTypes,
        specialties: specialties.map((specialty) => {
          const existingSpecialty = existingDepartment?.specialties?.find((item) => (item.name || item) === specialty);
          return typeof existingSpecialty === 'string'
            ? { name: specialty }
            : { name: specialty, code: existingSpecialty?.code || null, codeSystem: existingSpecialty?.codeSystem || null, display: existingSpecialty?.display || specialty };
        }),
        isActive: input.isActive !== false
      });
    }

    const removedDepartments = (hospital.departments || []).map((item) => item.name).filter((name) => !seenDepartments.has(name.trim().toLowerCase()));
    if (removedDepartments.length) {
      const linkedWard = await Ward.findOne({ hospitalId, departmentName: { $in: removedDepartments }, isActive: true }).select('wardName departmentName').lean();
      if (linkedWard) {
        return res.status(409).json({ success: false, message: `Deactivate or reassign ${linkedWard.wardName} before removing its ${linkedWard.departmentName} department` });
      }
    }
    const before = hospital.departments.map((item) => ({ name: item.name, specialties: item.specialties || [], serviceTypes: item.serviceTypes || [], isActive: item.isActive }));
    hospital.departments = departments;
    await hospital.save();
    const audit = await AuditLog.create({
      action: 'HOSPITAL_STRUCTURE_UPDATED',
      actorId: req.user._id,
      hospitalId,
      resourceType: 'HOSPITAL',
      authorized: true,
      changeDetails: { before, after: departments },
      description: 'Administrator updated hospital departments and specialties'
    });
    return res.json({ success: true, departments: hospital.departments, eventId: String(audit._id) });
  } catch (error) {
    console.error('[HOSPITAL STRUCTURE] Save error:', error);
    return res.status(500).json({ success: false, message: 'Failed to save hospital structure' });
  }
};

export const getHospitalWards = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const [hospital, wards] = await Promise.all([
      Hospital.findById(hospitalId).select('departments').lean(),
      Ward.find({ hospitalId }).sort({ wardName: 1 }).lean()
    ]);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    return res.json({ success: true, departments: hospital.departments || [], wards });
  } catch (error) {
    console.error('[HOSPITAL WARDS] Load error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load hospital wards' });
  }
};

export const saveHospitalWards = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const hospital = await Hospital.findById(hospitalId).select('departments');
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    const activeDepartments = (hospital.departments || []).filter((department) => department.isActive !== false);
    const activeDepartmentsByName = new Map(activeDepartments.map((department) => [department.name.trim().toLowerCase(), department]));
    const requested = req.body?.wards;
    if (!Array.isArray(requested) || requested.length > 300) {
      return res.status(400).json({ success: false, message: 'Provide a valid ward list (maximum 300 wards)' });
    }

    const seenIds = new Set();
    const normalized = [];
    for (const input of requested) {
      const wardId = String(input?.wardId || '').trim();
      const wardName = String(input?.wardName || '').trim();
      const requestedDepartmentName = String(input?.departmentName || '').trim();
      const configuredDepartment = activeDepartmentsByName.get(requestedDepartmentName.toLowerCase());
      const departmentName = configuredDepartment?.name || requestedDepartmentName;
      const careTier = String(input?.careTier || '').trim();
      const acuityLevelAllowed = [...new Set(Array.isArray(input?.acuityLevelAllowed) ? input.acuityLevelAllowed : [])];
      const totalBeds = Number(input?.totalBeds);
      if (!wardId || !wardName || (input?.isActive !== false && !configuredDepartment)) {
        return res.status(400).json({ success: false, message: `Ward ${wardName || wardId || '(unnamed)'} needs a name, ID, and active hospital department` });
      }
      if (seenIds.has(wardId)) return res.status(400).json({ success: false, message: `Duplicate ward ID: ${wardId}` });
      seenIds.add(wardId);
      if (!['CRITICAL_CARE', 'STEP_DOWN', 'GENERAL_INPATIENT', 'ISOLATION'].includes(careTier)) {
        return res.status(400).json({ success: false, message: `Ward ${wardName} needs a valid care tier` });
      }
      if (!acuityLevelAllowed.length || acuityLevelAllowed.some((color) => !['Red', 'Orange', 'Yellow', 'Green', 'Blue'].includes(color))) {
        return res.status(400).json({ success: false, message: `Ward ${wardName} needs one or more valid SATS acuity colors` });
      }
      if (!Number.isInteger(totalBeds) || totalBeds < 1 || totalBeds > 2000) {
        return res.status(400).json({ success: false, message: `Ward ${wardName} capacity must be between 1 and 2000 beds` });
      }
      const normalizedDepartmentName = departmentName.toUpperCase();
      const department = normalizedDepartmentName === 'ED' ? 'ED'
        : normalizedDepartmentName === 'ICU' ? 'ICU'
          : normalizedDepartmentName === 'OR' ? 'OR'
            : normalizedDepartmentName === 'OB' ? 'MATERNITY'
              : normalizedDepartmentName === 'PEDIATRICS' ? 'PEDIATRICS'
                : normalizedDepartmentName === 'PSYCHIATRY' ? 'PSYCH'
                  : 'GENERAL';
      normalized.push({
        wardId,
        wardName,
        department,
        departmentId: configuredDepartment?.departmentId || departmentCodeFromName(departmentName),
        departmentName,
        fhirLocationType: 'wa',
        careTier,
        acuityLevelAllowed,
        totalBeds,
        location: {
          floor: Math.max(0, Number(input?.location?.floor) || 1),
          wing: String(input?.location?.wing || 'Main').trim(),
          area: String(input?.location?.area || '').trim()
        },
        isActive: input?.isActive !== false
      });
    }

    const previous = await Ward.find({ hospitalId }).lean();
    const wardsToSave = [];
    for (const wardData of normalized) {
      const ward = await Ward.findOne({ hospitalId, wardId: wardData.wardId }) || new Ward({ hospitalId, wardId: wardData.wardId });
      const occupiedCapacity = (ward.occupiedBeds || 0) + (ward.cleaningBeds || 0) + (ward.maintenanceBeds || 0);
      if (wardData.totalBeds < occupiedCapacity) {
        return res.status(409).json({ success: false, message: `Ward ${wardData.wardName} capacity cannot be below its occupied, cleaning, and maintenance beds (${occupiedCapacity})` });
      }
      Object.assign(ward, wardData, { hospitalId });
      wardsToSave.push(ward);
    }
    for (const ward of wardsToSave) {
      await ward.save();
    }
    const audit = await AuditLog.create({
      action: 'HOSPITAL_STRUCTURE_UPDATED',
      actorId: req.user._id,
      hospitalId,
      resourceType: 'WARD',
      authorized: true,
      changeDetails: { previousWardCount: previous.length, savedWardCount: normalized.length, wards: normalized },
      description: 'Administrator created or updated hospital ward configuration'
    });
    const wards = await Ward.find({ hospitalId }).sort({ wardName: 1 }).lean();
    return res.json({ success: true, wards, eventId: String(audit._id) });
  } catch (error) {
    console.error('[HOSPITAL WARDS] Save error:', error);
    return res.status(500).json({ success: false, message: 'Failed to save hospital wards' });
  }
};

const requireHospital = (req, res) => {
  const hospitalId = hospitalIdFor(req);
  if (!hospitalId) {
    res.status(401).json({ success: false, message: 'Hospital context is required' });
    return null;
  }
  return hospitalId;
};

export const getHospitalOperationsStatus = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const today = dayStart();
    const [staff, shifts, encounters, beds, pharmaceuticals] = await Promise.all([
      ClinicalStaff.find({ hospitalId, status: { $ne: 'TERMINATED' } }).select('firstName lastName systemRole role assignment status').lean(),
      Shift.find({ hospitalId, date: { $gte: today } }).select('shiftName department shiftStart shiftEnd status assignedStaff teamAssignment date').lean(),
      Encounter.countDocuments({ hospitalId }),
      Bed.countDocuments({ hospitalId }),
      Pharmaceutical.countDocuments({ hospitalId })
    ]);
    return res.json({
      success: true,
      hospitalId: String(hospitalId),
      staff: { total: staff.length, byRole: staff.reduce((result, member) => { const role = roleOf(member); result[role] = (result[role] || 0) + 1; return result; }, {}) },
      shifts: { today: shifts.length, published: shifts.filter((shift) => shift.status === 'PUBLISHED').length, assigned: shifts.reduce((total, shift) => total + (shift.assignedStaff?.length || 0), 0) },
      encounters: { total: encounters },
      beds: { total: beds },
      pharmaceuticals: { total: pharmaceuticals },
      readiness: { staff: staff.length > 0, shifts: shifts.length > 0, encounters: encounters > 0, beds: beds > 0, pharmaceuticals: pharmaceuticals > 0 }
    });
  } catch (error) {
    console.error('[HOSPITAL OPERATIONS STATUS] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load hospital operations status' });
  }
};

export const getHospitalActivity = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const limit = Math.min(Math.max(Number.parseInt(req.query?.limit, 10) || 40, 1), 100);
    const [activity, staff, shifts, encounters, beds, pharmaceuticals] = await Promise.all([
      AuditLog.find({ hospitalId }).sort({ timestamp: -1 }).limit(limit).select('action actorId resourceType changeDetails authorized timestamp description').lean(),
      ClinicalStaff.countDocuments({ hospitalId, status: { $ne: 'TERMINATED' } }),
      Shift.countDocuments({ hospitalId }),
      Encounter.countDocuments({ hospitalId }),
      Bed.countDocuments({ hospitalId }),
      Pharmaceutical.countDocuments({ hospitalId })
    ]);
    return res.json({
      success: true,
      hospitalId: String(hospitalId),
      summary: { staff, shifts, encounters, beds, pharmaceuticals },
      activity: activity.map((event) => ({ ...event, id: String(event._id), actorId: event.actorId ? String(event.actorId) : null }))
    });
  } catch (error) {
    console.error('[HOSPITAL ACTIVITY] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to load hospital activity' });
  }
};

export const recordHospitalShiftEvent = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const event = String(req.body?.event || '').toUpperCase();
    if (!['STARTED', 'ENDED'].includes(event)) return res.status(400).json({ success: false, message: 'Shift event must be STARTED or ENDED' });
    const shiftInput = req.body?.shift || {};
    const requestedOperatingDate = String(req.body?.operatingDate || '').trim();
    if (!shiftInput.id && !shiftInput.shiftID) {
      return res.status(400).json({ success: false, message: 'A persisted shift ID is required' });
    }
    let shift = await Shift.findOne({
      hospitalId,
      ...(shiftInput.id ? { _id: shiftInput.id } : { shiftID: shiftInput.shiftID }),
    });
    if (!shift) return res.status(404).json({ success: false, message: 'Shift not found for this hospital' });

    const actorId = req.user?._id || req.user?.id;
    const occurredAt = new Date();
    if (event === 'STARTED') {
      if (shift.status !== 'PUBLISHED') {
        return res.status(409).json({ success: false, message: `Only published shifts can be started (current status: ${shift.status})` });
      }
      const hasAssignedStaff = (shift.assignedStaff || []).some((assignment) => assignment.status !== 'ABSENT');
      if (!hasAssignedStaff) {
        return res.status(409).json({ success: false, message: 'Assign at least one staff member before starting the shift' });
      }
      const shiftDay = shift.date ? new Date(shift.date).toISOString().slice(0, 10) : null;
      if (!shiftDay) return res.status(409).json({ success: false, message: 'A dated shift is required before it can be started' });
      const operatingDate = /^\d{4}-\d{2}-\d{2}$/.test(requestedOperatingDate)
        ? requestedOperatingDate
        : occurredAt.toISOString().slice(0, 10);
      if (shiftDay !== operatingDate) {
        return res.status(409).json({ success: false, message: 'A shift can only be started on its scheduled date' });
      }
      shift.status = 'IN_PROGRESS';
      shift.approvedBy = actorId;
      shift.approvedAt = occurredAt;
      shift.startedBy = actorId;
      shift.startedAt = occurredAt;
      shift.completedBy = null;
      shift.completedAt = null;
      shift.completedSource = null;
    } else {
      if (shift.status !== 'IN_PROGRESS') {
        return res.status(409).json({ success: false, message: `Only an in-progress shift can be ended (current status: ${shift.status})` });
      }
      const scheduledEndAt = getScheduledShiftEndAt(shift);
      const autoEndAt = getScheduledShiftAutoEndAt(shift);
      if (!scheduledEndAt || !autoEndAt) {
        return res.status(409).json({ success: false, message: 'The scheduled shift end could not be verified. The shift cannot be ended manually.' });
      }
      if (occurredAt < scheduledEndAt) {
        return res.status(409).json({ success: false, message: 'Early hospital shift stops require the separate dual-admin hard-stop workflow, which is not enabled.' });
      }
      if (occurredAt >= autoEndAt) {
        return res.status(409).json({ success: false, message: 'The five-minute scheduled-end authorization window has expired; the scheduled completion job will close this shift.' });
      }
      const completedShift = await Shift.findOneAndUpdate(
        { _id: shift._id, status: 'IN_PROGRESS' },
        { $set: { status: 'COMPLETED', completedBy: actorId, completedAt: occurredAt, completedSource: 'ADMIN_SCHEDULED' } },
        { new: true }
      );
      if (!completedShift) {
        return res.status(409).json({ success: false, message: 'The shift changed before administrator end could be recorded. Refresh the shift status.' });
      }
      shift = completedShift;
    }

    if (event === 'STARTED') await shift.save();
    const auditAction = `HOSPITAL_SHIFT_${event}`;
    const audit = await AuditLog.create({
      action: auditAction,
      actorId,
      hospitalId,
      resourceId: shift._id,
      resourceType: 'SHIFT',
      authorized: true,
      changeDetails: {
        shiftId: String(shift._id),
        shiftID: shift.shiftID,
        shiftName: shift.shiftName,
        department: shift.department,
        specialty: shift.specialty,
        wardId: shift.wardId ? String(shift.wardId) : null,
        shiftStart: shift.shiftStart,
        shiftEnd: shift.shiftEnd,
        date: shift.date,
        staffCount: shift.assignedStaff?.filter((assignment) => assignment.status !== 'ABSENT').length || 0,
        operatingDate: /^\d{4}-\d{2}-\d{2}$/.test(requestedOperatingDate) ? requestedOperatingDate : occurredAt.toISOString().slice(0, 10),
        status: shift.status,
        approvedBy: shift.approvedBy ? String(shift.approvedBy) : null,
        approvedAt: shift.approvedAt,
        startedAt: shift.startedAt,
        completedAt: shift.completedAt,
        completedSource: shift.completedSource,
      },
      description: `Administrator ${event.toLowerCase()} hospital shift ${shift.shiftName}`
    });
    return res.status(201).json({ success: true, event: audit });
  } catch (error) {
    console.error('[HOSPITAL SHIFT EVENT] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to record hospital shift event' });
  }
};

export const bulkOnboardHospitalStaff = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const input = Array.isArray(req.body?.staff) ? req.body.staff : [];
    if (!input.length || input.length > 500) return res.status(400).json({ success: false, message: 'Provide between 1 and 500 staff members' });

    const hospital = await Hospital.findById(hospitalId).select('departments').lean();
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    const allowedDepartments = new Map();
    (hospital.departments || []).filter((item) => item.isActive !== false).forEach((item) => {
      allowedDepartments.set(String(item.name).trim().toUpperCase(), item);
      if (item.departmentId) allowedDepartments.set(String(item.departmentId).trim().toUpperCase(), item);
    });
    const allowedTitles = ['MD', 'DO', 'RN', 'NP', 'PA', 'PhD'];
    const existingEmails = new Set((await ClinicalStaff.find({ hospitalId }).select('email').lean()).map((staff) => String(staff.email).toLowerCase()));
    const seenEmails = new Set();
    const documents = [];

    for (let index = 0; index < input.length; index += 1) {
      const item = input[index] || {};
      const firstName = String(item.firstName || item.name || '').trim();
      const lastName = String(item.lastName || item.surname || '').trim();
      const email = String(item.email || '').trim().toLowerCase();
      const positionCode = String(item.positionCode || '').trim();
      const position = positionCode
        ? findHospitalStaffRole(positionCode)
        : legacyHospitalStaffRole(item.role || item.systemRole || 'NURSE');
      const departmentInput = String(item.departmentId || item.department || '').trim();
      const configuredDepartment = allowedDepartments.get(departmentInput.toUpperCase());
      const department = String(configuredDepartment?.name || '').trim();
      const specialty = String(item.specialty || item.clinicalSpecialty || '').trim();
      const title = String(item.title || position?.defaultTitle || '').trim();
      const password = String(item.password || '').trim();
      const licenseNumber = String(item.licenseNumber || item.license || (position?.requiresCredentials ? `PENDING-${Date.now()}-${index + 1}` : '')).trim();

      if (!firstName || !lastName || !email || !password) return res.status(400).json({ success: false, message: `Staff row ${index + 1} requires firstName, lastName, email, and password` });
      if (!position) return res.status(400).json({ success: false, message: `Staff row ${index + 1} has an invalid staff role` });
      if (!configuredDepartment) return res.status(400).json({ success: false, message: `Staff row ${index + 1} must use an active department configured for this hospital` });
      if (title && !allowedTitles.includes(title)) return res.status(400).json({ success: false, message: `Staff row ${index + 1} has an invalid credential title` });
      if (position.requiresCredentials && (!title || !licenseNumber)) return res.status(400).json({ success: false, message: `Staff row ${index + 1} requires a credential title and license number` });
      if (password.length < 8) return res.status(400).json({ success: false, message: `Staff row ${index + 1} password must be at least 8 characters` });
      if (existingEmails.has(email) || seenEmails.has(email)) return res.status(409).json({ success: false, message: `Duplicate staff email: ${email}` });
      seenEmails.add(email);
      const configuredSpecialties = (configuredDepartment.specialties || [])
        .map((configured) => String(configured?.name || configured || '').trim());
      const canonicalSpecialty = configuredSpecialties.find((configured) => configured.toLowerCase() === specialty.toLowerCase());
      if (!canonicalSpecialty) {
        return res.status(400).json({
          success: false,
          code: 'SPECIALTY_NOT_CONFIGURED',
          message: `Staff row ${index + 1} must use a specialty configured for ${department}`
        });
      }

      documents.push({
        hospitalId,
        firstName,
        lastName,
        email,
        password: await bcrypt.hash(password, 12),
        systemRole: position.systemRole,
        positionCode: position.code,
        positionTitle: position.label,
        clinicalSpecialty: canonicalSpecialty,
        credentials: {
          ...(title ? { title } : {}),
          ...(licenseNumber ? { licenseNumber } : {}),
          ...(item.npiNumber ? { npiNumber: String(item.npiNumber).trim() } : {})
        },
        assignment: { department, isOnCall: Boolean(item.isOnCall) },
        status: 'ACTIVE',
        metadata: { createdBy: req.user._id }
      });
    }

    const created = await ClinicalStaff.insertMany(documents, { ordered: true });
    await AuditLog.create({
      action: 'HOSPITAL_STAFF_ONBOARDED', actorId: req.user._id, hospitalId,
      resourceType: 'STAFF', authorized: true,
      changeDetails: {
        requested: input.length,
        created: created.length,
        roles: created.reduce((result, staff) => { result[staff.positionTitle] = (result[staff.positionTitle] || 0) + 1; return result; }, {})
      },
      description: 'Administrator onboarded hospital staff in bulk'
    });
    return res.status(201).json({
      success: true,
      message: `${created.length} staff members onboarded`,
      staff: created.map((staff) => ({
        id: String(staff._id),
        firstName: staff.firstName,
        lastName: staff.lastName,
        email: staff.email,
        role: staff.systemRole,
        positionCode: staff.positionCode,
        positionTitle: staff.positionTitle,
        department: staff.assignment?.department,
        specialty: staff.clinicalSpecialty
      }))
    });
  } catch (error) {
    console.error('[HOSPITAL STAFF ONBOARDING] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to onboard hospital staff', detail: error.message });
  }
};

export const updateHospitalStaff = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const [hospital, staff] = await Promise.all([
      Hospital.findById(hospitalId).select('departments').lean(),
      ClinicalStaff.findOne({ _id: req.params.staffID, hospitalId }),
    ]);
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    if (!staff) return res.status(404).json({ success: false, message: 'Staff member not found in this hospital' });

    const positionCode = String(req.body?.positionCode || staff.positionCode || '').trim();
    const position = findHospitalStaffRole(positionCode);
    if (!position) {
      return res.status(400).json({
        success: false,
        code: 'INVALID_STAFF_POSITION',
        message: 'Select a valid staff position from the hospital role catalog'
      });
    }

    const departmentInput = String(req.body?.assignment?.department || staff.assignment?.department || '').trim().toUpperCase();
    const department = (hospital.departments || []).find((item) => (
      item.isActive !== false
      && (
        String(item.name || '').trim().toUpperCase() === departmentInput
        || String(item.departmentId || '').trim().toUpperCase() === departmentInput
      )
    ));
    if (!department) {
      return res.status(400).json({
        success: false,
        code: 'DEPARTMENT_NOT_CONFIGURED',
        message: 'Select an active department configured for this hospital'
      });
    }

    const requestedSpecialty = String(req.body?.clinicalSpecialty || req.body?.specialty || staff.clinicalSpecialty || '').trim();
    const specialty = (department.specialties || [])
      .map((item) => String(item?.name || item || '').trim())
      .find((item) => item.toLowerCase() === requestedSpecialty.toLowerCase());
    if (!specialty) {
      return res.status(400).json({
        success: false,
        code: 'SPECIALTY_NOT_CONFIGURED',
        message: `Select a specialty configured for ${department.name}`
      });
    }

    const firstName = String(req.body?.firstName ?? staff.firstName).trim();
    const lastName = String(req.body?.lastName ?? staff.lastName).trim();
    const email = String(req.body?.email ?? staff.email).trim().toLowerCase();
    if (!firstName || !lastName || !email) {
      return res.status(400).json({ success: false, message: 'First name, last name, and work email are required' });
    }
    const duplicateEmail = await ClinicalStaff.findOne({
      hospitalId,
      email,
      _id: { $ne: staff._id }
    });
    if (duplicateEmail) return res.status(409).json({ success: false, message: `Duplicate staff email: ${email}` });

    const credentialsInput = req.body?.credentials || {};
    const title = String(credentialsInput.title || '').trim();
    const licenseNumber = String(credentialsInput.licenseNumber || '').trim();
    const allowedTitles = ['MD', 'DO', 'RN', 'NP', 'PA', 'PhD'];
    if (position.requiresCredentials && (!allowedTitles.includes(title) || !licenseNumber)) {
      return res.status(400).json({
        success: false,
        code: 'STAFF_CREDENTIALS_REQUIRED',
        message: `${position.label} requires a valid credential title and license number`
      });
    }
    if (title && !allowedTitles.includes(title)) {
      return res.status(400).json({ success: false, message: 'Select a valid credential title' });
    }
    const status = String(req.body?.status || staff.status || 'ACTIVE');
    if (!['ACTIVE', 'ON_LEAVE', 'TERMINATED', 'PENDING_CREDENTIALING'].includes(status)) {
      return res.status(400).json({ success: false, message: 'Select a valid staff status' });
    }

    const password = String(req.body?.password || '').trim();
    if (password && password.length < 8) {
      return res.status(400).json({ success: false, message: 'Password must be at least 8 characters long' });
    }

    staff.firstName = firstName;
    staff.lastName = lastName;
    staff.email = email;
    staff.systemRole = position.systemRole;
    staff.positionCode = position.code;
    staff.positionTitle = position.label;
    staff.positionCodeNeedsBackfill = false;
    if (staff.clinicalSpecialty !== specialty) staff.clinicalSpecialtyId = null;
    staff.clinicalSpecialty = specialty;
    staff.assignment = { department: department.name, isOnCall: staff.assignment?.isOnCall || false };
    staff.credentials = {
      ...(position.requiresCredentials ? { title, licenseNumber } : {}),
      ...(String(credentialsInput.npiNumber || '').trim() ? { npiNumber: String(credentialsInput.npiNumber).trim() } : {})
    };
    staff.status = status;
    if (password) staff.password = await bcrypt.hash(password, 12);
    await staff.save();

    return res.status(200).json({
      success: true,
      message: 'Staff member updated successfully',
      staff: {
        id: String(staff._id),
        firstName: staff.firstName,
        lastName: staff.lastName,
        email: staff.email,
        role: staff.systemRole,
        positionCode: staff.positionCode,
        positionTitle: staff.positionTitle,
        department: staff.assignment?.department,
        specialty: staff.clinicalSpecialty,
        status: staff.status
      }
    });
  } catch (error) {
    console.error('[HOSPITAL STAFF UPDATE] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to update staff credentials' });
  }
};

export const bulkCreateHospitalShifts = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const shifts = Array.isArray(req.body?.shifts) ? req.body.shifts : [];
    if (!shifts.length || shifts.length > 500) return res.status(400).json({ success: false, message: 'Provide between 1 and 500 shifts' });
    const hospital = await Hospital.findById(hospitalId).select('departments').lean();
    if (!hospital) return res.status(404).json({ success: false, message: 'Hospital not found' });
    const configuredDepartments = new Map();
    for (const [index, input] of shifts.entries()) {
      const configuredDepartment = resolveConfiguredDepartment(hospital.departments, input.departmentId || input.department);
      if (!configuredDepartment) {
        return res.status(400).json({
          success: false,
          code: 'DEPARTMENT_NOT_CONFIGURED',
          message: `Shift row ${index + 1}: select an active department configured for this hospital`
        });
      }
      configuredDepartments.set(index, configuredDepartment);
      const coverageError = await validateHospitalShiftCoverage({
        hospitalId,
        departmentName: configuredDepartment.departmentName,
        specialty: input.specialty,
        wardId: input.wardId,
      });
      if (coverageError) return res.status(400).json({ success: false, message: `Shift row ${index + 1}: ${coverageError}` });
      if (!Array.isArray(input.staffIds) || !input.staffIds.length) {
        return res.status(400).json({ success: false, message: `Shift row ${index + 1}: assign at least one staff member before provisioning` });
      }
      if (input.status && !['DRAFT', 'PUBLISHED'].includes(input.status)) {
        return res.status(400).json({ success: false, message: `Shift row ${index + 1}: only draft or published status is allowed` });
      }
    }
    const requestedShiftIds = shifts.map((input, index) => input.shiftID || `ADMIN-${String(hospitalId).slice(-6)}-${new Date(input.date || new Date()).toISOString().slice(0, 10)}-${index + 1}`);
    if (new Set(requestedShiftIds).size !== requestedShiftIds.length) {
      return res.status(400).json({ success: false, message: 'Each provisioned shift must have a unique shiftID' });
    }
    const protectedShifts = await Shift.find({
      hospitalId,
      shiftID: { $in: requestedShiftIds },
      status: { $nin: ['DRAFT', 'PUBLISHED'] },
    }).select('shiftID status').lean();
    if (protectedShifts.length) {
      return res.status(409).json({
        success: false,
        message: 'Provisioning cannot reopen a shift that has started, been cancelled, or completed',
        shifts: protectedShifts.map(({ shiftID, status }) => ({ shiftID, status })),
      });
    }
    const [staff, configuredServices] = await Promise.all([
      ClinicalStaff.find({ hospitalId, status: { $ne: 'TERMINATED' } }).lean(),
      getConfiguredHospitalServiceSpecialties(hospitalId)
    ]);
    const staffById = new Map(staff.map((member) => [String(member._id), member]));
    const assignmentRolesByShift = new Map();
    for (const [index, input] of shifts.entries()) {
      const assignmentsForShift = new Map();
      const unrecognizedStaffId = input.staffIds.map(String).find((staffId) => !staffById.has(staffId));
      if (unrecognizedStaffId) {
        return res.status(400).json({ success: false, message: `Shift row ${index + 1}: a selected staff member is inactive, missing, or belongs to another hospital` });
      }
      for (const staffId of input.staffIds.map(String)) {
        const member = staffById.get(staffId);
        const team = input.teamAssignment || {};
        const teamRole = String(team.triageNurseId || '') === staffId ? 'TRIAGE_NURSE'
          : String(team.clinicalNurseId || '') === staffId ? 'CLINICAL_NURSE'
            : String(team.physicianId || '') === staffId ? 'PHYSICIAN' : '';
        const allowedDuties = getAllowedShiftDuties(member.positionCode, member.systemRole);
        const requestedDuty = input.assignmentRoleByStaff?.[staffId] || teamRole
          || (allowedDuties.length === 1 ? allowedDuties[0] : '');
        try {
          assignmentsForShift.set(staffId, normalizeShiftAssignmentRole(requestedDuty, member.systemRole, member.positionCode));
        } catch (error) {
          return res.status(400).json({
            success: false,
            code: error.code || 'INVALID_SHIFT_ASSIGNMENT',
            message: `Shift row ${index + 1}: ${nameOf(member)}: ${error.message}`,
            positionCodeNeedsBackfill: !member.positionCode
          });
        }
        const requestedService = input.serviceCoverageByStaff?.[staffId];
        const serviceCoverage = requestedService
          ? canonicalizeServiceCoverage(requestedService, configuredServices)
          : inferAssignmentServiceCoverage({ staff: member, configuredSpecialties: configuredServices });
        if (!serviceCoverage) {
          return res.status(400).json({
            success: false,
            message: `Shift row ${index + 1}: choose a hospital-mapped service for ${nameOf(member)}`
          });
        }
      }
      assignmentRolesByShift.set(index, assignmentsForShift);
    }
    const operations = shifts.map((input, index) => {
      const date = new Date(input.date || new Date());
      const assignments = (Array.isArray(input.staffIds) ? input.staffIds : []).map(String);
      const assignmentRoles = assignmentRolesByShift.get(index);
      const assignedStaff = assignments.map((id) => staffById.get(id)).filter(Boolean).map((member) => ({
        staffID: member._id,
        type: roleOf(member),
        assignmentRole: assignmentRoles.get(String(member._id)),
        positionCodeNeedsBackfill: !member.positionCode,
        name: member.firstName,
        surname: member.lastName,
        specialty: member.clinicalSpecialty,
        serviceCoverage: input.serviceCoverageByStaff?.[String(member._id)]
          ? canonicalizeServiceCoverage(input.serviceCoverageByStaff[String(member._id)], configuredServices)
          : inferAssignmentServiceCoverage({ staff: member, configuredSpecialties: configuredServices }),
        status: 'CONFIRMED'
      }));
      const team = input.teamAssignment || {};
      const shiftID = input.shiftID || `ADMIN-${String(hospitalId).slice(-6)}-${date.toISOString().slice(0, 10)}-${index + 1}`;
      return {
        updateOne: {
          filter: { hospitalId, shiftID, status: { $in: ['DRAFT', 'PUBLISHED'] } },
          update: {
            $set: {
              hospitalId,
              shiftID,
              shiftName: input.shiftName || `${input.department || 'Hospital'} clinical shift`,
              department: configuredDepartments.get(index).departmentName,
              departmentId: configuredDepartments.get(index).departmentId,
              departmentName: configuredDepartments.get(index).departmentName,
              specialty: input.specialty,
              wardId: input.wardId || null,
              shiftStart: input.shiftStart || '08:00',
              shiftEnd: input.shiftEnd || '16:00',
              duration: durationHours(input.shiftStart || '08:00', input.shiftEnd || '16:00'),
              date,
              maxStaffRequired: Number(input.maxStaffRequired) || Math.max(1, assignedStaff.length),
              assignedStaff,
              status: input.status || 'PUBLISHED',
              teamAssignment: {
                triageNurseId: team.triageNurseId || null,
                clinicalNurseId: team.clinicalNurseId || null,
                clinicalNurseContext: team.clinicalNurseContext || 'CLINICAL_NURSE_ER',
                physicianId: team.physicianId || null,
                physicianContext: team.physicianContext || 'PHYSICIAN_ER',
                teamAssignedAt: new Date(),
                assignedBy: req.user._id
              },
              shiftContext: input.shiftContext || 'ER',
              notes: input.notes || 'Administrator hospital operations setup'
            },
            $setOnInsert: { createdBy: req.user._id }
          },
          upsert: true
        }
      };
    });
    const result = await Shift.bulkWrite(operations, { ordered: false });
    await AuditLog.create({ action: 'HOSPITAL_SHIFTS_PROVISIONED', actorId: req.user._id, hospitalId, resourceType: 'SHIFT', authorized: true, changeDetails: { requested: shifts.length, upserted: result.upsertedCount, modified: result.modifiedCount }, description: 'Administrator provisioned hospital shifts and staff assignments' });
    return res.json({ success: true, message: 'Hospital shifts provisioned; staff clock-in remains blocked until each shift is explicitly started', requested: shifts.length, upserted: result.upsertedCount, modified: result.modifiedCount });
  } catch (error) {
    console.error('[HOSPITAL SHIFTS] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to provision hospital shifts', detail: error.message });
  }
};

export const bulkCreateHospitalEncounters = async (req, res) => {
  try {
    const hospitalId = requireHospital(req, res);
    if (!hospitalId) return;
    const count = Math.min(Math.max(Number.parseInt(req.body?.count, 10) || 1, 1), 1000);
    let historicalDays;
    try {
      historicalDays = normalizeHistoricalDays(req.body?.historicalDays);
    } catch (error) {
      return res.status(400).json({ success: false, message: error.message });
    }
    if (req.body?.lifecycleStatus && String(req.body.lifecycleStatus).toUpperCase() !== 'DISCHARGED') {
      return res.status(400).json({ success: false, message: 'Synthetic history can only create discharged encounters; start new triage through the nurse workflow' });
    }
    const patientNames = Array.isArray(req.body?.patientNames)
      ? req.body.patientNames.map((name) => String(name || '').trim()).filter(Boolean).slice(0, 1000)
      : [];
    const complaint = String(req.body?.complaint || '').trim();
    const complaintPresets = {
      CHEST_PAIN: 'Chest pain and shortness of breath',
      ABDOMINAL_PAIN: 'Acute abdominal pain',
      FEVER: 'Fever and generalized weakness',
      TRAUMA: 'Trauma following an accident',
      HEADACHE: 'Severe headache',
      RESPIRATORY: 'Respiratory distress',
      CUSTOM: complaint
    };
    const complaintType = String(req.body?.complaintType || 'FEVER').toUpperCase();
    const selectedComplaint = complaintPresets[complaintType] || complaint || complaintPresets.FEVER;
    const batch = `admin-${Date.now()}`;
    const now = new Date();
    const historyStart = new Date(now.getTime() - (historicalDays * 24 * 60 * 60 * 1000));
    const documents = Array.from({ length: count }, (_, index) => {
      const patientName = patientNames[index] || `Synthetic Patient ${index + 1}`;
      const encounterDate = historicalEncounterDate({ index, count, historicalDays, now });
      return createSyntheticHistoricalEncounter({
        hospitalId,
        actorId: req.user._id,
        batch,
        encounterDate,
        patientId: `SYN-${batch}-${index + 1}`,
        patientName,
        complaint: selectedComplaint
      });
    });
    const result = await Encounter.insertMany(documents, { ordered: false });
    await AuditLog.create({ action: 'HOSPITAL_ENCOUNTERS_PROVISIONED', actorId: req.user._id, hospitalId, resourceType: 'ENCOUNTER', authorized: true, changeDetails: { count: result.length, historicalDays, historyStart, historyEnd: now, batch, status: 'DISCHARGED' }, description: 'Administrator provisioned synthetic historical discharged encounters for EHR context' });
    return res.status(201).json({
      success: true,
      message: `${result.length} discharged synthetic history encounters created across the ${historicalDays}-day lookback window`,
      count: result.length,
      batch,
      historicalDays,
      historyStart,
      historyEnd: now,
      lifecycleStatus: 'DISCHARGED',
      complaint: selectedComplaint,
      patientNames: patientNames.length
    });
  } catch (error) {
    console.error('[HOSPITAL ENCOUNTERS] Error:', error);
    return res.status(500).json({ success: false, message: 'Failed to provision hospital encounters', detail: error.message });
  }
};
