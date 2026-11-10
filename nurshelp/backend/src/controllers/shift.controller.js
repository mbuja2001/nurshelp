/**
 * Shift Management Controller
 * Handles all shift-related operations for managers
 */

import { Shift } from '../models/shift.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import {
  buildAssignedStaff,
  getClinicalTeamRoleField,
  normalizeShiftAssignmentRole,
  validateClinicalTeam,
} from '../utils/shiftTeamAssignment.js';

// Helper function to generate Shift ID
const generateShiftID = () => `SHIFT-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`;

// ============ CREATE SHIFT ============
export const createShift = async (req, res) => {
  try {
    const { shiftName, department, shiftStart, shiftEnd, date, dayOfWeek, recurrence, recurrenceEndDate, maxStaffRequired, notes, assignedStaff } = req.body;
    const managerID = req.user.id;
    const hospitalId = req.hospitalId;

    console.log('\n[SHIFT CREATE]');
    console.log('  Manager:', managerID);
    console.log('  Hospital:', hospitalId);
    console.log('  Shift Name:', shiftName);
    console.log('  Department:', department);
    console.log('  Time:', `${shiftStart} - ${shiftEnd}`);
    console.log('  Max Staff Required:', maxStaffRequired);
    console.log('  Assigned Staff Count:', assignedStaff?.length || 0);
    if (assignedStaff && assignedStaff.length > 0) {
      assignedStaff.forEach((staff, i) => {
        console.log(`    [${i+1}] ${staff.name} ${staff.surname} (${staff.type})`);
      });
    }

    // Validate required fields
    if (!shiftName || !shiftStart || !shiftEnd || !maxStaffRequired) {
      return res.status(400).json({
        success: false,
        message: 'Missing required fields: shiftName, shiftStart, shiftEnd, maxStaffRequired'
      });
    }

    // Calculate duration
    const [startHour, startMin] = shiftStart.split(':').map(Number);
    const [endHour, endMin] = shiftEnd.split(':').map(Number);
    const startTime = startHour * 60 + startMin;
    const endTime = endHour * 60 + endMin;
    const duration = (endTime - startTime) / 60;

    const rosterInputs = Array.isArray(assignedStaff) ? assignedStaff : [];
    if (rosterInputs.length > Number(maxStaffRequired)) {
      return res.status(400).json({ success: false, message: 'Assigned staff exceeds maxStaffRequired' });
    }

    const staffIds = rosterInputs.map((entry) => String(entry.staffID || ''));
    if (staffIds.some((id) => !id) || new Set(staffIds).size !== staffIds.length) {
      return res.status(400).json({ success: false, message: 'Every shift assignment must include a unique staffID' });
    }
    const staffDocs = await ClinicalStaff.find({ _id: { $in: staffIds }, hospitalId, status: 'ACTIVE' }).lean();
    const staffById = new Map(staffDocs.map((staff) => [String(staff._id), staff]));
    if (staffById.size !== staffIds.length) {
      return res.status(400).json({ success: false, message: 'One or more assigned staff are inactive, missing, or outside this hospital' });
    }

    const canonicalAssignments = [];
    const teamAssignment = {
      triageNurseId: null,
      clinicalNurseId: null,
      clinicalNurseContext: 'CLINICAL_NURSE_ER',
      physicianId: null,
      physicianContext: 'PHYSICIAN_ER',
      physicianSpecialty: 'GENERAL',
      teamAssignedAt: rosterInputs.length ? new Date() : null,
      assignedBy: managerID,
    };
    for (const input of rosterInputs) {
      const canonicalStaff = staffById.get(String(input.staffID));
      let assignmentRole;
      try {
        assignmentRole = normalizeShiftAssignmentRole(input.assignmentRole || input.role, canonicalStaff.systemRole, canonicalStaff.positionCode);
      } catch (error) {
        const missingDuty = !String(input.assignmentRole || input.role || '').trim();
        return res.status(400).json({
          success: false,
          code: missingDuty
            ? 'SHIFT_DUTY_REQUIRED'
            : canonicalStaff.positionCode
              ? 'DUTY_NOT_PERMITTED_FOR_POSITION'
              : 'DUTY_NOT_PERMITTED_FOR_SYSTEM_ROLE',
          ...(missingDuty && !canonicalStaff.positionCode ? { positionCodeNeedsBackfill: true } : {}),
          message: `${canonicalStaff.firstName} ${canonicalStaff.lastName}: ${error.message}`,
        });
      }
      const assignment = {
        staffID: canonicalStaff._id,
        type: canonicalStaff.systemRole,
        assignmentRole,
        name: `${canonicalStaff.firstName || ''} ${canonicalStaff.lastName || ''}`.trim(),
        surname: canonicalStaff.lastName || '',
        specialty: canonicalStaff.clinicalSpecialty || '',
        status: 'ASSIGNED',
      };
      canonicalAssignments.push(assignment);
      const teamRoleField = getClinicalTeamRoleField(assignmentRole);
      if (teamRoleField) teamAssignment[teamRoleField] = canonicalStaff._id;
    }

    const newShift = new Shift({
      hospitalId,
      shiftID: generateShiftID(),
      shiftName,
      department: typeof department === 'string' && department.trim() ? department.trim() : null,
      shiftStart,
      shiftEnd,
      duration,
      date: date ? new Date(date) : null,
      dayOfWeek,
      recurrence: recurrence || 'ONCE',
      recurrenceEndDate: recurrenceEndDate ? new Date(recurrenceEndDate) : null,
      maxStaffRequired,
      assignedStaff: canonicalAssignments,
      teamAssignment,
      shiftContext: teamAssignment.triageNurseId ? 'TRIAGE' : 'WARD',
      status: 'PUBLISHED',
      createdBy: managerID,
      notes
    });

    await newShift.save();

    console.log(`✓ Shift created successfully!`);
    console.log(`  Shift ID: ${newShift.shiftID}`);
    console.log(`  DB ID: ${newShift._id}`);
    console.log(`  Staff assigned: ${newShift.assignedStaff.length}/${maxStaffRequired}\n`);

    res.status(201).json({
      success: true,
      message: 'Shift created successfully',
      shift: newShift
    });
  } catch (err) {
    console.error('[SHIFT CREATE ERROR]', err);
    res.status(500).json({
      success: false,
      message: 'Failed to create shift',
      error: err.message
    });
  }
};

// ============ GET SHIFTS BY DEPARTMENT & DATE RANGE ============
export const getShifts = async (req, res) => {
  try {
    const { department, startDate, endDate, dayOfWeek, status } = req.query;

    const query = {
      hospitalId: req.hospitalId  // Filter by user's hospital
    };
    if (department) query.department = department;
    if (status) query.status = status;
    if (dayOfWeek) query.dayOfWeek = dayOfWeek;

    if (startDate && endDate) {
      query.date = {
        $gte: new Date(startDate),
        $lte: new Date(endDate)
      };
    }

    const shifts = await Shift.find(query).sort({ date: 1, shiftStart: 1 });

    res.status(200).json({
      success: true,
      message: 'Shifts retrieved successfully',
      count: shifts.length,
      shifts
    });
  } catch (err) {
    console.error('Error fetching shifts:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch shifts',
      error: err.message
    });
  }
};

// ============ GET SHIFTS FOR SPECIFIC DAY ============
export const getShiftsForDay = async (req, res) => {
  try {
    const { date, department } = req.query;

    if (!date) {
      return res.status(400).json({
        success: false,
        message: 'Date parameter is required'
      });
    }

    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    const query = {
      date: { $gte: startOfDay, $lte: endOfDay },
      hospitalId: req.hospitalId  // Filter by user's hospital
    };

    if (department) query.department = department;

    const shifts = await Shift.find(query).sort({ shiftStart: 1 });

    res.status(200).json({
      success: true,
      message: 'Shifts for day retrieved successfully',
      date,
      count: shifts.length,
      shifts
    });
  } catch (err) {
    console.error('Error fetching daily shifts:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch daily shifts',
      error: err.message
    });
  }
};

// ============ GET SHIFTS FOR WEEK ============
export const getShiftsForWeek = async (req, res) => {
  try {
    const { startDate, department } = req.query;

    if (!startDate) {
      return res.status(400).json({
        success: false,
        message: 'startDate parameter is required'
      });
    }

    const weekStart = new Date(startDate);
    weekStart.setHours(0, 0, 0, 0);
    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekEnd.getDate() + 7);

    const query = {
      date: { $gte: weekStart, $lte: weekEnd },
      hospitalId: req.hospitalId
    };

    if (department) query.department = department;

    const shifts = await Shift.find(query).sort({ date: 1, shiftStart: 1 });

    const shiftsByDay = {};
    shifts.forEach(shift => {
      const dayKey = shift.date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
      if (!shiftsByDay[dayKey]) shiftsByDay[dayKey] = [];
      shiftsByDay[dayKey].push(shift);
    });

    res.status(200).json({
      success: true,
      message: 'Shifts for week retrieved successfully',
      startDate: weekStart.toISOString().split('T')[0],
      endDate: weekEnd.toISOString().split('T')[0],
      count: shifts.length,
      shifts,
      shiftsByDay
    });
  } catch (err) {
    console.error('Error fetching weekly shifts:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch weekly shifts',
      error: err.message
    });
  }
};

// ============ GET SHIFTS FOR MONTH ============
export const getShiftsForMonth = async (req, res) => {
  try {
    const { year, month, department } = req.query;

    if (!year || !month) {
      return res.status(400).json({
        success: false,
        message: 'Year and month parameters are required (e.g., ?year=2026&month=4)'
      });
    }

    const monthStart = new Date(year, month - 1, 1);
    const monthEnd = new Date(year, month, 0);
    monthEnd.setHours(23, 59, 59, 999);

    const query = {
      date: { $gte: monthStart, $lte: monthEnd },
      hospitalId: req.hospitalId
    };

    if (department) query.department = department;

    const shifts = await Shift.find(query).sort({ date: 1, shiftStart: 1 });

    const shiftsByWeek = {};
    shifts.forEach(shift => {
      const weekNum = Math.ceil(shift.date.getDate() / 7);
      const weekKey = `Week ${weekNum}`;
      if (!shiftsByWeek[weekKey]) shiftsByWeek[weekKey] = [];
      shiftsByWeek[weekKey].push(shift);
    });

    res.status(200).json({
      success: true,
      message: 'Shifts for month retrieved successfully',
      year: parseInt(year),
      month: parseInt(month),
      count: shifts.length,
      shifts,
      shiftsByWeek
    });
  } catch (err) {
    console.error('Error fetching monthly shifts:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch monthly shifts',
      error: err.message
    });
  }
};

// ============ GET ASSIGNED SHIFTS FOR AUTHENTICATED USER ============
export const getAssignedShiftsForUser = async (req, res) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({ success: false, message: 'User not authenticated' });
    }

    // By default return upcoming shifts (today onwards)
    const start = new Date();
    start.setHours(0,0,0,0);

    const query = {
      hospitalId: req.hospitalId,
      'assignedStaff.staffID': userId,
      date: { $gte: start }
    };

    console.log('[GET ASSIGNED SHIFTS] User:', userId);
    console.log('  Query:', JSON.stringify(query));

    const shifts = await Shift.find(query).sort({ date: 1, shiftStart: 1 });

    console.log(`  Found: ${shifts.length} shifts`);
    shifts.forEach((shift, i) => {
      console.log(`    [${i+1}] ${shift.shiftName} (${shift.department}) - ${shift.shiftStart} to ${shift.shiftEnd}`);
    });

    res.status(200).json({
      success: true,
      message: 'Assigned shifts retrieved',
      count: shifts.length,
      shifts
    });
  } catch (err) {
    console.error('[GET ASSIGNED SHIFTS ERROR]:', err);
    res.status(500).json({ success: false, message: 'Failed to fetch assigned shifts', error: err.message });
  }
};

// ============ GET TODAY'S ASSIGNED SHIFT FOR CLOCK-IN VALIDATION ============
// CRITICAL: Returns shifts with ACTIVE TEAM ROLE assignments, prioritizing recently assigned roles
// When multiple shifts exist: returns shift where user has explicit non-null role, sorted by assignment timestamp
export const getTodayAssignedShift = async (req, res) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({ success: false, message: 'User not authenticated' });
    }

    // Get today's date (at start of day)
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const userIdString = userId.toString();

    console.log('\n[GET TODAY ASSIGNED SHIFT]');
    console.log('  User ID:', userIdString);
    console.log('  Date range:', today.toISOString(), 'to', tomorrow.toISOString());

    // PRIORITY 1: Find shifts where user is explicitly assigned to a TEAM ROLE
    // Key: Only match shifts where the role field is NOT null (actively assigned)
    const teamRoleQuery = {
      hospitalId: req.hospitalId,
      date: { $gte: today, $lt: tomorrow },
      $or: [
        { 'teamAssignment.triageNurseId': userId },
        { 'teamAssignment.triageNurseId': userIdString },
        { 'teamAssignment.clinicalNurseId': userId },
        { 'teamAssignment.clinicalNurseId': userIdString },
        { 'teamAssignment.physicianId': userId },
        { 'teamAssignment.physicianId': userIdString }
      ]
    };

    // Sort: most recently assigned first, then by creation time
    let shift = await Shift.findOne(teamRoleQuery)
      .sort({ 
        'teamAssignment.teamAssignedAt': -1,  // Most recent team assignment first
        '_id': -1                               // Tiebreaker: most recently created
      })
      .exec();

    console.log('  PRIORITY 1 (active team roles):', shift ? `Found: ${shift.shiftName} (ID: ${shift._id})` : 'NOT FOUND');
    if (shift) {
      const ta = shift.teamAssignment || {};
      console.log('    Team Assignment:');
      console.log('      triageNurseId:', ta.triageNurseId ? ta.triageNurseId.toString() : null);
      console.log('      clinicalNurseId:', ta.clinicalNurseId ? ta.clinicalNurseId.toString() : null);
      console.log('      physicianId:', ta.physicianId ? ta.physicianId.toString() : null);
      console.log('      teamAssignedAt:', ta.teamAssignedAt);
    }

    // PRIORITY 2: If no active team role found, fall back to assignedStaff
    if (!shift) {
      console.log('  PRIORITY 2: Checking assignedStaff fallback...');
      const assignedStaffQuery = {
        hospitalId: req.hospitalId,
        'assignedStaff.staffID': userId,
        date: { $gte: today, $lt: tomorrow }
      };
      shift = await Shift.findOne(assignedStaffQuery).sort({ createdAt: -1 }).exec();
      console.log('  PRIORITY 2 (assignedStaff):', shift ? `Found: ${shift.shiftName} (ID: ${shift._id})` : 'NOT FOUND');
    }

    res.status(200).json({
      success: true,
      message: 'Today shift retrieved',
      shift: shift || null
    });
  } catch (err) {
    console.error('[GET TODAY ASSIGNED SHIFT ERROR]', err);
    res.status(500).json({ success: false, message: 'Failed to fetch today shift', error: err.message });
  }
};

// ============ PUBLISH SHIFT (make available for assignment) ============
export const publishShift = async (req, res) => {
  try {
    const { shiftID } = req.params;

    const shift = await Shift.findOneAndUpdate(
      { shiftID, hospitalId: req.hospitalId },
      { status: 'PUBLISHED' },
      { new: true }
    );

    if (!shift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found'
      });
    }

    res.status(200).json({
      success: true,
      message: 'Shift published successfully',
      shift
    });
  } catch (err) {
    console.error('Error publishing shift:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to publish shift',
      error: err.message
    });
  }
};

// ============ ASSIGN STAFF TO SHIFT ============
export const assignStaffToShift = async (req, res) => {
  try {
    const { shiftID } = req.params;
    const { staffID, assignmentRole } = req.body;

    if (!staffID || !assignmentRole) {
      return res.status(400).json({
        success: false,
        message: 'staffID and assignmentRole are required'
      });
    }

    const shift = await Shift.findOne({ shiftID, hospitalId: req.hospitalId });
    if (!shift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found'
      });
    }

    // CRITICAL: Verify staff belongs to the same hospital (prevent cross-hospital assignment)
    const staff = await ClinicalStaff.findById(staffID);
    if (!staff) {
      return res.status(404).json({
        success: false,
        message: 'Staff member not found'
      });
    }

    if (staff.hospitalId.toString() !== req.hospitalId.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Staff member does not belong to this hospital. Cannot assign to this shift.'
      });
    }

    if (staff.status !== 'ACTIVE') {
      return res.status(400).json({ success: false, message: 'Inactive staff cannot be assigned to a shift' });
    }

    let canonicalRole;
    try {
      canonicalRole = normalizeShiftAssignmentRole(assignmentRole, staff.systemRole, staff.positionCode);
    } catch (error) {
      return res.status(400).json({ success: false, message: error.message });
    }

    // Check if staff is already assigned
    const alreadyAssigned = shift.assignedStaff.some((entry) => String(entry.staffID?._id || entry.staffID) === String(staffID));
    if (alreadyAssigned) {
      return res.status(400).json({
        success: false,
        message: 'Staff already assigned to this shift'
      });
    }

    // Check if shift is full
    if (shift.assignedStaff.length >= shift.maxStaffRequired) {
      return res.status(400).json({
        success: false,
        message: 'Shift is already full'
      });
    }

    const roleField = getClinicalTeamRoleField(canonicalRole);
    if (roleField) {
      if (shift.teamAssignment?.[roleField] && String(shift.teamAssignment[roleField]) !== String(staffID)) {
        return res.status(409).json({ success: false, message: `This shift already has an assigned ${canonicalRole.replaceAll('_', ' ').toLowerCase()}` });
      }
      shift.teamAssignment[roleField] = staff._id;
      shift.teamAssignment.teamAssignedAt = new Date();
      shift.teamAssignment.assignedBy = req.user.id;
      if (canonicalRole === 'TRIAGE_NURSE') shift.shiftContext = 'TRIAGE';
      if (canonicalRole === 'CLINICAL_NURSE') shift.teamAssignment.clinicalNurseContext = shift.teamAssignment.clinicalNurseContext || 'CLINICAL_NURSE_ER';
    }

    shift.assignedStaff.push({
      staffID: staff._id,
      type: staff.systemRole,
      assignmentRole: canonicalRole,
      name: `${staff.firstName || ''} ${staff.lastName || ''}`.trim(),
      surname: staff.lastName,
      specialty: staff.clinicalSpecialty,
      status: 'ASSIGNED'
    });

    await shift.save();

    res.status(200).json({
      success: true,
      message: 'Staff assigned to shift successfully',
      shift
    });
  } catch (err) {
    console.error('Error assigning staff:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to assign staff',
      error: err.message
    });
  }
};

// ============ REMOVE STAFF FROM SHIFT ============
export const removeStaffFromShift = async (req, res) => {
  try {
    const { shiftID } = req.params;
    const { staffID } = req.body;

    if (!staffID) {
      return res.status(400).json({
        success: false,
        message: 'staffID is required'
      });
    }

    const shift = await Shift.findOne({ shiftID, hospitalId: req.hospitalId });
    if (!shift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found'
      });
    }

    shift.assignedStaff = shift.assignedStaff.filter(staff => staff.staffID !== staffID);
    await shift.save();

    res.status(200).json({
      success: true,
      message: 'Staff removed from shift successfully',
      shift
    });
  } catch (err) {
    console.error('Error removing staff:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to remove staff',
      error: err.message
    });
  }
};

// ============ UPDATE SHIFT ============
export const updateShift = async (req, res) => {
  try {
    const { shiftID } = req.params;
    const updateData = req.body;

    // Recalculate duration if times changed
    if (updateData.shiftStart && updateData.shiftEnd) {
      const [startHour, startMin] = updateData.shiftStart.split(':').map(Number);
      const [endHour, endMin] = updateData.shiftEnd.split(':').map(Number);
      const startTime = startHour * 60 + startMin;
      const endTime = endHour * 60 + endMin;
      updateData.duration = (endTime - startTime) / 60;
    }

    const shift = await Shift.findOneAndUpdate(
      { shiftID, hospitalId: req.hospitalId },
      updateData,
      { new: true, runValidators: true }
    );

    if (!shift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found'
      });
    }

    res.status(200).json({
      success: true,
      message: 'Shift updated successfully',
      shift
    });
  } catch (err) {
    console.error('Error updating shift:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to update shift',
      error: err.message
    });
  }
};

// ============ DELETE SHIFT ============
export const deleteShift = async (req, res) => {
  try {
    const { shiftID } = req.params;

    const shift = await Shift.findOneAndDelete({ shiftID, hospitalId: req.hospitalId });

    if (!shift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found'
      });
    }

    res.status(200).json({
      success: true,
      message: 'Shift deleted successfully',
      shift
    });
  } catch (err) {
    console.error('Error deleting shift:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to delete shift',
      error: err.message
    });
  }
};

export const cancelShift = async (req, res) => {
  try {
    const { shiftID } = req.params;
    const cancellationReason = typeof req.body?.reason === 'string'
      ? req.body.reason.trim()
      : '';
    if (!cancellationReason) {
      return res.status(400).json({
        success: false,
        message: 'A cancellation reason is required'
      });
    }
    if (cancellationReason.length > 500) {
      return res.status(400).json({
        success: false,
        message: 'Cancellation reason must be 500 characters or fewer'
      });
    }

    const shift = await Shift.findOneAndUpdate(
      { shiftID, hospitalId: req.hospitalId, status: { $in: ['DRAFT', 'PUBLISHED'] } },
      {
        $set: {
          status: 'CANCELLED',
          cancelledBy: req.user.id || req.user._id,
          cancelledAt: new Date(),
          cancellationReason
        }
      },
      { new: true, runValidators: true }
    );
    if (!shift) {
      const existingShift = await Shift.findOne({ shiftID, hospitalId: req.hospitalId });
      if (!existingShift) {
        return res.status(404).json({ success: false, message: 'Shift not found' });
      }
      return res.status(409).json({
        success: false,
        message: `Cannot cancel a ${String(existingShift.status).toLowerCase()} shift`
      });
    }

    return res.status(200).json({
      success: true,
      message: 'Shift cancelled successfully',
      shift
    });
  } catch (err) {
    console.error('Error cancelling shift:', err);
    return res.status(500).json({
      success: false,
      message: 'Failed to cancel shift',
      error: err.message
    });
  }
};

// ============ GET AVAILABLE STAFF FOR ASSIGNMENT ============
export const getAvailableStaff = async (req, res) => {
  try {
    const { department, specialty, type } = req.query;

    console.log('\n[GET AVAILABLE STAFF] Request received');
    console.log('  req.hospitalId:', req.hospitalId);
    console.log('  Query params - department:', department, 'specialty:', specialty, 'type:', type);

    const query = { 
      isActive: true,
      hospitalId: req.hospitalId
    };
    if (department) query.department = department;
    if (specialty) query.specialty = specialty;
    if (type) query.role = type; // NURSE or PHYSICIAN

    console.log('[GET AVAILABLE STAFF] Query object:', JSON.stringify(query, null, 2));

    const staff = await ClinicalStaff.find(query).select('_id name surname email department specialty role hospitalId isActive').limit(100);

    console.log('[GET AVAILABLE STAFF] Query result:');
    console.log('  Count:', staff.length);
    if (staff.length > 0) {
      console.log('  First staff member:', {
        id: staff[0]._id,
        name: staff[0].name,
        email: staff[0].email,
        hospitalId: staff[0].hospitalId,
        isActive: staff[0].isActive
      });
    }

    // Debug: Check all staff in hospital
    const allStaffInHospital = await ClinicalStaff.find({ hospitalId: req.hospitalId }).select('_id name surname email hospitalId isActive');
    console.log('[GET AVAILABLE STAFF] All staff in this hospital (including inactive):', allStaffInHospital.length);
    if (allStaffInHospital.length > 0) {
      allStaffInHospital.forEach((s, i) => {
        console.log(`  [${i+1}] ${s.name} ${s.surname} - isActive: ${s.isActive}, hospitalId: ${s.hospitalId}`);
      });
    }

    // CRITICAL DEBUG: Check ALL nurses in system to see if any staff exists at all
    const allStaff = await ClinicalStaff.find({}).select('_id name surname email hospitalId isActive');
    console.log('[GET AVAILABLE STAFF] DEBUG - ALL staff in entire system:', allStaff.length);
    if (allStaff.length > 0) {
      allStaff.forEach((s, i) => {
        console.log(`  [${i+1}] ${s.name} ${s.surname} - hospitalId: ${s.hospitalId}, isActive: ${s.isActive}`);
      });
    }

    res.status(200).json({
      success: true,
      message: 'Available staff retrieved successfully',
      count: staff.length,
      staff
    });
  } catch (err) {
    console.error('Error fetching available staff:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch available staff',
      error: err.message
    });
  }
};

// ============ ASSIGN CLINICAL TEAM TO SHIFT ============
// Assigns Triage Nurse, Clinical Nurse, and Physician to a shift with context roles
// This is the critical endpoint for Full Day Simulation: determines which dashboard each staff sees
export const assignClinicalTeam = async (req, res) => {
  try {
    // Accept either `shiftId` (new) or `shiftID` (legacy) for compatibility
    const body = req.body || {};
    const shiftId = body.shiftId || body.shiftID || body.shift || null;
    // Accept multiple possible property names for staff ids
    const triageNurseId = body.triageNurseId || body.triageNurse || null;
    const clinicalNurseId = body.clinicalNurseId || body.clinicalNurse || null;
    const clinicalNurseContext = body.clinicalNurseContext || body.clinicalContext || 'CLINICAL_NURSE_ER';
    const physicianId = body.physicianId || body.physician || null;
    const physicianContext = body.physicianContext || body.physicianRole || 'PHYSICIAN_ER';
    const physicianSpecialty = body.physicianSpecialty || body.physician_specialty || 'GENERAL';

    console.log('\n[ASSIGN CLINICAL TEAM] 📋 Request received');
    console.log('  Shift ID:', shiftId);
    console.log('  🔵 Triage Nurse:', triageNurseId || 'NONE');
    console.log('  🔴 Clinical Nurse:', clinicalNurseId || 'NONE', clinicalNurseId ? `(${clinicalNurseContext})` : '');
    console.log('  🟩 Physician:', physicianId || 'NONE', physicianId ? `(${physicianContext})` : '');

    // Validate required fields
    if (!shiftId) {
      return res.status(400).json({
        success: false,
        message: 'shiftId is required'
      });
    }

    // Find the shift
    const shift = await Shift.findById(shiftId);
    if (!shift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found'
      });
    }

    const staffIds = [triageNurseId, clinicalNurseId, physicianId].filter(Boolean);
    const staffDocs = await ClinicalStaff.find({
      _id: { $in: staffIds },
      hospitalId: req.hospitalId,
      status: 'ACTIVE',
    }).lean().exec();
    const staffById = new Map(staffDocs.map((staff) => [staff._id.toString(), staff]));

    let normalizedTeam;
    try {
      normalizedTeam = validateClinicalTeam({
        shiftId,
        triageNurseId,
        clinicalNurseId,
        physicianId,
        staffById,
        hospitalId: req.hospitalId,
      });
    } catch (error) {
      return res.status(400).json({
        success: false,
        message: error.message,
      });
    }

    const assignedStaff = buildAssignedStaff(
      shift.assignedStaff,
      staffById,
      normalizedTeam,
    );

    // Determine shiftContext based on assigned roles
    let shiftContext = 'WARD';
    if (physicianContext === 'PHYSICIAN_ICU' || clinicalNurseContext === 'CLINICAL_NURSE_ICU') {
      shiftContext = 'ICU';
    } else if (physicianContext === 'PHYSICIAN_SURGICAL' || clinicalNurseContext === 'CLINICAL_NURSE_SURGICAL') {
      shiftContext = 'SURGICAL';
    } else if (triageNurseId) {
      shiftContext = 'TRIAGE';
    }

    // Update shift using findByIdAndUpdate for reliable persistence
    const updateData = {
      'teamAssignment.triageNurseId': normalizedTeam.triageNurseId || null,
      'teamAssignment.clinicalNurseId': normalizedTeam.clinicalNurseId || null,
      'teamAssignment.clinicalNurseContext': clinicalNurseContext || 'CLINICAL_NURSE_ER',
      'teamAssignment.physicianId': normalizedTeam.physicianId || null,
      'teamAssignment.physicianContext': physicianContext || 'PHYSICIAN_ER',
      'teamAssignment.physicianSpecialty': physicianSpecialty || 'GENERAL',
      'teamAssignment.teamAssignedAt': new Date(),
      'teamAssignment.assignedBy': req.user?.id,
      assignedStaff,
      shiftContext,
      status: shift.status === 'PUBLISHED' ? 'PUBLISHED' : 'ACTIVE'
    };

    const updatedShift = await Shift.findByIdAndUpdate(shiftId, updateData, { new: true, runValidators: true });

    if (!updatedShift) {
      return res.status(404).json({
        success: false,
        message: 'Shift not found or update failed'
      });
    }

    console.log('✓ Clinical team assigned successfully!');
    console.log(`  Triage Nurse: ${triageNurseId ? 'YES' : 'NO'}`);
    console.log(`  Clinical Nurse: ${clinicalNurseId ? 'YES' : 'NO'}`);
    console.log(`  Physician: ${physicianId ? 'YES' : 'NO'}`);
    console.log(`  Shift Context: ${updatedShift.shiftContext}`);

    res.status(200).json({
      success: true,
      message: 'Clinical team assigned to shift successfully',
      shift: updatedShift,
      team: updatedShift.teamAssignment
    });
  } catch (err) {
    console.error('[ASSIGN CLINICAL TEAM ERROR]', err);
    res.status(500).json({
      success: false,
      message: 'Failed to assign clinical team',
      error: err.message
    });
  }
};
