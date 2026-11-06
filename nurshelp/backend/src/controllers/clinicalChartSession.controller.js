import mongoose from 'mongoose';
import { ClinicalChartSession } from '../models/clinicalChartSession.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import { Encounter } from '../models/encounter.model.js';
import { Shift } from '../models/shift.model.js';
import { AuditLog } from '../models/auditLog.model.js';

const auditSessionChange = async (req, session, action, description, changeDetails = {}) => {
  await AuditLog.create({
    action,
    actorId: req.user._id,
    hospitalId: session.hospitalId,
    resourceId: session.encounterId,
    resourceType: 'CLINICAL_CHART_SESSION',
    authorized: true,
    ipAddress: req.ip,
    userAgent: req.get('user-agent'),
    changeDetails: {
      sessionId: String(session._id),
      staffId: String(session.staffId),
      encounterId: String(session.encounterId),
      shiftId: session.shiftId ? String(session.shiftId) : null,
      status: session.status,
      hasUnsavedChanges: session.hasUnsavedChanges,
      ...changeDetails
    },
    description
  });
};

export const openClinicalChartSession = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    const staffId = req.user?._id;
    const sessionKey = String(req.body?.sessionKey || '').trim();
    const encounterId = String(req.body?.encounterId || '').trim();
    if (!hospitalId || !staffId || !sessionKey || sessionKey.length > 128 || !mongoose.isValidObjectId(encounterId)) {
      return res.status(400).json({ success: false, message: 'A valid encounter and chart session key are required' });
    }

    const staff = await ClinicalStaff.findOne({ _id: staffId, hospitalId, status: 'ACTIVE' }).select('_id');
    if (!staff) return res.status(403).json({ success: false, message: 'Only active clinical staff can open chart sessions' });
    const encounter = await Encounter.findOne({ _id: encounterId, hospitalId }).select('_id');
    if (!encounter) return res.status(404).json({ success: false, message: 'Encounter not found for this hospital' });

    const activeShift = await Shift.findOne({
      hospitalId,
      status: 'IN_PROGRESS',
      'assignedStaff.staffID': staffId
    }).select('_id');
    const now = new Date();
    const upsertResult = await ClinicalChartSession.updateOne(
      { hospitalId, staffId, sessionKey },
      {
        $set: {
          encounterId,
          shiftId: activeShift?._id || null,
          status: 'OPEN',
          hasUnsavedChanges: false,
          lastHeartbeatAt: now,
          closedAt: null,
          transferNote: null
        },
        $setOnInsert: { openedAt: now }
      },
      { upsert: true, runValidators: true, setDefaultsOnInsert: true }
    );
    const session = await ClinicalChartSession.findOne({ hospitalId, staffId, sessionKey });
    if (!session) return res.status(500).json({ success: false, message: 'Chart session could not be retrieved after opening' });
    if (upsertResult.upsertedCount > 0) {
      await auditSessionChange(req, session, 'CLINICAL_CHART_SESSION_OPENED', 'Clinical staff opened an encounter chart session');
    }
    return res.status(200).json({ success: true, session });
  } catch (error) {
    console.error('[CLINICAL CHART SESSION] Open failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to open a tracked chart session' });
  }
};

export const updateClinicalChartSession = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    const staffId = req.user?._id;
    const { sessionId } = req.params;
    const status = String(req.body?.status || 'OPEN').toUpperCase();
    if (!mongoose.isValidObjectId(sessionId) || !['OPEN', 'TRANSFERRED', 'CLOSED'].includes(status)) {
      return res.status(400).json({ success: false, message: 'Invalid chart session update' });
    }
    const session = await ClinicalChartSession.findOne({ _id: sessionId, hospitalId, staffId });
    if (!session) return res.status(404).json({ success: false, message: 'Chart session not found for this staff member and hospital' });
    if (session.status !== 'OPEN') {
      return res.status(409).json({ success: false, message: 'Only an open chart session can be updated' });
    }

    const previousStatus = session.status;
    const previousDirty = session.hasUnsavedChanges;
    session.lastHeartbeatAt = new Date();
    if (status === 'OPEN') {
      if (typeof req.body?.hasUnsavedChanges !== 'boolean') {
        return res.status(400).json({ success: false, message: 'hasUnsavedChanges must be a boolean for a heartbeat' });
      }
      session.hasUnsavedChanges = req.body.hasUnsavedChanges;
    } else if (status === 'TRANSFERRED') {
      const transferNote = String(req.body?.transferNote || '').trim();
      if (transferNote.length < 3 || transferNote.length > 500) {
        return res.status(400).json({ success: false, message: 'A transfer note between 3 and 500 characters is required' });
      }
      session.status = 'TRANSFERRED';
      session.hasUnsavedChanges = false;
      session.transferNote = transferNote;
    } else {
      if (session.hasUnsavedChanges) {
        return res.status(409).json({ success: false, message: 'Confirm the chart is saved or transfer the session before closing it' });
      }
      if (req.body?.attestedSaved !== true) {
        return res.status(400).json({ success: false, message: 'Confirm that chart changes were saved or handed over before closing the session' });
      }
      session.status = 'CLOSED';
      session.hasUnsavedChanges = false;
      session.closedAt = new Date();
    }
    await session.save();

    if (session.status !== previousStatus || session.hasUnsavedChanges !== previousDirty) {
      const action = session.status === 'CLOSED'
        ? 'CLINICAL_CHART_SESSION_CLOSED'
        : session.status === 'TRANSFERRED'
          ? 'CLINICAL_CHART_SESSION_TRANSFERRED'
          : 'CLINICAL_CHART_SESSION_UPDATED';
      await auditSessionChange(req, session, action, `Clinical chart session ${session.status.toLowerCase()}`, {
        previousStatus,
        previousHasUnsavedChanges: previousDirty,
        transferNote: session.transferNote
      });
    }
    return res.json({ success: true, session });
  } catch (error) {
    console.error('[CLINICAL CHART SESSION] Update failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to update the tracked chart session' });
  }
};

export const listActiveShiftChartSessions = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    const { shiftId } = req.params;
    if (!mongoose.isValidObjectId(shiftId)) return res.status(400).json({ success: false, message: 'A valid shift ID is required' });
    const shift = await Shift.findOne({ _id: shiftId, hospitalId }).select('_id');
    if (!shift) return res.status(404).json({ success: false, message: 'Shift not found for this hospital' });

    const sessions = await ClinicalChartSession.find({ hospitalId, shiftId, status: 'OPEN' })
      .populate('staffId', 'firstName lastName email systemRole')
      .select('staffId encounterId hasUnsavedChanges lastHeartbeatAt openedAt')
      .sort({ openedAt: 1 })
      .lean();
    return res.json({ success: true, sessions });
  } catch (error) {
    console.error('[CLINICAL CHART SESSION] Active shift query failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to query active chart sessions' });
  }
};

export const listActiveStaffChartSessions = async (req, res) => {
  try {
    const hospitalId = req.user?.hospitalId || req.hospitalId;
    const { staffId } = req.params;
    if (!mongoose.isValidObjectId(staffId)) return res.status(400).json({ success: false, message: 'A valid staff ID is required' });
    const staff = await ClinicalStaff.findOne({ _id: staffId, hospitalId }).select('_id');
    if (!staff) return res.status(404).json({ success: false, message: 'Staff member not found for this hospital' });
    const sessions = await ClinicalChartSession.find({ hospitalId, staffId, status: 'OPEN' })
      .select('encounterId shiftId hasUnsavedChanges lastHeartbeatAt openedAt')
      .sort({ openedAt: 1 })
      .lean();
    return res.json({ success: true, sessions });
  } catch (error) {
    console.error('[CLINICAL CHART SESSION] Active staff query failed:', error);
    return res.status(500).json({ success: false, message: 'Unable to query active staff chart sessions' });
  }
};
