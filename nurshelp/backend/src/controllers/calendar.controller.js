import mongoose from "mongoose";
import { Encounter } from "../models/encounter.model.js";
import { Appointment } from "../models/hmsMisc.model.js";
import {
  hasPhaseZeroClinicalRole,
  identityValue,
  isAssignedSurgeon,
  PHASE_ZERO_DISABLED_CATEGORIES
} from "../utils/phaseZeroCalendarScope.js";

const buildLocationLabel = (encounter) => {
  const ward = encounter.assigned_ward?.name || encounter.assigned_ward?.wardName || encounter.assigned_ward?.department;
  const bed = encounter.assigned_bed?.bedNumber || encounter.assigned_bed?.bed_number || encounter.assigned_bed?.bedId;
  if (ward && bed) return `${ward} / ${bed}`;
  if (ward) return ward;
  if (bed) return `Bed ${bed}`;
  return "Clinical area";
};

const parseDateRange = (query) => {
  const now = new Date();
  const defaultStart = new Date(now);
  defaultStart.setDate(now.getDate() - 7);
  const defaultEnd = new Date(now);
  defaultEnd.setDate(now.getDate() + 30);

  const startDate = query.startDate ? new Date(query.startDate) : defaultStart;
  const endDate = query.endDate ? new Date(query.endDate) : defaultEnd;
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime()) || startDate >= endDate) {
    return null;
  }
  return { startDate, endDate };
};

const getScheduledEnd = (record, start) => {
  if (record.scheduled_end) {
    const end = new Date(record.scheduled_end);
    if (!Number.isNaN(end.getTime()) && end > start) return end;
  }

  const durationMinutes = Number(record.durationMinutes);
  if (Number.isFinite(durationMinutes) && durationMinutes > 0) {
    return new Date(start.getTime() + durationMinutes * 60000);
  }
  return null;
};

const emptyCalendarResponse = () => ({
  success: true,
  events: [],
  disabledCategories: PHASE_ZERO_DISABLED_CATEGORIES
});

export const getCalendarEvents = async (req, res) => {
  try {
    const userId = identityValue(req.user?._id || req.user?.id);
    const hospitalId = identityValue(req.hospitalId);
    const userHospitalId = identityValue(req.user?.hospitalId);
    if (!userId || !mongoose.Types.ObjectId.isValid(userId)) {
      return res.status(401).json({ success: false, message: "Authenticated staff identity is required" });
    }
    if (!hospitalId || !mongoose.Types.ObjectId.isValid(hospitalId)) {
      return res.status(401).json({ success: false, message: "Verified hospital context is required" });
    }
    if (userHospitalId && String(userHospitalId) !== String(hospitalId)) {
      return res.status(403).json({ success: false, message: "Hospital context mismatch" });
    }

    const range = parseDateRange(req.query || {});
    if (!range) {
      return res.status(400).json({ success: false, message: "A valid startDate before endDate is required" });
    }

    const requestedCategory = String(req.query?.category || '').trim().toLowerCase();
    if (PHASE_ZERO_DISABLED_CATEGORIES.includes(requestedCategory)) {
      return res.status(200).json(emptyCalendarResponse());
    }
    if (requestedCategory && !['surgery', 'appointment'].includes(requestedCategory)) {
      return res.status(400).json({ success: false, message: "Unsupported calendar category" });
    }

    const role = req.user?.role || req.userRole;
    if (!hasPhaseZeroClinicalRole(role)) {
      return res.status(200).json(emptyCalendarResponse());
    }

    const hospitalObjectId = new mongoose.Types.ObjectId(hospitalId);
    const userObjectId = new mongoose.Types.ObjectId(userId);
    const encounters = await Encounter.find({
      hospitalId: hospitalObjectId,
      surgery_records: {
        $elemMatch: {
          surgeon_id: userObjectId,
          scheduled_at: { $gte: range.startDate, $lte: range.endDate },
          status: { $in: ['SCHEDULED', 'PREOP_READY', 'IN_PROGRESS', 'COMPLETED'] }
        }
      }
    })
      .select("patient surgery_records assigned_ward assigned_bed")
      .lean();

    const events = [];
    encounters.forEach((encounter) => {
      if (String(identityValue(encounter.hospitalId)) !== String(hospitalObjectId)) return;
      const location = buildLocationLabel(encounter);
      (encounter.surgery_records || []).forEach((surgery) => {
        if (requestedCategory && requestedCategory !== 'surgery') return;
        if (!isAssignedSurgeon(surgery, userObjectId)) return;
        if (!['SCHEDULED', 'PREOP_READY', 'IN_PROGRESS', 'COMPLETED'].includes(surgery.status)) return;

        const start = surgery.scheduled_at ? new Date(surgery.scheduled_at) : null;
        if (!start || Number.isNaN(start.getTime()) || start < range.startDate || start > range.endDate) return;
        const end = getScheduledEnd(surgery, start);
        if (!end) return;

        const surgeryId = surgery._id || surgery.surgery_id;
        if (!surgeryId) return;
        events.push({
          id: `surgery-${encounter._id}-${surgeryId}`,
          title: `Surgery: ${surgery.procedure_name || 'Procedure'}`,
          type: 'surgery',
          start: start.toISOString(),
          end: end.toISOString(),
          status: surgery.status,
          encounterId: encounter._id,
          surgeryId,
          location
        });
      });
    });

    const appointments = await Appointment.find({
      hospitalId: hospitalObjectId,
      physicianId: userObjectId,
      scheduledDateTime: { $gte: range.startDate, $lte: range.endDate },
      status: { $nin: ['cancelled'] }
    })
      .select('hospitalId physicianId appointmentType scheduledDateTime duration status facilityId roomNumber')
      .lean();

    appointments.forEach((appointment) => {
      if (requestedCategory && requestedCategory !== 'appointment') return;
      if (String(appointment.hospitalId) !== String(hospitalObjectId)) return;
      if (String(appointment.physicianId) !== String(userObjectId)) return;

      const start = appointment.scheduledDateTime ? new Date(appointment.scheduledDateTime) : null;
      const durationMinutes = Number(appointment.duration);
      if (!start || Number.isNaN(start.getTime()) || start < range.startDate || start > range.endDate) return;
      if (!Number.isFinite(durationMinutes) || durationMinutes <= 0) return;

      const appointmentId = appointment._id;
      if (!appointmentId) return;
      events.push({
        id: `appointment-${appointmentId}`,
        title: appointment.appointmentType
          ? `Appointment: ${appointment.appointmentType}`
          : 'Appointment',
        type: 'appointment',
        start: start.toISOString(),
        end: new Date(start.getTime() + durationMinutes * 60000).toISOString(),
        status: appointment.status || 'scheduled',
        encounterId: null,
        appointmentId,
        location: appointment.roomNumber || appointment.facilityId || 'Appointment'
      });
    });

    events.sort((left, right) => left.start.localeCompare(right.start));
    return res.status(200).json({
      ...emptyCalendarResponse(),
      events,
      range: {
        start: range.startDate.toISOString(),
        end: range.endDate.toISOString()
      },
      asOf: new Date().toISOString()
    });
  } catch (err) {
    console.error('[Calendar Events] Error:', err);
    return res.status(500).json({ success: false, message: 'Failed to load calendar events' });
  }
};
