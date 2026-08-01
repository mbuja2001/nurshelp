import mongoose from "mongoose";
import { Encounter } from "../models/encounter.model.js";
import { Appointment } from "../models/hmsMisc.model.js";

const normalizeDate = (value, fallback) => {
  if (!value) return fallback;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? fallback : parsed;
};

const buildLocationLabel = (encounter) => {
  const ward = encounter.assigned_ward?.name || encounter.assigned_ward?.wardName || encounter.assigned_ward?.department;
  const bed = encounter.assigned_bed?.bedNumber || encounter.assigned_bed?.bed_number || encounter.assigned_bed?.bedId;
  if (ward && bed) return `${ward} / ${bed}`;
  if (ward) return ward;
  if (bed) return `Bed ${bed}`;
  return "Clinical area";
};

const buildEvent = ({
  id,
  title,
  type,
  start,
  end,
  status,
  patientName,
  encounterId,
  surgeryId,
  location,
  assignedPhysicianName
}) => ({
  id,
  title,
  start,
  end,
  type,
  status,
  patientName,
  encounterId,
  surgeryId,
  location,
  assignedPhysicianName,
  color: type === "surgery"
    ? "#2563eb"
    : type === "lab"
      ? "#f59e0b"
      : type === "med"
        ? "#14b8a6"
        : type === "meeting"
          ? "#8b5cf6"
          : type === "diagnosed"
            ? "#f97316"
            : type === "discharged"
              ? "#475569"
              : type === "administered"
                ? "#0f766e"
                : "#6366f1"
});

export const getCalendarEvents = async (req, res) => {
  try {
    const hospitalId = req.hospitalId;
    if (!hospitalId) {
      return res.status(400).json({ success: false, message: "Hospital ID not found" });
    }

    const now = new Date();
    const defaultStart = new Date(now);
    defaultStart.setDate(now.getDate() - 7);
    const defaultEnd = new Date(now);
    defaultEnd.setDate(now.getDate() + 30);

    const startDate = normalizeDate(req.query.startDate, defaultStart);
    const endDate = normalizeDate(req.query.endDate, defaultEnd);

    const hospitalObjectId = mongoose.Types.ObjectId.isValid(hospitalId)
      ? new mongoose.Types.ObjectId(hospitalId)
      : hospitalId;

    const query = {
      hospitalId: hospitalObjectId,
      $or: [
        { "surgery_records.scheduled_at": { $gte: startDate, $lte: endDate } },
        { "lab_requests.requested_at": { $gte: startDate, $lte: endDate } },
        { "medication_requests.requested_at": { $gte: startDate, $lte: endDate } },
        { "medication_administrations.administered_at": { $gte: startDate, $lte: endDate } },
        { "clinical_handovers.handover_at": { $gte: startDate, $lte: endDate } },
        { "assigned_physician.diagnosed_at": { $gte: startDate, $lte: endDate } },
        { status_history: { $elemMatch: { status: "DIAGNOSED", changed_at: { $gte: startDate, $lte: endDate } } } },
        { status_history: { $elemMatch: { status: "DISCHARGED", changed_at: { $gte: startDate, $lte: endDate } } } }
      ]
    };

    const encounters = await Encounter.find(query)
      .select("patient assignedPhysicianName assigned_physician triage.assigned_physician assigned_bed assigned_ward surgery_records lab_requests medication_requests medication_administrations status status_history diagnosis")
      .lean();

    const events = [];

    encounters.forEach((encounter) => {
      const patientName = encounter.patient?.name || encounter.patient?.firstName || encounter.patient?.surname || "Patient";
      const location = buildLocationLabel(encounter);
      const assignedPhysicianName = encounter.assignedPhysicianName || encounter.assigned_physician?.name || encounter.triage?.assigned_physician?.name || "Physician";

      (encounter.surgery_records || []).forEach((surgery) => {
        const surgeryStart = surgery.scheduled_at ? new Date(surgery.scheduled_at) : null;
        if (!surgeryStart || surgeryStart < startDate || surgeryStart > endDate) return;

        const end = new Date(surgeryStart.getTime() + (surgery.durationMinutes || 90) * 60000);
        events.push(buildEvent({
          id: `surgery-${encounter._id}-${surgery._id || surgery.surgery_id}`,
          title: `Surgery: ${surgery.procedure_name || 'Procedure'} · ${patientName}`,
          type: "surgery",
          start: surgeryStart.toISOString(),
          end: end.toISOString(),
          status: surgery.status || "SCHEDULED",
          patientName,
          encounterId: encounter._id,
          surgeryId: surgery._id || surgery.surgery_id,
          location,
          assignedPhysicianName
        }));
      });

      (encounter.lab_requests || []).forEach((labRequest) => {
        const requestTime = labRequest.requested_at ? new Date(labRequest.requested_at) : null;
        if (!requestTime || requestTime < startDate || requestTime > endDate) return;

        const end = new Date(requestTime.getTime() + 30 * 60000);
        events.push(buildEvent({
          id: `lab-${encounter._id}-${labRequest.request_id}`,
          title: `Lab: ${labRequest.lab_type || 'Lab request'} · ${patientName}`,
          type: "lab",
          start: requestTime.toISOString(),
          end: end.toISOString(),
          status: labRequest.status || "PENDING",
          patientName,
          encounterId: encounter._id,
          surgeryId: labRequest.request_id,
          location,
          assignedPhysicianName
        }));
      });

      (encounter.medication_requests || []).forEach((medRequest, index) => {
        const requestTime = medRequest.requested_at ? new Date(medRequest.requested_at) : null;
        if (!requestTime || requestTime < startDate || requestTime > endDate) return;

        const end = new Date(requestTime.getTime() + 20 * 60000);
        events.push(buildEvent({
          id: `med-${encounter._id}-${medRequest.prescriptionId || medRequest._id || index}`,
          title: `Medication Order: ${medRequest.medication_name || 'Order'} · ${patientName}`,
          type: "med",
          start: requestTime.toISOString(),
          end: end.toISOString(),
          status: medRequest.status || "PENDING",
          patientName,
          encounterId: encounter._id,
          surgeryId: medRequest.prescriptionId || medRequest._id || index,
          location,
          assignedPhysicianName
        }));
      });

      (encounter.medication_administrations || []).forEach((admin, index) => {
        const administeredAt = admin.administered_at ? new Date(admin.administered_at) : null;
        if (!administeredAt || administeredAt < startDate || administeredAt > endDate) return;

        const end = new Date(administeredAt.getTime() + 15 * 60000);
        events.push(buildEvent({
          id: `admin-${encounter._id}-${admin._id || index}`,
          title: `Administered: ${admin.medication_name || 'Medication'} · ${patientName}`,
          type: "administered",
          start: administeredAt.toISOString(),
          end: end.toISOString(),
          status: admin.status || "ADMINISTERED",
          patientName,
          encounterId: encounter._id,
          surgeryId: admin._id || index,
          location,
          assignedPhysicianName
        }));
      });

      const diagnosedAt = encounter.assigned_physician?.diagnosed_at
        ? new Date(encounter.assigned_physician.diagnosed_at)
        : null;
      if (diagnosedAt && diagnosedAt >= startDate && diagnosedAt <= endDate) {
        const title = encounter.diagnosis ? `Diagnosed: ${encounter.diagnosis} · ${patientName}` : `Diagnosed · ${patientName}`;
        events.push(buildEvent({
          id: `diagnosed-${encounter._id}`,
          title,
          type: "diagnosed",
          start: diagnosedAt.toISOString(),
          end: new Date(diagnosedAt.getTime() + 30 * 60000).toISOString(),
          status: encounter.status || "DIAGNOSED",
          patientName,
          encounterId: encounter._id,
          surgeryId: null,
          location,
          assignedPhysicianName
        }));
      }

      (encounter.status_history || []).forEach((history, index) => {
        const changedAt = history.changed_at ? new Date(history.changed_at) : null;
        if (!changedAt || changedAt < startDate || changedAt > endDate) return;

        if (history.status === "DISCHARGED") {
          events.push(buildEvent({
            id: `discharged-${encounter._id}-${index}`,
            title: `Discharged · ${patientName}`,
            type: "discharged",
            start: changedAt.toISOString(),
            end: new Date(changedAt.getTime() + 20 * 60000).toISOString(),
            status: history.status,
            patientName,
            encounterId: encounter._id,
            surgeryId: null,
            location,
            assignedPhysicianName
          }));
        } else if (history.status === "DIAGNOSED" && !diagnosedAt) {
          events.push(buildEvent({
            id: `diagnosed-${encounter._id}-${index}`,
            title: `Diagnosed · ${patientName}`,
            type: "diagnosed",
            start: changedAt.toISOString(),
            end: new Date(changedAt.getTime() + 30 * 60000).toISOString(),
            status: history.status,
            patientName,
            encounterId: encounter._id,
            surgeryId: null,
            location,
            assignedPhysicianName
          }));
        }
      });

      (encounter.clinical_handovers || []).forEach((handover, index) => {
        const handoverAt = handover.handover_at ? new Date(handover.handover_at) : null;
        if (!handoverAt || handoverAt < startDate || handoverAt > endDate) return;

        const title = handover.report_type
          ? `${handover.report_type} Handover: ${handover.clinician_name || 'Clinical team'} · ${patientName}`
          : `Handover: ${handover.clinician_name || 'Clinical team'} · ${patientName}`;

        events.push(buildEvent({
          id: `meeting-${encounter._id}-${index}`,
          title,
          type: "meeting",
          start: handoverAt.toISOString(),
          end: new Date(handoverAt.getTime() + 30 * 60000).toISOString(),
          status: handover.handover_status || "NOTIFIED",
          patientName,
          encounterId: encounter._id,
          surgeryId: null,
          location,
          assignedPhysicianName
        }));
      });
    });

    const physicianId = req.user?._id;
    const appointmentQuery = {
      physicianId: physicianId,
      scheduledDateTime: { $gte: startDate, $lte: endDate },
      status: { $ne: 'cancelled' }
    };

    const appointments = await Appointment.find(appointmentQuery)
      .populate('patientId', 'name surname')
      .select('appointmentType appointmentNumber scheduledDateTime duration status facilityId roomNumber patientId')
      .lean();

    appointments.forEach((appointment, index) => {
      const appointmentStart = appointment.scheduledDateTime ? new Date(appointment.scheduledDateTime) : null;
      if (!appointmentStart) return;

      const appointmentEnd = appointment.duration
        ? new Date(appointmentStart.getTime() + appointment.duration * 60000)
        : new Date(appointmentStart.getTime() + 30 * 60000);

      const patientName = appointment.patientId?.name || appointment.patientId?.surname || 'Patient';
      const appointmentTypeLabel = appointment.appointmentType || 'Appointment';
      const title = `${appointmentTypeLabel}: ${patientName}`;

      events.push(buildEvent({
        id: `meeting-${appointment._id}-${index}`,
        title,
        type: 'meeting',
        start: appointmentStart.toISOString(),
        end: appointmentEnd.toISOString(),
        status: appointment.status || 'SCHEDULED',
        patientName,
        encounterId: null,
        surgeryId: appointment._id,
        location: appointment.roomNumber || appointment.facilityId || 'Appointment',
        assignedPhysicianName: req.user?.name || 'Physician'
      }));
    });

    return res.status(200).json({ success: true, events });
  } catch (err) {
    console.error('[Calendar Events] Error:', err);
    return res.status(500).json({ success: false, message: 'Failed to load calendar events', error: err.message });
  }
};
