export const buildNotificationOwnershipFilter = ({ notificationId, clinicianId, hospitalId }) => ({
  _id: notificationId,
  clinician_id: clinicianId,
  hospital_id: hospitalId
});
