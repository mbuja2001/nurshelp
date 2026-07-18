export const normalizeMedicationRequest = (payload = {}) => {
  const medicationName = payload.medication_name || payload.name || payload.medicationName || '';
  const dosage = payload.dosage || '';
  const frequency = payload.frequency || '';
  const route = payload.route || 'Oral';
  const indication = payload.indication || '';
  const requestedBy = payload.requested_by || payload.requestedBy || 'Physician';

  return {
    medication_name: medicationName.trim(),
    dosage: dosage.trim(),
    frequency: frequency.trim(),
    route: route.trim(),
    indication: indication.trim(),
    requested_by: requestedBy.trim(),
    status: 'PENDING',
    requested_at: payload.requested_at || new Date()
  };
};

export const isMedicationRequestValid = (payload = {}) => {
  const normalized = normalizeMedicationRequest(payload);
  return Boolean(normalized.medication_name);
};
