export const extractPatientDisplayName = (encounter) => {
  if (!encounter || typeof encounter !== 'object') return 'Unknown';

  return (
    encounter.patientName
    || encounter.patient?.name
    || encounter.patient?.firstName
    || encounter.firstName
    || encounter.name
    || encounter.fullName
    || encounter.displayName
    || encounter.patient?.fullName
    || encounter.patient?.displayName
    || 'Unknown'
  );
};

const isDischargedEncounter = (encounter) => {
  return encounter && typeof encounter === 'object' && String(encounter.status || '').toUpperCase() === 'DISCHARGED';
};

export const buildOccupantSummary = (bed, encounter) => {
  if (!encounter) {
    return bed.currentEncounterId
      ? {
          _id: bed.currentEncounterId._id || bed.currentEncounterId,
          patientName: 'Unknown',
          age: null,
          diagnosis: '',
          severity: 4,
          triage: null,
          vitals: null,
          status: null,
          assignedPhysicianName: null,
          assignedNurseName: null
        }
      : null;
  }

  if (isDischargedEncounter(encounter)) {
    return null;
  }

  const normalizedEncounter = typeof encounter === 'object' ? encounter : { _id: encounter };

  return {
    _id: normalizedEncounter._id || bed.currentEncounterId?._id || bed.currentEncounterId,
    patientName: extractPatientDisplayName(normalizedEncounter),
    firstName: normalizedEncounter.patient?.firstName || normalizedEncounter.firstName || null,
    lastName: normalizedEncounter.patient?.surname || normalizedEncounter.lastName || normalizedEncounter.patient?.lastName || null,
    gender: normalizedEncounter.patient?.gender || normalizedEncounter.gender || null,
    age: normalizedEncounter.age ?? normalizedEncounter.patient?.age ?? null,
    diagnosis: normalizedEncounter.diagnosis || normalizedEncounter.triage?.aiTriageReport?.primary_diagnosis || normalizedEncounter.triage?.primary_diagnosis || '',
    severity: normalizedEncounter.severity ?? normalizedEncounter.triage?.aiTriageReport?.esi_level ?? normalizedEncounter.triage?.esi_level ?? 4,
    triage: normalizedEncounter.triage || null,
    vitals: normalizedEncounter.vitals || normalizedEncounter.triage?.vitals || null,
    status: normalizedEncounter.status || null,
    assignedPhysicianName: normalizedEncounter.assignedPhysicianName || null,
    assignedNurseName: normalizedEncounter.assignedNurseName || null
  };
};
