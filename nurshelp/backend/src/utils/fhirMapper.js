const normalizeGender = (gender) => {
  const value = String(gender || '').toLowerCase();
  if (value === 'm' || value === 'male') return 'male';
  if (value === 'f' || value === 'female') return 'female';
  if (value === 'other') return 'other';
  return 'unknown';
};

export const mapEncounterToFHIRPatient = (encounter) => {
  const patient = encounter.patient || {};
  const identifier = patient.id_number || String(encounter._id);
  const telecomValue = patient.cellNumber || patient.cell_number || patient.phone;

  return {
    resourceType: 'Patient',
    id: String(encounter._id),
    active: encounter.status !== 'DISCHARGED',
    identifier: [
      {
        use: 'official',
        system: 'https://www.gov.za',
        value: identifier
      }
    ],
    name: [
      {
        use: 'official',
        family: patient.surname || '',
        given: patient.name ? [patient.name] : []
      }
    ],
    telecom: telecomValue
      ? [{ system: 'phone', value: telecomValue, use: 'mobile' }]
      : [],
    gender: normalizeGender(patient.gender)
  };
};

export const fhirOperationOutcome = (severity, code, diagnostics) => ({
  resourceType: 'OperationOutcome',
  issue: [{ severity, code, diagnostics }]
});