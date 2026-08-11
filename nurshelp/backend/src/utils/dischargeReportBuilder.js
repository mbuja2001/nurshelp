import crypto from 'crypto';

export const buildDischargeReportData = ({ encounter = {}, overrides = {} }) => {
  const patient = encounter.patient || {};
  const dischargeDate = overrideDate(encounter.discharge_date || encounter.discharge_scheduled_at || encounter.updatedAt || new Date());
  const report = {
    reportId: `DR-${crypto.randomBytes(4).toString('hex').toUpperCase()}`,
    patientId: patient.id_number || encounter.patientId || encounter.patient?.id || '',
    encounterId: encounter._id?.toString?.() || '',
    dischargeDate,
    status: 'draft',
    primaryDiagnosis: encounter.diagnosis || encounter.primaryDiagnosis || '',
    procedures: (encounter.surgery_records || []).filter(Boolean).map((entry) => ({
      procedureName: entry.procedure_name || entry.name || '',
      performedAt: entry.performed_at || entry.scheduled_at || null,
      durationMinutes: entry.duration_minutes || null
    })),
    medications: (encounter.medication_administrations || []).filter(Boolean).map((entry) => ({
      name: entry.medication_name || '',
      dose: entry.dosage || '',
      route: entry.route || '',
      frequency: entry.frequency || '',
      prescribingClinician: entry.prescribed_by || entry.administered_by || ''
    })),
    followUpInstructions: overrides.followUpInstructions || encounter.follow_up_instructions || encounter.treatment_plan || '',
    notes: overrides.notes || encounter.physician_notes || '',
    authorId: overrides.authorId || encounter.assigned_physician?.name || encounter.assignedPhysicianName || '',
    signedOffBy: overrides.signedOffBy || '',
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  return report;
};

function overrideDate(value) {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}
