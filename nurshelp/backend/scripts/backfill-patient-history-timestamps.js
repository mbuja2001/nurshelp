#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';
import { Prescription } from '../src/models/pharmacy.model.js';
import { buildDischargeReportData } from '../src/utils/dischargeReportBuilder.js';

dotenv.config();

const batch = process.argv[2];
if (!batch) throw new Error('Usage: node scripts/backfill-patient-history-timestamps.js <seed-batch>');
const uri = process.env.MONGO_URI || process.env.MONGODB_URI;
if (!uri) throw new Error('Missing MONGO_URI or MONGODB_URI');
const addMinutes = (value, minutes) => new Date(new Date(value).getTime() + minutes * 60 * 1000);

await mongoose.connect(uri);
try {
  const encounters = await Encounter.find({ seedBatch: batch, syntheticData: true });
  for (const encounter of encounters) {
    const base = encounter.date || encounter.createdAt;
    if (!base) continue;

    const ai = encounter.ai_triage_report || encounter.triage || {};
    ai.generated_at = addMinutes(base, 1);
    encounter.ai_triage_report = ai;
    encounter.triage = { ...(encounter.triage || {}), aiTriageReport: ai };
    encounter.markModified('ai_triage_report');
    encounter.markModified('triage');

    const manual = encounter.manual_triage_report || encounter.triage?.manualTriageReport || {};
    manual.confirmed_at = addMinutes(base, 2);
    encounter.manual_triage_report = manual;
    encounter.triage.manualTriageReport = manual;
    encounter.markModified('manual_triage_report');
    encounter.markModified('triage');

    encounter.assigned_physician = {
      ...(encounter.assigned_physician || {}),
      name: encounter.assigned_physician?.name || 'Dr. Synthetic Physician',
      specialty: encounter.assigned_physician?.specialty || 'Internal Medicine',
      diagnosed_at: addMinutes(base, 3)
    };

    (encounter.lab_requests || []).forEach((request, index) => {
      const offset = index === 0 ? 4 : 5;
      request.requested_at = addMinutes(base, offset);
      request.assigned_at = addMinutes(base, offset);
      request.created_at = addMinutes(base, offset);
      request.submitted_at = addMinutes(base, 5);
      request.reviewed_at = addMinutes(base, 5);
      request.updated_at = addMinutes(base, 5);
    });
    (encounter.medication_administrations || []).forEach((medication) => {
      medication.administered_at = addMinutes(base, 6);
      medication.recorded_at = addMinutes(base, 6);
    });
    (encounter.surgery_records || []).forEach((surgery) => {
      surgery.scheduled_at = addMinutes(base, 6);
      surgery.requested_at = addMinutes(base, 6);
      surgery.performed_at = addMinutes(base, 7);
      surgery.recorded_at = addMinutes(base, 7);
      surgery.updated_at = addMinutes(base, 7);
    });
    if (encounter.discharge_report) {
      const existingReportId = encounter.discharge_report.reportId;
      encounter.discharge_report = {
        ...buildDischargeReportData({
          encounter,
          overrides: {
            authorId: encounter.discharge_report.authorId || 'synthetic-discharge-clinician',
            signedOffBy: encounter.discharge_report.signedOffBy || 'Dr. Synthetic Physician',
            followUpInstructions: encounter.discharge_report.followUpInstructions,
            notes: encounter.discharge_report.notes
          }
        }),
        reportId: existingReportId || undefined,
        dischargeDate: addMinutes(base, 9),
        createdAt: addMinutes(base, 9),
        updatedAt: addMinutes(base, 9)
      };
      encounter.markModified('discharge_report');
    }
    await encounter.save();

    await Prescription.updateMany(
      { encounterId: encounter._id },
      { $set: { date_issued: addMinutes(base, 8), dateIssued: addMinutes(base, 8) } }
    );
  }
  console.log(JSON.stringify({ batch, updatedEncounters: encounters.length }, null, 2));
} finally {
  await mongoose.disconnect();
}
