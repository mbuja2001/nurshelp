#!/usr/bin/env node
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';
import PatientEHRRecord from '../src/models/patientEHRRecord.model.js';
import { Prescription } from '../src/models/pharmacy.model.js';

dotenv.config();

const batch = process.argv[2];
if (!batch) throw new Error('Usage: node scripts/verify-patient-history-reports.js <seed-batch>');

const uri = process.env.MONGO_URI || process.env.MONGODB_URI;
if (!uri) throw new Error('Missing MONGO_URI or MONGODB_URI');

await mongoose.connect(uri);
try {
  const encounters = await Encounter.find({ seedBatch: batch }).lean();
  const projections = await PatientEHRRecord.countDocuments({ seedBatch: batch });
  const encounterIds = encounters.map((encounter) => encounter._id);
  const prescriptions = await Prescription.countDocuments({ encounterId: { $in: encounterIds } });
  const completeBundles = encounters.filter((encounter) => (
    encounter.status === 'DISCHARGED'
    && encounter.isWaiting === false
    && Boolean(encounter.ai_triage_report?.esi_level)
    && Boolean(encounter.manual_triage_report?.selected_priority)
    && encounter.lab_requests?.length === 2
    && encounter.lab_requests.every((request) => ['COMPLETED', 'REVIEWED'].includes(request.status))
    && encounter.surgery_records?.length === 1
    && Boolean(encounter.discharge_report?.encounterId)
  ));

  console.log(JSON.stringify({
    batch,
    encounters: encounters.length,
    projections,
    prescriptions,
    activeQueue: encounters.filter((encounter) => encounter.isWaiting).length,
    completeReportBundles: completeBundles.length,
    labRequests: encounters.reduce((total, encounter) => total + (encounter.lab_requests?.length || 0), 0),
    surgeryReports: encounters.reduce((total, encounter) => total + (encounter.surgery_records?.length || 0), 0),
    dischargeReports: encounters.filter((encounter) => encounter.discharge_report).length
  }, null, 2));
} finally {
  await mongoose.disconnect();
}