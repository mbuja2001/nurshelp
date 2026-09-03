#!/usr/bin/env node
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { Encounter } from '../src/models/encounter.model.js';
import { buildLongitudinalHistoryContext } from '../src/utils/longitudinalPatientHistory.js';

dotenv.config();

const getArg = (name, fallback) => {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
};
const hasFlag = (name) => process.argv.includes(`--${name}`);

const batch = getArg('batch', 'history-stress');
const patientId = getArg('patient-id', 'SYNTH-HISTORY-000001');
const outputPath = getArg('out', `./artifacts/${batch}-${patientId}-payload.json`);
const endpoint = getArg('endpoint', process.env.ESI_ENGINE_URL || 'http://localhost:5000/llm-triage');
const maxCharacters = Number.parseInt(getArg('max-characters', '12000'), 10) || 12000;
const snapshotVersion = `snapshot-${crypto.createHash('sha256').update(`${batch}:${patientId}:${maxCharacters}`).digest('hex').slice(0, 16)}`;

async function main() {
  const uri = process.env.MONGO_URI || process.env.MONGODB_URI || 'mongodb://localhost:27017/nurse_helpdesk';
  await mongoose.connect(uri);
  try {
    const encounters = await Encounter.find({ seedBatch: batch, 'patient.id_number': patientId }).sort({ date: -1, createdAt: -1 }).lean();
    if (encounters.length < 2) throw new Error(`Expected at least 2 encounters for ${patientId} in batch ${batch}; seed the data first.`);

    const currentEncounter = encounters[0];
    const context = buildLongitudinalHistoryContext({ encounters, currentEncounter, asOf: currentEncounter.date, maxCharacters });
    const approved = hasFlag('approve');
    const approval = {
      status: approved ? 'approved' : 'pending',
      approvedBy: approved ? getArg('nurse-id', 'synthetic-triage-nurse') : null,
      approvedAt: approved ? new Date().toISOString() : null,
      snapshotVersion,
      editReason: getArg('edit-reason', approved ? 'Script-based synthetic approval' : null)
    };
    const payload = {
      patientId,
      patient: currentEncounter.patient,
      vitals: currentEncounter.vitals,
      chiefComplaint: currentEncounter.patient.symptoms,
      symptoms: currentEncounter.patient.symptoms,
      history: context.serialized,
      longitudinalHistory: context.payload,
      contextApproval: approval,
      syntheticTest: true,
      seedBatch: batch,
      snapshotVersion
    };

    await fs.mkdir('./artifacts', { recursive: true });
    await fs.writeFile(outputPath, JSON.stringify(payload, null, 2));
    const report = {
      mode: hasFlag('send') ? 'send' : 'preview',
      approved: approval.status === 'approved',
      patientId,
      batch,
      sourceEncounters: encounters.length,
      recentEncounters: context.payload.metadata.recent_encounter_count,
      relevantEncounters: context.payload.metadata.relevant_encounter_count,
      criticalEncounters: context.payload.critical_history?.length || 0,
      asOf: context.payload.metadata.as_of,
      estimatedCharacters: context.payload.metadata.estimated_characters,
      estimatedTokens: context.payload.metadata.estimated_tokens,
      truncated: context.payload.metadata.truncated,
      payloadFile: outputPath
    };

    if (hasFlag('send')) {
      if (!approved) throw new Error('Refusing to send: pass --approve after reviewing the generated snapshot.');
      const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(`ESI Engine rejected payload (${response.status}): ${JSON.stringify(body)}`);
      report.engineStatus = response.status;
      report.engineResponseKeys = Object.keys(body);
    }

    console.log(JSON.stringify(report, null, 2));
  } finally {
    await mongoose.disconnect();
  }
}

main().catch((error) => {
  console.error(`[test-patient-history-context-payload] ${error.message}`);
  process.exitCode = 1;
});
