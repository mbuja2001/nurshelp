import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import mongoose from 'mongoose';

dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nursehelpdesk';

function parseCsvRow(line) {
  const fields = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === ',' && !inQuotes) {
      fields.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  fields.push(cur);
  return fields.map(f => f.trim());
}

const specialtyCategoryMap = {
  'Cardiology': 'Medical',
  'Cardiothoracic Surgery': 'Surgical',
  'Cardiothoracic': 'Surgical',
  'CardiothoracicSurgery': 'Surgical',
  'Cardiothoracic': 'Surgical',
  'Vascular Surgery': 'Surgical',
  'General Surgery': 'Surgical',
  'Thoracic Surgery': 'Surgical',
  'Trauma Surgery': 'Surgical',
  'Pediatric Surgery': 'Surgical',
  'Plastic Surgery': 'Surgical',
  'Orthopedics': 'Surgical',
  'Neurosurgery': 'Surgical',
  'Interventional Radiology': 'Radiology',
  'Radiology': 'Radiology',
  'Pathology': 'Pathology',
  'Emergency Medicine': 'Medical',
  'Critical Care': 'Medical',
  'Pulmonology': 'Medical',
  'Gastroenterology': 'Medical',
  'Infectious Disease': 'Medical',
  'Neurology': 'Medical',
  'Oncology': 'Medical',
  'Urology': 'Surgical',
  'Dermatology': 'Medical',
  'Pediatrics': 'Medical',
  'ENT': 'Surgical',
  'OBGYN': 'Surgical',
  'Psychiatry': 'Medical',
  'Rheumatology': 'Medical',
  'Nephrology': 'Medical',
  'Endocrinology': 'Medical',
  'Geriatrics': 'Medical',
  'Optometrist': 'Allied Health',
  'Toxicology': 'Medical',
};

function makeCode(name, existing) {
  const base = name.replace(/[^A-Za-z0-9]/g, '').toUpperCase().slice(0, 8);
  let code = base || 'SPEC';
  let i = 1;
  while (existing.has(code)) {
    code = `${base}_${i}`;
    i++;
  }
  existing.add(code);
  return code;
}

async function main() {
  console.log('🔌 Connecting to MongoDB...');
  await mongoose.connect(MONGO_URI);
  console.log('✅ Connected');

  const { Specialty } = await import('../models/specialty.model.js');

  // Resolve CSV path robustly across platforms
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const csvPath = path.resolve(scriptDir, '..', '..', 'ML', 'physicians.csv');
  if (!fs.existsSync(csvPath)) {
    console.error('CSV not found at', csvPath);
    process.exit(1);
  }

  const text = fs.readFileSync(csvPath, 'utf8').trim();
  const lines = text.split(/\r?\n/);
  if (lines.length <= 1) {
    console.error('CSV seems empty');
    process.exit(1);
  }

  const specialties = new Set();
  for (let i = 1; i < lines.length; i++) {
    const row = parseCsvRow(lines[i]);
    if (row.length < 3) continue;
    const specialty = row[2].trim();
    if (specialty) specialties.add(specialty);
  }

  console.log(`Found ${specialties.size} unique specialties in CSV`);

  const existingCodes = new Set();
  const results = [];
  for (const name of specialties) {
    const category = specialtyCategoryMap[name] || 'Medical';
    const code = makeCode(name, existingCodes);
    const doc = {
      name,
      code,
      category,
      description: '',
      standardProcedures: [],
      isActive: true,
      requiresCredentialing: category === 'Surgical' || category === 'Radiology' || category === 'Pathology'
    };

    const upserted = await Specialty.findOneAndUpdate(
      { name },
      { $setOnInsert: doc },
      { upsert: true, returnDocument: 'after', setDefaultsOnInsert: true }
    );

    results.push({ name: upserted.name, id: upserted._id, code: upserted.code });
    console.log('Upserted specialty:', upserted.name, upserted.code);
  }

  console.log('\n✅ Specialties seeded:', results.length);
  await mongoose.connection.close();
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
