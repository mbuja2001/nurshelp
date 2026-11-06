/**
 * Reset the currently configured application database for a new, UI-driven QA
 * journey. This script intentionally does not seed or create application data.
 *
 * Dry run (default):
 *   node setup-hospital-auto-clockin.js
 *
 * Delete all documents after reviewing the dry-run output:
 *   node setup-hospital-auto-clockin.js --confirm-database <exact-database-name>
 *
 * The exact database name must match the database selected by MONGO_URI.
 */

import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import mongoose from 'mongoose';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(scriptDirectory, '.env') });

const confirmationArgumentIndex = process.argv.indexOf('--confirm-database');
const confirmedDatabaseName = confirmationArgumentIndex >= 0
  ? process.argv[confirmationArgumentIndex + 1]
  : null;

function validateConfiguration() {
  if (!process.env.MONGO_URI) {
    throw new Error('MONGO_URI is required; refusing to guess which database to reset.');
  }

  if (confirmationArgumentIndex >= 0 && !confirmedDatabaseName) {
    throw new Error('Pass the exact database name after --confirm-database.');
  }

  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log('Usage:');
    console.log('  node setup-hospital-auto-clockin.js');
    console.log('  node setup-hospital-auto-clockin.js --confirm-database <exact-database-name>');
    return false;
  }

  return true;
}

async function inspectCollections(database) {
  const collections = await database.listCollections({}, { nameOnly: false }).toArray();
  const applicationCollections = collections.filter(({ name, type }) => (
    !name.startsWith('system.') && type !== 'view'
  ));

  const collectionStates = [];
  for (const metadata of applicationCollections) {
    const collection = database.collection(metadata.name);
    const count = await collection.countDocuments({});
    const options = metadata.options || {};
    const unsupportedReason = options.capped
      ? 'capped collection (MongoDB does not permit deleting all documents)'
      : options.timeseries
        ? 'time-series collection (must be reset with its collection-specific policy)'
        : null;

    collectionStates.push({
      name: metadata.name,
      count,
      unsupportedReason
    });
  }

  return { collectionStates, viewNames: collections.filter(({ type }) => type === 'view').map(({ name }) => name) };
}

async function resetDatabase() {
  if (!validateConfiguration()) return;

  try {
    await mongoose.connect(process.env.MONGO_URI);
    const database = mongoose.connection.db;
    if (!database || !mongoose.connection.name) {
      throw new Error('MongoDB connected without a selected database; refusing to proceed.');
    }

    const databaseName = mongoose.connection.name;
    console.log(`Connected database: ${databaseName}`);
    console.log('Inspecting application collections; no database URI or credentials will be printed.');

    const { collectionStates, viewNames } = await inspectCollections(database);
    for (const state of collectionStates) {
      const note = state.unsupportedReason ? ` — NOT RESET: ${state.unsupportedReason}` : '';
      console.log(`  ${state.name}: ${state.count} document(s)${note}`);
    }
    for (const name of viewNames) {
      console.log(`  ${name}: view (preserved; derived from source collections)`);
    }

    const unsupported = collectionStates.filter(({ unsupportedReason }) => unsupportedReason);
    if (unsupported.length > 0) {
      throw new Error(
        `No collections were modified. Unsupported collection types require a deliberate reset policy: ${
          unsupported.map(({ name, unsupportedReason }) => `${name} (${unsupportedReason})`).join(', ')
        }`
      );
    }

    if (confirmedDatabaseName !== databaseName) {
      console.log('\nDRY RUN ONLY: no documents were deleted.');
      console.log(`To clear every listed application collection, rerun with: --confirm-database ${databaseName}`);
      return;
    }

    console.warn(`\nDESTRUCTIVE RESET CONFIRMED for database "${databaseName}".`);
    let totalDeleted = 0;
    for (const { name } of collectionStates) {
      const result = await database.collection(name).deleteMany({});
      totalDeleted += result.deletedCount;
      console.log(`  Cleared ${name}: ${result.deletedCount} document(s)`);
    }

    console.log(`\nDatabase reset complete: ${totalDeleted} document(s) deleted.`);
    console.log('Collections, indexes, validators, and views were preserved. No QA users, hospital, staff, shifts, or seed data were created.');
  } finally {
    await mongoose.disconnect();
  }
}

resetDatabase().catch((error) => {
  console.error(`Database reset failed: ${error.message}`);
  process.exitCode = 1;
});
