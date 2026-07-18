import assert from 'node:assert/strict';
import { findActiveHospitalizationForPatient, getEncounterPatientIdentifier } from './src/controllers/clinical.controller.js';

const makeEncounter = (patientIdentifier, id = 'enc-1') => ({
  _id: id,
  patient: { id_number: patientIdentifier }
});

const fakeEncounterModel = {
  findOne: async (query) => {
    if (query._id?.$ne === 'enc-1') {
      return { _id: 'enc-2' };
    }
    return null;
  }
};

const identifier = getEncounterPatientIdentifier(makeEncounter('MRN-100'));
assert.equal(identifier, 'MRN-100');

const active = await findActiveHospitalizationForPatient(makeEncounter('MRN-100'), fakeEncounterModel);
assert.deepEqual(active, { _id: 'enc-2' });

console.log('clinical admission guard test passed');
