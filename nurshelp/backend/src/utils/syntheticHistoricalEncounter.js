const DAY_MS = 24 * 60 * 60 * 1000;

export const normalizeHistoricalDays = (value = 90) => {
  const days = Number(value);
  if (!Number.isInteger(days) || days < 1 || days > 3650) {
    throw new RangeError('Historical lookback must be a whole number from 1 to 3650 days');
  }
  return days;
};

export const historicalEncounterDate = ({ index, count, historicalDays, now = new Date() }) => {
  if (!Number.isInteger(index) || index < 0 || !Number.isInteger(count) || count < 1) {
    throw new RangeError('Encounter index and count must be valid non-negative integers');
  }
  const days = normalizeHistoricalDays(historicalDays);
  const ageInDays = Math.max(1, Math.ceil(((index + 1) * days) / count));
  return new Date(new Date(now).getTime() - ageInDays * DAY_MS);
};

export const createSyntheticHistoricalEncounter = ({
  hospitalId,
  actorId,
  batch,
  encounterDate,
  patientId,
  patientName,
  complaint
}) => ({
  hospitalId,
  date: encounterDate,
  createdAt: encounterDate,
  syntheticData: true,
  seedBatch: batch,
  sourceMetadata: {
    source: 'administrator-hospital-operations',
    recordType: 'historical-discharged-encounter',
    seededBy: String(actorId)
  },
  patient: {
    id_number: patientId,
    name: patientName,
    surname: 'Hospital Record',
    symptoms: complaint,
    age: 25,
    gender: 'U'
  },
  status: 'DISCHARGED',
  isWaiting: false,
  status_history: [{
    status: 'DISCHARGED',
    changed_by: actorId,
    changed_at: encounterDate,
    notes: 'Administrator provisioned synthetic historical discharged encounter'
  }]
});
