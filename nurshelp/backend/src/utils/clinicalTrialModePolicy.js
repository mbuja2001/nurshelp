export const evaluateClinicalTrialModeChange = ({
  currentMode,
  requestedMode,
  hasActiveTrials,
}) => {
  if (typeof requestedMode !== 'boolean') {
    return { allowed: false, code: 'INVALID_TRIAL_MODE', changed: false };
  }

  if (Boolean(currentMode) === requestedMode) {
    return { allowed: true, changed: false };
  }

  if (currentMode && !requestedMode && hasActiveTrials) {
    return { allowed: false, code: 'ACTIVE_TRIALS_EXIST', changed: false };
  }

  return { allowed: true, changed: true };
};

export const evaluateClinicalTrialModeGate = ({ hospitalId, hospital }) => {
  if (!hospitalId || !hospital) {
    return { allowed: false, code: 'HOSPITAL_NOT_FOUND' };
  }
  if (hospital.clinicalTrialMode !== true) {
    return { allowed: false, code: 'TRIAL_MODE_DISABLED' };
  }
  return { allowed: true };
};

export const buildClinicalTrialModeUpdate = ({
  currentMode,
  requestedMode,
  reason,
  actorId,
  actorName,
  at,
}) => ({
  $set: {
    clinicalTrialMode: requestedMode,
    clinicalTrialModeUpdatedAt: at,
    clinicalTrialModeUpdatedBy: actorId,
    clinicalTrialModeUpdatedByName: actorName,
    clinicalTrialModeReason: reason,
  },
  $push: {
    clinicalTrialModeHistory: {
      eventType: 'MODE_CHANGED',
      from: Boolean(currentMode),
      to: requestedMode,
      reason,
      by: actorId,
      byName: actorName,
      at,
    },
  },
});

export const buildClinicalTrialModeFilter = ({
  hospitalId,
  ownerAccountId,
  currentMode,
  requestedMode,
}) => ({
  _id: hospitalId,
  ownerAccountId,
  $and: [
    currentMode
      ? { clinicalTrialMode: true }
      : { $or: [{ clinicalTrialMode: false }, { clinicalTrialMode: { $exists: false } }] },
    ...(!requestedMode
      ? [{ $or: [{ activeTrialCount: 0 }, { activeTrialCount: { $exists: false } }] }]
      : []),
  ],
});

export const activateClinicalTrialWithCounter = async ({
  reserveHospital,
  activateTrial,
  decrementHospital,
}) => {
  const reservation = await reserveHospital();
  if (!reservation) return { success: false, code: 'TRIAL_MODE_DISABLED' };

  const releaseReservation = async () => {
    const released = await decrementHospital();
    if (!released) throw new Error('Failed to release the hospital active-trial reservation');
  };
  let trial;
  try {
    trial = await activateTrial();
  } catch (error) {
    try {
      await releaseReservation();
    } catch (compensationError) {
      throw new AggregateError(
        [error, compensationError],
        'Trial activation failed and the hospital reservation could not be released'
      );
    }
    throw error;
  }

  if (trial) return { success: true, changed: true, trial };
  await releaseReservation();
  return { success: false, code: 'TRIAL_STATE_CHANGED' };
};

export const closeClinicalTrialWithCounter = async ({ closeTrial, decrementHospital }) => {
  const trial = await closeTrial();
  if (!trial) return { success: true, changed: false };

  const decremented = await decrementHospital();
  if (!decremented) {
    throw new Error('Trial closed but failed to update the hospital active-trial count');
  }
  return { success: true, changed: true, trial };
};

export const reconcileClinicalTrialCount = async ({
  activeTrialCount,
  activeTrialActualCount,
  recordMismatch,
}) => {
  if (activeTrialCount === activeTrialActualCount) {
    return { matches: true, activeTrialCount, activeTrialActualCount };
  }

  await recordMismatch({ activeTrialCount, activeTrialActualCount });
  return { matches: false, activeTrialCount, activeTrialActualCount };
};
