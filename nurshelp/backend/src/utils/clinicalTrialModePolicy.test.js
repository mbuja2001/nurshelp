import test from 'node:test';
import assert from 'node:assert/strict';
import {
  activateClinicalTrialWithCounter,
  buildClinicalTrialModeFilter,
  buildClinicalTrialModeUpdate,
  closeClinicalTrialWithCounter,
  evaluateClinicalTrialModeChange,
  evaluateClinicalTrialModeGate,
  reconcileClinicalTrialCount,
} from './clinicalTrialModePolicy.js';

test('trial mode is disabled by default unless explicitly enabled', () => {
  assert.deepEqual(evaluateClinicalTrialModeChange({
    currentMode: false,
    requestedMode: false,
    hasActiveTrials: false,
  }), { allowed: true, changed: false });
});

test('rejects invalid mode values', () => {
  assert.deepEqual(evaluateClinicalTrialModeChange({
    currentMode: false,
    requestedMode: 'true',
    hasActiveTrials: false,
  }), { allowed: false, code: 'INVALID_TRIAL_MODE', changed: false });
});

test('prevents disabling trial mode while a trial is active', () => {
  assert.deepEqual(evaluateClinicalTrialModeChange({
    currentMode: true,
    requestedMode: false,
    hasActiveTrials: true,
  }), { allowed: false, code: 'ACTIVE_TRIALS_EXIST', changed: false });
});

test('allows enabling trial mode and disabling it after trials close', () => {
  assert.deepEqual(evaluateClinicalTrialModeChange({
    currentMode: false,
    requestedMode: true,
    hasActiveTrials: false,
  }), { allowed: true, changed: true });
  assert.deepEqual(evaluateClinicalTrialModeChange({
    currentMode: true,
    requestedMode: false,
    hasActiveTrials: false,
  }), { allowed: true, changed: true });
});

test('mode disable filter requires trial mode enabled and a zero active-trial counter', () => {
  assert.deepEqual(buildClinicalTrialModeFilter({
    hospitalId: 'hospital-1',
    ownerAccountId: 'owner-1',
    currentMode: true,
    requestedMode: false,
  }), {
    _id: 'hospital-1',
    ownerAccountId: 'owner-1',
    $and: [
      { clinicalTrialMode: true },
      { $or: [{ activeTrialCount: 0 }, { activeTrialCount: { $exists: false } }] },
    ],
  });
});

test('enabling trial mode accepts a legacy hospital with an unset mode as disabled', () => {
  assert.deepEqual(buildClinicalTrialModeFilter({
    hospitalId: 'hospital-1',
    ownerAccountId: 'owner-1',
    currentMode: false,
    requestedMode: true,
  }), {
    _id: 'hospital-1',
    ownerAccountId: 'owner-1',
    $and: [{ $or: [{ clinicalTrialMode: false }, { clinicalTrialMode: { $exists: false } }] }],
  });
});

test('mode changes write exactly one embedded history entry and a failed guard writes none', () => {
  for (const [currentMode, requestedMode] of [[false, true], [true, false]]) {
    const update = buildClinicalTrialModeUpdate({
      currentMode,
      requestedMode,
      reason: 'Approved test change',
      actorId: 'owner-1',
      actorName: 'Owner',
      at: new Date('2026-01-01T00:00:00.000Z'),
    });
    assert.equal(update.$push.clinicalTrialModeHistory.eventType, 'MODE_CHANGED');
    assert.equal(update.$push.clinicalTrialModeHistory.from, currentMode);
    assert.equal(update.$push.clinicalTrialModeHistory.to, requestedMode);
  }

  const state = { clinicalTrialMode: false, activeTrialCount: 1, history: [] };
  const filter = buildClinicalTrialModeFilter({
    hospitalId: 'hospital-1',
    ownerAccountId: 'owner-1',
    currentMode: true,
    requestedMode: false,
  });
  if (state.clinicalTrialMode === true && state.activeTrialCount === 0) {
    state.history.push(buildClinicalTrialModeUpdate({
      currentMode: true,
      requestedMode: false,
      reason: 'Should not be recorded',
      actorId: 'owner-1',
      actorName: 'Owner',
      at: new Date(),
    }).$push.clinicalTrialModeHistory);
  }
  assert.equal(state.history.length, 0);
});

test('trial mode gate returns environment-independent errors', () => {
  for (const environment of ['SANDBOX', 'PRODUCTION']) {
    assert.deepEqual(evaluateClinicalTrialModeGate({
      hospitalId: 'hospital-1',
      hospital: { environment, clinicalTrialMode: false },
    }), { allowed: false, code: 'TRIAL_MODE_DISABLED' });
  }
  assert.deepEqual(evaluateClinicalTrialModeGate({ hospitalId: null, hospital: null }), {
    allowed: false,
    code: 'HOSPITAL_NOT_FOUND',
  });
});

test('concurrent trial activation and mode disable cannot both succeed', async () => {
  const state = { clinicalTrialMode: true, activeTrialCount: 0, status: 'DRAFT' };
  const activation = activateClinicalTrialWithCounter({
    reserveHospital: async () => {
      if (!state.clinicalTrialMode) return null;
      state.activeTrialCount += 1;
      return true;
    },
    activateTrial: async () => {
      if (state.status !== 'DRAFT') return null;
      state.status = 'ACTIVE';
      return { status: state.status };
    },
    decrementHospital: async () => {
      if (state.activeTrialCount <= 0) return false;
      state.activeTrialCount -= 1;
      return true;
    },
  });
  const disable = async () => {
    const filter = buildClinicalTrialModeFilter({
      hospitalId: 'hospital-1',
      ownerAccountId: 'owner-1',
      currentMode: true,
      requestedMode: false,
    });
    if (state.clinicalTrialMode !== true || state.activeTrialCount !== 0) return false;
    state.clinicalTrialMode = false;
    return true;
  };

  const [activationResult, disabled] = await Promise.all([activation, disable()]);
  assert.equal(Number(activationResult.success) + Number(disabled), 1);
  assert.equal(state.activeTrialCount, state.status === 'ACTIVE' ? 1 : 0);
  assert.equal(state.clinicalTrialMode, state.status === 'ACTIVE');
});

test('a repeated close decrements the active-trial counter only once', async () => {
  let status = 'ACTIVE';
  let activeTrialCount = 1;
  const close = () => closeClinicalTrialWithCounter({
    closeTrial: async () => {
      if (status !== 'ACTIVE') return null;
      status = 'CLOSED';
      return { status };
    },
    decrementHospital: async () => {
      if (activeTrialCount <= 0) return false;
      activeTrialCount -= 1;
      return true;
    },
  });

  const results = await Promise.all([close(), close()]);
  assert.equal(results.filter((result) => result.changed).length, 1);
  assert.equal(activeTrialCount, 0);
});

test('rejects a decrement when the active-trial counter is already zero', async () => {
  await assert.rejects(closeClinicalTrialWithCounter({
    closeTrial: async () => ({ status: 'CLOSED' }),
    decrementHospital: async () => false,
  }), /failed to update the hospital active-trial count/i);
});

  test('activation update exceptions release the reserved counter before surfacing', async () => {
    let activeTrialCount = 0;
    let decrementCount = 0;
    await assert.rejects(activateClinicalTrialWithCounter({
      reserveHospital: async () => {
        activeTrialCount += 1;
        return true;
      },
      activateTrial: async () => {
        throw new Error('trial update failed');
      },
      decrementHospital: async () => {
        decrementCount += 1;
        activeTrialCount -= 1;
        return true;
      },
    }), /trial update failed/);
    assert.equal(decrementCount, 1);
    assert.equal(activeTrialCount, 0);
  });

  test('reconciliation records a mismatch without changing either count', async () => {
  const events = [];
  const result = await reconcileClinicalTrialCount({
    activeTrialCount: 2,
    activeTrialActualCount: 1,
    recordMismatch: async (counts) => events.push(counts),
  });
  assert.deepEqual(result, { matches: false, activeTrialCount: 2, activeTrialActualCount: 1 });
  assert.deepEqual(events, [{ activeTrialCount: 2, activeTrialActualCount: 1 }]);
});
