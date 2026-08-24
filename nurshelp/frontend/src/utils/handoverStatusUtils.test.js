import { describe, it, expect } from 'vitest';
import { hasAnyHandoverStatus, isAwaitingPhysicianAcknowledgment, isAcknowledgedOrActiveHandover } from './handoverStatusUtils';

describe('handoverStatusUtils', () => {
  it('flags encounters with mixed handover states as acknowledged when any relevant status is active', () => {
    const encounter = {
      clinical_handovers: [
        { clinician_id: 'p1', handover_status: 'NOTIFIED' },
        { clinician_id: 'p2', handover_status: 'ACKNOWLEDGED' }
      ]
    };

    expect(hasAnyHandoverStatus(encounter, ['ACKNOWLEDGED', 'IN_PROGRESS', 'COMPLETED'])).toBe(true);
    expect(isAwaitingPhysicianAcknowledgment(encounter)).toBe(false);
    expect(isAcknowledgedOrActiveHandover(encounter)).toBe(true);
  });

  it('keeps purely unacknowledged handovers in the awaiting queue', () => {
    const encounter = {
      clinical_handovers: [
        { clinician_id: 'p1', handover_status: 'NOTIFIED' },
        { clinician_id: 'p2', handover_status: 'VIEWED' }
      ]
    };

    expect(hasAnyHandoverStatus(encounter, ['ACKNOWLEDGED', 'IN_PROGRESS', 'COMPLETED'])).toBe(false);
    expect(isAwaitingPhysicianAcknowledgment(encounter)).toBe(true);
    expect(isAcknowledgedOrActiveHandover(encounter)).toBe(false);
  });
});
