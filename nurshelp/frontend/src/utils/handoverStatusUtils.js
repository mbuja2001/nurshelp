export const ACKNOWLEDGED_HANDOVER_STATUSES = ['ACKNOWLEDGED', 'IN_PROGRESS', 'COMPLETED'];

export const hasAnyHandoverStatus = (encounter, statuses = []) => {
  const handovers = Array.isArray(encounter?.clinical_handovers) ? encounter.clinical_handovers : [];
  if (!statuses.length || !handovers.length) return false;

  return handovers.some((handover) => {
    const status = handover?.handover_status;
    return Boolean(status) && statuses.includes(status);
  });
};

export const isAwaitingPhysicianAcknowledgment = (encounter) => {
  if (!encounter) return false;
  const handovers = Array.isArray(encounter.clinical_handovers) ? encounter.clinical_handovers : [];
  if (!handovers.length) return false;

  const hasActiveAcknowledgement = hasAnyHandoverStatus(encounter, ACKNOWLEDGED_HANDOVER_STATUSES);
  if (hasActiveAcknowledgement) return false;

  return handovers.some((handover) => {
    const status = handover?.handover_status;
    return status === 'NOTIFIED' || status === 'VIEWED';
  });
};

export const isAcknowledgedOrActiveHandover = (encounter) => {
  return hasAnyHandoverStatus(encounter, ACKNOWLEDGED_HANDOVER_STATUSES);
};
