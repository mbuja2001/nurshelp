export const REVIEW_OUTCOME = {
  CORRECT: 'CORRECT',
  MISTRIAGE_MINOR: 'MISTRIAGE_MINOR',
  MISTRIAGE_MAJOR: 'MISTRIAGE_MAJOR',
  MISTRIAGE_CRITICAL: 'MISTRIAGE_CRITICAL',
  PENDING_CLINICAL_REVIEW: 'PENDING_CLINICAL_REVIEW',
};

const SEVERITY_WEIGHT = { Critical: 4, High: 3, Medium: 2, Low: 1 };

export const getSeverityWeight = (severity) => SEVERITY_WEIGHT[severity] ?? 0;

export const computeReviewOutcome = ({
  aiSuggestedUrgency,
  nurseApprovedUrgency,
  physicianFinalUrgency,
  ambiguousClinicalReview = false,
  dangerousCondition = false,
  mistriageFlag = false,
} = {}) => {
  if (ambiguousClinicalReview) {
    return REVIEW_OUTCOME.PENDING_CLINICAL_REVIEW;
  }

  if (mistriageFlag) {
    const reductionWeight = getSeverityWeight(physicianFinalUrgency) ?? 0;
    const originalWeight = getSeverityWeight(aiSuggestedUrgency) ?? 0;

    if (dangerousCondition || (originalWeight >= 3 && reductionWeight < originalWeight)) {
      return REVIEW_OUTCOME.MISTRIAGE_CRITICAL;
    }

    if (originalWeight >= 3 || getSeverityWeight(nurseApprovedUrgency) >= 3 || reductionWeight >= 3) {
      return REVIEW_OUTCOME.MISTRIAGE_MAJOR;
    }

    return REVIEW_OUTCOME.MISTRIAGE_MINOR;
  }

  return REVIEW_OUTCOME.CORRECT;
};

export const calculateMistriageRate = (cases = []) => {
  if (!Array.isArray(cases) || cases.length === 0) {
    return 0;
  }

  const flagged = cases.filter((entry) => Boolean(entry.mistriageFlag)).length;
  return Number(((flagged / cases.length) * 100).toFixed(1));
};

export const calculateTrendDelta = (currentValue, previousValue, { inverse = false } = {}) => {
  const current = Number(currentValue) || 0;
  const previous = Number(previousValue) || 0;

  if (!Number.isFinite(current) || !Number.isFinite(previous)) {
    return 0;
  }

  if (previous === 0) {
    return current > 0 ? 100 : 0;
  }

  const delta = ((current - previous) / Math.abs(previous)) * 100;
  return Number((inverse ? -delta : delta).toFixed(1));
};

export const getMetricConfidenceState = (value, sampleSize, minimumN = 30) => {
  const numericValue = Number(value);
  const numericSample = Number(sampleSize);

  if (!Number.isFinite(numericValue) || !Number.isFinite(numericSample) || numericSample <= 0) {
    return 'INSUFFICIENT';
  }

  if (numericSample < minimumN || numericSample < minimumN * 2) {
    return 'LOW_CONFIDENCE';
  }

  return 'RELIABLE';
};

export const getVisualizationDataState = (items = [], minimumN = 30) => {
  const normalized = Array.isArray(items) ? items : [];
  const totalObservations = normalized.reduce((sum, item) => {
    const value = Number(item?.count ?? item?.actual ?? item?.value ?? item?.total ?? 0);
    return sum + (Number.isFinite(value) ? value : 0);
  }, 0);

  if (!normalized.length || totalObservations === 0) {
    return {
      state: 'EMPTY',
      showPlaceholder: true,
      total: 0,
      minimumN,
      message: 'No live data has been recorded yet. The dashboard is waiting for hospital events.',
    };
  }

  if (totalObservations < minimumN) {
    return {
      state: 'INSUFFICIENT',
      showPlaceholder: true,
      total: totalObservations,
      minimumN,
      message: `Only ${totalObservations} observations are available; ${minimumN} are needed before the chart is treated as reliable.`,
    };
  }

  return {
    state: 'RELIABLE',
    showPlaceholder: false,
    total: totalObservations,
    minimumN,
    message: '',
  };
};

export const getBreachPriorityOrder = (rows = []) => {
  return [...rows]
    .map((item) => {
      const allocated = Number(item.allocated || 0);
      const actual = Number(item.actual || 0);
      const breachPercent = allocated > 0 && actual > allocated ? ((actual - allocated) / allocated) * 100 : 0;

      return {
        ...item,
        breachPercent,
        severityWeight: getSeverityWeight(item.severity),
      };
    })
    .sort((left, right) => {
      if (right.severityWeight !== left.severityWeight) {
        return right.severityWeight - left.severityWeight;
      }

      return (right.severityWeight * right.breachPercent) - (left.severityWeight * left.breachPercent) || right.actual - left.actual;
    });
};

export const summarizeStaffWorkload = (staffRows = []) => {
  const rows = staffRows.map((staff) => {
    const shiftHours = Number(staff.shift || 0);
    const breakHours = Number(staff.break || 0);
    const actualHours = Number(staff.actual || 0);
    const weightedCases = Number(staff.weightedCases || 0);
    const hoursWorked = Math.max(actualHours, 0.1);
    const weightedCasesPerHour = shiftHours > 0 ? weightedCases / hoursWorked : 0;

    return {
      ...staff,
      weightedCasesPerHour,
      hoursWorked,
    };
  });

  const values = rows.map((row) => row.weightedCasesPerHour);
  const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const variance = values.length
    ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
    : 0;

  return {
    rows,
    mean,
    variance,
    nearZeroVariance: values.length > 1 && variance <= 0.01,
  };
};
