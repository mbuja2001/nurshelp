import { describe, expect, it } from 'vitest';
import {
  calculateMistriageRate,
  calculateTrendDelta,
  computeReviewOutcome,
  getBreachPriorityOrder,
  getMetricConfidenceState,
  getVisualizationDataState,
} from './analyticsMetrics';
import { getAdminWorkspaceViewRegistration } from './adminDashboardViews';
import { resolveShiftDashboardContext } from './shiftDashboardContext';

describe('analytics metrics', () => {
  it('keeps live patient metrics visible when records exist, while only the empty sample remains insufficient', () => {
    expect(getMetricConfidenceState(40, 0, 30)).toBe('INSUFFICIENT');
    expect(getMetricConfidenceState(40, 5, 30)).toBe('LOW_CONFIDENCE');
    expect(getMetricConfidenceState(87.7, 35, 30)).toBe('LOW_CONFIDENCE');
    expect(getMetricConfidenceState(95.2, 120, 30)).toBe('RELIABLE');
  });

  it('computes mistriage only from urgency-category mismatch', () => {
    const cases = [
      { mistriageFlag: true, refinementOccurred: false },
      { mistriageFlag: false, refinementOccurred: true },
      { mistriageFlag: false, refinementOccurred: false },
    ];
    expect(calculateMistriageRate(cases)).toBe(33.3);
  });

  it('calculates real growth and decline percentages from the previous period', () => {
    expect(calculateTrendDelta(138, 120)).toBe(15);
    expect(calculateTrendDelta(42, 60, { inverse: true })).toBe(30);
    expect(calculateTrendDelta(9, 9)).toBe(0);
  });

  it('keeps ambiguous review outcomes out of the severity taxonomy until clinical review', () => {
    const outcome = computeReviewOutcome({
      aiSuggestedUrgency: 'Critical',
      nurseApprovedUrgency: 'High',
      physicianFinalUrgency: 'High',
      ambiguousClinicalReview: true,
    });

    expect(outcome).toBe('PENDING_CLINICAL_REVIEW');
  });

  it('prioritizes the worst breach by severity-weighted overshoot', () => {
    const rows = [
      { severity: 'Low', allocated: 120, actual: 95 },
      { severity: 'Critical', allocated: 5, actual: 8 },
      { severity: 'High', allocated: 15, actual: 28 },
    ];

    expect(getBreachPriorityOrder(rows)[0].severity).toBe('Critical');
  });

  it('classifies empty and low-volume chart data as insufficient while still rendering the visual', () => {
    expect(getVisualizationDataState([], 30)).toMatchObject({ state: 'EMPTY', showPlaceholder: true });
    expect(getVisualizationDataState([{ count: 2 }, { count: 2 }], 30)).toMatchObject({ state: 'INSUFFICIENT', showPlaceholder: true });
    expect(getVisualizationDataState([{ count: 30 }, { count: 40 }], 30)).toMatchObject({ state: 'RELIABLE', showPlaceholder: false });
  });

  it('routes admin workspace tabs to dedicated admin reporting pages instead of the specialist dashboards', () => {
    expect(getAdminWorkspaceViewRegistration('analytics')).toBe('TriageAnalyticsAdminPage');
    expect(getAdminWorkspaceViewRegistration('labs-configurations')).toBe('LabRequestsAdminPage');
    expect(getAdminWorkspaceViewRegistration('pharmacy-configurations')).toBe('PharmacyAdminPage');
  });

  it('routes nurses only from their explicit persisted shift assignment role', () => {
    const baseShift = { assignedStaff: [{ staffID: 'nurse-a', type: 'NURSE' }] };
    expect(resolveShiftDashboardContext({ userId: 'nurse-a', role: 'NURSE', shift: {
      assignedStaff: [{ ...baseShift.assignedStaff[0], assignmentRole: 'TRIAGE_NURSE' }]
    } })).toBe('TRIAGE_NURSE');
    expect(resolveShiftDashboardContext({ userId: 'nurse-a', role: 'NURSE', shift: {
      assignedStaff: [{ ...baseShift.assignedStaff[0], assignmentRole: 'CLINICAL_NURSE' }]
    } })).toBe('CLINICAL_NURSE_ER');
    expect(resolveShiftDashboardContext({ userId: 'nurse-a', role: 'NURSE', shift: baseShift })).toBe('NO_SHIFT');
  });

  it('continues to resolve legacy clinical team fields without guessing roster roles', () => {
    expect(resolveShiftDashboardContext({
      userId: 'legacy-nurse',
      role: 'NURSE',
      shift: { teamAssignment: { triageNurseId: 'legacy-nurse' } }
    })).toBe('TRIAGE_NURSE');
  });

  it('prefers the explicit roster duty over conflicting legacy team assignments', () => {
    expect(resolveShiftDashboardContext({
      userId: 'nurse-a',
      role: 'NURSE',
      shift: {
        assignedStaff: [{ staffID: 'nurse-a', type: 'NURSE', assignmentRole: 'TRIAGE_NURSE' }],
        teamAssignment: { clinicalNurseId: 'nurse-a', clinicalNurseContext: 'CLINICAL_NURSE_ER' }
      }
    })).toBe('TRIAGE_NURSE');
  });

  it('does not route a rostered nurse from unknown duty using stale legacy team fields', () => {
    expect(resolveShiftDashboardContext({
      userId: 'nurse-a',
      role: 'NURSE',
      shift: {
        assignedStaff: [{ staffID: 'nurse-a', type: 'NURSE' }],
        teamAssignment: { triageNurseId: 'nurse-a' }
      }
    })).toBe('NO_SHIFT');
  });
});
