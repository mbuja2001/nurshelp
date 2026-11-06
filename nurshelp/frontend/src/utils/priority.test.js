import { computePriority, waitStartFor } from './priority';

const waitingPatient = (overrides = {}) => ({
  triageCategory: 'Yellow',
  createdAt: '2026-09-03T10:00:00.000Z',
  vitals: { heartRate: 90, systolicBP: 120, oxygenSaturation: 97, temperature: 37 },
  ...overrides
});

describe('computePriority clinical event matrix', () => {
  it('raises a patient with shock index above 1', () => {
    const priority = computePriority(waitingPatient({ vitals: { heartRate: 130, systolicBP: 120 } }), Date.parse('2026-09-03T10:10:00.000Z'));
    expect(priority.physiologicalDrift.shockIndexAlert).toBe(true);
    expect(priority.queueSignals).toContain('Shock index > 1.0');
  });

  it('detects a hypoxic drop of at least three percent', () => {
    const priority = computePriority(waitingPatient({
      vitalsHistory: [{ oxygenSaturation: 97 }],
      vitals: { oxygenSaturation: 93 }
    }), Date.parse('2026-09-03T10:10:00.000Z'));
    expect(priority.physiologicalDrift.hypoxicTrendAlert).toBe(true);
    expect(priority.queueSignals).toContain('O2 drop >= 3%');
  });

  it('detects falling heart rate paired with falling blood pressure', () => {
    const priority = computePriority(waitingPatient({
      vitalsHistory: [{ heartRate: 120, systolicBP: 110 }],
      vitals: { heartRate: 80, systolicBP: 90 }
    }), Date.parse('2026-09-03T10:10:00.000Z'));
    expect(priority.physiologicalDrift.relativeBradycardia).toBe(true);
  });

  it('flags documented asplenic fever and time debt', () => {
    const priority = computePriority(waitingPatient({
      criticalHistory: { implantsOrProcedures: ['Recent splenectomy'] },
      vitals: { temperature: 39 }
    }), Date.parse('2026-09-03T11:01:00.000Z'));
    expect(priority.crossRisk.asplenicSepsisRisk).toBe(true);
    expect(priority.timeDebt).toBe(true);
    expect(priority.queueSignals).toEqual(expect.arrayContaining(['Asplenic fever risk', 'Time-to-target breached']));
  });

  it('does not invent history signals for a new patient', () => {
    const priority = computePriority(waitingPatient({ criticalHistory: undefined }), Date.parse('2026-09-03T10:01:00.000Z'));
    expect(priority.crossRisk.activeCoagulopathy).toBe(false);
    expect(priority.crossRisk.asplenicSepsisRisk).toBe(false);
    expect(priority.queueSignals).toEqual([]);
  });
});import { describe, it, expect } from 'vitest';
import { computePriority } from './priority';

describe('computePriority SATS logic', () => {
  it('returns Critical for Red SATS', () => {
    const now = 1_000_000_000_000;
    const p = { report: { satsCategory: 'Red' }, waitStartedAt: now - 1000 };
    const res = computePriority(p, now);
    expect(res.label).toBe('Critical');
    expect(res.score).toBeGreaterThan(90000);
  });

  it('escalates Green to High when wait exceeds threshold', () => {
    const now = 2_000_000_000_000;
    // Green threshold is 240 minutes => 240*60*1000 = 14_400_000 ms
    const threshold = 240 * 60 * 1000;
    const p = { /* no report => severity 1 -> Green */ waitStartedAt: now - Math.floor(threshold * 1.2) };
    const res = computePriority(p, now);
    expect(res.label).toBe('High');
    expect(res.waitRatio).toBeGreaterThan(1);
  });

  it('returns Med for Orange when waitRatio >= 0.5', () => {
    const now = 3_000_000_000_000;
    const threshold = 10 * 60 * 1000; // Orange
    const p = { report: { satsCategory: 'Orange' }, waitStartedAt: now - Math.floor(threshold * 0.6) };
    const res = computePriority(p, now);
    expect(res.label).toBe('Med');
  });

  it('returns Low for short waits', () => {
    const now = 4_000_000_000_000;
    const threshold = 60 * 60 * 1000; // Yellow
    const p = { report: { satsCategory: 'Yellow' }, waitStartedAt: now - Math.floor(threshold * 0.1) };
    const res = computePriority(p, now);
    expect(res.label).toBe('Low');
  });

  it('normalizes numeric and lowercase priority codes into the correct SATS category', () => {
    const now = 5_000_000_000_000;
    const numeric = computePriority({ report: { priority_code: 1 } }, now);
    const lowercase = computePriority({ report: { satsCategory: 'red' } }, now);
    expect(numeric.category).toBe('Red');
    expect(lowercase.category).toBe('Red');
  });

  it('uses API snake_case timestamps and nested triage reports', () => {
    const now = Date.parse('2026-09-11T12:00:00.000Z');
    const priority = computePriority({
      triage: { aiTriageReport: { priority_code: 'ORANGE' } },
      created_at: '2026-09-11T11:00:00.000Z'
    }, now);
    expect(priority.category).toBe('Orange');
    expect(priority.waitRatio).toBeGreaterThan(1);
  });

  it('prefers an explicit manual SATS category over numeric encounter severity', () => {
    const priority = computePriority({
      severity: 1,
      manual_triage_report: { satsLevel: 'GREEN' },
      ai_triage_report: { calculated_priority: 'YELLOW' }
    }, Date.parse('2026-09-11T12:00:00.000Z'));

    expect(priority.category).toBe('Green');
  });

  it('resolves the alert wait timestamp from snake_case encounter fields', () => {
    expect(waitStartFor({ created_at: '2026-09-11T11:00:00.000Z' })).toBe('2026-09-11T11:00:00.000Z');
  });
});
