import { describe, it, expect } from 'vitest';
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
});

