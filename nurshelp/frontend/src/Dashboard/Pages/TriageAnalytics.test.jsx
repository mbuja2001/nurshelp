// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import TriageAnalytics from './TriageAnalytics';

vi.mock('../../hooks/useUserFromToken', () => ({
  useUserFromToken: () => ({ userName: 'Admin', userRole: 'ADMIN' }),
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('TriageAnalytics', () => {
  it('shows an explicit unavailable state instead of sample analytics when the API fails', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 503 })));

    render(<TriageAnalytics embedded mode="admin" />);

    expect(await screen.findByText('Live analytics are unavailable.')).toBeTruthy();
    expect(screen.getByText(/No sample values are being shown/i)).toBeTruthy();
    expect(screen.queryByText(/Sarah Johnson/)).toBeNull();
    expect(screen.queryByText('65')).toBeNull();
    expect(screen.queryByText(/from yesterday/i)).toBeNull();
    expect(screen.queryByRole('heading', { name: 'Admin reporting configuration' })).toBeNull();
  });

  it('renders empty states for a successful live response with no recorded events', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({
        success: true,
        waitTimeVsSeverity: [],
        diagnoses: [],
        caseReviews: [],
        staffHours: [],
        dailyMetrics: {
          totalPatients: 0,
          triageAccuracy: null,
          triageAccuracySampleSize: 0,
          mistriageRate: null,
          mistriageSampleSize: 0,
          avgWaitTime: null,
          avgWaitTimeSampleSize: 0,
          assignedCasesPerHour: null,
          assignedCasesPerHourSampleSize: 0,
          totalStaffHours: null,
          totalStaffHoursSampleSize: 0,
        },
      }),
    })));

    render(<TriageAnalytics embedded mode="admin" />);

    expect(await screen.findAllByText(/No live data has been recorded yet/)).toHaveLength(2);
    expect(screen.getAllByText('No staff clock records are available for this period.')).toHaveLength(2);
    expect(screen.getByRole('heading', { name: 'Admin Analytics Dashboard' })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: 'Admin reporting configuration' })).toBeNull();
    expect(screen.queryByText(/Sarah Johnson/)).toBeNull();
  });

  it('retries a failed analytics request when requested', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const fetchMock = vi.fn(async () => ({ ok: false, status: 503 }));
    vi.stubGlobal('fetch', fetchMock);
    render(<TriageAnalytics embedded mode="admin" />);

    fireEvent.click(await screen.findByRole('button', { name: 'Retry' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByText('Live analytics are unavailable.')).toBeTruthy();
  });
});
