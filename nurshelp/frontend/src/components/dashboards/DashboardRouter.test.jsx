// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DashboardRouter from './DashboardRouter';

vi.mock('../../Dashboard/NurseDashboard', () => ({
  default: ({ navigate }) => React.createElement(
    'button',
    { type: 'button', onClick: () => navigate('/triaged') },
    'Assigned Patients'
  )
}));

const LocationProbe = () => {
  const location = useLocation();
  return <output data-testid="current-location">{`${location.pathname}${location.search}`}</output>;
};

const createToken = () => `header.${btoa(JSON.stringify({ id: 'staff-1', role: 'NURSE' }))}.signature`;

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('DashboardRouter fallback error page', () => {
  it('requests dashboard config after the assigned-shift request fails and lets staff sign out', async () => {
    localStorage.setItem('token', createToken());
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 401, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({}) });
    vi.stubGlobal('fetch', fetchMock);

    render(
      <MemoryRouter initialEntries={['/shift-dashboard']}>
        <DashboardRouter />
        <LocationProbe />
      </MemoryRouter>
    );

    expect(await screen.findByText('Clinical dashboard unavailable')).toBeTruthy();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(fetchMock.mock.calls[0][0]).toContain('/api/shifts/assigned/today');
    expect(fetchMock.mock.calls[1][0]).toContain('/api/dashboard/config');

    fireEvent.click(screen.getByRole('button', { name: /sign out/i }));

    expect(localStorage.getItem('token')).toBeNull();
    expect(screen.getByTestId('current-location').textContent).toBe('/dashboard?login=1');
  });

  it('passes router navigation to the triage nurse dashboard header buttons', async () => {
    localStorage.setItem('token', createToken());
    const fetchMock = vi.fn(async (url) => {
      if (String(url).includes('/api/shifts/assigned/today')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            shift: {
              _id: 'shift-1',
              shiftName: 'Day shift',
              shiftStart: '08:00',
              shiftEnd: '16:00',
              assignedStaff: [{ staffID: 'staff-1', assignmentRole: 'TRIAGE_NURSE' }]
            }
          })
        };
      }

      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, record: { status: 'CLOCKED_IN' } })
      };
    });
    vi.stubGlobal('fetch', fetchMock);

    render(
      <MemoryRouter initialEntries={['/shift-dashboard']}>
        <DashboardRouter />
        <LocationProbe />
      </MemoryRouter>
    );

    fireEvent.click(await screen.findByRole('button', { name: 'Assigned Patients' }));

    expect(screen.getByTestId('current-location').textContent).toBe('/triaged');
  });
});
