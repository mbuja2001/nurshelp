// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DashboardLoadError from './DashboardLoadError';

const error = {
  code: 'DASHBOARD_CONFIG_FAILED',
  message: 'The dashboard service encountered an internal error.',
  technicalMessage: 'Dashboard configuration request failed',
  failedRequest: '/api/dashboard/config',
  primaryStatus: 500,
  fallbackStatus: 503,
  occurredAt: '2026-11-12T10:00:00.000Z'
};

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('DashboardLoadError', () => {
  it('submits only the safe dashboard diagnostic fields and the staff description', async () => {
    localStorage.setItem('token', 'test-token');
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 201,
      json: async () => ({ success: true, ticket: { _id: 'ticket-123' } })
    }));
    vi.stubGlobal('fetch', fetchMock);
    render(<MemoryRouter><DashboardLoadError error={error} onRetry={vi.fn()} /></MemoryRouter>);

    fireEvent.click(screen.getByRole('button', { name: /report to it support/i }));
    fireEvent.change(screen.getByLabelText(/what happened/i), {
      target: { value: 'The assigned dashboard did not load after sign-in.' }
    });
    fireEvent.click(screen.getByRole('button', { name: /submit it ticket/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const request = fetchMock.mock.calls[0][1];
    const payload = JSON.parse(request.body);
    expect(request.headers.Authorization).toBe('Bearer test-token');
    expect(payload).toMatchObject({
      title: 'Dashboard unavailable',
      description: 'The assigned dashboard did not load after sign-in.',
      category: 'SYSTEM_ERROR',
      severity: 'HIGH',
      failedRequest: '/api/dashboard/config',
      primaryStatus: 500,
      fallbackStatus: 503
    });
    expect(payload).not.toHaveProperty('token');
    expect(payload).not.toHaveProperty('patientId');
    expect(payload).not.toHaveProperty('headers');
    expect(await screen.findByText(/IT support ticket submitted/i)).toBeTruthy();
  });

  it('clears the authenticated session from the error page sign-out action', () => {
    localStorage.setItem('token', 'test-token');
    localStorage.setItem('user', '{"id":"staff-1"}');
    render(<MemoryRouter><DashboardLoadError error={error} onRetry={vi.fn()} /></MemoryRouter>);

    fireEvent.click(screen.getByRole('button', { name: /sign out/i }));

    expect(localStorage.getItem('token')).toBeNull();
    expect(localStorage.getItem('user')).toBeNull();
  });
});
