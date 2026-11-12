// @vitest-environment jsdom

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SupportTicketInbox from './SupportTicketInbox';

afterEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('SupportTicketInbox', () => {
  it('loads active hospital tickets into the admin queue', async () => {
    localStorage.setItem('token', `header.${btoa(JSON.stringify({ id: 'manager-1', role: 'MANAGER' }))}.signature`);
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        success: true,
        tickets: [{
          _id: 'ticket-1',
          title: 'Dashboard unavailable',
          description: 'Staff dashboard failed to load.',
          category: 'SYSTEM_ERROR',
          severity: 'HIGH',
          status: 'NEW',
          reporterName: 'Nurse One',
          reporterRole: 'NURSE',
          createdAt: '2026-11-12T10:00:00.000Z'
        }]
      })
    }));
    vi.stubGlobal('fetch', fetchMock);

    render(<SupportTicketInbox />);

    expect(await screen.findByText('Dashboard unavailable')).toBeTruthy();
    expect(screen.getByText('Staff dashboard failed to load.')).toBeTruthy();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(fetchMock.mock.calls[0][0]).toContain('/api/support-tickets?status=OPEN');
  });
});
