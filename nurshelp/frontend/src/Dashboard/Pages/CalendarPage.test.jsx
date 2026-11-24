// @vitest-environment jsdom

import React from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CalendarPage from './CalendarPage';

vi.mock('@fullcalendar/react', () => ({
  default: ({ events }) => <div data-testid="calendar-events">{events.map((event) => event.title).join(',')}</div>
}));

vi.mock('../SidebarLayout', () => ({
  default: ({ children }) => <div data-testid="sidebar-layout">{children}</div>
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('CalendarPage containment behavior', () => {
  it('shows an explicit error and no fabricated events when the API fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      json: async () => ({ message: 'Calendar service unavailable' })
    })));

    render(<MemoryRouter><CalendarPage /></MemoryRouter>);

    expect((await screen.findByRole('alert')).textContent).toContain('No sample events are displayed');
    expect(screen.getByTestId('calendar-events').textContent).toBe('');
    expect(screen.queryByText(/Appendectomy|CT Chest|Kareem/i)).toBeNull();
    expect(screen.getByTestId('sidebar-layout')).toBeTruthy();
  });

  it('renders only allowed event types and does not show disabled-category counts', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({
        events: [
          { id: 'surgery-1', title: 'Assigned surgery', type: 'surgery', start: '2026-11-23T10:00:00.000Z', end: '2026-11-23T11:00:00.000Z' },
          { id: 'appointment-1', title: 'Assigned appointment', type: 'appointment', start: '2026-11-23T12:00:00.000Z', end: '2026-11-23T12:30:00.000Z' }
        ],
        disabledCategories: ['lab', 'med']
      })
    })));

    render(<MemoryRouter><CalendarPage /></MemoryRouter>);

    expect((await screen.findByText(/Some calendar categories are temporarily unavailable/)).textContent)
      .toContain('Lab requests, Medication orders. No records or counts');
    expect(screen.getByText('Surgeries')).toBeTruthy();
    expect(screen.getAllByText('Appointments')).toHaveLength(3);
    expect(screen.getByTestId('calendar-events').textContent).toBe('Assigned surgery,Assigned appointment');
    expect(screen.queryByText('Labs')).toBeNull();
  });

  it('fails closed if the API sends a disabled category', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({
        events: [{ id: 'lab-1', title: 'Unauthorized lab event', type: 'lab' }],
        disabledCategories: ['lab']
      })
    })));

    render(<MemoryRouter><CalendarPage /></MemoryRouter>);

    expect((await screen.findByRole('alert')).textContent).toContain('included a disabled event category');
    expect(screen.getByTestId('calendar-events').textContent).toBe('');
    expect(screen.queryByText('Unauthorized lab event')).toBeNull();
  });
});
