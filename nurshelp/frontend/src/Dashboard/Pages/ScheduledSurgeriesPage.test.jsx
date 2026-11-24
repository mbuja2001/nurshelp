// @vitest-environment jsdom

import React from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ScheduledSurgeriesPage from './ScheduledSurgeriesPage';

vi.mock('../SidebarLayout', () => ({
  default: ({ children }) => <div data-testid="sidebar-layout">{children}</div>
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('ScheduledSurgeriesPage containment behavior', () => {
  it('shows an explicit error and no fabricated patient records when the API fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      json: async () => ({ message: 'Surgery service unavailable' })
    })));

    render(<MemoryRouter><ScheduledSurgeriesPage /></MemoryRouter>);

    expect((await screen.findByRole('alert')).textContent).toContain('No sample records are displayed');
    expect(screen.getByRole('status').textContent).toContain('creation and updates are temporarily disabled');
    expect(screen.queryByRole('button', { name: 'New Surgery' })).toBeNull();
    expect(screen.queryByText(/John Doe|Jane Smith|Appendectomy|Cholecystectomy/i)).toBeNull();
    expect(screen.getByTestId('sidebar-layout')).toBeTruthy();
  });
});
