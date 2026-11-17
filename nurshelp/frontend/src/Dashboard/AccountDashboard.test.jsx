// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AccountDashboard from './AccountDashboard';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('sandbox account hospital workspaces', () => {
  it('shows the NurseHelpDesk logo and a structured hospital card', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        hospitals: [{
          id: 'hospital-1',
          hospitalID: 'HOSP-001',
          name: 'Rural Health Centre',
          country: 'South Africa',
          environment: 'SANDBOX',
          departments: [{ name: 'Emergency' }, { name: 'Laboratory' }],
          admins: [{ id: 'admin-1' }],
          clinicalTrialMode: false,
          integrationStatus: 'SANDBOX'
        }]
      })
    }));

    render(<AccountDashboard account={{ name: 'Account Owner', email: 'owner@example.test' }} onLogout={vi.fn()} />);

    expect(await screen.findByAltText('NurseHelpDesk')).toBeTruthy();
    expect(await screen.findByText('Rural Health Centre')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Hospitals' }));

    expect(await screen.findByText('HOSPITAL WORKSPACE · HOSP-001')).toBeTruthy();
    expect(screen.getByText('Administrators')).toBeTruthy();
    expect(screen.getByText('South Africa · 2 departments')).toBeTruthy();
  });
});
