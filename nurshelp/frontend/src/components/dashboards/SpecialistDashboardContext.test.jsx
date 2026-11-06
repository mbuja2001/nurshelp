// @vitest-environment jsdom

import React from 'react';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import SpecialistDashboardContext from './SpecialistDashboardContext';

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => ({ success: true, settings: { environment: 'SANDBOX', clinicalTrialMode: false } })
  })));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('SpecialistDashboardContext', () => {
  it('shows the shared authorization context and radiology modules without implying PACS is connected', async () => {
    const { container } = render(
      <SpecialistDashboardContext specialty="RADIOLOGY" config={{ shiftId: 'shift-1', shiftName: 'Radiology QA' }} />
    );

    await waitFor(() => expect(screen.getByText('Sandbox · mock data only')).toBeTruthy());
    expect(container.textContent).toContain('Clinical trial mode');
    expect(container.textContent).toContain('Clocked in · verified shift');
    expect(container.textContent).toContain('Imaging workbench');
    expect(container.textContent).toContain('Short-lived, per-user viewer links');
    expect(container.textContent).toContain('A second radiologist must sign');
    expect(container.textContent).toContain('Not connected yet');
    expect(container.textContent).toContain('IMG-VIEW');
    expect(container.textContent).not.toContain('DICOM image');
  });

  it('shows pathology results and metadata-only viewer scope without slide imagery', async () => {
    const { container } = render(
      <SpecialistDashboardContext specialty="PATHOLOGY" config={{ shiftId: 'shift-1' }} />
    );

    await waitFor(() => expect(screen.getByText('Sandbox · mock data only')).toBeTruthy());
    expect(container.textContent).toContain('Results, diagnosis and release');
    expect(container.textContent).toContain('simulated viewer contains no images');
    expect(container.textContent).toContain('A second pathologist must sign');
    expect(container.textContent).toContain('LAB-QC-APPROVE');
    expect(container.textContent).not.toContain('slide pixel data');
  });
});
