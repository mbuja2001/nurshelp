// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import RadiologyDashboard from './RadiologyDashboard';

const { navigate } = vi.hoisted(() => ({ navigate: vi.fn() }));

vi.mock('react-router-dom', () => ({ useNavigate: () => navigate }));
vi.mock('../../utils/jwt', () => ({ decodeJwt: () => ({ name: 'QA Radiologist' }) }));
vi.mock('../../Dashboard/SidebarLayout', () => ({
  default: ({ children }) => <div>{children}</div>
}));
vi.mock('../diagnostics/EhrContextPanel', () => ({
  default: () => <div>Encounter context</div>
}));

const makeResponse = (data, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => data
});

const request = {
  encounterId: 'encounter-qa-1',
  request_id: 'request-qa-1',
  specialty: 'RADIOLOGY',
  lab_type: 'Chest X-ray',
  description: 'Assess for acute chest pathology',
  priority: 'URGENT',
  status: 'PENDING',
  requested_by: 'QA Physician',
  requested_at: new Date().toISOString(),
  patientName: 'QA Test Patient',
  patientAge: 38,
  patientId: 'QA-TEST-001',
  ehrContext: null
};

let requests;

beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  });
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url).includes('/queue/requests')) {
      return makeResponse({ lab_requests: requests });
    }

    if (String(url).endsWith('/acknowledge')) {
      requests = [{ ...requests[0], status: 'ACKNOWLEDGED', assigned_technician: 'QA Radiologist' }];
      return makeResponse({ status: 'ACKNOWLEDGED' });
    }

    if (String(url).endsWith('/start')) {
      requests = [{ ...requests[0], status: 'IN_PROGRESS', started_by: 'QA Radiologist' }];
      return makeResponse({ status: 'IN_PROGRESS', started_by: 'QA Radiologist', started_at: new Date().toISOString() });
    }

    if (String(url).endsWith('/submit-report')) {
      const body = JSON.parse(options.body);
      requests = [{
        ...requests[0],
        status: 'COMPLETED',
        report_content: body.report_content,
        clinical_impression: body.clinical_impression,
        recommendations: body.recommendations,
        submitted_by: 'QA Radiologist',
        submitted_at: new Date().toISOString()
      }];
      return makeResponse({ status: 'COMPLETED' });
    }

    return makeResponse({ message: 'Unexpected request' }, false, 404);
  }));
  localStorage.setItem('token', 'qa-token');
});

afterEach(() => {
  cleanup();
  delete HTMLElement.prototype.scrollIntoView;
  vi.unstubAllGlobals();
  localStorage.clear();
  navigate.mockReset();
});

describe('RadiologyDashboard', () => {
  it('moves a live request through acceptance, work, and report submission', async () => {
    requests = [{ ...request }];
    render(<RadiologyDashboard config={{ shiftId: 'shift-qa-1', shiftName: 'QA radiology shift' }} />);

    await screen.findByText('Radiology request pipeline');
    const requestCard = await screen.findByRole('button', { name: /Chest X-ray/i });
    fireEvent.click(requestCard);

    fireEvent.click(await screen.findByRole('button', { name: 'Accept & vet request' }));
    await waitFor(() => expect(requests[0].status).toBe('ACKNOWLEDGED'));
    fireEvent.click(await screen.findByRole('button', { name: 'Start radiologist work' }));
    await waitFor(() => expect(requests[0].status).toBe('IN_PROGRESS'));

    fireEvent.click(await screen.findByRole('button', { name: 'Create & submit report' }));
    fireEvent.change(screen.getByPlaceholderText(/Describe imaging findings/i), {
      target: { value: 'No acute cardiopulmonary abnormality.' }
    });
    fireEvent.change(screen.getByPlaceholderText(/Radiologist's clinical interpretation/i), {
      target: { value: 'No acute cardiopulmonary abnormality.' }
    });
    fireEvent.click(screen.getByRole('button', { name: 'Submit report to EHR' }));

    await waitFor(() => expect(requests[0].status).toBe('COMPLETED'));
    expect(await screen.findByText('Report saved to the encounter and made available to the ordering physician.')).toBeTruthy();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/request-qa-1/submit-report'),
      expect.objectContaining({ method: 'PATCH' })
    );
    expect(JSON.parse(fetch.mock.calls.find(([url]) => String(url).endsWith('/submit-report'))[1].body))
      .not.toHaveProperty('report_images');
  });

  it('uses the seven-stage live queue and labels PACS/RIS-only stages without fabricated requests', async () => {
    requests = [];
    render(<RadiologyDashboard config={{ shiftId: 'shift-qa-1' }} />);

    expect(await screen.findByText('Radiology request pipeline')).toBeTruthy();
    expect(screen.getByRole('region', { name: 'New requests stage' })).toBeTruthy();
    expect(screen.getByRole('region', { name: 'Delivered stage' })).toBeTruthy();
    expect(screen.getAllByText('Not configured')).toHaveLength(6);
    expect(screen.getAllByText('No requests at this stage.').length).toBeGreaterThan(0);
    expect(screen.queryByText('John Doe')).toBeNull();
  });
});
