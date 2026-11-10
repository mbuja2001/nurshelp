// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import RadiologyDashboard from './RadiologyDashboard';

const { navigate, pdfSave } = vi.hoisted(() => ({ navigate: vi.fn(), pdfSave: vi.fn() }));

vi.mock('react-router-dom', () => ({ useNavigate: () => navigate }));
vi.mock('../../utils/jwt', () => ({ decodeJwt: () => ({ name: 'QA Radiologist' }) }));
vi.mock('jspdf', () => ({
  default: vi.fn(() => ({
    setFont: vi.fn(),
    setFontSize: vi.fn(),
    text: vi.fn(),
    splitTextToSize: (text) => [text],
    addPage: vi.fn(),
    save: pdfSave
  }))
}));
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

    if (String(url).endsWith('/radiology-workflow')) {
      const body = JSON.parse(options.body);
      const current = requests[0].imagingWorkflow || { stage: null, events: [] };
      const imagingWorkflow = {
        stage: body.stage,
        events: [...current.events, {
          stage: body.stage,
          occurred_at: new Date().toISOString(),
          recorded_by: 'QA Radiologist',
          source: 'MANUAL',
          viewer_url: body.viewer_url
        }]
      };
      requests = [{ ...requests[0], imagingWorkflow }];
      return makeResponse({ imagingWorkflow });
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
  pdfSave.mockReset();
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

    const workflow = [
      ['PATIENT_ARRIVED', 'Record: Patient arrived'],
      ['IDENTITY_VERIFIED', 'Record: Patient and order verified'],
      ['TECHNICIAN_HANDOFF', 'Record: Technician handoff'],
      ['ACQUISITION_STARTED', 'Record: External acquisition started'],
      ['ACQUISITION_COMPLETED', 'Record: External acquisition completed'],
      ['IMAGES_AVAILABLE', 'Record: Images available in PACS'],
      ['READY_FOR_INTERPRETATION', 'Record: Ready for interpretation']
    ];
    for (const [stage, buttonName] of workflow) {
      fireEvent.click(await screen.findByRole('button', { name: buttonName }));
      await waitFor(() => expect(requests[0].imagingWorkflow?.stage).toBe(stage));
    }

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

  it('shows empty live stages and clearly reports unavailable vendor integrations', async () => {
    requests = [];
    render(<RadiologyDashboard config={{ shiftId: 'shift-qa-1' }} />);

    expect(await screen.findByText('Radiology request pipeline')).toBeTruthy();
    expect(screen.getByRole('region', { name: 'New requests stage' })).toBeTruthy();
    expect(screen.getByRole('region', { name: 'Delivered stage' })).toBeTruthy();
    expect(screen.getAllByText('Not configured')).toHaveLength(6);
    expect(screen.getAllByText('No requests at this stage.').length).toBeGreaterThan(0);
    expect(screen.queryByText('John Doe')).toBeNull();
  });

  it('lets a radiology technician record manual status without specialist report actions', async () => {
    requests = [{ ...request, status: 'ACKNOWLEDGED', assigned_technician: 'QA Radiologist' }];
    render(<RadiologyDashboard config={{ shiftId: 'shift-qa-1' }} workflowRole="TECHNICIAN" />);

    await screen.findByText('Radiology request pipeline');
    fireEvent.click(await screen.findByRole('button', { name: /Chest X-ray/i }));

    expect(await screen.findByRole('button', { name: 'Record: Patient arrived' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Create & submit report' })).toBeNull();
    expect(screen.queryByText('Encounter context')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Record: Patient arrived' }));
    await waitFor(() => expect(requests[0].imagingWorkflow?.stage).toBe('PATIENT_ARRIVED'));
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/request-qa-1/radiology-workflow'),
      expect.objectContaining({ method: 'PATCH' })
    );
  });

  it('downloads a printable PDF for a saved radiology report', async () => {
    requests = [{
      ...request,
      status: 'COMPLETED',
      report_content: 'Synthetic findings.',
      clinical_impression: 'Synthetic impression.',
      submitted_by: 'QA Radiologist',
      submitted_at: new Date().toISOString()
    }];
    render(<RadiologyDashboard config={{ shiftId: 'shift-qa-1' }} />);

    await screen.findByText('Radiology request pipeline');
    fireEvent.click(await screen.findByRole('button', { name: /Chest X-ray/i }));
    fireEvent.click(await screen.findByRole('button', { name: 'Download printable PDF' }));
    expect(pdfSave).toHaveBeenCalledWith('radiology-report-request-qa-1.pdf');
  });
});
