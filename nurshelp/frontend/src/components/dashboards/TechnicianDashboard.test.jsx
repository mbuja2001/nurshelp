import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import TechnicianDashboard from './TechnicianDashboard';

vi.mock('../../Dashboard/SidebarLayout', () => ({
  default: ({ children }) => <div className="sidebar-layout">{children}</div>
}));

describe('TechnicianDashboard', () => {
  it('renders the pathology sample and custody workspace without invented study data or enabled actions', () => {
    const view = renderToStaticMarkup(
      <TechnicianDashboard
        specialty="PATHOLOGY"
        config={{ shiftName: 'Pathology coverage', department: 'General', shiftStartTime: '07:00', shiftEndTime: '15:00' }}
      />
    );

    expect(view).toContain('sidebar-layout');
    expect(view).toContain('Samples in progress');
    expect(view).toContain('Chain of custody');
    expect(view).toContain('Shipment to central laboratory');
    expect(view).toContain('Freezers and equipment');
    expect(view).toContain('Laboratory scientist only');
    expect(view).toContain('Not configured');
    expect(view).not.toContain('S-88214');
    expect(view).not.toContain('PT-0142');
    expect(view).not.toContain('<button');
  });
});
