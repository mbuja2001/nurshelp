import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import SidebarLayout from '../SidebarLayout';
import { FiBarChart2, FiMessageCircle, FiBell, FiSettings } from 'react-icons/fi';

const DashboardTools = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (location.hash) {
      const target = document.getElementById(location.hash.replace('#', ''));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, [location.hash]);

  return (
    <SidebarLayout>
      <div style={{ maxWidth: 1100, width: '100%' }}>
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <FiBarChart2 size={28} style={{ color: '#0f172a' }} />
            <div>
              <h1 style={{ margin: 0, fontSize: 32, fontWeight: 800, color: '#0f172a' }}>Toolbar</h1>
              <p style={{ margin: '6px 0 0', color: '#475569', fontSize: 15 }}>Clinical toolbar actions for physician workflow and rapid task context.</p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'stretch', justifyContent: 'space-between' }}>
          {[
            {
              id: 'calendar-events',
              title: 'Calendar Events',
              description: 'On-call rosters, appointment planning, and day-of schedule context.',
              accent: '#2563eb'
            },
            {
              id: 'history-actions',
              title: 'History Actions',
              description: 'Encounter review, timeline events, and clinical decision support.',
              accent: '#0f172a'
            },
            {
              id: 'lab-requests',
              title: 'Lab Requests',
              description: 'Create, prioritize, and track lab and radiology orders with patient context.',
              accent: '#0ea5e9'
            },
            {
              id: 'ordered-meds',
              title: 'Ordered Meds',
              description: 'Medication order review, pharmacist handoff, and administration details.',
              accent: '#0f766e'
            },
            {
              id: 'scheduled-surgeries',
              title: 'Scheduled Surgeries',
              description: 'Theatre schedules, pre-op checklists, and surgical case coordination.',
              accent: '#c2410c'
            }
          ].map((item) => (
            <div
              key={item.id}
              id={item.id}
              style={{
                minWidth: 220,
                flex: '1 1 220px',
                padding: 20,
                borderRadius: 18,
                border: '1px solid #e2e8f0',
                background: '#ffffff',
                boxShadow: '0 12px 24px rgba(15,23,42,0.06)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: 12,
                transition: 'transform 150ms ease, box-shadow 150ms ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 20px 40px rgba(15,23,42,0.1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0px)';
                e.currentTarget.style.boxShadow = '0 12px 24px rgba(15,23,42,0.06)';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'inline-flex', width: 44, height: 44, alignItems: 'center', justifyContent: 'center', borderRadius: 14, background: `${item.accent}1a`, color: item.accent, fontWeight: 700, fontSize: 18 }}>
                  {item.title.charAt(0)}
                </div>
                <span style={{ color: item.accent, fontWeight: 800, fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Toolbar</span>
              </div>
              <div>
                <h2 style={{ margin: '0 0 8px', fontSize: 18, fontWeight: 800, color: '#0f172a' }}>{item.title}</h2>
                <p style={{ margin: 0, color: '#475569', fontSize: 14, lineHeight: 1.65 }}>{item.description}</p>
              </div>
              <button style={{
                alignSelf: 'flex-start',
                padding: '10px 16px',
                borderRadius: 999,
                border: 'none',
                background: item.accent,
                color: '#ffffff',
                cursor: 'pointer',
                fontWeight: 700,
                fontSize: 13,
                transition: 'background 150ms ease'
              }}
              onClick={() => {
                // Navigate to dedicated pages when available
                if (item.id === 'ordered-meds') return navigate('/ordered-meds');
                if (item.id === 'calendar-events') return navigate('/calendar');
                if (item.id === 'history-actions') return navigate('/history-actions');
                if (item.id === 'lab-requests') return navigate('/lab-requests');
                if (item.id === 'scheduled-surgeries') return navigate('/scheduled-surgeries');

                // Fallback: scroll to section
                const target = document.getElementById(item.id);
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }}>
                Open
              </button>
            </div>
          ))}
        </div>      </div>
    </SidebarLayout>
  );
};

export default DashboardTools;
