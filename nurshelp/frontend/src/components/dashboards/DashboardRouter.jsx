import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiClock, FiCalendar, FiBarChart2 } from 'react-icons/fi';
import { FaHistory } from 'react-icons/fa';
import NurseDashboard from '../../Dashboard/NurseDashboard';
import ClockInOutBoard from '../../Dashboard/Pages/ClockInOutBoard';
import ClinicalNurseDashboard from './ClinicalNurseDashboard';
import PhysicianDashboard from './PhysicianDashboard';
import PharmacistDashboard from './PharmacistDashboard';
import RadiologyDashboard from './RadiologyDashboard';
import PathologyDashboard from './PathologyDashboard';
import TechnicianDashboard from './TechnicianDashboard';
import DashboardLogoutButton from './DashboardLogoutButton';
import { decodeJwt, normalizeRoleLabel } from '../../utils/jwt';
import { resolveShiftDashboardContext } from '../../utils/shiftDashboardContext';
import './DashboardRouter.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

/**
 * DashboardRouter Component
 * 
 * MAIN ORCHESTRATOR for shift-based conditional dashboard rendering
 * 
 * Flow:
 * 1. Resolve today's assigned shift using the authenticated staff session.
 * 2. Send staff with no verifiable shift to the shared clock-in board.
 * 3. Require an assigned shift and verified clock-in before rendering a work dashboard.
 */

const DashboardRouter = () => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [clockStatusLoading, setClockStatusLoading] = useState(false);
  const [isClockedIn, setIsClockedIn] = useState(false);
  const [clockRecord, setClockRecord] = useState(null);
  const [staffAccessTab, setStaffAccessTab] = useState('clock');
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDashboardConfig = async () => {
      try {
        setLoading(true);
        setError(null);

        // Get JWT token from localStorage
        const token = localStorage.getItem('token');
        if (!token) {
          setError('No authentication token. Please log in.');
          navigate('/login');
          return;
        }

        console.log('[DASHBOARD ROUTER] Decoding token and fetching assigned shift...');

        const decoded = decodeJwt(token);
        if (!decoded) {
          console.error('[DASHBOARD ROUTER] Unable to decode the session token; showing the shared clock-in board');
          setConfig({ context: 'SHIFT_UNVERIFIED', hasShiftToday: false, assignedRole: 'STAFF' });
          return;
        }

        if (decoded.identityType === 'ACCOUNT') {
          navigate('/dashboard', { replace: true });
          return;
        }

        const tokenRole = decoded.systemRole || decoded.role;
        const tokenSpecialty = decoded.specialty || decoded.clinicalSpecialty || decoded.department;
        const normalizedRole = normalizeRoleLabel(tokenRole, tokenSpecialty);
        // Primary flow: call authoritative endpoint that returns user's assigned shift for TODAY
        const shiftResp = await fetch('http://localhost:5001/api/shifts/assigned/today', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` }
        });

        if (!shiftResp.ok) {
          console.error(`[DASHBOARD ROUTER] Unable to verify assigned shift (HTTP ${shiftResp.status}); showing the shared clock-in board`);
          setConfig({
            context: 'SHIFT_UNVERIFIED',
            hasShiftToday: false,
            assignedRole: normalizedRole,
            department: decoded.department || ''
          });
          return;
        }

        const shiftData = await shiftResp.json();
        // shiftData: { success: true, shift: shiftObj | null }
        if (!shiftData || !shiftData.shift) {
          setConfig({
            context: 'NO_SHIFT',
            hasShiftToday: false,
            assignedRole: normalizedRole,
            department: decoded.department || ''
          });
          return;
        }

        const shift = shiftData.shift;
        const userId = decoded.id || decoded.sub || null;
        const context = resolveShiftDashboardContext({ userId, role: tokenRole, specialty: tokenSpecialty, shift });

        // Build a minimal config object compatible with existing components
        const cfg = {
          userId: userId,
          hasShiftToday: true,
          shiftId: shift._id,
          assignedRole: context.startsWith('PHYSICIAN') ? 'PHYSICIAN'
            : context === 'TRIAGE_NURSE' ? 'TRIAGE_NURSE'
              : context.startsWith('CLINICAL_NURSE') ? 'CLINICAL_NURSE'
                : normalizedRole,
          context,
          components: [],
          permissions: {},
          shiftName: shift.shiftName,
          shiftStartTime: shift.shiftStart,
          shiftEndTime: shift.shiftEnd,
          department: shift.department
        };

        setConfig(cfg);
      } catch (err) {
        if (localStorage.getItem('token')) {
          console.error('[DASHBOARD ROUTER] Shift verification failed; showing the shared clock-in board:', err);
          setConfig({ context: 'SHIFT_UNVERIFIED', hasShiftToday: false, assignedRole: 'STAFF' });
        } else {
          console.error('[DASHBOARD ROUTER] Error:', err);
          setError(err.message || 'Failed to load dashboard');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardConfig();
  }, [retryCount, navigate]);

  useEffect(() => {
    if (!config) return;
    let cancelled = false;

    const checkClockStatusOnce = async () => {
      setClockStatusLoading(true);
      try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${BACKEND_URL}/api/staff/today-record`, {
          headers: { Authorization: `Bearer ${token}` },
          cache: 'no-store'
        });
        const data = await response.json().catch(() => ({}));
        const record = response.ok && data.success ? data.record : null;
        if (!cancelled) {
          setClockRecord(record);
          setIsClockedIn(Boolean(config.hasShiftToday && ['CLOCKED_IN', 'ON_BREAK'].includes(record?.status)));
        }
      } catch {
        if (!cancelled) {
          setClockRecord(null);
          setIsClockedIn(false);
        }
      } finally {
        if (!cancelled) setClockStatusLoading(false);
      }
    };

    checkClockStatusOnce();
    return () => { cancelled = true; };
  }, [config?.context]);

  // LOADING STATE
  if (loading) {
    return (
      <div className="dashboard-router-loading">
        <div className="spinner"></div>
        <p>Loading your dashboard...</p>
      </div>
    );
  }

  if (clockStatusLoading) {
    return (
      <div className="dashboard-router-loading">
        <div className="spinner"></div>
        <p>Checking your shift and clock-in status...</p>
      </div>
    );
  }

  // ERROR STATE: Not assigned to shift
  if (error && typeof error === 'object' && error.status === 403) {
    return (
      <div className="dashboard-not-assigned">
        <div className="alert alert-warning">
          <h2>⚠️ Not Assigned to Team</h2>
          <p>{error.message}</p>
          {error.shift && (
            <div className="shift-info">
              <p><strong>Shift:</strong> {error.shift.shiftName}</p>
              <p><strong>Status:</strong> {error.shift.status}</p>
            </div>
          )}
          <p className="contact-manager">Please contact your manager to be assigned a role for today's shift.</p>
          <button 
            className="btn-retry"
            onClick={() => setRetryCount(c => c + 1)}
          >
            ♻️ Check Again
          </button>
        </div>
      </div>
    );
  }

  // ERROR STATE: Other errors
  if (error && typeof error === 'string') {
    return (
      <div className="dashboard-error">
        <div className="alert alert-danger">
          <h2>❌ Error Loading Dashboard</h2>
          <p>{error}</p>
          <button className="btn-retry" onClick={() => setRetryCount(c => c + 1)}>
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  // NO CONFIG
  if (!config) {
    return (
      <div className="dashboard-error">
        <div className="alert alert-danger">
          <h2>❌ No Dashboard Configuration</h2>
          <p>Unable to determine your dashboard role.</p>
          <button className="btn-retry" onClick={() => setRetryCount(c => c + 1)}>
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  const staffTabs = [
    { id: 'clock', label: 'Clock In/Out', icon: FiClock },
    { id: 'myshifts', label: 'My Schedule', icon: FiCalendar },
    { id: 'history', label: 'Attendance History', icon: FaHistory }
  ];
  const needsStaffAccess = ['NO_SHIFT', 'SHIFT_UNVERIFIED'].includes(config.context)
    || !config.hasShiftToday
    || !isClockedIn;
  if (needsStaffAccess) {
    const token = localStorage.getItem('token');
    const user = token ? decodeJwt(token) : {};
    const roleName = String(config.assignedRole || user?.role || 'staff').replaceAll('_', ' ').toLowerCase();
    return (
      <main style={{ minHeight: '100vh', width: '100%', boxSizing: 'border-box', padding: '32px 24px', background: '#f8fafc' }}>
        <section style={{ width: 'min(1200px, 100%)', margin: '0 auto', padding: 24, border: '1px solid #e2e8f0', borderRadius: 12, background: '#fff' }}>
          <header style={{ marginBottom: 24 }}>
            <h1 style={{ margin: '0 0 8px', color: '#0f172a' }}>Clock in to access your dashboard</h1>
            <p style={{ margin: 0, color: '#475569' }}>
              Your {roleName} dashboard and work queues are available after you are assigned to a shift and clock in.
            </p>
          </header>
          <nav aria-label="Staff schedule and attendance" style={{ display: 'flex', gap: 8, marginBottom: 20, overflowX: 'auto', borderBottom: '1px solid #e2e8f0' }}>
            {staffTabs.map(({ id, label, icon: Icon }) => (
              <button key={id} type="button" role="tab" aria-selected={staffAccessTab === id} onClick={() => setStaffAccessTab(id)} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flex: '0 0 auto', padding: '11px 13px', border: 0, borderBottom: `3px solid ${staffAccessTab === id ? '#0f766e' : 'transparent'}`, background: 'transparent', color: staffAccessTab === id ? '#0f766e' : '#475569', fontSize: 13, fontWeight: 700, cursor: 'pointer' }}>
                <Icon size={16} />{label}
              </button>
            ))}
          </nav>
          <section role="tabpanel" aria-label={staffTabs.find((tab) => tab.id === staffAccessTab)?.label}>
            <ClockInOutBoard token={token} user={user} activeTab={staffAccessTab} onClockStatusChange={(record) => {
              setClockRecord(record);
              if (record?.status === 'CLOCKED_IN') {
                setConfig((current) => ({ ...current, hasShiftToday: true, shiftId: record.shiftId?._id || record.shiftId || current.shiftId }));
              }
              setIsClockedIn(['CLOCKED_IN', 'ON_BREAK'].includes(record?.status));
            }} />
          </section>
        </section>
        <DashboardLogoutButton />
      </main>
    );
  }

  // RENDER APPROPRIATE DASHBOARD BASED ON CONTEXT
  const context = config.context;

  console.log('[DASHBOARD ROUTER] Rendering:', context);

  switch (context) {
    case 'TRIAGE_NURSE':
      return <NurseDashboard navigate={navigate} config={{ ...config, clockStatusVerified: true, isClockedIn, clockRecord }} />;

    case 'CLINICAL_NURSE_ER':
      return <ClinicalNurseDashboard config={config} context="ER" />;

    case 'CLINICAL_NURSE_ICU':
      return <ClinicalNurseDashboard config={config} context="ICU" />;

    case 'CLINICAL_NURSE_SURGICAL':
      return <ClinicalNurseDashboard config={config} context="SURGICAL" />;

    case 'PHYSICIAN_ER':
      return <><DashboardLogoutButton /><PhysicianDashboard config={config} context="ER" /></>;

    case 'PHYSICIAN_ICU':
      return <><DashboardLogoutButton /><PhysicianDashboard config={config} context="ICU" /></>;

    case 'PHYSICIAN_SURGICAL':
      return <><DashboardLogoutButton /><PhysicianDashboard config={config} context="SURGICAL" /></>;

    case 'PHARMACY':
      return <><DashboardLogoutButton /><PharmacistDashboard config={config} /></>;

    case 'RADIOLOGY':
      return <RadiologyDashboard config={config} />;

    case 'PATHOLOGY':
      return <PathologyDashboard config={config} />;

    case 'TECHNICIAN_PHARMACY':
      return <TechnicianDashboard config={config} specialty="PHARMACY" />;

    case 'TECHNICIAN_RADIOLOGY':
      return <TechnicianDashboard config={config} specialty="RADIOLOGY" />;

    case 'TECHNICIAN_PATHOLOGY':
      return <TechnicianDashboard config={config} specialty="PATHOLOGY" />;

    case 'TECHNICIAN_UNCONFIGURED':
      return (
        <main className="technician-access-denied">
          <section className="technician-access-card" role="alert">
            <p className="technician-eyebrow">Technician access</p>
            <h1>Specialty assignment required</h1>
            <p>Your account or shift does not have a recognized technician specialty. No specialist workspace has been opened.</p>
            <p>Please contact your coordinator to confirm your role, specialty, and shift assignment.</p>
            <button className="btn-retry" onClick={() => setRetryCount(count => count + 1)}>Check again</button>
          </section>
          <DashboardLogoutButton />
        </main>
      );

    default:
      return (
        <div className="dashboard-error">
          <div className="alert alert-danger">
            <h2>❌ Unknown Dashboard Context</h2>
            <p>Context: {context}</p>
            <button className="btn-retry" onClick={() => setRetryCount(c => c + 1)}>
              🔄 Retry
            </button>
          </div>
        </div>
      );
  }
};

export default DashboardRouter;
