import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import NurseDashboard from '../../Dashboard/NurseDashboard';
import ClinicalNurseDashboard from './ClinicalNurseDashboard';
import PhysicianDashboard from './PhysicianDashboard';
import OffDutyDashboard from './OffDutyDashboard';
import PharmacistDashboard from './PharmacistDashboard';
import Radiology from '../../Dashboard/Pages/Radiology';
import Pathology from '../../Dashboard/Pages/Pathology';
import { decodeJwt, normalizeRoleLabel } from '../../utils/jwt';
import './DashboardRouter.css';

/**
 * DashboardRouter Component
 * 
 * MAIN ORCHESTRATOR for shift-based conditional dashboard rendering
 * 
 * Flow:
 * 1. On mount, calls GET /api/dashboard/config
 * 2. Backend checks if user is assigned to today's shift (via JWT)
 * 3. Returns config.context based on which role matches (TRIAGE_NURSE, CLINICAL_NURSE_ER, PHYSICIAN_ER, NO_SHIFT)
 * 4. Conditionally renders the appropriate dashboard component
 * 5. If not assigned: Shows 403 error "Contact your manager"
 */

const DashboardRouter = () => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
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
          // fallback to server-side config
          console.log('[DASHBOARD ROUTER] Token decode failed, falling back to server config');
          const fallback = await fetch('http://localhost:5001/api/dashboard/config', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` }
          });
          if (!fallback.ok) throw new Error('Failed fallback dashboard config');
          const fallbackData = await fallback.json();
          setConfig(fallbackData.config);
          return;
        }

        const normalizedRole = normalizeRoleLabel(decoded.role, decoded.specialty || decoded.clinicalSpecialty || decoded.department);
        const isSpecialistRole = ['PHARMACIST', 'RADIOLOGIST', 'PATHOLOGIST'].includes(normalizedRole);

        if (isSpecialistRole) {
          const specialistContext = normalizedRole === 'PHARMACIST'
            ? 'PHARMACY'
            : normalizedRole === 'RADIOLOGIST'
              ? 'RADIOLOGY'
              : 'PATHOLOGY';

          setConfig({
            userId: decoded.id || decoded.sub || null,
            hasShiftToday: true,
            assignedRole: normalizedRole,
            context: specialistContext,
            components: [],
            permissions: {},
            department: decoded.department || specialistContext,
            shiftName: 'Specialist Workflow',
            shiftStartTime: null,
            shiftEndTime: null
          });
          return;
        }

        // Primary flow: call authoritative endpoint that returns user's assigned shift for TODAY
        const shiftResp = await fetch('http://localhost:5001/api/shifts/assigned/today', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` }
        });

        if (!shiftResp.ok) {
          // If server returns 401/403 or other, fallback to dashboard/config
          console.log('[DASHBOARD ROUTER] assigned/today returned', shiftResp.status, ' - falling back');
          const fallback = await fetch('http://localhost:5001/api/dashboard/config', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` }
          });
          if (!fallback.ok) throw new Error('Failed fallback dashboard config');
          const fallbackData = await fallback.json();
          setConfig(fallbackData.config);
          return;
        }

        const shiftData = await shiftResp.json();
        // shiftData: { success: true, shift: shiftObj | null }
        if (!shiftData || !shiftData.shift) {
          // No assigned shift today -> OFF_DUTY
          setConfig({ context: 'NO_SHIFT', hasShiftToday: false });
          return;
        }

        const shift = shiftData.shift;
        const team = shift.teamAssignment || {};
        const userId = decoded.id || decoded.sub || null;

        // Determine assigned role from teamAssignment
        let context = 'NO_SHIFT';
        if (userId && team) {
          const uid = userId.toString();
          if (team.triageNurseId && team.triageNurseId.toString() === uid) context = 'TRIAGE_NURSE';
          else if (team.clinicalNurseId && team.clinicalNurseId.toString() === uid) context = team.clinicalNurseContext || 'CLINICAL_NURSE_ER';
          else if (team.physicianId && team.physicianId.toString() === uid) context = team.physicianContext || 'PHYSICIAN_ER';
        }

        // Build a minimal config object compatible with existing components
        const cfg = {
          userId: userId,
          hasShiftToday: true,
          shiftId: shift._id,
          assignedRole: context.startsWith('PHYSICIAN') ? 'PHYSICIAN' : (context === 'TRIAGE_NURSE' ? 'TRIAGE_NURSE' : 'CLINICAL_NURSE'),
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
        console.error('[DASHBOARD ROUTER] Error:', err);
        setError(err.message || 'Failed to load dashboard');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardConfig();
  }, [retryCount, navigate]);

  // LOADING STATE
  if (loading) {
    return (
      <div className="dashboard-router-loading">
        <div className="spinner"></div>
        <p>Loading your dashboard...</p>
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

  // RENDER APPROPRIATE DASHBOARD BASED ON CONTEXT
  const context = config.context;

  console.log('[DASHBOARD ROUTER] Rendering:', context);

  switch (context) {
    case 'TRIAGE_NURSE':
      return <NurseDashboard config={config} />;

    case 'CLINICAL_NURSE_ER':
      return <ClinicalNurseDashboard config={config} context="ER" />;

    case 'CLINICAL_NURSE_ICU':
      return <ClinicalNurseDashboard config={config} context="ICU" />;

    case 'CLINICAL_NURSE_SURGICAL':
      return <ClinicalNurseDashboard config={config} context="SURGICAL" />;

    case 'PHYSICIAN_ER':
      return <PhysicianDashboard config={config} context="ER" />;

    case 'PHYSICIAN_ICU':
      return <PhysicianDashboard config={config} context="ICU" />;

    case 'PHYSICIAN_SURGICAL':
      return <PhysicianDashboard config={config} context="SURGICAL" />;

    case 'PHARMACY':
      return <PharmacistDashboard />;

    case 'RADIOLOGY':
      return <Radiology />;

    case 'PATHOLOGY':
      return <Pathology />;

    case 'NO_SHIFT':
      return <OffDutyDashboard config={config} />;

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
