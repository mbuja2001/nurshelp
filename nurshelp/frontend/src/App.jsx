// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate, useParams } from "react-router-dom";

import Dashboard from "./Dashboard/Dashboard.jsx";  // ensure exact filename
import PS from "./PatientSummary/PS.jsx";          // ensure exact filename
import VoiceIntake from "./Dashboard/VoiceIntake.jsx"; // ensure exact filename
import PatientReport from "./PatientSummary/PatientReport.jsx";
import DiagnosePage from "./PatientSummary/DiagnosePage.jsx";
import SurgeryPage from "./PatientSummary/SurgeryPage.jsx";
import ClinicianScheduler from "./PatientSummary/ClinicianScheduler.jsx";
import ConfirmingAIClinicalHandover from "./PatientSummary/ConfirmingAIClinicalHandover.jsx";
import ConfirmingManualClinicalHandover from "./PatientSummary/ConfirmingManualClinicalHandover.jsx";
import OfficialTriageReport from "./PatientSummary/OfficialTriageReport.jsx";
import ManualOfficialTriageReport from "./PatientSummary/ManualOfficialTriageReport.jsx";
import DoctorTriageReport from "./PatientSummary/DoctorTriageReport.jsx";
import InpatientReport from "./PatientSummary/InpatientReport.jsx";
import NurseConfirmedReportView from "./PatientSummary/NurseConfirmedReportView.jsx";
import ManualTriageReport from "./PatientSummary/manualtriagereport.jsx";
import OrderMedicationPage from "./PatientSummary/OrderMedicationPage.jsx";
import RequestLabPage from "./PatientSummary/RequestLabPage.jsx";
import LabRequestReport from "./PatientSummary/LabRequestReport.jsx";
import AssignPatientaWardAndBed from "./PatientSummary/AssignPatientaWardAndBed.jsx";
import ReferenceCase from "./PatientSummary/ReferenceCase.jsx";
import ReferenceCases from "./Dashboard/Pages/ReferenceCases.jsx";
import Profile from "./Dashboard/Pages/Profile.jsx";
import EHRRecords from "./Dashboard/Pages/EHRRecords.jsx";
import EHRLibrary from "./Dashboard/Pages/EHRLibrary.jsx";
import TriagedPatients from "./Dashboard/Pages/TriagedPatients.jsx";
import BedsAllocation from "./Dashboard/Pages/BedsAllocation.jsx";
import StaffScheduleManager from "./Dashboard/Pages/StaffScheduleManager.jsx";
import StaffStatusPage from "./Dashboard/Pages/StaffStatusPage.jsx";
import StaffTeamPage from "./Dashboard/Pages/StaffTeamPage.jsx";
import Pharmacy from "./Dashboard/Pages/Pharmacy.jsx";
import PharmacyPublic from "./Dashboard/Pages/PharmacyPublic.jsx";
import { decodeJwt, getUserFromToken, normalizeRoleLabel } from './utils/jwt';
import { apiFetch } from './utils/apiFetch';
import RadiologyDashboard from "./components/dashboards/RadiologyDashboard.jsx";
import PathologyDashboard from "./components/dashboards/PathologyDashboard.jsx";
import DashboardTools from "./Dashboard/Pages/DashboardTools.jsx";
import StaffTeamChat from "./Dashboard/Pages/StaffTeamChat.jsx";
import StaffToPharmacist from "./Dashboard/Pages/StaffToPharmacist.jsx";
import Settings from "./Dashboard/Pages/Settings.jsx";
import TriageAnalytics from "./Dashboard/Pages/TriageAnalytics.jsx";
import Help from "./Dashboard/Pages/Help.jsx";
import Notifications from "./Dashboard/Pages/Notifications.jsx";
import AlertsPage from "./Dashboard/Pages/AlertsPage.jsx";
import SystemPerformance from "./Dashboard/Pages/SystemPerformance.jsx";
import ClockInOutBoard from "./Dashboard/Pages/ClockInOutBoard.jsx";
import ClockDashboard from "./Dashboard/ClockDashboard.jsx";
import ManagerApprovalPage from "./Dashboard/Pages/ManagerApprovalPage.jsx";
import StatisticsPage from "./Dashboard/Pages/StatisticsPage.jsx";
import AdminSettingsPage from "./Dashboard/Pages/AdminSettingsPage.jsx";
import IntegratedSchedule from "./Dashboard/Pages/IntegratedSchedule.jsx";
import CalendarPage from "./Dashboard/Pages/CalendarPage.jsx";
import HistoryActionsPage from "./Dashboard/Pages/HistoryActionsPage.jsx";
import LabRequestsPage from "./Dashboard/Pages/LabRequestsPage.jsx";
import ScheduledSurgeriesPage from "./Dashboard/Pages/ScheduledSurgeriesPage.jsx";
import PrescriptionReport from "./components/DashboardPages/PrescriptionReport.jsx";
import OrderedMeds from "./Dashboard/Pages/OrderedMeds.jsx";
import DischargeReportPage from "./Dashboard/Pages/DischargeReportPage.jsx";
import LandingPage from "./LandingPage.jsx";
import Team from "./LandingPages/Team.jsx";
import WhatWeDo from "./LandingPages/WhatWeDo.jsx";
import EthicsResponsibility from "./LandingPages/EthicsResponsibility.jsx";
import ClinicalValidation from "./LandingPages/ClinicalValidation.jsx";
import SecurityCompliance from "./LandingPages/SecurityCompliance.jsx";
import ChartSessionBoundary from "./components/clinical/ChartSessionBoundary.jsx";

// Shift-based Conditional Dashboards (Full Day Simulation)
import DashboardRouter from "./components/dashboards/DashboardRouter.jsx";

import "./App.css";

/**
 * Route wrapper for VoiceIntake so it can keep using the same setView API.
 * When VoiceIntake calls `setView('patient')` we map that to navigate('/') to
 * preserve the same UX as when it was rendered inside the Dashboard controller.
 */
function VoiceIntakeRoute() {
  const navigate = useNavigate();

  const setView = (v) => {
    // preserve original behavior — go back to home for 'patient'
    if (v === "patient") navigate("/");
    else if (v === "nurse") navigate("/"); // fallback
    else navigate("/");
  };

  return <VoiceIntake setView={setView} />;
}

function TrackedChartRoute({ encounterParam = 'encounterId', children }) {
  const params = useParams();
  return <ChartSessionBoundary encounterId={params[encounterParam]}>{children}</ChartSessionBoundary>;
}

const routesAvailableBeforeClockIn = new Set([
  '/',
  '/dashboard',
  '/shift-dashboard',
  '/clock-in-out',
  '/clock-history',
  '/shift-management',
  '/staff-schedule',
  '/integrated-schedule',
  '/team',
  '/what-we-do',
  '/ethics',
  '/validation',
  '/security'
]);

function ShiftAccessBoundary({ children }) {
  const location = useLocation();
  const navigate = useNavigate();
  const token = localStorage.getItem('token');
  const user = token ? decodeJwt(token) : null;
  const identityType = String(user?.identityType || '').toUpperCase();
  const role = normalizeRoleLabel(user?.role, user?.specialty || user?.clinicalSpecialty || user?.department);
  const isExemptIdentity = identityType === 'ACCOUNT' || role === 'ADMIN' || role === 'MANAGER';
  const isExemptRoute = routesAvailableBeforeClockIn.has(location.pathname);
  const requiresClockIn = !isExemptIdentity && !isExemptRoute;
  const isAuthenticated = Boolean(token && user);
  const verificationKey = `${location.key}:${location.pathname}:${token || ''}`;
  const [verification, setVerification] = React.useState({ key: '', status: 'checking', error: '' });
  const [retryCount, setRetryCount] = React.useState(0);

  React.useEffect(() => {
    let cancelled = false;

    if (!requiresClockIn) return undefined;

    if (!isAuthenticated) {
      setVerification({ key: verificationKey, status: 'unauthenticated', error: '' });
      return undefined;
    }

    setVerification({ key: verificationKey, status: 'checking', error: '' });
    apiFetch('/api/staff/today-record', { cache: 'no-store' })
      .then((data) => {
        if (cancelled) return;
        if (!data.success) throw new Error("Unable to verify today's shift attendance.");

        const record = data.record;
        const isClockedIn = ['CLOCKED_IN', 'ON_BREAK'].includes(record?.status) && !record.clockOutTime;
        setVerification({ key: verificationKey, status: isClockedIn ? 'allowed' : 'blocked', error: '' });
      })
      .catch((error) => {
        if (!cancelled) {
          setVerification({
            key: verificationKey,
            status: 'error',
            error: error.message || 'Unable to verify your shift attendance.'
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [isAuthenticated, location.key, location.pathname, requiresClockIn, retryCount, token, verificationKey]);

  if (!requiresClockIn) return children;
  if (!isAuthenticated) return <Navigate to="/dashboard?login=1" replace state={{ from: location }} />;

  if (verification.key !== verificationKey || verification.status === 'checking') {
    return <main role="status" aria-live="polite" style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', color: '#475569', fontWeight: 700 }}>Verifying your shift and clock-in status...</main>;
  }

  if (verification.status === 'unauthenticated') {
    return <Navigate to="/dashboard?login=1" replace state={{ from: location }} />;
  }

  if (verification.status === 'blocked') {
    return <Navigate to="/dashboard" replace state={{ from: location }} />;
  }

  if (verification.status === 'error') {
    return (
      <main role="alert" style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: 24, color: '#334155' }}>
        <section>
          <p>Access is blocked because shift attendance could not be verified: {verification.error}</p>
          <button type="button" onClick={() => setRetryCount((count) => count + 1)}>Retry verification</button>
          <button type="button" onClick={() => navigate('/clock-in-out')}>Go to clock-in</button>
        </section>
      </main>
    );
  }

  return children;
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <ShiftAccessBoundary>
          <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          
          {/* Shift-based Conditional Dashboards (Full Day Simulation) */}
          <Route path="/shift-dashboard" element={<DashboardRouter />} />

          {/* CREATE (intake) */}
          <Route path="/summary/new" element={<PS />} />
          <Route path="/manual-triage-report" element={<ManualTriageReport />} />

          {/* DIAGNOSIS WORKFLOW */}
          <Route path="/diagnose/:id" element={<TrackedChartRoute encounterParam="id"><DiagnosePage /></TrackedChartRoute>} />
          <Route path="/surgery/:id" element={<TrackedChartRoute encounterParam="id"><SurgeryPage /></TrackedChartRoute>} />
          <Route path="/assign-ward-bed/:encounterId" element={<TrackedChartRoute><AssignPatientaWardAndBed /></TrackedChartRoute>} />

          {/* AI SUGGESTED REPORT (editable working draft) ✅ */}
          <Route path="/summary/:id" element={<TrackedChartRoute encounterParam="id"><PatientReport /></TrackedChartRoute>} />

          {/* CLINICIAN SCHEDULER (manually pick staff for encounter) */}
          <Route path="/scheduler" element={<ClinicianScheduler />} />
          <Route
            path="/shift-management"
            element={
              <StaffScheduleManager />
            }
          />
          <Route
            path="/staff-schedule"
            element={
              <StaffScheduleManager />
            }
          />

          {/* CLINICAL HANDOVER CONFIRMATIONS */}
          <Route path="/handover/ai/:encounterId" element={<TrackedChartRoute><ConfirmingAIClinicalHandover /></TrackedChartRoute>} />
          <Route path="/handover/confirm/:encounterId" element={<TrackedChartRoute><ConfirmingAIClinicalHandover /></TrackedChartRoute>} />
          <Route path="/handover/manual/:encounterId" element={<TrackedChartRoute><ConfirmingManualClinicalHandover /></TrackedChartRoute>} />

          {/* OFFICIAL TRIAGE REPORTS */}
          <Route path="/report/triage/:encounterId" element={<TrackedChartRoute><OfficialTriageReport /></TrackedChartRoute>} />
          <Route path="/report/manual-triage/:encounterId" element={<TrackedChartRoute><ManualOfficialTriageReport /></TrackedChartRoute>} />
          <Route path="/report/doctor/:encounterId" element={<DoctorTriageReport />} />
          <Route path="/report/inpatient/:encounterId" element={<InpatientReport />} />
          <Route path="/order-medication/:encounterId" element={<TrackedChartRoute><OrderMedicationPage /></TrackedChartRoute>} />
          <Route path="/request-lab/:encounterId" element={<TrackedChartRoute><RequestLabPage /></TrackedChartRoute>} />
          <Route path="/request-lab/:encounterId/:specialty" element={<TrackedChartRoute><RequestLabPage /></TrackedChartRoute>} />
          <Route path="/report/lab/:encounterId" element={<LabRequestReport />} />

          {/* NURSE CONFIRMED REPORT (read-only final) ✅ */}
          <Route path="/confirmed/:id" element={<NurseConfirmedReportView />} />
          <Route path="/confirmed/encounter/:encounterId" element={<NurseConfirmedReportView />} />

          {/* CASE DETAIL (reference case view) */}
          <Route path="/summary/:encounterId/case/:caseId" element={<ReferenceCase />} />

          <Route path="/voice-intake" element={<VoiceIntakeRoute />} />

          {/* New sidebar pages */}
          <Route path="/profile" element={<Profile />} />
          <Route path="/ehr" element={<EHRLibrary />} />
          <Route path="/pharmacy" element={
            (() => {
              const token = localStorage.getItem('token');
              const user = getUserFromToken(token) || {};
              return user.role === 'PHARMACIST' ? <Pharmacy /> : <OrderedMeds />;
            })()
          } />
          <Route path="/pharmacy-public" element={<OrderedMeds />} />
          <Route path="/staff-to-pharmacist" element={<StaffToPharmacist />} />
          <Route path="/dashboard-tools" element={<DashboardTools />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/system-performance" element={<SystemPerformance />} />
          <Route path="/staff-chat" element={<StaffTeamChat />} />
          <Route path="/staff-team" element={<StaffTeamPage />} />
          <Route path="/calendar" element={<CalendarPage />} />
          <Route path="/history-actions" element={<HistoryActionsPage />} />
          <Route path="/lab-requests" element={<LabRequestsPage />} />
          <Route path="/scheduled-surgeries" element={<ScheduledSurgeriesPage />} />
          <Route path="/ordered-meds" element={<OrderedMeds />} />
          <Route path="/radiology" element={
            (() => {
              const token = localStorage.getItem('token');
              const user = getUserFromToken(token) || {};
              return user.role === 'RADIOLOGIST' ? <RadiologyDashboard /> : <LabRequestsPage />;
            })()
          } />
          <Route path="/pathology" element={
            (() => {
              const token = localStorage.getItem('token');
              const user = getUserFromToken(token) || {};
              return user.role === 'PATHOLOGIST' ? <PathologyDashboard /> : <LabRequestsPage />;
            })()
          } />
          <Route path="/triaged" element={<TriagedPatients />} />
          <Route path="/beds" element={<BedsAllocation />} />
          <Route path="/staff-team" element={<StaffScheduleManager />} />
          <Route path="/integrated-schedule" element={<IntegratedSchedule token={localStorage.getItem('token')} user={JSON.parse(localStorage.getItem('user') || '{}')} />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/analytics" element={<TriageAnalytics />} />
          <Route path="/reference-cases" element={<ReferenceCases />} />
          <Route path="/help" element={<Help />} />
          <Route path="/notifications" element={<Notifications />} />
          <Route path="/prescription/:prescriptionId" element={<PrescriptionReport />} />
          <Route path="/discharge-report/:encounterId" element={<DischargeReportPage />} />

          {/* Clock Management Routes */}
          <Route path="/clock-in-out" element={<ClockInOutBoard token={localStorage.getItem('token')} user={JSON.parse(localStorage.getItem('user') || '{}')} />} />
          <Route path="/staff-status" element={<StaffStatusPage token={localStorage.getItem('token')} />} />
          <Route path="/dashboard/staff-status" element={<StaffStatusPage token={localStorage.getItem('token')} />} />
          <Route path="/approvals" element={<ManagerApprovalPage token={localStorage.getItem('token')} userRole={JSON.parse(localStorage.getItem('user') || '{}').role} />} />
          <Route path="/statistics" element={<StatisticsPage token={localStorage.getItem('token')} />} />
          <Route path="/admin-settings" element={<AdminSettingsPage token={localStorage.getItem('token')} />} />

          {/* Landing page sections */}
          <Route path="/team" element={<Team />} />
          <Route path="/what-we-do" element={<WhatWeDo />} />
          <Route path="/ethics" element={<EthicsResponsibility />} />
          <Route path="/validation" element={<ClinicalValidation />} />
          <Route path="/security" element={<SecurityCompliance />} />

          <Route path="*" element={<LandingPage />} />
          </Routes>
        </ShiftAccessBoundary>
      </div>
    </Router>
  );
}

export default App;
