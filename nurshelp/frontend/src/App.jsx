// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";

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
import DoctorTriageReport from "./PatientSummary/DoctorTriageReport.jsx";
import InpatientReport from "./PatientSummary/InpatientReport.jsx";
import NurseConfirmedReportView from "./PatientSummary/NurseConfirmedReportView.jsx";
import AssignPatientaWardAndBed from "./PatientSummary/AssignPatientaWardAndBed.jsx";
import ReferenceCase from "./PatientSummary/ReferenceCase.jsx";
import Profile from "./Dashboard/Pages/Profile.jsx";
import EHRRecords from "./Dashboard/Pages/EHRRecords.jsx";
import TriagedPatients from "./Dashboard/Pages/TriagedPatients.jsx";
import BedsAllocation from "./Dashboard/Pages/BedsAllocation.jsx";
import StaffScheduleManager from "./Dashboard/Pages/StaffScheduleManager.jsx";
import Pharmacy from "./Dashboard/Pages/Pharmacy.jsx";
import PharmacyPublic from "./Dashboard/Pages/PharmacyPublic.jsx";
import { getUserFromToken } from './utils/jwt';
import RadiologyDashboard from "./components/dashboards/RadiologyDashboard.jsx";
import PathologyDashboard from "./components/dashboards/PathologyDashboard.jsx";
import DashboardTools from "./Dashboard/Pages/DashboardTools.jsx";
import StaffToPharmacist from "./Dashboard/Pages/StaffToPharmacist.jsx";
import Settings from "./Dashboard/Pages/Settings.jsx";
import TriageAnalytics from "./Dashboard/Pages/TriageAnalytics.jsx";
import ReferenceCases from "./Dashboard/Pages/ReferenceCases.jsx";
import Help from "./Dashboard/Pages/Help.jsx";
import Notifications from "./Dashboard/Pages/Notifications.jsx";
import ClockInOutBoard from "./Dashboard/Pages/ClockInOutBoard.jsx";
import MyClockHistoryPage from "./Dashboard/Pages/MyClockHistoryPage.jsx";
import ManagerApprovalPage from "./Dashboard/Pages/ManagerApprovalPage.jsx";
import StatisticsPage from "./Dashboard/Pages/StatisticsPage.jsx";
import AdminSettingsPage from "./Dashboard/Pages/AdminSettingsPage.jsx";
import IntegratedSchedule from "./Dashboard/Pages/IntegratedSchedule.jsx";
import PrescriptionReport from "./components/DashboardPages/PrescriptionReport.jsx";
import OrderedMeds from "./Dashboard/Pages/OrderedMeds.jsx";
import LandingPage from "./LandingPage.jsx";
import Team from "./LandingPages/Team.jsx";
import WhatWeDo from "./LandingPages/WhatWeDo.jsx";
import EthicsResponsibility from "./LandingPages/EthicsResponsibility.jsx";
import ClinicalValidation from "./LandingPages/ClinicalValidation.jsx";
import SecurityCompliance from "./LandingPages/SecurityCompliance.jsx";

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

function App() {
  return (
    <Router>
      <div className="app-container">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          
          {/* Shift-based Conditional Dashboards (Full Day Simulation) */}
          <Route path="/shift-dashboard" element={<DashboardRouter />} />

          {/* CREATE (intake) */}
          <Route path="/summary/new" element={<PS />} />

          {/* DIAGNOSIS WORKFLOW */}
          <Route path="/diagnose/:id" element={<DiagnosePage />} />
          <Route path="/surgery/:id" element={<SurgeryPage />} />
          <Route path="/assign-ward-bed/:encounterId" element={<AssignPatientaWardAndBed />} />

          {/* AI SUGGESTED REPORT (editable working draft) ✅ */}
          <Route path="/summary/:id" element={<PatientReport />} />

          {/* CLINICIAN SCHEDULER (manually pick staff for encounter) */}
          <Route path="/scheduler" element={<ClinicianScheduler />} />

          {/* CLINICAL HANDOVER CONFIRMATIONS */}
          <Route path="/handover/ai/:encounterId" element={<ConfirmingAIClinicalHandover />} />
          <Route path="/handover/confirm/:encounterId" element={<ConfirmingAIClinicalHandover />} />
          <Route path="/handover/manual/:encounterId" element={<ConfirmingManualClinicalHandover />} />

          {/* OFFICIAL TRIAGE REPORTS */}
          <Route path="/report/triage/:encounterId" element={<OfficialTriageReport />} />
          <Route path="/report/doctor/:encounterId" element={<DoctorTriageReport />} />
          <Route path="/report/inpatient/:encounterId" element={<InpatientReport />} />

          {/* NURSE CONFIRMED REPORT (read-only final) ✅ */}
          <Route path="/confirmed/:id" element={<NurseConfirmedReportView />} />
          <Route path="/confirmed/encounter/:encounterId" element={<NurseConfirmedReportView />} />

          {/* CASE DETAIL (reference case view) */}
          <Route path="/summary/:encounterId/case/:caseId" element={<ReferenceCase />} />

          <Route path="/voice-intake" element={<VoiceIntakeRoute />} />

          {/* New sidebar pages */}
          <Route path="/profile" element={<Profile />} />
          <Route path="/ehr" element={<EHRRecords />} />
          <Route path="/pharmacy" element={
            (() => {
              const token = localStorage.getItem('token');
              const user = getUserFromToken(token) || {};
              return user.role === 'PHARMACIST' ? <Pharmacy /> : <PharmacyPublic />;
            })()
          } />
          <Route path="/pharmacy-public" element={<PharmacyPublic />} />
          <Route path="/staff-to-pharmacist" element={<StaffToPharmacist />} />
          <Route path="/dashboard-tools" element={<DashboardTools />} />
          <Route path="/ordered-meds" element={<OrderedMeds />} />
          <Route path="/radiology" element={<RadiologyDashboard />} />
          <Route path="/pathology" element={<PathologyDashboard />} />
          <Route path="/triaged" element={<TriagedPatients />} />
          <Route path="/beds" element={<BedsAllocation />} />
          <Route path="/schedule/staff" element={<StaffScheduleManager />} />
          <Route path="/integrated-schedule" element={<IntegratedSchedule token={localStorage.getItem('token')} user={JSON.parse(localStorage.getItem('user') || '{}')} />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/analytics" element={<TriageAnalytics />} />
          <Route path="/reference-cases" element={<ReferenceCases />} />
          <Route path="/help" element={<Help />} />
          <Route path="/notifications" element={<Notifications />} />
          <Route path="/prescription/:prescriptionId" element={<PrescriptionReport />} />

          {/* Clock Management Routes */}
          <Route path="/clock-in-out" element={<ClockInOutBoard token={localStorage.getItem('token')} user={JSON.parse(localStorage.getItem('user') || '{}')} />} />
          <Route path="/clock-history" element={<MyClockHistoryPage token={localStorage.getItem('token')} />} />
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
      </div>
    </Router>
  );
}

export default App;
