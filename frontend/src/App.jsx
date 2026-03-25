// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";

import Dashboard from "./Dashboard/Dashboard.jsx";  // ensure exact filename
import PS from "./PatientSummary/PS.jsx";          // ensure exact filename
import VoiceIntake from "./Dashboard/VoiceIntake.jsx"; // ensure exact filename
import PatientReport from "./PatientSummary/PatientReport.jsx";
import ReferenceCase from "./PatientSummary/ReferenceCase.jsx";
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
          <Route path="/" element={<Dashboard />} />

          {/* CREATE (intake) */}
          <Route path="/summary/new" element={<PS />} />

          {/* REPORT (final page) ✅ */}
          <Route path="/summary/:id" element={<PatientReport />} />

          {/* CASE DETAIL (reference case view) */}
          <Route path="/summary/:encounterId/case/:caseId" element={<ReferenceCase />} />

          <Route path="/voice-intake" element={<VoiceIntakeRoute />} />

          <Route path="*" element={<Dashboard />} />
          </Routes>
      </div>
    </Router>
  );
}

export default App;
