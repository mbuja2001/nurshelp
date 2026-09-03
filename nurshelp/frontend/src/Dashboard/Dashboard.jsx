import React, { useEffect, useState, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import NurseDashboard from "./NurseDashboard";
import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";
import StaffScheduleManager from "./Pages/StaffScheduleManager";
import ClockInOutBoard from "./Pages/ClockInOutBoard";
import PharmacistDashboard from "../components/dashboards/PharmacistDashboard";
import RadiologyDashboard from "../components/dashboards/RadiologyDashboard";
import PathologyDashboard from "../components/dashboards/PathologyDashboard";
import { styles } from "./styles";
import { decodeJwt, normalizeRoleLabel, getDashboardRoute } from "../utils/jwt";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * API helper (CRITICAL for consistency)
 */
const apiFetch = async (url, options = {}) => {
  const token = localStorage.getItem("token");

  const res = await fetch(`${BACKEND_URL}${url}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    }
  });

  if (res.status === 401) {
    // 🔥 token expired / invalid
    localStorage.removeItem("token");
    window.location.reload();
    return;
  }

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.message || "Request failed");
  }

  return data;
};

/** dedupe by _id */
function dedupeById(list) {
  const seen = new Set();
  return list.filter(item => {
    const id = item._id || item.id;
    if (!id || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

const Dashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const [view, setView] = useState("home");
  const [nurse, setNurse] = useState(null);

  const [waitingPatients, setWaitingPatients] = useState([]);
  const [nurseEncounters, setNurseEncounters] = useState([]);

  const [bedsAvailable] = useState(12);
  const [loading, setLoading] = useState(false);

  const pollRef = useRef(null);
  const POLL_INTERVAL = 5000;

  // -----------------------------
  // SESSION RESTORE
  // -----------------------------
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const decoded = decodeJwt(token);
    if (decoded?.id) {
      setNurse({
        id: decoded.id,
        email: decoded.email,
        name: decoded.name || "Staff",
        role: normalizeRoleLabel(decoded.role, decoded.specialty || decoded.department),
        specialty: decoded.specialty || decoded.department,
        hospitalId: decoded.hospitalId
      });
      setView("dashboard");
    }
  }, []);

  // -----------------------------
  // FETCH WAITING (PER USER NOW)
  // -----------------------------
  const fetchWaiting = useCallback(async () => {
    try {
      const data = await apiFetch("/api/encounters/waiting");

      setWaitingPatients(
        Array.isArray(data) ? data : (data.encounters ?? [])
      );
    } catch (err) {
      console.warn("Waiting fetch failed:", err.message);
    }
  }, []);

  // -----------------------------
  // FETCH MY ENCOUNTERS
  // -----------------------------
  const fetchMyEncounters = useCallback(async () => {
    try {
      setLoading(true);

      const data = await apiFetch("/api/encounters");

      setNurseEncounters(
        Array.isArray(data) ? data : (data.encounters ?? [])
      );
    } catch (err) {
      console.error("Encounter fetch failed:", err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // -----------------------------
  // POLLING LOOP
  // -----------------------------
  useEffect(() => {
    if (!nurse) return;

    // Only fetch patient data if user is staff (not admin/manager)
    const isAdmin = nurse.role === "ADMIN" || nurse.role === "MANAGER";
    
    if (!isAdmin) {
      fetchWaiting();
      fetchMyEncounters();

      pollRef.current = setInterval(() => {
        fetchWaiting();
      }, POLL_INTERVAL);
    }

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [nurse, fetchWaiting, fetchMyEncounters]);

  // -----------------------------
  // NAVIGATION STATE (NEW PATIENT or NEW ENCOUNTER from triage)
  // -----------------------------
  useEffect(() => {
    if (location.state?.newPatient) {
      const newP = location.state.newPatient;

      setNurseEncounters(prev => {
        if (prev.find(p => p._id === newP._id)) return prev;
        return [newP, ...prev];
      });

      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  // Handle newly created encounter from manual triage form
  useEffect(() => {
    if (location.state?.newEncounterId) {
      console.log(`[DASHBOARD] New encounter from triage: ${location.state.newEncounterId}`);
      
      // Immediately refresh the waiting queue to pick up the new encounter
      fetchWaiting().then(() => {
        console.log(`[DASHBOARD] Waiting queue refreshed after new triage`);
      }).catch(err => {
        console.error(`[DASHBOARD] Failed to refresh waiting queue:`, err);
      });
      
      // Clear the navigation state to avoid re-triggering
      window.history.replaceState({}, document.title);
    }
  }, [location.state?.newEncounterId, fetchWaiting]);

  // Get user role from JWT token
  const getUserRole = () => {
    const token = localStorage.getItem("token");
    if (!token) return null;
    const decoded = decodeJwt(token);
    return decoded?.role || null;
  };

  // Check if user is admin/manager
  const isManagerRole = () => {
    const role = getUserRole();
    return role === "ADMIN" || role === "MANAGER";
  };

  const getNormalizedRole = (role, specialty) => normalizeRoleLabel(role, specialty);

  const isSpecialistRole = (role, specialty) => {
    const normalized = getNormalizedRole(role, specialty);
    return [
      "PHARMACIST",
      "RADIOLOGIST",
      "PATHOLOGIST"
    ].includes(normalized);
  };

  // -----------------------------
  // AUTH
  // -----------------------------
  const handleLogin = async ({ email, password }) => {
    try {
      // Unified login endpoint - works for all user types
      const endpoint = "/api/auth/login";

      const data = await apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify({ email, password })
      });

      if (data.token) {
        localStorage.setItem("token", data.token);
        window.dispatchEvent(new Event("nursehelpdesk:auth-changed"));

        const decoded = decodeJwt(data.token);
        localStorage.setItem("currentNurseId", decoded.id);
        localStorage.setItem("userRole", decoded.role);

        setNurse({
          id: decoded.id,
          email: decoded.email,
          name: decoded.name || "Staff",
          role: normalizeRoleLabel(decoded.role, decoded.specialty || decoded.department),
          specialty: decoded.specialty || decoded.department
        });

        setView("dashboard");

        // Only fetch patient encounters if nurse/physician (not admin/manager)
        if (decoded.role !== "ADMIN" && decoded.role !== "MANAGER") {
          await fetchMyEncounters();
          await fetchWaiting();
        }
      }
    } catch (err) {
      alert(err.message);
    }
  };

  const handleRegister = async (payload) => {
    try {
      // Only hospital admin registration
      const endpoint = "/api/hospital/register";
      const registerData = {
        hospitalName: payload.hospitalName,
        managerName: payload.managerName,
        managerSurname: payload.managerSurname,
        managerEmail: payload.managerEmail,
        managerPassword: payload.managerPassword,
        department: payload.department || "All"
      };

      const data = await apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify(registerData)
      });

      if (data.success) {
        // Login first
        await handleLogin({
          email: payload.managerEmail,
          password: payload.managerPassword
        });

        // Create staff members if provided
        if (payload.staffMembers && payload.staffMembers.length > 0) {
          console.log('[REGISTER] Creating staff members:', payload.staffMembers.length);
          for (const staff of payload.staffMembers) {
            try {
              console.log('[REGISTER] Creating staff:', staff.name, staff.surname, 'Email:', staff.email);
              const staffRes = await apiFetch("/api/manager/staff/register", {
                method: "POST",
                body: JSON.stringify({
                  name: staff.name,
                  surname: staff.surname,
                  email: staff.email,
                  password: staff.password,
                  role: staff.role,
                  department: staff.department,
                  specialty: staff.specialty
                })
              });
              console.log('[REGISTER] Staff created successfully:', staffRes.staff?.id, staffRes.staff?.hospitalId);
            } catch (staffErr) {
              console.error(`Failed to create staff ${staff.name} ${staff.surname}:`, staffErr.message);
            }
          }
        }

        // Create beds if provided
        if (payload.beds && payload.beds.length > 0) {
          console.log('[REGISTER] Creating beds:', payload.beds.length);
          try {
            const bedsRes = await apiFetch("/api/beds/batch", {
              method: "POST",
              body: JSON.stringify({ beds: payload.beds })
            });
            console.log('[REGISTER] Beds created successfully:', bedsRes.count);
          } catch (bedsErr) {
            console.error('[REGISTER] Failed to create beds:', bedsErr.message);
          }
        }

        // Create pharmacy resources if provided
        if (payload.pharmaceuticals && payload.pharmaceuticals.length > 0) {
          console.log('[REGISTER] Creating pharmacy resources:', payload.pharmaceuticals.length);
          try {
            const pharmaRes = await apiFetch("/api/pharmacy/batch", {
              method: "POST",
              body: JSON.stringify({ pharmaceuticals: payload.pharmaceuticals })
            });
            console.log('[REGISTER] Pharmacy resources created successfully:', pharmaRes.count);
          } catch (pharmaErr) {
            console.error('[REGISTER] Failed to create pharmacy resources:', pharmaErr.message);
          }
        }
      }
    } catch (err) {
      alert(err.message);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("currentNurseId");
    localStorage.removeItem("userRole");
    setNurse(null);
    setWaitingPatients([]);
    setNurseEncounters([]);
    setView("login");
  };

  // -----------------------------
  // DATA MERGE + SORT
  // -----------------------------
  const combined = dedupeById([
    ...(waitingPatients || []),
    ...(nurseEncounters || [])
  ]);

  const sorted = combined.sort((a, b) => {
    const sa = a.severity || a.triage?.severity || 1;
    const sb = b.severity || b.triage?.severity || 1;

    if (sb !== sa) return sb - sa;

    const ta = new Date(a.createdAt || 0).getTime();
    const tb = new Date(b.createdAt || 0).getTime();

    return ta - tb;
  });

  // -----------------------------
  // UI
  // -----------------------------
  if (view === "home") {
    return (
      <div style={styles.patientBg}>
        <div style={{ textAlign: "center" }}>
          <h1 style={{ fontSize: "3rem", marginBottom: "20px" }}>
            Nurse Helpdesk
          </h1>
          <p style={{ fontSize: "18px", color: "#64748b", marginBottom: "40px" }}>
            Hospital Management & Staff Scheduling System
          </p>

          <div style={{ display: "flex", gap: "20px", justifyContent: "center", flexWrap: "wrap" }}>
            <button
              onClick={() => setView("login")}
              style={{
                ...styles.btnWhite,
                minWidth: "140px"
              }}
            >
              Login
            </button>

            <button
              onClick={() => setView("register")}
              style={{
                ...styles.btnOutline,
                minWidth: "220px"
              }}
            >
              Register Hospital
            </button>

            <button
              onClick={() => navigate("/")}
              style={{
                ...styles.btnOutline,
                minWidth: "140px"
              }}
            >
              Back to Home
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (view === "login") {
    return <LoginForm onLogin={handleLogin} onCancel={() => setView("home")} />;
  }

  if (view === "register") {
    return <RegisterForm onRegister={handleRegister} onCancel={() => setView("home")} />;
  }

  if (view === "dashboard" && nurse) {
    // Check user role from JWT
    const userRole = nurse.role || getUserRole();
    const userSpecialty = nurse.specialty || null;

    // Route based on role:
    // - ADMIN/MANAGER: Staff scheduling management page
    // - NURSE/PHYSICIAN: Staff dashboard (home page for they can view patients, clock in/out, etc.)
    const normalizedRole = getNormalizedRole(userRole, userSpecialty);

    if (userRole === "ADMIN" || userRole === "MANAGER" || normalizedRole === "ADMIN" || normalizedRole === "MANAGER") {
      return (
        <StaffScheduleManager
          hospitalId={nurse.hospitalId}
          userRole={userRole}
          userName={nurse.name}
          onLogout={handleLogout}
        />
      );
    } else if (normalizedRole === "NURSE" || normalizedRole === "PHYSICIAN") {
      // Staff dashboard - their home page with patient tracking and clock in/out
      return (
        <NurseDashboard
          navigate={navigate}
          setView={setView}
          username={nurse.name}
          patients={combined}
          bedsAvailable={bedsAvailable}
          sortedPatients={sorted}
          loading={loading}
          refresh={fetchMyEncounters}
          handleLogout={handleLogout}
          userRole={userRole}
          token={localStorage.getItem("token")}
          user={nurse}
          getWaitTime={(p) => {
            if (!p?.createdAt) return "N/A";
            const diff = Math.floor((Date.now() - new Date(p.createdAt)) / 60000);
            return diff < 60 ? `${diff}m` : `${Math.floor(diff / 60)}h`;
          }}
        />
      );
    } else if (isSpecialistRole(userRole, userSpecialty)) {
      const dashboardRoute = getDashboardRoute(userRole, userSpecialty);

      if (dashboardRoute === "PHARMACY") {
        return <PharmacistDashboard />;
      }

      if (dashboardRoute === "RADIOLOGY") {
        return <RadiologyDashboard />;
      }

      if (dashboardRoute === "PATHOLOGY") {
        return <PathologyDashboard />;
      }
    }

    // Fallback: render a visible dashboard shell so the user is never left with a blank page.
    return (
      <div style={{ padding: 24, fontFamily: "sans-serif" }}>
        <h2>Dashboard ready</h2>
        <p>Role received: {userRole || "unknown"}</p>
        <p>Normalized role: {normalizedRole || "unknown"}</p>
        <p>Please sign in with a supported role or refresh the page.</p>
      </div>
    );
  }

  return null;
};

/**
 * Admin/Manager Dashboard Component
 * Handles staff management, shift creation, and approval workflows
 */
const AdminManagerDashboard = ({ username, userRole, userEmail, handleLogout, apiFetch, BACKEND_URL }) => {
  const [activeTab, setActiveTab] = useState("overview");
  const [staffList, setStaffList] = useState([]);
  const [staffFormOpen, setStaffFormOpen] = useState(false);
  const [staffForm, setStaffForm] = useState({
    name: "",
    surname: "",
    email: "",
    password: "",
    role: "NURSE",
    department: "ED",
    specialty: ""
  });
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const token = localStorage.getItem("token");

  const API_URL = `${BACKEND_URL}/api`;

  useEffect(() => {
    loadStaff();
  }, []);

  const loadStaff = async () => {
    try {
      const res = await fetch(`${API_URL}/manager/staff/list`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.success) {
        setStaffList(data.staff || []);
      }
    } catch (err) {
      console.error("Error loading staff:", err);
    }
  };

  const handleAddStaff = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      const res = await fetch(`${API_URL}/manager/staff/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify(staffForm)
      });

      const data = await res.json();

      if (data.success) {
        setMessage(`✓ ${staffForm.role} added successfully!`);
        setStaffForm({
          name: "",
          surname: "",
          email: "",
          password: "",
          role: "NURSE",
          department: "ED",
          specialty: ""
        });
        loadStaff();
        setTimeout(() => setStaffFormOpen(false), 1000);
      } else {
        setMessage(`✗ ${data.message}`);
      }
    } catch (err) {
      setMessage(`✗ Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ backgroundColor: "#f8fafc", minHeight: "100vh" }}>
      {/* Top Bar */}
      <div style={{
        backgroundColor: "white",
        borderBottom: "1px solid #e2e8f0",
        padding: "16px 24px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center"
      }}>
        <div>
          <h1 style={{ margin: "0 0 4px 0", fontSize: "20px", fontWeight: "700", color: "#0f172a" }}>
            Hospital Management
          </h1>
          <p style={{ margin: 0, fontSize: "13px", color: "#64748b" }}>
            {username} ({userRole})
          </p>
        </div>
        <button
          onClick={handleLogout}
          style={{
            padding: "8px 16px",
            backgroundColor: "#ef4444",
            color: "white",
            border: "none",
            borderRadius: "6px",
            fontWeight: "600",
            cursor: "pointer",
            fontSize: "14px"
          }}
        >
          Logout
        </button>
      </div>

      {/* Tab Navigation */}
      <div style={{
        backgroundColor: "white",
        borderBottom: "2px solid #e2e8f0",
        padding: "0 24px",
        display: "flex",
        gap: "24px"
      }}>
        {[
          { id: "overview", label: "Overview", icon: "🏥" },
          { id: "staff", label: "Staff Management", icon: "👥" },
          { id: "shifts", label: "Shifts", icon: "📅" },
          { id: "approvals", label: "Approvals", icon: "✓" }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "16px 0",
              border: "none",
              borderBottom: activeTab === tab.id ? "3px solid #10b981" : "none",
              backgroundColor: "transparent",
              cursor: "pointer",
              fontWeight: activeTab === tab.id ? "700" : "600",
              color: activeTab === tab.id ? "#10b981" : "#64748b",
              fontSize: "14px",
              transition: "all 0.2s"
            }}
          >
            <span style={{ marginRight: "6px" }}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "24px 32px" }}>
        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div>
            <h2 style={{ margin: "0 0 24px 0", fontSize: "24px", fontWeight: "700" }}>
              Setup Overview
            </h2>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
              gap: "16px",
              marginBottom: "32px"
            }}>
              <div style={{
                padding: "20px",
                backgroundColor: "white",
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                textAlign: "center"
              }}>
                <h3 style={{ margin: "0 0 8px 0", fontSize: "32px", fontWeight: "700", color: "#10b981" }}>
                  {staffList.length}
                </h3>
                <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>Staff Members</p>
              </div>
              <div style={{
                padding: "20px",
                backgroundColor: "white",
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                textAlign: "center"
              }}>
                <h3 style={{ margin: "0 0 8px 0", fontSize: "32px", fontWeight: "700", color: "#0284c7" }}>
                  0
                </h3>
                <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>Active Shifts</p>
              </div>
              <div style={{
                padding: "20px",
                backgroundColor: "white",
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                textAlign: "center"
              }}>
                <h3 style={{ margin: "0 0 8px 0", fontSize: "32px", fontWeight: "700", color: "#f59e0b" }}>
                  0
                </h3>
                <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>Pending Approvals</p>
              </div>
            </div>
          </div>
        )}

        {/* Staff Management Tab */}
        {activeTab === "staff" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
              <h2 style={{ margin: 0, fontSize: "24px", fontWeight: "700" }}>Staff Management</h2>
              <button
                onClick={() => setStaffFormOpen(!staffFormOpen)}
                style={{
                  padding: "10px 16px",
                  backgroundColor: "#10b981",
                  color: "white",
                  border: "none",
                  borderRadius: "6px",
                  fontWeight: "600",
                  cursor: "pointer",
                  fontSize: "14px"
                }}
              >
                {staffFormOpen ? "✕ Cancel" : "+ Add Staff"}
              </button>
            </div>

            {staffFormOpen && (
              <div style={{
                padding: "24px",
                backgroundColor: "white",
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                marginBottom: "24px"
              }}>
                <h3 style={{ margin: "0 0 20px 0", fontSize: "16px", fontWeight: "700" }}>Register New Staff</h3>

                {message && (
                  <div style={{
                    padding: "12px",
                    backgroundColor: message.includes("✓") ? "#dcfce7" : "#fee2e2",
                    color: message.includes("✓") ? "#166534" : "#991b1b",
                    borderRadius: "6px",
                    marginBottom: "16px",
                    fontSize: "14px"
                  }}>
                    {message}
                  </div>
                )}

                <form onSubmit={handleAddStaff} style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "16px"
                }}>
                  <input
                    type="text"
                    placeholder="First Name"
                    value={staffForm.name}
                    onChange={(e) => setStaffForm({ ...staffForm, name: e.target.value })}
                    required
                    style={{
                      padding: "8px 12px",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "14px"
                    }}
                  />
                  <input
                    type="text"
                    placeholder="Last Name"
                    value={staffForm.surname}
                    onChange={(e) => setStaffForm({ ...staffForm, surname: e.target.value })}
                    required
                    style={{
                      padding: "8px 12px",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "14px"
                    }}
                  />
                  <input
                    type="email"
                    placeholder="Email"
                    value={staffForm.email}
                    onChange={(e) => setStaffForm({ ...staffForm, email: e.target.value })}
                    required
                    style={{
                      padding: "8px 12px",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "14px"
                    }}
                  />
                  <input
                    type="password"
                    placeholder="Password"
                    value={staffForm.password}
                    onChange={(e) => setStaffForm({ ...staffForm, password: e.target.value })}
                    required
                    style={{
                      padding: "8px 12px",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "14px"
                    }}
                  />
                  <select
                    value={staffForm.role}
                    onChange={(e) => setStaffForm({ ...staffForm, role: e.target.value })}
                    style={{
                      padding: "8px 12px",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "14px"
                    }}
                  >
                    <option value="NURSE">Nurse</option>
                    <option value="PHYSICIAN">Physician</option>
                  </select>
                  <select
                    value={staffForm.department}
                    onChange={(e) => setStaffForm({ ...staffForm, department: e.target.value })}
                    style={{
                      padding: "8px 12px",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "14px"
                    }}
                  >
                    <option value="ED">Emergency Department</option>
                    <option value="ICU">ICU</option>
                    <option value="OR">Operating Room</option>
                    <option value="OB">Obstetrics</option>
                    <option value="Medical-Surgical">Medical-Surgical</option>
                    <option value="Pediatrics">Pediatrics</option>
                    <option value="Psychiatry">Psychiatry</option>
                  </select>
                  <input
                    type="text"
                    placeholder="Specialty (optional)"
                    value={staffForm.specialty}
                    onChange={(e) => setStaffForm({ ...staffForm, specialty: e.target.value })}
                    style={{
                      padding: "8px 12px",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "14px"
                    }}
                  />
                  <button
                    type="submit"
                    disabled={loading}
                    style={{
                      gridColumn: "1 / -1",
                      padding: "10px",
                      backgroundColor: loading ? "#cbd5e1" : "#0f172a",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      fontWeight: "600",
                      cursor: loading ? "default" : "pointer",
                      fontSize: "14px"
                    }}
                  >
                    {loading ? "Adding..." : "Add Staff Member"}
                  </button>
                </form>
              </div>
            )}

            {/* Staff List Table */}
            <div style={{
              backgroundColor: "white",
              border: "1px solid #e2e8f0",
              borderRadius: "8px",
              overflow: "hidden"
            }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ backgroundColor: "#f8fafc", borderBottom: "2px solid #e2e8f0" }}>
                    <th style={{ padding: "12px", textAlign: "left", fontSize: "12px", fontWeight: "700", color: "#64748b", textTransform: "uppercase" }}>
                      Name
                    </th>
                    <th style={{ padding: "12px", textAlign: "left", fontSize: "12px", fontWeight: "700", color: "#64748b", textTransform: "uppercase" }}>
                      Role
                    </th>
                    <th style={{ padding: "12px", textAlign: "left", fontSize: "12px", fontWeight: "700", color: "#64748b", textTransform: "uppercase" }}>
                      Department
                    </th>
                    <th style={{ padding: "12px", textAlign: "left", fontSize: "12px", fontWeight: "700", color: "#64748b", textTransform: "uppercase" }}>
                      Email
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {staffList.length === 0 ? (
                    <tr>
                      <td colSpan="4" style={{ padding: "24px", textAlign: "center", color: "#94a3b8", fontSize: "14px" }}>
                        No staff members yet. Click "Add Staff" to register your first team member.
                      </td>
                    </tr>
                  ) : (
                    staffList.map((staff, idx) => (
                      <tr key={idx} style={{ borderBottom: "1px solid #e2e8f0" }}>
                        <td style={{ padding: "12px", fontSize: "14px", color: "#0f172a", fontWeight: "500" }}>
                          {staff.name} {staff.surname}
                        </td>
                        <td style={{ padding: "12px", fontSize: "14px", color: "#64748b" }}>
                          {staff.role}
                        </td>
                        <td style={{ padding: "12px", fontSize: "14px", color: "#64748b" }}>
                          {staff.department}
                        </td>
                        <td style={{ padding: "12px", fontSize: "14px", color: "#64748b" }}>
                          {staff.email}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Shifts Tab */}
        {activeTab === "shifts" && (
          <div>
            <h2 style={{ margin: "0 0 24px 0", fontSize: "24px", fontWeight: "700" }}>Shift Management</h2>
            <div style={{
              padding: "24px",
              backgroundColor: "white",
              border: "1px solid #e2e8f0",
              borderRadius: "8px",
              textAlign: "center"
            }}>
              <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
                Shift management features coming soon. Use API endpoint:
              </p>
              <p style={{ margin: "8px 0 0 0", fontSize: "13px", fontFamily: "monospace", fontWeight: "600", color: "#0f172a" }}>
                POST /api/shifts/create
              </p>
            </div>
          </div>
        )}

        {/* Approvals Tab */}
        {activeTab === "approvals" && (
          <div>
            <h2 style={{ margin: "0 0 24px 0", fontSize: "24px", fontWeight: "700" }}>Pending Approvals</h2>
            <div style={{
              padding: "24px",
              backgroundColor: "white",
              border: "1px solid #e2e8f0",
              borderRadius: "8px",
              textAlign: "center"
            }}>
              <p style={{ margin: 0, color: "#94a3b8", fontSize: "14px" }}>
                No pending approvals at this time.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;