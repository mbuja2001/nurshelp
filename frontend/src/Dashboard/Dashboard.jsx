import React, { useEffect, useState, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import NurseDashboard from "./NurseDashboard";
import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";
import { styles } from "./styles";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * Decode JWT safely
 */
function decodeJwt(token) {
  try {
    const payload = token.split(".")[1];
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(base64);
    return JSON.parse(decodeURIComponent(escape(json)));
  } catch {
    return null;
  }
}

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
        name: decoded.name || "Staff"
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

    fetchWaiting();
    fetchMyEncounters();

    pollRef.current = setInterval(() => {
      fetchWaiting();
    }, POLL_INTERVAL);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [nurse, fetchWaiting, fetchMyEncounters]);

  // -----------------------------
  // NAVIGATION STATE (NEW PATIENT)
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

  // -----------------------------
  // AUTH
  // -----------------------------
  const handleLogin = async ({ email, password }) => {
    try {
      const data = await apiFetch("/api/nurses/login", {
        method: "POST",
        body: JSON.stringify({ email, password })
      });

      if (data.token) {
        localStorage.setItem("token", data.token);

        const decoded = decodeJwt(data.token);

        setNurse({
          id: decoded.id,
          email: decoded.email,
          name: decoded.name || "Staff"
        });

        setView("dashboard");

        await fetchMyEncounters();
        await fetchWaiting();
      }
    } catch (err) {
      alert(err.message);
    }
  };

  const handleRegister = async (payload) => {
    try {
      await apiFetch("/api/nurses/register", {
        method: "POST",
        body: JSON.stringify(payload)
      });

      await handleLogin({
        email: payload.email,
        password: payload.password
      });
    } catch (err) {
      alert(err.message);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
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
          <h1 style={{ fontSize: "3rem", marginBottom: "40px" }}>
            Welcome to Nurse Helpdesk
          </h1>

          <div style={{ display: "flex", gap: "20px", justifyContent: "center" }}>
            <button onClick={() => setView("login")} style={styles.btnWhite}>
              Nurse Login
            </button>

            <button onClick={() => setView("register")} style={styles.btnOutline}>
              Register
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
    return (
      <NurseDashboard
        navigate={navigate}
        setView={setView}
        username={nurse.name}
        patients={combined}
        bedsAvailable={bedsAvailable}
        sortedPatients={sorted}
        loading={loading}
        refresh={fetchMyEncounters} // 🔥 future-proof hook
        handleLogout={handleLogout}
        getWaitTime={(p) => {
          if (!p?.createdAt) return "N/A";
          const diff = Math.floor((Date.now() - new Date(p.createdAt)) / 60000);
          return diff < 60 ? `${diff}m` : `${Math.floor(diff / 60)}h`;
        }}
      />
    );
  }

  return null;
};

export default Dashboard;