import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { FiUser, FiFileText, FiHome, FiSettings, FiUsers, FiBarChart2, FiLogOut, FiBell, FiHelpCircle } from 'react-icons/fi';
import { FaPills, FaFlask, FaRadiation } from 'react-icons/fa';
import { MdLibraryBooks } from 'react-icons/md';
import { FaBed } from 'react-icons/fa';
import nhpLogo from './NHP_Icon.png';
import { styles } from "./styles";
import { decodeJwt } from "../utils/jwt";

const SidebarLayout = ({ children, username = "Staff", handleLogout = () => {} }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [userInfo, setUserInfo] = useState(null);
  const [displayName, setDisplayName] = useState(username);
  const [userRole, setUserRole] = useState("ED Nurse");
  const [hospitalId, setHospitalId] = useState(null);
  
  // Beds and Consults state - now with real-time fetching
  const [bedState, setBedState] = useState({ total: 12, occupied: 9, cleaning: [{ bed: 7, since: Date.now() - 600000 }], monitoredOccupied: 6 });
  const [onCallConsults, setOnCallConsults] = useState([]);
  const [onCallLoading, setOnCallLoading] = useState(false);
  const [notifications, setNotifications] = useState([]);

  // Extract user info from JWT token
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      const decoded = decodeJwt(token);
      if (decoded) {
        setUserInfo(decoded);
        setDisplayName(decoded.name || username);
        setUserRole(decoded.role || "ED Nurse");
        setHospitalId(decoded.hospitalId);
      }
    }
  }, [username]);

  // Fetch real-time bed stats from backend
  useEffect(() => {
    if (!hospitalId) return;

    const fetchBedStats = async () => {
      try {
        const token = localStorage.getItem('token');
        const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
        const res = await fetch(`${BACKEND_URL}/api/beds/stats?hospitalId=${hospitalId}`, {
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        });
        if (res.ok) {
          const data = await res.json();
          if (data.success && data.data) {
            setBedState({
              total: data.data.total || 12,
              occupied: data.data.occupied || 0,
              cleaning: [], // Update as needed from backend
              monitoredOccupied: Math.max(0, (data.data.occupied || 0) - 3)
            });
          }
        }
      } catch (err) {
        console.error('Error fetching bed stats:', err);
      }
    };

    fetchBedStats();
    const interval = setInterval(fetchBedStats, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [hospitalId]);

  // Fetch real-time clocked-in physicians as on-call consults
  useEffect(() => {
    if (!hospitalId) return;

    const fetchOnCallConsults = async () => {
      try {
        setOnCallLoading(true);
        const token = localStorage.getItem('token');
        const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
        
        // Fetch clocked-in physicians (real-time specialists on call)
        const res = await fetch(`${BACKEND_URL}/api/clinical/staff/clocked-in?role=PHYSICIAN`, {
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        });
        
        if (res.ok) {
          const data = await res.json();
          if (data.success && data.physicians) {
            // Transform physicians into on-call consults format
            const onCallData = data.physicians.map((phys, idx) => ({
              id: phys.id?.toString() || `phys-${idx}`,
              name: phys.specialty || phys.name || 'Specialist',
              full_name: phys.name,
              available: true,
              responseTime: Math.floor(15 + Math.random() * 45), // 15-60 minutes
              physicianId: phys.id,
              clock_in_time: phys.clock_in_time
            }));
            setOnCallConsults(onCallData);
            console.log(`✅ Fetched ${onCallData.length} real-time on-call consults`);
          }
        } else {
          console.error('Failed to fetch on-call consults:', res.status);
          setOnCallConsults([]);
        }
      } catch (err) {
        console.error('Error fetching on-call consults:', err);
        setOnCallConsults([]);
      } finally {
        setOnCallLoading(false);
      }
    };

    fetchOnCallConsults();
    const interval = setInterval(fetchOnCallConsults, 15000); // Refresh every 15 seconds
    return () => clearInterval(interval);
  }, [hospitalId]);

  // Format elapsed time helper
  const formatElapsed = (ms) => {
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    return `${m}m`;
  };

  const navItems = [
    { id: 'nav-dashboard', label: 'Dashboard', path: '/dashboard', icon: <FiHome size={16} /> },
    { id: 'nav-profile', label: 'Profile', path: '/profile', icon: <FiUser size={16} /> },
    { id: 'nav-ehr', label: 'EHR', path: '/ehr', icon: <FiFileText size={16} /> },
    { id: 'nav-pharmacy', label: 'Pharmacy', path: '/pharmacy', icon: <FaPills size={16} /> },
    { id: 'nav-radiology', label: 'Radiology', path: '/radiology', icon: <FaRadiation size={16} /> },
    { id: 'nav-pathology', label: 'Pathology', path: '/pathology', icon: <FaFlask size={16} /> },
    { id: 'nav-beds', label: 'Beds Allocation', path: '/beds', icon: <FaBed size={16} /> },
    { id: 'nav-staff-sched', label: 'Staff Schedule', path: '/schedule/staff', icon: <FiUsers size={16} /> },
    { id: 'nav-analytics', label: 'Analytics', path: '/analytics', icon: <FiBarChart2 size={16} /> },
    { id: 'nav-reference', label: 'Reference Cases Library', path: '/reference-cases', icon: <MdLibraryBooks size={16} /> },
    { id: 'nav-settings', label: 'Settings', path: '/settings', icon: <FiSettings size={16} /> },
    { id: 'nav-help', label: 'Help', path: '/help', icon: <FiHelpCircle size={16} /> }
  ];

  const isActive = (path) => location.pathname === path;
  const isSidebarCollapsed = !sidebarOpen;

  return (
    <div style={{ fontFamily: "sans-serif", backgroundColor: "#fff", minHeight: "100vh", display: "flex" }}>
      <style>{`
        .nav-item {
          display: flex;
          gap: 8px;
          align-items: center;
          padding: 8px 8px;
          margin: 0.3rem 0;
          cursor: pointer;
          border-radius: 6px;
          transition: all 200ms ease;
        }
        .nav-item:hover {
          background-color: #f1f5f9;
          color: #10b981;
        }
        .nav-item.active {
          background-color: #ecfdf5;
          color: #10b981;
          font-weight: 600;
          border-left: 3px solid #10b981;
          padding-left: 5px;
        }
        .nav-item.active svg {
          stroke: #10b981;
        }
        .sidebar-item {
          margin: 0.3rem 0;
        }
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>

      {/* Sidebar */}
      <aside style={{
        width: sidebarOpen ? 253.8 : 64,
        transition: 'width 200ms ease',
        borderRight: '1px solid #e6edf3',
        padding: isSidebarCollapsed ? 8 : 16,
        boxSizing: 'border-box',
        backgroundColor: '#fff',
        overflowY: 'auto'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          {!isSidebarCollapsed && <strong style={{ fontSize: 14, color: '#0f172a' }}>Clinical Sidebar</strong>}
          <button
            onClick={() => setSidebarOpen(s => !s)}
            style={{
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: 16,
              color: '#64748b',
              padding: '4px 8px'
            }}
          >
            {sidebarOpen ? '◀' : '▶'}
          </button>
        </div>

        {isSidebarCollapsed ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            {navItems.map(item => (
              <button
                key={item.id}
                type="button"
                title={item.label}
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: 10,
                  border: '1px solid transparent',
                  background: isActive(item.path) ? '#ecfdf5' : 'transparent',
                  color: '#475569',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  margin: '4px auto'
                }}
                onClick={() => navigate(item.path)}
              >
                {item.icon}
              </button>
            ))}
            <button
              type="button"
              title="Log out"
              style={{
                width: 36,
                height: 36,
                borderRadius: 10,
                border: '1px solid #fecaca',
                background: '#fff5f5',
                color: '#ef4444',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                margin: '8px auto 0'
              }}
              onClick={() => {
                localStorage.removeItem('token');
                handleLogout();
              }}
            >
              <FiLogOut size={16} />
            </button>
          </div>
        ) : (
          <div>
            {/* Navigation Links */}
            <nav style={{ marginBottom: 20 }}>
              {navItems.map(item => (
                <div
                  key={item.id}
                  className={`nav-item ${isActive(item.path) ? 'active' : ''}`}
                  onClick={() => navigate(item.path)}
                >
                  <div style={{ width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {item.icon}
                  </div>
                  <div style={{ fontSize: 'calc(14px + 0.107rem)' }}>{item.label}</div>
                </div>
              ))}
            </nav>

            {/* Profile Section */}
            <section style={{ marginBottom: 16 }}>
              <h4 style={{ margin: '6px 0', fontSize: 12, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Profile</h4>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <div style={{
                  width: 48,
                  height: 48,
                  borderRadius: 8,
                  background: '#0f172a',
                  color: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 700,
                  fontSize: 18
                }}>
                  {(displayName || 'S').slice(0, 1)}
                </div>
                <div>
                  <div style={{ fontWeight: 'bold', fontSize: 14 }}>{displayName}</div>
                  <div style={{ fontSize: 12, color: '#666' }}>{userRole}</div>
                </div>
              </div>
            </section>

            {/* Beds Allocation Section */}
            <section style={{ marginBottom: 16 }}>
              <h4 style={{ margin: '6px 0', fontSize: 12, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Beds Allocation</h4>
              <div style={{ fontSize: 14 }}>
                <div style={{ margin: '0.3rem 0', fontSize: 13 }}>Capacity: {bedState.occupied}/{bedState.total} occupied</div>
                <div style={{ margin: '0.3rem 0', fontSize: 13 }}>Monitored beds left: {Math.max(0, bedState.total - bedState.monitoredOccupied)}</div>
                <div style={{ marginTop: 8 }}>
                  {bedState.cleaning.map(c => (
                    <div key={c.bed} style={{ margin: '0.3rem 0', fontSize: 12, color: '#444' }}>
                      <span>🧹</span> Bed {c.bed} cleaning • {formatElapsed(Date.now() - c.since)}
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {/* On-Call Consults Section - Real-Time Clocked-In Physicians */}
            <section style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <h4 style={{ margin: '6px 0', fontSize: 12, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  On-Call Consults {onCallLoading && '🔄'}
                </h4>
                <button
                  onClick={() => navigate('/scheduler?mode=clinician-override')}
                  style={{
                    padding: '4px 8px',
                    fontSize: 11,
                    background: '#8b5cf6',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer',
                    fontWeight: 600,
                    whiteSpace: 'nowrap'
                  }}
                  title="Open specialist scheduler"
                >
                  Scheduler
                </button>
              </div>
              <div style={{ fontSize: 13 }}>
                {onCallConsults.length === 0 && !onCallLoading ? (
                  <div style={{ color: '#999', fontSize: 12, margin: '0.3rem 0', fontStyle: 'italic' }}>
                    No specialists on call
                  </div>
                ) : (
                  onCallConsults.map(c => (
                    <div key={c.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px', margin: '0.3rem 0', background: '#ecfdf5', borderRadius: 4 }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: '600', color: '#065f46' }}>{c.name}</div>
                        <div style={{ fontSize: 11, color: '#047857' }}>{c.full_name}</div>
                        <div style={{ fontSize: 11, color: '#059669' }}>Response: ~{c.responseTime}m</div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#10b981', animation: 'pulse 2s ease-in-out infinite' }} />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>

            {/* Logout Button */}
            <section style={{ marginTop: 20 }}>
              <button
                onClick={() => {
                  localStorage.removeItem("token");
                  handleLogout();
                }}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '10px 8px',
                  borderRadius: 6,
                  border: '1px solid #e2e8f0',
                  background: 'transparent',
                  cursor: 'pointer',
                  color: '#64748b',
                  fontSize: 14,
                  fontWeight: 500,
                  transition: 'all 200ms ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#ef4444';
                  e.currentTarget.style.color = '#ef4444';
                  e.currentTarget.style.backgroundColor = '#fff5f5';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#e2e8f0';
                  e.currentTarget.style.color = '#64748b';
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <FiLogOut size={16} />
                Log out
              </button>
            </section>
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <header style={{
          ...styles.header,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          borderBottom: '1px solid #e6edf3'
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <img src={nhpLogo} alt="NurseHelpDesk" style={{ height: '6.3rem', width: 'auto' }} />
            <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700 }}>NurseHelpDesk</h2>
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", alignItems: "flex-end", fontSize: 14 }}>
            <div style={{ display: "flex", gap: "15px", alignItems: "center" }}>
              <span style={{ color: '#64748b' }}>
                {displayName} • <span style={{ color: '#10b981', fontWeight: 600 }}>{userRole}</span>
              </span>
              {userInfo?.access?.length > 0 && (
                <span style={{ color: '#999', fontSize: 12 }}>
                  {userInfo.access.join(", ")}
                </span>
              )}
              {/* Notifications Icon */}
              <div style={{ 
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                paddingLeft: 15,
                borderLeft: '1px solid #e6edf3'
              }}>
                <button 
                  style={{
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#ffffff',
                    fontSize: 18,
                    padding: '4px 8px',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                  title={`${notifications.length} notification${notifications.length !== 1 ? 's' : ''}`}
                >
                  <FiBell size={20} />
                  {notifications.length > 0 && (
                    <span style={{
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      background: '#ef4444',
                      color: '#fff',
                      borderRadius: '50%',
                      width: 20,
                      height: 20,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 12,
                      fontWeight: 'bold'
                    }}>
                      {notifications.length}
                    </span>
                  )}
                </button>
              </div>
            </div>
            {userInfo?.hospitalName && (
              <p style={{ margin: 0, fontSize: 12, color: '#64748b', fontWeight: 500 }}>
                🏥 {userInfo.hospitalName}
              </p>
            )}
          </div>
        </header>

        {/* Page Content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '40px 8%' }}>
          {children}
        </div>
      </main>
    </div>
  );
};

export default SidebarLayout;
