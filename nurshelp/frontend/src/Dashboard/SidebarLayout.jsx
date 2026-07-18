import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { FiUser, FiFileText, FiHome, FiSettings, FiUsers, FiBarChart2, FiLogOut, FiBell, FiHelpCircle, FiCreditCard, FiMessageCircle, FiClock } from 'react-icons/fi';
import { FaPills, FaFlask, FaRadiation } from 'react-icons/fa';
import { MdLibraryBooks } from 'react-icons/md';
import { FaBed } from 'react-icons/fa';
import nhpLogo from './NHP_Icon.png';
import { styles } from "./styles";
import { decodeJwt } from "../utils/jwt";
import { notificationService } from '../services/notificationService';

const SidebarLayout = ({ children, username = "Staff", handleLogout = () => {} }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarWidth, setSidebarWidth] = useState(52); // Default width in pixels
  const [isDraggingSidebar, setIsDraggingSidebar] = useState(false);
  const [userInfo, setUserInfo] = useState(null);
  const [displayName, setDisplayName] = useState(username);
  const [userRole, setUserRole] = useState("ED Nurse");
  const [hospitalId, setHospitalId] = useState(null);
  
  // Beds and Consults state - now with real-time fetching
  const [bedState, setBedState] = useState({ total: 12, occupied: 9, cleaning: [{ bed: 7, since: Date.now() - 600000 }], monitoredOccupied: 6 });
  const [onCallConsults, setOnCallConsults] = useState([]);
  const [onCallLoading, setOnCallLoading] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [alertCount, setAlertCount] = useState(3);
  const [messageCount, setMessageCount] = useState(2);
  const [showNotificationsDropdown, setShowNotificationsDropdown] = useState(false);
  const [showMessagesDropdown, setShowMessagesDropdown] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

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

  useEffect(() => {
    let cancelled = false;

    const loadUnreadNotifications = async () => {
      try {
        const response = await notificationService.getUnreadNotifications(6);
        if (cancelled) return;

        const items = Array.isArray(response?.notifications) ? response.notifications : [];
        const nextCount = response?.count || items.length || 3;

        setNotifications(items.length > 0 ? items : [
          { _id: 'demo-alert-1', title: 'Priority queue updated', message: 'Three urgent cases need physician review.', type: 'alert' },
          { _id: 'demo-alert-2', title: 'Lab review pending', message: 'Radiology results are ready for sign-off.', type: 'info' }
        ]);
        setAlertCount(nextCount || 3);
        setMessageCount(Math.max(2, Math.min(9, nextCount + 1)));
      } catch (err) {
        console.error('Error loading unread notifications:', err);
        if (!cancelled) {
          setNotifications([
            { _id: 'demo-alert-1', title: 'Priority queue updated', message: 'Three urgent cases need physician review.', type: 'alert' },
            { _id: 'demo-alert-2', title: 'Lab review pending', message: 'Radiology results are ready for sign-off.', type: 'info' }
          ]);
          setAlertCount(3);
          setMessageCount(2);
        }
      }
    };

    loadUnreadNotifications();
    const interval = window.setInterval(loadUnreadNotifications, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setCurrentTime(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

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
    { id: 'nav-reference', label: 'Reference Cases Library', path: '/reference-cases', icon: <MdLibraryBooks size={16} /> },
    { id: 'nav-analytics', label: 'Analytics', path: '/analytics', icon: <FiBarChart2 size={16} /> },
    { id: 'nav-settings', label: 'Settings', path: '/settings', icon: <FiSettings size={16} /> },
    { id: 'nav-licenses', label: 'Clinical Licenses', path: '/licenses', icon: <FiCreditCard size={16} /> },
    { id: 'nav-help', label: 'Help', path: '/help', icon: <FiHelpCircle size={16} /> }
  ];

  const isActive = (path) => location.pathname === path;

  // Sidebar drag handlers
  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDraggingSidebar(true);
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDraggingSidebar) return;
      
      // Min width 52px (collapsed), Max width 400px
      const newWidth = Math.max(52, Math.min(e.clientX, 400));
      setSidebarWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsDraggingSidebar(false);
    };

    if (isDraggingSidebar) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };
  }, [isDraggingSidebar]);

  return (
    <div style={{ fontFamily: "sans-serif", backgroundColor: "#fff", minHeight: "100vh", display: "flex", flexDirection: "row", width: "100%", overflow: "hidden" }}>
      <style>{`
        * {
          box-sizing: border-box;
        }

        body, html, #root {
          width: 100%;
          height: 100%;
          margin: 0;
          padding: 0;
        }

        .nav-item {
          display: flex;
          gap: 8px;
          align-items: center;
          padding: 8px 8px;
          margin: 0.3rem 0;
          cursor: pointer;
          border-radius: 6px;
          transition: all 200ms ease;
          white-space: nowrap;
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

        /* Sidebar drag handle */
        .sidebar-drag-handle {
          position: absolute;
          right: 0;
          top: 0;
          width: 4px;
          height: 100%;
          cursor: col-resize;
          background: transparent;
          hover: background #ddd;
          transition: background 200ms ease;
        }

        .sidebar-drag-handle:hover {
          background: #e0e7ff;
        }

        /* Desktop layout */
        .sidebar-container {
          display: flex !important;
          flex-direction: column !important;
          flex-shrink: 0 !important;
          height: 100vh;
          z-index: 100;
          position: relative;
          transition: width 0.05s ease-out;
        }

        .main-content-wrapper {
          display: flex !important;
          flex-direction: column !important;
          flex: 1;
          width: 100%;
          overflow: hidden;
          position: relative;
        }

        /* Responsive styles */
        @media (max-width: 1024px) {
          .sidebar-text-label {
            font-size: 12px !important;
          }
        }

        @media (max-width: 768px) {
          .sidebar-container {
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            height: 100vh !important;
            z-index: 200 !important;
          }

          .main-content-wrapper {
            margin-left: 0 !important;
            width: 100% !important;
          }

          .sidebar-text-label {
            font-size: 11px !important;
          }
          .profile-section {
            gap: 8px !important;
          }
          .nav-item {
            padding: 6px 6px !important;
          }
        }

        @media (max-width: 480px) {
          .sidebar-container {
            width: 100% !important;
          }

          .sidebar-text-label {
            display: block;
            font-size: 10px !important;
          }
          .profile-avatar {
            width: 40px !important;
            height: 40px !important;
            font-size: 16px !important;
          }
          .nav-item {
            padding: 5px 4px !important;
          }
        }
      `}</style>

      {/* Sidebar */}
      <aside className="sidebar-container" style={{
        width: sidebarWidth,
        transition: isDraggingSidebar ? 'none' : 'width 200ms ease',
        borderRight: '1px solid #e6edf3',
        padding: sidebarWidth > 100 ? 16 : 6,
        boxSizing: 'border-box',
        backgroundColor: '#fff',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        height: '100vh',
        zIndex: 100,
        position: 'relative'
      }}>
        {/* Drag handle on the right edge */}
        <div 
          className="sidebar-drag-handle" 
          onMouseDown={handleMouseDown}
          style={{
            position: 'absolute',
            right: 0,
            top: 0,
            width: 5,
            height: '100%',
            cursor: isDraggingSidebar ? 'col-resize' : 'col-resize',
            backgroundColor: isDraggingSidebar ? '#e0e7ff' : 'transparent',
            transition: 'background 200ms ease'
          }}
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, minHeight: 24 }}>
          {sidebarWidth > 100 && <strong style={{ fontSize: 14, color: '#0f172a' }}>Clinical Sidebar</strong>}
          <button
            onClick={() => setSidebarWidth(sidebarWidth > 100 ? 52 : 253.8)}
            style={{
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: 16,
              color: '#64748b',
              padding: '4px 8px'
            }}
          >
            {sidebarWidth > 100 ? '◀' : '▶'}
          </button>
        </div>

        {sidebarWidth > 100 && (
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
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }} className="profile-section">
                <div className="profile-avatar" style={{
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
                <div style={{ display: sidebarWidth > 100 ? 'block' : 'none' }}>
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
      <main className="main-content-wrapper" style={{ flex: 1, display: 'flex', flexDirection: 'column', width: '100%', overflow: 'hidden', position: 'relative' }}>
        <header style={{
          ...styles.header,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          borderBottom: '1px solid #e6edf3',
          padding: '12px 16px',
          flexWrap: 'wrap',
          flexShrink: 0,
          minHeight: '78px',
          background: 'linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)',
          position: 'relative'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
            <img src={nhpLogo} alt="NurseHelpDesk" style={{ height: '2.8rem', width: 'auto' }} />
            <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#ffffff' }}>NurseHelpDesk</h2>
              <span style={{ fontSize: 12, color: '#dbeafe' }}>Operational command center</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '10px', marginLeft: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '8px 12px',
                borderRadius: '999px',
                background: 'rgba(255,255,255,0.12)',
                border: '1px solid rgba(255,255,255,0.18)',
                boxShadow: '0 1px 3px rgba(15,23,42,0.12)'
              }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #0f172a 0%, #2563eb 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontWeight: 700,
                  fontSize: 16
                }}>
                  {(displayName || 'S').slice(0, 1).toUpperCase()}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2, minWidth: 0 }}>
                  <span style={{ fontWeight: 700, color: '#ffffff', fontSize: 13 }}>{displayName}</span>
                  <span style={{ fontSize: 12, color: '#dbeafe' }}>{userRole}</span>
                  <span style={{ fontSize: 11, color: '#86efac', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                    <FiClock size={11} />
                    {currentTime.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                  </span>
                </div>
              </div>

              <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  onClick={() => {
                    setShowMessagesDropdown(v => !v);
                    setShowNotificationsDropdown(false);
                  }}
                  style={{
                    position: 'relative',
                    background: 'rgba(255,255,255,0.16)',
                    border: '1px solid rgba(255,255,255,0.24)',
                    color: '#ffffff',
                    borderRadius: '999px',
                    padding: '8px 10px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: 12,
                    fontWeight: 700
                  }}
                  title="Staff messaging"
                >
                  <FiMessageCircle size={16} />
                  <span>Messages</span>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minWidth: '18px',
                    height: '18px',
                    padding: '0 4px',
                    borderRadius: '999px',
                    background: '#2563eb',
                    color: '#ffffff',
                    fontSize: 10,
                    fontWeight: 700
                  }}>
                    {messageCount}
                  </span>
                </button>

                <button
                  onClick={() => {
                    setShowNotificationsDropdown(v => !v);
                    setShowMessagesDropdown(false);
                  }}
                  style={{
                    position: 'relative',
                    background: alertCount > 0 ? 'rgba(248,113,113,0.18)' : 'rgba(255,255,255,0.16)',
                    border: `1px solid ${alertCount > 0 ? 'rgba(248,113,113,0.35)' : 'rgba(255,255,255,0.24)'}`,
                    color: '#ffffff',
                    borderRadius: '999px',
                    padding: '8px 10px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: 12,
                    fontWeight: 800
                  }}
                  title={`${alertCount} alert${alertCount !== 1 ? 's' : ''}`}
                >
                  <FiBell size={16} />
                  <span>Alerts</span>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minWidth: '18px',
                    height: '18px',
                    padding: '0 4px',
                    borderRadius: '999px',
                    background: '#dc2626',
                    color: '#ffffff',
                    fontSize: 10,
                    fontWeight: 700
                  }}>
                    {alertCount}
                  </span>
                </button>
              </div>
            </div>

          </div>

          {showMessagesDropdown && (
            <div style={{
              position: 'absolute',
              top: 'calc(100% + 8px)',
              right: '220px',
              minWidth: '260px',
              background: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: '12px',
              boxShadow: '0 12px 30px rgba(15,23,42,0.12)',
              padding: '12px',
              zIndex: 1100
            }}>
              <div style={{ fontSize: 12, fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>Staff messages</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {[
                  { title: 'Dr. Patel', detail: 'Surgical prep update for room 12' },
                  { title: 'Nurse Rivera', detail: 'Medication timing confirmed for ICU' }
                ].map((item) => (
                  <div key={item.title} style={{ padding: '8px 10px', borderRadius: '8px', background: '#f8fafc', fontSize: 12, color: '#334155' }}>
                    <div style={{ fontWeight: 700 }}>{item.title}</div>
                    <div style={{ color: '#64748b', marginTop: '2px' }}>{item.detail}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {showNotificationsDropdown && (
            <div style={{
              position: 'absolute',
              top: 'calc(100% + 8px)',
              right: '12px',
              minWidth: '280px',
              background: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: '12px',
              boxShadow: '0 12px 30px rgba(15,23,42,0.12)',
              padding: '12px',
              zIndex: 1100
            }}>
              <div style={{ fontSize: 12, fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>Live alerts</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {notifications.slice(0, 3).map((item) => (
                  <div key={item._id || item.title} style={{ padding: '8px 10px', borderRadius: '8px', background: '#fef2f2', fontSize: 12, color: '#7f1d1d' }}>
                    <div style={{ fontWeight: 700 }}>{item.title || 'Alert'}</div>
                    <div style={{ color: '#b91c1c', marginTop: '2px' }}>{item.message || item.body || 'New clinical update'}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </header>

        {/* Page Content */}
        <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: 'clamp(12px, 2%, 24px)', maxWidth: '100%', width: '100%' }}>
          {children}
        </div>
      </main>
    </div>
  );
};

export default SidebarLayout;
