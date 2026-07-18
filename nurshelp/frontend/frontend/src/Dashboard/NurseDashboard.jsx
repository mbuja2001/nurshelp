import React, { useEffect, useState, useRef } from "react";
import { useLocation } from "react-router-dom";
import { severityFromReport, waitStartFor, computePriority, formatWaitDetailed, normalizePatient, formatTimestamp } from "../utils/priority";
import { styles } from "./styles";
import { FiUser, FiFileText, FiCheckSquare, FiHome, FiSettings, FiUsers, FiBarChart2 } from 'react-icons/fi';
import { MdLibraryBooks } from 'react-icons/md';
import { FaBed } from 'react-icons/fa';
import nhpLogo from './NHP_Icon.png';
import SidebarLayout from './SidebarLayout';
import { useUserFromToken } from '../hooks/useUserFromToken';
import ClinicalNurseDashboard from '../components/dashboards/ClinicalNurseDashboard';
import PhysicianDashboard from '../components/dashboards/PhysicianDashboard';
import OffDutyDashboard from '../components/dashboards/OffDutyDashboard';
import { decodeJwt } from '../utils/jwt';
import './DashboardTheme.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const NurseDashboard = ({
  navigate = () => {},
  setView = () => {},
  username = "Staff",
  patients = [],
  bedsAvailable = 0,
  sortedPatients = [],
  toggleAssigned = () => {},
  getWaitTime = () => "N/A",
  handleLogout = () => {},
  userRole = null
}) => {

  useEffect(() => {
    if (!username) {
      setView("login");
      navigate("/login");
    }
  }, [username, navigate, setView]);

  const handleNewIntake = () => navigate("/summary/new");

  const location = useLocation();

  // Sidebar / CDS state
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Mock/placeholder specialists + consults state
  const [specialists] = useState([
    { id: 'neuro', name: 'Neurology', onCall: true, ertMinutes: 10 },
    { id: 'card', name: 'Cardiology', onCall: true, ertMinutes: 30 },
    { id: 'ortho', name: 'Orthopedics', onCall: false, ertMinutes: 60 }
  ]);

  const [consults, setConsults] = useState([
    // sample active consult: { id, specialtyId, pagedAt }
  ]);

  // beds / EVS sample state
  const [bedState] = useState({ total: bedsAvailable || 12, occupied: 9, cleaning: [{ bed: 7, since: Date.now() - 600000 }] , monitoredOccupied: 6 });

  // real-time clock tick to drive high-resolution wait timers (centiseconds)
  const [now, setNow] = useState(Date.now());
  const alarmRef = useRef(null);
  const [showAlertsPanel, setShowAlertsPanel] = useState(true);
  const [showEmergencyBanner, setShowEmergencyBanner] = useState(true);
  const prevUrgentIdsRef = useRef([]);

  const [confirmLoading, setConfirmLoading] = useState({});
  const [attendLoading, setAttendLoading] = useState({});
  const [currentRecord, setCurrentRecord] = useState(null);
  const [recordLoading, setRecordLoading] = useState(true);

  // Track assigned role after clock-in
  const [assignedRole, setAssignedRole] = useState(null);
  const [assignedContext, setAssignedContext] = useState(null);
  const [roleFetchLoading, setRoleFetchLoading] = useState(false);

  // Workflow-aware encounter state (newly initiated handovers)
  const [handedOverEncounters, setHandedOverEncounters] = useState([]);
  const [handoverLoading, setHandoverLoading] = useState(false);

  // Priority Queue assignment notification modal
  const [priorityQueueNotification, setPriorityQueueNotification] = useState(null);
  const [showPriorityNotificationModal, setShowPriorityNotificationModal] = useState(false);

  const markPatientAsAttended = async (p) => {
    const id = p._id || p.id;
    if (!id) return alert('Missing encounter id');
    setAttendLoading(prev => ({ ...prev, [id]: true }));
    try {
      const token = localStorage.getItem('token');
      const body = {
        encounter: id,
        specialistAttended: true,
        specialistAttendedAt: new Date().toISOString(),
        patient: p.patient || {},
      };

      const res = await fetch(`/api/encounters/${id}/mark-attended`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify(body)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || 'Update failed');
      // Update the patient object in the list to mark as attended
      // This will trigger re-render and move patient to attended table
      alert('Patient marked as attended');
    } catch (e) {
      console.error('Mark as attended error', e);
      alert('Failed to mark patient as attended: ' + (e.message || e));
    } finally {
      setAttendLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const createNurseConfirmed = async (p) => {
    const id = p._id || p.id;
    if (!id) return alert('Missing encounter id');
    setConfirmLoading(prev => ({ ...prev, [id]: true }));
    try {
      const token = localStorage.getItem('token');
      const body = {
        encounter: id,
        patient: p.patient || {},
        transcript: p.transcript || [],
        vitals: p.vitals || {},
        triage: p.triage || {},
        recommendations: p.recommendations || [],
        nurseNotes: p.nurseNotes || "",
        severity: p.severity ?? 1,
        isWaiting: typeof p.isWaiting !== 'undefined' ? p.isWaiting : true,
        attendedAt: p.attendedAt || p.specialistAttendedAt || null
      };

      const res = await fetch('/api/reports/nurse-confirmed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify(body)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || 'Create failed');
      // simple success feedback
      alert('Nurse-confirmed report saved');
    } catch (e) {
      console.error('Create nurse-confirmed error', e);
      alert('Failed to save nurse-confirmed report: ' + (e.message || e));
    } finally {
      setConfirmLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const pageSpecialist = (specId) => {
    const paged = { id: `${specId}-${Date.now()}`, specialtyId: specId, pagedAt: Date.now() };
    setConsults(prev => [paged, ...prev]);
  };

  const { userName } = useUserFromToken();

  // Fetch today's clock record from server to enforce triage permissions
  const fetchTodayRecord = async () => {
    const token = localStorage.getItem('token');
    if (!token) {
      setCurrentRecord(null);
      setRecordLoading(false);
      return;
    }
    try {
      setRecordLoading(true);
      const res = await fetch(`${BACKEND_URL}/api/staff/today-record`, {
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
      });
      const data = await res.json().catch(() => ({}));
      console.log('[fetchTodayRecord] response', data);
      if (res.ok && data.success) {
        console.log('[fetchTodayRecord] setting currentRecord ->', data.record);
        setCurrentRecord(data.record || null);
      } else setCurrentRecord(null);
    } catch (e) {
      console.error('Failed to fetch today record', e);
      setCurrentRecord(null);
    } finally {
      setRecordLoading(false);
    }
  };

  useEffect(() => {
    fetchTodayRecord();
    const iv = setInterval(fetchTodayRecord, 10000);
    return () => clearInterval(iv);
  }, []);

  // Fetch encounters that have been handed over but not yet seen by physician
  const fetchHandedOverEncounters = async () => {
    const token = localStorage.getItem('token');
    if (!token) {
      setHandedOverEncounters([]);
      return;
    }
    try {
      setHandoverLoading(true);
      console.log('[DASHBOARD_FETCH] Fetching handed over encounters...');
      const res = await fetch(`${BACKEND_URL}/api/encounters?status=AWAITING_PHYSICIAN`, {
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
      });
      
      console.log('[DASHBOARD_FETCH] Response status:', res.status);
      const data = await res.json().catch((e) => {
        console.error('[DASHBOARD_FETCH] Failed to parse response:', e);
        return {};
      });
      
      console.log('[DASHBOARD_FETCH] Response data:', data);
      
      if (res.ok && data.success && Array.isArray(data.encounters)) {
        console.log('[DASHBOARD_FETCH] Found', data.encounters.length, 'encounters awaiting physician');
        setHandedOverEncounters(data.encounters);
      } else {
        console.log('[DASHBOARD_FETCH] ❌ Invalid response format or not ok');
        setHandedOverEncounters([]);
      }
    } catch (e) {
      console.error('[DASHBOARD_FETCH] Error:', e);
      setHandedOverEncounters([]);
    } finally {
      setHandoverLoading(false);
    }
  };

  useEffect(() => {
    // Fetch handed over encounters every 5 seconds
    fetchHandedOverEncounters();
    const iv = setInterval(fetchHandedOverEncounters, 5000);
    return () => clearInterval(iv);
  }, []);

  // Update `now` frequently (every 100ms) to keep timers and priority calculations live
  useEffect(() => {
    const interval = setInterval(() => {
      setNow(Date.now());
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // Fetch assigned role after clock-in
  useEffect(() => {
    if (!currentRecord || currentRecord.status !== 'CLOCKED_IN') {
      setAssignedRole(null);
      setAssignedContext(null);
      return;
    }

    const fetchAssignedRole = async () => {
      setRoleFetchLoading(true);
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setAssignedRole(null);
          return;
        }

        const decoded = decodeJwt(token);
        const userId = decoded?.id;

        const res = await fetch(`${BACKEND_URL}/api/shifts/assigned/today`, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          }
        });

        if (!res.ok) {
          console.warn('[ASSIGNED ROLE] Failed to fetch shift:', res.status);
          setAssignedRole(null);
          return;
        }

        const data = await res.json();
        const shift = data.shift;

        if (!shift) {
          console.log('[ASSIGNED ROLE] No shift found');
          setAssignedRole(null);
          return;
        }

        const userIdStr = userId?.toString();
        const teamAssignment = shift.teamAssignment || {};

        // PRIORITY: Check teamAssignment first (authoritative for dashboard role)
        // This determines which dashboard role to display, NOT the generic assignedStaff type
        let role = null;
        let context = shift.department || 'ER';

        const triageNurseId = teamAssignment.triageNurseId?.toString?.() || teamAssignment.triageNurseId;
        const clinicalNurseId = teamAssignment.clinicalNurseId?.toString?.() || teamAssignment.clinicalNurseId;
        const physicianId = teamAssignment.physicianId?.toString?.() || teamAssignment.physicianId;

        console.log('[ASSIGNED ROLE] Checking teamAssignment...');
        console.log('  userId:', userIdStr);
        console.log('  triageNurseId:', triageNurseId);
        console.log('  clinicalNurseId:', clinicalNurseId);
        console.log('  physicianId:', physicianId);

        if (userIdStr && triageNurseId && userIdStr === triageNurseId.toString()) {
          role = 'TRIAGE_NURSE';
          context = 'TRIAGE';
          console.log('[ASSIGNED ROLE] ✓ User is TRIAGE_NURSE');
        } else if (userIdStr && clinicalNurseId && userIdStr === clinicalNurseId.toString()) {
          role = 'CLINICAL_NURSE';
          context = teamAssignment.clinicalNurseContext || 'ER';
          console.log('[ASSIGNED ROLE] ✓ User is CLINICAL_NURSE');
        } else if (userIdStr && physicianId && userIdStr === physicianId.toString()) {
          role = 'PHYSICIAN';
          context = teamAssignment.physicianContext || 'ER';
          console.log('[ASSIGNED ROLE] ✓ User is PHYSICIAN');
        } else {
          // Fallback to assignedStaff if not in teamAssignment
          console.log('[ASSIGNED ROLE] Not in teamAssignment, checking assignedStaff as fallback...');
          if (shift.assignedStaff && shift.assignedStaff.length > 0) {
            const staffEntry = shift.assignedStaff.find(staff => 
              (staff.staffID && staff.staffID.toString?.() === userIdStr) || staff.staffID === userIdStr || staff.staffID?._id === userId
            );
            if (staffEntry) {
              const roleMap = {
                'NURSE': 'CLINICAL_NURSE',
                'PHYSICIAN': 'PHYSICIAN',
                'SPECIALIST': 'SPECIALIST'
              };
              role = roleMap[staffEntry.type] || staffEntry.type;
              console.log(`[ASSIGNED ROLE] Fallback: User assigned as ${role} from assignedStaff`);
            }
          }
        }

        if (role) {
          setAssignedRole(role);
          setAssignedContext(context);
          console.log(`[ASSIGNED ROLE] Final: ${role} in ${context}`);
        } else {
          console.log('[ASSIGNED ROLE] User not assigned to any role');
          setAssignedRole(null);
        }
      } catch (err) {
        console.error('[ASSIGNED ROLE] Error:', err);
        setAssignedRole(null);
      } finally {
        setRoleFetchLoading(false);
      }
    };

    fetchAssignedRole();
  }, [currentRecord]);

  const triageAllowed = !!(currentRecord && currentRecord.status === 'CLOCKED_IN');

  const formatElapsed = (ms) => {
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    return `${m}m`;
  };

  

  const severityColor = (sev) => {
    if (sev === 1) return "#ef4444";  // RED
    if (sev === 2) return "#f97316";  // ORANGE
    if (sev === 3) return "#eab308";  // YELLOW
    if (sev === 4) return "#22c55e";  // GREEN
    return "#94a3b8";  // Unknown
  };

  // normalize all patients first so attendance fields are consistent
  const normalizedPatients = (sortedPatients || []).map(p => normalizePatient(p));

  const isAttended = (p) => {
    // Treat as attended ONLY when specialist attendance is recorded.
    // Prefer explicit specialist fields to avoid treating 'confirmed' status as attended.
    if (p.specialistAttended === true) return true;
    if (p.specialistAttendedAt) return true;
    // if backend provides an explicit attendedAt that represents specialist attendance, use it as a fallback
    if (p.attendedAt && p.source === 'specialist') return true;
    return false;
  };

  const pendingPatients = normalizedPatients.filter(p => !isAttended(p) && ((p.isWaiting === true) || (p.status === "pending")));
  const attendedPatients = normalizedPatients.filter(p => isAttended(p));

  // enrich pending patients with live wait, severity and priority, then produce nurse queue
  const pendingWithMeta = pendingPatients.map(p => {
    const start = waitStartFor(p);
    const waitMs = start ? Math.max(0, now - start) : null;
    const sev = severityFromReport(p);
    const priority = computePriority(p, now);
    const attended = typeof p.specialistAttended === 'boolean' ? p.specialistAttended : (typeof p.attended === 'boolean' ? p.attended : !!p.assigned);
    return { patient: p, waitMs, sev, priority, attended };
  }).sort((a,b) => b.priority.score - a.priority.score || (b.waitMs || 0) - (a.waitMs || 0));

  // determine urgent alerts: when waitRatio >= 1 (surpassed SATS threshold) or Critical
  const urgentAlerts = pendingWithMeta.filter(item => {
    const pr = item.priority || {};
    if (!pr) return false;
    if (pr.label === 'Critical' || pr.category === 'Red') return true;
    return (pr.waitRatio || 0) >= 1;
  });

  // play an audible beep when the urgent alert set changes
  useEffect(() => {
    const ids = urgentAlerts.map(a => a.patient._id || a.patient.id).filter(Boolean);
    // if the urgent set changed since last render, reopen the emergency banner and play alarm
    const prev = prevUrgentIdsRef.current || [];
    const changed = ids.length !== prev.length || ids.some((id, idx) => prev[idx] !== id);
    if (changed) {
      try {
        if (ids.length > 0) {
          if (!alarmRef.current) alarmRef.current = new Audio('/alarm.mp3');
          alarmRef.current.play?.().catch(() => {});
        }
      } catch (e) {}
      setShowEmergencyBanner(true);
    }
    prevUrgentIdsRef.current = ids;
  }, [urgentAlerts]);

  return (
    <SidebarLayout username={userName} handleLogout={handleLogout}>
      <style>{`
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.6);} 70% { box-shadow: 0 0 0 10px rgba(239,68,68,0);} 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0);} }
        .pulse { animation: pulse 1.5s infinite; }
        .cleaning-icon { display: inline-block; width: 14px; height: 14px; border-radius: 50%; background: #f59e0b; margin-right: 8px; }
      `}</style>

      {/* Emergency Alert Modal (refurbished professional design) */}
      {urgentAlerts.length > 0 && showEmergencyBanner && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          zIndex: 999,
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'flex-end',
          paddingTop: '40px',
          paddingRight: '20px',
          backdropFilter: 'blur(2px)'
        }}>
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            width: '420px',
            maxHeight: 'calc(100vh - 100px)',
            boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            animation: 'slideIn 0.3s ease-out'
          }}>
            {/* Header with Gradient Background */}
            <div style={{
              background: 'linear-gradient(135deg, #fee2e2 0%, #fef3c7 100%)',
              padding: '24px 24px',
              borderBottom: '2px solid #fecaca',
              position: 'relative'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '10px',
                    background: '#991b1b',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '24px',
                    fontWeight: 700,
                    color: '#fef2f2',
                    boxShadow: '0 0 12px rgba(153,27,27,0.4)'
                  }}>
                    🚨
                  </div>
                  <div>
                    <div style={{
                      fontSize: '16px',
                      fontWeight: 700,
                      color: '#991b1b',
                      letterSpacing: '0.5px'
                    }}>
                      CRITICAL ALERT
                    </div>
                    <div style={{
                      fontSize: '12px',
                      color: '#7f1d1d',
                      marginTop: '2px'
                    }}>
                      {urgentAlerts.length} patient{urgentAlerts.length !== 1 ? 's' : ''} require immediate attention
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setShowEmergencyBanner(false)}
                  style={{
                    background: 'rgba(155,27,27,0.1)',
                    border: 'none',
                    color: '#991b1b',
                    cursor: 'pointer',
                    fontSize: '20px',
                    width: '32px',
                    height: '32px',
                    borderRadius: '6px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = 'rgba(155,27,27,0.2)';
                    e.target.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'rgba(155,27,27,0.1)';
                    e.target.style.boxShadow = 'none';
                  }}
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Alert List */}
            <div style={{
              flex: 1,
              overflowY: 'auto',
              padding: '16px 0'
            }}>
              {urgentAlerts.map((item, idx) => {
                const id = item.patient._id || item.patient.id;
                const severityColors = {
                  CRITICAL: { bg: '#fee2e2', border: '#fecaca', text: '#991b1b' },
                  HIGH: { bg: '#ffedd5', border: '#fed7aa', text: '#9a3412' },
                  MODERATE: { bg: '#fef3c7', border: '#fde68a', text: '#92400e' },
                  LOW: { bg: '#dcfce7', border: '#bbf7d0', text: '#166534' }
                };
                const severity = item.priority.label.toUpperCase();
                const colors = severityColors[severity] || severityColors.CRITICAL;

                return (
                  <div
                    key={id}
                    style={{
                      padding: '16px 16px',
                      borderBottom: idx === urgentAlerts.length - 1 ? 'none' : '1px solid #f1f5f9',
                      background: idx % 2 === 0 ? '#ffffff' : '#f9fafb',
                      transition: 'background 0.2s ease'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = '#eff6ff'}
                    onMouseLeave={(e) => e.currentTarget.style.background = idx % 2 === 0 ? '#ffffff' : '#f9fafb'}
                  >
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      {/* Severity Badge */}
                      <div style={{
                        width: '44px',
                        height: '44px',
                        borderRadius: '8px',
                        background: colors.bg,
                        border: `2px solid ${colors.border}`,
                        color: colors.text,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 700,
                        fontSize: '14px',
                        flexShrink: 0
                      }}>
                        {String(item.patient.id || id).slice(-2)}
                      </div>

                      {/* Patient Info */}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{
                          fontWeight: 700,
                          color: '#1e293b',
                          fontSize: '14px',
                          marginBottom: '4px'
                        }}>
                          {item.patient.patient?.name ?? (`#${String(item.patient.id || id).slice(-6)}`)}
                        </div>
                        <div style={{
                          fontSize: '12px',
                          color: '#64748b',
                          display: 'flex',
                          gap: '8px',
                          alignItems: 'center',
                          flexWrap: 'wrap'
                        }}>
                          <span style={{ fontWeight: 600, color: colors.text }}>
                            {item.priority.label}
                          </span>
                          <span style={{ color: '#94a3b8' }}>•</span>
                          <span style={{ fontFamily: 'monospace', fontWeight: 500 }}>
                            {item.waitMs ? formatWaitDetailed(item.waitMs) : 'N/A'} waiting
                          </span>
                        </div>
                      </div>

                      {/* Action Button */}
                      <button
                        onClick={() => navigate(`/handover/ai/${id}`)}
                        style={{
                          padding: '8px 14px',
                          borderRadius: '6px',
                          border: 'none',
                          background: '#991b1b',
                          color: '#ffffff',
                          cursor: 'pointer',
                          fontWeight: 600,
                          fontSize: '12px',
                          transition: 'all 0.2s ease',
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = '#7f1d1d';
                          e.target.style.boxShadow = '0 4px 8px rgba(153,27,27,0.3)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = '#991b1b';
                          e.target.style.boxShadow = 'none';
                        }}
                      >
                        📋 View
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Footer with Summary */}
            <div style={{
              padding: '12px 16px',
              background: '#f9fafb',
              borderTop: '1px solid #e5e7eb',
              fontSize: '12px',
              color: '#64748b',
              textAlign: 'center'
            }}>
              {urgentAlerts.length > 1 ? `${urgentAlerts.length} patients overdue for triage` : '1 patient overdue for triage'}
            </div>
          </div>
        </div>
      )}

      {/* Priority Queue Assignment Notification Modal */}
      {showPriorityNotificationModal && priorityQueueNotification && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          zIndex: 999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backdropFilter: 'blur(2px)'
        }}>
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            width: '420px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            animation: 'scaleIn 0.3s ease-out'
          }}>
            {/* Header with Gradient */}
            <div style={{
              background: priorityQueueNotification.attended 
                ? 'linear-gradient(135deg, #dcfce7 0%, #d1fae5 100%)'
                : 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
              padding: '24px 24px',
              borderBottom: `2px solid ${priorityQueueNotification.attended ? '#86efac' : '#93c5fd'}`,
              position: 'relative'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '10px',
                    background: priorityQueueNotification.attended ? '#059669' : '#2563eb',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '24px',
                    fontWeight: 700,
                    color: '#ffffff',
                    boxShadow: `0 0 12px ${priorityQueueNotification.attended ? 'rgba(5,150,105,0.4)' : 'rgba(37,99,235,0.4)'}`
                  }}>
                    {priorityQueueNotification.attended ? '✓' : '📋'}
                  </div>
                  <div>
                    <div style={{
                      fontSize: '16px',
                      fontWeight: 700,
                      color: priorityQueueNotification.attended ? '#059669' : '#1e40af',
                      letterSpacing: '0.5px'
                    }}>
                      {priorityQueueNotification.attended ? 'PATIENT ATTENDED' : 'ASSIGNMENT CONFIRMED'}
                    </div>
                    <div style={{
                      fontSize: '12px',
                      color: priorityQueueNotification.attended ? '#047857' : '#1e3a8a',
                      marginTop: '2px'
                    }}>
                      {priorityQueueNotification.attended ? 'Successfully marked as attended' : 'Patient assigned to queue'}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setShowPriorityNotificationModal(false)}
                  style={{
                    background: priorityQueueNotification.attended ? 'rgba(5,150,105,0.1)' : 'rgba(37,99,235,0.1)',
                    border: 'none',
                    color: priorityQueueNotification.attended ? '#059669' : '#2563eb',
                    cursor: 'pointer',
                    fontSize: '20px',
                    width: '32px',
                    height: '32px',
                    borderRadius: '6px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = priorityQueueNotification.attended 
                      ? 'rgba(5,150,105,0.2)' 
                      : 'rgba(37,99,235,0.2)';
                    e.target.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = priorityQueueNotification.attended 
                      ? 'rgba(5,150,105,0.1)' 
                      : 'rgba(37,99,235,0.1)';
                    e.target.style.boxShadow = 'none';
                  }}
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Patient Info */}
            <div style={{
              padding: '24px 24px',
              borderBottom: '1px solid #e5e7eb'
            }}>
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, marginBottom: 4 }}>
                  PATIENT NAME
                </div>
                <div style={{
                  fontSize: '16px',
                  fontWeight: 700,
                  color: '#1e293b'
                }}>
                  {priorityQueueNotification.patientName}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, marginBottom: 4 }}>
                    PRIORITY LEVEL
                  </div>
                  <div style={{
                    fontSize: '14px',
                    fontWeight: 700,
                    color: priorityQueueNotification.severity === 1 ? '#991b1b' 
                      : priorityQueueNotification.severity === 2 ? '#9a3412'
                      : priorityQueueNotification.severity === 3 ? '#92400e'
                      : '#166534',
                    background: priorityQueueNotification.severity === 1 ? '#fee2e2'
                      : priorityQueueNotification.severity === 2 ? '#ffedd5'
                      : priorityQueueNotification.severity === 3 ? '#fef3c7'
                      : '#dcfce7',
                    padding: '8px 12px',
                    borderRadius: 6,
                    textAlign: 'center'
                  }}>
                    {priorityQueueNotification.priorityLabel}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, marginBottom: 4 }}>
                    WAIT TIME
                  </div>
                  <div style={{
                    fontSize: '14px',
                    fontWeight: 700,
                    color: '#1e293b',
                    fontFamily: 'monospace',
                    background: '#f1f5f9',
                    padding: '8px 12px',
                    borderRadius: 6,
                    textAlign: 'center'
                  }}>
                    {priorityQueueNotification.waitTime}
                  </div>
                </div>
              </div>
            </div>

            {/* Footer Buttons */}
            <div style={{
              padding: '16px 24px',
              display: 'flex',
              gap: '12px',
              background: '#f9fafb'
            }}>
              <button
                onClick={() => setShowPriorityNotificationModal(false)}
                style={{
                  flex: 1,
                  padding: '10px 16px',
                  borderRadius: '6px',
                  border: '1px solid #e5e7eb',
                  background: '#ffffff',
                  color: '#475569',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '13px',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = '#f1f5f9';
                  e.target.style.borderColor = '#cbd5e1';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = '#ffffff';
                  e.target.style.borderColor = '#e5e7eb';
                }}
              >
                Close
              </button>
              <button
                onClick={() => {
                  setShowPriorityNotificationModal(false);
                  if (priorityQueueNotification.patientId) {
                    navigate(`/summary/${priorityQueueNotification.patientId}`);
                  }
                }}
                style={{
                  flex: 1,
                  padding: '10px 16px',
                  borderRadius: '6px',
                  border: 'none',
                  background: priorityQueueNotification.attended ? '#059669' : '#2563eb',
                  color: '#ffffff',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '13px',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = priorityQueueNotification.attended ? '#047857' : '#1d4ed8';
                  e.target.style.boxShadow = priorityQueueNotification.attended
                    ? '0 4px 8px rgba(5,150,105,0.3)'
                    : '0 4px 8px rgba(37,99,235,0.3)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = priorityQueueNotification.attended ? '#059669' : '#2563eb';
                  e.target.style.boxShadow = 'none';
                }}
              >
                View Details
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes slideIn {
          from {
            transform: translateX(400px);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        @keyframes scaleIn {
          from {
            transform: scale(0.95);
            opacity: 0;
          }
          to {
            transform: scale(1);
            opacity: 1;
          }
        }
      `}</style>
      )}

      {/* If assigned a specific role (not triage nurse), render role-specific dashboard */}
      {assignedRole && assignedRole !== 'TRIAGE_NURSE' && triageAllowed ? (
        <>
          {assignedRole === 'CLINICAL_NURSE' && (
            <ClinicalNurseDashboard 
              config={{
                staffName: username,
                staffDepartment: assignedContext || 'ER',
                shiftStartTime: currentRecord?.clockInTime ? new Date(currentRecord.clockInTime).toLocaleTimeString() : 'N/A',
                shiftEndTime: '—'
              }}
              context={assignedContext || 'ER'}
            />
          )}
          {assignedRole === 'PHYSICIAN' && (
            <PhysicianDashboard 
              config={{
                staffName: username,
                staffDepartment: assignedContext || 'ER',
                shiftStartTime: currentRecord?.clockInTime ? new Date(currentRecord.clockInTime).toLocaleTimeString() : 'N/A',
                shiftEndTime: '—'
              }}
              context={assignedContext || 'ER'}
            />
          )}
        </>
      ) : (
        <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={styles.statGrid}>
            {[
              ["Active", patients.length],
              ["Pending", pendingPatients.length],
              ["Beds Allocation", bedsAvailable]
            ].map(([label, count]) => (
              <div key={label} style={styles.statCard}>
                <p>{label}</p>
                <h1 style={{ fontSize: "40px" }}>{count}</h1>
              </div>
            ))}
          </div>

          <div style={styles.queueBox}>
            {recordLoading ? (
              <div style={{ padding: 16, color: '#6b7280' }}>Checking shift status...</div>
            ) : (!currentRecord || currentRecord.status === 'CLOCKED_OUT') ? (
              <div style={{ padding: 16, color: '#92400e', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>Clock in to work on patients</div>
                <button onClick={() => navigate('/clock-in-out')} style={{ ...styles.btnLarge, padding: '8px 12px' }}>Clock In</button>
              </div>
            ) : (currentRecord.status === 'ON_BREAK') ? (
              <div style={{ padding: 16, color: '#92400e', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>You are currently on break — end break to work on patients</div>
                <button onClick={async () => {
                  try {
                    const token = localStorage.getItem('token');
                    const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
                    const res = await fetch(`${BACKEND_URL}/api/staff/break-end`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
                      body: JSON.stringify({ breakDuration: 0 })
                    });
                    const data = await res.json().catch(() => ({}));
                    if (res.ok && data.success) {
                      await fetchTodayRecord();
                      alert('Break ended');
                    } else {
                      console.error('[END BREAK] Error:', data);
                      alert('Failed to end break: ' + (data.message || 'server error'));
                    }
                  } catch (e) {
                    console.error('End break error', e);
                    alert('Error ending break: ' + e.message);
                  }
                }} style={{ ...styles.btnLarge, padding: '8px 12px' }}>End Break</button>
              </div>
            ) : null}



          </div>

          {attendedPatients.length > 0 && (
            <div style={{ ...styles.queueBox, marginTop: 40 }}>
              <h3>Attended Patients</h3>
              <table style={styles.table}>
                <thead>
                  <tr style={styles.tableHeadRow}>
                    <th>ID</th>
                    <th>Patient</th>
                    <th>Severity</th>
                    <th>Attending Doctor</th>
                    <th>Nurse Notes</th>
                    <th style={{ textAlign: "right" }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {attendedPatients.map(p0 => {
                    const p = normalizePatient(p0);
                    const doctor = p.specialistDoctor ?? (p.attendedByName || p.attendedBy || null);
                    return (
                      <tr key={p._id || p.id} style={styles.tableRow}>
                        <td>#{String(p.id || (p._id || "")).padStart(3, "0")}</td>
                        <td>{p.patient?.name ?? "N/A"}</td>
                        <td style={{ fontWeight: "bold", color: severityColor(p.severity || (p.triage?.severity)) }}>
                          {(p.severity || p.triage?.severity) === 3 ? "High" : ((p.severity || p.triage?.severity) === 2 ? "Med" : "Low")}
                        </td>
                        <td style={{ textAlign: 'center' }} title={p.specialistAttendedAt ? formatTimestamp(p.specialistAttendedAt) : ''}>
                          {doctor ? <span style={{ display: 'inline-block' }}>{doctor}</span> : <span style={{ color: '#777' }}>—</span>}
                          {p.specialistAttendedAt && (
                            <div style={{ fontSize: 11, color: '#666' }}>{new Date(p.specialistAttendedAt).toLocaleTimeString()}</div>
                          )}
                        </td>
                        <td>{p.nurseNotes ?? ""}</td>
                        <td style={{ textAlign: "right" }}>
                          <button onClick={() => navigate(`/confirmed/encounter/${p._id || p.id}`)} style={{ ...styles.btnLarge, padding: "10px", fontSize: "12px", width: "auto" }}>
                            View Final Report
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Unattended / Waiting Section - Professional Dashboard */}
          {handedOverEncounters.length > 0 && (
            <div style={{ 
              marginTop: 40, 
              borderRadius: 12,
              background: '#ffffff',
              boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 4px 6px rgba(0,0,0,0.05)',
              border: '1px solid #e5e7eb',
              overflow: 'hidden'
            }}>
              {/* Header section */}
              <div style={{ 
                padding: '24px 28px',
                background: 'linear-gradient(135deg, #fff5f7 0%, #f0f9ff 100%)',
                borderBottom: '1px solid #e5e7eb',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <div>
                  <h3 style={{ 
                    margin: '0 0 4px 0',
                    fontSize: '20px',
                    fontWeight: 700,
                    color: '#1e293b',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    📋 Awaiting Clinician Review
                  </h3>
                  <p style={{ 
                    margin: 0,
                    fontSize: '13px',
                    color: '#64748b'
                  }}>
                    {handedOverEncounters.length} patient{handedOverEncounters.length !== 1 ? 's' : ''} waiting for physician/nurse acknowledgment
                  </p>
                </div>
              </div>

              {/* Table container with better scrolling */}
              <div style={{ overflowX: 'auto' }}>
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: '14px'
                }}>
                  <thead>
                    <tr style={{ 
                      background: '#f8fafc',
                      borderBottom: '2px solid #e2e8f0'
                    }}>
                      <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>ID</th>
                      <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Patient</th>
                      <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Specialty</th>
                      <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Physician</th>
                      <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Physician Status</th>
                      <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Nurse</th>
                      <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Nurse Status</th>
                      <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Severity</th>
                      <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Wait Time</th>
                      <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Priority</th>
                      <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {handedOverEncounters.map((enc, idx) => {
                      const pt = enc.patient || {};
                      const fullName = pt.name || `${pt.firstName || ''} ${pt.lastName || ''}`.trim() || 'Unknown';
                      const specialty = enc.assigned_specialty || 'General';
                      const clinicians = enc.assigned_clinicians || [];
                      const handovers = enc.clinical_handovers || [];
                      
                      const physicianNames = clinicians
                        .filter(c => c.role === 'PHYSICIAN')
                        .map(c => c.name || c.specialty)
                        .join(', ') || '—';
                      const physicianHandover = handovers.find(h => 
                        clinicians.find(c => c._id?.toString?.() === h.clinician_id?.toString?.() && c.role === 'PHYSICIAN')
                      );
                      const physicianStatus = physicianHandover?.handover_status || 'NOTIFIED';
                      
                      const nurseNames = clinicians
                        .filter(c => c.role === 'NURSE')
                        .map(c => c.name)
                        .join(', ') || '—';
                      const nurseHandover = handovers.find(h => 
                        clinicians.find(c => c._id?.toString?.() === h.clinician_id?.toString?.() && c.role === 'NURSE')
                      );
                      const nurseStatus = nurseHandover?.handover_status || 'NOTIFIED';
                      
                      // Professional status badge
                      const StatusBadge = ({ status }) => {
                        const statusConfig = {
                          'NOTIFIED': { label: 'Notified', bg: '#fef3c7', color: '#92400e', border: '#fcd34d' },
                          'VIEWED': { label: 'Viewed', bg: '#dbeafe', color: '#1e40af', border: '#93c5fd' },
                          'ACKNOWLEDGED': { label: 'Acknowledged', bg: '#d1fae5', color: '#065f46', border: '#86efac' },
                          'IN_PROGRESS': { label: 'In Progress', bg: '#ede9fe', color: '#5b21b6', border: '#ddd6fe' },
                          'COMPLETED': { label: 'Completed', bg: '#cffafe', color: '#164e63', border: '#67e8f9' }
                        };
                        const config = statusConfig[status] || statusConfig['NOTIFIED'];
                        return (
                          <span style={{ 
                            padding: '6px 12px', 
                            borderRadius: 6, 
                            background: config.bg, 
                            color: config.color,
                            fontSize: '12px', 
                            fontWeight: 600,
                            display: 'inline-block',
                            border: `1px solid ${config.border}`,
                            whiteSpace: 'nowrap'
                          }}>
                            {config.label}
                          </span>
                        );
                      };

                      const rpt = enc.triage?.aiTriageReport || enc.triage?.manualTriageReport || {};
                      const esiLevel = rpt.esiLevel || rpt.ESI || enc.severity || 4;
                      
                      const severityConfig = {
                        1: { label: 'CRITICAL', color: '#991b1b', bg: '#fee2e2', icon: '🔴' },
                        2: { label: 'HIGH', color: '#9a3412', bg: '#ffedd5', icon: '🟠' },
                        3: { label: 'MODERATE', color: '#92400e', bg: '#fef3c7', icon: '🟡' },
                        4: { label: 'LOW', color: '#166534', bg: '#dcfce7', icon: '🟢' },
                        5: { label: 'LOW', color: '#166534', bg: '#dcfce7', icon: '🟢' }
                      };
                      const severityConfig_item = severityConfig[esiLevel] || severityConfig[4];

                      const priority = computePriority(enc, now);
                      const displayPriority = priority.label;
                      
                      const priorityConfig = {
                        'Critical': { color: '#991b1b', bg: '#fee2e2', border: '#fca5a5' },
                        'High': { color: '#b91c1c', bg: '#fecaca', border: '#fca5a5' },
                        'Med': { color: '#ea580c', bg: '#fed7aa', border: '#fdba74' },
                        'Low': { color: '#059669', bg: '#d1fae5', border: '#6ee7b7' }
                      };
                      const priorityConfig_item = priorityConfig[displayPriority] || priorityConfig['Low'];

                      const handoverTime = enc.handover_completed_at ? new Date(enc.handover_completed_at).getTime() : null;
                      const elapsedMs = handoverTime ? Math.max(0, now - handoverTime) : 0;
                      const hours = Math.floor(elapsedMs / 3600000);
                      const mins = Math.floor((elapsedMs % 3600000) / 60000);
                      const secs = Math.floor((elapsedMs % 60000) / 1000);
                      const elapsedDisplay = `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

                      return (
                        <tr 
                          key={enc._id} 
                          style={{ 
                            borderBottom: '1px solid #e5e7eb',
                            background: idx % 2 === 0 ? '#ffffff' : '#f9fafb',
                            transition: 'all 0.2s ease',
                            ':hover': { background: '#f0f9ff' }
                          }}
                          onMouseEnter={(e) => e.currentTarget.style.background = '#eff6ff'}
                          onMouseLeave={(e) => e.currentTarget.style.background = idx % 2 === 0 ? '#ffffff' : '#f9fafb'}
                        >
                          <td style={{ padding: '14px 16px', fontWeight: 600, color: '#1e293b', fontSize: '13px', fontFamily: 'monospace' }}>
                            #{String(enc._id || "").slice(-6).padStart(6, "0")}
                          </td>
                          <td style={{ padding: '14px 16px', fontWeight: 600, color: '#1e293b' }}>
                            {fullName}
                          </td>
                          <td style={{ padding: '14px 16px', color: '#64748b', fontSize: '13px' }}>
                            {specialty}
                          </td>
                          <td style={{ padding: '14px 16px', color: '#475569', fontSize: '13px' }}>
                            {physicianNames}
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                            <StatusBadge status={physicianStatus} />
                          </td>
                          <td style={{ padding: '14px 16px', color: '#475569', fontSize: '13px' }}>
                            {nurseNames}
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                            <StatusBadge status={nurseStatus} />
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                            <span style={{ 
                              padding: '6px 12px',
                              borderRadius: 6,
                              background: severityConfig_item.bg,
                              color: severityConfig_item.color,
                              fontWeight: 700,
                              fontSize: '12px',
                              display: 'inline-block',
                              textTransform: 'uppercase',
                              letterSpacing: '0.5px'
                            }}>
                              {severityConfig_item.label}
                            </span>
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center', fontSize: '13px', fontFamily: 'monospace', fontWeight: 600, color: '#1e293b' }}>
                            {elapsedDisplay}
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                            <span style={{ 
                              padding: '6px 12px',
                              borderRadius: 6,
                              background: priorityConfig_item.bg,
                              color: priorityConfig_item.color,
                              fontWeight: 700,
                              fontSize: '12px',
                              display: 'inline-block',
                              border: `1px solid ${priorityConfig_item.border}`
                            }}>
                              {displayPriority}
                            </span>
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                            <button
                              onClick={() => {
                                const reportType = enc.clinical_handovers?.[0]?.report_type || 'AI';
                                const path = reportType === 'MANUAL' 
                                  ? `/handover/manual/${enc._id}`
                                  : `/handover/ai/${enc._id}`;
                                navigate(path);
                              }}
                              style={{
                                padding: '8px 14px',
                                background: '#3b82f6',
                                color: '#ffffff',
                                border: 'none',
                                borderRadius: 6,
                                fontWeight: 600,
                                fontSize: '13px',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '6px',
                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                              }}
                              onMouseEnter={(e) => {
                                e.target.style.background = '#2563eb';
                                e.target.style.boxShadow = '0 4px 6px rgba(59, 130, 246, 0.3)';
                              }}
                              onMouseLeave={(e) => {
                                e.target.style.background = '#3b82f6';
                                e.target.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)';
                              }}
                            >
                              📋 View
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div style={styles.actionBox}>
            {triageAllowed ? (
              <button onClick={handleNewIntake} style={styles.btnLarge}>New Intake</button>
            ) : (
              <button disabled title="Clock in to triage" style={{ ...styles.btnLarge, opacity: 0.6, cursor: 'not-allowed' }}>New Intake</button>
            )}
          </div>
        </div>

        {triageAllowed ? (
          <aside style={{ 
            width: 360, 
            flexShrink: 0, 
            background: '#ffffff',
            borderLeft: '1px solid #e5e7eb',
            boxShadow: '-1px 0 3px rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column'
          }}>
            {/* Priority Queue Header */}
            <div style={{ 
              padding: '20px 20px',
              background: 'linear-gradient(135deg, #fef2f2 0%, #f0f9ff 100%)',
              borderBottom: '1px solid #e5e7eb'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <h3 style={{ 
                  margin: 0,
                  fontSize: '16px',
                  fontWeight: 700,
                  color: '#1e293b',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  🎯 Priority Queue
                </h3>
                <div style={{ 
                  fontSize: '12px', 
                  fontWeight: 600,
                  color: '#475569',
                  background: '#f1f5f9',
                  padding: '4px 10px',
                  borderRadius: 4
                }}>
                  {pendingWithMeta.length} waiting
                </div>
              </div>
              <p style={{ 
                margin: 0,
                fontSize: '12px',
                color: '#64748b'
              }}>
                Sorted by severity & wait time
              </p>
            </div>

            {/* Alerts Panel */}
            {showAlertsPanel && urgentAlerts.length > 0 && (
              <div style={{ 
                padding: '16px 16px',
                background: '#fef2f2',
                borderBottom: '1px solid #e5e7eb',
                margin: 0
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#991b1b', fontSize: '13px' }}>
                    🚨 Urgent Alerts • {urgentAlerts.length}
                  </div>
                  <button 
                    onClick={() => setShowAlertsPanel(false)} 
                    style={{ 
                      border: 'none', 
                      background: 'transparent', 
                      cursor: 'pointer',
                      color: '#991b1b',
                      fontSize: '16px'
                    }}
                  >
                    ✕
                  </button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 200, overflowY: 'auto' }}>
                  {urgentAlerts.map(item => {
                    const id = item.patient._id || item.patient.id;
                    return (
                      <div 
                        key={id} 
                        style={{ 
                          display: 'flex', 
                          justifyContent: 'space-between', 
                          alignItems: 'center',
                          padding: '10px 12px',
                          background: '#ffffff',
                          borderRadius: 6,
                          border: '1px solid #fecaca'
                        }}
                      >
                        <div>
                          <div style={{ fontWeight: 600, color: '#1e293b', fontSize: '13px' }}>
                            {item.patient.patient?.name ?? (`#${String(item.patient.id || id).slice(-6)}`)
                            }
                          </div>
                          <div style={{ fontSize: '11px', color: '#64748b', marginTop: 2 }}>
                            {item.priority.label} • {item.waitMs ? formatWaitDetailed(item.waitMs) : 'N/A'}
                          </div>
                        </div>
                        <button 
                          onClick={() => navigate(`/summary/${id}`)} 
                          style={{ 
                            padding: '6px 10px', 
                            borderRadius: 5, 
                            border: 'none', 
                            cursor: 'pointer', 
                            background: '#fca5a5', 
                            color: '#7f1d1d',
                            fontWeight: 600,
                            fontSize: '11px',
                            transition: 'all 0.2s ease'
                          }}
                          onMouseEnter={(e) => e.target.style.background = '#f87171'}
                          onMouseLeave={(e) => e.target.style.background = '#fca5a5'}
                        >
                          View
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Queue List */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
              {pendingWithMeta.slice(0, 20).map(({ patient: p, waitMs, sev, priority, attended }, idx) => {
                const severityConfig = {
                  1: { bg: '#fee2e2', color: '#991b1b', label: 'CRITICAL', icon: '🔴' },
                  2: { bg: '#ffedd5', color: '#9a3412', label: 'HIGH', icon: '🟠' },
                  3: { bg: '#fef3c7', color: '#92400e', label: 'MODERATE', icon: '🟡' },
                  4: { bg: '#dcfce7', color: '#166534', label: 'LOW', icon: '🟢' },
                  5: { bg: '#dcfce7', color: '#166534', label: 'LOW', icon: '🟢' }
                };
                const sevConfig = severityConfig[sev] || severityConfig[4];

                return (
                  <div 
                    key={(p._id || p.id) + '-prio'} 
                    style={{ 
                      padding: '12px 12px',
                      borderBottom: idx === pendingWithMeta.slice(0, 20).length - 1 ? 'none' : '1px solid #f1f5f9',
                      marginX: '8px',
                      transition: 'all 0.2s ease',
                      background: idx % 2 === 0 ? '#ffffff' : '#f9fafb',
                      cursor: 'pointer'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = '#eff6ff'}
                    onMouseLeave={(e) => e.currentTarget.style.background = idx % 2 === 0 ? '#ffffff' : '#f9fafb'}
                  >
                    <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                      {/* Severity Badge */}
                      <div style={{ 
                        width: 44, 
                        height: 44, 
                        borderRadius: 8, 
                        background: sevConfig.bg, 
                        color: sevConfig.color, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center', 
                        fontWeight: 700,
                        fontSize: '14px',
                        flexShrink: 0,
                        border: `2px solid ${sevConfig.color}`
                      }}>
                        {(p.id || p._id || '').toString().slice(-2)}
                      </div>

                      {/* Patient Info */}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ 
                          fontWeight: 700, 
                          color: '#1e293b',
                          fontSize: '13px',
                          marginBottom: 4,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}>
                          {p.patient?.name ?? (`#${String(p.id || (p._id || '')).padStart(3,'0')}`)
                          }
                        </div>
                        <div style={{ 
                          fontSize: '11px', 
                          color: '#64748b',
                          display: 'flex',
                          gap: '6px',
                          alignItems: 'center'
                        }}>
                          <span style={{ fontWeight: 600, color: '#475569' }}>
                            {priority.label}
                          </span>
                          <span style={{ color: '#94a3b8' }}>•</span>
                          <span style={{ fontFamily: 'monospace', fontWeight: 500 }}>
                            {waitMs ? formatWaitDetailed(waitMs) : (getWaitTime(p) || 'N/A')}
                          </span>
                        </div>
                      </div>

                      {/* View Button */}
                      <button 
                        onClick={() => navigate(`/handover/ai/${p._id || p.id}`)}
                        style={{ 
                          padding: '6px 10px', 
                          borderRadius: 5, 
                          border: 'none', 
                          cursor: 'pointer', 
                          background: '#3b82f6', 
                          color: '#ffffff',
                          fontSize: '11px',
                          fontWeight: 600,
                          transition: 'all 0.2s ease',
                          flexShrink: 0,
                          whiteSpace: 'nowrap'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = '#2563eb';
                          e.target.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = '#3b82f6';
                          e.target.style.boxShadow = 'none';
                        }}
                      >
                        📋 View
                      </button>
                    </div>
                  </div>
                );
              })}
              
              {pendingWithMeta.length === 0 && (
                <div style={{ 
                  color: '#94a3b8', 
                  padding: '20px 16px',
                  textAlign: 'center',
                  fontSize: '13px'
                }}>
                  ✓ No patients in queue
                </div>
              )}
            </div>
          </aside>
        ) : (
          <aside style={{ width: 320, flexShrink: 0, background: '#fafbfd', padding: 12 }}>
            <div style={{ color: '#666' }}>Patient queue hidden — please clock in to view and triage patients.</div>
          </aside>
        )}
      </div>
      )}
    </SidebarLayout>
  );
};

export default NurseDashboard;