import React, { useEffect, useState, useRef } from "react";
import { useLocation } from "react-router-dom";
import { severityFromReport, waitStartFor, computePriority, formatWaitDetailed, normalizePatient, formatTimestamp } from "../utils/priority";
import { styles } from "./styles";
import { FiUser, FiFileText, FiCheckSquare, FiHome, FiSettings, FiUsers, FiBarChart2, FiClock, FiPlusCircle } from 'react-icons/fi';
import { MdLibraryBooks } from 'react-icons/md';
import { FaBed, FaSearch, FaHistory } from 'react-icons/fa';
import nhpLogo from './NHP_Icon.png';
import SidebarLayout from './SidebarLayout';
import { useUserFromToken } from '../hooks/useUserFromToken';
import ClinicalNurseDashboard from '../components/dashboards/ClinicalNurseDashboard';
import PhysicianDashboard from '../components/dashboards/PhysicianDashboard';
import OffDutyDashboard from '../components/dashboards/OffDutyDashboard';
import { decodeJwt } from '../utils/jwt';
import { isAwaitingPhysicianAcknowledgment, ACKNOWLEDGED_HANDOVER_STATUSES } from '../utils/handoverStatusUtils';
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

  // Bed and inpatient metrics are refreshed from the hospital APIs below.
  const [bedState, setBedState] = useState({ total: 0, occupied: 0, cleaning: [], monitoredOccupied: 0 });
  const [metricsLoading, setMetricsLoading] = useState(true);

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

  // Acknowledged encounters state
  const [acknowledgedEncounters, setAcknowledgedEncounters] = useState([]);
  const [acknowledgedLoading, setAcknowledgedLoading] = useState(false);

  // Priority Queue assignment notification modal
  const [priorityQueueNotification, setPriorityQueueNotification] = useState(null);
  const [showPriorityNotificationModal, setShowPriorityNotificationModal] = useState(false);
  const [showPriorityQueueModal, setShowPriorityQueueModal] = useState(false);

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

  useEffect(() => {
    let cancelled = false;

    const fetchLiveMetrics = async () => {
      const token = localStorage.getItem('token');
      if (!token || cancelled) return;
      const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` };
      const hospitalId = decodeJwt(token)?.hospitalId;
      const statsUrl = hospitalId
        ? `${BACKEND_URL}/api/beds/stats?hospitalId=${encodeURIComponent(hospitalId)}`
        : `${BACKEND_URL}/api/beds/stats`;

      try {
        const [bedsResponse, hospitalizedResponse] = await Promise.all([
          fetch(statsUrl, { headers }),
          fetch(`${BACKEND_URL}/api/encounters?status=HOSPITALIZED&limit=500&bypassNurseFilter=true`, { headers })
        ]);
        const bedsData = await bedsResponse.json().catch(() => ({}));
        const hospitalizedData = await hospitalizedResponse.json().catch(() => ({}));
        const bedStats = bedsData?.data || bedsData?.stats || {};
        const hospitalized = Array.isArray(hospitalizedData)
          ? hospitalizedData
          : hospitalizedData?.encounters || hospitalizedData?.data || [];

        if (!cancelled) {
          setBedState({
            total: Number(bedStats.total) || 0,
            occupied: Number(bedStats.occupied) || hospitalized.length,
            cleaning: [],
            monitoredOccupied: Number(bedStats.occupied) || hospitalized.length
          });
          setMetricsLoading(false);
        }
      } catch (error) {
        console.error('Failed to fetch live dashboard metrics:', error);
        if (!cancelled) setMetricsLoading(false);
      }
    };

    fetchLiveMetrics();
    const interval = window.setInterval(fetchLiveMetrics, 10000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  // Fetch encounters that have been handed over but not yet acknowledged by physician
  // Status: AWAITING_PHYSICIAN + handover_status: NOTIFIED (physician not yet acknowledged)
  const fetchHandedOverEncounters = async () => {
    const token = localStorage.getItem('token');
    if (!token) {
      setHandedOverEncounters([]);
      return;
    }
    try {
      setHandoverLoading(true);
      console.log('[DASHBOARD_FETCH] Fetching handed over encounters (awaiting acknowledgment)...');
      const res = await fetch(`${BACKEND_URL}/api/encounters?status=AWAITING_PHYSICIAN&handover_status=NOTIFIED`, {
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
      });
      
      console.log('[DASHBOARD_FETCH] Response status:', res.status);
      const data = await res.json().catch((e) => {
        console.error('[DASHBOARD_FETCH] Failed to parse response:', e);
        return {};
      });
      
      console.log('[DASHBOARD_FETCH] Response data:', data);
      
      if (res.ok && data.success && Array.isArray(data.encounters)) {
        console.log('[DASHBOARD_FETCH] Found', data.encounters.length, 'encounters from backend');
        
        // Frontend filter: exclude encounters that have ANY acknowledged handover
        // (in case multiple handovers exist for same encounter with mixed statuses)
        const filtered = data.encounters.filter((enc) => isAwaitingPhysicianAcknowledgment(enc));
        
        console.log('[DASHBOARD_FETCH] After filtering acknowledged:', filtered.length, 'encounters awaiting physician acknowledgment');
        setHandedOverEncounters(filtered);
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

  // Fetch acknowledged encounters
  const fetchAcknowledgedEncounters = async () => {
    const token = localStorage.getItem('token');
    if (!token) {
      setAcknowledgedEncounters([]);
      return;
    }
    try {
      setAcknowledgedLoading(true);
      console.log('[DASHBOARD_FETCH] Fetching acknowledged encounters...');
      const res = await fetch(`${BACKEND_URL}/api/encounters?handover_status=ACKNOWLEDGED,IN_PROGRESS`, {
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
      });
      
      const data = await res.json().catch((e) => {
        console.error('[DASHBOARD_FETCH] Failed to parse acknowledged response:', e);
        return {};
      });
      
      if (res.ok && data.success && Array.isArray(data.encounters)) {
        const filtered = data.encounters.filter((enc) => {
          const acknowledged = Array.isArray(enc.clinical_handovers)
            ? enc.clinical_handovers.some((handover) => ACKNOWLEDGED_HANDOVER_STATUSES.includes(handover?.handover_status))
            : false;
          return acknowledged;
        });

        console.log('[DASHBOARD_FETCH] Found', filtered.length, 'active specialist encounters after filtering');
        setAcknowledgedEncounters(filtered);
      } else {
        console.log('[DASHBOARD_FETCH] No active specialist encounters');
        setAcknowledgedEncounters([]);
      }
    } catch (e) {
      console.error('[DASHBOARD_FETCH] Error fetching acknowledged encounters:', e);
      setAcknowledgedEncounters([]);
    } finally {
      setAcknowledgedLoading(false);
    }
  };

  useEffect(() => {
    // Poll for handed over and acknowledged encounters
    // Handed over encounters: only poll if nurse is clocked in (active work)
    // Acknowledged encounters: always poll (status visibility for all staff)
    
    // Always fetch acknowledged encounters (regardless of clock-in status)
    fetchAcknowledgedEncounters();
    const acknowledgedInterval = setInterval(() => {
      fetchAcknowledgedEncounters();
    }, 5000);
    
    // Only fetch handed over encounters if nurse is clocked in
    let handoverInterval = null;
    if (currentRecord && currentRecord.status === 'CLOCKED_IN') {
      fetchHandedOverEncounters();
      handoverInterval = setInterval(() => {
        fetchHandedOverEncounters();
      }, 5000);
    }
    
    return () => {
      clearInterval(acknowledgedInterval);
      if (handoverInterval) clearInterval(handoverInterval);
    };
  }, [currentRecord]);

  // Update `now` frequently (every 100ms) to keep timers and priority calculations live
  useEffect(() => {
    const interval = setInterval(() => {
      setNow(Date.now());
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // Fetch assigned role (not dependent on clock-in - all staff should see their role-specific dashboard)
  useEffect(() => {
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
          // Fall through to JWT fallback below
        } else {
          const data = await res.json();
          const shift = data.shift;

          if (!shift) {
            console.log('[ASSIGNED ROLE] No shift found');
          } else {
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
              return; // Successfully set role, exit
            }
          }
        }

        // If we reach here, shift lookup failed or user not in shift - use JWT fallback
        console.log('[ASSIGNED ROLE] Not found in shift assignment, checking JWT role as fallback...');
        const jwtRole = decoded?.role;
        const jwtNsgRole = decoded?.nsgRole;
        
        if (jwtRole || jwtNsgRole) {
          
          console.log('[ASSIGNED ROLE] JWT role:', jwtRole, 'nsgRole:', jwtNsgRole);
          
          // Map JWT role to dashboard role
          if (jwtNsgRole === 'TRIAGE_NURSE') {
            setAssignedRole('TRIAGE_NURSE');
            setAssignedContext('TRIAGE');
            console.log('[ASSIGNED ROLE] Fallback: Using JWT TRIAGE_NURSE role');
          } else if (jwtNsgRole === 'CLINICAL_NURSE') {
            setAssignedRole('CLINICAL_NURSE');
            setAssignedContext('ER');
            console.log('[ASSIGNED ROLE] Fallback: Using JWT CLINICAL_NURSE role');
          } else if (jwtRole === 'PHYSICIAN') {
            setAssignedRole('PHYSICIAN');
            setAssignedContext('ER');
            console.log('[ASSIGNED ROLE] Fallback: Using JWT PHYSICIAN role');
          } else {
            console.log('[ASSIGNED ROLE] No matching role found');
            setAssignedRole(null);
          }
        }
      } catch (err) {
        console.error('[ASSIGNED ROLE] Error:', err);
        setAssignedRole(null);
      } finally {
        setRoleFetchLoading(false);
      }
    };

    fetchAssignedRole();
  }, []);

  const triageAllowed = !!(currentRecord && currentRecord.status === 'CLOCKED_IN');

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

  const nurseMetrics = {
    totalBeds: bedState.total,
    hospitalizedPatients: bedState.occupied,
    pendingTriagedPatients: pendingPatients.length
  };

  const formatElapsed = (ms) => {
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    return `${m}m`;
  };

  const currentTime = new Date(now);

  const severityColor = (sev) => {
    if (sev === 1) return "#ef4444";  // RED
    if (sev === 2) return "#f97316";  // ORANGE
    if (sev === 3) return "#eab308";  // YELLOW
    if (sev === 4) return "#22c55e";  // GREEN
    return "#94a3b8";  // Unknown
  };

  // normalize all patients first so attendance fields are consistent

  // enrich pending patients with live wait, severity and priority, then produce nurse queue
  const pendingWithMeta = pendingPatients.map(p => {
    const start = waitStartFor(p);
    const waitMs = start ? Math.max(0, now - start) : null;
    const sev = severityFromReport(p);
    const priority = computePriority(p, now);
    const attended = typeof p.specialistAttended === 'boolean' ? p.specialistAttended : (typeof p.attended === 'boolean' ? p.attended : !!p.assigned);
    return { patient: p, waitMs, sev, priority, attended };
  }).sort((a,b) => b.priority.score - a.priority.score || (b.waitMs || 0) - (a.waitMs || 0));

  const queueManagerEntries = pendingWithMeta.slice(0, 5).map((item, index) => {
    const patient = item.patient || {};
    const patientName = patient.patient?.name || patient.name || patient.patientName || `Patient ${index + 1}`;
    const waitMs = item.waitMs || (index + 1) * 5 * 60 * 1000;
    const status = index === 0 ? 'IN PROGRESS' : index === 1 ? 'NEXT UP' : patient.priority?.category === 'Red' || patient.priority?.queueSignals?.includes('Time-to-target breached') ? 'AI TRIAGING' : 'WAITING';
    const waitLabel = index === 0 ? 'NOW' : `${Math.max(1, Math.round(waitMs / 60000))} min`;
    return {
      id: item.id || item._id || patient._id || patient.id || `queue-${index}`,
      name: patientName,
      type: patient.patient?.chiefComplaint || patient.chiefComplaint || patient.reason || (index === 0 ? 'Urgent Walk-in' : index === 1 ? 'Follow-up' : index === 2 ? 'New Patient' : index === 3 ? 'Lab Follow-up' : 'Walk-in'),
      waitMinutes: Math.max(4, Math.round(waitMs / 60000)),
      waitLabel,
      status,
      category: patient.priority?.category || 'Yellow',
      priorityLabel: patient.priority?.label || 'Moderate'
    };
  });

  const providerLoadEntries = [];
  pendingWithMeta.forEach(({ patient }) => {
    const clinician = patient.assigned_clinician || patient.assignedClinician || patient.triage?.assigned_physician;
    const name = clinician?.name || patient.specialistDoctor || patient.attendingDoctor || 'Unassigned';
    const existing = providerLoadEntries.find((provider) => provider.name === name);
    if (existing) existing.count += 1;
    else providerLoadEntries.push({ name, count: 1, percent: 0 });
  });
  const maxProviderCount = Math.max(1, ...providerLoadEntries.map((provider) => provider.count));
  providerLoadEntries.forEach((provider) => {
    provider.percent = Math.min(100, Math.round((provider.count / maxProviderCount) * 100));
  });

  const congestionMessage = providerLoadEntries.some((provider) => provider.count >= 3)
    ? `Congestion Alert: ${providerLoadEntries.find((provider) => provider.count >= 3).name}'s queue is exceeding the current workload threshold.`
    : null;

  // determine urgent alerts: when waitRatio >= 1 (surpassed SATS threshold) or Critical
  // Only show alerts if nurse is clocked in
  const urgentAlerts = triageAllowed ? pendingWithMeta.filter(item => {
    const pr = item.priority || {};
    if (!pr) return false;
    if (pr.label === 'Critical' || pr.category === 'Red') return true;
    return (pr.waitRatio || 0) >= 1;
  }) : [];

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
        .priority-queue-launcher { display: none; }
        @media (max-width: 1180px) {
          .nurse-priority-queue { display: none !important; }
          .priority-queue-launcher { display: flex; position: fixed; right: 24px; bottom: 28px; z-index: 40; width: 62px; height: 62px; align-items: center; justify-content: center; border: 2px solid #0b3b70; border-radius: 50%; background: #ffffff; box-shadow: 0 10px 28px rgba(11,59,112,.3); cursor: pointer; font-size: 27px; }
          .priority-queue-launcher:focus-visible { outline: 3px solid #93c5fd; outline-offset: 3px; }
        }
        .priority-queue-launcher-count { position: absolute; top: -5px; right: -5px; min-width: 21px; height: 21px; padding: 0 5px; display: grid; place-items: center; border-radius: 999px; background: #dc2626; color: #fff; border: 2px solid #fff; font-size: 10px; font-weight: 800; }
        .priority-queue-modal-backdrop { position: fixed; inset: 0; z-index: 50; display: flex; align-items: center; justify-content: center; padding: 24px; background: rgba(15,23,42,.48); }
        .priority-queue-modal { width: min(560px, 100%); max-height: min(78vh, 720px); overflow: hidden; border-radius: 14px; background: #ffffff; box-shadow: 0 24px 70px rgba(15,23,42,.35); }
        .priority-queue-modal-list { max-height: calc(min(78vh, 720px) - 82px); overflow-y: auto; }
        @media (max-width: 640px) {
          .priority-queue-modal-backdrop { padding: 16px; }
          .priority-queue-modal { max-height: 84vh; border-radius: 12px; }
          .priority-queue-modal-list { max-height: calc(84vh - 82px); }
        }
        .nurse-responsive-table-shell { width: 100%; min-width: 0; max-width: 100%; overflow-x: auto; overflow-y: hidden; -webkit-overflow-scrolling: touch; }
        .nurse-responsive-table { min-width: 860px; }
        @media (max-width: 640px) {
          .nurse-dashboard-header-status {
            width: 100%;
            flex-direction: row !important;
            align-items: stretch !important;
            justify-content: flex-start !important;
            flex-wrap: wrap;
          }
          .nurse-dashboard-status-pill {
            min-height: 40px;
            flex: 1 1 150px;
            justify-content: center;
          }
          .nurse-dashboard-status-row {
            width: 100%;
            flex-wrap: wrap;
          }
          .nurse-dashboard-actions {
            width: 100%;
            margin-top: 0 !important;
            flex-direction: row !important;
            justify-content: flex-start !important;
          }
          .nurse-responsive-table-shell { overflow-x: visible; }
          .nurse-responsive-table { min-width: 0 !important; width: 100% !important; display: block; }
          .nurse-responsive-table thead { display: none; }
          .nurse-responsive-table tbody,
          .nurse-responsive-table tr,
          .nurse-responsive-table td { display: block; width: 100%; box-sizing: border-box; }
          .nurse-responsive-table tr { padding: 12px 14px; border-bottom: 1px solid #e5e7eb; }
          .nurse-responsive-table td { display: grid; grid-template-columns: minmax(92px, 34%) minmax(0, 1fr); gap: 10px; align-items: start; padding: 7px 0 !important; text-align: left !important; white-space: normal; overflow-wrap: anywhere; }
          .nurse-responsive-table td::before { color: #64748b; font-size: 10px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
          .nurse-table-attended td:nth-child(1)::before { content: 'ID'; }
          .nurse-table-attended td:nth-child(2)::before { content: 'Patient'; }
          .nurse-table-attended td:nth-child(3)::before { content: 'Severity'; }
          .nurse-table-attended td:nth-child(4)::before { content: 'Attending doctor'; }
          .nurse-table-attended td:nth-child(5)::before { content: 'Nurse notes'; }
          .nurse-table-attended td:nth-child(6)::before { content: 'Action'; }
          .nurse-table-awaiting td:nth-child(1)::before { content: 'ID'; }
          .nurse-table-awaiting td:nth-child(2)::before { content: 'Patient'; }
          .nurse-table-awaiting td:nth-child(3)::before { content: 'Specialty'; }
          .nurse-table-awaiting td:nth-child(4)::before { content: 'Physician'; }
          .nurse-table-awaiting td:nth-child(5)::before { content: 'Physician status'; }
          .nurse-table-awaiting td:nth-child(6)::before { content: 'Nurse'; }
          .nurse-table-awaiting td:nth-child(7)::before { content: 'Nurse status'; }
          .nurse-table-awaiting td:nth-child(8)::before { content: 'Severity'; }
          .nurse-table-awaiting td:nth-child(9)::before { content: 'Wait time'; }
          .nurse-table-awaiting td:nth-child(10)::before { content: 'Priority'; }
          .nurse-table-awaiting td:nth-child(11)::before { content: 'Action'; }
          .nurse-table-acknowledged td:nth-child(1)::before { content: 'ID'; }
          .nurse-table-acknowledged td:nth-child(2)::before { content: 'Patient'; }
          .nurse-table-acknowledged td:nth-child(3)::before { content: 'Physician'; }
          .nurse-table-acknowledged td:nth-child(4)::before { content: 'Physician ack'; }
          .nurse-table-acknowledged td:nth-child(5)::before { content: 'Specialty'; }
          .nurse-table-acknowledged td:nth-child(6)::before { content: 'Diagnosed'; }
          .nurse-table-acknowledged td:nth-child(7)::before { content: 'Nurse'; }
          .nurse-table-acknowledged td:nth-child(8)::before { content: 'Nurse ack'; }
          .nurse-table-acknowledged td:nth-child(9)::before { content: 'Cared for'; }
          .nurse-table-acknowledged td:nth-child(10)::before { content: 'Severity'; }
          .nurse-table-acknowledged td:nth-child(11)::before { content: 'Action'; }
          .nurse-responsive-table td > button { justify-self: start; }
        }
      `}</style>

      {false && (
        <div style={{
          width: '100%',
          maxWidth: 960,
          margin: '8px auto 20px',
          background: '#f8fafc',
          borderRadius: 18,
          border: '1px solid #e2e8f0',
          boxShadow: '0 8px 24px rgba(15, 23, 42, 0.08)',
          padding: 0,
          overflow: 'hidden'
        }}>
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

          <div style={{ padding: 20, background: '#ffffff' }}>
            <div style={{ marginBottom: 18 }}>
              <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#64748b', fontWeight: 800, marginBottom: 12 }}>Provider Load</div>
              {providerLoadEntries.map((provider) => (
                <div key={provider.name} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  <div style={{ fontSize: 14, color: '#1e293b', fontWeight: 600 }}>{provider.name}</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ position: 'relative', height: 10, flex: 1, background: '#e2e8f0', borderRadius: 999, overflow: 'hidden' }}>
                      <div style={{
                        width: `${provider.percent}%`,
                        height: '100%',
                        background: provider.name === 'Dr. S. Tan' ? '#ef4444' : provider.name === 'Dr. J. Wong' ? '#f59e0b' : '#2bb3c2',
                        borderRadius: 999
                      }} />
                    </div>
                    <div style={{ fontSize: 12, color: '#475569', fontWeight: 700, minWidth: 38, textAlign: 'right' }}>{provider.percent}%</div>
                  </div>
                </div>
              ))}
            </div>

            <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: 18 }}>
              <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#64748b', fontWeight: 800, marginBottom: 8 }}>Patient Queue</div>
              <div style={{ display: 'grid', gap: 6 }}>
                {queueManagerEntries.map((patient, index) => (
                  <div key={patient.id} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 12,
                    padding: '10px 0',
                    borderBottom: index === queueManagerEntries.length - 1 ? 'none' : '1px solid #f1f5f9'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1, minWidth: 0 }}>
                      <span style={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        background: patient.category === 'Red' ? '#ef4444' : patient.category === 'Orange' ? '#f59e0b' : patient.category === 'Yellow' ? '#eab308' : patient.category === 'Purple' ? '#8b5cf6' : '#22c55e',
                        display: 'inline-block',
                        boxShadow: '0 0 0 3px rgba(255,255,255,0.8)'
                      }} />
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ fontSize: 14, color: '#0f172a', fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{patient.name}</div>
                        <div style={{ fontSize: 12, color: '#64748b' }}>{patient.type}</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ fontSize: 12, color: '#475569', minWidth: 36, textAlign: 'right' }}>{patient.waitLabel}</span>
                      <span style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        minWidth: 78,
                        fontSize: 11,
                        fontWeight: 700,
                        color: patient.status === 'IN PROGRESS' ? '#0f172a' : patient.status === 'NEXT UP' ? '#b45309' : patient.status === 'AI TRIAGING' ? '#7c3aed' : '#475569',
                        background: patient.status === 'IN PROGRESS' ? '#dbeafe' : patient.status === 'NEXT UP' ? '#fef3c7' : patient.status === 'AI TRIAGING' ? '#f3e8ff' : '#e2e8f0',
                        borderRadius: 999,
                        padding: '6px 10px',
                        textTransform: 'uppercase',
                        letterSpacing: '0.04em'
                      }}>{patient.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {congestionMessage && (
              <div style={{
                marginTop: 18,
                background: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: 10,
                padding: '14px 16px',
                color: '#991b1b',
                fontSize: 14,
                fontWeight: 600,
                lineHeight: 1.45,
                display: 'flex',
                alignItems: 'flex-start',
                gap: 10
              }}>
                <span style={{ fontSize: 18 }}>⚠</span>
                <span>{congestionMessage}</span>
              </div>
            )}
          </div>
        </div>
      )}

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

                      <span style={{
                        background: colors.bg,
                        color: colors.text,
                        border: `1px solid ${colors.border}`,
                        borderRadius: 999,
                        padding: '6px 10px',
                        fontSize: '11px',
                        fontWeight: 700,
                        whiteSpace: 'nowrap'
                      }}>
                        {severity}
                      </span>
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
      {triageAllowed && showPriorityNotificationModal && priorityQueueNotification && (
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
                  if (priorityQueueNotification.encounterId) {
                    navigate(`/report/triage/${priorityQueueNotification.encounterId}`);
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

      {/* If assigned a specific role (not triage nurse), render role-specific dashboard */}
      {/* NOTE: Clinical nurses and physicians don't need to clock-in to see their dashboards */}
      {assignedRole && assignedRole !== 'TRIAGE_NURSE' ? (
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
        <>
          <div style={{ marginBottom: 24, display: 'grid', gap: 20 }}>
            <header style={{
              padding: '24px',
              background: 'linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%)',
              borderRadius: 12,
              border: '1px solid #e2e8f0',
              boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              flexWrap: 'wrap',
              gap: 20
            }}>
              <div style={{ minWidth: '280px' }}>
                <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Triage Nurse Dashboard</h1>
                <p style={{ margin: 0, fontSize: '14px', color: '#475569' }}>
                  {userName} • {assignedContext || 'ER'}
                </p>
              </div>

              <div className="nurse-dashboard-header-status" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 12, minWidth: '260px' }}>
                <div className="nurse-dashboard-status-row" style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8
                }}>
                  <button
                    type="button"
                    className="nurse-dashboard-status-pill"
                    onClick={() => navigate('/shift-management')}
                    title="Open shift management"
                    style={{
                    fontSize: '12px',
                    color: '#64748b',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                    ,
                    padding: '10px 14px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '999px',
                    background: '#ffffff',
                    cursor: 'pointer',
                    fontWeight: 700,
                    whiteSpace: 'nowrap'
                  }}>
                    <FiClock /> Shift: {currentRecord?.clockInTime ? new Date(currentRecord.clockInTime).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'Pending'} - {currentRecord?.clockOutTime ? new Date(currentRecord.clockOutTime).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : '—'}
                  </button>
                  <button
                    type="button"
                    className="nurse-dashboard-status-pill"
                    onClick={() => navigate('/shift-management')}
                    title="Open shift management"
                    style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 14px',
                    borderRadius: '999px',
                    background: '#0f172a',
                    color: '#f8fafc',
                    boxShadow: '0 1px 2px rgba(15,23,42,0.12)',
                    fontSize: '12px',
                    fontWeight: 700,
                    cursor: 'pointer',
                    border: '1px solid #0f172a',
                    whiteSpace: 'nowrap'
                  }}>
                    <FiClock size={14} />
                    {currentTime.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}
                    <span style={{ color: '#93c5fd' }}>•</span>
                    <span>{currentTime.toLocaleDateString([], { month: 'short', day: 'numeric' })}</span>
                  </button>
                </div>

                <div className="nurse-dashboard-actions" style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  flexDirection: 'row-reverse',
                  justifyContent: 'flex-end',
                  gap: '10px',
                  marginTop: '12px'
                }}>
                  {[
                
                    { icon: <FiUsers />, label: 'Assigned Patients', action: () => navigate('/triaged') },
                    { icon: <FaSearch />, label: 'EHR Search', action: () => navigate('/ehr') },
                    { icon: <FiPlusCircle size={14} />, label: 'Triage New Patient', action: handleNewIntake },
                    { icon: <FaBed />, label: 'Bed Allocation', action: () => navigate('/beds') },
                    { icon: <FiUsers />, label: 'Available Specialists', action: () => navigate('/staff-team') },
                    { icon: <FaHistory />, label: 'History Actions', action: () => navigate('/history-actions') }
                  ].map((item) => (
                    <button
                      key={item.label}
                      onClick={item.action}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '10px 14px',
                        background: '#ffffff',
                        color: '#0f172a',
                        border: '1px solid #dbeafe',
                        borderRadius: '999px',
                        cursor: 'pointer',
                        fontSize: '13px',
                        fontWeight: 700,
                        boxShadow: '0 1px 2px rgba(15,23,42,0.06)'
                      }}
                    >
                      <span style={{ fontSize: '14px' }}>{item.icon}</span>
                      <span>{item.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </header>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '16px',
              fontFamily: 'Georgia, Cambria, serif'
            }}>
              <div style={{
                padding: '22px',
                background: 'linear-gradient(145deg, #ffffff 0%, #eefbf5 100%)',
                borderRadius: '12px',
                border: '1px solid #ccefe0',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.95), 0 2px 0 #b7decf, 0 8px 16px rgba(15, 118, 87, 0.14)',
                minHeight: 132
              }}>
                <p style={{ margin: '0 0 12px 0', fontSize: '15px', color: '#14532d', fontWeight: 800, letterSpacing: '0.01em' }}>Total Beds</p>
                <h2 style={{ margin: 0, fontSize: '44px', lineHeight: 1, color: '#047857', fontWeight: 900, textShadow: '0 2px 0 rgba(255,255,255,0.9)' }}>
                  {metricsLoading ? '—' : nurseMetrics.totalBeds}
                </h2>
              </div>
              <div style={{
                padding: '22px',
                background: 'linear-gradient(145deg, #ffffff 0%, #fff1f2 100%)',
                borderRadius: '12px',
                border: '1px solid #fecdd3',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.95), 0 2px 0 #f3b5bd, 0 8px 16px rgba(190, 24, 93, 0.12)',
                minHeight: 132
              }}>
                <p style={{ margin: '0 0 12px 0', fontSize: '15px', color: '#881337', fontWeight: 800, letterSpacing: '0.01em' }}>Hospitalized Patients</p>
                <h2 style={{ margin: 0, fontSize: '44px', lineHeight: 1, color: '#e11d48', fontWeight: 900, textShadow: '0 2px 0 rgba(255,255,255,0.9)' }}>
                  {metricsLoading ? '—' : nurseMetrics.hospitalizedPatients}
                </h2>
              </div>
              <div style={{
                padding: '22px',
                background: 'linear-gradient(145deg, #ffffff 0%, #fffbeb 100%)',
                borderRadius: '12px',
                border: '1px solid #fde68a',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.95), 0 2px 0 #e9d38a, 0 8px 16px rgba(161, 98, 7, 0.14)',
                minHeight: 132
              }}>
                <p style={{ margin: '0 0 12px 0', fontSize: '15px', color: '#713f12', fontWeight: 800, letterSpacing: '0.01em' }}>Pending Triaged Patients</p>
                <h2 style={{ margin: 0, fontSize: '44px', lineHeight: 1, color: '#ca8a04', fontWeight: 900, textShadow: '0 2px 0 rgba(255,255,255,0.9)' }}>
                  {nurseMetrics.pendingTriagedPatients}
                </h2>
              </div>
            </div>
          </div>
          <div className="nurse-dashboard-columns" style={{ display: 'flex', gap: 16, minWidth: 0, width: '100%' }}>
        <div className="nurse-dashboard-main" style={{ flex: '1 1 0', minWidth: 0, width: 0 }}>
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
              <div className="nurse-responsive-table-shell">
              <table className="nurse-responsive-table nurse-table-attended" style={styles.table}>
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
            </div>
          )}

          {/* Both tables only render when clocked in */}
          {triageAllowed && (
              <>
            {/* AWAITING CLINICIAN REVIEW TABLE - Always visible */}
            <div style={{
              marginTop: '24px',
              background: '#ffffff',
              borderRadius: 12,
              border: '1px solid #e5e7eb',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              overflow: 'hidden'
            }}>
              {/* Header section with total count */}
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
                    Total: <strong>{handedOverEncounters.length}</strong> patient{handedOverEncounters.length !== 1 ? 's' : ''} waiting for acknowledgment
                  </p>
                </div>
              </div>

              {/* Empty state or table */}
              {handedOverEncounters.length === 0 ? (
                <div style={{
                  padding: '40px 28px',
                  textAlign: 'center',
                  color: '#94a3b8'
                }}>
                  <p style={{ fontSize: '14px', margin: 0 }}>No awaiting patients yet</p>
                </div>
              ) : (
                <div className="nurse-responsive-table-shell">
                  <table className="nurse-responsive-table nurse-table-awaiting" style={{
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
                        const clinicians = enc.assigned_clinicians || [];
                        const handovers = enc.clinical_handovers || [];
                      
                        // Extract physician info - try assigned_clinicians first, then fall back to triage.assigned_physician
                        let physician = clinicians.find(c => c.role === 'PHYSICIAN');
                        if (!physician?.name && enc.triage?.assigned_physician) {
                          physician = {
                            ...physician,
                            name: enc.triage.assigned_physician.name || physician?.name,
                            specialty: enc.triage.assigned_physician.specialty || physician?.specialty,
                            _id: enc.triage.assigned_physician._id || enc.triage.assigned_physician.id || physician?._id
                          };
                        }
                        const physicianHandover = handovers.find(h => 
                          physician && physician._id?.toString?.() === h.clinician_id?.toString?.()
                        );
                        const physicianStatus = physicianHandover?.handover_status || 'NOTIFIED';
                      
                        // Extract nurse info - try assigned_clinicians first, then fall back to triage.assigned_nurse
                        let nurse = clinicians.find(c => c.role === 'NURSE');
                        if (!nurse?.name && enc.triage?.assigned_nurse) {
                          nurse = {
                            ...nurse,
                            name: enc.triage.assigned_nurse.name || nurse?.name,
                            specialty: enc.triage.assigned_nurse.specialty || nurse?.specialty,
                            _id: enc.triage.assigned_nurse._id || enc.triage.assigned_nurse.id || nurse?._id
                          };
                        }
                        const nurseHandover = handovers.find(h => 
                          nurse && nurse._id?.toString?.() === h.clinician_id?.toString?.()
                        );
                        const nurseStatus = nurseHandover?.handover_status || 'NOTIFIED';
                      
                        // Get specialty from assigned physician first, fall back to aiTriageReport or assigned_specialty
                        const specialty = physician?.specialty || enc.triage?.specialty || enc.triage?.assigned_specialty || 'General';
                      
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
                      
                        const statusColors = {
                          'NOTIFIED': { color: '#f59e0b', label: 'NOTIFIED' },
                          'VIEWED': { color: '#3b82f6', label: 'VIEWED' },
                          'ACKNOWLEDGED': { color: '#10b981', label: 'ACKNOWLEDGED' },
                          'IN_PROGRESS': { color: '#8b5cf6', label: 'IN PROGRESS' },
                          'COMPLETED': { color: '#6b7280', label: 'COMPLETED' }
                        };
                      
                        const physicianStatusConfig = statusColors[physicianStatus] || statusColors['NOTIFIED'];
                        const nurseStatusConfig = statusColors[nurseStatus] || statusColors['NOTIFIED'];
                      
                        const createdAt = enc.created_at || enc.createdAt;
                        const waitTime = createdAt ? formatWaitDetailed(Date.now() - new Date(createdAt).getTime()) : 'N/A';

                        return (
                          <tr 
                            key={enc._id} 
                            style={{ 
                              borderBottom: '1px solid #e5e7eb',
                              background: idx % 2 === 0 ? '#ffffff' : '#f9fafb',
                              transition: 'all 0.2s ease'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.background = '#f0fdf4'}
                            onMouseLeave={(e) => e.currentTarget.style.background = idx % 2 === 0 ? '#ffffff' : '#f9fafb'}
                          >
                            <td style={{ padding: '14px 16px', fontSize: '12px', color: '#64748b' }}>{enc._id?.substring(0, 8)}</td>
                            <td style={{ padding: '14px 16px', fontWeight: 600, color: '#1e293b' }}>{fullName}</td>
                            <td style={{ padding: '14px 16px', color: '#475569' }}>{specialty}</td>
                            <td style={{ padding: '14px 16px', color: '#475569', fontWeight: 500 }}>{physician?.name || '—'}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                              <span style={{
                                padding: '6px 12px',
                                borderRadius: 4,
                                background: `${physicianStatusConfig.color}20`,
                                color: physicianStatusConfig.color,
                                fontSize: '11px',
                                fontWeight: 600,
                                display: 'inline-block',
                                whiteSpace: 'nowrap'
                              }}>
                                {physicianStatusConfig.label}
                              </span>
                            </td>
                            <td style={{ padding: '14px 16px', color: '#475569', fontWeight: 500 }}>{nurse?.name || '—'}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                              <span style={{
                                padding: '6px 12px',
                                borderRadius: 4,
                                background: `${nurseStatusConfig.color}20`,
                                color: nurseStatusConfig.color,
                                fontSize: '11px',
                                fontWeight: 600,
                                display: 'inline-block',
                                whiteSpace: 'nowrap'
                              }}>
                                {nurseStatusConfig.label}
                              </span>
                            </td>
                            <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                              <span style={{
                                padding: '6px 12px',
                                borderRadius: 6,
                                background: severityConfig_item.bg,
                                color: severityConfig_item.color,
                                fontSize: '12px',
                                fontWeight: 600,
                                display: 'inline-block',
                                whiteSpace: 'nowrap'
                              }}>
                                {severityConfig_item.icon} {severityConfig_item.label}
                              </span>
                            </td>
                            <td style={{ padding: '14px 16px', textAlign: 'center', fontSize: '12px', color: '#64748b' }}>{waitTime}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center', fontSize: '12px', fontWeight: 600, color: '#1e293b' }}>{enc.priority || 'N/A'}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                              <button
                                onClick={() => {
                                  navigate(`/report/triage/${enc._id}`);
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
              )}
            </div>

            {/* ACKNOWLEDGED PATIENTS TABLE - Always visible */}
            <div style={{
              marginTop: '24px',
              background: '#ffffff',
              borderRadius: 12,
              border: '1px solid #e5e7eb',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              overflow: 'hidden'
            }}>
              {/* Header section with total count */}
              <div style={{ 
                padding: '24px 28px',
                background: 'linear-gradient(135deg, #f0fdf4 0%, #dbeafe 100%)',
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
                    ✓ Acknowledged Patients
                  </h3>
                  <p style={{ 
                    margin: 0,
                    fontSize: '13px',
                    color: '#64748b'
                  }}>
                    Total: <strong>{acknowledgedEncounters.filter((enc) => {
                      const handovers = Array.isArray(enc.clinical_handovers) ? enc.clinical_handovers : [];
                      return handovers.some((handover) => ACKNOWLEDGED_HANDOVER_STATUSES.includes(handover?.handover_status));
                    }).length}</strong> acknowledged
                  </p>
                </div>
              </div>

              {/* Table or empty state */}
              {acknowledgedEncounters.length === 0 ? (
                <div style={{
                  padding: '40px 28px',
                  textAlign: 'center',
                  color: '#94a3b8'
                }}>
                  <p style={{ fontSize: '14px', margin: 0 }}>No acknowledged patients yet</p>
                </div>
              ) : (
                <div className="nurse-responsive-table-shell">
                  <table className="nurse-responsive-table nurse-table-acknowledged" style={{
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
                        <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Physician</th>
                        <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Phys. Ack</th>
                        <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Specialty</th>
                        <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Diagnosed</th>
                        <th style={{ padding: '14px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Nurse</th>
                        <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Nurse. Ack</th>
                        <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Cared For</th>
                        <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Severity</th>
                        <th style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#475569', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {acknowledgedEncounters.map((enc, idx) => {
                        const pt = enc.patient || {};
                        const fullName = pt.name || `${pt.firstName || ''} ${pt.lastName || ''}`.trim() || 'Unknown';
                        const clinicians = enc.assigned_clinicians || [];
                        const handovers = enc.clinical_handovers || [];
                      
                        // Extract physician info - try assigned_clinicians first, then fall back to triage.assigned_physician
                        let physician = clinicians.find(c => c.role === 'PHYSICIAN');
                        if (!physician?.name && enc.triage?.assigned_physician) {
                          physician = {
                            ...physician,
                            name: enc.triage.assigned_physician.name || physician?.name,
                            specialty: enc.triage.assigned_physician.specialty || physician?.specialty,
                            _id: enc.triage.assigned_physician._id || enc.triage.assigned_physician.id || physician?._id
                          };
                        }
                        const physicianHandover = handovers.find((h) => {
                          const idMatch = physician && physician._id?.toString?.() === h.clinician_id?.toString?.();
                          const nameMatch = physician?.name && h.clinician_name === physician.name;
                          const roleMatch = h.clinician_role === 'PHYSICIAN';
                          return idMatch || nameMatch || roleMatch;
                        }) || handovers.find((h) => h.clinician_role === 'PHYSICIAN');
                        const physicianAckStatus = ACKNOWLEDGED_HANDOVER_STATUSES.includes(physicianHandover?.handover_status) ? '✓' : '—';
                        const physicianSpecialty = physician?.specialty || '—';
                        const physicianDiagnosed = enc.diagnosis?.trim() ? '✓' : '—';
                      
                        // Extract nurse info - try assigned_clinicians first, then fall back to triage.assigned_nurse
                        let nurse = clinicians.find(c => c.role === 'NURSE');
                        if (!nurse?.name && enc.triage?.assigned_nurse) {
                          nurse = {
                            ...nurse,
                            name: enc.triage.assigned_nurse.name || nurse?.name,
                            specialty: enc.triage.assigned_nurse.specialty || nurse?.specialty,
                            _id: enc.triage.assigned_nurse._id || enc.triage.assigned_nurse.id || nurse?._id
                          };
                        }
                        const nurseHandover = handovers.find((h) => {
                          const idMatch = nurse && nurse._id?.toString?.() === h.clinician_id?.toString?.();
                          const nameMatch = nurse?.name && h.clinician_name === nurse.name;
                          const roleMatch = h.clinician_role === 'NURSE';
                          return idMatch || nameMatch || roleMatch;
                        }) || handovers.find((h) => h.clinician_role === 'NURSE');
                        const nurseAckStatus = ACKNOWLEDGED_HANDOVER_STATUSES.includes(nurseHandover?.handover_status) ? '✓' : '—';
                        const nurseCaredFor = nurseHandover?.notes?.includes('cared') ? '✓' : '—';
                      
                        const hasAcknowledged = handovers.some((handover) => ACKNOWLEDGED_HANDOVER_STATUSES.includes(handover?.handover_status));
                        if (!hasAcknowledged) return null;
                      
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

                        return (
                          <tr 
                            key={enc._id} 
                            style={{ 
                              borderBottom: '1px solid #e5e7eb',
                              background: idx % 2 === 0 ? '#ffffff' : '#f9fafb',
                              transition: 'all 0.2s ease'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.background = '#f0fdf4'}
                            onMouseLeave={(e) => e.currentTarget.style.background = idx % 2 === 0 ? '#ffffff' : '#f9fafb'}
                          >
                            <td style={{ padding: '14px 16px', fontSize: '12px', color: '#64748b' }}>{enc._id?.substring(0, 8)}</td>
                            <td style={{ padding: '14px 16px', fontWeight: 600, color: '#1e293b' }}>{fullName}</td>
                            <td style={{ padding: '14px 16px', color: '#475569', fontWeight: 500 }}>{physician?.name || '—'}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center', fontSize: '14px', fontWeight: 600, color: physicianAckStatus === '✓' ? '#059669' : '#64748b' }}>{physicianAckStatus}</td>
                            <td style={{ padding: '14px 16px', color: '#475569' }}>{physicianSpecialty}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center', fontSize: '14px', fontWeight: 600, color: physicianDiagnosed === '✓' ? '#059669' : '#64748b' }}>{physicianDiagnosed}</td>
                            <td style={{ padding: '14px 16px', color: '#475569', fontWeight: 500 }}>{nurse?.name || '—'}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center', fontSize: '14px', fontWeight: 600, color: nurseAckStatus === '✓' ? '#059669' : '#64748b' }}>{nurseAckStatus}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center', fontSize: '14px', fontWeight: 600, color: nurseCaredFor === '✓' ? '#059669' : '#64748b' }}>{nurseCaredFor}</td>
                            <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                              <span style={{
                                padding: '6px 12px',
                                borderRadius: 6,
                                background: severityConfig_item.bg,
                                color: severityConfig_item.color,
                                fontSize: '12px',
                                fontWeight: 600,
                                display: 'inline-block',
                                whiteSpace: 'nowrap'
                              }}>
                                {severityConfig_item.icon} {severityConfig_item.label}
                              </span>
                            </td>
                            <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                              <button
                                onClick={() => {
                                  navigate(`/report/triage/${enc._id}`);
                                }}
                                style={{
                                  padding: '8px 14px',
                                  background: '#059669',
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
                                  e.target.style.background = '#047857';
                                  e.target.style.boxShadow = '0 4px 6px rgba(5, 150, 105, 0.3)';
                                }}
                                onMouseLeave={(e) => {
                                  e.target.style.background = '#059669';
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
              )}
            </div>
            </>
          )}

        </div>

        {triageAllowed && (
          <button
            className="priority-queue-launcher"
            type="button"
            aria-label={`Open priority queue, ${pendingWithMeta.length} waiting`}
            title="Open priority queue"
            onClick={() => setShowPriorityQueueModal(true)}
          >
            🎯
            {pendingWithMeta.length > 0 && <span className="priority-queue-launcher-count">{pendingWithMeta.length > 99 ? '99+' : pendingWithMeta.length}</span>}
          </button>
        )}

        {showPriorityQueueModal && triageAllowed && (
          <div className="priority-queue-modal-backdrop" role="presentation" onMouseDown={() => setShowPriorityQueueModal(false)}>
            <section className="priority-queue-modal" role="dialog" aria-modal="true" aria-label="Priority Queue" onMouseDown={(event) => event.stopPropagation()}>
              <div style={{ padding: '18px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'linear-gradient(135deg, #fef2f2 0%, #f0f9ff 100%)', borderBottom: '1px solid #e5e7eb' }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: 17, color: '#1e293b' }}>🎯 Priority Queue</h3>
                  <div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>{pendingWithMeta.length} waiting</div>
                </div>
                <button type="button" aria-label="Close priority queue" title="Close priority queue" onClick={() => setShowPriorityQueueModal(false)} style={{ border: 0, background: 'transparent', color: '#475569', fontSize: 24, cursor: 'pointer', lineHeight: 1 }}>×</button>
              </div>
              <div className="priority-queue-modal-list">
                {pendingWithMeta.slice(0, 20).map(({ patient: p, waitMs, priority }, idx) => (
                  <div key={`${p._id || p.id}-modal-prio`} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '13px 18px', borderBottom: '1px solid #f1f5f9', background: idx % 2 === 0 ? '#fff' : '#f9fafb' }}>
                    <div style={{ minWidth: 42, padding: '8px 5px', textAlign: 'center', borderRadius: 7, background: priority.category === 'Red' ? '#fee2e2' : priority.category === 'Orange' ? '#ffedd5' : priority.category === 'Yellow' ? '#fef3c7' : '#dcfce7', color: priority.category === 'Red' ? '#991b1b' : priority.category === 'Orange' ? '#9a3412' : priority.category === 'Yellow' ? '#92400e' : '#166534', fontWeight: 800, fontSize: 11 }}>{priority.category}</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 700, color: '#1e293b', fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.patient?.name || `#${p._id || p.id}`}</div>
                      <div style={{ marginTop: 3, color: '#64748b', fontSize: 11 }}>{priority.label} • {waitMs ? formatWaitDetailed(waitMs) : 'N/A'}</div>
                      {priority.queueSignals?.length > 0 && <div style={{ marginTop: 3, color: '#b45309', fontSize: 10 }}>{priority.queueSignals.join(' • ')}</div>}
                    </div>
                    <button type="button" onClick={() => { setShowPriorityQueueModal(false); navigate(`/report/triage/${p._id || p.id}`); }} style={{ padding: '7px 9px', border: 0, borderRadius: 5, background: '#3b82f6', color: '#fff', fontSize: 11, fontWeight: 700, cursor: 'pointer' }}>View</button>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {triageAllowed ? (
          <aside className="nurse-priority-queue" style={{ 
            width: 360, 
            flexShrink: 0, 
            background: '#ffffff',
            borderLeft: '1px solid #e5e7eb',
            boxShadow: '-1px 0 3px rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ padding: '20px', background: 'linear-gradient(135deg, #fef2f2 0%, #f0f9ff 100%)', borderBottom: '1px solid #e5e7eb' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#1e293b', display: 'flex', alignItems: 'center', gap: 8 }}>🎯 Priority Queue</h3>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', background: '#f1f5f9', padding: '4px 10px', borderRadius: 4 }}>{pendingWithMeta.length} waiting</div>
              </div>
              <p style={{ margin: 0, fontSize: 12, color: '#64748b' }}>Sorted by severity & wait time</p>
            </div>
            <div style={{ padding: 16, overflowY: 'auto' }}>
              <div style={{ marginBottom: 18 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#64748b', fontWeight: 800, marginBottom: 12 }}>Provider Load</div>
                {providerLoadEntries.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 12 }}>No active provider workload</div> : providerLoadEntries.slice(0, 3).map((provider) => (
                  <div key={provider.name} style={{ display: 'grid', gridTemplateColumns: '92px 1fr 34px', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                    <div style={{ fontSize: 12, color: '#1e293b', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{provider.name}</div>
                    <div style={{ height: 9, background: '#e2e8f0', borderRadius: 999, overflow: 'hidden' }}><div style={{ width: `${provider.percent}%`, height: '100%', background: provider.percent >= 80 ? '#ef4444' : provider.percent >= 50 ? '#f59e0b' : '#2bb3c2', borderRadius: 999 }} /></div>
                    <div style={{ fontSize: 11, color: '#475569', fontWeight: 700, textAlign: 'right' }}>{provider.percent}%</div>
                  </div>
                ))}
              </div>
              <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: 16 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#64748b', fontWeight: 800, marginBottom: 8 }}>Patient Queue</div>
                {queueManagerEntries.length === 0 ? <div style={{ padding: '18px 0', color: '#94a3b8', fontSize: 13, textAlign: 'center' }}>No patients in queue</div> : queueManagerEntries.map((patient, index) => (
                  <div key={patient.id} style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '10px 0', borderBottom: index === queueManagerEntries.length - 1 ? 'none' : '1px solid #f1f5f9' }}>
                    <span style={{ width: 9, height: 9, flexShrink: 0, borderRadius: '50%', background: patient.category === 'Red' ? '#ef4444' : patient.category === 'Orange' ? '#f59e0b' : patient.category === 'Purple' ? '#8b5cf6' : '#eab308' }} />
                    <div style={{ flex: 1, minWidth: 0 }}><div style={{ fontSize: 12, color: '#0f172a', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{patient.name}</div><div style={{ fontSize: 10, color: '#64748b' }}>{patient.type}</div></div>
                    <span style={{ fontSize: 11, color: '#475569', whiteSpace: 'nowrap' }}>{patient.waitLabel}</span>
                    <span style={{ fontSize: 9, fontWeight: 700, color: patient.status === 'NEXT UP' ? '#b45309' : '#475569', background: patient.status === 'NEXT UP' ? '#fef3c7' : '#f1f5f9', borderRadius: 999, padding: '5px 7px', whiteSpace: 'nowrap' }}>{patient.status}</span>
                  </div>
                ))}
              </div>
              {congestionMessage && <div style={{ marginTop: 16, padding: 12, borderRadius: 8, background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', fontSize: 11, lineHeight: 1.4 }}>⚠ {congestionMessage}</div>}
            </div>
          </aside>
        ) : null}
      </div>
            </>
      )}
    </SidebarLayout>
  );
};

export default NurseDashboard;