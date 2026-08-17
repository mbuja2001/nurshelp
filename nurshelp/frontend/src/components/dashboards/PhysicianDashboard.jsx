import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaCalendarAlt, FaFlask, FaCapsules, FaProcedures, FaClock, FaSearch, FaTimes, FaHistory } from 'react-icons/fa';
import { decodeJwt, getRoleDisplayDepartment } from '../../utils/jwt';
import { notificationService } from '../../services/notificationService';
import './PhysicianDashboard.css';
import '../../Dashboard/DashboardTheme.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const LAB_TYPE_OPTIONS = {
  RADIOLOGY: [
    'Chest X-ray',
    'CT Scan',
    'MRI',
    'Ultrasound',
    'Echocardiogram',
    'Fluoroscopy',
    'Mammogram'
  ],
  PATHOLOGY: [
    'Complete Blood Count',
    'Basic Metabolic Panel',
    'Coagulation Profile',
    'Blood Culture',
    'Urinalysis',
    'Histopathology',
    'Cytology'
  ]
};

const SURGERY_PROCEDURE_OPTIONS = [
  'Appendectomy',
  'Cholecystectomy',
  'Hernia Repair',
  'Laparotomy',
  'Orthopaedic Fixation',
  'Cardiac Bypass',
  'Emergency Laparotomy',
  'Exploratory Surgery'
];

const SURGERY_TYPE_OPTIONS = ['ELECTIVE', 'EMERGENCY', 'URGENT'];

/**
 * PhysicianDashboard Component - REFACTORED
 * 
 * PHYSICIAN WARD MANAGEMENT
 * ════════════════════════════════════════════════════════════════════════
 * 
 * Role: Clinical decision-making, patient assessment, disposition management
 * 
 * Primary Features:
 *   • BED GRID VIEW: Visual representation of all beds in department
 *     - Occupied beds show patient info with severity color-coding
 *     - Available beds shown as empty slots
 *     - Real-time occupancy status
 * 
 *   • PATIENTS AWAITING ACKNOWLEDGMENT
 *     - Priority queue styled list similar to triage dashboard
 *     - Sorted by severity and wait time
 *     - One-click acknowledge buttons
 *     - Color-coded severity badges
 * 
 *   • DEPARTMENT OVERVIEW
 *     - Bed occupancy metrics
 *     - Total patients under care
 *     - Urgent/critical patient count
 * 
 * Design Inspiration: Triage dashboard themes
 */

const PhysicianDashboard = ({ config = {}, context = 'ER' }) => {
  // ============ STATE ============
  const [assignedPatients, setAssignedPatients] = useState([]);
  const [beds, setBeds] = useState([]);
  const [wards, setWards] = useState([]);
  const [bedMetrics, setBedMetrics] = useState({ total: 0, occupied: 0, available: 0, pendingTriage: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [acknowledging, setAcknowledging] = useState({});
  const [hospitalizing, setHospitalizing] = useState({});
  const [diagnosing, setDiagnosing] = useState({});
  const [selectedWard, setSelectedWard] = useState('ALL');
  const [selectedBedModal, setSelectedBedModal] = useState(null);
  const [bedModalData, setBedModalData] = useState(null);
  const [bedModalLoading, setBedModalLoading] = useState(false);
  const [selectedBedObject, setSelectedBedObject] = useState(null);
  const [severitySelection, setSeveritySelection] = useState(null);
  const [severitySaving, setSeveritySaving] = useState(false);
  const [showSeverityConfirmModal, setShowSeverityConfirmModal] = useState(false);
  const [pendingSeveritySelection, setPendingSeveritySelection] = useState(null);
  const [showMedicationModal, setShowMedicationModal] = useState(false);
  const [showPatientSearchModal, setShowPatientSearchModal] = useState(false);
  const [patientSearchQuery, setPatientSearchQuery] = useState('');
  const [notifications, setNotifications] = useState([]);
  const [unreadNotificationsCount, setUnreadNotificationsCount] = useState(0);
  const [showCalendarEventsModal, setShowCalendarEventsModal] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [orderedMedsCount, setOrderedMedsCount] = useState(0);
  const [showQueueOpen, setShowQueueOpen] = useState(true);
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1400);
  const [showHistoryActionsModal, setShowHistoryActionsModal] = useState(false);
  const [pharmacyInventory, setPharmacyInventory] = useState([]);
  const [medicationForm, setMedicationForm] = useState({
    medicationId: '',
    medicationName: '',
    dosage: '',
    frequency: '',
    route: 'Oral',
    indication: '',
    prescriptionNotes: '',
    administrationTime: new Date().toISOString().slice(0, 16)
  });
  const [administeingMed, setAdmisteringMed] = useState(false);

  // ============ LAB REQUEST MODAL STATE ============
  const [showLabRequestModal, setShowLabRequestModal] = useState(false);
  const [submittingLabRequest, setSubmittingLabRequest] = useState(false);
  const [labRequestForm, setLabRequestForm] = useState({
    specialty: 'RADIOLOGY',
    lab_type: '',
    description: '',
    priority: 'MEDIUM'
  });

  const [showSurgeryModal, setShowSurgeryModal] = useState(false);
  const [submittingSurgery, setSubmittingSurgery] = useState(false);
  const [surgeryForm, setSurgeryForm] = useState({
    procedure_name: '',
    surgery_type: 'ELECTIVE',
    priority: 'MEDIUM',
    rationale: '',
    scheduled_at: '',
    surgeon: '',
    surgical_team: '',
    notes: ''
  });
  const [showDischargeModal, setShowDischargeModal] = useState(false);
  const [submittingDischarge, setSubmittingDischarge] = useState(false);
  const [dischargeChecks, setDischargeChecks] = useState(null);
  const [dischargeForm, setDischargeForm] = useState({
    scheduled_at: '',
    disposition_notes: ''
  });

  const navigate = useNavigate();

  // Get specialty and role from JWT
  const [userInfo, setUserInfo] = useState({ specialty: 'General', role: 'PHYSICIAN', name: '', email: '' });
  const [currentTime, setCurrentTime] = useState(new Date());

  // ============ MEDICATION ORDER MODAL STATE ============
  const [showMedicationOrderModal, setShowMedicationOrderModal] = useState(false);
  const [submittingMedicationOrder, setSubmittingMedicationOrder] = useState(false);
  const [medicationOrderForm, setMedicationOrderForm] = useState({
    medicationId: '',
    medication_name: '',
    dosage: '',
    frequency: '',
    route: 'Oral',
    schedule: 'Schedule 3',
    indication: '',
    priority: 'MEDIUM',
    prescriptionNotes: ''
  });

  useEffect(() => {
    const token = localStorage.getItem('token');
    const decoded = decodeJwt(token);
    setUserInfo({
      specialty: decoded?.specialty || 'General',
      role: decoded?.role || 'PHYSICIAN',
      name: decoded?.name || '',
      email: decoded?.email || ''
    });
  }, []);

  useEffect(() => {
    const tick = () => setCurrentTime(new Date());
    tick();
    const intervalId = window.setInterval(tick, 1000);
    return () => window.clearInterval(intervalId);
  }, []);

  // Fetch real specialty from clocked-in staff (like the sidebar does for consults)
  useEffect(() => {
    const fetchPhysicianSpecialty = async () => {
      try {
        const token = localStorage.getItem('token');
        const decoded = decodeJwt(token);
        const clinicianId = decoded?.id;

        if (!clinicianId) {
          console.log('[PHYSICIAN DASHBOARD] No clinician ID in token, skipping specialty fetch');
          return;
        }

        console.log('[PHYSICIAN DASHBOARD] Fetching real specialty from clocked-in staff...');
        const res = await fetch(`${BACKEND_URL}/api/clinical/staff/clocked-in?role=PHYSICIAN`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });

        if (!res.ok) {
          console.error('[PHYSICIAN DASHBOARD] Failed to fetch clocked-in physicians:', res.status);
          return;
        }

        const data = await res.json();
        console.log('[PHYSICIAN DASHBOARD] Clocked-in physicians response:', data);

        if (data.success && data.physicians) {
          // Find the current user in the list
          const currentPhysician = data.physicians.find(p => p.id?.toString() === clinicianId.toString());
          
          if (currentPhysician && currentPhysician.specialty) {
            console.log('[PHYSICIAN DASHBOARD] Found real specialty:', currentPhysician.specialty);
            setUserInfo(prev => ({
              ...prev,
              specialty: currentPhysician.specialty
            }));
          } else {
            console.log('[PHYSICIAN DASHBOARD] Physician not found in clocked-in list or no specialty');
          }
        }
      } catch (err) {
        console.error('[PHYSICIAN DASHBOARD] Error fetching specialty:', err);
      }
    };

    fetchPhysicianSpecialty();
  }, []);

  // ============ RESPONSIVE WINDOW LISTENER ============
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // ============ FETCH WARDS AND BEDS ============
  const fetchWardsAndBeds = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      console.log('[PHYSICIAN DASHBOARD] Fetching wards and beds...');

      // Fetch all wards with bed status
      const wardsRes = await fetch(`${BACKEND_URL}/api/beds/wards`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!wardsRes.ok) {
        console.error('[PHYSICIAN DASHBOARD] Failed to fetch wards:', wardsRes.status);
        return;
      }

      const wardsData = await wardsRes.json();
      console.log('[PHYSICIAN DASHBOARD] Wards data:', wardsData);

      if (wardsData.success && wardsData.data) {
        setWards(wardsData.data);

        // Fetch detailed beds for the selected ward or first ward
        let wardToFetch = wardsData.data[0]?._id;
        if (selectedWard !== 'ALL' && wardsData.data.length > 0) {
          const foundWard = wardsData.data.find(w => w.department === selectedWard);
          wardToFetch = foundWard?._id || wardsData.data[0]?._id;
        }

        if (wardToFetch) {
          const wardDetailRes = await fetch(`${BACKEND_URL}/api/beds/ward/${wardToFetch}`, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json'
            }
          });

          if (wardDetailRes.ok) {
            const wardDetail = await wardDetailRes.json();
            console.log('[PHYSICIAN DASHBOARD] Ward detail:', wardDetail);

            if (wardDetail.success && wardDetail.data) {
              const bedsList = wardDetail.data.beds || [];
              const enrichedBeds = await Promise.all(bedsList.map(async (bed) => {
                const equipment = await fetchBedEquipment(bed.bedId || bed._id);
                return { ...bed, equipment };
              }));
              setBeds(enrichedBeds);

              // Update metrics
              const statusCounts = wardDetail.data.statusCounts || {};
              setBedMetrics(prev => ({
                ...prev,
                total: bedsList.length,
                occupied: statusCounts.occupied || 0,
                available: statusCounts.available || 0
              }));
              setLoading(false);
            }
          }
        }
      }
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Error fetching wards/beds:', err);
      setError('Failed to fetch ward and bed information');
      setLoading(false);
    }
  }, [selectedWard]);

  // ============ FETCH ENCOUNTER DETAILS FOR BED MODAL ============
  const fetchEncounterForBed = useCallback(async (encounterId) => {
    try {
      const token = localStorage.getItem('token');
      console.log('[PHYSICIAN DASHBOARD] Fetching encounter details:', encounterId);

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!res.ok) {
        console.error('[PHYSICIAN DASHBOARD] Failed to fetch encounter:', res.status);
        return null;
      }

      const data = await res.json();
      console.log('[PHYSICIAN DASHBOARD] Encounter data:', data);
      return data.encounter || data.case || data;
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Error fetching encounter:', err);
      return null;
    }
  }, []);
  // Fetch live bed equipment/status for a specific bed
  const fetchBedEquipment = useCallback(async (bedId) => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/beds/${encodeURIComponent(bedId)}/equipment`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!res.ok) {
        console.error('[PHYSICIAN DASHBOARD] Failed to fetch bed equipment:', res.status);
        return [];
      }

      const data = await res.json();
      return data?.equipment || [];
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Error fetching bed equipment:', err);
      return [];
    }
  }, []);

  const fetchLiveBedEquipment = useCallback(async (bedId) => {
    try {
      const equipment = await fetchBedEquipment(bedId);
      return equipment;
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Error fetching live bed equipment:', err);
      return [];
    }
  }, [fetchBedEquipment]);

  const normalizeSeverityValue = useCallback((severity) => {
    const numericSeverity = Number(severity);
    if (!Number.isFinite(numericSeverity)) return 4;
    if (numericSeverity >= 4) return 4;
    if (numericSeverity < 1) return 1;
    return numericSeverity;
  }, []);

  const buildEquipmentDisplayList = useCallback((bedLike = {}) => {
    const displayList = [];
    const addEquipment = (name) => {
      if (!name) return;
      const normalizedName = String(name).trim();
      if (!normalizedName) return;
      const alreadyExists = displayList.some((entry) => entry.toLowerCase() === normalizedName.toLowerCase());
      if (!alreadyExists) {
        displayList.push(normalizedName);
      }
    };

    const equipmentItems = Array.isArray(bedLike?.equipment) ? bedLike.equipment : [];
    equipmentItems.forEach((item) => {
      if (typeof item === 'string') {
        addEquipment(item);
      } else if (item?.name) {
        addEquipment(item.name);
      }
    });

    if (bedLike?.hasVentilator || bedLike?.ventilator) addEquipment('Mechanical Ventilator');
    if (bedLike?.hasMonitor || bedLike?.monitor) addEquipment('Cardiac Monitor');
    if (bedLike?.hasDialysis || bedLike?.dialysis) addEquipment('Dialysis Machine');
    if (bedLike?.canIsolate || bedLike?.isolationCapable || bedLike?.isolationEnabled) addEquipment('Isolation Setup');

    return displayList;
  }, []);

  const normalizeBedModalEncounter = useCallback((encounterData, fallbackBed = null) => {
    if (!encounterData) {
      return null;
    }

    const normalizedEncounter = typeof encounterData === 'object' ? encounterData : { _id: encounterData };
    const encounterId = normalizedEncounter._id || normalizedEncounter.id || fallbackBed?.currentEncounterId?._id || fallbackBed?.currentEncounterId;
    const latestProgressNote = normalizedEncounter.progression_notes?.[0] || null;
    const vitalsSources = [
      normalizedEncounter.vitals,
      latestProgressNote?.vitals_snapshot,
      normalizedEncounter.triage?.vitals,
      normalizedEncounter.triage?.aiTriageReport?.vitals,
      normalizedEncounter.ai_triage_report?.vitals
    ].filter(Boolean);
    const mergedVitals = vitalsSources.reduce((acc, source) => ({ ...acc, ...source }), {});

    const resolveVital = (aliases) => {
      for (const alias of aliases) {
        const value = mergedVitals?.[alias];
        if (value !== undefined && value !== null && value !== '') {
          return value;
        }
      }
      return null;
    };

    const systolicValue = resolveVital(['bp_systolic', 'bp_left_systolic', 'bp_right_systolic', 'systolic', 'systolicBP', 'blood_pressure_systolic']);
    const diastolicValue = resolveVital(['bp_diastolic', 'bp_left_diastolic', 'bp_right_diastolic', 'diastolic', 'diastolicBP', 'blood_pressure_diastolic']);
    const vitals = {
      bp: resolveVital(['bp']) || [systolicValue, diastolicValue].filter(Boolean).join('/') || null,
      hr: resolveVital(['hr', 'heartRate', 'heart_rate', 'pulse']) ?? null,
      temp: resolveVital(['temp', 'temperature', 'temperatureC', 'temperature_c']) ?? null,
      o2: resolveVital(['o2', 'oxygenSaturation', 'oxygen_saturation', 'spo2']) ?? null,
      resp: resolveVital(['resp', 'respiratoryRate', 'respiratory_rate', 'rr']) ?? null,
      avpu: resolveVital(['avpu']) ?? null
    };
    const normalizedSeverity = normalizeSeverityValue(normalizedEncounter.severity ?? normalizedEncounter.triage?.aiTriageReport?.esi_level ?? normalizedEncounter.triage?.esi_level ?? 4);

    return {
      ...normalizedEncounter,
      _id: encounterId,
      patientName: normalizedEncounter.patientName || normalizedEncounter.patient?.name || normalizedEncounter.firstName || normalizedEncounter.name || normalizedEncounter.fullName || normalizedEncounter.displayName || normalizedEncounter.patient?.fullName || normalizedEncounter.patient?.displayName || 'Unknown',
      age: normalizedEncounter.age ?? normalizedEncounter.patient?.age ?? null,
      diagnosis: normalizedEncounter.diagnosis || normalizedEncounter.triage?.aiTriageReport?.primary_diagnosis || normalizedEncounter.ai_triage_report?.primary_diagnosis || normalizedEncounter.triage?.primary_diagnosis || '',
      treatment_plan: normalizedEncounter.treatment_plan || normalizedEncounter.triage?.aiTriageReport?.recommended_treatment || normalizedEncounter.triage?.recommended_treatment || '',
      physician_notes: normalizedEncounter.physician_notes || normalizedEncounter.notes || '',
      treatments: normalizedEncounter.treatments || normalizedEncounter.treatment_plan?.treatments || [],
      recommendations: normalizedEncounter.recommendations || normalizedEncounter.triage?.aiTriageReport?.recommendations || normalizedEncounter.ai_triage_report?.recommendations || [],
      progression_notes: normalizedEncounter.progression_notes || [],
      medication_requests: normalizedEncounter.medication_requests || normalizedEncounter.medicationRequests || [],
      medication_administrations: normalizedEncounter.medication_administrations || normalizedEncounter.medicationAdministrations || [],
      lab_requests: normalizedEncounter.lab_requests || normalizedEncounter.labRequests || [],
      surgery_records: normalizedEncounter.surgery_records || normalizedEncounter.surgeryRecords || [],
      discharge_scheduled_at: normalizedEncounter.discharge_scheduled_at || normalizedEncounter.dischargeScheduledAt || null,
      discharge_scheduled_by: normalizedEncounter.discharge_scheduled_by || normalizedEncounter.dischargeScheduledBy || null,
      discharge_schedule_notes: normalizedEncounter.discharge_schedule_notes || normalizedEncounter.dischargeScheduleNotes || '',
      triageSummary: normalizedEncounter.triage?.aiTriageReport?.ai_summary || normalizedEncounter.ai_triage_report?.ai_summary || normalizedEncounter.triage?.summary || normalizedEncounter.summary || '',
      severity: normalizedSeverity,
      vitals,
      triage: normalizedEncounter.triage || null,
      latestProgressNote,
      status: normalizedEncounter.status || fallbackBed?.status || null
    };
  }, [normalizeSeverityValue]);

  // Keep the bed modal current with live encounter and equipment data while open
  useEffect(() => {
    if (!selectedBedModal || !bedModalData?._id) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const [liveEquipment, refreshedEncounter] = await Promise.all([
          fetchLiveBedEquipment(selectedBedObject?.bedId || selectedBedModal),
          fetchEncounterForBed(bedModalData._id)
        ]);

        if (cancelled) return;

        setSelectedBedObject(prev => prev ? ({ ...prev, equipment: liveEquipment }) : prev);
        if (refreshedEncounter) {
          setBedModalData(prev => normalizeBedModalEncounter(refreshedEncounter, prev || selectedBedObject));
        }
      } catch (err) {
        console.error('[PHYSICIAN DASHBOARD] Bed modal refresh error:', err);
      }
    };

    poll();
    const interval = setInterval(poll, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [selectedBedModal, bedModalData?._id, fetchEncounterForBed, fetchLiveBedEquipment, normalizeBedModalEncounter, selectedBedObject?.bedId]);

  const refreshBedModalData = async () => {
    if (!bedModalData?._id) return;
    try {
      const refreshed = await fetchEncounterForBed(bedModalData._id);
      if (refreshed) {
        setBedModalData(normalizeBedModalEncounter(refreshed, selectedBedObject));
      }
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Failed to refresh bed modal data:', err);
    }
  };

  useEffect(() => {
    if (!bedModalData) return;
    const severityValue = normalizeSeverityValue(bedModalData?.severity || bedModalData?.triage?.aiTriageReport?.esi_level || 4);
    setSeveritySelection(severityValue);
  }, [bedModalData, normalizeSeverityValue]);

  const handleSeveritySelectionChange = (nextValue) => {
    const normalizedValue = normalizeSeverityValue(nextValue);
    setPendingSeveritySelection(normalizedValue);
    setSeveritySelection(normalizedValue);
    setShowSeverityConfirmModal(true);
  };

  const cancelSeverityChange = () => {
    const currentSeverity = normalizeSeverityValue(bedModalData?.severity || bedModalData?.triage?.aiTriageReport?.esi_level || 4);
    setSeveritySelection(currentSeverity);
    setPendingSeveritySelection(null);
    setShowSeverityConfirmModal(false);
  };

  const confirmSeverityChange = async () => {
    if (!bedModalData?._id || pendingSeveritySelection == null) {
      setShowSeverityConfirmModal(false);
      return;
    }

    setShowSeverityConfirmModal(false);
    await handleSaveSeverity(pendingSeveritySelection);
    setPendingSeveritySelection(null);
  };

  const handleSaveSeverity = async (value = severitySelection) => {
    if (!bedModalData?._id || value == null) return;
    const normalizedValue = normalizeSeverityValue(value);
    setSeveritySaving(true);
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/encounters/${bedModalData._id}/ai-fields`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          changedFields: { severity: normalizedValue },
          nurseNotes: `Physician updated severity to ${getSeverityConfig(normalizedValue).label} via bed modal.`
        })
      });

      if (!res.ok) {
        const errorBody = await res.text();
        throw new Error(`Failed to update severity: ${res.status} ${errorBody}`);
      }

      const data = await res.json();
      const updatedEncounter = data?.encounter || data;
      setBedModalData(prev => ({
        ...prev,
        ...updatedEncounter,
        severity: normalizedValue,
        triage: {
          ...prev?.triage,
          ...updatedEncounter?.triage
        }
      }));
      console.log('[PHYSICIAN DASHBOARD] Severity updated:', normalizedValue);
      await refreshBedModalData();
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Error saving severity:', err);
    } finally {
      setSeveritySaving(false);
    }
  };

  // ============ HANDLE BED CLICK ============
  const handleBedClick = async (bed) => {
    try {
      const bedNumber = bed.bedId || bed._id;
      console.log('[PHYSICIAN DASHBOARD] Bed clicked:', bedNumber, bed);
      setSelectedBedModal(bedNumber);
      setSelectedBedObject(bed);
      setBedModalLoading(true);
      console.log('[PHYSICIAN DASHBOARD] Modal should now be visible for bed:', bedNumber);

      const liveEquipment = await fetchLiveBedEquipment(bed.bedId || bed._id);
      const enrichedBed = {
        ...bed,
        equipment: liveEquipment,
        bedType: bed.bedType || bed.bed_type || 'Standard',
        acuityLevel: bed.acuityLevel || bed.acuity_level || 'Moderate',
        canIsolate: Boolean(bed.canIsolate || bed.isolationCapable || bed.isolationEnabled),
        hasMonitor: Boolean(bed.hasMonitor || bed.monitor),
        hasVentilator: Boolean(bed.hasVentilator || bed.ventilator),
        hasDialysis: Boolean(bed.hasDialysis || bed.dialysis),
        rawBed: bed
      };

      setSelectedBedObject(enrichedBed);

      if (bed.currentEncounterId) {
        const encounterId = typeof bed.currentEncounterId === 'object'
          ? (bed.currentEncounterId._id || bed.currentEncounterId.id)
          : bed.currentEncounterId;

        if (encounterId) {
          const encounterData = await fetchEncounterForBed(encounterId);
          const fallbackEncounter = typeof bed.currentEncounterId === 'object' ? bed.currentEncounterId : null;
          setBedModalData(normalizeBedModalEncounter(encounterData || fallbackEncounter, enrichedBed));
        } else {
          setBedModalData(normalizeBedModalEncounter(bed.currentEncounterId, enrichedBed));
        }
      } else {
        console.log('[PHYSICIAN DASHBOARD] No encounter data for bed:', bedNumber, '- setting empty modal');
        setBedModalData(null);
      }
      setBedModalLoading(false);
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Error handling bed click:', err);
      setBedModalData(null);
      setBedModalLoading(false);
    }
  };

  // ============ FETCH ASSIGNED PATIENTS ============
  const fetchAssignedPatients = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      const clinicianId = decoded?.id;

      console.log('[PHYSICIAN DASHBOARD] Fetching assigned patients...');
      console.log('[PHYSICIAN DASHBOARD] Clinician ID:', clinicianId);
      console.log('[PHYSICIAN DASHBOARD] Context:', context);

      if (!clinicianId) {
        console.error('[PHYSICIAN DASHBOARD] Could not identify clinician from token');
        setError('Could not identify clinician');
        setLoading(false);
        return;
      }

      const url = `${BACKEND_URL}/api/encounters/clinician/${clinicianId}`;
      console.log('[PHYSICIAN DASHBOARD] Fetching from:', url);

      const res = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      console.log('[PHYSICIAN DASHBOARD] Response status:', res.status);

      if (!res.ok) {
        throw new Error(`Failed to fetch assigned patients: ${res.status}`);
      }

      const data = await res.json();
      console.log('[PHYSICIAN DASHBOARD] Response data:', data);
      console.log('[PHYSICIAN DASHBOARD] Cases received:', data.cases?.length || 0);

      if (data.cases) {
        data.cases.forEach((pt, idx) => {
          console.log(`[PHYSICIAN DASHBOARD] Case ${idx}:`, {
            id: pt._id?.substring(0, 8),
            patient: pt.patientName || pt.firstName,
            status: pt.handover?.handover_status,
            severity: pt.triage?.aiTriageReport?.esi_level
          });
        });
      }

      setAssignedPatients(data.cases || []);

      // Mock bed data - in production would come from backend
      const totalBeds = 12;
      const hospitalized = (data.cases || []).filter(p => p.status === 'HOSPITALIZED').length;
      const pendingTriage = (data.cases || []).filter(p => {
        const status = String(p.status || '').toUpperCase();
        return p.status !== 'HOSPITALIZED' && (!status || ['PENDING', 'TRIAGED', 'WAITING', 'QUEUED'].includes(status));
      }).length;
      setBedMetrics({
        total: totalBeds,
        occupied: hospitalized,
        available: totalBeds - hospitalized,
        pendingTriage
      });

      setError(null);
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Fetch error:', err);
      console.error('[PHYSICIAN DASHBOARD] Error message:', err.message);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [context]);

  const loadUnreadNotifications = useCallback(async () => {
    try {
      setHistoryLoading(true);
      const response = await notificationService.getUnreadNotifications(12);
      const items = Array.isArray(response.notifications) ? response.notifications : (Array.isArray(response.data) ? response.data : []);
      setNotifications(items);
      setUnreadNotificationsCount(typeof response.count === 'number' ? response.count : items.length);
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Failed to load notifications:', err);
      setNotifications([]);
      setUnreadNotificationsCount(0);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const handleHistoryActions = async () => {
    if (!showHistoryActionsModal) {
      await loadUnreadNotifications();
    }
    setShowHistoryActionsModal((prev) => !prev);
  };

  const handleMarkNotificationRead = async (notificationId) => {
    if (!notificationId) return;
    try {
      await notificationService.markAsRead(notificationId);
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Failed to mark notification read:', err);
    } finally {
      await loadUnreadNotifications();
    }
  };

  const handleOpenCalendarEvents = () => {
    setShowCalendarEventsModal(true);
  };

  const handleOpenSurgeryModal = (encounterId, patientName) => {
    if (!encounterId) {
      alert('Unable to schedule surgery: missing encounter identifier');
      return;
    }

    const baseEncounter = bedModalData || { _id: encounterId, patientName: patientName || 'Patient' };
    setBedModalData(baseEncounter);
    setSurgeryForm({
      procedure_name: '',
      surgery_type: 'ELECTIVE',
      priority: 'MEDIUM',
      rationale: baseEncounter.diagnosis || baseEncounter.triageSummary || `Surgery request for ${patientName || 'patient'}`,
      scheduled_at: '',
      surgeon: userInfo.name || '',
      surgical_team: '',
      notes: ''
    });
    setShowSurgeryModal(true);
  };

  const handleRequestLabs = (encounterId, patientName) => {
    if (!encounterId) {
      alert('Unable to request labs: missing encounter identifier');
      return;
    }

    navigate(`/request-lab/${encounterId}`, { state: { encounter: bedModalData || { _id: encounterId, patientName: patientName || 'Patient' } } });
  };

  const handleAcknowledge = async (encounterId) => {
    try {
      console.log('[PHYSICIAN ACKNOWLEDGE] Starting acknowledgment for encounter:', encounterId);
      setAcknowledging(prev => ({ ...prev, [encounterId]: true }));
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      
      console.log('[PHYSICIAN ACKNOWLEDGE] Clinician ID:', decoded?.id);
      console.log('[PHYSICIAN ACKNOWLEDGE] Clinician Name:', decoded?.name);

      const body = {
        acknowledgedNotes: 'Patient acknowledged by physician'
      };

      console.log('[PHYSICIAN ACKNOWLEDGE] Request body:', body);

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover/acknowledge`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });

      console.log('[PHYSICIAN ACKNOWLEDGE] Response status:', res.status);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        console.error('[PHYSICIAN ACKNOWLEDGE] Error response:', errData);
        throw new Error(errData.message || 'Failed to acknowledge patient');
      }

      const responseData = await res.json();
      console.log('[PHYSICIAN ACKNOWLEDGE] Success! Response:', responseData);
      console.log('[PHYSICIAN ACKNOWLEDGE] Updated encounter status:', responseData.encounter?.status);
      console.log('[PHYSICIAN ACKNOWLEDGE] Handover status:', responseData.handover?.handover_status);

      // Refresh the patient list
      console.log('[PHYSICIAN ACKNOWLEDGE] Refreshing patient list...');
      await fetchAssignedPatients();
      
      console.log('[PHYSICIAN ACKNOWLEDGE] ✓ Acknowledgment complete and list refreshed');
    } catch (err) {
      console.error('[PHYSICIAN ACKNOWLEDGE] Error:', err);
      console.error('[PHYSICIAN ACKNOWLEDGE] Error message:', err.message);
      alert(`Error: ${err.message}`);
    } finally {
      setAcknowledging(prev => ({ ...prev, [encounterId]: false }));
    }
  };

  const isEncounterHospitalized = (patientLike) => Boolean(
    patientLike?.isHospitalized ||
    patientLike?.status === 'HOSPITALIZED' ||
    patientLike?.disposition === 'HOSPITALIZED'
  );

  const isPatientAcknowledged = (patientLike) => Boolean(
    patientLike?.handover?.handover_status === 'ACKNOWLEDGED' ||
    patientLike?.handover_status === 'ACKNOWLEDGED' ||
    patientLike?.status === 'ACKNOWLEDGED' ||
    patientLike?.status === 'DIAGNOSED' ||
    patientLike?.status === 'HOSPITALIZED' ||
    patientLike?.status === 'DISCHARGED' ||
    Array.isArray(patientLike?.clinical_handovers) && patientLike.clinical_handovers.some((handover) => ['ACKNOWLEDGED', 'IN_PROGRESS', 'COMPLETED'].includes(handover?.handover_status))
  );

  // ============ HOSPITALIZE PATIENT ============
  const handleHospitalize = async (patientLike, patientName) => {
    try {
      const encounterId = patientLike?._id || patientLike?.id || patientLike?.encounterId || patientLike?.encounter_id;
      const encounterPayload = patientLike && typeof patientLike === 'object' ? patientLike : bedModalData || null;

      if (!encounterId) {
        alert('Unable to admit patient: missing encounter identifier');
        return;
      }

      if (isEncounterHospitalized(encounterPayload)) {
        alert('This patient is already hospitalized. Use the inpatient workflow for ongoing care.');
        return;
      }

      console.log('[PHYSICIAN HOSPITALIZE] Opening ward and bed assignment for:', patientName);
      setHospitalizing(prev => ({ ...prev, [encounterId]: true }));
      navigate(`/assign-ward-bed/${encounterId}`, { state: { encounter: encounterPayload } });
    } catch (err) {
      console.error('[PHYSICIAN HOSPITALIZE] Error:', err);
      alert(`Error: ${err.message}`);
    } finally {
      setHospitalizing(prev => ({ ...prev, [patientLike?._id || patientLike?.id || patientLike?.encounterId || patientLike?.encounter_id]: false }));
    }
  };

  // ============ VIEW TRIAGE REPORT ============
  const handleViewTriageReport = (patient) => {
    const encounterId = patient?._id || patient?.id || patient?.encounterId || patient?.encounter_id;
    if (!encounterId) {
      console.error('[PHYSICIAN TRIAGE REPORT] Missing encounter id for patient:', patient);
      alert('Unable to open triage report: missing encounter identifier');
      return;
    }

    console.log('[PHYSICIAN TRIAGE REPORT] Navigating to physician triage report for:', encounterId);
    navigate(`/report/doctor/${encounterId}`);
  };

  // ============ VIEW INPATIENT REPORT ============
  const handleViewInpatientReport = (patient) => {
    const encounterId = patient?._id || patient?.id || patient?.encounterId || patient?.encounter_id;
    if (!encounterId) {
      console.error('[PHYSICIAN INPATIENT REPORT] Missing encounter id for patient:', patient);
      alert('Unable to open inpatient report: missing encounter identifier');
      return;
    }

    console.log('[PHYSICIAN INPATIENT REPORT] Navigating to inpatient report for:', encounterId);
    navigate(`/report/inpatient/${encounterId}`);
  };

  // ============ TRANSFER SPECIALTY CARE ============
  const handleTransferSpecialtyCare = async (encounterId, patientName) => {
    try {
      if (!encounterId) {
        alert('Missing encounter identifier for specialty transfer');
        return;
      }

      const specialty = window.prompt('Transfer care to which specialty?', bedModalData?.triage?.assigned_specialty || 'Specialty Care');
      if (!specialty) {
        console.log('[PHYSICIAN TRANSFER] Specialty transfer cancelled by user');
        return;
      }

      console.log('[PHYSICIAN TRANSFER] Transferring specialty care for encounter:', encounterId, 'specialty:', specialty);
      const token = localStorage.getItem('token');

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/assign`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ specialty })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        console.error('[PHYSICIAN TRANSFER] Error response:', errData);
        throw new Error(errData.message || 'Failed to transfer specialty care');
      }

      const responseData = await res.json();
      console.log('[PHYSICIAN TRANSFER] Success:', responseData);
      alert(`✓ ${patientName || 'Patient'} transferred to ${specialty} care`);
      await fetchAssignedPatients();
      await refreshBedModalData();
    } catch (err) {
      console.error('[PHYSICIAN TRANSFER] Error:', err);
      alert(`Error: ${err.message}`);
    }
  };

  // ============ DISCHARGE PATIENT ============
  const handleDischarge = async (encounterId, patientName) => {
    if (!encounterId) {
      alert('Unable to schedule discharge: missing encounter identifier');
      return;
    }

    setDischargeChecks(null);
    setDischargeForm({
      scheduled_at: new Date().toISOString().slice(0, 16),
      disposition_notes: `Discharge scheduled by ${userInfo.name || 'physician'}`
    });
    setShowDischargeModal(true);
  };

  const handleSubmitDischarge = async () => {
    if (!bedModalData?._id) {
      alert('Unable to schedule discharge: missing encounter data');
      return;
    }

    if (!dischargeForm.scheduled_at) {
      alert('Please select a discharge date and time');
      return;
    }

    setSubmittingDischarge(true);
    setDischargeChecks(null);
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`${BACKEND_URL}/api/encounters/${bedModalData._id}/disposition`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          disposition: 'DISCHARGED',
          disposition_notes: dischargeForm.disposition_notes || `Discharge scheduled by ${userInfo.name || 'physician'}`,
          scheduled_at: dischargeForm.scheduled_at
        })
      });

      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        console.error('[PHYSICIAN DISCHARGE] Error response:', payload);
        setDischargeChecks({
          blocked: true,
          message: payload.message || 'Failed to schedule discharge',
          details: payload.details || null,
          warnings: payload.warnings || null
        });
        throw new Error(payload.message || 'Failed to schedule discharge');
      }

      setDischargeChecks({
        blocked: false,
        message: payload.message || `Discharge scheduled for ${new Date(dischargeForm.scheduled_at).toLocaleString()}`,
        warnings: payload.warnings || null
      });

      alert(`✓ Discharge scheduled for ${new Date(dischargeForm.scheduled_at).toLocaleString()}`);
      setShowDischargeModal(false);
      setDischargeForm({ scheduled_at: '', disposition_notes: '' });
      await fetchAssignedPatients();
      await fetchWardsAndBeds();
      await refreshBedModalData();
    } catch (err) {
      console.error('[PHYSICIAN DISCHARGE] Error:', err);
      alert(`Error: ${err.message}`);
    } finally {
      setSubmittingDischarge(false);
    }
  };

  // ============ DIAGNOSE PATIENT (PHYSICIAN) ============
  const handleDiagnose = (encounterId, encounterData = null) => {
    if (!encounterId) {
      alert('Unable to open diagnosis page: missing encounter identifier');
      return;
    }

    const patientIsAcknowledged = isPatientAcknowledged(encounterData);

    if (!patientIsAcknowledged) {
      alert('Patient must be acknowledged by a clinician before diagnosis can begin.');
      return;
    }

    console.log('[PHYSICIAN DIAGNOSE] Navigating to diagnosis page for:', encounterId);
    navigate(`/diagnose/${encounterId}`, { state: { encounter: encounterData } });
  };

  const handleOrderMedication = async () => {
    if (!bedModalData?._id) {
      alert('Unable to order medication: missing patient data');
      return;
    }

    navigate(`/order-medication/${bedModalData._id}`, { state: { encounter: bedModalData } });
  };

  const handleSubmitMedicationOrder = async () => {
    if (!bedModalData?._id) {
      alert('Unable to order medication: missing patient encounter');
      return;
    }

    const decoded = decodeJwt(localStorage.getItem('token'));
    const physicianId = decoded?.id;
    const patientId = bedModalData?.patient?.id_number || bedModalData?.patient?.id || bedModalData?.patient?._id || bedModalData?._id;
    const selectedMedication = pharmacyInventory.find((item) => item.id === medicationOrderForm.medicationId || item.name === medicationOrderForm.medication_name);
    const medicationName = selectedMedication?.name || medicationOrderForm.medication_name;

    if (!medicationName || !physicianId || !patientId) {
      alert('Please select a medication and confirm the patient/physician context is available');
      return;
    }

    setSubmittingMedicationOrder(true);
    try {
      const token = localStorage.getItem('token');
      const prescriptionResponse = await fetch(`${BACKEND_URL}/api/prescriptions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          encounterId: bedModalData._id,
          patientId,
          physicianId,
          medicationId: selectedMedication?.id || medicationOrderForm.medicationId || null,
          medicationName,
          prescriber: {
            name: userInfo.name,
            email: userInfo.email,
            role: userInfo.role,
            specialty: userInfo.specialty
          },
          patient_details: {
            name: bedModalData?.patientName || bedModalData?.patient?.name || 'Patient',
            id_number: bedModalData?.patient?.id_number || patientId,
            age: bedModalData?.age || bedModalData?.patient?.age || null,
            address: bedModalData?.patient?.address || 'N/A',
            medical_aid: bedModalData?.patient?.medical_aid || 'N/A',
            medical_aid_number: bedModalData?.patient?.medical_aid_number || 'N/A'
          },
          items: [
            {
              medicationId: selectedMedication?.id || medicationOrderForm.medicationId || null,
              medication_name: medicationName,
              dosage: medicationOrderForm.dosage,
              frequency: medicationOrderForm.frequency,
              route: medicationOrderForm.route,
              schedule: medicationOrderForm.schedule,
              indication: medicationOrderForm.indication,
              quantity: 1,
              unit: 'Tablets'
            }
          ],
          substitution_permitted: true,
          special_instructions: medicationOrderForm.prescriptionNotes || medicationOrderForm.indication || ''
        })
      });

      if (!prescriptionResponse.ok) {
        const errData = await prescriptionResponse.json().catch(() => ({}));
        throw new Error(errData.error || errData.message || 'Failed to create prescription');
      }

      const prescriptionData = await prescriptionResponse.json();
      const prescriptionId = prescriptionData.prescription?._id || prescriptionData._id;
      const prescriptionNumber = prescriptionData.prescription_number;

      alert(`✓ Prescription ${prescriptionNumber} created successfully`);
      setShowMedicationOrderModal(false);
      setSelectedBedModal(null);
      setMedicationOrderForm({
        medicationId: '',
        medication_name: '',
        dosage: '',
        frequency: '',
        route: 'Oral',
        schedule: 'Schedule 3',
        indication: '',
        priority: 'MEDIUM',
        prescriptionNotes: ''
      });

      navigate(`/prescription/${prescriptionId}`);
    } catch (err) {
      alert('Failed to create prescription: ' + (err.message || 'Unknown error'));
    } finally {
      setSubmittingMedicationOrder(false);
    }
  };

  const handleSubmitLabRequest = async () => {
    if (!labRequestForm.lab_type || !labRequestForm.description || !bedModalData?._id) {
      alert('Please select lab type and enter indication');
      return;
    }

    setSubmittingLabRequest(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${BACKEND_URL}/api/lab-requests/${bedModalData._id}/create`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          specialty: labRequestForm.specialty,
          lab_type: labRequestForm.lab_type,
          description: labRequestForm.description,
          priority: labRequestForm.priority,
          requested_by: userInfo.name || 'Physician'
        })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.message || 'Failed to request lab');
      }

      const data = await response.json();
      alert(`✓ Lab request submitted: ${labRequestForm.lab_type} (${labRequestForm.specialty})`);
      console.log('[PHYSICIAN LAB REQUEST] Success:', data);
      
      await refreshBedModalData();
      setShowLabRequestModal(false);
      setLabRequestForm({
        specialty: 'RADIOLOGY',
        lab_type: '',
        description: '',
        priority: 'MEDIUM'
      });
    } catch (err) {
      console.error('[PHYSICIAN LAB REQUEST] Error:', err);
      alert(`Error requesting lab: ${err.message}`);
    } finally {
      setSubmittingLabRequest(false);
    }
  };

  const handleSubmitSurgery = async () => {
    if (!bedModalData?._id || !surgeryForm.procedure_name || !surgeryForm.scheduled_at) {
      alert('Please enter procedure name and scheduled date/time');
      return;
    }

    setSubmittingSurgery(true);
    try {
      const token = localStorage.getItem('token');
      const payload = {
        procedure_name: surgeryForm.procedure_name,
        surgery_type: surgeryForm.surgery_type,
        priority: surgeryForm.priority,
        rationale: surgeryForm.rationale,
        scheduled_at: surgeryForm.scheduled_at,
        surgeon: surgeryForm.surgeon,
        surgical_team: surgeryForm.surgical_team.split(',').map((item) => item.trim()).filter(Boolean),
        notes: surgeryForm.notes
      };

      const response = await fetch(`${BACKEND_URL}/api/encounters/${bedModalData._id}/surgery`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.message || 'Failed to schedule surgery');
      }

      const data = await response.json();
      alert(`✓ Surgery requested: ${surgeryForm.procedure_name}`);
      console.log('[PHYSICIAN SURGERY REQUEST] Success:', data);
      setShowSurgeryModal(false);
      setSurgeryForm({
        procedure_name: '',
        surgery_type: 'ELECTIVE',
        priority: 'MEDIUM',
        rationale: '',
        scheduled_at: '',
        surgeon: userInfo.name || '',
        surgical_team: '',
        notes: ''
      });
      await refreshBedModalData();
    } catch (err) {
      console.error('[PHYSICIAN SURGERY REQUEST] Error:', err);
      alert(`Error scheduling surgery: ${err.message}`);
    } finally {
      setSubmittingSurgery(false);
    }
  };

  const handleAdministerCare = (encounterId, patientName) => {
    if (!encounterId) {
      alert('Unable to administer care: missing encounter identifier');
      return;
    }

    console.log('[PHYSICIAN ADMINISTER CARE] Opening care details for:', encounterId);
    alert(`Care actions page opened for ${patientName || 'patient'}.
Update medications, procedures, or notes there.`);
    navigate(`/summary/${encounterId}`);
  };

  // ============ MEDICATION ADMINISTRATION ============
  const fetchPharmacyInventory = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      const decoded = decodeJwt(token);
      
      const response = await fetch(
        `${BACKEND_URL}/api/pharmacy/inventory?hospitalId=${decoded?.hospitalId}`,
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        const inventoryArray = Array.isArray(data.data) ? data.data : [];
        
        const transformed = inventoryArray.map(inv => ({
          id: inv._id || inv.medicationId,
          name: inv.medicationName || inv.name,
          strength: inv.strength || '',
          dosageForm: inv.form || inv.dosageForm || 'Tablet',
          quantity: inv.quantity || 0,
          minStock: inv.minStockLevel || 10,
          route: inv.route || 'PO'
        })).filter(med => med.quantity > 0);
        
        setPharmacyInventory(transformed);
      }
    } catch (err) {
      console.error('[PHYSICIAN] Error fetching pharmacy inventory:', err);
    }
  }, []);

  const handleOpenMedicationRequest = async () => {
    if (!bedModalData?._id) {
      alert('Unable to open medication workflow: missing patient encounter');
      return;
    }

    console.log('[PHYSICIAN MEDICATION] Opening diagnosis workflow for medication request:', bedModalData._id);
    navigate(`/diagnose/${bedModalData._id}`, { state: { encounter: bedModalData } });
  };

  const handleAdministerMedication = async () => {
    if (!medicationForm.medicationName || !medicationForm.dosage || !bedModalData?._id) {
      alert('Please select medication, dosage, and ensure patient is loaded');
      return;
    }

    setAdmisteringMed(true);

    try {
      const token = localStorage.getItem('token');
      const requestPayload = {
        diagnosis: bedModalData?.diagnosis || '',
        treatment_plan: bedModalData?.treatment_plan || '',
        notes: bedModalData?.physician_notes || '',
        medication_requests: [{
          medication_name: medicationForm.medicationName,
          dosage: medicationForm.dosage,
          frequency: medicationForm.frequency || 'As prescribed',
          route: medicationForm.route || 'Oral',
          indication: medicationForm.indication || 'Requested for active treatment',
          requested_by: userInfo.name || 'Physician'
        }]
      };

      const response = await fetch(`${BACKEND_URL}/api/encounters/${bedModalData._id}/inpatient-report`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestPayload)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.message || 'Failed to request pharmacist review');
      }

      alert(`✓ Pharmacist review requested: ${medicationForm.medicationName}`);
      await refreshBedModalData();
      setShowMedicationModal(false);
    } catch (err) {
      console.error('[PHYSICIAN MEDICATION] Error:', err);
      alert(`Error: ${err.message}`);
    } finally {
      setAdmisteringMed(false);
    }
  };

  // ============ SURGERY ADMINISTRATION ============
  const handleAdministerSurgery = (encounterId, patientName) => {
    if (!encounterId) {
      alert('Unable to open surgery workflow: missing encounter identifier');
      return;
    }

    console.log('[PHYSICIAN SURGERY] Navigating to surgery page for:', encounterId);
    navigate(`/surgery/${encounterId}`, { state: { encounter: bedModalData } });
  };

  const goToDashboardToolsSection = (section) => {
    navigate(`/dashboard-tools#${section}`);
  };

  // ============ CARE FOR PATIENT (CLINICAL NURSE) ============
  const handleCareFor = async (encounterId, patientName) => {
    try {
      console.log('[NURSE CARE FOR] Marking patient as cared for:', encounterId);
      setDiagnosing(prev => ({ ...prev, [encounterId]: true }));
      const token = localStorage.getItem('token');

      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover/acknowledge`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          acknowledgedNotes: 'Care provided to patient'
        })
      });

      console.log('[NURSE CARE FOR] Response status:', res.status);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        console.error('[NURSE CARE FOR] Error response:', errData);
        throw new Error(errData.message || 'Failed to mark patient as cared for');
      }

      const responseData = await res.json();
      console.log('[NURSE CARE FOR] ✓ Patient marked as cared for');

      await fetchAssignedPatients();
      alert(`✓ ${patientName} - care recorded`);
    } catch (err) {
      console.error('[NURSE CARE FOR] Error:', err.message);
      alert(`Error: ${err.message}`);
    } finally {
      setDiagnosing(prev => ({ ...prev, [encounterId]: false }));
    }
  };

  // ============ EFFECTS ============
  // Fetch wards and beds on component mount and when ward selection changes
  useEffect(() => {
    fetchWardsAndBeds();
  }, [selectedWard, fetchWardsAndBeds]);

  useEffect(() => {
    fetchAssignedPatients();
    fetchWardsAndBeds();
    const interval = setInterval(() => {
      fetchAssignedPatients();
      fetchWardsAndBeds();
    }, 15000); // Refresh every 15s
    return () => clearInterval(interval);
  }, [fetchAssignedPatients, fetchWardsAndBeds]);

  const loadOrderedMedsCount = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      const res = await fetch(`${BACKEND_URL}/api/prescriptions/ordered-meds?countOnly=true`, {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        cache: 'no-store'
      });

      const data = await res.json().catch(() => ({}));
      if (res.ok && typeof data.count === 'number') {
        setOrderedMedsCount(data.count);
      } else {
        console.warn('[PHYSICIAN DASHBOARD] Failed to load ordered meds count', data);
        setOrderedMedsCount(0);
      }
    } catch (err) {
      console.error('[PHYSICIAN DASHBOARD] Error loading ordered meds count:', err);
      setOrderedMedsCount(0);
    }
  };

  useEffect(() => {
    loadOrderedMedsCount();
    const interval = setInterval(loadOrderedMedsCount, 15000);
    return () => clearInterval(interval);
  }, []);

  // ============ HELPER FUNCTIONS ============
  const getSeverityConfig = (severity) => {
    const normalizedSeverity = normalizeSeverityValue(severity);
    const severityMap = {
      1: { label: 'CRITICAL', color: '#991b1b', bg: '#fee2e2', icon: '🔴' },
      2: { label: 'HIGH', color: '#9a3412', bg: '#ffedd5', icon: '🟠' },
      3: { label: 'MODERATE', color: '#92400e', bg: '#fef3c7', icon: '🟡' },
      4: { label: 'LOW', color: '#166534', bg: '#dcfce7', icon: '🟢' }
    };
    return severityMap[normalizedSeverity] || severityMap[4];
  };

  const getStatusConfig = (status) => {
    const statusMap = {
      'NOTIFIED': { color: '#f59e0b', label: '🔔 NOTIFIED' },
      'VIEWED': { color: '#3b82f6', label: '👁️ VIEWED' },
      'ACKNOWLEDGED': { color: '#10b981', label: '✓ ACKNOWLEDGED' },
      'IN_PROGRESS': { color: '#8b5cf6', label: '⏱️ IN PROGRESS' },
      'COMPLETED': { color: '#6b7280', label: '✓ COMPLETED' },
      'DIAGNOSED': { color: '#9333ea', label: '🩺 DIAGNOSED' },
      'HOSPITALIZED': { color: '#1d4ed8', label: '🏥 HOSPITALIZED' },
      'DISCHARGED': { color: '#047857', label: '🚪 DISCHARGED' }
    };
    return statusMap[status] || statusMap['NOTIFIED'];
  };

  const pendingWithMeta = assignedPatients
    .filter(p => p.handover?.handover_status !== 'ACKNOWLEDGED' && p.handover?.handover_status !== 'COMPLETED')
    .map((patient) => {
      const severity = patient.triage?.aiTriageReport?.esi_level || patient.severity || 4;
      const waitMs = patient.createdAt ? Date.now() - new Date(patient.createdAt).getTime() : 0;
      return {
        patient,
        waitMs,
        sev: severity,
        priority: { label: getSeverityConfig(severity).label },
        attended: false
      };
    })
    .sort((a, b) => {
      if (a.sev !== b.sev) return a.sev - b.sev;
      return a.waitMs - b.waitMs;
    });

  const formatWaitTime = (createdAt, referenceTime = new Date()) => {
    if (!createdAt) return 'N/A';
    const diff = Math.floor((referenceTime.getTime() - new Date(createdAt).getTime()) / 60000);
    if (diff < 1) return 'Now';
    if (diff < 60) return `${diff}m`;
    return `${Math.floor(diff / 60)}h ${diff % 60}m`;
  };

  const getBedSearchResults = (query = patientSearchQuery) => {
    const normalizedQuery = (query || '').trim().toLowerCase();

    if (!beds.length) return [];

    return beds
      .map((bed) => {
        const encounter = bed.currentEncounterId;
        const patientName = encounter?.patientName || encounter?.firstName || encounter?.patient?.name || encounter?.name || '';
        const patientSurname = encounter?.patient?.surname || encounter?.surname || '';
        const patientGender = encounter?.patient?.gender || encounter?.gender || '';
        const patientAge = encounter?.age ?? encounter?.patient?.age ?? '';
        const diagnosis = encounter?.diagnosis || encounter?.triageSummary || encounter?.physician_notes || encounter?.treatment_plan || '';
        const summary = encounter?.patient?.symptoms || encounter?.patient?.history || encounter?.triage?.summary || '';
        const bedLabel = `bed ${bed.bedId || bed._id || ''}`;
        const searchableText = [patientName, patientSurname, patientGender, patientAge, diagnosis, summary, bedLabel].filter(Boolean).join(' ').toLowerCase();

        let score = 0;
        if (!normalizedQuery) {
          score = 1;
        } else {
          const queryTerms = normalizedQuery.split(/\s+/).filter(Boolean);
          queryTerms.forEach((term) => {
            if (searchableText.includes(term)) score += 3;
            if (patientName.toLowerCase().includes(term)) score += 4;
            if (patientSurname.toLowerCase().includes(term)) score += 4;
            if (diagnosis.toLowerCase().includes(term)) score += 3;
            if (summary.toLowerCase().includes(term)) score += 2;
            if (bedLabel.includes(term)) score += 2;
          });
        }

        return { bed, encounter, score, patientName, diagnosis, summary, bedLabel };
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
  };

  // ============ RENDER ============
  const severityOptions = [1, 2, 3, 4];
  const latestProgressNote = bedModalData?.progression_notes?.[0];
  const modalVitals = bedModalData?.vitals || latestProgressNote?.vitals_snapshot || null;
  const equipmentSummary = buildEquipmentDisplayList(selectedBedObject || {});
  const modalEquipmentCount = equipmentSummary.length;
  const normalizedEquipmentSummary = [...new Set(equipmentSummary)].slice(0, 4).join(', ') || 'Standard setup';
  const clinicalSummary = bedModalData?.diagnosis || bedModalData?.triageSummary || bedModalData?.physician_notes || bedModalData?.treatment_plan || 'No active diagnosis documented yet.';

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      gap: '24px',
      padding: '24px',
      background: '#f8fafc',
      minHeight: '100vh'
    }}>
      {/* ===== HEADER ===== */}
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '24px',
        background: 'linear-gradient(135deg, #fef2f2 0%, #f0f9ff 100%)',
        borderRadius: '12px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
      }}>
        <div>
          <h1 style={{ 
            margin: '0 0 4px 0',
            fontSize: '28px',
            fontWeight: 700,
            color: '#1e293b'
          }}>
            {getRoleDisplayDepartment(userInfo.role, userInfo.specialty || context)}
          </h1>
          <p style={{ 
            margin: 0,
            fontSize: '14px',
            color: '#64748b'
          }}>
            {config?.staffName || userInfo.name} • {userInfo.role || 'PHYSICIAN'}
          </p>
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          gap: '8px'
        }}>
          <div style={{
            fontSize: '12px',
            color: '#64748b',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <FaClock /> Shift: {config?.shiftStartTime} - {config?.shiftEndTime || '—'}
          </div>

          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 12px',
            borderRadius: '999px',
            background: '#0f172a',
            color: '#f8fafc',
            fontSize: '12px',
            fontWeight: 700,
            boxShadow: '0 1px 2px rgba(15,23,42,0.12)'
          }}>
            <FaClock />
            <span>{currentTime.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}</span>
            <span style={{ color: '#93c5fd' }}>•</span>
            <span>{currentTime.toLocaleDateString([], { month: 'short', day: 'numeric' })}</span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            flexWrap: 'wrap',
            justifyContent: 'flex-end'
          }}>
            {[
                { icon: <FaCalendarAlt />, label: 'Calendar Events', action: () => navigate('/calendar') },
                { icon: <FaHistory />, label: 'History Actions', action: () => navigate('/history-actions'), badge: unreadNotificationsCount },
                { icon: <FaFlask />, label: 'Lab Requests', action: () => navigate('/lab-requests') },
                { icon: <FaCapsules />, label: 'Ordered Meds', action: () => navigate('/ordered-meds'), badge: orderedMedsCount },
                { icon: <FaProcedures />, label: 'Scheduled Surgeries', action: () => navigate('/scheduled-surgeries') }
            ].map((item) => (
              <button
                key={item.label}
                onClick={item.action}
                style={{
                  position: 'relative',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '8px 12px',
                  background: '#ffffff',
                  color: '#0f172a',
                  border: '1px solid #dbeafe',
                  borderRadius: '999px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: 700,
                  boxShadow: '0 1px 2px rgba(15,23,42,0.06)'
                }}
              >
                <span style={{ fontSize: '13px' }}>{item.icon}</span>
                    <span>{item.label}</span>
                {item.badge > 0 && (
                  <span style={{
                    marginLeft: 6,
                    minWidth: 20,
                    height: 20,
                    padding: '0 6px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: 999,
                    background: '#ef4444',
                    color: '#ffffff',
                    fontSize: 11,
                    fontWeight: 800
                  }}>
                    {item.badge > 99 ? '99+' : item.badge}
                  </span>
                )}
              </button>
            ))}
          </div>

        </div>
      </header>

      {/* ===== METRICS ===== */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px'
      }}>
        <div style={{
          padding: '20px',
          background: 'white',
          borderRadius: '10px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
        }}>
          <p style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#000000', fontWeight: 750 }}>Total Beds</p>
          <h2 style={{ margin: 0, fontSize: '32px', color: '#10b981', fontWeight: 800 }}>
            {bedMetrics.total}
          </h2>
        </div>
        <div style={{
          padding: '20px',
          background: 'white',
          borderRadius: '10px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
        }}>
          <p style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#000000', fontWeight: 750 }}>Hospitalized Patients</p>
          <h2 style={{ margin: 0, fontSize: '32px', color: '#ef4444', fontWeight: 800 }}>
            {bedMetrics.occupied}
          </h2>
        </div>
        <div style={{
          padding: '20px',
          background: 'white',
          borderRadius: '10px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
        }}>
          <p style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#000000', fontWeight: 750 }}>Pending Triaged Patients</p>
          <h2 style={{ margin: 0, fontSize: '32px', color: '#facc15', fontWeight: 800 }}>
            {bedMetrics.pendingTriage}
          </h2>
        </div>
      </div>

      {/* ===== BED GRID VIEW & PRIORITY QUEUE (Side-by-side, Responsive) ===== */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: windowWidth >= 1400 ? 'minmax(0, 1fr) 320px' : '1fr',
        gap: '20px',
        alignItems: 'start'
      }}>
        {/* LEFT: BED GRID */}
        <div style={{
          padding: '24px',
          background: 'white',
          borderRadius: '12px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
          overflow: 'hidden'
        }}>
          {/* BED GRID HEADER - Reorganized & Responsive */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '24px',
            paddingBottom: '16px',
            borderBottom: '2px solid #e5e7eb',
            gap: '16px',
            flexWrap: windowWidth < 768 ? 'wrap' : 'nowrap'
          }}>
            {/* LEFT: Title */}
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '12px', 
              minWidth: '200px',
              order: 1
            }}>
              <span style={{ fontSize: '24px' }}>🛏️</span>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: '16px', fontWeight: 700, color: '#1e293b', margin: 0 }}>Department Beds</div>
                <div style={{ 
                  fontSize: '12px', 
                  color: '#64748b', 
                  margin: '4px 0 0 0',
                  display: windowWidth < 640 ? 'none' : 'block'
                }}>Real-time ward occupancy and patient care status</div>
              </div>
            </div>

            {/* RIGHT: Search and Filter Controls (responsive, stacking on mobile) */}
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: windowWidth < 640 ? '8px' : '10px',
              justifyContent: windowWidth < 768 ? 'flex-start' : 'flex-end',
              flexWrap: 'wrap',
              order: windowWidth < 768 ? 3 : 2,
              width: windowWidth < 768 ? '100%' : 'auto'
            }}>
              <button
                onClick={() => setShowPatientSearchModal(true)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: windowWidth < 640 ? '8px 10px' : '10px 14px',
                  borderRadius: '8px',
                  border: '1px solid #cbd5e1',
                  background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
                  color: '#0f172a',
                  cursor: 'pointer',
                  fontSize: windowWidth < 640 ? '12px' : '13px',
                  fontWeight: 600,
                  transition: 'all 0.2s ease',
                  whiteSpace: 'nowrap',
                  flex: windowWidth < 640 ? 1 : 'auto'
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = 'linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)';
                  e.target.style.borderColor = '#94a3b8';
                  e.target.style.boxShadow = '0 2px 4px rgba(15,23,42,0.08)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)';
                  e.target.style.borderColor = '#cbd5e1';
                  e.target.style.boxShadow = 'none';
                }}
              >
                <FaSearch size={windowWidth < 640 ? 12 : 13} style={{ flexShrink: 0 }} />
                <span style={{ display: windowWidth < 640 ? 'none' : 'inline' }}>Search</span>
              </button>

              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 0',
                flex: windowWidth < 640 ? 1 : 'auto'
              }}>
                <label style={{
                  fontSize: windowWidth < 640 ? '11px' : '12px',
                  color: '#64748b',
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                  display: windowWidth < 640 ? 'none' : 'block'
                }}>
                  Ward:
                </label>
                <select
                  value={selectedWard}
                  onChange={(e) => setSelectedWard(e.target.value)}
                  style={{
                    padding: windowWidth < 640 ? '8px 10px' : '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
                    color: '#0f172a',
                    cursor: 'pointer',
                    fontSize: windowWidth < 640 ? '12px' : '13px',
                    fontWeight: 600,
                    transition: 'all 0.2s ease',
                    appearance: 'none',
                    paddingRight: '28px',
                    backgroundImage: 'url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 24 24%27 fill=%27none%27 stroke=%27%230f172a%27 stroke-width=%272%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27%3e%3cpolyline points=%276 9 12 15 18 9%27%3e%3c/polyline%3e%3c/svg%3e")',
                    backgroundRepeat: 'no-repeat',
                    backgroundPosition: 'right 8px center',
                    backgroundSize: '16px'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = 'linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)';
                    e.target.style.borderColor = '#94a3b8';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)';
                    e.target.style.borderColor = '#cbd5e1';
                  }}
                >
                  <option value="ALL">All Wards</option>
                  {wards.map((ward) => (
                    <option key={ward._id} value={ward.department}>
                      {ward.wardName || ward.department}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>



        {showHistoryActionsModal && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.72)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1200,
            padding: '20px'
          }}>
            <div style={{
              width: '100%',
              maxWidth: '640px',
              background: '#ffffff',
              borderRadius: '16px',
              boxShadow: '0 20px 50px rgba(15, 23, 42, 0.24)',
              padding: '20px',
              color: '#0f172a'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <div>
                  <div style={{ fontSize: '17px', fontWeight: 800 }}>History Actions</div>
                  <div style={{ fontSize: '13px', color: '#64748b', marginTop: '4px' }}>Review recent physician activity and encounter history.</div>
                </div>
                <button
                  onClick={() => setShowHistoryActionsModal(false)}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    color: '#64748b',
                    cursor: 'pointer',
                    fontSize: '16px'
                  }}
                >
                  <FaTimes />
                </button>
              </div>

              <div style={{ marginTop: '14px', display: 'grid', gap: '10px', maxHeight: '320px', overflowY: 'auto' }}>
                {assignedPatients.length === 0 ? (
                  <div style={{ padding: '16px', borderRadius: '10px', background: '#f8fafc', color: '#64748b', fontSize: '13px' }}>
                    No recent history actions available.
                  </div>
                ) : (
                  assignedPatients.map((patient) => (
                    <div key={patient._id} style={{ padding: '14px', border: '1px solid #e2e8f0', borderRadius: '10px', background: '#f9fafb' }}>
                      <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>{patient.patientName || patient.firstName || 'Unknown patient'}</div>
                      <div style={{ marginTop: '6px', fontSize: '12px', color: '#475569' }}>
                        Status: {patient.handover?.handover_status || 'Unknown'} • Created: {new Date(patient.createdAt || patient.created_at || Date.now()).toLocaleString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {showCalendarEventsModal && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.72)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1200,
            padding: '20px'
          }} onClick={() => setShowCalendarEventsModal(false)}>
            <div style={{
              width: '100%',
              maxWidth: '760px',
              background: '#ffffff',
              borderRadius: '18px',
              padding: '24px',
              boxShadow: '0 24px 64px rgba(15,23,42,0.24)',
              overflowY: 'auto',
              maxHeight: '90vh'
            }} onClick={(e) => e.stopPropagation()}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
                <div>
                  <div style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a' }}>Calendar Events</div>
                  <div style={{ color: '#64748b', fontSize: '13px', marginTop: '6px' }}>View your upcoming shift, scheduled procedures, and clinical planning events.</div>
                </div>
                <button
                  onClick={() => setShowCalendarEventsModal(false)}
                  style={{ border: 'none', background: 'transparent', color: '#475569', cursor: 'pointer', fontSize: '20px' }}>
                  ×
                </button>
              </div>
              <div style={{ display: 'grid', gap: '16px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div style={{ padding: '18px', borderRadius: '16px', border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.12em' }}>Upcoming shift</div>
                    <div style={{ marginTop: '10px', fontSize: '18px', fontWeight: 700, color: '#111827' }}>Day shift</div>
                    <div style={{ marginTop: '8px', color: '#475569', fontSize: '14px' }}>Today • 08:00 – 17:00</div>
                  </div>
                  <div style={{ padding: '18px', borderRadius: '16px', border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.12em' }}>Next surgery block</div>
                    <div style={{ marginTop: '10px', fontSize: '18px', fontWeight: 700, color: '#111827' }}>OR 3 • 14:00</div>
                    <div style={{ marginTop: '8px', color: '#475569', fontSize: '14px' }}>Urgent laparotomy for patient in bed {selectedBedModal || 'TBD'}</div>
                  </div>
                </div>
                <div style={{ padding: '18px', borderRadius: '16px', border: '1px solid #e2e8f0', background: '#fff' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                    <div>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.12em' }}>Event timeline</div>
                    </div>
                    <span style={{ fontSize: '12px', color: '#64748b' }}>Updated now</span>
                  </div>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    {['08:00 • Ward rounds', '10:00 • Clinical review', '13:00 • Consultant huddle', '16:00 • Case handover'].map((entry) => (
                      <div key={entry} style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                        <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#0f172a', marginTop: 6 }} />
                        <div>
                          <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>{entry.split(' • ')[0]}</div>
                          <div style={{ color: '#475569', fontSize: '13px' }}>{entry.split(' • ')[1]}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {showSurgeryModal && bedModalData && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(6, 15, 35, 0.88)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1150,
            backdropFilter: 'blur(8px)',
            padding: '20px',
            overflowY: 'auto'
          }}
          onClick={() => setShowSurgeryModal(false)}>
            <div style={{
              background: 'linear-gradient(180deg, rgba(7, 18, 36, 0.98), rgba(4, 11, 25, 0.98))',
              borderRadius: '24px',
              border: '1px solid rgba(59, 130, 246, 0.2)',
              padding: '32px',
              maxWidth: '640px',
              width: '100%',
              boxShadow: '0 25px 50px rgba(0, 0, 0, 0.5)',
              color: '#f8fafc'
            }}
            onClick={(e) => e.stopPropagation()}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '22px' }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: '24px', color: '#f8fafc', fontWeight: 700 }}>🩺 Schedule Surgery</h2>
                  <p style={{ margin: '10px 0 0 0', color: '#cbd5e1', fontSize: '14px' }}>Create a surgical care plan and schedule the procedure for the selected patient.</p>
                </div>
                <button
                  onClick={() => setShowSurgeryModal(false)}
                  style={{ border: 'none', background: 'transparent', color: '#cbd5e1', cursor: 'pointer', fontSize: '24px' }}>
                  ×
                </button>
              </div>
              <div style={{ display: 'grid', gap: '18px' }}>
                <div>
                  <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Procedure</label>
                  <input
                    value={surgeryForm.procedure_name}
                    onChange={(e) => setSurgeryForm((prev) => ({ ...prev, procedure_name: e.target.value }))}
                    placeholder="e.g. Appendectomy"
                    style={{ width: '100%', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px' }}
                  />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                  <div>
                    <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Type</label>
                    <select
                      value={surgeryForm.surgery_type}
                      onChange={(e) => setSurgeryForm((prev) => ({ ...prev, surgery_type: e.target.value }))}
                      style={{ width: '100%', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px', cursor: 'pointer' }}
                    >
                      {SURGERY_TYPE_OPTIONS.map((type) => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Priority</label>
                    <select
                      value={surgeryForm.priority}
                      onChange={(e) => setSurgeryForm((prev) => ({ ...prev, priority: e.target.value }))}
                      style={{ width: '100%', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px', cursor: 'pointer' }}
                    >
                      <option value="URGENT">URGENT</option>
                      <option value="HIGH">HIGH</option>
                      <option value="MEDIUM">MEDIUM</option>
                      <option value="LOW">LOW</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Schedule</label>
                  <input
                    type="datetime-local"
                    value={surgeryForm.scheduled_at}
                    onChange={(e) => setSurgeryForm((prev) => ({ ...prev, scheduled_at: e.target.value }))}
                    style={{ width: '100%', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Surgeon</label>
                  <input
                    value={surgeryForm.surgeon}
                    onChange={(e) => setSurgeryForm((prev) => ({ ...prev, surgeon: e.target.value }))}
                    placeholder="Surgeon name"
                    style={{ width: '100%', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Surgical team</label>
                  <input
                    value={surgeryForm.surgical_team}
                    onChange={(e) => setSurgeryForm((prev) => ({ ...prev, surgical_team: e.target.value }))}
                    placeholder="Comma separated names"
                    style={{ width: '100%', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Rationale</label>
                  <textarea
                    value={surgeryForm.rationale}
                    onChange={(e) => setSurgeryForm((prev) => ({ ...prev, rationale: e.target.value }))}
                    placeholder="Clinical indication for the surgery"
                    style={{ width: '100%', minHeight: '100px', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px', resize: 'vertical', fontFamily: 'inherit' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.14em' }}>Notes</label>
                  <textarea
                    value={surgeryForm.notes}
                    onChange={(e) => setSurgeryForm((prev) => ({ ...prev, notes: e.target.value }))}
                    placeholder="Pre-op notes or special instructions"
                    style={{ width: '100%', minHeight: '90px', padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(15,23,42,0.88)', color: '#f8fafc', fontSize: '14px', resize: 'vertical', fontFamily: 'inherit' }}
                  />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '10px' }}>
                  <button
                    type="button"
                    onClick={() => setShowSurgeryModal(false)}
                    style={{ padding: '14px 18px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(255,255,255,0.08)', color: '#cbd5e1', fontWeight: 700, cursor: 'pointer', fontSize: '14px' }}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleSubmitSurgery}
                    disabled={submittingSurgery}
                    style={{ padding: '14px 18px', borderRadius: '14px', border: 'none', background: submittingSurgery ? '#475569' : '#22c55e', color: '#ffffff', fontWeight: 700, cursor: submittingSurgery ? 'wait' : 'pointer', fontSize: '14px' }}
                  >
                    {submittingSurgery ? 'Saving…' : 'Schedule Surgery'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#64748b' }}>
            Loading bed information...
          </div>
        ) : beds.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#64748b' }}>
            No beds available for the selected ward
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: '12px'
          }}>
            {beds.map((bed) => {
              const bedNumber = bed.bedId || bed._id || 'Unknown';
              const encounter = bed.currentEncounterId;
              const patientName = encounter?.patientName || encounter?.firstName || encounter?.patient?.name || encounter?.name || 'Empty Bed';
              const patientGender = encounter?.patient?.gender || encounter?.gender || '';
              const patientAge = encounter?.age ?? encounter?.patient?.age ?? null;
              const patientDetails = [patientGender, patientAge ? `${patientAge} yrs` : null].filter(Boolean).join(' • ');
              const severity = encounter?.severity ?? encounter?.triage?.aiTriageReport?.esi_level ?? bed.severity ?? 4;
              const severityConfig = getSeverityConfig(severity);
              const equipmentDisplay = buildEquipmentDisplayList(bed);
              const equipmentCount = equipmentDisplay.length;
              const isOccupied = Boolean(encounter);
              const cardClinicalSummary = isOccupied
                ? (encounter?.diagnosis || encounter?.triageSummary || encounter?.physician_notes || encounter?.treatment_plan || 'No active diagnosis documented yet.')
                : 'Empty bed ready for admission and immediate care setup.';

              const emptyCardBackground = 'linear-gradient(180deg, #07152f 0%, #0f1f46 100%)';
              const emptyCardBorder = '1px solid rgba(59, 130, 246, 0.35)';
              const emptyCardText = '#f8fafc';
              const emptyCardSubtext = '#c7d9ff';

              return (
                <div
                  key={`bed-${bedNumber}`}
                  onClick={() => handleBedClick(bed)}
                  style={{
                    padding: '18px',
                    background: isOccupied ? '#f8fafc' : emptyCardBackground,
                    border: isOccupied ? `1px solid ${severityConfig.bg}` : emptyCardBorder,
                    borderRadius: '16px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    minHeight: '170px',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '12px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow = '0 8px 24px rgba(15,23,42,0.08)';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', alignItems: 'flex-start' }}>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: '#475569' }}>
                        Bed {bedNumber}
                      </div>
                      <div style={{ marginTop: '10px', fontSize: '16px', fontWeight: 700, color: isOccupied ? '#0f172a' : emptyCardText, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {patientName}
                      </div>
                      {isOccupied && patientDetails ? (
                        <div style={{ marginTop: '8px', fontSize: '12px', color: '#64748b' }}>
                          {patientDetails}
                        </div>
                      ) : null}
                    </div>
                    {!isOccupied ? (
                      <div style={{
                        width: '44px',
                        height: '44px',
                        borderRadius: '50%',
                        background: 'rgba(96, 165, 250, 0.18)',
                        color: '#f8fafc',
                        display: 'grid',
                        placeItems: 'center',
                        fontSize: '22px',
                        lineHeight: 1
                      }}>
                        +
                      </div>
                    ) : (
                      <div style={{
                        padding: '8px 10px',
                        borderRadius: '999px',
                        background: severityConfig.bg,
                        color: severityConfig.color,
                        fontSize: '12px',
                        fontWeight: 700,
                        whiteSpace: 'nowrap'
                      }}>
                        {severityConfig.icon} {severityConfig.label}
                      </div>
                    )}
                  </div>

                  <div style={{ display: 'grid', gap: '8px' }}>
                    {isOccupied ? (
                      <>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: '#475569', fontSize: '13px' }}>
                          <span>{equipmentCount} equipment</span>
                          <span style={{ padding: '4px 10px', borderRadius: '999px', background: '#e2e8f0', color: '#0f172a', fontWeight: 700 }}>
                            {bed.status || 'Occupied'}
                          </span>
                        </div>
                        <div style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.5 }}>
                          {cardClinicalSummary}
                        </div>
                      </>
                    ) : (
                      <div style={{ textAlign: 'center', color: emptyCardSubtext, fontSize: '13px' }}>
                        Empty bed — click to assign or view details
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        </div>

        {/* RIGHT: Priority Queue Panel (Nurse-style) */}
        {showQueueOpen && (
          <aside style={{
            display: 'flex',
            flexDirection: 'column',
            background: 'white',
            borderRadius: '12px',
            border: '1px solid #e5e7eb',
            boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            overflow: 'hidden',
            height: 'fit-content',
            maxHeight: '80vh'
          }}>
            {/* Queue Header */}
            <div style={{ 
              padding: '20px',
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

            {/* Queue List */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
              {pendingWithMeta.slice(0, 20).map(({ patient: p, waitMs, sev, priority }, idx) => {
                const severityConfig = {
                  1: { bg: '#fee2e2', color: '#991b1b', label: 'CRITICAL', icon: '🔴' },
                  2: { bg: '#ffedd5', color: '#9a3412', label: 'HIGH', icon: '🟠' },
                  3: { bg: '#fef3c7', color: '#92400e', label: 'MODERATE', icon: '🟡' },
                  4: { bg: '#dcfce7', color: '#166534', label: 'LOW', icon: '🟢' },
                  5: { bg: '#dcfce7', color: '#166534', label: 'LOW', icon: '🟢' }
                };
                const sevConfig = severityConfig[sev] || severityConfig[4];
                const patientName = p.patientName || p.firstName || p.patient?.name || `#${String(p.id || p._id || '').slice(-6)}`;
                const waitLabel = formatWaitTime(p.createdAt, currentTime);

                return (
                  <div 
                    key={(p._id || p.id) + '-prio'} 
                    style={{ 
                      padding: '12px 12px',
                      borderBottom: idx === pendingWithMeta.slice(0, 20).length - 1 ? 'none' : '1px solid #f1f5f9',
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
                          {patientName}
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
                            {waitLabel}
                          </span>
                        </div>
                      </div>

                      {/* View Button */}
                      <button 
                        onClick={() => handleViewTriageReport(p)}
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
        )}
      </div>

      {/* ===== ASSIGNED PATIENTS TABLE (Full-width below) ===== */}
      <div style={{
        padding: '24px',
        background: 'white',
        borderRadius: '12px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
        overflow: 'hidden'
      }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '20px',
            paddingBottom: '16px',
            borderBottom: '2px solid #e5e7eb'
          }}>
            <h2 style={{
              margin: 0,
              fontSize: '18px',
              fontWeight: 700,
              color: '#1e293b',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              📋 Assigned Patients
            </h2>
            <span style={{
              fontSize: '12px',
              fontWeight: 600,
              color: '#475569',
              background: '#f1f5f9',
              padding: '6px 12px',
              borderRadius: '20px'
            }}>
              {assignedPatients.length} total
            </span>
          </div>

          {loading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#64748b' }}>
              Loading patients...
            </div>
          ) : error ? (
            <div style={{ 
              padding: '20px', 
              background: '#fee2e2', 
              color: '#991b1b', 
              borderRadius: '8px',
              fontSize: '14px'
            }}>
              ⚠️ Error: {error}
            </div>
          ) : assignedPatients.length === 0 ? (
            <div style={{
              padding: '60px 20px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '40px', marginBottom: '12px' }}>✨</div>
              <p style={{ fontSize: '16px', fontWeight: 600, color: '#1e293b', marginBottom: '4px' }}>
                No Patients Assigned
              </p>
              <p style={{ fontSize: '14px', color: '#64748b', margin: 0 }}>
                You'll see assigned patients here once they're handed over from triage
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {assignedPatients.map(patient => {
                const patientName = patient.patientName || patient.firstName || 'Unknown';
                const severity = patient.triage?.aiTriageReport?.esi_level || patient.severity || 4;
                const severityConfig = getSeverityConfig(severity);
                const handoverStatus = patient.handover?.handover_status || 'NOTIFIED';
                const statusConfig = getStatusConfig(handoverStatus);
                const waitTime = formatWaitTime(patient.createdAt, currentTime);
                const specialty = patient.triage?.specialty || 'General';

                return (
                  <div
                    key={patient._id}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '16px',
                      background: '#f9fafb',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = '#f0fdf4';
                      e.currentTarget.style.borderColor = '#10b981';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = '#f9fafb';
                      e.currentTarget.style.borderColor = '#e5e7eb';
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        marginBottom: '8px',
                        flexWrap: 'wrap'
                      }}>
                        <span style={{
                          display: 'inline-block',
                          padding: '4px 10px',
                          background: severityConfig.bg,
                          color: severityConfig.color,
                          borderRadius: '6px',
                          fontSize: '11px',
                          fontWeight: 700,
                          whiteSpace: 'nowrap'
                        }}>
                          {severityConfig.icon} {severityConfig.label}
                        </span>
                        <span style={{
                          fontSize: '13px',
                          fontWeight: 700,
                          color: '#1e293b',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {patientName}
                        </span>
                      </div>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        fontSize: '12px',
                        color: '#64748b',
                        flexWrap: 'wrap'
                      }}>
                        <span>{specialty}</span>
                        <span>•</span>
                        <span>Wait: {waitTime}</span>
                        <span style={{
                          padding: '2px 8px',
                          background: `${statusConfig.color}20`,
                          color: statusConfig.color,
                          borderRadius: '4px',
                          fontWeight: 600,
                          fontSize: '10px',
                          whiteSpace: 'nowrap'
                        }}>
                          {statusConfig.label}
                        </span>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginLeft: '12px' }}>
                      <button
                        onClick={() => handleAcknowledge(patient._id)}
                        disabled={acknowledging[patient._id] || isPatientAcknowledged(patient)}
                        title={isPatientAcknowledged(patient) ? 'This patient has already been acknowledged' : 'Acknowledge patient'}
                        aria-label="Acknowledge patient"
                        style={{
                          padding: '8px 12px',
                          background: isPatientAcknowledged(patient) ? '#94a3b8' : '#10b981',
                          color: 'white',
                          border: 'none',
                          borderRadius: '6px',
                          cursor: acknowledging[patient._id] || isPatientAcknowledged(patient) ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          fontSize: '11px',
                          transition: 'all 0.2s ease',
                          opacity: acknowledging[patient._id] || isPatientAcknowledged(patient) ? 0.7 : 1,
                          whiteSpace: 'nowrap'
                        }}
                        onMouseEnter={(e) => {
                          if (!acknowledging[patient._id] && !isPatientAcknowledged(patient)) e.target.style.background = '#059669';
                        }}
                        onMouseLeave={(e) => e.target.style.background = isPatientAcknowledged(patient) ? '#94a3b8' : '#10b981'}
                      >
                        {acknowledging[patient._id] ? '⏳ Acknowledging' : isPatientAcknowledged(patient) ? '✓ Acknowledged' : '✓ Acknowledge'}
                      </button>

                      <button
                        onClick={() => handleViewTriageReport(patient)}
                        title="View triage report"
                        aria-label="View triage report"
                        style={{
                          padding: '8px 12px',
                          background: '#8b5cf6',
                          color: 'white',
                          border: 'none',
                          borderRadius: '6px',
                          cursor: 'pointer',
                          fontWeight: 600,
                          fontSize: '11px',
                          transition: 'all 0.2s ease',
                          whiteSpace: 'nowrap'
                        }}
                        onMouseEnter={(e) => e.target.style.background = '#7c3aed'}
                        onMouseLeave={(e) => e.target.style.background = '#8b5cf6'}
                      >
                        📄 Triage Report
                      </button>

                      <button
                        onClick={() => handleHospitalize(patient, patientName)}
                        disabled={isEncounterHospitalized(patient) || hospitalizing[patient._id]}
                        title={isEncounterHospitalized(patient) ? 'Already hospitalized' : 'Hospitalize patient'}
                        aria-label={isEncounterHospitalized(patient) ? 'Already hospitalized' : 'Hospitalize patient'}
                        style={{
                          padding: '8px 12px',
                          background: isEncounterHospitalized(patient) ? '#94a3b8' : '#2563eb',
                          color: 'white',
                          border: 'none',
                          borderRadius: '6px',
                          cursor: isEncounterHospitalized(patient) || hospitalizing[patient._id] ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          fontSize: '11px',
                          transition: 'all 0.2s ease',
                          opacity: isEncounterHospitalized(patient) || hospitalizing[patient._id] ? 0.7 : 1,
                          whiteSpace: 'nowrap'
                        }}
                      >
                        🏥 Hospitalize
                      </button>

                      {userInfo.role === 'PHYSICIAN' ? (
                        <button
                          onClick={() => handleDiagnose(patient._id, patient)}
                          disabled={diagnosing[patient._id] || !isPatientAcknowledged(patient)}
                          title={!isPatientAcknowledged(patient) ? 'Patient must be acknowledged before diagnosis' : 'Open diagnosis workflow'}
                          aria-label="Open diagnosis workflow"
                          style={{
                            padding: '8px 12px',
                            background: !isPatientAcknowledged(patient) ? '#cbd5e1' : '#f59e0b',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: diagnosing[patient._id] || !isPatientAcknowledged(patient) ? 'not-allowed' : 'pointer',
                            fontWeight: 600,
                            fontSize: '11px',
                            transition: 'all 0.2s ease',
                            opacity: diagnosing[patient._id] || !isPatientAcknowledged(patient) ? 0.7 : 1,
                            whiteSpace: 'nowrap'
                          }}
                          onMouseEnter={(e) => {
                            if (!diagnosing[patient._id] && isPatientAcknowledged(patient)) e.target.style.background = '#d97706';
                          }}
                          onMouseLeave={(e) => e.target.style.background = !isPatientAcknowledged(patient) ? '#cbd5e1' : '#f59e0b'}
                        >
                          {isPatientAcknowledged(patient) ? '🩺 Diagnose' : '🔒 Diagnose'}
                        </button>
                      ) : (
                        <button
                          onClick={() => handleCareFor(patient._id, patientName)}
                          disabled={diagnosing[patient._id]}
                          title="Record care for patient"
                          aria-label="Record care for patient"
                          style={{
                            padding: '8px 12px',
                            background: '#06b6d4',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: diagnosing[patient._id] ? 'wait' : 'pointer',
                            fontWeight: 600,
                            fontSize: '11px',
                            transition: 'all 0.2s ease',
                            opacity: diagnosing[patient._id] ? 0.7 : 1,
                            whiteSpace: 'nowrap'
                          }}
                          onMouseEnter={(e) => {
                            if (!diagnosing[patient._id]) e.target.style.background = '#0891b2';
                          }}
                          onMouseLeave={(e) => e.target.style.background = '#06b6d4'}
                        >
                          💊 Record Care
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

      {/* Search Modal */}
      {showPatientSearchModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(15, 23, 42, 0.72)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1200,
          padding: '20px'
        }}>
          <div style={{
            width: '100%',
            maxWidth: '640px',
            background: '#ffffff',
            borderRadius: '16px',
            boxShadow: '0 20px 50px rgba(15, 23, 42, 0.24)',
            padding: '20px',
            color: '#0f172a'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div>
                <div style={{ fontSize: '17px', fontWeight: 800 }}>Search patient bed records</div>
                <div style={{ fontSize: '13px', color: '#64748b', marginTop: '4px' }}>Find the best bed match by patient name, age, diagnosis, or care notes.</div>
              </div>
              <button
                onClick={() => setShowPatientSearchModal(false)}
                style={{
                  border: 'none',
                  background: 'transparent',
                  color: '#64748b',
                  cursor: 'pointer',
                  fontSize: '16px'
                }}
              >
                <FaTimes />
              </button>
            </div>

            <input
              value={patientSearchQuery}
              onChange={(e) => setPatientSearchQuery(e.target.value)}
              placeholder="Search by patient name, diagnosis, age, or symptoms"
              style={{
                width: '100%',
                padding: '12px 14px',
                borderRadius: '10px',
                border: '1px solid #dbeafe',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box'
              }}
            />

            <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '320px', overflowY: 'auto' }}>
              {getBedSearchResults().length === 0 ? (
                <div style={{ padding: '16px', borderRadius: '10px', background: '#f8fafc', color: '#64748b', fontSize: '13px' }}>
                  No matches found for this search.
                </div>
              ) : (
                getBedSearchResults().map((item) => {
                  const displayName = item.patientName || 'Unassigned bed';
                  const bedNumber = item.bed?.bedId || item.bed?._id || 'Unknown';
                  const summary = item.diagnosis || item.summary || 'No clinical summary yet';
                  return (
                    <button
                      key={`search-${bedNumber}`}
                      onClick={() => {
                        setShowPatientSearchModal(false);
                        handleBedClick(item.bed);
                      }}
                      style={{
                        textAlign: 'left',
                        border: '1px solid #e2e8f0',
                        background: '#f8fafc',
                        borderRadius: '10px',
                        padding: '12px 14px',
                        cursor: 'pointer'
                      }}
                    >
                      <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>{displayName}</div>
                      <div style={{ fontSize: '12px', color: '#475569', marginTop: '4px' }}>Bed {bedNumber}</div>
                      <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>{summary}</div>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}

      {selectedBedModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(6, 15, 35, 0.88)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          backdropFilter: 'blur(8px)',
          padding: '20px',
          overflowY: 'auto'
        }}
        onClick={() => {
          setSelectedBedModal(null);
          setBedModalData(null);
        }}>
          <style>{`
            .physician-bed-modal-body {
              padding: 24px;
              display: flex;
              flex-direction: column;
              min-height: calc(90vh - 104px);
              max-height: calc(90vh - 104px);
              overflow-y: auto;
              overflow-x: hidden;
              scrollbar-width: thin;
              scrollbar-color: rgba(56, 189, 248, 0.35) transparent;
            }
            .physician-bed-modal-body::-webkit-scrollbar {
              width: 8px;
            }
            .physician-bed-modal-body::-webkit-scrollbar-track {
              background: transparent;
            }
            .physician-bed-modal-body::-webkit-scrollbar-thumb {
              background: rgba(56, 189, 248, 0.35);
              border-radius: 999px;
            }
            .physician-bed-modal-grid {
              display: grid;
              grid-template-columns: minmax(0, 1.65fr) minmax(320px, 1fr);
              gap: 24px;
              min-height: 520px;
              min-height: 0;
              height: 100%;
              align-items: start;
              overflow: visible;
            }
            .physician-bed-modal-grid > section,
            .physician-bed-modal-grid > aside {
              min-height: 0;
            }
            .physician-bed-modal-aside {
              display: flex;
              flex-direction: column;
              gap: 22px;
              min-height: 0;
              padding-right: 2px;
            }
            .physician-bed-modal-aside::-webkit-scrollbar {
              width: 8px;
            }
            .physician-bed-modal-aside::-webkit-scrollbar-track {
              background: transparent;
            }
            .physician-bed-modal-aside::-webkit-scrollbar-thumb {
              background: rgba(99, 102, 241, 0.32);
              border-radius: 999px;
            }
            @media (max-width: 980px) {
              .physician-bed-modal-grid {
                grid-template-columns: 1fr;
                min-height: auto;
              }
            }
            @media (max-width: 640px) {
              .physician-bed-modal-container {
                border-radius: 18px;
              }
              .physician-bed-modal-body {
                padding: 18px;
              }
            }
          `}</style>
          <div className="physician-bed-modal-container" style={{
            background: 'linear-gradient(180deg, rgba(7, 18, 36, 0.98), rgba(4, 11, 25, 0.98))',
            borderRadius: '24px',
            boxShadow: '0 32px 110px rgba(0,0,0,0.55)',
            width: '100%',
            maxWidth: '1120px',
            maxHeight: '90vh',
            overflow: 'hidden',
            position: 'relative',
            border: '1px solid rgba(14, 165, 233, 0.18)'
          }}
          onClick={(e) => e.stopPropagation()}>
            
            {/* Modal Header */}
            <div style={{
              padding: '24px 28px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(59, 130, 246, 0.18))',
              borderBottom: '1px solid rgba(255,255,255,0.08)'
            }}>
              <div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px'
                }}>
                  <div style={{
                    width: '52px',
                    height: '52px',
                    borderRadius: '18px',
                    background: 'rgba(255,255,255,0.08)',
                    display: 'grid',
                    placeItems: 'center',
                    fontSize: '24px'
                  }}>
                    🛏️
                  </div>
                  <div>
                    <h2 style={{ margin: 0, fontSize: '24px', lineHeight: 1.1, color: '#f8fafc' }}>
                      Bed {selectedBedModal}
                    </h2>
                    <p style={{ margin: '8px 0 0 0', color: '#cbd5e1', fontSize: '14px' }}>
                      Real-time bed command center for patient care and workflows.
                    </p>
                  </div>
                </div>
              </div>
              <button
                onClick={() => {
                  setSelectedBedModal(null);
                  setBedModalData(null);
                }}
                style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '16px',
                  border: '1px solid rgba(255,255,255,0.1)',
                  background: 'rgba(255,255,255,0.08)',
                  color: '#f8fafc',
                  fontSize: '22px',
                  cursor: 'pointer'
                }}
              >
                ✕
              </button>
            </div>

            {/* Modal Body */}
            <div className="physician-bed-modal-body">
              {bedModalLoading ? (
                <div style={{
                  padding: '60px 40px',
                  textAlign: 'center',
                  color: '#94a3b8'
                }}>
                  <div style={{ fontSize: '48px', marginBottom: '16px' }}>⏳</div>
                  <p style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
                    Loading patient dashboard...
                  </p>
                  <p style={{ margin: 0, fontSize: '14px', color: '#94a3b8' }}>
                    Updating live encounter and resource details
                  </p>
                </div>
              ) : (
                <div className="physician-bed-modal-grid">
                  <section style={{ display: 'flex', flexDirection: 'column', gap: '22px', minHeight: 0 }}>
                    <div style={{
                      padding: '22px',
                      borderRadius: '24px',
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.08)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '18px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', flexWrap: 'wrap' }}>
                            <div>
                              <p style={{ margin: 0, fontSize: '12px', color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
                                Patient
                              </p>
                              <h3 style={{ margin: '10px 0 0 0', color: '#f8fafc', fontSize: '22px' }}>
                                {bedModalData?.patientName || bedModalData?.patient?.name || bedModalData?.firstName || 'Empty Bed'}
                              </h3>
                            </div>
                              </div>
                          <p style={{ margin: '10px 0 0 0', color: '#f8fafc', fontSize: '15px', fontWeight: 600 }}>
                            {bedModalData?.diagnosis || bedModalData?.triage?.aiTriageReport?.primary_diagnosis || bedModalData?.triage?.primary_diagnosis || 'No active diagnosis'}
                          </p>
                          <p style={{ margin: '8px 0 0 0', color: '#94a3b8', fontSize: '13px', lineHeight: 1.6 }}>
                            {bedModalData?.triageSummary || bedModalData?.physician_notes || 'No triage summary or physician note available yet.'}
                          </p>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end', alignSelf: 'flex-start', marginLeft: 'auto' }}>
                          <div style={{
                            fontSize: '11px',
                            fontWeight: 700,
                            letterSpacing: '0.12em',
                            textTransform: 'uppercase',
                            color: '#93c5fd',
                            padding: '0 2px'
                          }}>
                            Click to adjust severity
                          </div>
                          <div style={{
                            width: 'fit-content',
                            minWidth: '148px',
                            padding: '9px 11px',
                            borderRadius: '999px',
                            background: 'rgba(255,255,255,0.04)',
                            border: '1px solid rgba(255,255,255,0.08)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}>
                            <select
                              value={severitySelection ?? (bedModalData?.severity || bedModalData?.triage?.aiTriageReport?.esi_level || 4)}
                              onChange={(e) => handleSeveritySelectionChange(Number(e.target.value))}
                              style={{
                                width: '100%',
                                padding: '6px 8px',
                                borderRadius: '999px',
                                border: '1px solid rgba(148,163,184,0.35)',
                                background: 'rgba(15,23,42,0.92)',
                                color: '#f8fafc',
                                fontSize: '12px',
                                fontWeight: 700,
                                appearance: 'none',
                                textAlign: 'center'
                              }}
                            >
                              {severityOptions.map((value) => {
                                const config = getSeverityConfig(value);
                                return (
                                  <option key={value} value={value}>{`${config.icon} ${config.label}`}</option>
                                );
                              })}
                            </select>
                          </div>
                          <div style={{
                            minWidth: '180px',
                            padding: '16px',
                            borderRadius: '20px',
                            background: 'rgba(14, 165, 233, 0.12)',
                            border: '1px solid rgba(56, 189, 248, 0.2)'
                          }}>
                            <p style={{ margin: 0, fontSize: '12px', color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
                              Bed {selectedBedObject?.bedId || selectedBedModal}
                            </p>
                            <p style={{ margin: '10px 0 0 0', color: '#f8fafc', fontSize: '18px', fontWeight: 700 }}>
                              {selectedBedObject?.status || 'Unknown'}
                            </p>
                          </div>
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '14px', marginTop: '24px' }}>
                        <div style={{
                          padding: '16px',
                          borderRadius: '18px',
                          background: 'rgba(255,255,255,0.02)',
                          border: '1px solid rgba(255,255,255,0.08)'
                        }}>
                          <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>Patient ID</p>
                          <p style={{ margin: '10px 0 0 0', color: '#f8fafc', fontWeight: 700 }}>{bedModalData?._id?.slice(-8) || bedModalData?.patient?.id_number || 'N/A'}</p>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '18px',
                          background: 'rgba(255,255,255,0.02)',
                          border: '1px solid rgba(255,255,255,0.08)'
                        }}>
                          <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>Age</p>
                          <p style={{ margin: '10px 0 0 0', color: '#f8fafc', fontWeight: 700 }}>{bedModalData?.age ?? bedModalData?.patient?.age ?? '—'}</p>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '18px',
                          background: 'rgba(255,255,255,0.02)',
                          border: '1px solid rgba(255,255,255,0.08)'
                        }}>
                          <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>Severity</p>
                          <p style={{ margin: '10px 0 0 0', color: '#f8fafc', fontWeight: 700 }}>{getSeverityConfig(bedModalData?.triage?.aiTriageReport?.esi_level || bedModalData?.severity || 4).label}</p>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '18px',
                          background: 'rgba(255,255,255,0.02)',
                          border: '1px solid rgba(255,255,255,0.08)'
                        }}>
                          <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>Equipment</p>
                          <p style={{ margin: '10px 0 0 0', color: '#f8fafc', fontWeight: 700 }}>
                            {selectedBedObject?.equipment && selectedBedObject.equipment.length > 0 ?
                              `${selectedBedObject.equipment.length} items` : (selectedBedObject?.rawBed ? 'Configured' : 'Pending')}
                          </p>
                          <p style={{ margin: '6px 0 0 0', color: '#94a3b8', fontSize: '12px' }}>{normalizedEquipmentSummary}</p>
                        </div>
                      </div>
                    </div>

                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '18px'
                    }}>
                      <div style={{
                        padding: '20px',
                        borderRadius: '22px',
                        background: 'rgba(15,23,42,0.88)',
                        border: '1px solid rgba(255,255,255,0.08)'
                      }}>
                        <p style={{ margin: 0, fontSize: '12px', color: '#8b98a6', textTransform: 'uppercase', letterSpacing: '0.12em' }}>Vitals</p>
                        <div style={{ marginTop: '18px', display: 'grid', gap: '12px' }}>
                          {[
                            { label: 'Blood Pressure', value: modalVitals?.bp || '—' },
                            { label: 'Heart Rate', value: modalVitals?.hr != null ? `${modalVitals.hr} bpm` : '—' },
                            { label: 'Temperature', value: modalVitals?.temp != null ? `${modalVitals.temp}°C` : '—' },
                            { label: 'O₂ Saturation', value: modalVitals?.o2 != null ? `${modalVitals.o2}%` : '—' },
                            { label: 'Respiratory Rate', value: modalVitals?.resp != null ? `${modalVitals.resp}/min` : '—' },
                            { label: 'AVPU', value: modalVitals?.avpu || '—' }
                          ].map((item, idx) => (
                            <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', color: '#e2e8f0' }}>
                              <span>{item.label}</span>
                              <span>{item.value}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div style={{
                        padding: '20px',
                        borderRadius: '22px',
                        background: 'rgba(15,23,42,0.88)',
                        border: '1px solid rgba(255,255,255,0.08)'
                      }}>
                        <p style={{ margin: 0, fontSize: '12px', color: '#8b98a6', textTransform: 'uppercase', letterSpacing: '0.12em' }}>Assessment & Plan</p>
                        <div style={{ marginTop: '18px', display: 'grid', gap: '14px' }}>
                          <div>
                            <p style={{ margin: 0, color: '#9ca3af', fontSize: '12px' }}>Diagnosis</p>
                            <p style={{ margin: '6px 0 0 0', color: '#f8fafc', fontWeight: 700 }}>{bedModalData?.diagnosis || bedModalData?.triage?.aiTriageReport?.primary_diagnosis || bedModalData?.triage?.primary_diagnosis || 'Not available'}</p>
                          </div>
                          <div>
                            <p style={{ margin: 0, color: '#9ca3af', fontSize: '12px' }}>Treatment plan</p>
                            {bedModalData?.treatments?.length ? (
                              <div style={{ marginTop: '6px', display: 'grid', gap: '6px' }}>
                                {bedModalData.treatments.slice(0, 3).map((treatment, idx) => (
                                  <div key={idx} style={{ color: '#f8fafc', fontSize: '13px' }}>
                                    • {treatment.name || 'Intervention'}{treatment.dosage ? ` · ${treatment.dosage}` : ''}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p style={{ margin: '6px 0 0 0', color: '#f8fafc', fontWeight: 700 }}>{bedModalData?.treatment_plan || 'Pending'}</p>
                            )}
                          </div>
                          <div>
                            <p style={{ margin: 0, color: '#9ca3af', fontSize: '12px' }}>Recommendations</p>
                            {bedModalData?.recommendations?.length ? (
                              <div style={{ marginTop: '6px', display: 'grid', gap: '6px' }}>
                                {bedModalData.recommendations.slice(0, 3).map((recommendation, idx) => (
                                  <div key={idx} style={{ color: '#f8fafc', fontSize: '13px' }}>
                                    • {recommendation.description || recommendation.name || 'Recommendation'}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p style={{ margin: '6px 0 0 0', color: '#f8fafc', fontWeight: 700 }}>{bedModalData?.triage?.aiTriageReport?.recommended_treatment || bedModalData?.triage?.recommended_treatment || 'Pending'}</p>
                            )}
                          </div>
                          {bedModalData?.progression_notes?.length ? (
                            <div>
                              <p style={{ margin: 0, color: '#9ca3af', fontSize: '12px' }}>Progression note</p>
                              <p style={{ margin: '6px 0 0 0', color: '#f8fafc', fontSize: '13px', lineHeight: 1.6 }}>
                                {bedModalData.progression_notes[0]?.content || 'Progress note available'}
                              </p>
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </div>

                    <div style={{
                      padding: '22px',
                      borderRadius: '24px',
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid rgba(255,255,255,0.08)'
                    }}>
                      <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
                        Clinical snapshot
                      </p>
                      <div style={{ marginTop: '16px', display: 'grid', gap: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1' }}>
                          <span>Treatments</span><span>{bedModalData?.treatments?.length || 0}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1' }}>
                          <span>Recommendations</span><span>{bedModalData?.recommendations?.length || 0}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1' }}>
                          <span>Progression notes</span><span>{bedModalData?.progression_notes?.length || 0}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1' }}>
                          <span>Equipment</span><span>{modalEquipmentCount || (selectedBedObject?.rawBed ? 'Present' : 'Missing')}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1' }}>
                          <span>Medication requests</span><span>{bedModalData?.medication_requests?.length || 0}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1' }}>
                          <span>Medication administrations</span><span>{bedModalData?.medication_administrations?.length || 0}</span>
                        </div>
                        {selectedBedObject?.equipment && selectedBedObject.equipment.length > 0 && (
                          <div style={{ marginTop: '8px', padding: '10px', borderRadius: '10px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                            {selectedBedObject.equipment.map((eq, idx) => (
                              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: idx < selectedBedObject.equipment.length - 1 ? '1px dashed rgba(255,255,255,0.03)' : 'none' }}>
                                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                  <div style={{ fontSize: '16px' }}>⚙️</div>
                                  <div>
                                    <div style={{ color: '#f8fafc', fontWeight: 700 }}>{eq.name}</div>
                                    <div style={{ color: '#94a3b8', fontSize: '12px' }}>{eq.status}</div>
                                  </div>
                                </div>
                                <div style={{ color: '#cbd5e1', fontWeight: 700 }}>{eq.usageCount ?? 0}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </section>

                  <aside className="physician-bed-modal-aside" style={{
                    padding: '0',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '22px',
                    minHeight: 0
                  }}>
                    <div style={{
                      padding: '22px',
                      borderRadius: '24px',
                      background: 'linear-gradient(180deg, rgba(15,23,42,0.88), rgba(30,41,59,0.95))',
                      border: '1px solid rgba(255,255,255,0.08)'
                    }}>
                      <p style={{ margin: 0, fontSize: '12px', color: '#7c3aed', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
                        Clinical Actions
                      </p>
                      <div style={{ marginTop: '18px', display: 'grid', gap: '12px' }}>
                        <div style={{ display: 'grid', gap: '12px' }}>
                          {[
                            { label: 'View Triage Review', icon: '📋', handler: () => handleViewTriageReport(bedModalData), disabled: !bedModalData },
                            { label: 'View Inpatient Report', icon: '📄', handler: () => handleViewInpatientReport(bedModalData), disabled: !bedModalData || !isEncounterHospitalized(bedModalData) },
                            { label: 'Open Diagnosis Workflow', icon: '🩺', handler: () => handleDiagnose(bedModalData?._id, bedModalData), disabled: !bedModalData },
                            { label: 'Order Medication', icon: '💊', handler: () => handleOrderMedication(), disabled: !bedModalData },
                            { label: 'Request Labs', icon: '🧪', handler: () => handleRequestLabs(bedModalData?._id, bedModalData?.patientName || 'Patient'), disabled: !bedModalData },
                            { label: 'Administer medication', icon: '💉', handler: () => handleOpenMedicationRequest(), disabled: !bedModalData || bedModalData.status === 'DISCHARGED' },
                            { label: 'Administer Surgery', icon: '🏥', handler: () => handleAdministerSurgery(bedModalData?._id, bedModalData?.patientName || 'Patient'), disabled: !bedModalData || bedModalData.status === 'DISCHARGED' },
                            { label: 'Discharge Patient', icon: '🚪', handler: () => handleDischarge(bedModalData?._id, bedModalData?.patientName || 'Patient'), disabled: !bedModalData || bedModalData.status === 'DISCHARGED' },
                            { label: 'Transfer Specialty Care', icon: '🩺', handler: () => handleTransferSpecialtyCare(bedModalData?._id, bedModalData?.patientName || 'Patient'), disabled: !bedModalData }
                          ].map((item, idx) => (
                            <button
                              key={idx}
                              type="button"
                              onClick={item.handler}
                              disabled={item.disabled}
                              title={item.label}
                              aria-label={item.label}
                              style={{
                                width: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                gap: '10px',
                                padding: '16px 18px',
                                borderRadius: '18px',
                                border: item.disabled ? '1px solid rgba(148, 163, 184, 0.25)' : '1px solid rgba(59,130,246,0.36)',
                                background: item.disabled ? 'rgba(15, 23, 42, 0.68)' : 'linear-gradient(135deg, rgba(16, 185, 129, 0.22), rgba(14, 165, 233, 0.16))',
                                color: item.disabled ? '#8b98a6' : '#eff6ff',
                                cursor: item.disabled ? 'not-allowed' : 'pointer',
                                fontWeight: 700,
                                fontSize: '14px',
                                textAlign: 'left'
                              }}
                            >
                              <span>{item.icon}</span>
                              <span>{item.label}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div style={{
                      padding: '22px',
                      borderRadius: '24px',
                      background: 'linear-gradient(180deg, rgba(99, 102, 241, 0.16), rgba(59, 130, 246, 0.12))',
                      border: '2px solid rgba(99, 102, 241, 0.3)',
                      boxShadow: '0 8px 24px rgba(99, 102, 241, 0.15)'
                    }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '12px',
                        marginBottom: '16px',
                        paddingBottom: '12px',
                        borderBottom: '1px solid rgba(255,255,255,0.12)'
                      }}>
                        <div>
                          <p style={{ margin: 0, fontSize: '14px', color: '#a5b4fc', letterSpacing: '0.14em', textTransform: 'uppercase', fontWeight: 700 }}>
                            📋 Latest Inpatient Report
                          </p>
                        </div>
                        <div style={{ fontSize: '12px', color: '#a5b4fc', fontWeight: 600 }}>
                          {bedModalData?.progression_notes?.[0]?.recorded_at ?
                            new Date(bedModalData.progression_notes[0].recorded_at).toLocaleString('en-US', {
                              month: 'short',
                              day: 'numeric',
                              year: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit'
                            }) : null}
                        </div>
                      </div>
                      <div style={{ marginTop: '14px', display: 'grid', gap: '12px', color: '#e2e8f0', fontSize: '13px', lineHeight: 1.6 }}>
                        <div>
                          <div style={{ color: '#60a5fa', fontSize: '12px', fontWeight: 700, marginBottom: '6px' }}>Assessment</div>
                          <div style={{ color: '#cbd5e1' }}>{bedModalData?.diagnosis || 'No diagnosis documented yet.'}</div>
                        </div>
                        <div>
                          <div style={{ color: '#60a5fa', fontSize: '12px', fontWeight: 700, marginBottom: '6px' }}>Current Plan</div>
                          <div style={{ color: '#cbd5e1' }}>{bedModalData?.treatment_plan || bedModalData?.physician_notes || 'No current treatment plan documented yet.'}</div>
                        </div>
                        <div>
                          <div style={{ color: '#60a5fa', fontSize: '12px', fontWeight: 700, marginBottom: '6px' }}>Vitals</div>
                          <div style={{ color: '#cbd5e1', fontSize: '12px' }}>
                            BP: {modalVitals?.bp || '—'} • HR: {modalVitals?.hr || '—'} • Temp: {modalVitals?.temp || '—'} • O₂: {modalVitals?.o2 || '—'}
                          </div>
                        </div>
                        {latestProgressNote?.content && (
                          <div>
                            <div style={{ color: '#60a5fa', fontSize: '12px', fontWeight: 700, marginBottom: '6px' }}>Latest Progress</div>
                            <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: 1.5 }}>{latestProgressNote.content}</div>
                          </div>
                        )}
                        {bedModalData?.treatments?.length > 0 && (
                          <div>
                            <div style={{ color: '#60a5fa', fontSize: '12px', fontWeight: 700, marginBottom: '6px' }}>Active Treatments</div>
                            <div style={{ display: 'grid', gap: '4px' }}>
                              {bedModalData.treatments.slice(0, 2).map((t, idx) => (
                                <div key={idx} style={{ color: '#cbd5e1', fontSize: '12px' }}>
                                  • {t.name} {t.dosage ? `(${t.dosage})` : ''} · {t.frequency || 'as prescribed'}
                                </div>
                              ))}
                              {bedModalData.treatments.length > 2 && (
                                <div style={{ color: '#60a5fa', fontSize: '11px', marginTop: '4px' }}>
                                  +{bedModalData.treatments.length - 2} more...
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    <div style={{
                      padding: '18px',
                      borderRadius: '20px',
                      background: 'rgba(15,23,42,0.88)',
                      border: '1px solid rgba(255,255,255,0.08)'
                    }}>
                      <button
                        onClick={() => handleViewInpatientReport(bedModalData)}
                        disabled={!bedModalData || !isEncounterHospitalized(bedModalData)}
                        style={{
                          width: '100%',
                          padding: '14px 16px',
                          borderRadius: '16px',
                          border: 'none',
                          background: !bedModalData || !isEncounterHospitalized(bedModalData) ? '#4b5563' : '#2563eb',
                          color: 'white',
                          fontWeight: 700,
                          cursor: !bedModalData || !isEncounterHospitalized(bedModalData) ? 'not-allowed' : 'pointer'
                        }}
                      >
                        View Inpatient Report
                      </button>
                      <button
                        onClick={() => {
                          setSelectedBedModal(null);
                          setBedModalData(null);
                        }}
                        style={{
                          width: '100%',
                          marginTop: '12px',
                          padding: '14px 16px',
                          borderRadius: '16px',
                          border: '1px solid rgba(255,255,255,0.12)',
                          background: 'rgba(255,255,255,0.06)',
                          color: '#cbd5e1',
                          fontWeight: 700,
                          cursor: 'pointer'
                        }}
                      >
                        Close
                      </button>
                    </div>
                  </aside>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {showDischargeModal && bedModalData && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(2, 8, 23, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1300,
          padding: '20px'
        }}>
          <div style={{
            width: '100%',
            maxWidth: '500px',
            padding: '24px',
            borderRadius: '24px',
            background: 'linear-gradient(180deg, rgba(7, 18, 36, 0.98), rgba(4, 11, 25, 0.98))',
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow: '0 24px 80px rgba(0,0,0,0.45)'
          }}>
            <p style={{ margin: 0, fontSize: '12px', color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.14em', fontWeight: 700 }}>
              Schedule Discharge
            </p>
            <h3 style={{ margin: '12px 0 0 0', color: '#f8fafc', fontSize: '22px' }}>
              Discharge patient from bed
            </h3>
            <p style={{ margin: '10px 0 0 0', color: '#cbd5e1', fontSize: '14px', lineHeight: 1.6 }}>
              Select a date and time to schedule discharge. The system will block discharge if pending urgent or emergency surgery is present.
            </p>
            <div style={{ marginTop: '20px', display: 'grid', gap: '16px' }}>
              <label style={{ display: 'grid', gap: '8px', color: '#cbd5e1', fontSize: '13px' }}>
                Scheduled discharge
                <input
                  type="datetime-local"
                  value={dischargeForm.scheduled_at}
                  onChange={e => setDischargeForm(prev => ({ ...prev, scheduled_at: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    borderRadius: '14px',
                    border: '1px solid rgba(148,163,184,0.25)',
                    background: 'rgba(15,23,42,0.95)',
                    color: '#f8fafc'
                  }}
                />
              </label>
              <label style={{ display: 'grid', gap: '8px', color: '#cbd5e1', fontSize: '13px' }}>
                Notes
                <textarea
                  rows={3}
                  value={dischargeForm.disposition_notes}
                  onChange={e => setDischargeForm(prev => ({ ...prev, disposition_notes: e.target.value }))}
                  placeholder="Add optional discharge notes..."
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    borderRadius: '14px',
                    border: '1px solid rgba(148,163,184,0.25)',
                    background: 'rgba(15,23,42,0.95)',
                    color: '#f8fafc',
                    resize: 'vertical'
                  }}
                />
              </label>

              <div style={{ padding: '18px', borderRadius: '18px', background: '#0f172a', border: '1px solid rgba(56,189,248,0.18)' }}>
                <div style={{ marginBottom: '10px', fontSize: '12px', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.12em' }}>Discharge readiness checks</div>
                <div style={{ display: 'grid', gap: '10px', color: '#cbd5e1', fontSize: '13px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                    <span>Pending lab requests</span>
                    <strong>{bedModalData?.lab_requests?.filter((item) => item && !['COMPLETED', 'DECLINED'].includes((item.status || '').toUpperCase())).length || 0}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                    <span>Ordered meds not yet administered</span>
                    <strong>{bedModalData?.medication_administrations?.filter((item) => item && !['ADMINISTERED', 'COMPLETED', 'DECLINED'].includes((item.status || '').toUpperCase())).length || 0}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                    <span>Pending medication orders</span>
                    <strong>{bedModalData?.medication_requests?.filter((item) => item && !['COMPLETED', 'DECLINED'].includes((item.status || '').toUpperCase())).length || 0}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                    <span>Urgent/Emergency surgeries pending</span>
                    <strong>{bedModalData?.surgery_records?.filter((item) => item && ['URGENT', 'EMERGENCY'].includes((item.surgery_type || '').toUpperCase()) && !['COMPLETED', 'CANCELLED'].includes((item.status || '').toUpperCase())).length || 0}</strong>
                  </div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>
                    Pending labs and medications can often be managed after discharge if clinically appropriate. Discharge is only blocked when urgent or emergency surgery is still pending.
                  </div>
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '22px' }}>
              <button
                onClick={() => setShowDischargeModal(false)}
                style={{
                  padding: '12px 16px',
                  borderRadius: '999px',
                  border: '1px solid rgba(255,255,255,0.12)',
                  background: 'rgba(255,255,255,0.06)',
                  color: '#cbd5e1',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitDischarge}
                disabled={submittingDischarge || !dischargeForm.scheduled_at}
                style={{
                  padding: '12px 16px',
                  borderRadius: '999px',
                  border: 'none',
                  background: submittingDischarge || !dischargeForm.scheduled_at ? '#4b5563' : '#2563eb',
                  color: 'white',
                  cursor: submittingDischarge || !dischargeForm.scheduled_at ? 'not-allowed' : 'pointer'
                }}
              >
                {submittingDischarge ? 'Scheduling…' : 'Schedule Discharge'}
              </button>
            </div>
            {dischargeChecks && (
              <div style={{ marginTop: '16px', padding: '16px', borderRadius: '18px', background: dischargeChecks.blocked ? 'rgba(254, 226, 226, 0.9)' : 'rgba(220, 248, 198, 0.9)', border: dischargeChecks.blocked ? '1px solid #fca5a5' : '1px solid #34d399', color: dischargeChecks.blocked ? '#b91c1c' : '#166534' }}>
                <div style={{ fontWeight: 700, marginBottom: '8px' }}>
                  {dischargeChecks.blocked ? 'Discharge check blocked' : 'Discharge check completed'}
                </div>
                <div style={{ fontSize: '13px', lineHeight: 1.6 }}>
                  {dischargeChecks.message}
                </div>
                {dischargeChecks.details?.blockingSurgeries?.length > 0 && (
                  <div style={{ marginTop: '10px' }}>
                    <div style={{ fontWeight: 700 }}>Urgent/Emergency surgeries still pending:</div>
                    <ul style={{ margin: '8px 0 0 16px', padding: 0, listStyleType: 'disc' }}>
                      {dischargeChecks.details.blockingSurgeries.map((surgery, index) => (
                        <li key={index} style={{ marginBottom: '6px', fontSize: '13px' }}>
                          {surgery.procedure_name || 'Scheduled surgery'} • {surgery.surgery_type} • {new Date(surgery.scheduled_at).toLocaleString()}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {showSeverityConfirmModal && bedModalData && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(2, 8, 23, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1300,
          padding: '20px'
        }}>
          <div style={{
            width: '100%',
            maxWidth: '420px',
            padding: '24px',
            borderRadius: '24px',
            background: 'linear-gradient(180deg, rgba(7, 18, 36, 0.98), rgba(4, 11, 25, 0.98))',
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow: '0 24px 80px rgba(0,0,0,0.45)'
          }}>
            <p style={{ margin: 0, fontSize: '12px', color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.14em', fontWeight: 700 }}>
              Confirm severity change
            </p>
            <h3 style={{ margin: '12px 0 0 0', color: '#f8fafc', fontSize: '20px' }}>
              Change patient severity?
            </h3>
            <p style={{ margin: '10px 0 0 0', color: '#cbd5e1', fontSize: '14px', lineHeight: 1.6 }}>
              This will reduce the patient&apos;s priority status. Continue with the severity change?
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '20px' }}>
              <button
                onClick={cancelSeverityChange}
                style={{
                  padding: '10px 14px',
                  borderRadius: '999px',
                  border: '1px solid rgba(255,255,255,0.12)',
                  background: 'rgba(255,255,255,0.06)',
                  color: '#cbd5e1',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmSeverityChange}
                disabled={severitySaving}
                style={{
                  padding: '10px 14px',
                  borderRadius: '999px',
                  border: 'none',
                  background: severitySaving ? '#4b5563' : '#2563eb',
                  color: 'white',
                  fontWeight: 700,
                  cursor: severitySaving ? 'not-allowed' : 'pointer'
                }}
              >
                {severitySaving ? 'Updating...' : 'Confirm'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ===== MEDICATION ADMINISTRATION MODAL ===== */}
      {showMedicationModal && bedModalData && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(6, 15, 35, 0.88)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1100,
          backdropFilter: 'blur(8px)',
          padding: '20px',
          overflowY: 'auto'
        }}
        onClick={() => setShowMedicationModal(false)}>
          <div style={{
            background: 'linear-gradient(180deg, rgba(7, 18, 36, 0.98), rgba(4, 11, 25, 0.98))',
            borderRadius: '24px',
            boxShadow: '0 32px 110px rgba(0,0,0,0.55)',
            width: '100%',
            maxWidth: '600px',
            maxHeight: '90vh',
            overflow: 'auto',
            border: '1px solid rgba(14, 165, 233, 0.18)',
            padding: '28px'
          }}
          onClick={(e) => e.stopPropagation()}>
            
            {/* Modal Header */}
            <div style={{ marginBottom: '28px', paddingBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.12)' }}>
              <h2 style={{ margin: 0, fontSize: '24px', color: '#f8fafc', fontWeight: 700 }}>
                💉 Create Medication Request
              </h2>
              <p style={{ margin: '8px 0 0 0', color: '#cbd5e1', fontSize: '14px' }}>
                Create a medication request for this patient and route it to the diagnosis workflow for pharmacy review.
              </p>
              <div style={{ marginTop: '12px', padding: '12px', borderRadius: '12px', background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)' }}>
                <p style={{ margin: 0, color: '#60a5fa', fontSize: '13px', fontWeight: 600 }}>
                  Patient: <strong>{bedModalData?.patientName || 'Unknown'}</strong>
                </p>
              </div>
            </div>

            {/* Form */}
            <div style={{ display: 'grid', gap: '20px', marginBottom: '24px' }}>
              
              {/* Select Medication from Inventory */}
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Medication *
                </label>
                <select
                  value={medicationForm.medicationId}
                  onChange={(e) => {
                    const selected = pharmacyInventory.find(m => m.id === e.target.value);
                    if (selected) {
                      setMedicationForm({
                        ...medicationForm,
                        medicationId: selected.id,
                        medicationName: selected.name
                      });
                    }
                  }}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px',
                    fontWeight: 500,
                    cursor: 'pointer'
                  }}
                >
                  <option value="">-- Select from inventory --</option>
                  {pharmacyInventory.map(med => (
                    <option key={med.id} value={med.id}>
                      {med.name} ({med.strength ? `${med.strength}` : ''}) • Stock: {med.quantity}
                    </option>
                  ))}
                </select>
                <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
                  {pharmacyInventory.length} medications available in pharmacy inventory
                </p>
              </div>

              {/* Dosage */}
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Dosage *
                </label>
                <input
                  type="text"
                  placeholder="e.g., 500mg, 2 tablets, 5ml"
                  value={medicationForm.dosage}
                  onChange={(e) => setMedicationForm({ ...medicationForm, dosage: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px'
                  }}
                />
              </div>

              {/* Frequency & Route */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                    Frequency
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., Q4-6h, BID, TID"
                    value={medicationForm.frequency}
                    onChange={(e) => setMedicationForm({ ...medicationForm, frequency: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '12px',
                      borderRadius: '12px',
                      border: '1px solid rgba(255,255,255,0.12)',
                      background: 'rgba(15,23,42,0.88)',
                      color: '#f8fafc',
                      fontSize: '14px'
                    }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                    Route
                  </label>
                  <select
                    value={medicationForm.route}
                    onChange={(e) => setMedicationForm({ ...medicationForm, route: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '12px',
                      borderRadius: '12px',
                      border: '1px solid rgba(255,255,255,0.12)',
                      background: 'rgba(15,23,42,0.88)',
                      color: '#f8fafc',
                      fontSize: '14px',
                      cursor: 'pointer'
                    }}
                  >
                    <option>Oral</option>
                    <option>IV</option>
                    <option>IM</option>
                    <option>SC</option>
                    <option>Topical</option>
                    <option>Inhalation</option>
                  </select>
                </div>
              </div>

              {/* Indication */}
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Indication (Why?)
                </label>
                <input
                  type="text"
                  placeholder="Clinical reason for this medication"
                  value={medicationForm.indication}
                  onChange={(e) => setMedicationForm({ ...medicationForm, indication: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px'
                  }}
                />
              </div>

              {/* Prescription Notes */}
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Prescription Notes
                </label>
                <textarea
                  placeholder="Special instructions, warnings, or contraindications..."
                  value={medicationForm.prescriptionNotes}
                  onChange={(e) => setMedicationForm({ ...medicationForm, prescriptionNotes: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px',
                    minHeight: '80px',
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
              </div>

              {/* Administration Time */}
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Administration Time *
                </label>
                <input
                  type="datetime-local"
                  value={medicationForm.administrationTime}
                  onChange={(e) => setMedicationForm({ ...medicationForm, administrationTime: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px',
                    cursor: 'pointer'
                  }}
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <button
                onClick={() => setShowMedicationModal(false)}
                style={{
                  padding: '14px 18px',
                  borderRadius: '12px',
                  border: '1px solid rgba(255,255,255,0.12)',
                  background: 'rgba(255,255,255,0.06)',
                  color: '#cbd5e1',
                  fontWeight: 700,
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleAdministerMedication}
                disabled={administeingMed}
                style={{
                  padding: '14px 18px',
                  borderRadius: '12px',
                  border: 'none',
                  background: administeingMed ? '#475569' : '#10b981',
                  color: 'white',
                  fontWeight: 700,
                  cursor: administeingMed ? 'wait' : 'pointer',
                  fontSize: '14px',
                  opacity: administeingMed ? 0.7 : 1
                }}
              >
                {administeingMed ? '⏳ Sending...' : '✓ Send Request'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ===== LAB REQUEST MODAL ===== */}
      {showLabRequestModal && bedModalData && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(6, 15, 35, 0.88)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1150,
          backdropFilter: 'blur(8px)',
          padding: '20px',
          overflowY: 'auto'
        }}
        onClick={() => setShowLabRequestModal(false)}>
          <div style={{
            background: 'linear-gradient(180deg, rgba(7, 18, 36, 0.98), rgba(4, 11, 25, 0.98))',
            borderRadius: '24px',
            boxShadow: '0 32px 110px rgba(0,0,0,0.55)',
            width: '100%',
            maxWidth: '620px',
            maxHeight: '90vh',
            overflow: 'auto',
            border: '1px solid rgba(14, 165, 233, 0.18)',
            padding: '28px'
          }}
          onClick={(e) => e.stopPropagation()}>
            <div style={{ marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.12)' }}>
              <h2 style={{ margin: 0, fontSize: '24px', color: '#f8fafc', fontWeight: 700 }}>🧪 Request Labs</h2>
              <p style={{ margin: '10px 0 0 0', color: '#cbd5e1', fontSize: '14px' }}>
                Send a specialty lab order to Radiology or Pathology for this patient.
              </p>
              <div style={{ marginTop: '14px', padding: '14px', borderRadius: '14px', background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.14)' }}>
                <p style={{ margin: 0, color: '#60a5fa', fontSize: '13px', fontWeight: 700 }}>
                  Patient: <strong>{bedModalData?.patientName || bedModalData?.patient?.name || 'Unknown'}</strong>
                </p>
              </div>
            </div>

            <div style={{ display: 'grid', gap: '18px', marginBottom: '24px' }}>
              <div style={{ display: 'grid', gap: '10px' }}>
                <label style={{ color: '#f8fafc', fontSize: '13px', fontWeight: 700, textTransform: 'uppercase' }}>Specialty</label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  {['RADIOLOGY', 'PATHOLOGY'].map((option) => (
                    <button
                      key={option}
                      type="button"
                      onClick={() => {
                        const nextType = LAB_TYPE_OPTIONS[option][0] || '';
                        setLabRequestForm(prev => ({
                          ...prev,
                          specialty: option,
                          lab_type: prev.specialty === option ? prev.lab_type : nextType
                        }));
                      }}
                      style={{
                        padding: '14px 16px',
                        borderRadius: '14px',
                        border: labRequestForm.specialty === option ? '1px solid #38bdf8' : '1px solid rgba(255,255,255,0.12)',
                        background: labRequestForm.specialty === option ? 'rgba(56, 189, 248, 0.18)' : 'rgba(255,255,255,0.04)',
                        color: '#f8fafc',
                        cursor: 'pointer',
                        fontWeight: 700,
                        minHeight: '48px'
                      }}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Test / Procedure Type
                </label>
                <select
                  value={labRequestForm.lab_type}
                  onChange={(e) => setLabRequestForm(prev => ({ ...prev, lab_type: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px',
                    cursor: 'pointer'
                  }}
                >
                  <option value="">-- Select a lab type --</option>
                  {(LAB_TYPE_OPTIONS[labRequestForm.specialty] || []).map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Priority
                </label>
                <select
                  value={labRequestForm.priority}
                  onChange={(e) => setLabRequestForm(prev => ({ ...prev, priority: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px',
                    cursor: 'pointer'
                  }}
                >
                  <option value="URGENT">URGENT</option>
                  <option value="HIGH">HIGH</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="LOW">LOW</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Clinical Indication
                </label>
                <textarea
                  placeholder="Describe the reason for this lab request, urgency, and relevant clinical context."
                  value={labRequestForm.description}
                  onChange={(e) => setLabRequestForm(prev => ({ ...prev, description: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px',
                    minHeight: '110px',
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <button
                  type="button"
                  onClick={() => setShowLabRequestModal(false)}
                  style={{
                    padding: '14px 18px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(255,255,255,0.06)',
                    color: '#cbd5e1',
                    fontWeight: 700,
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSubmitLabRequest}
                  disabled={submittingLabRequest}
                  style={{
                    padding: '14px 18px',
                    borderRadius: '12px',
                    border: 'none',
                    background: submittingLabRequest ? '#475569' : '#22c55e',
                    color: 'white',
                    fontWeight: 700,
                    cursor: submittingLabRequest ? 'wait' : 'pointer',
                    fontSize: '14px',
                    opacity: submittingLabRequest ? 0.8 : 1
                  }}
                >
                  {submittingLabRequest ? '⏳ Sending request...' : '✓ Submit Lab Request'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== MEDICATION ORDER MODAL ===== */}
      {showMedicationOrderModal && bedModalData && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(6, 15, 35, 0.88)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1150,
          backdropFilter: 'blur(8px)',
          padding: '20px',
          overflowY: 'auto'
        }}
        onClick={() => setShowMedicationOrderModal(false)}>
          <div style={{
            background: 'linear-gradient(180deg, rgba(7, 18, 36, 0.98), rgba(4, 11, 25, 0.98))',
            borderRadius: '24px',
            border: '1px solid rgba(59, 130, 246, 0.2)',
            padding: '48px',
            maxWidth: '600px',
            width: '100%',
            boxShadow: '0 25px 50px rgba(0, 0, 0, 0.5)',
            color: '#f8fafc'
          }} onClick={e => e.stopPropagation()}>
            <h2 style={{ margin: 0, fontSize: '24px', color: '#f8fafc', fontWeight: 700, marginBottom: '24px' }}>💊 Order Medication</h2>
            <div style={{ display: 'grid', gap: '20px', marginBottom: '24px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Medication Name *
                </label>
                <select
                  value={medicationOrderForm.medicationId}
                  onChange={(e) => {
                    const selected = pharmacyInventory.find(item => item.id === e.target.value);
                    setMedicationOrderForm(prev => ({
                      ...prev,
                      medicationId: selected?.id || '',
                      medication_name: selected?.name || prev.medication_name
                    }));
                  }}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px'
                  }}
                >
                  <option value="">Select medication from hospital inventory</option>
                  {pharmacyInventory.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name} — {item.strength || item.dosageForm || 'form'} ({item.quantity} available)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Dosage
                </label>
                <input
                  type="text"
                  placeholder="e.g., 500mg, 1 tablet, 10ml"
                  value={medicationOrderForm.dosage}
                  onChange={(e) => setMedicationOrderForm(prev => ({ ...prev, dosage: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px'
                  }}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                    Frequency
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., Every 6 hours, Once daily"
                    value={medicationOrderForm.frequency}
                    onChange={(e) => setMedicationOrderForm(prev => ({ ...prev, frequency: e.target.value }))}
                    style={{
                      width: '100%',
                      padding: '12px',
                      borderRadius: '12px',
                      border: '1px solid rgba(255,255,255,0.12)',
                      background: 'rgba(15,23,42,0.88)',
                      color: '#f8fafc',
                      fontSize: '14px'
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                    Schedule
                  </label>
                  <select
                    value={medicationOrderForm.schedule}
                    onChange={(e) => setMedicationOrderForm(prev => ({ ...prev, schedule: e.target.value }))}
                    style={{
                      width: '100%',
                      padding: '12px',
                      borderRadius: '12px',
                      border: '1px solid rgba(255,255,255,0.12)',
                      background: 'rgba(15,23,42,0.88)',
                      color: '#f8fafc',
                      fontSize: '14px',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="Schedule 1">Schedule 1</option>
                    <option value="Schedule 2">Schedule 2</option>
                    <option value="Schedule 3">Schedule 3</option>
                    <option value="Schedule 4">Schedule 4</option>
                    <option value="Schedule 5">Schedule 5</option>
                    <option value="OTC">OTC</option>
                  </select>
                </div>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 700, color: '#f8fafc', textTransform: 'uppercase' }}>
                  Clinical Indication
                </label>
                <textarea
                  placeholder="Describe the reason for this medication order and any relevant clinical notes."
                  value={medicationOrderForm.indication}
                  onChange={(e) => setMedicationOrderForm(prev => ({ ...prev, indication: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(15,23,42,0.88)',
                    color: '#f8fafc',
                    fontSize: '14px',
                    minHeight: '100px',
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <button
                  type="button"
                  onClick={() => setShowMedicationOrderModal(false)}
                  style={{
                    padding: '14px 18px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(255,255,255,0.06)',
                    color: '#cbd5e1',
                    fontWeight: 700,
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSubmitMedicationOrder}
                  disabled={submittingMedicationOrder}
                  style={{
                    padding: '14px 18px',
                    borderRadius: '12px',
                    border: 'none',
                    background: submittingMedicationOrder ? '#475569' : '#06b6d4',
                    color: 'white',
                    fontWeight: 700,
                    cursor: submittingMedicationOrder ? 'wait' : 'pointer',
                    fontSize: '14px',
                    opacity: submittingMedicationOrder ? 0.8 : 1
                  }}
                >
                  {submittingMedicationOrder ? '⏳ Sending order...' : '✓ Order Medication'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PhysicianDashboard;