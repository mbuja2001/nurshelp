import React, { useState, useEffect, useRef, useId } from 'react';
// Staff view for non-admins
import ClockInOutBoard from './ClockInOutBoard';
import { FiX, FiPlus, FiSearch, FiFilter, FiClock, FiUsers, FiSliders, FiSave, FiActivity, FiRefreshCw, FiDatabase, FiShield, FiCheckCircle, FiAlertTriangle, FiChevronLeft, FiChevronRight } from 'react-icons/fi';
import { decodeJwt } from '../../utils/jwt';
import SidebarLayout from '../SidebarLayout';
import BedsAllocation from './BedsAllocation';
import Pharmacy from './Pharmacy';
import TriageAnalytics from './TriageAnalytics';
import LabRequestsPage from './LabRequestsPage';
import AdminAnalyticsPage from './AdminAnalyticsPage';
import AdminLabRequestsPage from './AdminLabRequestsPage';
import AdminPharmacyPage from './AdminPharmacyPage';
import SupportTicketInbox from './SupportTicketInbox';
import { getAdminWorkspaceViewRegistration } from '../../utils/adminDashboardViews';
import { HOSPITAL_DEPARTMENTS, HOSPITAL_SPECIALTIES } from '../../config/hospitalStructureCatalog';
import {
  getAllowedShiftDuties,
  HOSPITAL_STAFF_ROLES,
  resolveShiftAssignmentDuty,
  SHIFT_DUTY_LABELS
} from '../../config/hospitalStaffRoleCatalog';
import {
  getConfiguredDepartmentSpecialties,
  resolveConfiguredDepartmentName,
  resolveStaffPositionCode
} from '../../utils/hospitalStaffOnboarding';
import { ehrConfigurationBlueprint, protocolAdapterInterface } from './ehrConfigBlueprint';

const OverviewDiagnosisPieChart = ({ data }) => {
  const [animateSlices, setAnimateSlices] = useState(false);
  const chartRef = useRef(null);
  const colors = ['#0369a1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6b7280'];

  useEffect(() => {
    setAnimateSlices(false);
    const chart = chartRef.current;
    if (!chart || typeof IntersectionObserver === 'undefined') {
      setAnimateSlices(true);
      return undefined;
    }

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setAnimateSlices(true);
        observer.disconnect();
      }
    }, { threshold: 0.35 });

    observer.observe(chart);
    return () => observer.disconnect();
  }, [data]);

  const items = Array.isArray(data) ? data.slice(0, 5) : [];
  const total = items.reduce((sum, item) => sum + Number(item.count || 0), 0);

  if (!items.length || total <= 0) {
    return <div style={{ width: 180, height: 180, flex: '0 0 180px', borderRadius: '50%', border: '8px dashed #cbd5e1', display: 'grid', placeItems: 'center', color: '#64748b', fontSize: 13, fontWeight: 800 }}>n/a</div>;
  }

  let currentAngle = 0;
  const radius = 40;
  const polarToCartesian = (centerX, centerY, radiusValue, angleInDegrees) => {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180;
    return {
      x: centerX + radiusValue * Math.cos(angleInRadians),
      y: centerY + radiusValue * Math.sin(angleInRadians)
    };
  };

  const describeSlice = (startAngle, sliceAngle) => {
    if (sliceAngle >= 359.999) return null;
    const start = polarToCartesian(50, 50, radius, startAngle);
    const end = polarToCartesian(50, 50, radius, startAngle + sliceAngle);
    const largeArc = sliceAngle > 180 ? 1 : 0;
    return `M 50 50 L ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y} Z`;
  };

  const slices = items.map((item, index) => {
    const sliceAngle = (Number(item.count || 0) / total) * 360;
    const startAngle = currentAngle;
    currentAngle += sliceAngle;
    const labelAngle = startAngle + sliceAngle / 2;
    const labelPoint = polarToCartesian(50, 50, 27, labelAngle);
    return {
      color: colors[index % colors.length],
      path: describeSlice(startAngle, sliceAngle),
      label: `${Number(item.percentage || 0).toFixed(1)}%`,
      labelX: labelPoint.x,
      labelY: labelPoint.y,
    };
  });

  return (
    <div ref={chartRef} style={{ width: 180, height: 180, flex: '0 0 180px' }}>
      <svg width={180} height={180} viewBox="0 0 100 100" role="img" aria-label="Diagnosis distribution chart">
        <circle cx="50" cy="50" r={radius} fill="#e2e8f0" />
        {slices.map((slice, index) => (
          <g key={index} style={{ opacity: animateSlices ? 1 : 0, transform: animateSlices ? 'scale(1)' : 'scale(0)', transformOrigin: '50px 50px', transition: `opacity 0.25s ease ${index * 100}ms, transform 1s cubic-bezier(.2,.9,.2,1) ${index * 100}ms` }}>
            {slice.path ? <path d={slice.path} fill={slice.color} /> : <circle cx="50" cy="50" r={radius} fill={slice.color} />}
            <text x={slice.labelX} y={slice.labelY} textAnchor="middle" dominantBaseline="middle" style={{ opacity: animateSlices ? 1 : 0, transition: `opacity 0.3s ease ${index * 100 + 700}ms`, fontSize: 7, fontWeight: 800, fill: '#0f172a' }}>{slice.label}</text>
          </g>
        ))}
      </svg>
    </div>
  );
};

const createStaffDraft = () => ({
  firstName: '', lastName: '', email: '', password: '', positionCode: 'TRIAGE_NURSE', role: 'NURSE', department: '',
  specialty: '', title: 'RN', licenseNumber: '', npiNumber: ''
});

const createShiftDraft = () => ({
  shiftID: '', shiftName: '', department: 'ED', date: new Date().toISOString().split('T')[0],
  shiftStart: '08:00', shiftEnd: '16:00', maxStaffRequired: 1, staffIds: []
});

const PasswordField = ({ label, value, onChange, placeholder = '', style = {} }) => {
  const [showPassword, setShowPassword] = useState(false);
  const inputId = useId();

  return (
    <div style={style}>
      <label htmlFor={inputId} style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
        {label}
      </label>
      <div style={{ position: 'relative' }}>
        <input
          id={inputId}
          type={showPassword ? 'text' : 'password'}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          style={{ ...policyInputStyle, paddingRight: 64 }}
        />
        <button
          type="button"
          onClick={() => setShowPassword((current) => !current)}
          aria-label={showPassword ? 'Hide password' : 'Show password'}
          style={{
            position: 'absolute',
            top: '50%',
            right: 10,
            transform: 'translateY(-50%)',
            border: 0,
            background: 'transparent',
            padding: '4px 2px',
            fontSize: 12,
            color: '#0f172a',
            fontWeight: 700,
            cursor: 'pointer'
          }}
        >
          {showPassword ? 'Hide' : 'Show'}
        </button>
      </div>
    </div>
  );
};

const createHospitalSetupSample = ({ staffCount = 49 } = {}) => {
  const suffix = `${Date.now()}${Math.floor(Math.random() * 90 + 10)}`;
  const managerPassword = `Admin!${suffix.slice(-8)}`;
  const staffPassword = `Staff!${suffix.slice(-8)}`;
  const legacyStaffTemplates = [
    ['Sarah Chen', 'PHYSICIAN', 'GENERAL_WARD', 'Internal Medicine'], ['Raj Patel', 'PHYSICIAN', 'GENERAL_WARD', 'General Surgery'],
    ['Maria Garcia', 'PHYSICIAN', 'EMERGENCY', 'Emergency Medicine'], ['James Wilson', 'PHYSICIAN', 'PEDIATRICS', 'Pediatrics'],
    ['Lisa Anderson', 'PHYSICIAN', 'CARDIOLOGY', 'Cardiology'], ['Robert Brown', 'PHYSICIAN', 'RADIOLOGY', 'Radiology'],
    ['Emily Davis', 'PHYSICIAN', 'PEDIATRICS', 'Neonatal Medicine'], ['Michael Torres', 'PHYSICIAN', 'ICU', 'Pediatric ICU'],
    ['Jennifer Lee', 'PHYSICIAN', 'CARDIOLOGY', 'Cardiothoracic Surgery'], ['David Kumar', 'PHYSICIAN', 'ICU', 'Neurosurgery'],
    ['Amanda Wright', 'PHYSICIAN', 'ICU', 'Pulmonology'], ['Christopher Martinez', 'PHYSICIAN', 'GENERAL_WARD', 'Nephrology'],
    ['Patricia Johnson', 'PHYSICIAN', 'GENERAL_WARD', 'Orthopedics'], ['Daniel O\'Brien', 'PHYSICIAN', 'GENERAL_WARD', 'Gastroenterology'],
    ['Susan Blake', 'PHYSICIAN', 'GENERAL_WARD', 'Neurology'], ['George Harris', 'PHYSICIAN', 'GENERAL_WARD', 'Oncology'],
    ['Diana Turner', 'PHYSICIAN', 'GENERAL_WARD', 'Infectious Disease'], ['Paul Thompson', 'PHYSICIAN', 'GENERAL_WARD', 'Geriatrics'],
    ['Helen White', 'PHYSICIAN', 'GENERAL_WARD', 'OBGYN'], ['Kevin Zhang', 'PHYSICIAN', 'GENERAL_WARD', 'Psychiatry'],
    ['Nancy Garcia', 'PHYSICIAN', 'GENERAL_WARD', 'Clinical Pathology'], ['Emma Thompson', 'NURSE', 'EMERGENCY', 'Emergency Nursing'],
    ['Michael Chen', 'NURSE', 'EMERGENCY', 'Emergency Nursing'], ['Amanda Foster', 'NURSE', 'ICU', 'Critical Care Nursing'],
    ['Jessica Lee', 'NURSE', 'PEDIATRICS', 'Neonatal Nursing'], ['Thomas Martinez', 'NURSE', 'PEDIATRICS', 'Pediatric Nursing'],
    ['Sophie Robinson', 'NURSE', 'ICU', 'ICU Nursing'], ['David Kumar', 'NURSE', 'ICU', 'Neurosurgical Nursing'],
    ['Helen White', 'NURSE', 'CARDIOLOGY', 'Cardiac Nursing'], ['Oliver Green', 'NURSE', 'GENERAL_WARD', 'General Nursing'],
    ['Victoria Brown', 'NURSE', 'GENERAL_WARD', 'General Nursing'], ['Rachel Murphy', 'NURSE', 'GENERAL_WARD', 'General Nursing'],
    ['Marcus Johnson', 'NURSE', 'OR', 'Operating Room Nursing'], ['Carla Smith', 'NURSE', 'GENERAL_WARD', 'Orthopedic Nursing'],
    ['James Wilson', 'NURSE', 'GENERAL_WARD', 'General Nursing'], ['Patricia Davis', 'NURSE', 'GENERAL_WARD', 'General Nursing'],
    ['Robert Miller', 'NURSE', 'GENERAL_WARD', 'Oncology Nursing'], ['Linda Anderson', 'NURSE', 'GENERAL_WARD', 'Infection Control Nursing'],
    ['George Taylor', 'NURSE', 'GENERAL_WARD', 'Geriatric Nursing'], ['Susan Harris', 'NURSE', 'GENERAL_WARD', 'Maternity Nursing'],
    ['Frank Jackson', 'NURSE', 'GENERAL_WARD', 'Psychiatric Nursing'], ['Peter Mitchell', 'RADIOLOGIST', 'RADIOLOGY', 'X-Ray Technician'],
    ['Andrew Morris', 'RADIOLOGIST', 'RADIOLOGY', 'CT Technician'], ['Jennifer Garcia', 'RADIOLOGIST', 'RADIOLOGY', 'MRI Technician'],
    ['Marcus White', 'RADIOLOGIST', 'RADIOLOGY', 'Ultrasound Technician'], ['Leonard Roberts', 'PATHOLOGIST', 'GENERAL_WARD', 'Lab Technician'],
    ['Victoria King', 'PATHOLOGIST', 'GENERAL_WARD', 'Hematology Technician'], ['Mark Johnson', 'PHARMACIST', 'GENERAL_WARD', 'Clinical Pharmacist'],
    ['Nancy Davis', 'PHARMACIST', 'GENERAL_WARD', 'Hospital Pharmacist']
  ].map(([fullName, role, department, specialty]) => {
    const nameParts = fullName.split(' ');
    const positionCodeByRole = {
      NURSE: 'REGISTERED_NURSE',
      PHYSICIAN: 'PHYSICIAN',
      TECHNICIAN: 'TECHNICIAN',
      ADMINISTRATIVE: 'ADMINISTRATIVE_SUPPORT',
      PHARMACIST: 'PHARMACIST',
      RADIOLOGIST: 'RADIOLOGIST',
      PATHOLOGIST: 'PATHOLOGIST'
    };
    const departmentAliases = {
      EMERGENCY: 'ED',
      GENERAL_WARD: 'General',
      CARDIOLOGY: 'General',
      RADIOLOGY: 'General'
    };
    return {
      firstName: nameParts.shift(),
      lastName: nameParts.join(' '),
      role,
      positionCode: positionCodeByRole[role] || 'REGISTERED_NURSE',
      department: departmentAliases[department] || department,
      specialty,
      title: role === 'NURSE' ? 'RN' : role === 'PHARMACIST' || role === 'PATHOLOGIST' ? 'PhD' : 'MD'
    };
  });
  const staff = Array.from({ length: Math.max(1, Number(staffCount) || 1) }, (_, index) => {
    const template = legacyStaffTemplates[index % legacyStaffTemplates.length];
    const cycle = Math.floor(index / legacyStaffTemplates.length);
    const firstName = cycle ? `${template.firstName}${cycle + 1}` : template.firstName;
    return {
      ...template,
      firstName,
      email: `${firstName}.${template.lastName}.${suffix}.${index + 1}@example.test`.toLowerCase(),
      password: staffPassword,
      licenseNumber: `${template.title}-${suffix.slice(-6)}-${String(index + 1).padStart(2, '0')}`
    };
  });

  return {
    hospital: {
      hospitalName: `Northstar Medical Center ${suffix.slice(-4)}`,
      managerName: 'Amelia',
      managerSurname: 'Carter',
      managerEmail: `amelia.carter.${suffix}@example.test`.toLowerCase(),
      managerPassword,
      department: 'All',
      departments: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'General']
    },
    staff
  };
};

const StaffScheduleManager = ({ onLogout = null }) => {
  const token = localStorage.getItem('token');
  const decoded = decodeJwt(token);
  const userRole = decoded?.role || 'NURSE';
  const isAdmin = userRole === 'ADMIN' || userRole === 'MANAGER';
  const isCompanySuperAdmin = (decoded?.role || '').toUpperCase() === 'SUPERUSER' || (decoded?.email || '').toLowerCase() === 'superadmin@nurshelpdesk.com' || new URLSearchParams(window.location.search).get('internal_company_login') === 'true';
  
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

  // ============ STATE ============
  const [activeTab, setActiveTab] = useState(isAdmin ? 'hospital-operations' : 'schedule'); // unified shift management view
  const [targetDate, setTargetDate] = useState(new Date().toISOString().split('T')[0]);
  const [systemMode, setSystemMode] = useState(() => {
    try {
      const fallbackMode = isCompanySuperAdmin ? 'live' : 'mock';
      return localStorage.getItem('nursehelpdesk:system-mode') || fallbackMode;
    } catch (error) {
      return isCompanySuperAdmin ? 'live' : 'mock';
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem('nursehelpdesk:system-mode', systemMode);
    } catch (error) {
      console.warn('[SYSTEM MODE] Unable to persist system mode:', error);
    }
  }, [systemMode]);
  const isLiveSystemMode = systemMode === 'live';
  const isMockSystemMode = systemMode === 'mock';
  
  // Staff search state
  const [staffList, setStaffList] = useState([]);
  const [filteredStaff, setFilteredStaff] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('all');
  const [roleFilter, setRoleFilter] = useState('all');
  const [searchLoading, setSearchLoading] = useState(false);

  // Schedule view state
  const [todaySchedule, setTodaySchedule] = useState([]);
  const [scheduleLoading, setScheduleLoading] = useState(false);
  const [selectedShiftId, setSelectedShiftId] = useState('');
  const [assignmentStaffId, setAssignmentStaffId] = useState('');
  const [assignmentRole, setAssignmentRole] = useState('');
  const [editingShiftId, setEditingShiftId] = useState('');
  const [editingShiftName, setEditingShiftName] = useState('');

  // Add shift form state (admin only)
  const [shiftForm, setShiftForm] = useState({
    shiftName: '',
    shiftStart: '08:00',
    shiftEnd: '16:00',
    shiftDate: new Date().toISOString().split('T')[0], // Today's date YYYY-MM-DD
    maxStaffRequired: 1,
    assignedStaff: []
  });
  
  // Role assignments (integrated into shift creation)
  const [roleAssignments, setRoleAssignments] = useState({
    triageNurseId: '',
    clinicalNurseId: '',
    clinicalNurseContext: 'CLINICAL_NURSE_ER',
    physicianId: '',
    physicianContext: 'PHYSICIAN_ER'
  });
  
  const [selectedStaffForShift, setSelectedStaffForShift] = useState([]);
  const [shiftMessages, setShiftMessages] = useState({ success: '', error: '' });
  
  // Per-staff role assignment mapping: { [staffId]: { role: 'TRIAGE_NURSE'|'CLINICAL_NURSE'|'PHYSICIAN', context: 'CLINICAL_NURSE_ER'|... } }
  const [selectedStaffRoles, setSelectedStaffRoles] = useState({});
  
  // Staff assignment search/filter state
  const [staffSearchQuery, setStaffSearchQuery] = useState('');
  const [staffDepartmentFilter, setStaffDepartmentFilter] = useState('all');
  const [staffRoleFilter, setStaffRoleFilter] = useState('all');

  const [bedConfigurations, setBedConfigurations] = useState([]);
  const [bedConfigurationsLoading, setBedConfigurationsLoading] = useState(false);
  const [bedConfigurationSaving, setBedConfigurationSaving] = useState(false);
  const [bedEditor, setBedEditor] = useState(null);

  // Messages
  const [messages, setMessages] = useState({ success: '', error: '' });
  const [hospitalActivity, setHospitalActivity] = useState(null);
  const [hospitalActivityLoading, setHospitalActivityLoading] = useState(false);
  const [hospitalActivityOpen, setHospitalActivityOpen] = useState(false);
  const [hospitalStructure, setHospitalStructure] = useState({ departments: [], departmentOptions: [], specialties: [], staffRoles: HOSPITAL_STAFF_ROLES });
  const [hospitalStructureDraft, setHospitalStructureDraft] = useState([]);
  const [hospitalStructureDepartmentName, setHospitalStructureDepartmentName] = useState('');
  const [hospitalStructureDepartmentError, setHospitalStructureDepartmentError] = useState('');
  const [hospitalStructureLoading, setHospitalStructureLoading] = useState(false);
  const [hospitalStructureSaving, setHospitalStructureSaving] = useState(false);
  const [hospitalStructureWizardOpen, setHospitalStructureWizardOpen] = useState(false);
  const [hospitalStructureStep, setHospitalStructureStep] = useState(0);
  const [hospitalStructureDepartmentIndex, setHospitalStructureDepartmentIndex] = useState(0);
  const [hospitalStructureWardIndex, setHospitalStructureWardIndex] = useState(0);
  const [hospitalWards, setHospitalWards] = useState([]);
  const [hospitalWardDrafts, setHospitalWardDrafts] = useState([]);
  const [hospitalWardsLoading, setHospitalWardsLoading] = useState(false);
  const [hospitalWardsSaving, setHospitalWardsSaving] = useState(false);
  const [hospitalDayRunning, setHospitalDayRunning] = useState(false);
  const [shiftSession, setShiftSession] = useState(null);
  const shiftSessionLog = (hospitalActivity?.activity || [])
    .filter((event) => ['HOSPITAL_SHIFT_STARTED', 'HOSPITAL_SHIFT_ENDED'].includes(event.action))
    .map((event) => ({
      event: event.action.endsWith('STARTED') ? 'STARTED' : 'ENDED',
      ...(event.changeDetails || {}),
      shiftName: event.changeDetails?.shiftName || 'Hospital shift',
      startedAt: event.timestamp,
      endedAt: event.timestamp
    }));
  const [hospitalDayWizardOpen, setHospitalDayWizardOpen] = useState(false);
  const [hospitalDayStep, setHospitalDayStep] = useState(0);
  const [daySystemReadiness, setDaySystemReadiness] = useState({
    beds: true,
    staff: true,
    imaging: true,
    lab: true,
    pharmacy: true,
    telemetry: true
  });
  const [dayShiftAssignments, setDayShiftAssignments] = useState({
    triageNurseId: '',
    pharmacistId: '',
    radiologistId: '',
    pathologistId: ''
  });
  const [daySelectedStaffIds, setDaySelectedStaffIds] = useState([]);
  const [daySelectedShiftId, setDaySelectedShiftId] = useState('');
  const [dayShiftMode, setDayShiftMode] = useState('existing');
  const [dayShiftName, setDayShiftName] = useState('Hospital Day Coverage');
  const [dayShiftStart, setDayShiftStart] = useState('08:00');
  const [dayShiftEnd, setDayShiftEnd] = useState('16:00');
  const [dayClockedInStaff, setDayClockedInStaff] = useState([]);
  const [staffActivityLoading, setStaffActivityLoading] = useState(false);
  const [hospitalDayPatientRows, setHospitalDayPatientRows] = useState([
    { name: 'Amina Okafor', note: '' },
    { name: 'Daniel Mensah', note: '' },
    { name: 'Grace Williams', note: '' }
  ]);
  const activeShiftSession = shiftSession || (hospitalDayRunning ? {
    shiftName: 'Active hospital shift',
    department: 'Hospital operations',
    shiftStart: '--:--',
    shiftEnd: '--:--',
    startedAt: null,
    staffCount: 0
  } : null);

  const isShiftCurrent = (shift) => {
    if (!shift?.shiftStart || !shift?.shiftEnd) return false;
    const now = new Date();
    const currentMinutes = now.getHours() * 60 + now.getMinutes();
    const [startHour, startMinute] = shift.shiftStart.split(':').map(Number);
    const [endHour, endMinute] = shift.shiftEnd.split(':').map(Number);
    const startMinutes = startHour * 60 + startMinute;
    const endMinutes = endHour * 60 + endMinute;
    return startMinutes <= endMinutes
      ? currentMinutes >= startMinutes && currentMinutes <= endMinutes
      : currentMinutes >= startMinutes || currentMinutes <= endMinutes;
  };

  const getShiftDuration = (start, end) => {
    if (!start || !end) return 0;
    const [startHour, startMinute] = start.split(':').map(Number);
    const [endHour, endMinute] = end.split(':').map(Number);
    let minutes = (endHour * 60 + endMinute) - (startHour * 60 + startMinute);
    if (minutes <= 0) minutes += 24 * 60;
    return Number((minutes / 60).toFixed(2));
  };

  const buildDefaultCoverageAssignments = (roster = []) => {
    const nextAssignments = { triageNurseId: '', pharmacistId: '', radiologistId: '', pathologistId: '' };
    const matchRole = (patterns) => {
      const staff = roster.find((person) => {
        const role = (person.systemRole || person.role || person.userRole || '').toUpperCase();
        const specialty = (person.clinicalSpecialty || person.specialty || '').toUpperCase();
        return patterns.some((pattern) => role.includes(pattern) || specialty.includes(pattern));
      });
      return staff?._id || '';
    };

    nextAssignments.triageNurseId = matchRole(['TRIAGE_NURSE', 'NURSE', 'EMERGENCY']);
    nextAssignments.pharmacistId = matchRole(['PHARMACIST']);
    nextAssignments.radiologistId = matchRole(['RADIOLOGIST', 'RADIOLOGY']);
    nextAssignments.pathologistId = matchRole(['PATHOLOGIST', 'LAB']);

    return nextAssignments;
  };

  // Workload policy state. The API remains the source of truth and enforces RBAC.
  const [capacityConfig, setCapacityConfig] = useState(null);
  const [capacityPolicyLoading, setCapacityPolicyLoading] = useState(false);
  const [pharmacyInventory, setPharmacyInventory] = useState([]);
  const [pharmacyInventoryLoading, setPharmacyInventoryLoading] = useState(false);
  const [capacityPolicySaving, setCapacityPolicySaving] = useState(false);
  const [capacityRole, setCapacityRole] = useState('TRIAGE_NURSE');
  const [capacityValue, setCapacityValue] = useState('');
  const [savedCapacityValue, setSavedCapacityValue] = useState(null);
  const [acuityWeights, setAcuityWeights] = useState({ RED: 1, ORANGE: 1, YELLOW: 1, GREEN: 1 });
  const [acuityValidationMessage, setAcuityValidationMessage] = useState('');
  const [acuitySaving, setAcuitySaving] = useState(false);
  const [hospitalOperationsStatus, setHospitalOperationsStatus] = useState(null);
  const [hospitalOperationsLoading, setHospitalOperationsLoading] = useState(false);
  const [hospitalOperationsAction, setHospitalOperationsAction] = useState('');
  const [hospitalOperationsEncounterCount, setHospitalOperationsEncounterCount] = useState(8);
  const [hospitalOperationsHistoricalDays, setHospitalOperationsHistoricalDays] = useState(30);
  const [hospitalOperationsPatientNames, setHospitalOperationsPatientNames] = useState('Amina Okafor\nDaniel Mensah\nGrace Williams');
  const [hospitalOperationsComplaintType, setHospitalOperationsComplaintType] = useState('FEVER');
  const [hospitalOperationsCustomComplaint, setHospitalOperationsCustomComplaint] = useState('');
  const [hospitalOperationsPriority, setHospitalOperationsPriority] = useState('MIXED');
  const [hospitalOperationsLifecycle, setHospitalOperationsLifecycle] = useState('AWAITING_PHYSICIAN');
  const [hospitalOperationsTriageNurseId, setHospitalOperationsTriageNurseId] = useState('');
  const [hospitalOperationsStaffRows, setHospitalOperationsStaffRows] = useState([createStaffDraft()]);
  const [staffEditorOpen, setStaffEditorOpen] = useState(false);
  const [staffEditorSaving, setStaffEditorSaving] = useState(false);
  const [staffEditorError, setStaffEditorError] = useState('');
  const [staffEditor, setStaffEditor] = useState(null);
  const [staffOnboardingOpen, setStaffOnboardingOpen] = useState(false);
  const [hospitalCredentialsOpen, setHospitalCredentialsOpen] = useState(false);
  const [workloadPolicyOpen, setWorkloadPolicyOpen] = useState(false);
  const [hospitalOperationsShiftRows, setHospitalOperationsShiftRows] = useState([createShiftDraft()]);
  const [managedHospitals, setManagedHospitals] = useState([]);
  const [managedHospitalLoading, setManagedHospitalLoading] = useState(false);
  const [workspaceView, setWorkspaceView] = useState('overview');
  const [ehrConfig, setEhrConfig] = useState({
    adapters: [
      { id: 'hl7v2', name: 'HL7v2', status: 'Enabled', connection: 'Healthy', host: 'mlp.lab.internal', port: '2575', lastTest: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), inboundFailures: 2, outboundRoutes: ['Encounter intake', 'Lab results'], messageCount: 1183 },
      { id: 'cda', name: 'CDA', status: 'Planned', connection: 'Not configured', host: '—', port: '—', lastTest: null, inboundFailures: 0, outboundRoutes: ['Not scheduled'], messageCount: 0 },
      { id: 'dicom', name: 'DICOM', status: 'Queued', connection: 'Planned', host: '—', port: '—', lastTest: null, inboundFailures: 0, outboundRoutes: ['Not scheduled'], messageCount: 0 }
    ],
    deadLetter: {
      inboundFailures: 2,
      retryQueue: 1,
      failedMessages: [
        { id: 'HL7-2026-1001', reason: 'Malformed PID segment', payload: 'MSH|^~\\&|PASS|HOSP1|...|PID|1||ABC123||DOE^JOHN|...\nPV1|1|I|ICU^101||||...', status: 'Queued for review' },
        { id: 'HL7-2026-1002', reason: 'Unmapped terminology code: UNMAPPED-17', payload: 'OBX|1|NM|^GLUCOSE|89|mg/dL|...\nOBX|2|NM|UNMAPPED-17|4.9|...', status: 'Needs review' }
      ],
      retryItems: [
        { id: 'RETRY-17', reason: 'Temporary upstream socket timeout', action: 'Retry', queueSlot: 'Manual review queue' }
      ]
    },
    retention: [
      { id: 'primary-records', category: 'Primary clinical record', scope: 'Primary record', window: '4 years', minimum: '6 years', compliant: false },
      { id: 'lab-results', category: 'Lab results', scope: 'Processing artifact', window: '2 years', minimum: '3 years', compliant: false },
      { id: 'audit-logs', category: 'Audit logs', scope: 'Primary record', window: '7 years', minimum: '6 years', compliant: true },
      { id: 'location-pings', category: 'Location pings', scope: 'Processing artifact', window: '90 days', minimum: '90 days', compliant: true }
    ],
    terminologyRows: [
      { code: 'RED', status: 'Mapped', value: 'Priority = RED', color: '#ef4444' },
      { code: 'ORANGE', status: 'Mapped', value: 'Priority = ORANGE', color: '#f97316' },
      { code: 'YELLOW', status: 'Mapped', value: 'Priority = YELLOW', color: '#eab308' },
      { code: 'GREEN', status: 'Mapped', value: 'Priority = GREEN', color: '#22c55e' },
      { code: 'UNMAPPED-17', status: 'Needs review', value: 'Missing SNOMED crosswalk', color: '#a855f7', needsAction: true }
    ],
    accessMatrix: [
      { role: 'Physician', canView: true, canEdit: true, scope: 'Full encounter access' },
      { role: 'Nurse', canView: true, canEdit: false, scope: 'Ward-scoped records' },
      { role: 'Pharmacist', canView: true, canEdit: false, scope: 'Medication records' },
      { role: 'Radiologist', canView: true, canEdit: false, scope: 'Imaging review' },
      { role: 'Pathologist', canView: true, canEdit: false, scope: 'Lab review' },
      { role: 'Technician', canView: true, canEdit: false, scope: 'Workflow tasks' },
      { role: 'Administrative', canView: true, canEdit: false, scope: 'Scheduling and admin' }
    ]
  });
  const [ehrSelectedFailure, setEhrSelectedFailure] = useState(null);
  const [ehrSelectedMapping, setEhrSelectedMapping] = useState(null);
  const [ehrToast, setEhrToast] = useState(null);
  const canEditEhr = ['ADMIN', 'SUPERUSER', 'MANAGER'].includes((decoded?.role || '').toUpperCase());
  const [ehrRetryReview, setEhrRetryReview] = useState(null);
  const [newHospitalEhrState, setNewHospitalEhrState] = useState('new');
  const [ehrConnectionTick, setEhrConnectionTick] = useState(0);
  const [ehrConfigMode, setEhrConfigMode] = useState('partial');
  const [ehrLastAction, setEhrLastAction] = useState('');

  const formatRelativeTime = (dateValue) => {
    if (!dateValue) return 'No test yet';
    const diffMs = Date.now() - new Date(dateValue).getTime();
    const diffMinutes = Math.max(1, Math.round(diffMs / 60000));
    if (diffMinutes < 60) return `${diffMinutes} minute${diffMinutes === 1 ? '' : 's'} ago`;
    const diffHours = Math.round(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
    const diffDays = Math.round(diffHours / 24);
    return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
  };

  const adapterStateStyles = {
    Enabled: { background: '#dcfce7', color: '#166534', border: '1px solid #86efac', icon: '●' },
    Healthy: { background: '#dcfce7', color: '#166534', border: '1px solid #86efac', icon: '●' },
    Planned: { background: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1', icon: '○' },
    Queued: { background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d', icon: '◔' },
    Degraded: { background: '#fef2f2', color: '#b91c1c', border: '1px solid #fca5a5', icon: '◉' },
    'Not configured': { background: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1', icon: '○' }
  };

  const getEhrStatusStyle = (value = 'Enabled') => adapterStateStyles[value] || adapterStateStyles.Enabled;

  const updateEhrAdapter = (id, updater) => {
    setEhrConfig((current) => ({
      ...current,
      adapters: current.adapters.map((adapter) => (adapter.id === id ? { ...adapter, ...updater(adapter) } : adapter))
    }));
  };

  const handleTestConnection = (adapterId) => {
    updateEhrAdapter(adapterId, (adapter) => ({
      connection: 'Healthy',
      status: adapter.id === 'hl7v2' ? 'Enabled' : adapter.status,
      lastTest: new Date().toISOString()
    }));
    setEhrToast({ tone: 'success', message: `${adapterId.toUpperCase()} connection test passed.` });
    setEhrLastAction(`Tested ${adapterId.toUpperCase()} connection`);
  };

  const handleRetryQueueAction = (itemId, action) => {
    setEhrConfig((current) => ({
      ...current,
      deadLetter: {
        ...current.deadLetter,
        retryQueue: Math.max(0, current.deadLetter.retryQueue - (action === 'retry' ? 1 : 0)),
        failedMessages: current.deadLetter.failedMessages.filter((message) => message.id !== itemId)
      }
    }));
    setEhrToast({ tone: action === 'retry' ? 'success' : 'warning', message: action === 'retry' ? 'Queued message re-submitted.' : 'Message routed to permanent failure.' });
    setEhrLastAction(action === 'retry' ? 'Retry queued message' : 'Marked message as permanent failure');
  };

  const handleEhrMappingSave = (code) => {
    setEhrToast({ tone: 'success', message: `Terminology mapping for ${code} saved.` });
    setEhrLastAction(`Saved terminology mapping for ${code}`);
  };

  const handleRetentionSave = (id) => {
    setEhrToast({ tone: 'success', message: `Retention policy ${id} saved.` });
    setEhrLastAction(`Saved retention policy ${id}`);
  };

  const handleAccessMatrixSave = (role) => {
    setEhrToast({ tone: 'success', message: `Access matrix for ${role} saved.` });
    setEhrLastAction(`Saved access matrix for ${role}`);
  };

  const handleEhrSettingsSave = () => {
    setEhrToast({ tone: 'success', message: 'EHR settings saved successfully.' });
    setEhrLastAction('Saved EHR settings');
  };

  const clearEhrToast = () => setEhrToast(null);

  const openHospitalDayWizard = async () => {
    console.log('[HOSPITAL DAY] Opening start-day wizard', {
      targetDate,
      hospitalId: decoded?.hospitalId,
      currentStaffCount: staffList.length,
      currentShiftCount: todaySchedule.length
    });
    setHospitalDayStep(0);
    setHospitalDayWizardOpen(true);
    try {
      const [status, activity, capacity, staffResults, shifts, clockedInResults] = await Promise.all([
        fetchHospitalOperationsStatus(),
        fetchHospitalActivity(),
        fetchCapacityPolicy(),
        fetchStaff(),
        fetchTodaySchedule(),
        fetchCurrentStaffStatus()
      ]);

      const roster = staffResults?.length ? staffResults : (filteredStaff.length ? filteredStaff : staffList);
      const currentShift = (shifts || todaySchedule).find(isShiftCurrent) || (shifts || todaySchedule).find((shift) => shift.status === 'PUBLISHED');
      console.log('[HOSPITAL DAY] Readiness data loaded', {
        staffCount: roster.length,
        scheduledShiftCount: (shifts || todaySchedule).length,
        selectedExistingShift: currentShift ? { id: currentShift._id, shiftID: currentShift.shiftID, name: currentShift.shiftName, start: currentShift.shiftStart, end: currentShift.shiftEnd, assigned: currentShift.assignedStaff?.length || 0 } : null,
        clockedInCount: clockedInResults?.length || 0
      });
      setDaySelectedShiftId(currentShift?._id || '');
      setDayShiftMode(currentShift ? 'existing' : 'new');
      if (currentShift) {
        setDayShiftName(currentShift.shiftName || 'Hospital Day Coverage');
        setDayShiftStart(currentShift.shiftStart || '08:00');
        setDayShiftEnd(currentShift.shiftEnd || '16:00');
      }
      const existingAssignedIds = (currentShift?.assignedStaff || []).map((staff) => String(staff.staffID?._id || staff.staffID));
      const defaults = buildDefaultCoverageAssignments(roster);
      setDaySelectedStaffIds(existingAssignedIds.length ? existingAssignedIds : [...new Set(Object.values(defaults).filter(Boolean).map(String))]);
      setDayShiftAssignments((current) => ({
        ...current,
        ...Object.fromEntries(Object.entries(defaults).filter(([, value]) => value))
      }));
    } catch (error) {
      setMessages({ success: '', error: error.message || 'Unable to refresh hospital day readiness.' });
    }
  };

  const fetchCurrentStaffStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/manager/staff-status`, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load current staff status (${response.status})`);
      setDayClockedInStaff(data.staff || []);
      return data.staff || [];
    } catch (error) {
      setDayClockedInStaff([]);
      setMessages({ success: '', error: error.message });
      return [];
    }
  };

  const recordHospitalShiftEvent = async (event, session) => {
    const response = await fetch(`${BACKEND_URL}/api/admin/hospital-operations/shift-events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ event, shift: session })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) throw new Error(data.message || `Unable to record shift ${event.toLowerCase()} event (${response.status})`);
    console.log('[HOSPITAL ACTIVITY] Shift lifecycle event recorded', { event, auditId: data.event?._id });
    return data.event;
  };

  const refreshStaffActivity = async () => {
    if (!isAdmin || !token) return [];
    setStaffActivityLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/manager/staff-status`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load staff clock activity (${response.status})`);
      setDayClockedInStaff(data.staff || []);
      return data.staff || [];
    } catch (error) {
      setMessages({ success: '', error: error.message });
      return [];
    } finally {
      setStaffActivityLoading(false);
    }
  };

  const autoClockOutStaff = async () => {
    const response = await fetch(`${BACKEND_URL}/api/staff/auto-clockout`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) throw new Error(data.message || `Unable to auto-clock out staff (${response.status})`);
    await refreshStaffActivity();
    await fetchHospitalActivity();
    return data;
  };

  const closeHospitalDayWizard = () => {
    setHospitalDayWizardOpen(false);
    setHospitalDayStep(0);
  };

  const pullShiftCoverageRoster = async () => {
    try {
      await Promise.all([fetchStaff(), fetchTodaySchedule()]);
      setMessages({ success: 'Live staff roster and shift coverage refreshed from the shift tab.', error: '' });
      return true;
    } catch (error) {
      setMessages({ success: '', error: error.message || 'Unable to refresh the live shift roster.' });
      return false;
    }
  };

  const createCoverageShiftSchedule = async () => {
    console.log('[HOSPITAL DAY] Start requested', {
      mode: dayShiftMode,
      selectedShiftId: daySelectedShiftId,
      selectedStaffCount: daySelectedStaffIds.length,
      selectedStaffIds: daySelectedStaffIds,
      shift: { name: dayShiftName, start: dayShiftStart, end: dayShiftEnd, duration: getShiftDuration(dayShiftStart, dayShiftEnd) }
    });
    if (dayShiftMode === 'existing' && !daySelectedShiftId) {
      console.warn('[HOSPITAL DAY] Cannot start: no existing shift selected');
      setMessages({ success: '', error: 'Select a scheduled shift or choose Create new shift before starting the hospital day.' });
      return false;
    }
    let selectedIds = daySelectedStaffIds.filter(Boolean).map(String);
    if (!selectedIds.length) {
      const defaults = buildDefaultCoverageAssignments(filteredStaff.length ? filteredStaff : staffList);
      selectedIds = [...new Set(Object.values(defaults).filter(Boolean).map(String))];
      setDaySelectedStaffIds(selectedIds);
      setDayShiftAssignments((current) => ({ ...current, ...defaults }));
    }

    if (!selectedIds.length) {
      setMessages({ success: '', error: 'Select at least one staff member for coverage before starting the hospital day.' });
      return false;
    }

    const roster = (filteredStaff.length ? filteredStaff : staffList).filter((person) =>
      selectedIds.includes(String(person._id))
    );

    console.log('[HOSPITAL DAY] Resolved selected roster', {
      requestedCount: selectedIds.length,
      matchedCount: roster.length,
      matchedStaff: roster.map((person) => ({ id: person._id, name: `${person.firstName || person.name || ''} ${person.lastName || person.surname || ''}`.trim(), role: person.systemRole || person.role }))
    });

    if (!roster.length) {
      setMessages({ success: '', error: 'No live staff roster records matched the selected coverage assignments.' });
      return false;
    }

    try {
      const dutyByHospitalDayAssignment = {
        triageNurseId: 'TRIAGE_NURSE',
        pharmacistId: 'PHARMACIST',
        radiologistId: 'RADIOLOGIST',
        pathologistId: 'PATHOLOGIST',
      };
      const assignedStaffData = roster.map((person) => {
        const personId = String(person._id);
        const selectedDuty = Object.entries(dayShiftAssignments)
          .find(([field, assignedId]) => dutyByHospitalDayAssignment[field] && String(assignedId) === personId);
        const assignmentRole = resolveShiftAssignmentDuty(
          person,
          selectedDuty ? dutyByHospitalDayAssignment[selectedDuty[0]] : ''
        );
        return {
          staffID: person._id,
          type: (person.systemRole || person.role || 'NURSE').toUpperCase().replace(/\s+/g, '_'),
          assignmentRole,
          name: `${person.firstName || person.name || ''}`.trim(),
          surname: `${person.lastName || person.surname || ''}`.trim(),
          specialty: person.specialty || person.clinicalSpecialty || '',
          status: 'ASSIGNED'
        };
      });
      const staffWithoutDuty = assignedStaffData.find((assignment) => !assignment.assignmentRole);
      if (staffWithoutDuty) {
        const staffMember = roster.find((person) => String(person._id) === String(staffWithoutDuty.staffID));
        const staffName = `${staffMember?.firstName || staffMember?.name || ''} ${staffMember?.lastName || staffMember?.surname || ''}`.trim();
        throw new Error(`Select a shift duty supported by ${staffName || 'each selected staff member'}'s job position before starting the hospital day.`);
      }

      const shiftDate = targetDate || new Date().toISOString().split('T')[0];
      const existingShift = dayShiftMode === 'existing' ? todaySchedule.find((shift) => String(shift._id) === String(daySelectedShiftId)) : null;

      if (dayShiftMode === 'existing' && !existingShift) {
        throw new Error('The selected scheduled shift is no longer available. Refresh the shift list and try again.');
      }

      if (existingShift) {
        console.log('[HOSPITAL DAY] Reusing scheduled shift', {
          id: existingShift._id,
          shiftID: existingShift.shiftID,
          name: existingShift.shiftName,
          window: `${existingShift.shiftStart}-${existingShift.shiftEnd}`,
          alreadyAssigned: existingShift.assignedStaff?.length || 0,
          toAssign: roster.length
        });
        const assignedIds = new Set((existingShift.assignedStaff || []).map((staff) => String(staff.staffID?._id || staff.staffID)));
        for (const staff of roster.filter((person) => !assignedIds.has(String(person._id)))) {
          console.log('[HOSPITAL DAY] Assigning staff to existing shift', { shiftID: existingShift.shiftID, staffId: staff._id, name: staff.firstName || staff.name });
          const assignResponse = await fetch(`${BACKEND_URL}/api/shifts/${existingShift.shiftID}/assign-staff`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({
              staffID: staff._id,
              assignmentRole: assignedStaffData.find((assignment) => String(assignment.staffID) === String(staff._id))?.assignmentRole,
              staffType: (() => {
                const role = (staff.systemRole || staff.role || 'NURSE').toUpperCase().replace(/\s+/g, '_');
                return ['MANAGER', 'NURSE', 'PHYSICIAN', 'SPECIALIST', 'SUPPORT'].includes(role) ? role : 'SPECIALIST';
              })(),
              name: staff.firstName || staff.name || '',
              surname: staff.lastName || staff.surname || '',
              specialty: staff.specialty || staff.clinicalSpecialty || ''
            })
          });
          const assignData = await assignResponse.json().catch(() => ({}));
          console.log('[HOSPITAL DAY] Existing-shift assignment response', { shiftID: existingShift.shiftID, staffId: staff._id, status: assignResponse.status, success: assignData.success, message: assignData.message });
          if (!assignResponse.ok || !assignData.success) throw new Error(assignData.message || `Staff assignment failed (${assignResponse.status})`);
        }
        await fetchTodaySchedule();
        setMessages({ success: `Using ${existingShift.shiftName}; ${roster.length} selected staff assigned to the scheduled shift.`, error: '' });
        return existingShift;
      }

      const shiftPayload = {
        shiftName: dayShiftName.trim() || 'Hospital Day Coverage',
        shiftStart: dayShiftStart,
        shiftEnd: dayShiftEnd,
        date: shiftDate,
        maxStaffRequired: assignedStaffData.length,
        assignedStaff: assignedStaffData
      };

      console.log('[HOSPITAL DAY] Creating new shift payload', shiftPayload);

      const res = await fetch(`${BACKEND_URL}/api/shifts/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(shiftPayload)
      });

      const data = await res.json();
      console.log('[HOSPITAL DAY] Shift creation response', { status: res.status, success: data.success, message: data.message, shiftId: data.shift?._id, shiftID: data.shift?.shiftID, assignedCount: data.shift?.assignedStaff?.length });
      if (!res.ok || !data.success) {
        throw new Error(data.message || `Shift creation failed (${res.status})`);
      }

      await fetchTodaySchedule();
      if (!data.shift?._id && !data.shift?.shiftID) {
        throw new Error('The shift was created, but the server did not return its persisted ID. Refresh the schedule before starting the hospital day.');
      }
      console.log('[HOSPITAL DAY] Created coverage shift from live roster:', data.shift._id || data.shift.shiftID);
      return data.shift;
    } catch (error) {
      setMessages({ success: '', error: error.message || 'Failed to create the coverage shift schedule.' });
      return false;
    }
  };

  const startHospitalDay = async () => {
    console.log('[HOSPITAL DAY] Final start action clicked');
    const persistedShift = await createCoverageShiftSchedule();
    console.log('[HOSPITAL DAY] Schedule/assignment result', { shiftId: persistedShift?._id, shiftID: persistedShift?.shiftID });
    if (!persistedShift) {
      console.warn('[HOSPITAL DAY] Start aborted because schedule setup failed');
      return;
    }

    const startedAt = new Date().toISOString();
    const session = {
      id: persistedShift._id,
      shiftID: persistedShift.shiftID,
      shiftName: dayShiftName || 'Hospital Day Coverage',
      shiftStart: dayShiftStart,
      shiftEnd: dayShiftEnd,
      startedAt,
      staffCount: daySelectedStaffIds.length,
      autoClockOutAt: (() => {
        const deadline = new Date(startedAt);
        const [hours, minutes] = dayShiftEnd.split(':').map(Number);
        deadline.setHours(hours, minutes, 0, 0);
        if (deadline <= new Date(startedAt)) deadline.setDate(deadline.getDate() + 1);
        deadline.setMinutes(deadline.getMinutes() + 5);
        return deadline.toISOString();
      })()
    };
    try {
      await recordHospitalShiftEvent('STARTED', session);
      await fetchHospitalActivity();
    } catch (error) {
      console.error('[HOSPITAL DAY] Shift start audit failed', error);
      setMessages({ success: '', error: error.message });
      return;
    }
    setHospitalDayRunning(true);
    setShiftSession(session);
    setHospitalDayWizardOpen(false);
    setWorkspaceView('shifts');
    setActiveTab('shifts');
    setMessages({ success: 'Hospital day started and the coverage shift schedule was created from the live roster.', error: '' });
    console.log('[HOSPITAL DAY] Hospital day marked active', { selectedShiftId: daySelectedShiftId, selectedStaffCount: daySelectedStaffIds.length });
    setHospitalDayStep(0);
  };

  const endHospitalDay = async () => {
    const endedAt = new Date().toISOString();
    try {
      await autoClockOutStaff();
      await recordHospitalShiftEvent('ENDED', { ...(shiftSession || { shiftName: 'Hospital shift', staffCount: 0 }), endedAt });
      await fetchHospitalActivity();
    } catch (error) {
      console.error('[HOSPITAL DAY] Shift end audit failed', error);
      setMessages({ success: '', error: error.message });
      return;
    }
    setHospitalDayRunning(false);
    setShiftSession(null);
    setMessages({ success: 'Hospital shift ended. The shift has been recorded in the shift log.', error: '' });
  };

  useEffect(() => {
    if (!hospitalDayRunning || !shiftSession?.autoClockOutAt) return undefined;
    const delay = new Date(shiftSession.autoClockOutAt).getTime() - Date.now();
    if (delay <= 0) {
      autoClockOutStaff().catch((error) => setMessages({ success: '', error: error.message }));
      return undefined;
    }
    const timer = setTimeout(() => {
      autoClockOutStaff()
        .then(() => setMessages({ success: 'The shift-end grace period elapsed. Active staff were clocked out automatically.', error: '' }))
        .catch((error) => setMessages({ success: '', error: error.message }));
    }, delay);
    return () => clearTimeout(timer);
  }, [hospitalDayRunning, shiftSession?.autoClockOutAt]);

  useEffect(() => {
    if (!hospitalDayRunning || !shiftSession || shiftSession.autoClockOutAt) return undefined;
    const deadline = new Date(shiftSession.startedAt || Date.now());
    const [hours, minutes] = (shiftSession.shiftEnd || '16:00').split(':').map(Number);
    deadline.setHours(hours, minutes, 0, 0);
    if (deadline <= new Date(shiftSession.startedAt || Date.now())) deadline.setDate(deadline.getDate() + 1);
    deadline.setMinutes(deadline.getMinutes() + 5);
    setShiftSession((current) => current ? { ...current, autoClockOutAt: deadline.toISOString() } : current);
    return undefined;
  }, [hospitalDayRunning, shiftSession]);

  useEffect(() => {
    if (workspaceView !== 'ehr-configurations') return undefined;
    const timer = setInterval(() => {
      setEhrConnectionTick((current) => current + 1);
      setEhrConfig((current) => ({
        ...current,
        adapters: current.adapters.map((adapter) => {
          if (adapter.id !== 'hl7v2' || adapter.status === 'Planned') return adapter;
          return {
            ...adapter,
            connection: current.adapters[0]?.connection === 'Healthy' ? 'Healthy' : 'Healthy',
            lastTest: new Date().toISOString()
          };
        })
      }));
    }, 15000);
    return () => clearInterval(timer);
  }, [workspaceView]);

  useEffect(() => {
    if (!ehrToast) return undefined;
    const timeout = setTimeout(() => setEhrToast(null), 2500);
    return () => clearTimeout(timeout);
  }, [ehrToast]);

  const [hospitalCredentials, setHospitalCredentials] = useState({
    name: '',
    surname: '',
    email: '',
    password: '',
    phone: ''
  });
  const [newHospitalName, setNewHospitalName] = useState('');
  const [hospitalBootstrap, setHospitalBootstrap] = useState({
    hospitalName: '',
    managerName: '',
    managerSurname: '',
    managerEmail: '',
    managerPassword: '',
    department: 'All',
    departments: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'General']
  });
  const [bootstrapStaffRows, setBootstrapStaffRows] = useState([
    { firstName: '', lastName: '', email: '', password: '', role: 'NURSE', positionCode: 'TRIAGE_NURSE', department: 'ED', specialty: '' }
  ]);
  const [hospitalSetupOpen, setHospitalSetupOpen] = useState(false);
  const [setupStep, setSetupStep] = useState(0);
  const [setupAutoFilled, setSetupAutoFilled] = useState(false);
  const [setupStaffCount, setSetupStaffCount] = useState(49);
  const [setupEncounterCount, setSetupEncounterCount] = useState(8);
  const [setupHistoricalDays, setSetupHistoricalDays] = useState(30);
  const [overviewDiagnoses, setOverviewDiagnoses] = useState([]);

  // ============ DEPARTMENTS ============
  const DEPARTMENTS = hospitalStructure.departments.filter((department) => department.isActive !== false).map((department) => department.name);
  const ROLES = [...new Set(HOSPITAL_STAFF_ROLES.map((position) => position.systemRole))];

  const fetchCapacityPolicy = async () => {
    setCapacityPolicyLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/capacity-config`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load workload policy (${response.status})`);
      setCapacityConfig(data);
      setAcuityWeights({
        RED: data.policies?.acuityWeightByPriority?.RED ?? 1,
        ORANGE: data.policies?.acuityWeightByPriority?.ORANGE ?? 1,
        YELLOW: data.policies?.acuityWeightByPriority?.YELLOW ?? 1,
        GREEN: data.policies?.acuityWeightByPriority?.GREEN ?? 1
      });
      setSavedCapacityValue(data.policies?.defaultCapacityByRole?.[capacityRole] ?? null);
      setCapacityValue('');
      setAcuityValidationMessage('');
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setCapacityPolicyLoading(false);
    }
  };

  const fetchPharmacyInventory = async () => {
    setPharmacyInventoryLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/pharmacy/inventory`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load pharmacy inventory (${response.status})`);
      setPharmacyInventory(Array.isArray(data.data) ? data.data : []);
    } catch (error) {
      console.error('[PHARMACY INVENTORY] Unable to load overview inventory:', error);
      setPharmacyInventory([]);
    } finally {
      setPharmacyInventoryLoading(false);
    }
  };

  useEffect(() => {
    if (workspaceView === 'overview' && decoded?.hospitalId) fetchPharmacyInventory();
  }, [workspaceView, decoded?.hospitalId]);

  const fetchBedConfigurations = async () => {
    setBedConfigurationsLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/beds/configuration`, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load bed configurations (${response.status})`);
      setBedConfigurations(data.beds || []);
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setBedConfigurationsLoading(false);
    }
  };

  const saveBedConfiguration = async (event) => {
    event.preventDefault();
    if (!bedEditor?.bed_id?.trim() || !bedEditor?.ward_unit) return setMessages({ success: '', error: 'Bed ID and ward are required.' });
    setBedConfigurationSaving(true);
    try {
      const isEditing = Boolean(bedEditor.originalBedId);
      const response = await fetch(`${BACKEND_URL}/api/beds/configuration${isEditing ? `/${encodeURIComponent(bedEditor.originalBedId)}` : ''}`, {
        method: isEditing ? 'PATCH' : 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ ...bedEditor, equipment_baseline: bedEditor.equipmentText.split(',').map((item) => item.trim()).filter(Boolean), special_characteristics: bedEditor.characteristicsText.split(',').map((item) => item.trim()).filter(Boolean) })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to save bed configuration (${response.status})`);
      setBedEditor(null);
      setMessages({ success: isEditing ? `Bed ${bedEditor.bed_id} updated.` : `Bed ${bedEditor.bed_id} created.`, error: '' });
      await fetchBedConfigurations();
      await fetchHospitalOperationsStatus();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setBedConfigurationSaving(false);
    }
  };

  const deleteBedConfiguration = async (bed) => {
    if (!window.confirm(`Delete bed ${bed.bed_id}? Its equipment configuration will also be removed.`)) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/beds/configuration/${encodeURIComponent(bed.bed_id)}`, { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to delete bed (${response.status})`);
      setMessages({ success: data.message || `Bed ${bed.bed_id} deleted.`, error: '' });
      await fetchBedConfigurations();
      await fetchHospitalOperationsStatus();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    }
  };

  const saveHospitalCapacity = async () => {
    console.log('[WORKLOAD POLICY] Saving hospital capacity configuration');
    setCapacityPolicySaving(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/capacity`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ level: 'hospital', role: capacityRole, capacity: capacityValue === '' ? null : Number(capacityValue) })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to save capacity (${response.status})`);
      setMessages({ success: `${capacityRole} hospital capacity saved and audited.`, error: '' });
      setSavedCapacityValue(Number(capacityValue) || null);
      setCapacityValue('');
      await fetchCapacityPolicy();
      await fetchHospitalActivity();
    } catch (error) {
      if (String(error.message).includes('(404)')) {
        setManagedHospitals(decoded?.hospitalId ? [{ _id: decoded.hospitalId, name: decoded.hospitalName || decoded.name || decoded.hospitalId, hospitalID: decoded.hospitalId }] : []);
      } else {
        setMessages({ success: '', error: error.message });
      }
    } finally {
      setCapacityPolicySaving(false);
    }
  };

  const saveHospitalAcuityWeights = async () => {
    console.log('[WORKLOAD POLICY] Saving acuity weights configuration');
    const weights = Object.fromEntries(Object.entries(acuityWeights).map(([key, value]) => [key, Number(value)]));
    const ordered = ['RED', 'ORANGE', 'YELLOW', 'GREEN'];
    const invalid = ordered.find((priority, index) => {
      const current = Number(weights[priority]);
      if (!Number.isFinite(current) || current <= 0) return true;
      const nextPriority = ordered[index + 1];
      if (!nextPriority) return false;
      return current < Number(weights[nextPriority]);
    });
    if (invalid) {
      const message = 'Acuity weights must follow RED >= ORANGE >= YELLOW >= GREEN.';
      setAcuityValidationMessage(message);
      setMessages({ success: '', error: message });
      return;
    }
    setAcuityValidationMessage('');
    setAcuitySaving(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/capacity/acuity-weights`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ weights })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to save acuity weights (${response.status})`);
      setMessages({ success: 'Acuity weights saved and audited.', error: '' });
      await fetchCapacityPolicy();
      await fetchHospitalActivity();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setAcuitySaving(false);
    }
  };

  const fetchHospitalOperationsStatus = async () => {
    setHospitalOperationsLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/admin/hospital-operations/status`, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (response.status === 404) {
          setHospitalOperationsStatus({
            staff: { total: 0 },
            shifts: { today: 0 },
            encounters: { total: 0 },
            beds: { total: 0 },
            pharmaceuticals: { total: 0 },
            readiness: { staff: false, shifts: false, encounters: false, beds: false, pharmaceuticals: false }
          });
          return;
        }
        throw new Error(data.message || `Unable to load hospital operations status (${response.status})`);
      }
      setHospitalOperationsStatus(data);
    } catch (error) {
      if (!String(error.message).includes('(404)')) {
        setMessages({ success: '', error: error.message });
      }
    } finally {
      setHospitalOperationsLoading(false);
    }
  };

  const fetchHospitalActivity = async () => {
    if (!isAdmin || !token) return;
    setHospitalActivityLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/admin/hospital-operations/activity?limit=60`, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load hospital activity (${response.status})`);
      setHospitalActivity(data);
      const latestShiftEvent = (data.activity || []).find((event) => ['HOSPITAL_SHIFT_STARTED', 'HOSPITAL_SHIFT_ENDED'].includes(event.action));
      if (latestShiftEvent?.action === 'HOSPITAL_SHIFT_STARTED') {
        const session = latestShiftEvent.changeDetails || {};
        setHospitalDayRunning(true);
        setShiftSession({
          ...session,
          startedAt: session.startedAt || latestShiftEvent.timestamp,
          autoClockOutAt: session.autoClockOutAt || null
        });
      } else if (latestShiftEvent?.action === 'HOSPITAL_SHIFT_ENDED') {
        setHospitalDayRunning(false);
        setShiftSession(null);
      }
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setHospitalActivityLoading(false);
    }
  };

  const fetchHospitalStructure = async () => {
    setHospitalStructureLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/hospital-structure`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      let structureData = data;
      if (!response.ok && response.status === 404) {
        const legacyResponse = await fetch(`${BACKEND_URL}/api/operations/capacity-config`, {
          headers: { Authorization: `Bearer ${token}` },
          cache: 'no-store'
        });
        const legacyData = await legacyResponse.json().catch(() => ({}));
        if (!legacyResponse.ok) throw new Error(legacyData.message || data.message || 'Unable to load hospital departments');
        structureData = {
          success: true,
          departments: legacyData.departments || [],
          departmentOptions: hospitalStructure.departmentOptions,
          specialties: hospitalStructure.specialties
        };
      } else if (!response.ok) {
        throw new Error(data.message || 'Unable to load hospital departments');
      }
      const currentDepartments = (structureData.departments || []).map((department) => ({
        name: department.name,
        isActive: department.isActive !== false,
        serviceTypes: department.serviceTypes?.length ? department.serviceTypes : ['INPATIENT'],
        specialties: (department.specialties || []).map((specialty) => typeof specialty === 'string' ? specialty : specialty.name)
      }));
      setHospitalStructure({
        departments: currentDepartments,
        departmentOptions: structureData.departmentOptions?.length ? structureData.departmentOptions : HOSPITAL_DEPARTMENTS,
        specialties: structureData.specialties?.length ? structureData.specialties : HOSPITAL_SPECIALTIES,
        staffRoles: structureData.staffRoles?.length ? structureData.staffRoles : HOSPITAL_STAFF_ROLES
      });
      setHospitalStructureDraft(currentDepartments);
      await fetchHospitalWards(currentDepartments);
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setHospitalStructureLoading(false);
    }
  };

  const fetchHospitalWards = async (departments = hospitalStructureDraft) => {
    setHospitalWardsLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/hospital-wards`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to load hospital wards');
      const wards = data.wards || [];
      setHospitalWards(wards);
      setHospitalWardDrafts(wards.map((ward) => ({
        wardId: ward.wardId,
        wardName: ward.wardName,
        departmentName: ward.departmentName || departments.find((department) => department.name === ward.department)?.name || departments[0]?.name || '',
        careTier: ward.careTier || 'GENERAL_INPATIENT',
        acuityLevelAllowed: ward.acuityLevelAllowed?.length ? ward.acuityLevelAllowed : ['Yellow', 'Green', 'Blue'],
        totalBeds: ward.totalBeds || 1,
        location: { floor: ward.location?.floor || 1, wing: ward.location?.wing || 'Main', area: ward.location?.area || '' },
        isActive: ward.isActive !== false
      })));
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setHospitalWardsLoading(false);
    }
  };

  const saveHospitalWards = async () => {
    setHospitalWardsSaving(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/hospital-wards`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ wards: hospitalWardDrafts })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to save hospital wards');
      const wards = data.wards || [];
      setHospitalWards(wards);
      setHospitalWardDrafts(wards.map((ward) => ({
        wardId: ward.wardId,
        wardName: ward.wardName,
        departmentName: ward.departmentName,
        careTier: ward.careTier,
        acuityLevelAllowed: ward.acuityLevelAllowed || [],
        totalBeds: ward.totalBeds,
        location: ward.location || { floor: 1, wing: 'Main', area: '' },
        isActive: ward.isActive !== false
      })));
      setMessages({ success: `Ward configuration saved and audited (event ${data.eventId}).`, error: '' });
      await fetchHospitalActivity();
      return true;
    } catch (error) {
      setMessages({ success: '', error: error.message });
      return false;
    } finally {
      setHospitalWardsSaving(false);
    }
  };

  const addHospitalWard = () => {
    const departmentName = hospitalStructureDraft[0]?.name || '';
    const nextNumber = hospitalWardDrafts.length + 1;
    const idPart = departmentName.toUpperCase().replace(/[^A-Z0-9]+/g, '-') || 'WARD';
    setHospitalWardDrafts((current) => [...current, {
      wardId: `${idPart}-WARD-${String(nextNumber).padStart(2, '0')}`,
      wardName: '',
      departmentName,
      careTier: 'GENERAL_INPATIENT',
      acuityLevelAllowed: ['Yellow', 'Green', 'Blue'],
      totalBeds: 1,
      location: { floor: 1, wing: 'Main', area: '' },
      isActive: true
    }]);
  };

  const updateWardDraft = (index, field, value) => {
    setHospitalWardDrafts((current) => current.map((ward, wardIndex) => wardIndex === index ? { ...ward, [field]: value } : ward));
  };

  const saveHospitalStructure = async () => {
    if (!hospitalStructureDraft.length || hospitalStructureDraft.some((department) => !department.specialties.length || !department.serviceTypes.length)) {
      setMessages({ success: '', error: 'Each active department needs at least one specialty and service type.' });
      return false;
    }
    setHospitalStructureSaving(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/operations/hospital-structure`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ departments: hospitalStructureDraft.map((department) => ({
          ...department,
          specialties: department.specialties.map((specialty) => typeof specialty === 'string' ? specialty : specialty?.name).filter(Boolean)
        })) })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || 'Unable to save hospital departments');
      const savedDepartments = (data.departments || []).map((department) => ({
        ...department,
        specialties: (department.specialties || []).map((specialty) => typeof specialty === 'string' ? specialty : specialty?.name).filter(Boolean)
      }));
      setHospitalStructure((current) => ({ ...current, departments: savedDepartments }));
      setHospitalStructureDraft(savedDepartments);
      setMessages({ success: `Hospital structure saved and audited (event ${data.eventId}).`, error: '' });
      await fetchHospitalActivity();
      return true;
    } catch (error) {
      setMessages({ success: '', error: error.message });
      return false;
    } finally {
      setHospitalStructureSaving(false);
    }
  };

  const toggleHospitalDepartment = (name) => {
    const normalizedName = name.trim().toLowerCase();
    const option = hospitalStructure.departmentOptions.find((department) => department.name.toLowerCase() === normalizedName);
    setHospitalStructureDraft((current) => current.some((department) => department.name.trim().toLowerCase() === normalizedName)
      ? current.filter((department) => department.name.trim().toLowerCase() !== normalizedName)
      : current.length >= 100
        ? current
        : [...current, {
          name: option?.name || name,
          isActive: true,
          serviceTypes: option?.serviceTypes || ['CLINICAL_SUPPORT'],
          specialties: []
        }]);
  };

  const addHospitalDepartment = () => {
    const name = hospitalStructureDepartmentName.trim().replace(/\s+/g, ' ');
    const normalizedName = name.toLowerCase();
    if (name.length < 2 || name.length > 60 || !/^[\p{L}\p{N}][\p{L}\p{N} &().'-]*$/u.test(name)) {
      setHospitalStructureDepartmentError('Use 2-60 letters, numbers, spaces, or basic punctuation for the department name.');
      return;
    }
    if (hospitalStructureDraft.some((department) => department.name.trim().toLowerCase() === normalizedName)) {
      setHospitalStructureDepartmentError('That department is already selected.');
      return;
    }
    if (hospitalStructureDraft.length >= 100) {
      setHospitalStructureDepartmentError('A hospital can configure up to 100 departments.');
      return;
    }
    const option = hospitalStructure.departmentOptions.find((department) => department.name.toLowerCase() === normalizedName);
    setHospitalStructureDraft((current) => [...current, {
      name: option?.name || name,
      isActive: true,
      serviceTypes: option?.serviceTypes || ['CLINICAL_SUPPORT'],
      specialties: []
    }]);
    setHospitalStructureDepartmentName('');
    setHospitalStructureDepartmentError('');
  };

  const updateDepartmentSelection = (departmentName, field, value) => {
    setHospitalStructureDraft((current) => current.map((department) => department.name !== departmentName
      ? department
      : { ...department, [field]: value }));
  };

  const fetchOverviewDiagnosisMix = async () => {
    if (!token) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/analytics/triage?range=week`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load analytics overview (${response.status})`);
      setOverviewDiagnoses(Array.isArray(data.diagnoses) ? data.diagnoses : []);
    } catch (error) {
      setOverviewDiagnoses([]);
    }
  };

  const fetchManagedHospitals = async () => {
    if (!isAdmin || !token) return;
    setManagedHospitalLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/hospital/managed`, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to load managed hospitals (${response.status})`);
      setManagedHospitals(data.hospitals || []);
      const selectedHospitalId = decoded?.hospitalId || data.hospitals?.[0]?._id;
      if (selectedHospitalId) {
        setHospitalCredentials((current) => ({
          ...current,
          name: decoded?.name || current.name || '',
          surname: decoded?.surname || current.surname || '',
          email: decoded?.email || current.email || '',
          phone: current.phone || ''
        }));
      }
    } catch (error) {
      if (String(error.message).includes('(404)')) {
        setManagedHospitals(decoded?.hospitalId ? [{ _id: decoded.hospitalId, name: decoded.hospitalName || decoded.name || decoded.hospitalId, hospitalID: decoded.hospitalId }] : []);
      } else setMessages({ success: '', error: error.message });
    } finally {
      setManagedHospitalLoading(false);
    }
  };

  const updateManagedHospitalCredentials = async () => {
    const selectedHospitalId = decoded?.hospitalId || managedHospitals[0]?._id;
    if (!selectedHospitalId) {
      setMessages({ success: '', error: 'Select a hospital before updating admin credentials.' });
      return;
    }

    if (!hospitalCredentials.email?.trim()) {
      setMessages({ success: '', error: 'An admin email is required.' });
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/hospital/managed/${selectedHospitalId}/credentials`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          name: hospitalCredentials.name.trim(),
          surname: hospitalCredentials.surname.trim(),
          email: hospitalCredentials.email.trim(),
          password: hospitalCredentials.password.trim() || undefined,
          phone: hospitalCredentials.phone.trim() || undefined
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to update credentials (${response.status})`);
      if (data.token) {
        localStorage.setItem('token', data.token);
      }
      setMessages({ success: 'Hospital credentials updated successfully.', error: '' });
      setHospitalCredentials((current) => ({ ...current, password: '' }));
      await fetchHospitalActivity();
      window.location.reload();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    }
  };

  const createManagedHospital = async () => {
    console.log('[HOSPITAL] Add hospital requested', { hospitalName: newHospitalName.trim() });
    if (!newHospitalName.trim()) return setMessages({ success: '', error: 'Enter a hospital name first.' });
    setManagedHospitalLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/hospital/managed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ hospitalName: newHospitalName.trim() })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to create hospital (${response.status})`);
      setNewHospitalName('');
      setMessages({ success: `${data.hospital?.name || 'Hospital'} created.`, error: '' });
      await fetchManagedHospitals();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setManagedHospitalLoading(false);
    }
  };

  const submitHospitalBootstrap = async () => {
    console.log('[HOSPITAL] Add hospital bootstrap requested', hospitalBootstrap);
    const { hospitalName, managerName, managerSurname, managerEmail, managerPassword } = hospitalBootstrap;
    if (!hospitalName || !managerName || !managerSurname || !managerEmail || !managerPassword) {
      setMessages({ success: '', error: 'Hospital, manager, and login details are required before creating the facility.' });
      return;
    }

    const enteredStaff = bootstrapStaffRows.filter((row) => row.firstName.trim() || row.lastName.trim() || row.email.trim());
    const invalidStaff = enteredStaff.find((row) => {
      const position = HOSPITAL_STAFF_ROLES.find((item) => item.code === row.positionCode);
      return !row.firstName.trim()
        || !row.lastName.trim()
        || !row.email.trim()
        || row.password.trim().length < 8
        || !position
        || !hospitalBootstrap.departments.includes(row.department)
        || !HOSPITAL_SPECIALTIES.some((item) => item.name === row.specialty);
    });
    if (invalidStaff) {
      setMessages({ success: '', error: 'Complete every staff row with a valid position, selected hospital department, configured specialty, and password of at least 8 characters.' });
      setSetupStep(1);
      return;
    }

    setManagedHospitalLoading(true);
    try {
      const departmentSpecialties = [...new Set(enteredStaff.map((row) => row.department))].map((department) => ({
        department,
        specialties: [...new Set(enteredStaff.filter((row) => row.department === department).map((row) => row.specialty))]
      }));
      const response = await fetch(`${BACKEND_URL}/api/hospital/managed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          hospitalName: hospitalName.trim(),
          adminName: managerName.trim(),
          adminSurname: managerSurname.trim(),
          adminEmail: managerEmail.trim(),
          adminPassword: managerPassword,
          department: hospitalBootstrap.department,
          departments: hospitalBootstrap.departments,
          departmentSpecialties
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to create managed hospital (${response.status})`);
      const hospitalToken = data.token || token;
      if (data.token) localStorage.setItem('token', data.token);

      const staff = enteredStaff.map((row, index) => {
        const position = HOSPITAL_STAFF_ROLES.find((item) => item.code === row.positionCode);
        return {
          ...row,
          role: position.systemRole,
          title: position.defaultTitle || '',
          licenseNumber: position.requiresCredentials ? (row.licenseNumber || `PENDING-${Date.now()}-${index + 1}`) : ''
        };
      });
      if (staff.length) {
        const staffResponse = await fetch(`${BACKEND_URL}/api/admin/hospital-operations/staff/bulk`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${hospitalToken}` },
          body: JSON.stringify({ staff })
        });
        const staffData = await staffResponse.json().catch(() => ({}));
        if (!staffResponse.ok) throw new Error(staffData.message || `Hospital created, but staff onboarding failed (${staffResponse.status})`);
      }

      const encounterResponse = await fetch(`${BACKEND_URL}/api/admin/hospital-operations/encounters/bulk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${hospitalToken}` },
        body: JSON.stringify({ count: Number(setupEncounterCount), historicalDays: Number(setupHistoricalDays), complaintType: 'FEVER', lifecycleStatus: 'AWAITING_PHYSICIAN' })
      });
      const encounterData = await encounterResponse.json().catch(() => ({}));
      if (!encounterResponse.ok) throw new Error(encounterData.message || `Hospital and staff were created, but encounter seeding failed (${encounterResponse.status})`);

      setMessages({ success: `${data.hospital?.name || hospitalName}, ${staff.length} staff credentials, and ${encounterData.count || setupEncounterCount} encounters were created under your administrator account.`, error: '' });
      setHospitalBootstrap({
        hospitalName: '',
        managerName: '',
        managerSurname: '',
        managerEmail: '',
        managerPassword: '',
        department: 'All',
        departments: ['ED', 'ICU', 'OR', 'OB', 'Medical-Surgical', 'Pediatrics', 'Psychiatry', 'General']
      });
      setBootstrapStaffRows([{ firstName: '', lastName: '', email: '', password: '', role: 'NURSE', positionCode: 'TRIAGE_NURSE', department: 'ED', specialty: '' }]);
      setSetupAutoFilled(false);
      await fetchManagedHospitals();
      setTimeout(() => window.location.reload(), 600);
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setManagedHospitalLoading(false);
    }
  };

  const updateBootstrapStaffRow = (index, field, value) => setBootstrapStaffRows((rows) => rows.map((row, rowIndex) => {
    if (rowIndex !== index) return row;
    if (field === 'positionCode') {
      const position = HOSPITAL_STAFF_ROLES.find((item) => item.code === value);
      return { ...row, positionCode: value, role: position?.systemRole || '' };
    }
    return { ...row, [field]: value };
  }));

  const autoFillHospitalSetup = () => {
    const sample = createHospitalSetupSample({ staffCount: setupStaffCount });
    setHospitalBootstrap(sample.hospital);
    setBootstrapStaffRows(sample.staff);
    setSetupAutoFilled(true);
    setSetupStep(0);
    setMessages({ success: 'Sample hospital, manager, and staff credentials filled. Review each step before creating.', error: '' });
  };

  const selectManagedHospital = async (hospitalId) => {
    if (!hospitalId || hospitalId === decoded?.hospitalId) return;
    setManagedHospitalLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/hospital/managed/${hospitalId}/select`, { method: 'POST', headers: { Authorization: `Bearer ${token}` } });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to select hospital (${response.status})`);
      localStorage.setItem('token', data.token);
      window.location.reload();
    } catch (error) {
      setMessages({ success: '', error: error.message });
      setManagedHospitalLoading(false);
    }
  };

  const runHospitalOperationsAction = async (path, body = undefined) => {
    setHospitalOperationsAction(path);
    try {
      const response = await fetch(`${BACKEND_URL}/api/admin/hospital-operations/${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: body ? JSON.stringify(body) : undefined
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Hospital operations action failed (${response.status})`);
      setMessages({ success: data.message || 'Hospital operations updated.', error: '' });
      await fetchHospitalOperationsStatus();
      await fetchHospitalActivity();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    } finally {
      setHospitalOperationsAction('');
    }
  };

  const onboardHospitalStaff = async () => {
    console.log('[STAFF] Add staff requested', hospitalOperationsStaffRows);
    try {
      const staff = hospitalOperationsStaffRows.filter((row) => row.firstName.trim() || row.lastName.trim() || row.email.trim());
      if (!staff.length) throw new Error('Add at least one staff member');
      const incomplete = staff.find((row) => {
        const position = hospitalStructure.staffRoles.find((item) => item.code === row.positionCode);
        return !row.firstName.trim()
          || !row.lastName.trim()
          || !row.email.trim()
          || !row.password.trim()
          || row.password.trim().length < 8
          || !position
          || !hospitalStructure.departments.some((department) => department.name === row.department && department.isActive !== false)
          || !getConfiguredDepartmentSpecialties(hospitalStructure.departments, row.department).includes(row.specialty)
          || (position.requiresCredentials && (!row.title.trim() || !row.licenseNumber.trim()));
      });
      if (incomplete) throw new Error('Complete each staff row using a configured department and specialty. Licensed positions require a title and license number; passwords must be at least 8 characters.');
      await runHospitalOperationsAction('staff/bulk', { staff });
      setHospitalOperationsStaffRows([createStaffDraft()]);
      await fetchStaff();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    }
  };

  const openStaffEditor = (staff) => {
    const positionCode = resolveStaffPositionCode(staff);
    const department = resolveConfiguredDepartmentName(
      hospitalStructure.departments,
      staff.assignment?.department || staff.department
    );
    const availableSpecialties = getConfiguredDepartmentSpecialties(hospitalStructure.departments, department);
    const specialty = availableSpecialties.find((item) => item.toLowerCase() === String(staff.clinicalSpecialty || staff.specialty || '').trim().toLowerCase()) || '';
    setStaffEditor({
      _id: staff._id,
      firstName: staff.firstName || '',
      lastName: staff.lastName || '',
      email: staff.email || '',
      password: '',
      positionCode,
      role: HOSPITAL_STAFF_ROLES.find((position) => position.code === positionCode)?.systemRole || '',
      department,
      specialty,
      title: staff.credentials?.title || HOSPITAL_STAFF_ROLES.find((position) => position.code === positionCode)?.defaultTitle || '',
      licenseNumber: staff.credentials?.licenseNumber || '',
      npiNumber: staff.credentials?.npiNumber || '',
      status: staff.status || 'ACTIVE'
    });
    setStaffEditorError('');
    setStaffEditorOpen(true);
  };

  const updateStaffEditor = (field, value) => {
    setStaffEditorError('');
    setStaffEditor((current) => {
      if (field === 'positionCode') {
        const position = HOSPITAL_STAFF_ROLES.find((item) => item.code === value);
        return {
          ...current,
          positionCode: value,
          role: position?.systemRole || '',
          title: position?.defaultTitle || '',
          licenseNumber: position?.requiresCredentials ? current.licenseNumber : ''
        };
      }
      if (field === 'department') {
        const specialties = getConfiguredDepartmentSpecialties(hospitalStructure.departments, value);
        return { ...current, department: value, specialty: specialties.includes(current.specialty) ? current.specialty : '' };
      }
      return { ...current, [field]: value };
    });
  };

  const saveStaffEditor = async () => {
    if (!staffEditor?._id) return;
    setStaffEditorSaving(true);
    setStaffEditorError('');
    try {
      const position = HOSPITAL_STAFF_ROLES.find((item) => item.code === staffEditor.positionCode);
      if (!position) throw new Error('Select a valid staff position.');
      if (!hospitalStructure.departments.some((department) => department.name === staffEditor.department && department.isActive !== false)) {
        throw new Error('Select an active department configured for this hospital.');
      }
      if (!getConfiguredDepartmentSpecialties(hospitalStructure.departments, staffEditor.department).includes(staffEditor.specialty)) {
        throw new Error('Select a specialty configured for the selected department.');
      }
      if (position.requiresCredentials && (!staffEditor.title || !staffEditor.licenseNumber.trim())) {
        throw new Error(`${position.label} requires a credential title and license number.`);
      }
      if (staffEditor.password?.trim() && staffEditor.password.trim().length < 8) {
        throw new Error('Password must be at least 8 characters long.');
      }

      const payload = {
        firstName: staffEditor.firstName.trim(),
        lastName: staffEditor.lastName.trim(),
        email: staffEditor.email.trim(),
        positionCode: position.code,
        clinicalSpecialty: staffEditor.specialty.trim(),
        assignment: { department: staffEditor.department },
        credentials: {
          ...(position.requiresCredentials ? { title: staffEditor.title, licenseNumber: staffEditor.licenseNumber.trim() } : {}),
          ...(staffEditor.npiNumber.trim() ? { npiNumber: staffEditor.npiNumber.trim() } : {})
        },
        status: staffEditor.status
      };

      if (staffEditor.password?.trim()) {
        payload.password = staffEditor.password.trim();
      }

      const response = await fetch(`${BACKEND_URL}/api/manager/staff/${staffEditor._id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(payload)
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to update staff credentials (${response.status})`);
      setMessages({ success: 'Staff credentials updated successfully.', error: '' });
      setStaffEditorError('');
      setStaffEditorOpen(false);
      setStaffEditor(null);
      await fetchStaff();
      await fetchHospitalActivity();
    } catch (error) {
      setStaffEditorError(error.message);
      setMessages({ success: '', error: error.message });
    } finally {
      setStaffEditorSaving(false);
    }
  };

  const deleteStaffMember = async (staffId, staffName = 'this staff member') => {
    console.log('[STAFF] Delete staff requested', { staffId, staffName });
    if (!staffId) return;
    const confirmed = window.confirm(`Remove ${staffName} from this hospital? This will deactivate their access.`);
    if (!confirmed) return;

    try {
      const response = await fetch(`${BACKEND_URL}/api/manager/staff/${staffId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `Unable to remove staff (${response.status})`);
      setMessages({ success: data.message || 'Staff member removed from the hospital.', error: '' });
      setStaffEditorOpen(false);
      setStaffEditor(null);
      await fetchStaff();
      await fetchHospitalActivity();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    }
  };

  const updateStaffRow = (index, field, value) => setHospitalOperationsStaffRows((rows) => rows.map((row, rowIndex) => {
    if (rowIndex !== index) return row;
    if (field === 'positionCode') {
      const position = hospitalStructure.staffRoles.find((item) => item.code === value);
      return {
        ...row,
        positionCode: value,
        role: position?.systemRole || '',
        title: position?.defaultTitle || '',
        licenseNumber: position?.requiresCredentials ? row.licenseNumber : ''
      };
    }
    if (field === 'department') {
      const specialties = getConfiguredDepartmentSpecialties(hospitalStructure.departments, value);
      return { ...row, department: value, specialty: specialties.includes(row.specialty) ? row.specialty : '' };
    }
    return { ...row, [field]: value };
  }));
  const updateShiftRow = (index, field, value) => setHospitalOperationsShiftRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, [field]: value } : row));
  const provisionHospitalShifts = async () => {
    const shifts = hospitalOperationsShiftRows.filter((row) => row.shiftName.trim());
    if (!shifts.length) return setMessages({ success: '', error: 'Add at least one shift' });
    await runHospitalOperationsAction('shifts/bulk', { shifts });
  };

  // ============ FETCH STAFF ============
  const fetchStaff = async () => {
    console.log('[STAFF SEARCH] Searching staff', { searchQuery, departmentFilter, roleFilter, isAdmin });
    if (!token) {
      setMessages({ success: '', error: 'Not authenticated' });
      return;
    }

    setSearchLoading(true);
    try {
      let url = `${BACKEND_URL}/api/manager/staff/list?`;
      if (departmentFilter !== 'all') url += `department=${departmentFilter}&`;
      if (roleFilter !== 'all') url += `role=${roleFilter}&`;
      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();

      if (data.success) {
        let staff = data.staff || [];
        setStaffList(staff);

        // Filter by search query
        if (searchQuery) {
          staff = staff.filter(s =>
            (s.firstName || s.name)?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            (s.lastName || s.surname)?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.email?.toLowerCase().includes(searchQuery.toLowerCase())
          );
        }

        setFilteredStaff(staff);
                return data.staff || [];
      } else {
        setMessages({ success: '', error: 'Failed to load staff' });
                return [];
      }
    } catch (err) {
      setMessages({ success: '', error: err.message });
              return [];
    } finally {
      setSearchLoading(false);
    }
  };

  useEffect(() => {
    fetchManagedHospitals();
  }, []);

  // If user is not admin/manager, render staff Clock In/Out view within the same route
  const [staffActiveTab, setStaffActiveTab] = useState('clock'); // 'clock' | 'myshifts'

  if (!isAdmin) {
    return (
      <SidebarLayout username={decoded?.name || 'Staff'} handleLogout={onLogout}>
        <div style={{ maxWidth: 1200 }}>
          {/* Header (styled like Settings) */}
          <div style={{ marginBottom: 32 }}>
            <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a' }}>
              <FiClock size={32} style={{ marginRight: 12, verticalAlign: 'middle', color: '#0369a1' }} />
              Time Schedule
            </h1>
            <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>Clock in/out, view assigned shifts, and manage your schedule</p>
          </div>

          {/* Minimal Tab Navigation to match Settings styling */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 24, borderBottom: '2px solid #e5e7eb', overflowX: 'auto' }}>
            {[
              { id: 'clock', label: 'Clock' },
              { id: 'myshifts', label: 'My Shifts' },
              { id: 'history', label: 'Attendance History' },
              { id: 'profile', label: 'Profile' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setStaffActiveTab(tab.id)}
                style={{
                  padding: '12px 16px',
                  backgroundColor: 'transparent',
                  border: 'none',
                  borderBottom: staffActiveTab === tab.id ? '3px solid #0f172a' : 'none',
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: staffActiveTab === tab.id ? 700 : 500,
                  color: staffActiveTab === tab.id ? '#0f172a' : '#6b7280',
                  transition: 'all 0.2s',
                  whiteSpace: 'nowrap'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Card container matching Settings content */}
          <div style={{
            backgroundColor: 'white',
            borderRadius: 12,
            padding: 24,
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
            border: '1px solid #e5e7eb',
          }}>
            <ClockInOutBoard
              token={token}
              user={decoded}
              activeTab={staffActiveTab}
              onLogout={onLogout}
            />
          </div>
        </div>
      </SidebarLayout>
    );
  }

  // ============ FETCH TODAY'S SCHEDULE ============
  const fetchTodaySchedule = async () => {
    if (!token) {
      setMessages({ success: '', error: 'Not authenticated' });
      return;
    }

    setScheduleLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/shifts/day?date=${targetDate}`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();

      if (data.success) {
        // Format shifts for 24-hour view
        setTodaySchedule(data.shifts || []);
        return data.shifts || [];
      } else {
        setMessages({ success: '', error: 'Failed to load schedule' });
        return [];
      }
    } catch (err) {
      setMessages({ success: '', error: err.message });
      return [];
    } finally {
      setScheduleLoading(false);
    }
  };

  // ============ CREATE SHIFT (Admin only) ============
  // ============ CREATE SHIFT & ASSIGN ROLES ============
  const handleCreateShift = async (e) => {
    e.preventDefault();

    console.log('[FRONTEND] CREATE SHIFT CALLED');
    console.log('  Shift Form:', shiftForm);
    console.log('  Selected Staff IDs:', selectedStaffForShift);
    console.log('  Role Assignments:', roleAssignments);

    if (!shiftForm.shiftName) {
      console.log('  ❌ VALIDATION FAILED: Missing required fields');
      setShiftMessages({ success: '', error: 'Please fill all required fields' });
      return;
    }

    const selectedStaff = selectedStaffForShift.map((staffId) => ({
      staffId,
      staff: staffList.find((person) => String(person._id) === String(staffId)),
    }));
    const invalidAssignment = selectedStaff.find(({ staff, staffId }) => (
      !staff || !resolveShiftAssignmentDuty(staff, selectedStaffRoles[staffId]?.role || '')
    ));
    if (invalidAssignment) {
      const staffName = invalidAssignment.staff
        ? `${invalidAssignment.staff.firstName || invalidAssignment.staff.name || ''} ${invalidAssignment.staff.lastName || invalidAssignment.staff.surname || ''}`.trim()
        : invalidAssignment.staffId;
      setShiftMessages({
        success: '',
        error: `Choose a shift duty compatible with ${staffName}, or refresh the staff roster.`,
      });
      return;
    }

    try {
      // Convert frontend role names to backend enum values
      const convertRoleToBackendType = (frontendRole) => {
        const roleMap = {
          'TRIAGE_NURSE': 'NURSE',
          'CLINICAL_NURSE': 'NURSE',
          'PHYSICIAN': 'PHYSICIAN',
          'SPECIALIST': 'SPECIALIST',
          'SUPPORT': 'SUPPORT',
          'MANAGER': 'MANAGER'
        };
        return roleMap[frontendRole] || 'NURSE';
      };

      // Map selected staff IDs to staff objects with full details
      const assignedStaffData = selectedStaff.map(({ staffId, staff }) => {
        const assignedRole = resolveShiftAssignmentDuty(staff, selectedStaffRoles[staffId]?.role || '');
        const backendType = staff.systemRole || staff.role || convertRoleToBackendType(assignedRole);
        return {
          staffID: staffId,
          type: backendType,
          assignmentRole: assignedRole,
          name: (staff?.firstName || staff?.name) + ' ' + (staff?.lastName || staff?.surname) || '',
          surname: staff?.lastName || staff?.surname || '',
          specialty: staff?.specialty || '',
          status: 'ASSIGNED'
        };
      });

      console.log('  ✓ Staff Data:', assignedStaffData);

      const shiftData = {
        shiftName: shiftForm.shiftName,
        shiftStart: shiftForm.shiftStart,
        shiftEnd: shiftForm.shiftEnd,
        date: shiftForm.shiftDate, // Include the date!
        maxStaffRequired: parseInt(shiftForm.maxStaffRequired),
        assignedStaff: assignedStaffData
      };

      console.log('  📤 Sending to backend:', JSON.stringify(shiftData, null, 2));
      console.log(`  URL: ${BACKEND_URL}/api/shifts/create`);
      console.log(`  Token: ${token?.substring(0, 20)}...`);

      const res = await fetch(`${BACKEND_URL}/api/shifts/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(shiftData)
      });

      console.log('  Status:', res.status);

      const data = await res.json();
      console.log('  📥 Response:', data);

      if (data.success) {
        console.log('  ✓ Shift created successfully!');
        console.log('  📍 Shift DB ID:', data.shift._id);

        setShiftMessages({ success: 'Shift created with explicit staff roles.', error: '' });

        // Reset forms
        setShiftForm({
          shiftName: '',
          shiftStart: '08:00',
          shiftEnd: '16:00',
          shiftDate: new Date().toISOString().split('T')[0],
          maxStaffRequired: 1,
          assignedStaff: []
        });
        setRoleAssignments({
          triageNurseId: '',
          clinicalNurseId: '',
          clinicalNurseContext: 'CLINICAL_NURSE_ER',
          physicianId: '',
          physicianContext: 'PHYSICIAN_ER'
        });
        setSelectedStaffRoles({});
        setSelectedStaffForShift([]);

        // Refresh schedule and return to the unified shift workspace
        await fetchTodaySchedule();
        setWorkspaceView('shifts');
        setActiveTab('shifts');
      } else {
        console.log('  ❌ Backend error:', data.message);
        setShiftMessages({ success: '', error: data.message || 'Failed to create shift' });
      }
    } catch (err) {
      console.log('  ❌ Exception:', err);
      setShiftMessages({ success: '', error: err.message });
    }
  };

  // ============ ASSIGN STAFF TO SHIFT ============
  const handleAssignStaffToShift = async (shiftId, staffId, role) => {
    const staff = staffList.find((person) => String(person._id) === String(staffId));
    if (!shiftId || !staffId || !staff || !role) {
      setMessages({ success: '', error: 'Select a shift, staff member, and explicit shift role before assigning coverage.' });
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/api/shifts/${shiftId}/assign-staff`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          staffID: staffId,
          assignmentRole: role
        })
      });

      const data = await res.json();
      if (data.success) {
        setMessages({ success: 'Staff assigned successfully', error: '' });
        setAssignmentStaffId('');
        setAssignmentRole('');
        await fetchTodaySchedule();
      } else {
        setMessages({ success: '', error: data.message || 'Failed to assign staff' });
      }
    } catch (err) {
      setMessages({ success: '', error: err.message });
    }
  };

  const handleRenameShift = async (shiftID) => {
    const nextName = editingShiftName.trim();
    if (!shiftID || !nextName) {
      setMessages({ success: '', error: 'Enter a shift name before saving.' });
      return;
    }
    try {
      const response = await fetch(`${BACKEND_URL}/api/shifts/${shiftID}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ shiftName: nextName })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) throw new Error(data.message || `Unable to rename shift (${response.status})`);
      setEditingShiftId('');
      setEditingShiftName('');
      setMessages({ success: 'Shift renamed successfully.', error: '' });
      await fetchTodaySchedule();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    }
  };

  const handleDeleteShift = async (shift) => {
    if (!shift?.shiftID) return;
    const reason = window.prompt(`Why are you cancelling ${shift.shiftName}?`);
    if (reason === null) return;
    const cancellationReason = reason.trim();
    if (!cancellationReason) {
      setMessages({ success: '', error: 'A cancellation reason is required.' });
      return;
    }
    const confirmed = window.confirm(`Cancel ${shift.shiftName}? The shift and assignments will remain in the audit history.`);
    if (!confirmed) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/shifts/${shift.shiftID}/cancel`, {
        method: 'POST',
        body: new Blob([JSON.stringify({ reason: cancellationReason })], { type: 'application/json' }),
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) throw new Error(data.message || `Unable to cancel shift (${response.status})`);
      if (String(selectedShiftId) === String(shift._id)) setSelectedShiftId('');
      setMessages({ success: 'Shift cancelled successfully.', error: '' });
      await fetchTodaySchedule();
    } catch (error) {
      setMessages({ success: '', error: error.message });
    }
  };

  // ============ AUTO-LOAD DATA ============
  useEffect(() => {
    if (activeTab === 'shifts') {
      fetchTodaySchedule();
      fetchStaff();
    } else if (activeTab === 'workload-policy') {
      fetchCapacityPolicy();
    } else if (activeTab === 'hospital-operations') {
      fetchHospitalOperationsStatus();
    }
    if (workspaceView === 'beds') fetchBedConfigurations();
    if (workspaceView === 'staff') {
      fetchStaff();
      fetchHospitalStructure();
    }
    if (workspaceView === 'hospital') {
      fetchHospitalActivity();
      fetchHospitalStructure();
    }
    if (workspaceView === 'shifts') fetchHospitalStructure();
    if (workspaceView === 'hospital-operations' || workspaceView === 'overview') {
      fetchHospitalActivity();
      fetchOverviewDiagnosisMix();
      if (workspaceView === 'overview') fetchHospitalOperationsStatus();
    }
    if (workspaceView === 'shifts') refreshStaffActivity();
  }, [activeTab, workspaceView, searchQuery, departmentFilter, roleFilter, targetDate]);

  useEffect(() => {
    if (!isAdmin || workspaceView !== 'shifts') return undefined;
    const interval = setInterval(refreshStaffActivity, 30000);
    return () => clearInterval(interval);
  }, [workspaceView, isAdmin, token]);

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const renderStaffActivity = () => (
    <section style={{ marginTop: 0, padding: 18, borderRadius: 14, border: '1px solid #dbeafe', background: '#f8fbff' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Live attendance</div>
          <h3 style={{ margin: '6px 0 0', color: '#0f172a' }}>Staff clock-in and clock-out activity</h3>
          <p style={{ margin: '5px 0 0', color: '#64748b', fontSize: 12 }}>All staff records for today, including active shifts, breaks, and completed clock-outs.</p>
        </div>
        <button type="button" onClick={refreshStaffActivity} disabled={staffActivityLoading} style={{ ...secondaryButtonStyle, minHeight: 36, padding: '8px 12px', fontSize: 12 }}><FiRefreshCw size={15} /> {staffActivityLoading ? 'Refreshing...' : 'Refresh attendance'}</button>
      </div>
      <div style={{ marginTop: 14, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {['CLOCKED_IN', 'ON_BREAK', 'CLOCKED_OUT'].map((status) => (
          <span key={status} style={{ padding: '6px 10px', borderRadius: 999, background: status === 'CLOCKED_IN' ? '#dcfce7' : status === 'ON_BREAK' ? '#fef3c7' : '#e2e8f0', color: status === 'CLOCKED_IN' ? '#166534' : status === 'ON_BREAK' ? '#92400e' : '#475569', fontSize: 11, fontWeight: 800 }}>
            {status.replace('_', ' ')}: {dayClockedInStaff.filter((record) => record.status === status).length}
          </span>
        ))}
      </div>
      {staffActivityLoading && !dayClockedInStaff.length ? <div style={{ marginTop: 14, color: '#64748b', fontSize: 13 }}>Loading staff attendance...</div> : dayClockedInStaff.length ? (
        <div style={{ marginTop: 14, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, background: '#fff' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}><th style={{ padding: 10 }}>Staff member</th><th style={{ padding: 10 }}>Status</th><th style={{ padding: 10 }}>Clocked in</th><th style={{ padding: 10 }}>Clocked out</th><th style={{ padding: 10 }}>Location</th></tr></thead>
            <tbody>{dayClockedInStaff.map((record) => {
              const staff = record.staff || {};
              return <tr key={record._id} style={{ borderBottom: '1px solid #eef2f7' }}><td style={{ padding: 10, fontWeight: 700 }}>{`${staff.firstName || ''} ${staff.lastName || ''}`.trim() || staff.email || 'Unknown staff'}</td><td style={{ padding: 10 }}><span style={{ color: record.status === 'CLOCKED_IN' ? '#166534' : record.status === 'ON_BREAK' ? '#92400e' : '#475569', fontWeight: 800 }}>{record.status?.replace('_', ' ') || 'UNKNOWN'}</span></td><td style={{ padding: 10 }}>{record.clockInTime ? new Date(record.clockInTime).toLocaleTimeString() : '--'}</td><td style={{ padding: 10 }}>{record.clockOutTime ? new Date(record.clockOutTime).toLocaleTimeString() : '--'}</td><td style={{ padding: 10 }}>{record.location || '--'}</td></tr>;
            })}</tbody>
          </table>
        </div>
      ) : <div style={{ marginTop: 14, padding: 14, borderRadius: 10, background: '#fff', color: '#64748b', fontSize: 13 }}>No staff clock activity has been recorded today.</div>}
    </section>
  );

  const renderWorkspaceView = () => {
    const overviewDiagnosisMix = (overviewDiagnoses || []).slice(0, 5);
    const diagnosisColors = ['#0369a1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
    const pharmacyStockTotal = pharmacyInventory.reduce((total, item) => total + Number(item.quantity || 0), 0);

    if (workspaceView === 'it-support' && isAdmin) {
      return <SupportTicketInbox />;
    }

    if (workspaceView === 'overview') {
      return (
        <div id="hospital-operations-overview" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: 16, minWidth: 0, width: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, minWidth: 0 }}>
            {[
              ['Configured hospital', managedHospitals.length || '0', 'Managed facilities'],
              ['Active staff', hospitalOperationsStatus?.staff?.total || '0', 'Rostered team'],
              ['Open shifts', hospitalOperationsStatus?.shifts?.today || '0', 'Scheduled today'],
              ['Patients', hospitalOperationsStatus?.encounters?.total || '0', 'Recorded encounters'],
            ].map(([label, value, note]) => (
              <div key={label} style={{ minWidth: 0, padding: 16, borderRadius: 12, border: '1px solid #dbeafe', background: 'linear-gradient(180deg, #f8fbff 0%, #ffffff 100%)', overflowWrap: 'anywhere' }}>
                <div style={{ color: '#64748b', fontSize: 11, lineHeight: 1.35, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{label}</div>
                <div style={{ marginTop: 10, fontSize: 26, lineHeight: 1.15, fontWeight: 800, color: '#0f172a', overflowWrap: 'anywhere' }}>{value}</div>
                <div style={{ marginTop: 6, fontSize: 12, lineHeight: 1.4, color: '#475569' }}>{note}</div>
              </div>
            ))}
            <div style={{ minWidth: 0, padding: 16, borderRadius: 12, border: '1px solid #dbeafe', background: 'linear-gradient(180deg, #f8fbff 0%, #ffffff 100%)', overflowWrap: 'anywhere' }}>
              <div style={{ color: '#64748b', fontSize: 11, lineHeight: 1.35, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Pharmacy inventory</div>
              <div style={{ marginTop: 10, fontSize: 26, lineHeight: 1.15, fontWeight: 800, color: '#0f172a', overflowWrap: 'anywhere' }}>{pharmacyInventoryLoading ? '...' : pharmacyStockTotal}</div>
              <div style={{ marginTop: 6, fontSize: 12, lineHeight: 1.4, color: '#475569' }}>Units on hand</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12, minWidth: 0 }}>
            <div style={{ minWidth: 0, padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: 'linear-gradient(180deg, #f8fbff 0%, #ffffff 100%)' }}>
              <div style={{ color: '#64748b', fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Workload baseline</div>
              <div style={{ marginTop: 10, fontSize: 26, fontWeight: 800, color: '#0f172a' }}>{capacityConfig?.policies?.defaultCapacityByRole ? 'Configured' : 'Unconfigured'}</div>
              <div style={{ marginTop: 6, fontSize: 12, color: '#475569' }}>{capacityConfig?.policies?.defaultCapacityByRole ? 'Clinical workload baseline configured' : 'Set baseline capacity before live utilization metrics are trusted'}</div>
            </div>

            <div style={{ minWidth: 0, padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: 'linear-gradient(180deg, #f8fbff 0%, #ffffff 100%)' }}>
              <div style={{ color: '#64748b', fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Most common diagnosis mix</div>
              <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 18, minHeight: 180, flexWrap: 'wrap', minWidth: 0 }}>
                {overviewDiagnosisMix.length ? (
                  <>
                    <OverviewDiagnosisPieChart data={overviewDiagnosisMix} />
                    <div style={{ display: 'grid', gap: 6, flex: 1 }}>
                      {overviewDiagnosisMix.slice(0, 4).map((item, index) => (
                        <div key={`${item.name}-${index}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, fontSize: 12, color: '#334155' }}>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 10, height: 10, borderRadius: 3, background: diagnosisColors[index % diagnosisColors.length], display: 'inline-block' }} />
                            {item.name}
                          </span>
                          <strong>{Number(item.percentage || 0).toFixed(1)}%</strong>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div style={{ fontSize: 12, color: '#64748b' }}>No live diagnosis mix recorded yet.</div>
                )}
              </div>
            </div>
          </div>

          {isCompanySuperAdmin && (
            <div style={{ padding: 16, borderRadius: 12, border: '1px solid #dbeafe', background: 'linear-gradient(180deg, #effdf5 0%, #f8fbff 100%)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#166534' }}>Nurshelpdesk oversight</div>
                  <h3 style={{ margin: '8px 0 0', color: '#0f172a' }}>Company admin monitoring layer</h3>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ padding: '6px 10px', borderRadius: 999, background: '#dcfce7', color: '#166534', fontSize: 11, fontWeight: 800 }}>Hospitals: {managedHospitals.length || 0}</span>
                  <span style={{ padding: '6px 10px', borderRadius: 999, background: '#dbeafe', color: '#1d4ed8', fontSize: 11, fontWeight: 800 }}>Admins: {managedHospitals.length || 0}</span>
                  <span style={{ padding: '6px 10px', borderRadius: 999, background: isMockSystemMode ? '#fef3c7' : '#dcfce7', color: isMockSystemMode ? '#92400e' : '#166534', fontSize: 11, fontWeight: 800 }}>{isMockSystemMode ? 'Mock sandbox' : 'Company live'} </span>
                </div>
              </div>
              <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
                {managedHospitals.length ? managedHospitals.slice(0, 4).map((hospital) => (
                  <div key={hospital._id || hospital.id} style={{ padding: 12, borderRadius: 10, border: '1px solid #dbeafe', background: '#fff' }}>
                    <div style={{ fontWeight: 800, color: '#0f172a' }}>{hospital.name}</div>
                    <div style={{ marginTop: 6, fontSize: 12, color: '#64748b' }}>{hospital.country || 'South Africa'} • {hospital.departments?.length || 0} departments</div>
                  </div>
                )) : (
                  <div style={{ padding: 12, borderRadius: 10, border: '1px solid #dbeafe', background: '#fff', color: '#64748b', fontSize: 12 }}>
                    No managed hospitals yet. Register a hospital to create the first admin-owned operational unit.
                  </div>
                )}
              </div>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
            <div style={{ padding: 16, borderRadius: 12, border: '1px solid #dbeafe', background: '#f8fbff' }}>
              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#475569' }}>Managed hospital</div>
              <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{selectedHospitalName}</div>
              <div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>{decoded?.hospitalId || 'No hospital selected yet'}</div>
            </div>
            <div style={{ padding: 16, borderRadius: 12, border: '1px solid #dbeafe', background: '#f8fbff' }}>
              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#475569' }}>Workload policy status</div>
              <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{capacityConfig?.policies?.defaultCapacityByRole ? 'Operational' : 'Needs setup'}</div>
              <div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>{capacityConfig?.policies?.defaultCapacityByRole ? 'Clinical workload baseline configured' : 'Set baseline capacity before live utilization metrics are trusted'}</div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button type="button" onClick={() => setHospitalSetupOpen(true)} style={policyButtonStyle}><FiPlus size={15} /> Create New Hospital</button>
            <button type="button" aria-label={hospitalDayRunning ? 'End Hospital Shift' : 'Start Hospital Shift'} title={hospitalDayRunning ? 'End Hospital Shift' : 'Start Hospital Shift'} onClick={hospitalDayRunning ? endHospitalDay : openHospitalDayWizard} style={hospitalDayRunning ? { ...policyButtonStyle, background: 'linear-gradient(135deg, #be123c 0%, #e11d48 100%)', boxShadow: '0 10px 24px rgba(225, 29, 72, 0.2)' } : policyButtonStyle}>
              <FiClock size={15} /> {hospitalDayRunning ? 'End Hospital Shift' : 'Start Hospital Shift'}
            </button>
            <button type="button" onClick={() => setWorkspaceView('staff')} style={secondaryButtonStyle}><FiUsers size={15} /> Onboard New Staff</button>
            <button type="button" onClick={() => setWorkspaceView('staff')} style={secondaryButtonStyle}><FiX size={15} /> Remove Existing Staff</button>
            <button type="button" onClick={() => setWorkspaceView('shifts')} style={secondaryButtonStyle}><FiPlus size={15} /> Schedule Shift</button>
            <button type="button" onClick={() => setWorkspaceView('beds')} style={secondaryButtonStyle}><FiSliders size={15} /> Manage Beds</button>
          </div>

          {hospitalDayRunning && activeShiftSession && (
            <section style={{ marginTop: 18, padding: 18, borderRadius: 14, border: '1px solid #86efac', background: '#ecfdf5', boxShadow: '0 10px 24px rgba(22, 101, 52, 0.08)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#166534' }}>Live operations</div>
                  <h3 style={{ margin: '6px 0 0', color: '#14532d' }}>Shift in session</h3>
                  <div style={{ marginTop: 5, color: '#166534', fontSize: 13 }}>{activeShiftSession.shiftName} • {activeShiftSession.department} • {activeShiftSession.shiftStart}-{activeShiftSession.shiftEnd}</div>
                </div>
                <span style={{ padding: '6px 10px', borderRadius: 999, background: '#bbf7d0', color: '#166534', fontSize: 11, fontWeight: 800 }}>ACTIVE</span>
              </div>
              <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
                <div style={{ padding: 10, borderRadius: 10, background: '#fff', color: '#166534' }}><strong>Staff assigned</strong><div style={{ marginTop: 4, fontSize: 18, fontWeight: 800 }}>{activeShiftSession.staffCount}</div></div>
                <div style={{ padding: 10, borderRadius: 10, background: '#fff', color: '#166534' }}><strong>Started</strong><div style={{ marginTop: 4, fontSize: 12 }}>{activeShiftSession.startedAt ? new Date(activeShiftSession.startedAt).toLocaleString() : 'Existing active session'}</div></div>
                <div style={{ padding: 10, borderRadius: 10, background: '#fff', color: '#166534' }}><strong>Approval</strong><div style={{ marginTop: 4, fontSize: 12 }}>Approved by admin</div></div>
              </div>
            </section>
          )}

          <section style={{ marginTop: 18, padding: 18, borderRadius: 14, border: '1px solid #dbeafe', background: '#f8fbff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Operational record</div>
                <h3 style={{ margin: '6px 0 0', color: '#0f172a' }}>Shift start and end log</h3>
              </div>
              <span style={{ fontSize: 12, color: '#64748b' }}>{shiftSessionLog.length} recorded events</span>
            </div>
            <div style={{ marginTop: 12, display: 'grid', gap: 8 }}>
              {shiftSessionLog.length ? shiftSessionLog.slice(0, 8).map((entry, index) => (
                <div key={`${entry.event}-${entry.startedAt || entry.endedAt}-${index}`} style={{ display: 'grid', gridTemplateColumns: 'auto minmax(0, 1fr) auto', gap: 10, alignItems: 'center', padding: '10px 12px', borderRadius: 10, background: '#fff', border: '1px solid #e2e8f0' }}>
                  <span style={{ padding: '4px 7px', borderRadius: 999, background: entry.event === 'STARTED' ? '#dcfce7' : '#ffe4e6', color: entry.event === 'STARTED' ? '#166534' : '#be123c', fontSize: 10, fontWeight: 800 }}>{entry.event}</span>
                  <div style={{ minWidth: 0, color: '#334155', fontSize: 12, overflowWrap: 'anywhere' }}>{entry.shiftName} • {entry.shiftStart}-{entry.shiftEnd} • {entry.staffCount || 0} staff</div>
                  <time style={{ color: '#64748b', fontSize: 11 }}>{new Date(entry.startedAt || entry.endedAt).toLocaleString()}</time>
                </div>
              )) : <div style={{ color: '#64748b', fontSize: 12 }}>No hospital shifts have been started or ended in this session.</div>}
            </div>
          </section>

          <section style={{ marginTop: 18, padding: 18, borderRadius: 14, border: '1px solid #dbeafe', background: '#f8fbff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Schedule overview</div>
                <h3 style={{ margin: '6px 0 0', color: '#0f172a' }}>Next upcoming shift</h3>
                <p style={{ margin: '5px 0 0', color: '#64748b', fontSize: 12 }}>Review scheduled shifts before approving one with Start Hospital Shift.</p>
              </div>
              <button type="button" onClick={() => setWorkspaceView('shifts')} style={{ ...secondaryButtonStyle, minHeight: 36, padding: '8px 12px', fontSize: 12 }}>Open shift calendar</button>
            </div>
            <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: 'repeat(7, minmax(90px, 1fr))', gap: 8, overflowX: 'auto', paddingBottom: 4 }}>
              {Array.from({ length: 7 }, (_, index) => {
                const date = new Date();
                date.setHours(0, 0, 0, 0);
                date.setDate(date.getDate() + index);
                const dateKey = date.toISOString().split('T')[0];
                const dayShifts = todaySchedule.filter((shift) => String(shift.date || '').slice(0, 10) === dateKey || (index === 0 && !shift.date));
                return (
                  <div key={dateKey} style={{ minHeight: 118, padding: 10, borderRadius: 10, border: index === 0 ? '2px solid #0f766e' : '1px solid #dbeafe', background: index === 0 ? '#ecfeff' : '#fff' }}>
                    <div style={{ fontSize: 11, fontWeight: 800, color: '#475569' }}>{date.toLocaleDateString(undefined, { weekday: 'short' })}</div>
                    <div style={{ marginTop: 3, fontSize: 16, fontWeight: 800, color: '#0f172a' }}>{date.getDate()}</div>
                    <div style={{ marginTop: 9, display: 'grid', gap: 5 }}>
                      {dayShifts.length ? dayShifts.slice(0, 3).map((shift) => <div key={shift._id} style={{ padding: '5px 6px', borderRadius: 6, background: '#dff7f2', color: '#115e59', fontSize: 10, fontWeight: 800, overflowWrap: 'anywhere' }}>{shift.shiftName}<br />{shift.shiftStart}-{shift.shiftEnd}</div>) : <span style={{ color: '#94a3b8', fontSize: 10 }}>No shifts</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

        </div>
      );
    }

    if (workspaceView === 'hospital') {
      return (
        <div id="hospital-setup-panel" style={{ display: 'grid', gap: 18 }}>
          <section style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#f8fbff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Hospital context</div>
                <h3 style={{ margin: '8px 0 0', color: '#0f172a' }}>Select, create, and manage hospital access</h3>
                <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Credentials, staff access, workload policy, and operational events stay scoped to {selectedHospitalName}.</p>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                <button type="button" onClick={() => setHospitalSetupOpen(true)} style={policyButtonStyle}><FiPlus size={15} /> Create new hospital</button>
                <button type="button" onClick={() => setWorkspaceView('staff')} style={secondaryButtonStyle}><FiUsers size={15} /> Manage staff</button>
                <button type="button" onClick={() => { setWorkloadPolicyOpen(true); fetchCapacityPolicy(); }} style={secondaryButtonStyle}><FiSliders size={15} /> Workload policy</button>
                <button type="button" onClick={() => { setHospitalActivityOpen(true); fetchHospitalActivity(); }} style={secondaryButtonStyle}><FiActivity size={15} /> Hospital activity</button>
                <button type="button" onClick={fetchHospitalActivity} disabled={hospitalActivityLoading} style={secondaryButtonStyle}><FiRefreshCw size={15} /> {hospitalActivityLoading ? 'Refreshing...' : 'Refresh activity'}</button>
              </div>
            </div>

            <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: 'minmax(220px, 300px) 1fr', gap: 12, alignItems: 'end' }}>
              <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                Active hospital
                <select value={decoded?.hospitalId || managedHospitals[0]?._id || ''} onChange={(event) => selectManagedHospital(event.target.value)} disabled={managedHospitalLoading} style={policyInputStyle}>
                  {managedHospitals.map((hospital) => (
                    <option key={String(hospital._id)} value={String(hospital._id)}>{hospital.name}</option>
                  ))}
                </select>
              </label>
              <div style={{ padding: '10px 12px', borderRadius: 10, background: '#fff', border: '1px solid #e2e8f0', color: '#475569', fontSize: 12 }}>
                One administrator can manage multiple hospitals. Switch the selected hospital here to keep the dashboard scoped to the correct operations view.
              </div>
            </div>

            <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10 }}>
              {Object.entries(hospitalActivity?.summary || { staff: 0, shifts: 0, encounters: 0, beds: 0, pharmaceuticals: 0 }).map(([label, value]) => (
                <div key={label} style={{ padding: 12, borderRadius: 10, background: '#fff', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 800 }}>{label}</div>
                  <strong style={{ display: 'block', marginTop: 7, fontSize: 22, color: '#0f172a' }}>{value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section style={{ padding: 16, borderRadius: 10, border: '1px solid #dbeafe', background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Hospital structure</div>
                <h3 style={{ margin: '5px 0 0', color: '#0f172a' }}>Departments, specialties and wards</h3>
                <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 12 }}>{hospitalStructure.departments.length} departments · {hospitalStructure.departments.reduce((count, department) => count + (department.specialties?.length || 0), 0)} specialties · {hospitalWards.length} wards</p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button type="button" onClick={fetchHospitalStructure} disabled={hospitalStructureLoading} style={secondaryButtonStyle}><FiRefreshCw size={14} /> Reload</button>
                <button type="button" onClick={() => { setHospitalStructureStep(0); setHospitalStructureDepartmentIndex(0); setHospitalStructureWardIndex(0); setHospitalStructureWizardOpen(true); }} disabled={hospitalStructureLoading} style={policyButtonStyle}><FiSliders size={14} /> Configure structure</button>
              </div>
            </div>
          </section>

          {false && <section style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Hospital structure</div>
                <h3 style={{ margin: '7px 0 0', color: '#0f172a' }}>Departments and specialties</h3>
                <p style={{ margin: '6px 0 0', maxWidth: 760, color: '#64748b', fontSize: 13 }}>Choose from the hospital’s fixed department catalog, then assign relevant clinical specialties. Departments are organizational services; wards are separate physical inpatient units with their own acuity, bed capabilities, and occupancy.</p>
              </div>
              <button type="button" onClick={fetchHospitalStructure} disabled={hospitalStructureLoading} style={secondaryButtonStyle}><FiRefreshCw size={15} /> {hospitalStructureLoading ? 'Refreshing...' : 'Reload saved structure'}</button>
            </div>

            {hospitalStructureLoading ? <p style={{ margin: '18px 0 0', color: '#64748b' }}>Loading saved hospital structure...</p> : <>
              <div style={{ marginTop: 16, padding: 14, borderRadius: 10, background: '#f8fbff', border: '1px solid #dbeafe' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                  <strong style={{ color: '#0f172a', fontSize: 13 }}>Available departments</strong>
                  <span style={{ color: '#64748b', fontSize: 12 }}>{hospitalStructureDraft.length} of 8 selected</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 10 }}>
                  {hospitalStructure.departmentOptions.map((option) => {
                    const selected = hospitalStructureDraft.some((department) => department.name === option.name);
                    return <button key={option.name} type="button" aria-pressed={selected} onClick={() => toggleHospitalDepartment(option.name)} style={{ padding: '8px 11px', borderRadius: 6, border: `1px solid ${selected ? '#0f766e' : '#cbd5e1'}`, background: selected ? '#ccfbf1' : '#fff', color: selected ? '#115e59' : '#475569', fontWeight: 700, fontSize: 12, cursor: 'pointer' }}>{selected ? '✓ ' : '+ '}{option.name}</button>;
                  })}
                </div>
              </div>

              {!hospitalStructureDraft.length && <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: '#fff7ed', color: '#9a3412', fontSize: 12 }}>Select at least one department before saving.</div>}
              <div style={{ display: 'grid', gap: 12, marginTop: 14 }}>
                {hospitalStructureDraft.map((department) => (
                  <article key={department.name} style={{ padding: 14, border: '1px solid #e2e8f0', borderRadius: 8, background: '#fff' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                      <div><strong style={{ color: '#0f172a' }}>{department.name}</strong><div style={{ marginTop: 3, color: '#64748b', fontSize: 11 }}>Department configuration</div></div>
                      <button type="button" onClick={() => toggleHospitalDepartment(department.name)} style={removeRowButtonStyle}>Remove department</button>
                    </div>
                    <div style={{ marginTop: 12 }}>
                      <strong style={{ display: 'block', marginBottom: 7, color: '#334155', fontSize: 12 }}>Service model</strong>
                      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                        {[['OUTPATIENT', 'Outpatient'], ['INPATIENT', 'Inpatient'], ['CLINICAL_SUPPORT', 'Clinical support']].map(([value, label]) => <label key={value} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: '#475569', fontSize: 12 }}><input type="checkbox" checked={department.serviceTypes.includes(value)} onChange={(event) => updateDepartmentSelection(department.name, 'serviceTypes', event.target.checked ? [...department.serviceTypes, value] : department.serviceTypes.filter((item) => item !== value))} />{label}</label>)}
                      </div>
                    </div>
                    <div style={{ marginTop: 12 }}>
                      <strong style={{ display: 'block', marginBottom: 8, color: '#334155', fontSize: 12 }}>Specialties</strong>
                      {['Medical', 'Surgical', 'Maternal & Child', 'Diagnostic & Support', 'Nursing'].map((category) => {
                        const options = hospitalStructure.specialties.filter((specialty) => specialty.category === category);
                        return <div key={category} style={{ marginTop: 10 }}>
                          <div style={{ marginBottom: 6, color: '#64748b', fontSize: 11, fontWeight: 800, textTransform: 'uppercase' }}>{category}</div>
                          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            {options.map((specialty) => {
                              const checked = department.specialties.includes(specialty.name);
                              return <label key={specialty.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '6px 8px', border: '1px solid #e2e8f0', borderRadius: 6, background: checked ? '#f0fdfa' : '#fff', color: '#334155', fontSize: 11 }}><input type="checkbox" checked={checked} onChange={(event) => updateDepartmentSelection(department.name, 'specialties', event.target.checked ? [...department.specialties, specialty.name] : department.specialties.filter((item) => item !== specialty.name))} />{specialty.name}</label>;
                            })}
                          </div>
                        </div>;
                      })}
                    </div>
                  </article>
                ))}
              </div>
              <div style={{ marginTop: 14, padding: 12, borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0', color: '#475569', fontSize: 12 }}>
                <strong>Wards stay separate from departments.</strong> Configure each ward as a physical care unit under its responsible department, with a care tier (critical care, step-down, general inpatient, or isolation), accepted SATS acuity colors, bed inventory, and live occupancy. Beds remain individually tracked records.
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 14 }}><button type="button" onClick={saveHospitalStructure} disabled={hospitalStructureSaving || hospitalStructureLoading || !hospitalStructureDraft.length} style={policyButtonStyle}><FiSave size={15} /> {hospitalStructureSaving ? 'Saving...' : 'Save departments and specialties'}</button></div>
            </>}
          </section>}

          {false && <section style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Physical care locations</div>
                <h3 style={{ margin: '7px 0 0', color: '#0f172a' }}>Wards and units</h3>
                <p style={{ margin: '6px 0 0', maxWidth: 760, color: '#64748b', fontSize: 13 }}>A ward is a physical inpatient unit with a responsible department, care tier, acuity acceptance, and capacity. Live beds remain separately tracked so encounter assignments and status history stay authoritative.</p>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button type="button" onClick={fetchHospitalWards} disabled={hospitalWardsLoading} style={secondaryButtonStyle}><FiRefreshCw size={15} /> {hospitalWardsLoading ? 'Refreshing...' : 'Reload wards'}</button>
                <button type="button" onClick={addHospitalWard} disabled={!hospitalStructureDraft.length || hospitalWardDrafts.length >= 300} style={policyButtonStyle}><FiPlus size={15} /> Add ward</button>
              </div>
            </div>
            {hospitalWardsLoading ? <p style={{ marginTop: 18, color: '#64748b' }}>Loading ward configuration...</p> : !hospitalWardDrafts.length ? <div style={{ marginTop: 14, padding: 16, borderRadius: 8, background: '#f8fafc', color: '#64748b', fontSize: 13 }}>No wards configured. Save departments first, then add each physical ward or unit.</div> : <div style={{ display: 'grid', gap: 12, marginTop: 14 }}>
              {hospitalWardDrafts.map((ward, index) => (
                <article key={`${ward.wardId}-${index}`} style={{ padding: 14, border: '1px solid #e2e8f0', borderRadius: 8, background: ward.isActive ? '#f8fbff' : '#f1f5f9', opacity: ward.isActive ? 1 : 0.72 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                    <strong style={{ color: '#0f172a' }}>{ward.wardName || `New ward ${index + 1}`}{!ward.isActive && ' (inactive)'}</strong>
                    <button type="button" onClick={() => updateWardDraft(index, 'isActive', !ward.isActive)} style={removeRowButtonStyle}>{ward.isActive ? 'Deactivate ward' : 'Reactivate ward'}</button>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10, marginTop: 12 }}>
                    <label style={adminFieldStyle}>Ward ID<input value={ward.wardId} onChange={(event) => updateWardDraft(index, 'wardId', event.target.value)} style={policyInputStyle} /></label>
                    <label style={adminFieldStyle}>Ward name<input value={ward.wardName} onChange={(event) => updateWardDraft(index, 'wardName', event.target.value)} placeholder="e.g. Medical Ward A" style={policyInputStyle} /></label>
                    <label style={adminFieldStyle}>Responsible department<select value={ward.departmentName} onChange={(event) => updateWardDraft(index, 'departmentName', event.target.value)} style={policyInputStyle}>{hospitalStructureDraft.map((department) => <option key={department.name} value={department.name}>{department.name}</option>)}</select></label>
                    <label style={adminFieldStyle}>Care tier<select value={ward.careTier} onChange={(event) => updateWardDraft(index, 'careTier', event.target.value)} style={policyInputStyle}><option value="CRITICAL_CARE">Critical care</option><option value="STEP_DOWN">Step-down</option><option value="GENERAL_INPATIENT">General inpatient</option><option value="ISOLATION">Isolation</option></select></label>
                    <label style={adminFieldStyle}>Total bed capacity<input type="number" min="1" max="2000" value={ward.totalBeds} onChange={(event) => updateWardDraft(index, 'totalBeds', Number(event.target.value))} style={policyInputStyle} /></label>
                    <label style={adminFieldStyle}>Floor<input type="number" min="0" value={ward.location.floor} onChange={(event) => updateWardDraft(index, 'location', { ...ward.location, floor: Number(event.target.value) })} style={policyInputStyle} /></label>
                    <label style={adminFieldStyle}>Wing<input value={ward.location.wing} onChange={(event) => updateWardDraft(index, 'location', { ...ward.location, wing: event.target.value })} style={policyInputStyle} /></label>
                    <label style={adminFieldStyle}>Area<input value={ward.location.area} onChange={(event) => updateWardDraft(index, 'location', { ...ward.location, area: event.target.value })} style={policyInputStyle} /></label>
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <strong style={{ display: 'block', marginBottom: 7, color: '#334155', fontSize: 12 }}>Accepted SATS acuity colors</strong>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>{[['Red', '#b91c1c'], ['Orange', '#c2410c'], ['Yellow', '#a16207'], ['Green', '#15803d'], ['Blue', '#1d4ed8']].map(([color, textColor]) => <label key={color} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: textColor, fontSize: 12, fontWeight: 700 }}><input type="checkbox" checked={ward.acuityLevelAllowed.includes(color)} onChange={(event) => updateWardDraft(index, 'acuityLevelAllowed', event.target.checked ? [...ward.acuityLevelAllowed, color] : ward.acuityLevelAllowed.filter((item) => item !== color))} />{color}</label>)}</div>
                  </div>
                  <div style={{ marginTop: 9, color: '#64748b', fontSize: 11 }}>Live bed status and encounter assignment are managed in Beds, not copied into this ward configuration.</div>
                </article>
              ))}
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}><button type="button" onClick={saveHospitalWards} disabled={hospitalWardsSaving || hospitalWardsLoading} style={policyButtonStyle}><FiSave size={15} /> {hospitalWardsSaving ? 'Saving...' : 'Save wards'}</button></div>
            </div>}
          </section>}

          {hospitalStructureWizardOpen && (
            <div role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setHospitalStructureWizardOpen(false); }} style={{ position: 'fixed', inset: 0, zIndex: 120, display: 'grid', placeItems: 'center', padding: 16, background: 'rgba(15, 23, 42, 0.58)' }}>
              <section role="dialog" aria-modal="true" aria-labelledby="hospital-structure-title" style={{ width: 'min(760px, 100%)', maxHeight: 'min(760px, calc(100vh - 32px))', display: 'flex', flexDirection: 'column', overflow: 'hidden', borderRadius: 10, border: '1px solid #cbd5e1', background: '#fff', boxShadow: '0 28px 80px rgba(15,23,42,.28)' }}>
                <header style={{ padding: '18px 20px 14px', borderBottom: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
                    <div><div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.1em', color: '#0f766e', textTransform: 'uppercase' }}>Hospital structure · {selectedHospitalName}</div><h2 id="hospital-structure-title" style={{ margin: '5px 0 0', color: '#0f172a', fontSize: 21 }}>{['Departments', 'Specialties', 'Wards and units', 'Review and save'][hospitalStructureStep]}</h2></div>
                    <button type="button" aria-label="Close structure setup" onClick={() => setHospitalStructureWizardOpen(false)} style={{ ...removeRowButtonStyle, width: 36, height: 36, padding: 0, justifyContent: 'center' }}><FiX size={17} /></button>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 5, marginTop: 14 }}>
                    {['Departments', 'Specialties', 'Wards', 'Review'].map((label, index) => <div key={label} style={{ height: 4, borderRadius: 4, background: index <= hospitalStructureStep ? '#0f766e' : '#e2e8f0' }} title={label} />)}
                  </div>
                </header>

                <div style={{ padding: 20, overflowY: 'auto', minHeight: 300 }}>
                  {hospitalStructureStep === 0 && <div>
                    <p style={{ margin: '0 0 14px', color: '#64748b', fontSize: 13 }}>Select the hospital's departments or add a custom department. Wards are configured separately in a later step.</p>
                    <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                      <input
                        aria-label="New department name"
                        value={hospitalStructureDepartmentName}
                        onChange={(event) => { setHospitalStructureDepartmentName(event.target.value); setHospitalStructureDepartmentError(''); }}
                        onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); addHospitalDepartment(); } }}
                        placeholder="e.g. Pharmacy, Radiology, Pathology"
                        maxLength={60}
                        style={{ ...policyInputStyle, flex: 1 }}
                      />
                      <button type="button" onClick={addHospitalDepartment} style={secondaryButtonStyle}><FiPlus size={14} /> Add department</button>
                    </div>
                    {hospitalStructureDepartmentError && <p role="alert" style={{ margin: '-4px 0 10px', color: '#b91c1c', fontSize: 12 }}>{hospitalStructureDepartmentError}</p>}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
                      {hospitalStructure.departmentOptions.map((option) => {
                        const selected = hospitalStructureDraft.some((department) => department.name.trim().toLowerCase() === option.name.toLowerCase());
                        return <button key={option.name} type="button" aria-pressed={selected} onClick={() => toggleHospitalDepartment(option.name)} style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '12px 10px', borderRadius: 6, border: `1px solid ${selected ? '#0f766e' : '#dbe2ea'}`, background: selected ? '#f0fdfa' : '#fff', color: '#334155', textAlign: 'left', fontWeight: 700, cursor: 'pointer' }}><span style={{ width: 18, height: 18, display: 'grid', placeItems: 'center', borderRadius: 4, border: `1px solid ${selected ? '#0f766e' : '#94a3b8'}`, background: selected ? '#0f766e' : '#fff', color: '#fff', fontSize: 12 }}>{selected ? '✓' : ''}</span>{option.name}</button>;
                      })}
                      {hospitalStructureDraft.filter((department) => !hospitalStructure.departmentOptions.some((option) => option.name.toLowerCase() === department.name.trim().toLowerCase())).map((department) => <button key={department.name} type="button" aria-pressed="true" onClick={() => toggleHospitalDepartment(department.name)} style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '12px 10px', borderRadius: 6, border: '1px solid #0f766e', background: '#f0fdfa', color: '#334155', textAlign: 'left', fontWeight: 700, cursor: 'pointer' }}><span style={{ width: 18, height: 18, display: 'grid', placeItems: 'center', borderRadius: 4, border: '1px solid #0f766e', background: '#0f766e', color: '#fff', fontSize: 12 }}>✓</span>{department.name} <span style={{ color: '#64748b', fontSize: 10, fontWeight: 500 }}>Custom</span></button>)}
                    </div>
                    <div style={{ marginTop: 12, color: '#64748b', fontSize: 12 }}>{hospitalStructureDraft.length} selected</div>
                  </div>}

                  {hospitalStructureStep === 1 && <div>
                    {!hospitalStructureDraft.length ? <p style={{ color: '#92400e' }}>Select at least one department first.</p> : (() => {
                      const department = hospitalStructureDraft[hospitalStructureDepartmentIndex];
                      const allSpecialtyNames = hospitalStructure.specialties.map((specialty) => specialty.name);
                      const allSpecialtiesSelected = allSpecialtyNames.length > 0 && allSpecialtyNames.every((name) => department.specialties.includes(name));
                      return <>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                          <button type="button" aria-label="Previous department" title="Previous department" disabled={hospitalStructureDepartmentIndex === 0} onClick={() => setHospitalStructureDepartmentIndex((index) => Math.max(0, index - 1))} style={{ ...secondaryButtonStyle, width: 40, minWidth: 40, padding: 0, justifyContent: 'center' }}><FiChevronLeft size={18} /></button>
                          <div style={{ textAlign: 'center' }}><strong style={{ color: '#0f172a', fontSize: 17 }}>{department.name}</strong><div style={{ color: '#64748b', fontSize: 11 }}>{hospitalStructureDepartmentIndex + 1} of {hospitalStructureDraft.length}</div></div>
                          <button type="button" aria-label="Next department" title="Next department" disabled={hospitalStructureDepartmentIndex >= hospitalStructureDraft.length - 1} onClick={() => setHospitalStructureDepartmentIndex((index) => Math.min(hospitalStructureDraft.length - 1, index + 1))} style={{ ...secondaryButtonStyle, width: 40, minWidth: 40, padding: 0, justifyContent: 'center' }}><FiChevronRight size={18} /></button>
                        </div>
                        <div style={{ padding: 12, borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                          <strong style={{ color: '#334155', fontSize: 12 }}>Service model</strong>
                          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 8 }}>{[['OUTPATIENT', 'Outpatient'], ['INPATIENT', 'Inpatient'], ['CLINICAL_SUPPORT', 'Clinical support']].map(([value, label]) => <label key={value} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: '#475569', fontSize: 12 }}><input type="checkbox" checked={department.serviceTypes.includes(value)} onChange={(event) => updateDepartmentSelection(department.name, 'serviceTypes', event.target.checked ? [...department.serviceTypes, value] : department.serviceTypes.filter((item) => item !== value))} />{label}</label>)}</div>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, marginTop: 14 }}>
                          <strong style={{ color: '#334155', fontSize: 13 }}>Specialties</strong>
                          <button type="button" onClick={() => updateDepartmentSelection(department.name, 'specialties', allSpecialtiesSelected ? [] : allSpecialtyNames)} style={{ ...secondaryButtonStyle, minHeight: 34, padding: '7px 10px', fontSize: 12 }}>{allSpecialtiesSelected ? 'Clear all' : 'Select all specialties'}</button>
                        </div>
                        <div style={{ display: 'grid', gap: 12, marginTop: 14, maxHeight: 360, overflowY: 'auto' }}>
                          {['Medical', 'Surgical', 'Maternal & Child', 'Diagnostic & Support', 'Nursing'].map((category) => {
                            const specialties = hospitalStructure.specialties.filter((item) => item.category === category);
                            return <fieldset key={category} style={{ margin: 0, padding: 10, border: '1px solid #e2e8f0', borderRadius: 6 }}><legend style={{ padding: '0 5px', color: '#475569', fontSize: 11, fontWeight: 800 }}>{category}</legend><div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 4 }}>{specialties.map((specialty) => <label key={specialty.name} style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '5px 4px', color: '#334155', fontSize: 12 }}><input type="checkbox" checked={department.specialties.includes(specialty.name)} onChange={(event) => updateDepartmentSelection(department.name, 'specialties', event.target.checked ? [...department.specialties, specialty.name] : department.specialties.filter((item) => item !== specialty.name))} />{specialty.name}</label>)}</div></fieldset>;
                          })}
                        </div>
                        <p style={{ margin: '10px 0 0', color: '#64748b', fontSize: 11 }}>{department.specialties.length} specialties selected</p>
                      </>;
                    })()}
                  </div>}

                  {hospitalStructureStep === 2 && <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                      <div><strong style={{ color: '#0f172a' }}>Wards and units</strong><div style={{ marginTop: 3, color: '#64748b', fontSize: 12 }}>Separate physical inpatient areas linked to a department</div></div>
                      <button type="button" onClick={() => { addHospitalWard(); setHospitalStructureWardIndex(hospitalWardDrafts.length); }} disabled={!hospitalStructureDraft.length || hospitalWardDrafts.length >= 300} style={secondaryButtonStyle}><FiPlus size={14} /> Add ward</button>
                    </div>
                    {!hospitalWardDrafts.length ? <div style={{ padding: 16, border: '1px dashed #cbd5e1', borderRadius: 6, color: '#64748b', fontSize: 13 }}>No wards configured. Add a ward if this hospital has inpatient capacity.</div> : (() => {
                      const ward = hospitalWardDrafts[hospitalStructureWardIndex];
                      return <>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                          <button type="button" aria-label="Previous ward" title="Previous ward" disabled={hospitalStructureWardIndex === 0} onClick={() => setHospitalStructureWardIndex((index) => Math.max(0, index - 1))} style={{ ...secondaryButtonStyle, width: 40, minWidth: 40, padding: 0, justifyContent: 'center' }}><FiChevronLeft size={18} /></button>
                          <div style={{ textAlign: 'center' }}><strong style={{ color: '#0f172a' }}>{ward.wardName || 'New ward'}</strong><div style={{ color: '#64748b', fontSize: 11 }}>{hospitalStructureWardIndex + 1} of {hospitalWardDrafts.length}</div></div>
                          <button type="button" aria-label="Next ward" title="Next ward" disabled={hospitalStructureWardIndex >= hospitalWardDrafts.length - 1} onClick={() => setHospitalStructureWardIndex((index) => Math.min(hospitalWardDrafts.length - 1, index + 1))} style={{ ...secondaryButtonStyle, width: 40, minWidth: 40, padding: 0, justifyContent: 'center' }}><FiChevronRight size={18} /></button>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 10 }}>
                          <label style={adminFieldStyle}>Ward ID<input value={ward.wardId} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'wardId', event.target.value)} style={policyInputStyle} /></label>
                          <label style={adminFieldStyle}>Ward name<input value={ward.wardName} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'wardName', event.target.value)} placeholder="e.g. Medical Ward A" style={policyInputStyle} /></label>
                          <label style={adminFieldStyle}>Department<select value={ward.departmentName} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'departmentName', event.target.value)} style={policyInputStyle}>{hospitalStructureDraft.map((department) => <option key={department.name} value={department.name}>{department.name}</option>)}</select></label>
                          <label style={adminFieldStyle}>Care tier<select value={ward.careTier} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'careTier', event.target.value)} style={policyInputStyle}><option value="CRITICAL_CARE">Critical care</option><option value="STEP_DOWN">Step-down</option><option value="GENERAL_INPATIENT">General inpatient</option><option value="ISOLATION">Isolation</option></select></label>
                          <label style={adminFieldStyle}>Bed capacity<input type="number" min="1" max="2000" value={ward.totalBeds} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'totalBeds', Number(event.target.value))} style={policyInputStyle} /></label>
                          <label style={adminFieldStyle}>Floor<input type="number" min="0" value={ward.location.floor} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'location', { ...ward.location, floor: Number(event.target.value) })} style={policyInputStyle} /></label>
                          <label style={adminFieldStyle}>Wing<input value={ward.location.wing} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'location', { ...ward.location, wing: event.target.value })} style={policyInputStyle} /></label>
                        </div>
                        <div style={{ marginTop: 14 }}><strong style={{ display: 'block', marginBottom: 7, color: '#334155', fontSize: 12 }}>Accepted SATS acuity</strong><div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>{['Red', 'Orange', 'Yellow', 'Green', 'Blue'].map((color) => <label key={color} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: '#475569', fontSize: 12 }}><input type="checkbox" checked={ward.acuityLevelAllowed.includes(color)} onChange={(event) => updateWardDraft(hospitalStructureWardIndex, 'acuityLevelAllowed', event.target.checked ? [...ward.acuityLevelAllowed, color] : ward.acuityLevelAllowed.filter((item) => item !== color))} />{color}</label>)}</div></div>
                        <button type="button" onClick={() => setHospitalWardDrafts((wards) => wards.map((item, index) => index === hospitalStructureWardIndex ? { ...item, isActive: !item.isActive } : item))} style={{ ...removeRowButtonStyle, marginTop: 12 }}>{ward.isActive ? 'Deactivate this ward' : 'Reactivate this ward'}</button>
                      </>;
                    })()}
                  </div>}

                  {hospitalStructureStep === 3 && <div style={{ display: 'grid', gap: 14 }}>
                    <div style={{ padding: 14, borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0' }}><strong style={{ color: '#0f172a' }}>{hospitalStructureDraft.length} departments</strong><div style={{ marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>{hospitalStructureDraft.map((department) => <span key={department.name} style={{ padding: '4px 7px', borderRadius: 4, background: '#e2e8f0', color: '#334155', fontSize: 11 }}>{department.name}</span>)}</div></div>
                    <div style={{ padding: 14, borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0' }}><strong style={{ color: '#0f172a' }}>{hospitalStructureDraft.reduce((count, department) => count + department.specialties.length, 0)} specialty assignments</strong><div style={{ marginTop: 7, display: 'grid', gap: 6 }}>{hospitalStructureDraft.map((department) => <div key={department.name} style={{ fontSize: 12, color: '#475569' }}><strong>{department.name}:</strong> {department.specialties.map((specialty) => typeof specialty === 'string' ? specialty : specialty?.name).filter(Boolean).join(', ') || 'None selected'}</div>)}</div></div>
                    <div style={{ padding: 14, borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0' }}><strong style={{ color: '#0f172a' }}>{hospitalWardDrafts.length} wards / units</strong><div style={{ marginTop: 7, display: 'grid', gap: 6 }}>{hospitalWardDrafts.map((ward) => <div key={ward.wardId} style={{ fontSize: 12, color: '#475569' }}><strong>{ward.wardName || ward.wardId}:</strong> {ward.departmentName} · {ward.careTier.replaceAll('_', ' ')} · {ward.totalBeds} beds · {ward.acuityLevelAllowed.join(', ')}</div>)}</div></div>
                    <p style={{ margin: 0, color: '#64748b', fontSize: 11 }}>Saving writes hospital departments and ward records separately. It does not create or alter individual live bed assignments.</p>
                  </div>}
                </div>

                <footer style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, padding: '12px 20px', borderTop: '1px solid #e2e8f0', background: '#f8fafc' }}>
                  <button type="button" onClick={() => hospitalStructureStep === 0 ? setHospitalStructureWizardOpen(false) : setHospitalStructureStep((step) => step - 1)} style={secondaryButtonStyle}>{hospitalStructureStep === 0 ? 'Cancel' : <><FiChevronLeft size={15} /> Previous</>}</button>
                  <span style={{ color: '#64748b', fontSize: 11 }}>Step {hospitalStructureStep + 1} of 4</span>
                  {hospitalStructureStep < 3 ? <button type="button" onClick={() => {
                    if (hospitalStructureStep === 0 && !hospitalStructureDraft.length) {
                      setMessages({ success: '', error: 'Select at least one department.' });
                      return;
                    }
                    if (hospitalStructureStep === 1) {
                      const missing = hospitalStructureDraft.find((department) => !department.specialties.length || !department.serviceTypes.length);
                      if (missing) {
                        setMessages({ success: '', error: `Select a service model and at least one specialty for ${missing.name}.` });
                        setHospitalStructureDepartmentIndex(hospitalStructureDraft.indexOf(missing));
                        return;
                      }
                    }
                    setMessages({ success: '', error: '' });
                    setHospitalStructureStep((step) => step + 1);
                  }} style={policyButtonStyle}>Next <FiChevronRight size={15} /></button> : <button type="button" disabled={hospitalStructureSaving || hospitalWardsSaving} onClick={async () => {
                    const departmentsSaved = await saveHospitalStructure();
                    if (!departmentsSaved) return;
                    if (hospitalWardDrafts.length) {
                      const wardsSaved = await saveHospitalWards();
                      if (!wardsSaved) return;
                    }
                    setHospitalStructureWizardOpen(false);
                  }} style={policyButtonStyle}><FiSave size={15} /> {hospitalStructureSaving || hospitalWardsSaving ? 'Saving...' : 'Save structure'}</button>}
                </footer>
              </section>
            </div>
          )}

          {hospitalCredentialsOpen && (
            <div style={{ position: 'fixed', inset: 0, zIndex: 70, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
              <div role="dialog" aria-modal="true" style={{ width: 'min(680px, 100%)', background: '#fff', borderRadius: 16, boxShadow: '0 26px 64px rgba(15, 23, 42, 0.24)', border: '1px solid #dbeafe', padding: 22 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
                  <div><div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Admin credentials</div><h3 style={{ margin: '8px 0 0', color: '#0f172a' }}>Update login details for {selectedHospitalName}</h3></div>
                  <button type="button" onClick={() => setHospitalCredentialsOpen(false)} style={removeRowButtonStyle}><FiX size={14} /> Close</button>
                </div>
                <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                  <label style={adminFieldStyle}>First name<input value={hospitalCredentials.name} onChange={(event) => setHospitalCredentials((current) => ({ ...current, name: event.target.value }))} style={policyInputStyle} /></label>
                  <label style={adminFieldStyle}>Surname<input value={hospitalCredentials.surname} onChange={(event) => setHospitalCredentials((current) => ({ ...current, surname: event.target.value }))} style={policyInputStyle} /></label>
                  <label style={adminFieldStyle}>Email<input type="email" value={hospitalCredentials.email} onChange={(event) => setHospitalCredentials((current) => ({ ...current, email: event.target.value }))} style={policyInputStyle} /></label>
                  <label style={adminFieldStyle}>Phone<input value={hospitalCredentials.phone} onChange={(event) => setHospitalCredentials((current) => ({ ...current, phone: event.target.value }))} style={policyInputStyle} /></label>
                  <PasswordField
                    label="New password"
                    value={hospitalCredentials.password}
                    onChange={(event) => setHospitalCredentials((current) => ({ ...current, password: event.target.value }))}
                    placeholder="Leave blank to keep current"
                    style={adminFieldStyle}
                  />
                </div>
                <div style={{ marginTop: 18, display: 'flex', justifyContent: 'flex-end', gap: 10 }}><button type="button" onClick={updateManagedHospitalCredentials} style={policyButtonStyle}>Save credentials</button></div>
              </div>
            </div>
          )}

          {hospitalActivityOpen && (
            <div style={{ position: 'fixed', inset: 0, zIndex: 70, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
              <div role="dialog" aria-modal="true" style={{ width: 'min(900px, 100%)', maxHeight: '85vh', overflowY: 'auto', background: '#fff', borderRadius: 16, boxShadow: '0 26px 64px rgba(15, 23, 42, 0.24)', border: '1px solid #dbeafe', padding: 22 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
                  <div><div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><FiActivity color="#0f766e" /><div><div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Hospital activity</div><h3 style={{ margin: '6px 0 0', color: '#0f172a' }}>Audited activity for {selectedHospitalName}</h3></div></div><p style={{ margin: '8px 0 0', color: '#64748b', fontSize: 13 }}>Every audited hospital action by administrators, doctors, nurses, and other staff.</p></div>
                  <button type="button" onClick={() => setHospitalActivityOpen(false)} style={removeRowButtonStyle}><FiX size={14} /> Close</button>
                </div>
                <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                  {Object.entries(hospitalActivity?.summary || { staff: 0, shifts: 0, encounters: 0, beds: 0, pharmaceuticals: 0 }).map(([label, value]) => <div key={label} style={{ padding: 12, borderRadius: 10, background: '#f8fafc', border: '1px solid #e2e8f0' }}><div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', fontWeight: 800 }}>{label}</div><strong style={{ display: 'block', marginTop: 6, fontSize: 22, color: '#0f172a' }}>{value}</strong></div>)}
                </div>
                <div style={{ marginTop: 16, display: 'flex', justifyContent: 'flex-end' }}><button type="button" onClick={fetchHospitalActivity} disabled={hospitalActivityLoading} style={secondaryButtonStyle}><FiRefreshCw size={15} /> {hospitalActivityLoading ? 'Refreshing...' : 'Refresh activity'}</button></div>
                <div style={{ marginTop: 14 }}>
                  {hospitalActivityLoading && <div style={{ padding: 18, color: '#64748b' }}>Loading activity...</div>}
                  {!hospitalActivityLoading && !(hospitalActivity?.activity?.length) && <div style={{ padding: 18, borderRadius: 10, background: '#f8fafc', color: '#64748b', fontSize: 13 }}>No audited activity has been recorded for this hospital yet.</div>}
                  {!hospitalActivityLoading && hospitalActivity?.activity?.length > 0 && <div style={{ display: 'grid', gap: 8 }}>{hospitalActivity.activity.map((event) => {
                    const changeDetails = event.changeDetails || {};
                    const actor = changeDetails.actor || event.actorId || 'Unknown actor';
                    const oldValue = changeDetails.oldValue ?? changeDetails.previousValue ?? null;
                    const newValue = changeDetails.newValue ?? changeDetails.value ?? null;
                    const timestamp = changeDetails.timestamp || event.timestamp || null;
                    return <div key={event.id} style={{ display: 'grid', gridTemplateColumns: 'minmax(150px, 0.7fr) minmax(0, 2fr) auto', gap: 12, alignItems: 'center', padding: '11px 12px', border: '1px solid #e2e8f0', borderRadius: 10, background: '#f8fafc' }}><div><strong style={{ display: 'block', fontSize: 12, color: '#0f172a' }}>{String(event.action).replaceAll('_', ' ')}</strong><span style={{ fontSize: 11, color: '#64748b' }}>{event.resourceType || 'Hospital'}</span></div><div style={{ fontSize: 12, color: '#475569' }}><div>{event.description || 'Audited hospital activity'}</div>{(oldValue !== null || newValue !== null || actor) && <div style={{ marginTop: 6, color: '#334155', lineHeight: 1.5 }}><strong>Actor:</strong> {String(actor)}{oldValue !== null && <><br /><strong>Old:</strong> {JSON.stringify(oldValue)} </>}{newValue !== null && <><br /><strong>New:</strong> {JSON.stringify(newValue)} </>}{timestamp && <><br /><strong>Time:</strong> {new Date(timestamp).toLocaleString()}</>}</div>}</div><time style={{ fontSize: 11, color: '#64748b', whiteSpace: 'nowrap' }}>{timestamp ? new Date(timestamp).toLocaleString() : event.timestamp ? new Date(event.timestamp).toLocaleString() : 'Unknown time'}</time></div>;
                  })}</div>}
                </div>
              </div>
            </div>
          )}

          {workloadPolicyOpen && (
            <div style={{ position: 'fixed', inset: 0, zIndex: 70, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
              <div role="dialog" aria-modal="true" style={{ width: 'min(820px, 100%)', maxHeight: '88vh', overflowY: 'auto', background: '#fff', borderRadius: 16, boxShadow: '0 26px 64px rgba(15, 23, 42, 0.24)', border: '1px solid #dbeafe', padding: 22 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}><div><div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Workload policy</div><h3 style={{ margin: '8px 0 0', color: '#0f172a' }}>Configure workload for {selectedHospitalName}</h3><p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Set capacity baselines and acuity weights without leaving Hospital management.</p></div><button type="button" onClick={() => setWorkloadPolicyOpen(false)} style={removeRowButtonStyle}><FiX size={14} /> Close</button></div>
                {capacityPolicyLoading ? <p style={{ marginTop: 20, color: '#64748b' }}>Loading policy...</p> : <>
                  <section style={{ marginTop: 18, padding: 16, borderRadius: 12, background: '#f8fbff', border: '1px solid #dbeafe' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(180px, 1fr) minmax(180px, 1fr) auto', gap: 12, alignItems: 'end' }}>
                      <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>Role<select value={capacityRole} onChange={(event) => { const nextRole = event.target.value; setCapacityRole(nextRole); setSavedCapacityValue(capacityConfig?.policies?.defaultCapacityByRole?.[nextRole] ?? null); setCapacityValue(''); }} style={policyInputStyle}><option value="TRIAGE_NURSE">Triage nurse</option><option value="CLINICAL_NURSE">Clinical nurse</option><option value="PHYSICIAN">Physician</option></select></label>
                      <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>Capacity in workload units (1 unit = 1 acuity-weighted workload point)<input type="number" min="1" step="1" value={capacityValue} onChange={(event) => setCapacityValue(event.target.value)} placeholder="Enter a capacity value" style={policyInputStyle} /></label>
                      <button type="button" onClick={saveHospitalCapacity} disabled={capacityPolicySaving} style={policyButtonStyle}><FiSave size={15} /> {capacityPolicySaving ? 'Saving...' : 'Save baseline'}</button>
                    </div>
                    <p style={{ margin: '10px 0 0', color: '#64748b', fontSize: 12 }}>This is the lowest-precedence hospital-wide role default. Select a role to view its current saved baseline, then update it. The current saved value is: <strong>{savedCapacityValue ?? 'not set'}</strong>.</p>
                  </section>
                  <section style={{ marginTop: 14, padding: 16, borderRadius: 12, background: '#fff', border: '1px solid #dbeafe' }}>
                    <h4 style={{ margin: 0, color: '#0f172a' }}>Acuity workload weights</h4>
                    <p style={{ margin: '6px 0 12px', color: '#64748b', fontSize: 13 }}>These multipliers convert active cases into workload units. The required ordering is RED ≥ ORANGE ≥ YELLOW ≥ GREEN.</p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(100px, 1fr))', gap: 10 }}>{Object.keys(acuityWeights).map((priority) => <label key={priority} style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>{priority}<input type="number" min="0.1" step="0.1" value={acuityWeights[priority]} onChange={(event) => setAcuityWeights((current) => ({ ...current, [priority]: event.target.value }))} style={policyInputStyle} /></label>)}</div>
                    {acuityValidationMessage && <div style={{ marginTop: 12, padding: '8px 10px', borderRadius: 8, background: '#fef2f2', color: '#991b1b', fontSize: 12, fontWeight: 600 }}>{acuityValidationMessage}</div>}
                    <button type="button" onClick={saveHospitalAcuityWeights} disabled={acuitySaving} style={{ ...policyButtonStyle, marginTop: 14 }}><FiSave size={15} /> {acuitySaving ? 'Saving...' : 'Save acuity weights'}</button>
                  </section>
                </>}
              </div>
            </div>
          )}
        </div>
      );
    }

    if (workspaceView === 'beds') {
      return <BedsAllocation embedded />;
      /*
      return (
        <section id="beds-management" style={{ display: 'grid', gap: 16 }}>
          <div style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#0f766e' }}>Resource configuration</div>
                <h3 style={{ margin: '6px 0 0', color: '#0f172a' }}>Beds and equipment</h3>
                <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Manage the live inventory for {selectedHospitalName}. Changes here update the hospital records used by bed availability and clinical workflows.</p>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button type="button" onClick={fetchBedConfigurations} disabled={bedConfigurationsLoading} style={secondaryButtonStyle}><FiRefreshCw size={15} /> {bedConfigurationsLoading ? 'Refreshing...' : 'Refresh inventory'}</button>
                <button type="button" onClick={() => setBedEditor({ bed_id: '', ward_unit: 'ICU', room_number: '', sub_unit: '', equipmentText: '', characteristicsText: '', infection_control_tier: 'Standard' })} style={policyButtonStyle}><FiPlus size={15} /> Add bed</button>
              </div>
            </div>
            <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
              {[['Total beds', bedConfigurations.length], ['Available', bedConfigurations.filter((bed) => (bed.schedule?.status || '').toUpperCase() === 'AVAILABLE').length], ['Occupied', bedConfigurations.filter((bed) => (bed.schedule?.status || '').toUpperCase() === 'OCCUPIED').length], ['Equipment items', [...new Set(bedConfigurations.flatMap((bed) => bed.equipment_baseline || []))].length]].map(([label, value]) => <div key={label} style={{ padding: 12, borderRadius: 8, background: '#f8fbff', border: '1px solid #dbeafe' }}><div style={{ fontSize: 11, color: '#64748b', fontWeight: 800, textTransform: 'uppercase' }}>{label}</div><strong style={{ display: 'block', marginTop: 6, fontSize: 22, color: '#0f172a' }}>{value}</strong></div>)}
            </div>
          </div>

          <div style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}><div><h3 style={{ margin: 0, color: '#0f172a' }}>Bed inventory</h3><p style={{ margin: '5px 0 0', color: '#64748b', fontSize: 12 }}>Every bed and its editable equipment baseline.</p></div><span style={{ fontSize: 12, color: '#64748b' }}>{bedConfigurations.length} records</span></div>
            {bedConfigurationsLoading ? <div style={{ padding: 28, color: '#64748b' }}>Loading live bed inventory...</div> : !bedConfigurations.length ? <div style={{ marginTop: 14, padding: 18, borderRadius: 8, background: '#f8fafc', color: '#64748b', fontSize: 13 }}>No bed configurations exist for this hospital yet. Add the first bed to establish live capacity.</div> : <div style={{ marginTop: 14, display: 'grid', gap: 10 }}>{bedConfigurations.map((bed) => <div key={bed.bed_id} style={{ display: 'grid', gridTemplateColumns: 'minmax(120px, 0.7fr) minmax(130px, 0.8fr) minmax(120px, 0.8fr) minmax(0, 2fr) auto', gap: 12, alignItems: 'center', padding: 14, border: '1px solid #e2e8f0', borderRadius: 10, background: '#f8fbff' }}><div><strong style={{ color: '#0f172a' }}>{bed.bed_id}</strong><div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>Bed identifier</div></div><div><strong style={{ color: '#334155', fontSize: 12 }}>{bed.ward_unit}</strong><div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>{bed.sub_unit || 'No sub-unit'}</div></div><div><strong style={{ color: '#334155', fontSize: 12 }}>{bed.room_number || bed.schedule?.roomNumber || 'No room'}</strong><div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>Room</div></div><div><div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{(bed.equipment_baseline || []).length ? bed.equipment_baseline.map((item) => <span key={item} style={{ padding: '5px 8px', borderRadius: 6, background: '#fff', border: '1px solid #cbd5e1', color: '#334155', fontSize: 11 }}>{item}</span>) : <span style={{ color: '#94a3b8', fontSize: 12 }}>No equipment configured</span>}</div><span style={{ display: 'inline-block', marginTop: 6, padding: '4px 7px', borderRadius: 999, background: (bed.schedule?.status || '').toUpperCase() === 'OCCUPIED' ? '#fee2e2' : '#dcfce7', color: (bed.schedule?.status || '').toUpperCase() === 'OCCUPIED' ? '#991b1b' : '#166534', fontSize: 10, fontWeight: 800 }}>{bed.schedule?.status || bed.status || 'Available'}</span></div><div style={{ display: 'flex', gap: 6 }}><button type="button" onClick={() => setBedEditor({ ...bed, originalBedId: bed.bed_id, equipmentText: (bed.equipment_baseline || []).join(', '), characteristicsText: (bed.special_characteristics || []).join(', ') })} style={secondaryButtonStyle}>Edit</button><button type="button" onClick={() => deleteBedConfiguration(bed)} style={removeRowButtonStyle}>Delete</button></div></div>)}</div>}
          </div>

          {bedEditor && <div role="dialog" aria-modal="true" style={{ position: 'fixed', inset: 0, zIndex: 70, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}><form onSubmit={saveBedConfiguration} style={{ width: 'min(680px, 100%)', background: '#fff', borderRadius: 14, padding: 22, border: '1px solid #dbeafe', boxShadow: '0 26px 64px rgba(15, 23, 42, 0.24)' }}><div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}><div><div style={{ fontSize: 11, color: '#0f766e', fontWeight: 800, textTransform: 'uppercase' }}>{bedEditor.originalBedId ? 'Edit configuration' : 'New configuration'}</div><h3 style={{ margin: '7px 0 0', color: '#0f172a' }}>{bedEditor.originalBedId ? `Configure ${bedEditor.originalBedId}` : 'Add a hospital bed'}</h3></div><button type="button" onClick={() => setBedEditor(null)} style={removeRowButtonStyle}><FiX size={14} /> Close</button></div><div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}><label style={adminFieldStyle}>Bed ID<input value={bedEditor.bed_id} disabled={Boolean(bedEditor.originalBedId)} onChange={(event) => setBedEditor((current) => ({ ...current, bed_id: event.target.value }))} style={policyInputStyle} /></label><label style={adminFieldStyle}>Ward<select value={bedEditor.ward_unit} onChange={(event) => setBedEditor((current) => ({ ...current, ward_unit: event.target.value }))} style={policyInputStyle}>{['ICU', 'NICU', 'PICU', 'HDU', 'Medical Ward', 'Surgical Ward', 'Pediatric Ward', 'Maternity Ward', 'Psychiatric Ward', 'Infectious Disease Ward'].map((ward) => <option key={ward} value={ward}>{ward}</option>)}</select></label><label style={adminFieldStyle}>Sub-unit<input value={bedEditor.sub_unit || ''} onChange={(event) => setBedEditor((current) => ({ ...current, sub_unit: event.target.value }))} style={policyInputStyle} /></label><label style={adminFieldStyle}>Infection control<select value={bedEditor.infection_control_tier || 'Standard'} onChange={(event) => setBedEditor((current) => ({ ...current, infection_control_tier: event.target.value }))} style={policyInputStyle}>{['Standard', 'Contact', 'Droplet', 'Airborne', 'Enhanced Precautions'].map((tier) => <option key={tier} value={tier}>{tier}</option>)}</select></label><label style={{ ...adminFieldStyle, gridColumn: '1 / -1' }}>Equipment, comma separated<input value={bedEditor.equipmentText} onChange={(event) => setBedEditor((current) => ({ ...current, equipmentText: event.target.value }))} placeholder="Cardiac Monitor, Oxygen Delivery, Infusion Pump" style={policyInputStyle} /></label><label style={{ ...adminFieldStyle, gridColumn: '1 / -1' }}>Bed characteristics, comma separated<input value={bedEditor.characteristicsText} onChange={(event) => setBedEditor((current) => ({ ...current, characteristicsText: event.target.value }))} placeholder="Electric Bed, Isolation Capsule" style={policyInputStyle} /></label></div><div style={{ marginTop: 18, display: 'flex', justifyContent: 'flex-end', gap: 8 }}><button type="button" onClick={() => setBedEditor(null)} style={secondaryButtonStyle}>Cancel</button><button type="submit" disabled={bedConfigurationSaving} style={policyButtonStyle}>{bedConfigurationSaving ? 'Saving...' : bedEditor.originalBedId ? 'Save configuration' : 'Create bed'}</button></div></form></div>}
        </section>
      );
      */
    }

    if (workspaceView === 'ehr-configurations') {
      console.log('[EHR CONFIG] EHR Configurations workspace opened');
      const adapterCards = ehrConfig.adapters || [];
      const terminologyRows = ehrConfig.terminologyRows || [];
      const retentionRows = ehrConfig.retention || [];
      const alertRows = ehrConfig.alertRouting || [];
      const accessRows = ehrConfig.accessMatrix || [];
      const deadLetter = ehrConfig.deadLetter || { inboundFailures: 0, retryQueue: 0, failedMessages: [], retryItems: [] };
      return (
        <section id="ehr-configurations" style={{ display: 'grid', gap: 18, background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
          {ehrToast && (
            <div role="status" aria-live="polite" style={{ padding: '10px 12px', borderRadius: 10, background: ehrToast.tone === 'warning' ? '#fff7ed' : '#dcfce7', border: `1px solid ${ehrToast.tone === 'warning' ? '#fdba74' : '#86efac'}`, color: ehrToast.tone === 'warning' ? '#9a4d00' : '#166534', fontSize: 12, fontWeight: 700 }}>
              {ehrToast.message}
            </div>
          )}

          <div>
            <div style={{ marginBottom: 6, fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#0f766e' }}>Hospital administration</div>
            <h2 style={{ margin: 0, fontSize: 28, fontWeight: 700, color: '#0f172a' }}>EHR Configurations</h2>
            <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Clinical configuration settings and hospital EHR policy controls live here. The canonical model remains FHIR while adapters translate HL7v2 and future protocol feeds into the same internal representation.</p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, flexWrap: 'wrap' }}>
            <button type="button" onClick={handleEhrSettingsSave} style={policyButtonStyle}>Save EHR settings</button>
            <button type="button" onClick={() => handleTestConnection('hl7v2')} style={secondaryButtonStyle}>Test HL7v2</button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
            <div style={{ padding: 14, borderRadius: 10, border: '1px solid #dbeafe', background: '#f8fbff' }}>
              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Hospital context</div>
              <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{selectedHospitalName}</div>
            </div>
            <div style={{ padding: 14, borderRadius: 10, border: '1px solid #dbeafe', background: '#f8fbff' }}>
              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Canonical format</div>
              <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{ehrConfig.canonicalFormat || 'FHIR R4'}</div>
            </div>
            <div style={{ padding: 14, borderRadius: 10, border: '1px solid #dbeafe', background: '#f8fbff' }}>
              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Workload policy</div>
              <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{capacityConfig?.policies?.defaultCapacityByRole ? 'Configured' : 'Needs setup'}</div>
            </div>
          </div>

          <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#f8fbff' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <FiDatabase color="#0f766e" size={18} />
              <h4 style={{ margin: 0, color: '#0f172a' }}>Protocol-agnostic adapter contract</h4>
            </div>
            <p style={{ margin: '0 0 12px', color: '#475569', fontSize: 13 }}>Each adapter translates to and from the canonical FHIR model. The core workflow never needs protocol-specific logic, and future protocols can be added without rewriting the core.</p>
            <div style={{ display: 'grid', gap: 8 }}>
              {protocolAdapterInterface.contract.map((item) => (
                <div key={item} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', borderRadius: 10, background: '#fff', border: '1px solid #dbeafe' }}>
                  <FiCheckCircle size={14} color="#0f766e" />
                  <span style={{ fontSize: 12, color: '#334155' }}>{item}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <FiShield color="#0f766e" size={18} />
              <h4 style={{ margin: 0, color: '#0f172a' }}>Interoperability / protocol status</h4>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
              {adapterCards.map((adapter) => (
                <div key={adapter.id || adapter.protocolName} style={{ border: '1px solid #dbeafe', borderRadius: 12, background: '#f8fbff', padding: 14 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                    <strong style={{ color: '#0f172a', fontSize: 15 }}>{adapter.name || adapter.protocolName}</strong>
                    <span style={{ padding: '4px 8px', borderRadius: 999, background: adapter.status === 'Enabled' ? '#dcfce7' : '#e2e8f0', color: adapter.status === 'Enabled' ? '#166534' : '#475569', fontSize: 11, fontWeight: 800 }}>{adapter.status}</span>
                  </div>
                  <div style={{ marginTop: 10, fontSize: 12, color: '#475569', display: 'grid', gap: 6 }}>
                    <div><strong>Transport:</strong> {adapter.protocol || adapter.transport}</div>
                    <div><strong>Listener:</strong> {adapter.host ? `${adapter.host}:${adapter.port}` : adapter.listener}</div>
                    <div><strong>Connection:</strong> {adapter.connection || adapter.connectionStatus}</div>
                    <div><strong>Messages:</strong> {(adapter.outboundRoutes || adapter.supportedMessages || []).join(', ') || 'None configured'}</div>
                    <div><strong>Last test:</strong> {adapter.lastTest || 'Not run'}</div>
                  </div>
                  <div style={{ marginTop: 12, display: 'flex', justifyContent: 'flex-end' }}>
                    <button type="button" onClick={() => handleTestConnection(adapter.id || adapter.protocolName)} style={secondaryButtonStyle}>Test connection</button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <FiAlertTriangle color="#f59e0b" size={18} />
              <h4 style={{ margin: 0, color: '#0f172a' }}>Clinical configuration and policy controls</h4>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
              <div style={{ padding: 12, borderRadius: 10, border: '1px solid #dbeafe', background: '#f8fbff' }}>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>SATS protocol</div>
                <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{ehrConfig.protocolPolicy?.satsVersion || 'SATS v3.1'}</div>
              </div>
              <div style={{ padding: 12, borderRadius: 10, border: '1px solid #dbeafe', background: '#f8fbff' }}>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>TEWS protocol</div>
                <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{ehrConfig.protocolPolicy?.tewsVersion || 'TEWS v2.4'}</div>
              </div>
              <div style={{ padding: 12, borderRadius: 10, border: '1px solid #dbeafe', background: '#f8fbff' }}>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Report requirements</div>
                <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>{ehrConfig.protocolPolicy?.reportRequirements || 'System default'}</div>
              </div>
            </div>
          </div>

          <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
              <h4 style={{ margin: 0, color: '#0f172a' }}>Terminology mapping & review queue</h4>
              <button type="button" onClick={() => handleEhrSettingsSave()} style={secondaryButtonStyle}>Save mappings</button>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', color: '#334155' }}>
                    <th style={{ textAlign: 'left', padding: 10, borderBottom: '1px solid #dbeafe' }}>Local code</th>
                    <th style={{ textAlign: 'left', padding: 10, borderBottom: '1px solid #dbeafe' }}>Standard code</th>
                    <th style={{ textAlign: 'left', padding: 10, borderBottom: '1px solid #dbeafe' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: 10, borderBottom: '1px solid #dbeafe' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {terminologyRows.map((row) => (
                    <tr key={row.code || row.localCode}>
                      <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0', color: '#0f172a' }}>{row.code || row.localCode}</td>
                      <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0', color: '#475569' }}>{row.value || row.standardCode || 'Awaiting review'}</td>
                      <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>
                        <select value={row.status} onChange={(event) => setEhrConfig((current) => ({ ...current, terminologyRows: current.terminologyRows.map((item) => item.code === row.code || item.localCode === row.localCode ? { ...item, status: event.target.value } : item) }))} style={policyInputStyle}>
                          <option value="Mapped">Mapped</option>
                          <option value="Needs review">Needs review</option>
                        </select>
                      </td>
                      <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>
                        <button type="button" onClick={() => handleEhrMappingSave(row.code || row.localCode)} style={secondaryButtonStyle}>Save mapping</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16 }}>
            <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
              <h4 style={{ margin: '0 0 12px', color: '#0f172a' }}>Retention policy</h4>
              <div style={{ display: 'grid', gap: 10 }}>
                {retentionRows.map((item) => (
                  <div key={item.id || item.category} style={{ padding: 10, borderRadius: 10, background: '#f8fbff', border: '1px solid #dbeafe' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                      <strong style={{ display: 'block', color: '#0f172a', marginBottom: 4 }}>{item.category}</strong>
                      <button type="button" onClick={() => handleRetentionSave(item.id || item.category)} style={secondaryButtonStyle}>Save</button>
                    </div>
                    <label style={{ display: 'grid', gap: 4, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                      Retention window
                      <input value={item.window || item.retention || ''} onChange={(event) => setEhrConfig((current) => ({ ...current, retention: current.retention.map((entry) => (entry.id === item.id || entry.category === item.category ? { ...entry, window: event.target.value } : entry)) }))} style={policyInputStyle} />
                    </label>
                    <div style={{ marginTop: 8, fontSize: 12, color: '#475569' }}>Minimum: {item.minimum || 'Not set'}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
              <h4 style={{ margin: '0 0 12px', color: '#0f172a' }}>Alert routing</h4>
              <div style={{ display: 'grid', gap: 10 }}>
                {alertRows.map((item) => (
                  <div key={item.alertType} style={{ padding: 10, borderRadius: 10, background: '#f8fbff', border: '1px solid #dbeafe' }}>
                    <strong style={{ display: 'block', color: '#0f172a', marginBottom: 4 }}>{item.alertType}</strong>
                    <div style={{ fontSize: 12, color: '#475569' }}>Targets: {(item.targets || []).join(', ')}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
              <h4 style={{ margin: '0 0 12px', color: '#0f172a' }}>Access matrix</h4>
              <div style={{ display: 'grid', gap: 10 }}>
                {accessRows.map((item) => (
                  <div key={item.role} style={{ padding: 10, borderRadius: 10, background: '#f8fbff', border: '1px solid #dbeafe' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
                      <strong style={{ display: 'block', color: '#0f172a', marginBottom: 4 }}>{item.role}</strong>
                      <button type="button" onClick={() => handleAccessMatrixSave(item.role)} style={secondaryButtonStyle}>Save access</button>
                    </div>
                    <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
                      <label style={{ display: 'grid', gap: 4, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                        View access
                        <select value={item.canView !== undefined ? String(item.canView) : 'true'} onChange={(event) => setEhrConfig((current) => ({ ...current, accessMatrix: current.accessMatrix.map((entry) => entry.role === item.role ? { ...entry, canView: event.target.value === 'true' } : entry) }))} style={policyInputStyle}>
                          <option value="true">Allow view</option>
                          <option value="false">Hidden</option>
                        </select>
                      </label>
                      <label style={{ display: 'grid', gap: 4, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                        Edit access
                        <select value={item.canEdit !== undefined ? String(item.canEdit) : 'false'} onChange={(event) => setEhrConfig((current) => ({ ...current, accessMatrix: current.accessMatrix.map((entry) => entry.role === item.role ? { ...entry, canEdit: event.target.value === 'true' } : entry) }))} style={policyInputStyle}>
                          <option value="true">Allow edit</option>
                          <option value="false">View only</option>
                        </select>
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div style={{ padding: 18, borderRadius: 12, border: '1px solid #dbeafe', background: '#f8fbff' }}>
            <h4 style={{ margin: '0 0 12px', color: '#0f172a' }}>Dead-letter / retry queue</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
              <div style={{ padding: 10, borderRadius: 10, background: '#fff', border: '1px solid #dbeafe' }}><strong>Inbound failures</strong><div style={{ marginTop: 6, fontSize: 18, fontWeight: 700 }}>{deadLetter.inboundFailures}</div></div>
              <div style={{ padding: 10, borderRadius: 10, background: '#fff', border: '1px solid #dbeafe' }}><strong>Retry queue</strong><div style={{ marginTop: 6, fontSize: 18, fontWeight: 700 }}>{deadLetter.retryQueue}</div></div>
              <div style={{ padding: 10, borderRadius: 10, background: '#fff', border: '1px solid #dbeafe' }}><strong>Last failure</strong><div style={{ marginTop: 6, fontSize: 12, color: '#475569' }}>{deadLetter.failedMessages?.[0]?.reason || deadLetter.lastFailure || 'No recent failures'}</div></div>
            </div>
            <div style={{ marginTop: 14, display: 'grid', gap: 8 }}>
              {(deadLetter.failedMessages || []).map((message) => (
                <div key={message.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, padding: 10, borderRadius: 10, background: '#fff', border: '1px solid #fbd38d' }}>
                  <div>
                    <strong style={{ display: 'block', color: '#0f172a' }}>{message.id}</strong>
                    <span style={{ fontSize: 12, color: '#475569' }}>{message.reason}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button type="button" onClick={() => handleRetryQueueAction(message.id, 'retry')} style={secondaryButtonStyle}>Retry</button>
                    <button type="button" onClick={() => handleRetryQueueAction(message.id, 'permanent')} style={removeRowButtonStyle}>Permanent failure</button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      );
    }

    if (workspaceView === 'labs-configurations') {
      const registration = getAdminWorkspaceViewRegistration(workspaceView);
      if (registration === 'LabRequestsAdminPage') return <AdminLabRequestsPage />;
      return <LabRequestsPage />;
    }
    if (workspaceView === 'pharmacy-configurations') {
      const registration = getAdminWorkspaceViewRegistration(workspaceView);
      if (registration === 'PharmacyAdminPage') return <AdminPharmacyPage />;
      return <Pharmacy embedded />;
    }
    if (workspaceView === 'analytics') {
      const registration = getAdminWorkspaceViewRegistration(workspaceView);
      if (registration === 'TriageAnalyticsAdminPage') return <AdminAnalyticsPage />;
      return <TriageAnalytics embedded />;
    }

    if (workspaceView === 'staff') {
      return (
        <section id="staff-management" style={{ display: 'grid', gap: 16 }}>
          <div style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
              <div>
                <div style={{ marginBottom: 6, fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#0f766e' }}>Hospital administration</div>
                <h2 style={{ margin: 0, fontSize: 28, fontWeight: 700, color: '#0f172a' }}>Staff</h2>
                <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Review and edit existing credentials for {selectedHospitalName}. Onboarding new staff is available as a separate action below.</p>
              </div>
              <button type="button" onClick={() => { setStaffOnboardingOpen(true); fetchHospitalStructure(); }} style={policyButtonStyle}><FiPlus size={15} /> Onboard new staff</button>
            </div>

            <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, padding: 14, background: '#f8fafc', borderRadius: 10, border: '1px solid #e2e8f0' }}>
              <div>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6, color: '#334155' }}>Search by name/email</label>
                <input type="text" placeholder="Search staff..." value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', border: '1px solid #cbd5e1', borderRadius: 6, background: '#fff', color: '#0f172a', fontSize: 14 }} />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6, color: '#334155' }}>Department</label>
                <select value={departmentFilter} onChange={(event) => setDepartmentFilter(event.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', border: '1px solid #cbd5e1', borderRadius: 6, background: '#fff', color: '#0f172a', fontSize: 14 }}>
                  <option value="all">All departments</option>
                  {DEPARTMENTS.map((department) => <option key={department} value={department}>{department}</option>)}
                </select>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6, color: '#334155' }}>Role</label>
                <select value={roleFilter} onChange={(event) => setRoleFilter(event.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', border: '1px solid #cbd5e1', borderRadius: 6, background: '#fff', color: '#0f172a', fontSize: 14 }}>
                  <option value="all">All roles</option>
                  {ROLES.map((role) => <option key={role} value={role}>{role}</option>)}
                </select>
              </div>
            </div>

            <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
              {searchLoading && <div style={{ padding: 18, color: '#64748b' }}>Loading existing staff...</div>}
              {!searchLoading && filteredStaff.length === 0 && <div style={{ padding: 18, borderRadius: 10, background: '#f8fafc', color: '#64748b', fontSize: 13 }}>No existing staff credentials found for this hospital.</div>}
              {!searchLoading && filteredStaff.map((staff) => (
                <div key={staff._id} style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(150px, 1fr) auto auto', gap: 12, alignItems: 'center', padding: '12px 14px', border: '1px solid #e2e8f0', borderRadius: 10, background: '#f8fafc' }}>
                  <div><strong style={{ color: '#0f172a' }}>{staff.firstName || staff.name} {staff.lastName || staff.surname}</strong><div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>{staff.email}</div></div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{staff.positionTitle || HOSPITAL_STAFF_ROLES.find((position) => position.code === staff.positionCode)?.label || staff.systemRole || staff.role} • {staff.assignment?.department || staff.department}<div style={{ marginTop: 3, color: '#64748b' }}>{staff.clinicalSpecialty || staff.specialty || 'No specialty'}{staff.systemRole && <span> • Permission role: {staff.systemRole}</span>}</div></div>
                  <button type="button" onClick={() => openStaffEditor(staff)} disabled={hospitalStructureLoading} style={secondaryButtonStyle}><FiSliders size={14} /> Edit credentials</button>
                  <button type="button" onClick={() => deleteStaffMember(staff._id, `${staff.firstName || staff.name} ${staff.lastName || staff.surname}`.trim() || 'this staff member')} style={{ ...removeRowButtonStyle, padding: '8px 12px' }}><FiX size={14} /> Delete</button>
                </div>
              ))}
            </div>
          </div>

          {staffOnboardingOpen && <div style={{ position: 'fixed', inset: 0, zIndex: 70, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
          <div role="dialog" aria-modal="true" style={{ width: 'min(900px, 100%)', maxHeight: '88vh', overflowY: 'auto', background: '#fff', borderRadius: 16, boxShadow: '0 26px 64px rgba(15, 23, 42, 0.24)', border: '1px solid #dbeafe', padding: 22 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}><div><div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Staff onboarding</div><h3 style={{ margin: '8px 0 0', color: '#0f172a' }}>Onboard new staff</h3><p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Create authenticated staff records for {selectedHospitalName}.</p></div><button type="button" onClick={() => setStaffOnboardingOpen(false)} style={removeRowButtonStyle}><FiX size={14} /> Close</button></div>
          {hospitalStructureLoading && <div role="status" style={{ marginTop: 12, color: '#475569', fontSize: 13 }}>Loading configured hospital departments and specialties...</div>}
          {!hospitalStructureLoading && !hospitalStructure.departments.some((department) => department.isActive !== false) && <div role="alert" style={{ marginTop: 12, padding: 10, borderRadius: 8, background: '#fff7ed', color: '#9a3412', fontSize: 13 }}>Configure at least one active hospital department before onboarding staff.</div>}
          {!hospitalStructureLoading && hospitalStructure.departments.some((department) => department.isActive !== false) && !hospitalStructure.departments.some((department) => department.isActive !== false && getConfiguredDepartmentSpecialties(hospitalStructure.departments, department.name).length) && <div role="alert" style={{ marginTop: 12, padding: 10, borderRadius: 8, background: '#fff7ed', color: '#9a3412', fontSize: 13 }}>Configure at least one specialty for an active department before onboarding staff.</div>}
          <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
            {hospitalOperationsStaffRows.map((row, index) => (
              <div key={`staff-${index}`} style={{ padding: 14, border: '1px solid #e2e8f0', borderRadius: 8, background: '#f8fafc' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 8 }}>
                  {[
                    ['firstName', 'First name'], ['lastName', 'Last name'], ['email', 'Work email']
                  ].map(([field, label]) => <label key={field} style={adminFieldStyle}>{label}<input type={field === 'email' ? 'email' : 'text'} value={row[field]} onChange={(event) => updateStaffRow(index, field, event.target.value)} placeholder={label} style={policyInputStyle} /></label>)}
                  <PasswordField
                    label="Account password"
                    value={row.password}
                    onChange={(event) => updateStaffRow(index, 'password', event.target.value)}
                    placeholder="Create password"
                    style={adminFieldStyle}
                  />
                  <label style={adminFieldStyle}>Position / job<select value={row.positionCode} onChange={(event) => updateStaffRow(index, 'positionCode', event.target.value)} style={policyInputStyle}>
                    <option value="">Select a position</option>
                    {hospitalStructure.staffRoles.map((position) => <option key={position.code} value={position.code}>{position.label} — {position.category}</option>)}
                  </select></label>
                  <label style={adminFieldStyle}>Department<select value={row.department} onChange={(event) => updateStaffRow(index, 'department', event.target.value)} style={policyInputStyle}>
                    <option value="">Select an active department</option>
                    {hospitalStructure.departments.filter((department) => department.isActive !== false).map((department) => <option key={department.name} value={department.name}>{department.name}</option>)}
                  </select></label>
                  <label style={adminFieldStyle}>Configured specialty<select value={row.specialty} onChange={(event) => updateStaffRow(index, 'specialty', event.target.value)} disabled={!getConfiguredDepartmentSpecialties(hospitalStructure.departments, row.department).length} style={policyInputStyle}>
                    <option value="">{getConfiguredDepartmentSpecialties(hospitalStructure.departments, row.department).length ? 'Select a specialty' : 'No specialties configured'}</option>
                    {getConfiguredDepartmentSpecialties(hospitalStructure.departments, row.department).map((specialty) => <option key={specialty} value={specialty}>{specialty}</option>)}
                  </select></label>
                  {hospitalStructure.staffRoles.find((position) => position.code === row.positionCode)?.requiresCredentials && (
                    <>
                      <label style={adminFieldStyle}>Credential title<select value={row.title} onChange={(event) => updateStaffRow(index, 'title', event.target.value)} style={policyInputStyle}><option value="">Select a title</option>{['MD', 'DO', 'RN', 'NP', 'PA', 'PhD'].map((title) => <option key={title} value={title}>{title}</option>)}</select></label>
                      <label style={adminFieldStyle}>License number<input value={row.licenseNumber} onChange={(event) => updateStaffRow(index, 'licenseNumber', event.target.value)} placeholder="License number" style={policyInputStyle} /></label>
                    </>
                  )}
                  <label style={adminFieldStyle}>NPI number (optional)<input value={row.npiNumber} onChange={(event) => updateStaffRow(index, 'npiNumber', event.target.value)} placeholder="NPI number" style={policyInputStyle} /></label>
                </div>
                {hospitalOperationsStaffRows.length > 1 && <button type="button" onClick={() => setHospitalOperationsStaffRows((rows) => rows.filter((_, rowIndex) => rowIndex !== index))} style={removeRowButtonStyle}><FiX size={14} /> Remove staff row</button>}
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <button type="button" onClick={() => setHospitalOperationsStaffRows((rows) => [...rows, createStaffDraft()])} style={secondaryButtonStyle}><FiPlus size={15} /> Add staff row</button>
            <button type="button" onClick={onboardHospitalStaff} disabled={hospitalOperationsAction === 'staff/bulk'} style={policyButtonStyle}><FiUsers size={15} /> {hospitalOperationsAction === 'staff/bulk' ? 'Onboarding...' : 'Onboard staff batch'}</button>
          </div>
          </div></div>}

          {staffEditorOpen && staffEditor && <div style={{ position: 'fixed', inset: 0, zIndex: 70, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
            <div role="dialog" aria-modal="true" style={{ width: 'min(760px, 100%)', maxHeight: '88vh', overflowY: 'auto', background: '#fff', borderRadius: 16, boxShadow: '0 26px 64px rgba(15, 23, 42, 0.24)', border: '1px solid #dbeafe', padding: 22 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}><div><div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0f766e' }}>Existing staff credential</div><h3 style={{ margin: '8px 0 0', color: '#0f172a' }}>Edit {staffEditor.firstName} {staffEditor.lastName}</h3></div><button type="button" onClick={() => setStaffEditorOpen(false)} style={removeRowButtonStyle}><FiX size={14} /> Close</button></div>
              {staffEditorError && <div role="alert" style={{ marginTop: 14, padding: 10, borderRadius: 8, background: '#fee2e2', color: '#991b1b', fontSize: 13 }}>{staffEditorError}</div>}
              {!staffEditor.positionCode && <div role="status" style={{ marginTop: 14, padding: 10, borderRadius: 8, background: '#fff7ed', color: '#9a3412', fontSize: 13 }}>This legacy record has no specific position. Select the correct job before saving; the permission role alone cannot distinguish triage from clinical nursing.</div>}
              {!staffEditor.positionCode && <div role="alert" style={{ marginTop: 14, padding: 10, borderRadius: 8, background: '#fff7ed', color: '#9a3412', fontSize: 13 }}>This legacy staff record has no position code. Choose the correct position before saving so its permission role is backfilled accurately.</div>}
              <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                {['firstName', 'lastName', 'email'].map((field) => <label key={field} style={adminFieldStyle}>{field === 'firstName' ? 'First name' : field === 'lastName' ? 'Last name' : 'Work email'}<input type={field === 'email' ? 'email' : 'text'} value={staffEditor[field]} onChange={(event) => updateStaffEditor(field, event.target.value)} style={policyInputStyle} /></label>)}
                <label style={adminFieldStyle}>Position / job<select value={staffEditor.positionCode} onChange={(event) => updateStaffEditor('positionCode', event.target.value)} style={policyInputStyle}>
                  <option value="">Select a position</option>
                  {HOSPITAL_STAFF_ROLES.map((position) => <option key={position.code} value={position.code}>{position.label} — {position.category}</option>)}
                </select>
                  <span style={{ display: 'block', marginTop: 4, color: '#64748b', fontSize: 11 }}>Permission role: {staffEditor.role || 'selected from position'}</span>
                </label>
                <label style={adminFieldStyle}>Department<select value={staffEditor.department} onChange={(event) => updateStaffEditor('department', event.target.value)} style={policyInputStyle}>
                  <option value="">Select an active department</option>
                  {hospitalStructure.departments.filter((department) => department.isActive !== false).map((department) => <option key={department.name} value={department.name}>{department.name}</option>)}
                </select></label>
                <label style={adminFieldStyle}>Configured specialty<select value={staffEditor.specialty} onChange={(event) => updateStaffEditor('specialty', event.target.value)} disabled={!getConfiguredDepartmentSpecialties(hospitalStructure.departments, staffEditor.department).length} style={policyInputStyle}>
                  <option value="">{getConfiguredDepartmentSpecialties(hospitalStructure.departments, staffEditor.department).length ? 'Select a specialty' : 'No specialties configured'}</option>
                  {getConfiguredDepartmentSpecialties(hospitalStructure.departments, staffEditor.department).map((specialty) => <option key={specialty} value={specialty}>{specialty}</option>)}
                </select></label>
                <PasswordField
                  label="Password"
                  value={staffEditor.password || ''}
                  onChange={(event) => updateStaffEditor('password', event.target.value)}
                  placeholder="Leave blank to keep current"
                  style={adminFieldStyle}
                />
                {HOSPITAL_STAFF_ROLES.find((position) => position.code === staffEditor.positionCode)?.requiresCredentials && (
                  <>
                    <label style={adminFieldStyle}>Credential title<select value={staffEditor.title} onChange={(event) => updateStaffEditor('title', event.target.value)} style={policyInputStyle}><option value="">Select a title</option>{['MD', 'DO', 'RN', 'NP', 'PA', 'PhD'].map((title) => <option key={title} value={title}>{title}</option>)}</select></label>
                    <label style={adminFieldStyle}>License number<input value={staffEditor.licenseNumber} onChange={(event) => updateStaffEditor('licenseNumber', event.target.value)} style={policyInputStyle} /></label>
                  </>
                )}
                <label style={adminFieldStyle}>NPI number (optional)<input value={staffEditor.npiNumber} onChange={(event) => updateStaffEditor('npiNumber', event.target.value)} style={policyInputStyle} /></label>
                <label style={adminFieldStyle}>Status<select value={staffEditor.status} onChange={(event) => updateStaffEditor('status', event.target.value)} style={policyInputStyle}><option value="ACTIVE">Active</option><option value="ON_LEAVE">On leave</option><option value="TERMINATED">Terminated</option><option value="PENDING_CREDENTIALING">Pending credentialing</option></select></label>
              </div>
              <div style={{ marginTop: 18, display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center' }}>
                <button type="button" onClick={() => deleteStaffMember(staffEditor._id, `${staffEditor.firstName} ${staffEditor.lastName}`.trim() || 'this staff member')} style={{ ...removeRowButtonStyle, padding: '10px 14px' }}><FiX size={15} /> Delete staff</button>
                <button type="button" onClick={saveStaffEditor} disabled={staffEditorSaving} style={policyButtonStyle}><FiSave size={15} /> {staffEditorSaving ? 'Saving...' : 'Save staff credentials'}</button>
              </div>
            </div>
          </div>}
        </section>
      );
    }

    if (workspaceView === 'patients') {
      return (
        <section id="patients-seeding" style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
          <div style={{ marginBottom: 6, fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#0f766e' }}>Hospital administration</div>
          <h2 style={{ margin: 0, fontSize: 28, fontWeight: 700, color: '#0f172a' }}>Patients</h2>
          <p style={{ margin: '6px 0 14px', color: '#64748b', fontSize: 13 }}>Provision hospital-scoped encounter records for triage, role dashboards, workload, and downstream actions. This does not modify real patient records.</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 240px))', gap: 12, alignItems: 'end' }}>
            <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>Encounter count<input type="number" min="1" max="1000" value={hospitalOperationsEncounterCount} onChange={(event) => setHospitalOperationsEncounterCount(event.target.value)} style={policyInputStyle} /></label>
            <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>Historical span in days<input type="number" min="0" max="3650" value={hospitalOperationsHistoricalDays} onChange={(event) => setHospitalOperationsHistoricalDays(event.target.value)} style={policyInputStyle} /></label>
          </div>
          <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>Complaint<select value={hospitalOperationsComplaintType} onChange={(event) => setHospitalOperationsComplaintType(event.target.value)} style={policyInputStyle}><option value="FEVER">Fever and weakness</option><option value="CHEST_PAIN">Chest pain</option><option value="ABDOMINAL_PAIN">Abdominal pain</option><option value="TRAUMA">Trauma</option><option value="HEADACHE">Severe headache</option><option value="RESPIRATORY">Respiratory distress</option><option value="CUSTOM">Custom complaint</option></select></label>
            <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>Priority<select value={hospitalOperationsPriority} onChange={(event) => setHospitalOperationsPriority(event.target.value)} style={policyInputStyle}><option value="MIXED">Mixed acuity</option><option value="RED">RED</option><option value="ORANGE">ORANGE</option><option value="YELLOW">YELLOW</option><option value="GREEN">GREEN</option></select></label>
            <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>Lifecycle status<select value={hospitalOperationsLifecycle} onChange={(event) => setHospitalOperationsLifecycle(event.target.value)} style={policyInputStyle}><option value="TRIAGE_IN_PROGRESS">Triage in progress</option><option value="OFFICIAL_REPORT">Official report</option><option value="AWAITING_PHYSICIAN">Awaiting physician</option><option value="SEEN">Seen</option><option value="DIAGNOSED">Diagnosed</option><option value="HOSPITALIZED">Hospitalized</option><option value="DISCHARGED">Discharged</option></select></label>
          </div>
          {hospitalOperationsComplaintType === 'CUSTOM' && <label style={{ display: 'grid', gap: 5, marginTop: 12, fontSize: 12, fontWeight: 700, color: '#334155' }}>Custom complaint<input value={hospitalOperationsCustomComplaint} onChange={(event) => setHospitalOperationsCustomComplaint(event.target.value)} placeholder="Describe the presenting complaint" style={policyInputStyle} /></label>}
          <label style={{ display: 'grid', gap: 5, marginTop: 12, fontSize: 12, fontWeight: 700, color: '#334155' }}>Patient names, one per line<input value={hospitalOperationsPatientNames} onChange={(event) => setHospitalOperationsPatientNames(event.target.value)} style={policyInputStyle} /></label>
          <div style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <button type="button" onClick={() => runHospitalOperationsAction('encounters/bulk', { count: Number(hospitalOperationsEncounterCount), historicalDays: Number(hospitalOperationsHistoricalDays), patientNames: hospitalOperationsPatientNames.split(/\r?\n|,/).map((name) => name.trim()).filter(Boolean), complaintType: hospitalOperationsComplaintType, complaint: hospitalOperationsCustomComplaint, priority: hospitalOperationsPriority === 'MIXED' ? '' : hospitalOperationsPriority, lifecycleStatus: hospitalOperationsLifecycle, triageNurseId: hospitalOperationsTriageNurseId || undefined })} disabled={hospitalOperationsAction === 'encounters/bulk'} style={policyButtonStyle}><FiPlus size={15} /> {hospitalOperationsAction === 'encounters/bulk' ? 'Provisioning...' : 'Provision encounters'}</button>
            <span style={{ color: '#64748b', fontSize: 12 }}>Names and selected lifecycle will be written into synthetic records and audit metadata.</span>
          </div>
        </section>
      );
    }

    if (workspaceView === 'shifts') {
      return (
        <div style={{ display: 'grid', gap: 20 }}>
          <section style={{ padding: 20, borderRadius: 8, background: '#fff', color: '#0f172a', border: '1px solid #dbeafe', boxShadow: '0 1px 3px rgba(15, 23, 42, 0.06)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 18, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#0f766e' }}>Workforce control</div>
                <h2 style={{ margin: '7px 0 0', fontSize: 26, color: '#0f172a' }}>Shifts & attendance</h2>
                <p style={{ margin: '7px 0 0', maxWidth: 620, color: '#64748b', fontSize: 13 }}>Coordinate today’s coverage, see who is currently working, and make schedule changes from one operational view.</p>
              </div>
              <label style={{ display: 'grid', gap: 6, minWidth: 180, color: '#475569', fontSize: 11, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Viewing date
                <input
                  type="date"
                  value={targetDate}
                  onChange={(e) => setTargetDate(e.target.value)}
                  style={{ padding: '10px 12px', border: '1px solid #5eead4', borderRadius: 8, background: '#fff', color: '#0f172a', fontSize: 14, fontWeight: 700 }}
                />
              </label>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10, marginTop: 18 }}>
              {[
                ['Scheduled shifts', todaySchedule.length, 'For selected date'],
                ['Clocked in', dayClockedInStaff.filter((record) => record.status === 'CLOCKED_IN').length, 'Working now'],
                ['On break', dayClockedInStaff.filter((record) => record.status === 'ON_BREAK').length, 'Temporarily away'],
                ['Clocked out', dayClockedInStaff.filter((record) => record.status === 'CLOCKED_OUT').length, 'Completed today']
              ].map(([label, value, note]) => (
                <div key={label} style={{ padding: '12px 14px', borderRadius: 8, background: '#f8fbff', border: '1px solid #dbeafe' }}>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
                  <strong style={{ display: 'block', marginTop: 5, fontSize: 24, color: '#0f172a' }}>{value}</strong>
                  <div style={{ marginTop: 2, fontSize: 11, color: '#64748b' }}>{note}</div>
                </div>
              ))}
            </div>
          </section>

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.45fr) minmax(320px, 0.8fr)', gap: 16, alignItems: 'start' }}>
          <div style={{ background: '#fff', padding: 20, borderRadius: 8, border: '1px solid #e2e8f0' }}>
            <div style={{ marginBottom: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 800, color: '#0f766e', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Coverage planning</div>
                <h2 style={{ margin: '5px 0 0', color: '#1e293b' }}>Shift roster</h2>
                <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Select a shift to assign coverage or update its details.</p>
              </div>
              <span style={{ padding: '6px 9px', borderRadius: 999, background: '#ecfeff', color: '#0f766e', fontSize: 11, fontWeight: 800 }}>{targetDate}</span>
            </div>

            {scheduleLoading ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>Loading shifts...</div>
            ) : todaySchedule.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>No shift coverage scheduled for this date</div>
            ) : (
              <div style={{ display: 'grid', gap: 15 }}>
                {todaySchedule.map((shift) => {
                  const isSelected = String(selectedShiftId) === String(shift._id);
                  const assignedIds = new Set((shift.assignedStaff || []).map((staff) => String(staff.staffID?._id || staff.staffID)));
                  const availableStaff = staffList.filter((staff) => !assignedIds.has(String(staff._id)));
                  const selectedAssignmentStaff = availableStaff.find((staff) => String(staff._id) === String(assignmentStaffId));
                  const allowedAssignmentDuties = getAllowedShiftDuties(selectedAssignmentStaff?.positionCode, selectedAssignmentStaff?.systemRole || selectedAssignmentStaff?.role);

                  return (
                  <div key={shift._id} style={{ padding: 15, border: isSelected ? '2px solid #0f766e' : '1px solid #cbd5e1', borderRadius: 8, background: isSelected ? '#f0fdfa' : '#f8fafc', boxShadow: isSelected ? '0 8px 20px rgba(15, 118, 110, 0.12)' : 'none' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 10, gap: 10, flexWrap: 'wrap' }}>
                      <div>
                        {String(editingShiftId) === String(shift._id) ? (
                          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
                            <input aria-label="Rename shift" value={editingShiftName} onChange={(event) => setEditingShiftName(event.target.value)} style={{ ...policyInputStyle, minWidth: 220, background: '#fff' }} />
                            <button type="button" onClick={() => handleRenameShift(shift.shiftID)} style={{ ...policyButtonStyle, minHeight: 34, padding: '7px 10px', fontSize: 11 }}>Save name</button>
                            <button type="button" onClick={() => { setEditingShiftId(''); setEditingShiftName(''); }} style={{ ...secondaryButtonStyle, minHeight: 34, padding: '7px 10px', fontSize: 11 }}>Cancel</button>
                          </div>
                        ) : <h3 style={{ margin: '0 0 5px 0', color: '#1e293b' }}>{shift.shiftName}</h3>}
                        <div style={{ fontSize: 14, color: '#64748b' }}>{shift.department && <><span style={{ fontWeight: 500 }}>{shift.department}</span> • </>}<span>{shift.shiftStart} - {shift.shiftEnd}</span></div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: 12, color: '#64748b' }}>{shift.assignedStaff?.length || 0} / {shift.maxStaffRequired} assigned</div>
                        {shift.status && (
                          <span style={{ display: 'inline-block', padding: '4px 8px', borderRadius: 4, fontSize: 12, fontWeight: 500, background: shift.status === 'PUBLISHED' ? '#dcfce7' : '#fef3c7', color: shift.status === 'PUBLISHED' ? '#166534' : '#92400e', marginTop: 5 }}>{shift.status}</span>
                        )}
                      </div>
                    </div>
                    {String(editingShiftId) !== String(shift._id) && <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
                      <button type="button" onClick={() => { setEditingShiftId(shift._id); setEditingShiftName(shift.shiftName || ''); }} style={{ ...secondaryButtonStyle, minHeight: 34, padding: '7px 10px', fontSize: 11 }}>Rename shift</button>
                      <button type="button" onClick={() => handleDeleteShift(shift)} style={{ ...removeRowButtonStyle, marginTop: 0, minHeight: 34, padding: '7px 10px', fontSize: 11 }}>Cancel shift</button>
                    </div>}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 12 }}>
                      <button type="button" onClick={() => { setSelectedShiftId(shift._id); setAssignmentStaffId(''); setAssignmentRole(''); }} style={{ ...secondaryButtonStyle, padding: '8px 12px', borderColor: isSelected ? '#0f766e' : '#9adbcf', background: isSelected ? '#ccfbf1' : '#fff' }}>
                        {isSelected ? 'Selected shift' : 'Select shift to assign staff'}
                      </button>
                      {isSelected && <span style={{ color: '#0f766e', fontSize: 12, fontWeight: 700 }}>Assignment controls are active for this schedule.</span>}
                    </div>
                    <div style={{ marginTop: 15 }}>
                      <h4 style={{ margin: '0 0 10px 0', fontSize: 13, color: '#64748b' }}>Assigned Staff:</h4>
                      {shift.assignedStaff && shift.assignedStaff.length > 0 ? (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                          {shift.assignedStaff.map((staff, idx) => (
                            <div key={idx} style={{ padding: '6px 10px', background: '#ecfdf5', border: '1px solid #a7f3d0', borderRadius: 4, fontSize: 13, color: '#065f46' }}>{staff.firstName || staff.name} {staff.lastName || staff.surname} ({String(staff.assignmentRole || staff.type || staff.systemRole || 'STAFF').replaceAll('_', ' ')})</div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ fontSize: 13, color: '#cbd5e1', fontStyle: 'italic' }}>No staff assigned yet</div>
                      )}
                    </div>
                    {isSelected && (
                      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(150px, 0.7fr) auto', gap: 10, alignItems: 'end', marginTop: 16, paddingTop: 14, borderTop: '1px solid #ccfbf1' }}>
                        <label style={{ display: 'grid', gap: 6, minWidth: 0, color: '#334155', fontSize: 12, fontWeight: 800 }}>
                          Add staff to this shift
                          <select value={assignmentStaffId} onChange={(event) => {
                            const nextStaffId = event.target.value;
                            const nextStaff = availableStaff.find((staff) => String(staff._id) === String(nextStaffId));
                            setAssignmentStaffId(nextStaffId);
                            setAssignmentRole(resolveShiftAssignmentDuty(nextStaff));
                          }} style={{ ...policyInputStyle, background: '#fff', minWidth: 0 }}>
                            <option value="">Select roster staff</option>
                            {availableStaff.map((staff) => (
                              <option key={staff._id} value={staff._id}>{staff.firstName || staff.name} {staff.lastName || staff.surname || ''} ({staff.positionTitle || staff.positionCode || staff.systemRole || staff.role || 'Team'})</option>
                            ))}
                          </select>
                        </label>
                        <label style={{ display: 'grid', gap: 6, color: '#334155', fontSize: 12, fontWeight: 800 }}>
                          Shift role
                          <select
                            value={assignmentRole}
                            onChange={(event) => setAssignmentRole(event.target.value)}
                            style={{ ...policyInputStyle, background: '#fff' }}
                          >
                            <option value="">{selectedAssignmentStaff ? 'Select shift duty' : 'Select staff first'}</option>
                            {allowedAssignmentDuties.map((duty) => <option key={duty} value={duty}>{SHIFT_DUTY_LABELS[duty] || duty.replaceAll('_', ' ')}</option>)}
                          </select>
                        </label>
                        <button type="button" onClick={() => handleAssignStaffToShift(shift.shiftID, assignmentStaffId, assignmentRole)} disabled={!assignmentStaffId || !assignmentRole || (shift.assignedStaff?.length || 0) >= shift.maxStaffRequired} style={{ ...policyButtonStyle, minWidth: 150 }}>
                          {shift.assignedStaff?.length >= shift.maxStaffRequired ? 'Shift fully staffed' : 'Assign staff'}
                        </button>
                      </div>
                    )}
                  </div>
                  );
                })}
              </div>
            )}
          </div>

          {renderStaffActivity()}
          </div>

          {isAdmin && (
            <div style={{ background: '#fff', padding: 20, borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <h2 style={{ marginTop: 0, color: '#1e293b', marginBottom: 20 }}>Schedule Shift</h2>
              {shiftMessages.success && <div style={{ padding: '12px', background: '#dcfce7', color: '#166534', borderRadius: '4px', marginBottom: '15px' }}>✓ {shiftMessages.success}</div>}
              {shiftMessages.error && <div style={{ padding: '12px', background: '#fee2e2', color: '#991b1b', borderRadius: '4px', marginBottom: '15px' }}>✕ {shiftMessages.error}</div>}
              <form onSubmit={handleCreateShift} style={{ maxWidth: 600 }}>
                <div style={{ marginBottom: '15px' }}><label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>Shift Name *</label><input type="text" placeholder="e.g., Morning Hospital Shift" value={shiftForm.shiftName} onChange={(e) => setShiftForm({ ...shiftForm, shiftName: e.target.value })} style={{ width: '100%', padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: '4px', fontSize: '14px', boxSizing: 'border-box' }} required /></div>
                <div style={{ marginBottom: '15px' }}><label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>Shift Date *</label><input type="date" value={shiftForm.shiftDate} onChange={(e) => setShiftForm({ ...shiftForm, shiftDate: e.target.value })} style={{ width: '100%', padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: '4px', fontSize: '14px', boxSizing: 'border-box' }} /></div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                  <div><label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>Start Time</label><input type="time" value={shiftForm.shiftStart} onChange={(e) => setShiftForm({ ...shiftForm, shiftStart: e.target.value })} style={{ width: '100%', padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: '4px', fontSize: '14px', boxSizing: 'border-box' }} /></div>
                  <div><label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>End Time</label><input type="time" value={shiftForm.shiftEnd} onChange={(e) => setShiftForm({ ...shiftForm, shiftEnd: e.target.value })} style={{ width: '100%', padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: '4px', fontSize: '14px', boxSizing: 'border-box' }} /></div>
                </div>
                <div style={{ marginBottom: '15px' }}><label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '5px', color: '#475569' }}>Max Staff Required</label><input type="number" min="1" value={shiftForm.maxStaffRequired} onChange={(e) => setShiftForm({ ...shiftForm, maxStaffRequired: Number(e.target.value) || 1 })} style={{ width: '100%', padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: '4px', fontSize: '14px', boxSizing: 'border-box' }} /></div>
                <div style={{ display: 'grid', gap: 10, padding: 14, marginBottom: 15, border: '1px solid #dbeafe', borderRadius: 12, background: '#f8fbff' }}>
                  <div>
                    <strong style={{ color: '#0f172a' }}>Assign staff to this scheduled shift</strong>
                    <div style={{ marginTop: 4, color: '#64748b', fontSize: 12 }}>Select the roster members who should be assigned now. Start Hospital Day remains the approval gate before clock-in.</div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 8, maxHeight: 240, overflowY: 'auto' }}>
                    {(filteredStaff.length ? filteredStaff : staffList).map((staff) => {
                      const staffId = String(staff._id);
                      const selected = selectedStaffForShift.includes(staffId);
                      const allowedDuties = getAllowedShiftDuties(staff.positionCode, staff.systemRole || staff.role);
                      const assignedRole = resolveShiftAssignmentDuty(staff, selectedStaffRoles[staffId]?.role || '');
                      const positionLabel = staff.positionTitle || staff.positionCode || staff.systemRole || staff.role || 'Team member';
                      return (
                        <label key={staffId} style={{ display: 'grid', gap: 6, padding: '9px 10px', border: selected ? '1px solid #0f766e' : '1px solid #dbeafe', borderRadius: 10, background: selected ? '#ccfbf1' : '#fff', cursor: 'pointer' }}>
                          <span style={{ display: 'flex', alignItems: 'flex-start', gap: 8, minWidth: 0, fontSize: 12, lineHeight: 1.35, color: '#334155' }}>
                            <input type="checkbox" checked={selected} onChange={() => setSelectedStaffForShift((current) => { const next = selected ? current.filter((id) => id !== staffId) : [...current, staffId]; if (!selected) setSelectedStaffRoles((roles) => ({ ...roles, [staffId]: { role: assignedRole, context: 'CLINICAL_NURSE_ER' } })); return next; })} style={{ marginTop: 2 }} />
                            <span style={{ minWidth: 0 }}><strong style={{ display: 'block' }}>{`${staff.firstName || staff.name || ''} ${staff.lastName || staff.surname || ''}`.trim()}</strong><span>{positionLabel} · {staff.systemRole || staff.role || 'Role unavailable'}{staff.email ? ` • ${staff.email}` : ''}</span></span>
                          </span>
                          {selected && (
                            <select value={assignedRole} onChange={(event) => setSelectedStaffRoles((roles) => ({ ...roles, [staffId]: { role: event.target.value, context: selectedStaffRoles[staffId]?.context || 'CLINICAL_NURSE_ER' } }))} style={{ width: '100%', padding: '5px 7px', border: '1px solid #94a3b8', borderRadius: 5, fontSize: 11 }}>
                              {!assignedRole && <option value="">{allowedDuties.length ? 'Select shift duty' : 'No configured shift duty'}</option>}
                              {allowedDuties.map((duty) => <option key={duty} value={duty}>{SHIFT_DUTY_LABELS[duty] || duty.replaceAll('_', ' ')}</option>)}
                            </select>
                          )}
                        </label>
                      );
                    })}
                  </div>
                  <div style={{ color: '#0f766e', fontSize: 12, fontWeight: 800 }}>{selectedStaffForShift.length} staff selected for this scheduled shift</div>
                </div>
                <button type="submit" style={{ ...policyButtonStyle, marginTop: '10px' }}>Schedule Shift</button>
              </form>
            </div>
          )}
        </div>
      );
    }

    if (workspaceView === 'workload') {
      return (
        <div style={{ display: 'grid', gap: 16 }}>
          <section style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'flex-start', marginBottom: 18 }}>
              <div>
                <h2 style={{ margin: 0, color: '#0f172a' }}>Workload policy</h2>
                <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Define the hospital baseline used by clinician workload and queue congestion calculations. Every save is audited.</p>
              </div>
              <span style={{ padding: '5px 9px', borderRadius: 999, background: '#dcfce7', color: '#166534', fontSize: 11, fontWeight: 800 }}>ADMIN ONLY</span>
            </div>
            {capacityPolicyLoading ? <p style={{ color: '#64748b' }}>Loading policy...</p> : (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(180px, 1fr) minmax(180px, 1fr) auto', gap: 12, alignItems: 'end' }}>
                  <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                    Role
                    <select value={capacityRole} onChange={(event) => { setCapacityRole(event.target.value); setCapacityValue(capacityConfig?.policies?.defaultCapacityByRole?.[event.target.value] ?? ''); }} style={policyInputStyle}>
                      <option value="TRIAGE_NURSE">Triage nurse</option>
                      <option value="CLINICAL_NURSE">Clinical nurse</option>
                      <option value="PHYSICIAN">Physician</option>
                    </select>
                  </label>
                  <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                    Capacity in workload units
                    <input type="number" min="1" step="1" value={capacityValue} onChange={(event) => setCapacityValue(event.target.value)} placeholder="No baseline" style={policyInputStyle} />
                  </label>
                  <button type="button" onClick={saveHospitalCapacity} disabled={capacityPolicySaving} style={policyButtonStyle}><FiSave size={15} /> {capacityPolicySaving ? 'Saving...' : 'Save baseline'}</button>
                </div>
                <p style={{ margin: '10px 0 0', color: '#64748b', fontSize: 12 }}>This is the lowest-precedence hospital-wide role default. Select a role to view its current saved baseline, then update it. Current saved value: <strong>{savedCapacityValue ?? 'not set'}</strong>.</p>
              </>
            )}
          </section>

          <section style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
            <div style={{ marginBottom: 14 }}>
              <h3 style={{ margin: 0, color: '#0f172a' }}>Acuity workload weights</h3>
              <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Multipliers convert active cases into workload units. These values affect workload and congestion alerts, not admission decisions. The required ordering is RED ≥ ORANGE ≥ YELLOW ≥ GREEN.</p>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(100px, 1fr))', gap: 10 }}>
              {Object.keys(acuityWeights).map((priority) => (
                <label key={priority} style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                  {priority}
                  <input type="number" min="0.1" step="0.1" value={acuityWeights[priority]} onChange={(event) => setAcuityWeights((current) => ({ ...current, [priority]: event.target.value }))} style={policyInputStyle} />
                </label>
              ))}
            </div>
            {acuityValidationMessage && <div style={{ marginTop: 12, padding: '8px 10px', borderRadius: 8, background: '#fef2f2', color: '#991b1b', fontSize: 12, fontWeight: 600 }}>{acuityValidationMessage}</div>}
            <button type="button" onClick={saveHospitalAcuityWeights} disabled={acuitySaving} style={{ ...policyButtonStyle, marginTop: 14 }}><FiSave size={15} /> {acuitySaving ? 'Saving...' : 'Save acuity weights'}</button>
          </section>
        </div>
      );
    }

    return null;
  };

  const renderMainAdminWorkspaceContent = () => {
    return renderWorkspaceView();
  };

  // ============ FILTER STAFF FOR ASSIGNMENT ============
  const getFilteredStaffForAssignment = () => {
    return filteredStaff.filter(staff => {
      // Search by name, surname, or email
      const matchesSearch = !staffSearchQuery ||
        (staff.firstName || staff.name)?.toLowerCase().includes(staffSearchQuery.toLowerCase()) ||
        (staff.lastName || staff.surname)?.toLowerCase().includes(staffSearchQuery.toLowerCase()) ||
        staff.email?.toLowerCase().includes(staffSearchQuery.toLowerCase());

      // Filter by department
      const matchesDept = staffDepartmentFilter === 'all' || (staff.assignment?.department || staff.department) === staffDepartmentFilter;

      // Filter by role
      const matchesRole = staffRoleFilter === 'all' || (staff.systemRole || staff.role) === staffRoleFilter;

      return matchesSearch && matchesDept && matchesRole;
    });
  };

  const selectedHospitalName = managedHospitals.find((hospital) => String(hospital._id) === String(decoded?.hospitalId))?.name || decoded?.hospitalName || managedHospitals[0]?.name || decoded?.hospitalId || 'No hospital selected';
  const hasLiveH7FhirIntegration = ehrConfig?.adapters?.some((adapter) => {
    const name = String(adapter?.name || '').toUpperCase();
    const status = String(adapter?.status || '').toUpperCase();
    const connection = String(adapter?.connection || '').toUpperCase();
    return (name.includes('HL7') || name.includes('FHIR')) && (status === 'ENABLED' || status === 'HEALTHY' || connection === 'HEALTHY');
  }) || false;
  const hasSyntheticSeedActivity = (hospitalActivity?.activity || []).some((event) => {
    const text = `${event?.action || ''} ${event?.description || ''}`.toLowerCase();
    return /synthetic|seed|mock simulation/i.test(text);
  });
  const isLiveIntegrationActive = systemMode === 'live' || (hasLiveH7FhirIntegration && !hasSyntheticSeedActivity);
  const isMockSimulationActive = systemMode === 'mock' || !isLiveIntegrationActive;
  const adminWorkspaceItems = [
    { label: 'Overview', target: 'overview' },
    { label: 'Hospital', target: 'hospital' },
    { label: 'Beds', target: 'beds' },
    { label: 'Staff', target: 'staff' },
    { label: 'Patients', target: 'patients' },
    { label: 'Shifts', target: 'shifts' },
    { label: 'Workload', target: 'workload' },
    { label: 'EHR Configurations', target: 'ehr-configurations' },
    { label: 'Labs Configurations', target: 'labs-configurations' },
    { label: 'Pharmacy Configurations', target: 'pharmacy-configurations' },
    { label: 'Analytics', target: 'analytics' },
    { label: 'IT Support', target: 'it-support' }
  ];

  // ============ RENDER ============
  return (
    <SidebarLayout username={decoded?.name || 'Staff'} handleLogout={onLogout}>
      <div className="admin-workspace-shell" style={{ padding: 'clamp(10px, 2vw, 20px)', maxWidth: '1400px', margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
        <style>{`
          .admin-workspace-shell,
          .admin-workspace-shell * { box-sizing: border-box; }
          .admin-workspace-header { min-width: 0; }
          .admin-workspace-context { min-width: 0 !important; max-width: 100%; flex: 1 1 260px; }
          .admin-workspace-tabs { min-width: 0; }
          .admin-workspace-tabs > div { min-width: 0; }
          @media (max-width: 600px) {
            .admin-workspace-header { padding: 16px !important; border-radius: 10px !important; }
            .admin-workspace-header h1 { font-size: clamp(22px, 8vw, 28px) !important; overflow-wrap: anywhere; }
            .admin-workspace-context { flex-basis: 100%; width: 100%; }
            .admin-workspace-context select { max-width: 100%; }
            .admin-workspace-tabs { padding: 10px !important; margin-bottom: 14px !important; overflow: hidden; }
            .admin-workspace-tabs > div { flex-wrap: nowrap !important; overflow-x: auto; padding-bottom: 4px; scrollbar-width: thin; }
            .admin-workspace-tabs button { flex: 0 0 auto; white-space: nowrap; }
            #hospital-operations-overview { gap: 12px !important; }
            #hospital-operations-overview > div { min-width: 0 !important; }
            #hospital-operations-overview > div:first-child,
            #hospital-operations-overview > div:nth-child(2) { grid-template-columns: minmax(0, 1fr) !important; }
            .overview-diagnosis-chart,
            .overview-diagnosis-chart svg { width: 130px !important; height: 130px !important; }
            .overview-diagnosis-chart { flex-basis: 130px !important; }
          }
        `}</style>
        {/* Header */}
        <div className="admin-workspace-header" style={{ marginBottom: 24, padding: '24px 26px', borderRadius: 14, background: 'linear-gradient(120deg, #0f172a 0%, #164e63 58%, #0e7490 100%)', color: '#f8fafc', boxShadow: '0 12px 28px rgba(15,23,42,0.16)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#a5f3fc', marginBottom: 8 }}>{isAdmin ? 'Hospital administration' : 'Clinical operations'}</div>
              <h1 style={{ margin: 0, fontSize: 28, letterSpacing: '-0.02em' }}>{isAdmin ? 'Hospital command center' : 'Staff scheduling'}</h1>
              <p style={{ margin: '8px 0 0', maxWidth: 680, color: '#cbd5e1', fontSize: 14, lineHeight: 1.55 }}>{isAdmin ? 'Establish the operating model for this hospital: populate its people and records, compose shifts, and make workload policy explicit.' : 'Review staff coverage, assignments, and the live schedule for your clinical area.'}</p>
            </div>
            {isAdmin && (
              <div className="admin-workspace-context" style={{ minWidth: 260, padding: '12px 14px', border: '1px solid rgba(165,243,252,0.3)', borderRadius: 10, background: 'rgba(15,23,42,0.25)' }}>
                <div style={{ color: '#a5f3fc', fontSize: 11, fontWeight: 800, textTransform: 'uppercase' }}>Active workspace</div>
                <strong style={{ display: 'block', marginTop: 5, fontSize: 16 }}>{selectedHospitalName}</strong>
                <select
                  value={decoded?.hospitalId || managedHospitals[0]?._id || ''}
                  onChange={(event) => selectManagedHospital(event.target.value)}
                  disabled={managedHospitalLoading}
                  style={{
                    width: '100%',
                    marginTop: 8,
                    padding: '8px 10px',
                    borderRadius: 8,
                    border: '1px solid rgba(165,243,252,0.28)',
                    background: 'rgba(15,23,42,0.35)',
                    color: '#f8fafc',
                    fontSize: 12
                  }}
                >
                  {managedHospitals.map((hospital) => (
                    <option key={String(hospital._id)} value={String(hospital._id)} style={{ color: '#0f172a' }}>{hospital.name}</option>
                  ))}
                </select>

                {isCompanySuperAdmin && (
                  <div style={{ marginTop: 12, display: 'grid', gap: 8 }}>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      <button
                        type="button"
                        onClick={() => setSystemMode('live')}
                        style={{
                          flex: 1,
                          minWidth: 110,
                          padding: '8px 10px',
                          borderRadius: 10,
                          border: isLiveSystemMode ? '1px solid #86efac' : '1px solid rgba(165,243,252,0.28)',
                          background: isLiveSystemMode ? '#dcfce7' : 'rgba(15,23,42,0.25)',
                          color: isLiveSystemMode ? '#166534' : '#e2e8f0',
                          fontWeight: 800,
                          cursor: 'pointer'
                        }}
                      >
                        Live company
                      </button>
                      <button
                        type="button"
                        onClick={() => setSystemMode('mock')}
                        style={{
                          flex: 1,
                          minWidth: 110,
                          padding: '8px 10px',
                          borderRadius: 10,
                          border: isMockSystemMode ? '1px solid #fbbf24' : '1px solid rgba(165,243,252,0.28)',
                          background: isMockSystemMode ? '#fef3c7' : 'rgba(15,23,42,0.25)',
                          color: isMockSystemMode ? '#92400e' : '#e2e8f0',
                          fontWeight: 800,
                          cursor: 'pointer'
                        }}
                      >
                        Mock sandbox
                      </button>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, padding: '10px 8px', borderRadius: 12, border: `1px solid ${isMockSimulationActive ? 'rgba(245,158,11,0.45)' : 'rgba(34,197,94,0.45)'}`, background: isMockSimulationActive ? 'rgba(245,158,11,0.12)' : 'rgba(34,197,94,0.12)' }}>
                      <span
                        style={{
                          width: 18,
                          height: 18,
                          borderRadius: '50%',
                          display: 'inline-block',
                          background: isMockSimulationActive ? '#f59e0b' : '#22c55e',
                          boxShadow: isMockSimulationActive ? '0 0 0 4px rgba(245,158,11,0.2)' : '0 0 0 4px rgba(34,197,94,0.17)',
                          flexShrink: 0
                        }}
                      />
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: isMockSimulationActive ? '#fcd34d' : '#bbf7d0' }}>
                          {isMockSimulationActive ? 'Mock Simulation' : 'Live Active HL7/FHIR Integration'}
                        </div>
                        <div style={{ fontSize: 11, color: '#f8fafc', fontWeight: 700, marginTop: 3 }}>
                          {isMockSimulationActive ? 'Synthetic hospital data is active' : 'Real-time EHR data is active'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Messages */}
        {messages.success && (
          <div style={{ padding: '12px', background: '#dcfce7', color: '#166534', borderRadius: '4px', marginBottom: '15px' }}>
            ✓ {messages.success}
          </div>
        )}
        {messages.error && (
          <div style={{ padding: '12px', background: '#fee2e2', color: '#991b1b', borderRadius: '4px', marginBottom: '15px' }}>
            ✕ {messages.error}
          </div>
        )}

        {isAdmin && (
          <div className="admin-workspace-tabs" style={{ marginBottom: 20, padding: 12, borderRadius: 14, border: '1px solid #dbeafe', background: '#fff', boxShadow: '0 8px 18px rgba(15, 23, 42, 0.04)' }}>
            <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b', marginBottom: 12 }}>Workspace</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {adminWorkspaceItems.map((item) => {
                const isSelected = workspaceView === item.target ||
                  (item.target === 'workload' && (activeTab === 'workload-policy' || workloadPolicyOpen));

                return (
                  <button
                    key={item.label}
                    type="button"
                    onClick={() => {
                      console.log('[NAV] Workspace tab selected', item.target);
                      setWorkspaceView(item.target);
                      if (item.target === 'staff') {
                        setActiveTab('hospital-operations');
                        setSearchQuery('');
                      } else if (item.target === 'shifts') setActiveTab('shifts');
                      else if (item.target === 'workload') { setWorkspaceView('hospital'); setActiveTab('hospital-operations'); setWorkloadPolicyOpen(true); fetchCapacityPolicy(); }
                      else if (item.target === 'ehr-configurations') { setActiveTab('hospital-operations'); }
                      else setActiveTab('hospital-operations');
                    }}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 10,
                      border: isSelected ? '1px solid #0e7490' : '1px solid #e2e8f0',
                      background: isSelected ? '#ecfeff' : '#f8fafc',
                      color: isSelected ? '#0f172a' : '#475569',
                      fontWeight: isSelected ? 700 : 600,
                      cursor: 'pointer'
                    }}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === 'workload-policy' && isAdmin && (
          <div style={{ display: 'grid', gap: 16 }}>
            <section style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'flex-start', marginBottom: 18 }}>
                <div>
                  <h2 style={{ margin: 0, color: '#0f172a' }}>Workload policy</h2>
                  <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Define the hospital baseline used by clinician workload and queue congestion calculations. Every save is audited.</p>
                </div>
                <span style={{ padding: '5px 9px', borderRadius: 999, background: '#dcfce7', color: '#166534', fontSize: 11, fontWeight: 800 }}>ADMIN ONLY</span>
              </div>
              {capacityPolicyLoading ? <p style={{ color: '#64748b' }}>Loading policy...</p> : (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'minmax(180px, 1fr) minmax(180px, 1fr) auto', gap: 12, alignItems: 'end' }}>
                    <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                      Role
                      <select value={capacityRole} onChange={(event) => { const nextRole = event.target.value; setCapacityRole(nextRole); setSavedCapacityValue(capacityConfig?.policies?.defaultCapacityByRole?.[nextRole] ?? null); setCapacityValue(''); }} style={policyInputStyle}>
                        <option value="TRIAGE_NURSE">Triage nurse</option>
                        <option value="CLINICAL_NURSE">Clinical nurse</option>
                        <option value="PHYSICIAN">Physician</option>
                      </select>
                    </label>
                    <label style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                      Capacity in workload units (1 unit = 1 acuity-weighted workload point)
                      <input type="number" min="1" step="1" value={capacityValue} onChange={(event) => setCapacityValue(event.target.value)} placeholder="Enter a capacity value" style={policyInputStyle} />
                    </label>
                    <button type="button" onClick={saveHospitalCapacity} disabled={capacityPolicySaving} style={policyButtonStyle}><FiSave size={15} /> {capacityPolicySaving ? 'Saving...' : 'Save baseline'}</button>
                  </div>
                  <p style={{ margin: '10px 0 0', color: '#64748b', fontSize: 12 }}>This becomes the fallback capacity for every clinician in the selected role unless a department or clinician override exists.</p>
                </>
              )}
            </section>

            <section style={{ background: '#fff', padding: 22, borderRadius: 8, border: '1px solid #dbeafe' }}>
              <div style={{ marginBottom: 14 }}>
                <h3 style={{ margin: 0, color: '#0f172a' }}>Acuity workload weights</h3>
                <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Multipliers convert active cases into workload units. These values affect workload and congestion alerts, not admission decisions.</p>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(100px, 1fr))', gap: 10 }}>
                {Object.keys(acuityWeights).map((priority) => (
                  <label key={priority} style={{ display: 'grid', gap: 5, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                    {priority}
                    <input type="number" min="0.1" step="0.1" value={acuityWeights[priority]} onChange={(event) => setAcuityWeights((current) => ({ ...current, [priority]: event.target.value }))} style={policyInputStyle} />
                  </label>
                ))}
              </div>
              <button type="button" onClick={saveHospitalAcuityWeights} disabled={acuitySaving} style={{ ...policyButtonStyle, marginTop: 14 }}><FiSave size={15} /> {acuitySaving ? 'Saving...' : 'Save acuity weights'}</button>
            </section>
          </div>
        )}

        {isAdmin && (
          <div style={{ display: 'grid', gap: 16 }}>
            <section style={{ background: '#fff', borderRadius: 18, border: '1px solid #dbeafe', overflow: 'hidden', boxShadow: '0 12px 26px rgba(15, 23, 42, 0.08)' }}>
              <div style={{ padding: 20 }}>
                <div style={{ display: 'grid', gap: 16 }}>
                  {renderMainAdminWorkspaceContent()}
                </div>
              </div>
            </section>

            {hospitalDayWizardOpen && (
              <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20, zIndex: 65 }}>
                <div style={{ width: 'min(900px, 100%)', maxHeight: '88vh', overflowY: 'auto', background: '#fff', borderRadius: 18, boxShadow: '0 26px 64px rgba(15, 23, 42, 0.22)', border: '1px solid #e2e8f0' }}>
                  <div id="hospital-day-start-panel" style={{ padding: '20px 22px 14px', borderBottom: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#0f766e' }}>Hospital day startup</div>
                      <h3 style={{ margin: '8px 0 0', fontSize: 24, color: '#0f172a' }}>Start Hospital Shift</h3>
                      <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Activate the hospital operating model, assign the coverage team, and bring specialist services online together.</p>
                    </div>
                    <button type="button" onClick={closeHospitalDayWizard} style={{ ...removeRowButtonStyle, background: '#f8fafc', borderColor: '#cbd5e1', color: '#475569' }}><FiX size={14} /> Close</button>
                  </div>

                  <div style={{ padding: 20 }}>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 18 }}>
                      {['Readiness', 'Workload policy', 'Shift & coverage', 'Synthetic data', 'Review & start'].map((stepLabel, index) => (
                        <div key={stepLabel} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', borderRadius: 999, border: index === hospitalDayStep ? '1px solid #0f766e' : '1px solid #dbeafe', background: index === hospitalDayStep ? '#ecfeff' : '#f8fafc', color: index === hospitalDayStep ? '#0f172a' : '#475569', fontWeight: 700, fontSize: 12 }}>
                          <span style={{ width: 20, height: 20, borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', background: index === hospitalDayStep ? '#0f766e' : '#dbeafe', color: index === hospitalDayStep ? '#fff' : '#0f172a' }}>{index + 1}</span>
                          {stepLabel}
                        </div>
                      ))}
                    </div>

                    {hospitalDayStep === 0 && (
                      <div style={{ display: 'grid', gap: 12 }}>
                        {(() => {
                          const selectedShift = todaySchedule.find((shift) => String(shift._id) === String(daySelectedShiftId));
                          const assignedIds = new Set((selectedShift?.assignedStaff || []).map((staff) => String(staff.staffID?._id || staff.staffID)));
                          const clockedIds = new Set((dayClockedInStaff || []).map((record) => String(record.staff?._id || record.staffId || record._id)));
                          const assignedRoster = (filteredStaff.length ? filteredStaff : staffList).filter((person) => assignedIds.has(String(person._id)));
                          const onCallRoster = (filteredStaff.length ? filteredStaff : staffList).filter((person) => person.assignment?.isOnCall || person.isOnCall);
                          return (
                            <div style={{ display: 'grid', gap: 10, padding: 14, border: '1px solid #99f6e4', borderRadius: 14, background: '#f0fdfa' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                                <div>
                                  <div style={{ fontSize: 13, fontWeight: 800, color: '#134e4a' }}>Scheduled shift for this hospital day</div>
                                  <div style={{ marginTop: 4, fontSize: 12, color: '#475569' }}>{selectedShift ? `${selectedShift.shiftName}${selectedShift.department ? ` • ${selectedShift.department}` : ''} • ${selectedShift.shiftStart}-${selectedShift.shiftEnd}` : 'No scheduled shift selected'}</div>
                                </div>
                                <span style={{ padding: '4px 8px', borderRadius: 999, background: selectedShift && isShiftCurrent(selectedShift) ? '#dcfce7' : '#fef3c7', color: selectedShift && isShiftCurrent(selectedShift) ? '#166534' : '#92400e', fontSize: 11, fontWeight: 800 }}>{selectedShift && isShiftCurrent(selectedShift) ? 'CURRENTLY ACTIVE' : 'SCHEDULED'}</span>
                              </div>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8, fontSize: 12 }}>
                                <div><strong>Assigned:</strong> {assignedRoster.length || selectedShift?.assignedStaff?.length || 0}</div>
                                <div><strong>On call:</strong> {onCallRoster.length}</div>
                                <div><strong>Clocked in:</strong> {assignedRoster.filter((person) => clockedIds.has(String(person._id))).length}</div>
                              </div>
                              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                                {assignedRoster.map((person) => <span key={person._id} style={{ padding: '5px 8px', borderRadius: 999, background: clockedIds.has(String(person._id)) ? '#dcfce7' : '#fff', border: '1px solid #99f6e4', color: '#134e4a', fontSize: 11 }}>{person.firstName || person.name} {person.lastName || person.surname || ''} {clockedIds.has(String(person._id)) ? '• CLOCKED IN' : '• SCHEDULED'}</span>)}
                                {onCallRoster.slice(0, 8).map((person) => <span key={`on-call-${person._id}`} style={{ padding: '5px 8px', borderRadius: 999, background: '#fff7ed', border: '1px solid #fed7aa', color: '#9a3412', fontSize: 11 }}>{person.firstName || person.name} {person.lastName || person.surname || ''} • ON CALL</span>)}
                              </div>
                              <button type="button" onClick={fetchCurrentStaffStatus} style={{ ...secondaryButtonStyle, justifySelf: 'start', minHeight: 34, padding: '7px 10px', fontSize: 11 }}>Refresh clock-in status</button>
                            </div>
                          );
                        })()}
                        <div style={{ padding: 14, borderRadius: 12, border: '1px solid #dbeafe', background: '#f8fbff' }}>
                          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Live hospital readiness</div>
                          <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: '#0f172a' }}>
                            {Number(hospitalOperationsStatus?.staff?.total || 0)} staff • {Number(hospitalOperationsStatus?.beds?.total || 0)} beds • {Number(hospitalOperationsStatus?.shifts?.today || 0)} shifts
                          </div>
                          <div style={{ marginTop: 6, color: '#475569', fontSize: 12 }}>Readiness is refreshed from the current hospital configuration, shift roster, and workforce state.</div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                          {[
                            {
                              key: 'beds',
                              label: 'Beds',
                              ready: Number(hospitalOperationsStatus?.beds?.total || 0) > 0,
                              detail: `${Number(hospitalOperationsStatus?.beds?.total || 0)} configured`
                            },
                            {
                              key: 'staff',
                              label: 'Staff clock-in',
                              ready: Number(hospitalOperationsStatus?.staff?.total || 0) > 0,
                              detail: `${Number(hospitalOperationsStatus?.staff?.total || 0)} on roster`
                            },
                            {
                              key: 'shifts',
                              label: 'Shift coverage',
                              ready: Number(hospitalOperationsStatus?.shifts?.today || 0) > 0 || todaySchedule.length > 0,
                              detail: `${Number(hospitalOperationsStatus?.shifts?.today || 0)} scheduled today`
                            },
                            {
                              key: 'workload',
                              label: 'Workload policy',
                              ready: Boolean(capacityConfig?.policies?.defaultCapacityByRole),
                              detail: capacityConfig?.policies?.defaultCapacityByRole ? 'Configured' : 'Needs setup'
                            },
                            {
                              key: 'syntheticPatients',
                              label: 'Synthetic patient load',
                              ready: Number(hospitalOperationsStatus?.encounters?.total || hospitalActivity?.summary?.encounters || 0) > 0,
                              detail: `${Number(hospitalOperationsStatus?.encounters?.total || hospitalActivity?.summary?.encounters || 0)} active cases`
                            }
                          ].map((item) => (
                            <button
                              key={item.key}
                              type="button"
                              onClick={() => setDaySystemReadiness((current) => ({ ...current, [item.key]: !current[item.key] }))}
                              style={{
                                display: 'grid',
                                gap: 8,
                                width: '100%',
                                minWidth: 0,
                                padding: 14,
                                borderRadius: 12,
                                border: item.ready ? '1px solid #86efac' : '1px solid #fbcfe8',
                                background: item.ready ? '#ecfdf5' : '#fff7ed',
                                color: '#0f172a',
                                textAlign: 'left',
                                cursor: 'pointer',
                                fontWeight: 700,
                                fontSize: 13,
                                lineHeight: 1.3,
                                overflowWrap: 'anywhere',
                                wordBreak: 'break-word'
                              }}
                            >
                              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', alignItems: 'center', gap: 8, minWidth: 0 }}>
                                <span style={{ minWidth: 0, overflowWrap: 'anywhere', wordBreak: 'break-word' }}>{item.label}</span>
                                <span style={{ flexShrink: 0, padding: '4px 8px', borderRadius: 999, background: item.ready ? '#dcfce7' : '#fcd34d', color: item.ready ? '#166534' : '#92400e', fontSize: 11, whiteSpace: 'nowrap' }}>{item.ready ? 'Ready' : 'Standby'}</span>
                              </div>
                              <div style={{ marginTop: 0, color: '#475569', fontSize: 11, fontWeight: 600, lineHeight: 1.4, minWidth: 0, overflowWrap: 'anywhere', wordBreak: 'break-word' }}>{item.detail}</div>
                            </button>
                          ))}
                        </div>

                        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 4 }}>
                          <button type="button" onClick={() => { setHospitalDayWizardOpen(false); setWorkspaceView('shifts'); }} style={{ ...secondaryButtonStyle, whiteSpace: 'normal', textAlign: 'center', minWidth: 0 }}>Open live shift roster</button>
                          <button type="button" onClick={() => { setHospitalDayWizardOpen(false); setWorkspaceView('workload'); setWorkloadPolicyOpen(true); fetchCapacityPolicy(); }} style={{ ...secondaryButtonStyle, whiteSpace: 'normal', textAlign: 'center', minWidth: 0 }}>Set workload policy</button>
                        </div>
                      </div>
                    )}

                    {hospitalDayStep === 1 && (
                      <div style={{ display: 'grid', gap: 14 }}>
                        <div style={{ padding: 14, borderRadius: 14, border: '1px solid #dbeafe', background: '#f8fbff' }}>
                          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Day workload policy</div>
                          <div style={{ marginTop: 6, color: '#475569', fontSize: 13 }}>Set the hospital-wide role baseline used for this operating day. The saved value is shown before you make a change.</div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, alignItems: 'end' }}>
                          <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                            Role
                            <select value={capacityRole} onChange={(event) => { const nextRole = event.target.value; setCapacityRole(nextRole); setSavedCapacityValue(capacityConfig?.policies?.defaultCapacityByRole?.[nextRole] ?? null); setCapacityValue(''); }} style={{ ...policyInputStyle, background: '#fff' }}>
                              <option value="TRIAGE_NURSE">Triage nurse</option>
                              <option value="CLINICAL_NURSE">Clinical nurse</option>
                              <option value="PHYSICIAN">Physician</option>
                            </select>
                          </label>
                          <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                            Capacity workload units
                            <input type="number" min="1" step="1" value={capacityValue} onChange={(event) => setCapacityValue(event.target.value)} placeholder={savedCapacityValue == null ? 'Enter capacity' : `Saved: ${savedCapacityValue}`} style={policyInputStyle} />
                          </label>
                          <button type="button" onClick={saveHospitalCapacity} disabled={capacityPolicySaving || capacityValue === ''} style={policyButtonStyle}><FiSave size={15} /> {capacityPolicySaving ? 'Saving...' : 'Save workload policy'}</button>
                        </div>
                        <div style={{ padding: '10px 12px', borderRadius: 10, background: '#ecfdf5', color: '#166534', fontSize: 12, fontWeight: 700 }}>Current saved baseline: {savedCapacityValue ?? 'not set'} workload units for {capacityRole}.</div>
                      </div>
                    )}

                    {hospitalDayStep === 2 && (
                      <div style={{ display: 'grid', gap: 14 }}>
                        <div style={{ display: 'grid', gap: 12, padding: 14, border: '1px solid #dbeafe', borderRadius: 14, background: '#f8fbff' }}>
                          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            <button type="button" onClick={() => setDayShiftMode('existing')} style={{ ...secondaryButtonStyle, background: dayShiftMode === 'existing' ? '#ccfbf1' : '#fff', borderColor: dayShiftMode === 'existing' ? '#0f766e' : '#9adbcf' }}>Use scheduled shift</button>
                            <button type="button" onClick={() => { setDayShiftMode('new'); setDaySelectedShiftId(''); }} style={{ ...secondaryButtonStyle, background: dayShiftMode === 'new' ? '#ccfbf1' : '#fff', borderColor: dayShiftMode === 'new' ? '#0f766e' : '#9adbcf' }}>Create new shift</button>
                          </div>
                          {dayShiftMode === 'existing' ? (
                            <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 800, color: '#334155' }}>
                              Select scheduled shift
                              <select value={daySelectedShiftId} onChange={(event) => { const next = todaySchedule.find((shift) => String(shift._id) === String(event.target.value)); setDaySelectedShiftId(event.target.value); if (next) { setDayShiftName(next.shiftName || 'Hospital Day Coverage'); setDayShiftStart(next.shiftStart || '08:00'); setDayShiftEnd(next.shiftEnd || '16:00'); } }} style={{ ...policyInputStyle, background: '#fff' }}>
                                <option value="">Choose a scheduled shift</option>
                                {todaySchedule.map((shift) => <option key={shift._id} value={shift._id}>{shift.shiftName} • {shift.shiftStart}-{shift.shiftEnd} • {shift.assignedStaff?.length || 0}/{shift.maxStaffRequired} assigned</option>)}
                              </select>
                            </label>
                          ) : (
                            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(180px, 1.4fr) minmax(150px, 1fr) repeat(2, minmax(120px, 0.7fr))', gap: 10, alignItems: 'end' }}>
                              <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 800, color: '#334155' }}>Shift name<input value={dayShiftName} onChange={(event) => setDayShiftName(event.target.value)} placeholder="e.g. Morning Hospital Shift" style={policyInputStyle} /></label>
                              <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 800, color: '#334155' }}>Starts<input type="time" value={dayShiftStart} onChange={(event) => setDayShiftStart(event.target.value)} style={policyInputStyle} /></label>
                              <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 800, color: '#334155' }}>Ends<input type="time" value={dayShiftEnd} onChange={(event) => setDayShiftEnd(event.target.value)} style={policyInputStyle} /></label>
                            </div>
                          )}
                          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, color: '#475569' }}><strong>{dayShiftName || 'Unnamed shift'}</strong><span>{dayShiftStart}–{dayShiftEnd}</span><span style={{ color: '#0f766e', fontWeight: 800 }}>{getShiftDuration(dayShiftStart, dayShiftEnd)} hour duration</span></div>
                        </div>
                        {(() => {
                          const selectedShift = todaySchedule.find((shift) => String(shift._id) === String(daySelectedShiftId));
                          const assignedIds = new Set((selectedShift?.assignedStaff || []).map((staff) => String(staff.staffID?._id || staff.staffID)));
                          const clockedIds = new Set((dayClockedInStaff || []).map((record) => String(record.staff?._id || record.staffId || record._id)));
                          const availableRoster = filteredStaff.length ? filteredStaff : staffList;
                          const assignedRoster = availableRoster.filter((person) => assignedIds.has(String(person._id)));
                          const onCallRoster = availableRoster.filter((person) => person.assignment?.isOnCall || person.isOnCall);
                          return (
                            <div style={{ display: 'grid', gap: 10, padding: 14, border: '1px solid #99f6e4', borderRadius: 14, background: '#f0fdfa' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                                <div>
                                  <div style={{ fontSize: 13, fontWeight: 800, color: '#134e4a' }}>Scheduled shift for this hospital day</div>
                                  <div style={{ marginTop: 4, fontSize: 12, color: '#475569' }}>{selectedShift ? `${selectedShift.shiftName}${selectedShift.department ? ` • ${selectedShift.department}` : ''} • ${selectedShift.shiftStart}-${selectedShift.shiftEnd}` : 'No scheduled shift selected'}</div>
                                </div>
                                <span style={{ padding: '4px 8px', borderRadius: 999, background: selectedShift && isShiftCurrent(selectedShift) ? '#dcfce7' : '#fef3c7', color: selectedShift && isShiftCurrent(selectedShift) ? '#166534' : '#92400e', fontSize: 11, fontWeight: 800 }}>{selectedShift && isShiftCurrent(selectedShift) ? 'CURRENTLY ACTIVE' : 'SCHEDULED'}</span>
                              </div>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8, fontSize: 12 }}>
                                <div><strong>Assigned:</strong> {assignedRoster.length || selectedShift?.assignedStaff?.length || 0}</div>
                                <div><strong>On call:</strong> {onCallRoster.length}</div>
                                <div><strong>Clocked in:</strong> {assignedRoster.filter((person) => clockedIds.has(String(person._id))).length}</div>
                              </div>
                              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                                {assignedRoster.map((person) => <span key={person._id} style={{ padding: '5px 8px', borderRadius: 999, background: clockedIds.has(String(person._id)) ? '#dcfce7' : '#fff', border: '1px solid #99f6e4', color: '#134e4a', fontSize: 11 }}>{person.firstName || person.name} {person.lastName || person.surname || ''} {clockedIds.has(String(person._id)) ? '• CLOCKED IN' : '• SCHEDULED'}</span>)}
                                {onCallRoster.slice(0, 8).map((person) => <span key={`on-call-${person._id}`} style={{ padding: '5px 8px', borderRadius: 999, background: '#fff7ed', border: '1px solid #fed7aa', color: '#9a3412', fontSize: 11 }}>{person.firstName || person.name} {person.lastName || person.surname || ''} • ON CALL</span>)}
                              </div>
                              <button type="button" onClick={fetchCurrentStaffStatus} style={{ ...secondaryButtonStyle, justifySelf: 'start', minHeight: 34, padding: '7px 10px', fontSize: 11 }}>Refresh clock-in status</button>
                            </div>
                          );
                        })()}
                        <div style={{ padding: 14, borderRadius: 14, border: '1px solid #dbeafe', background: 'linear-gradient(135deg, #f8fbff 0%, #ecfeff 100%)', color: '#475569', fontSize: 12 }}>
                          This step uses the live staff roster and current shift coverage. It pulls directly from the same shift-tab roster used to build schedules and assigns selected staff into the day coverage plan before the hospital opens.
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap', padding: '10px 12px', border: '1px solid #dbeafe', borderRadius: 12, background: '#fff' }}>
                          <div style={{ fontSize: 12, color: '#475569', fontWeight: 700 }}>
                            Live roster: {(filteredStaff.length || staffList.length || 0)} staff available for coverage
                          </div>
                          <button
                            type="button"
                            onClick={pullShiftCoverageRoster}
                            style={{ ...secondaryButtonStyle, background: '#f8fafc' }}
                          >
                            Refresh live roster
                          </button>
                        </div>

                        <div style={{ display: 'grid', gap: 10, padding: 14, border: '1px solid #99f6e4', borderRadius: 14, background: '#f0fdfa' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'baseline', flexWrap: 'wrap' }}>
                            <div>
                              <div style={{ fontSize: 13, fontWeight: 800, color: '#134e4a' }}>Staff for this shift</div>
                              <div style={{ marginTop: 4, fontSize: 12, color: '#475569' }}>Select the fetched roster members who will be assigned when the hospital day starts.</div>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                              <strong style={{ fontSize: 12, color: '#0f766e' }}>{daySelectedStaffIds.length} selected</strong>
                              <button type="button" onClick={() => setDaySelectedStaffIds((filteredStaff.length ? filteredStaff : staffList).map((person) => String(person._id)))} style={{ ...secondaryButtonStyle, minHeight: 32, padding: '6px 10px', fontSize: 11 }}>Select all staff</button>
                              <button type="button" onClick={() => setDaySelectedStaffIds([])} style={{ ...secondaryButtonStyle, minHeight: 32, padding: '6px 10px', fontSize: 11 }}>Clear selection</button>
                            </div>
                          </div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 8, maxHeight: 220, overflowY: 'auto', padding: 2 }}>
                            {(filteredStaff.length ? filteredStaff : staffList).map((person) => {
                              const personId = String(person._id);
                              const checked = daySelectedStaffIds.includes(personId);
                              return (
                                <label key={personId} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, minWidth: 0, padding: '9px 10px', border: checked ? '1px solid #0f766e' : '1px solid #dbeafe', borderRadius: 10, background: checked ? '#ccfbf1' : '#fff', cursor: 'pointer' }}>
                                  <input type="checkbox" checked={checked} onChange={() => setDaySelectedStaffIds((current) => checked ? current.filter((id) => id !== personId) : [...current, personId])} style={{ marginTop: 2, flexShrink: 0 }} />
                                  <span style={{ minWidth: 0, fontSize: 12, lineHeight: 1.35, color: '#334155', overflowWrap: 'anywhere' }}>
                                    <strong style={{ display: 'block' }}>{`${person.firstName || person.name || ''} ${person.lastName || person.surname || ''}`.trim()}</strong>
                                    <span>{person.systemRole || person.role || 'Team member'}</span>
                                  </span>
                                </label>
                              );
                            })}
                          </div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                          {[
                            { key: 'triageNurseId', label: 'Triage Nurse', defaults: ['TRIAGE_NURSE', 'NURSE', 'EMERGENCY'] },
                            { key: 'pharmacistId', label: 'Pharmacist', defaults: ['PHARMACIST'] },
                            { key: 'radiologistId', label: 'Radiologist', defaults: ['RADIOLOGIST', 'RADIOLOGY'] },
                            { key: 'pathologistId', label: 'Pathologist', defaults: ['PATHOLOGIST', 'LAB'] }
                          ].map((field) => {
                            const rosterStaff = (filteredStaff.length ? filteredStaff : staffList).filter((person) => {
                              const role = (person.systemRole || person.role || person.userRole || '').toUpperCase();
                              const specialty = (person.clinicalSpecialty || person.specialty || '').toUpperCase();
                              return field.defaults.some((target) => role.includes(target) || specialty.includes(target));
                            });

                            return (
                              <div key={field.key} style={{ padding: '12px 10px', borderRadius: 12, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                                <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 700, color: '#334155' }}>
                                  {field.label}
                                  <select value={dayShiftAssignments[field.key]} onChange={(event) => setDayShiftAssignments((current) => ({ ...current, [field.key]: event.target.value }))} style={{ ...policyInputStyle, background: '#fff' }}>
                                    <option value="">Unassigned</option>
                                    {rosterStaff.map((person) => (
                                      <option key={person._id || `${field.key}-${person.firstName}`} value={person._id || `${field.key}-${person.firstName}`}>
                                        {person.firstName || person.name} {person.lastName || person.surname || ''} ({person.systemRole || person.role || 'Team'})
                                      </option>
                                    ))}
                                  </select>
                                </label>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {hospitalDayStep === 3 && (
                      <div style={{ display: 'grid', gap: 14 }}>
                        <div style={{ padding: 14, borderRadius: 14, border: '1px solid #dbeafe', background: '#f8fbff' }}>
                          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Synthetic patient data</div>
                          <div style={{ marginTop: 6, color: '#475569', fontSize: 13 }}>Seed hospital-scoped synthetic encounters for the day simulation. These records are clearly synthetic and do not modify real patient records.</div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                          <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 700, color: '#334155' }}>Encounter count<input type="number" min="1" max="1000" value={hospitalOperationsEncounterCount} onChange={(event) => setHospitalOperationsEncounterCount(event.target.value)} style={policyInputStyle} /></label>
                          <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 700, color: '#334155' }}>Historical span in days<input type="number" min="0" max="3650" value={hospitalOperationsHistoricalDays} onChange={(event) => setHospitalOperationsHistoricalDays(event.target.value)} style={policyInputStyle} /></label>
                          <label style={{ display: 'grid', gap: 6, fontSize: 12, fontWeight: 700, color: '#334155' }}>Priority<select value={hospitalOperationsPriority} onChange={(event) => setHospitalOperationsPriority(event.target.value)} style={{ ...policyInputStyle, background: '#fff' }}><option value="MIXED">Mixed acuity</option><option value="RED">RED</option><option value="ORANGE">ORANGE</option><option value="YELLOW">YELLOW</option><option value="GREEN">GREEN</option></select></label>
                        </div>
                        <div style={{ display: 'grid', gap: 12, padding: 16, borderRadius: 14, border: '1px solid #cbd5e1', background: '#ffffff' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                            <div>
                              <div style={{ fontSize: 13, fontWeight: 800, color: '#0f172a' }}>Synthetic patient roster</div>
                              <div style={{ marginTop: 4, fontSize: 12, color: '#64748b', maxWidth: 620 }}>Add each simulated patient as a separate record. Use realistic names for scenario tracking; these records are synthetic and remain hospital-scoped.</div>
                            </div>
                            <span style={{ padding: '5px 9px', borderRadius: 999, background: '#ecfeff', color: '#0f766e', fontSize: 11, fontWeight: 800 }}>{hospitalDayPatientRows.filter((row) => row.name.trim()).length} named patients</span>
                          </div>
                          <div style={{ display: 'grid', gap: 8 }}>
                            {hospitalDayPatientRows.map((row, index) => (
                              <div key={`patient-row-${index}`} style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 1fr) auto', gap: 8, alignItems: 'end', padding: 10, borderRadius: 10, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                                <label style={{ display: 'grid', gap: 5, minWidth: 0, fontSize: 11, fontWeight: 800, color: '#475569' }}>
                                  Patient {index + 1} name
                                  <input value={row.name} onChange={(event) => setHospitalDayPatientRows((rows) => rows.map((current, rowIndex) => rowIndex === index ? { ...current, name: event.target.value } : current))} placeholder="e.g. Amina Okafor" style={{ ...policyInputStyle, background: '#fff' }} />
                                </label>
                                <label style={{ display: 'grid', gap: 5, minWidth: 0, fontSize: 11, fontWeight: 800, color: '#475569' }}>
                                  Scenario note <span style={{ fontWeight: 500, color: '#94a3b8' }}>(optional)</span>
                                  <input value={row.note} onChange={(event) => setHospitalDayPatientRows((rows) => rows.map((current, rowIndex) => rowIndex === index ? { ...current, note: event.target.value } : current))} placeholder="e.g. fever triage" style={{ ...policyInputStyle, background: '#fff' }} />
                                </label>
                                <button type="button" aria-label={`Remove patient ${index + 1}`} onClick={() => setHospitalDayPatientRows((rows) => rows.length > 1 ? rows.filter((_, rowIndex) => rowIndex !== index) : rows)} style={{ ...removeRowButtonStyle, marginTop: 0, minHeight: 40, padding: '8px 10px' }}><FiX size={14} /> Remove</button>
                              </div>
                            ))}
                          </div>
                          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                            <button type="button" onClick={() => setHospitalDayPatientRows((rows) => [...rows, { name: '', note: '' }])} style={{ ...secondaryButtonStyle, minHeight: 38, padding: '8px 12px' }}><FiPlus size={14} /> Add patient</button>
                            <span style={{ fontSize: 11, color: '#64748b' }}>The encounter count controls how many records are generated; named rows provide the scenario identities.</span>
                          </div>
                        </div>
                        <button type="button" onClick={() => runHospitalOperationsAction('encounters/bulk', { count: Number(hospitalOperationsEncounterCount), historicalDays: Number(hospitalOperationsHistoricalDays), patientNames: hospitalDayPatientRows.map((row) => row.name.trim()).filter(Boolean), complaintType: hospitalOperationsComplaintType, complaint: hospitalOperationsCustomComplaint, priority: hospitalOperationsPriority === 'MIXED' ? '' : hospitalOperationsPriority, lifecycleStatus: hospitalOperationsLifecycle, triageNurseId: hospitalOperationsTriageNurseId || undefined })} disabled={hospitalOperationsAction === 'encounters/bulk'} style={policyButtonStyle}><FiPlus size={15} /> {hospitalOperationsAction === 'encounters/bulk' ? 'Seeding...' : 'Seed synthetic encounters'}</button>
                      </div>
                    )}

                    {hospitalDayStep === 4 && (
                      <div style={{ display: 'grid', gap: 14 }}>
                        <div style={{ padding: 16, borderRadius: 14, border: '1px solid #dbeafe', background: 'linear-gradient(135deg, #f0fdfa 0%, #f8fbff 100%)' }}>
                          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Day summary</div>
                          <div style={{ marginTop: 8, fontSize: 22, fontWeight: 800, color: '#0f172a' }}>{selectedHospitalName}</div>
                          <div style={{ marginTop: 8, color: '#475569' }}>System status: {Object.values(daySystemReadiness).every(Boolean) ? 'Operational' : 'Partial readiness'}</div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
                          {[
                            ['Beds', daySystemReadiness.beds, `${Number(hospitalOperationsStatus?.beds?.total || 0)} configured`],
                            ['Staff clock-in', daySystemReadiness.staff, `${staffList.length || filteredStaff.length || 0} on roster`],
                            ['Shift coverage', daySystemReadiness.shifts, `${Number(hospitalOperationsStatus?.shifts?.today || todaySchedule.length || 0)} scheduled today`],
                            ['Workload policy', daySystemReadiness.workload, capacityConfig?.policies?.defaultCapacityByRole ? 'Configured' : 'Needs setup'],
                            ['Synthetic patient load', daySystemReadiness.syntheticPatients, `${Number(hospitalOperationsEncounterCount) || 0} encounters requested`]
                          ].map(([label, ready, detail]) => (
                            <div key={label} style={{ padding: 12, borderRadius: 10, border: '1px solid #dbeafe', background: '#fff' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 12, fontWeight: 800, color: '#334155' }}><span>{label}</span><span style={{ color: ready ? '#15803d' : '#b45309' }}>{ready ? 'Ready' : 'Standby'}</span></div>
                              <div style={{ marginTop: 6, fontSize: 11, color: '#64748b' }}>{detail}</div>
                            </div>
                          ))}
                        </div>
                        <div style={{ display: 'grid', gap: 8, padding: 14, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
                          <strong style={{ color: '#0f172a' }}>Workload policy saved for this day</strong>
                          <div style={{ fontSize: 12, color: '#475569' }}>Role: {capacityRole} • Capacity: {savedCapacityValue ?? 'not set'} workload units</div>
                          <div style={{ fontSize: 12, color: '#475569' }}>Acuity weights: RED {acuityWeights.RED} • ORANGE {acuityWeights.ORANGE} • YELLOW {acuityWeights.YELLOW} • GREEN {acuityWeights.GREEN}</div>
                        </div>
                        <div style={{ display: 'grid', gap: 8, padding: 14, borderRadius: 12, border: '1px solid #dbeafe', background: '#fff' }}>
                          <strong style={{ color: '#0f172a' }}>Shift to be created</strong>
                          <div style={{ fontSize: 12, color: '#475569' }}>{dayShiftMode === 'existing' ? 'Selected scheduled shift' : 'New shift to create'}: {dayShiftName || 'Hospital Day Coverage'} • {dayShiftStart}-{dayShiftEnd} • {getShiftDuration(dayShiftStart, dayShiftEnd)} hours • {targetDate} • {daySelectedStaffIds.length} staff selected</div>
                          <div style={{ fontSize: 12, color: '#475569' }}>Synthetic data: {hospitalOperationsEncounterCount} encounters • {hospitalOperationsHistoricalDays} historical days • {hospitalOperationsPriority} priority • {hospitalOperationsLifecycle}</div>
                          <div style={{ marginTop: 6, fontSize: 12, color: '#475569' }}><strong>Named scenarios:</strong> {hospitalDayPatientRows.filter((row) => row.name.trim()).map((row) => row.note.trim() ? `${row.name.trim()} (${row.note.trim()})` : row.name.trim()).join(' • ') || 'No named patients added'}</div>
                        </div>
                        <div style={{ display: 'grid', gap: 8 }}>
                          <strong style={{ color: '#0f172a' }}>{daySelectedStaffIds.length} staff selected for the hospital day shift</strong>
                          {daySelectedStaffIds.map((staffId) => {
                            const rosterMember = (filteredStaff.length ? filteredStaff : staffList).find((person) => String(person._id) === String(staffId));
                            return (
                              <div key={staffId} style={{ padding: '12px 14px', borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff' }}>
                                <strong style={{ display: 'block', marginBottom: 4 }}>{rosterMember ? `${rosterMember.firstName || rosterMember.name} ${rosterMember.lastName || rosterMember.surname || ''}`.trim() : staffId}</strong>
                                <div style={{ fontSize: 12, color: '#64748b' }}>
                                  {rosterMember?.systemRole || rosterMember?.role || 'Selected roster staff'}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    <div style={{ marginTop: 20, display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                      <div>
                        {hospitalDayStep > 0 && <button type="button" onClick={() => setHospitalDayStep((current) => Math.max(current - 1, 0))} style={secondaryButtonStyle}>Back</button>}
                      </div>
                      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                        {hospitalDayStep < 4 ? (
                          <button type="button" onClick={() => setHospitalDayStep((current) => Math.min(current + 1, 4))} style={policyButtonStyle}>Continue</button>
                        ) : (
                          <button type="button" onClick={startHospitalDay} style={{ ...policyButtonStyle, background: 'linear-gradient(135deg, #0f766e 0%, #0d9488 100%)' }}>Start Hospital Shift</button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {hospitalSetupOpen && (
              <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20, zIndex: 60 }}>
                <div style={{ width: 'min(900px, 100%)', maxHeight: '88vh', overflowY: 'auto', background: '#fff', borderRadius: 18, boxShadow: '0 26px 64px rgba(15, 23, 42, 0.22)', border: '1px solid #e2e8f0' }}>
                  <div id="hospital-setup-panel" style={{ padding: '20px 22px 14px', borderBottom: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#0f766e' }}>Guided hospital setup</div>
                      <h3 style={{ margin: '8px 0 0', fontSize: 24, color: '#0f172a' }}>Create a new hospital</h3>
                      <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 13 }}>Set up the facility, its administrator, and its first clinical team in one reviewed flow.</p>
                    </div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                      <button type="button" onClick={autoFillHospitalSetup} style={secondaryButtonStyle}><FiSliders size={14} /> Auto-fill sample</button>
                      <button type="button" onClick={() => setHospitalSetupOpen(false)} style={{ ...removeRowButtonStyle, background: '#f8fafc', borderColor: '#cbd5e1', color: '#475569' }}><FiX size={14} /> Close</button>
                    </div>
                  </div>

                  <div style={{ padding: 20 }}>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 18 }}>
                      {['Facility & admin', 'Staff credentials', 'Review & create'].map((stepLabel, index) => (
                        <div key={stepLabel} style={{
                          display: 'flex', alignItems: 'center', gap: 8,
                          padding: '8px 10px', borderRadius: 999,
                          border: index === setupStep ? '1px solid #0f766e' : '1px solid #dbeafe',
                          background: index === setupStep ? '#ecfeff' : '#f8fafc',
                          color: index === setupStep ? '#0f172a' : '#475569',
                          fontWeight: 700, fontSize: 12
                        }}>
                          <span style={{ width: 20, height: 20, borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', background: index === setupStep ? '#0f766e' : '#dbeafe', color: index === setupStep ? '#fff' : '#0f172a' }}>{index + 1}</span>
                          {stepLabel}
                        </div>
                      ))}
                    </div>

                    <div style={{ marginBottom: 16, padding: '12px 14px', borderRadius: 12, background: '#f0fdfa', border: '1px solid #99f6e4', color: '#115e59', fontSize: 13 }}>
                      {setupStep === 0 && 'Start with the hospital identity and the administrator who will own this workspace.'}
                      {setupStep === 1 && 'Add the first people who can work in this hospital. Each person receives an individual login.'}
                      {setupStep === 2 && 'Check the generated credentials and operating details before anything is created.'}
                    </div>

                    {setupStep === 0 && (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
                        <label style={adminFieldStyle}>Hospital name<input value={hospitalBootstrap.hospitalName} onChange={(event) => setHospitalBootstrap((current) => ({ ...current, hospitalName: event.target.value }))} placeholder="St. Mary’s Hospital" style={policyInputStyle} /></label>
                        <label style={adminFieldStyle}>Additional admin first name<input value={hospitalBootstrap.managerName} onChange={(event) => setHospitalBootstrap((current) => ({ ...current, managerName: event.target.value }))} placeholder="John" style={policyInputStyle} /></label>
                        <label style={adminFieldStyle}>Additional admin surname<input value={hospitalBootstrap.managerSurname} onChange={(event) => setHospitalBootstrap((current) => ({ ...current, managerSurname: event.target.value }))} placeholder="Smith" style={policyInputStyle} /></label>
                        <label style={adminFieldStyle}>Additional admin email<input type="email" value={hospitalBootstrap.managerEmail} onChange={(event) => setHospitalBootstrap((current) => ({ ...current, managerEmail: event.target.value }))} placeholder="admin@hospital.org" style={policyInputStyle} /></label>
                        <PasswordField
                          label="Additional admin password"
                          value={hospitalBootstrap.managerPassword}
                          onChange={(event) => setHospitalBootstrap((current) => ({ ...current, managerPassword: event.target.value }))}
                          placeholder="Minimum 8 characters"
                          style={adminFieldStyle}
                        />
                        <label style={adminFieldStyle}>Default department<select value={hospitalBootstrap.department} onChange={(event) => setHospitalBootstrap((current) => ({ ...current, department: event.target.value }))} style={policyInputStyle}><option value="All">All</option><option value="ED">ED</option><option value="ICU">ICU</option><option value="OR">OR</option><option value="OB">OB</option><option value="General Ward">General Ward</option><option value="Pediatrics">Pediatrics</option></select></label>
                        <label style={adminFieldStyle}>Staff users to seed<input type="number" min="1" max="500" value={setupStaffCount} onChange={(event) => setSetupStaffCount(Math.min(500, Math.max(1, Number(event.target.value) || 1)))} style={policyInputStyle} /></label>
                        <label style={adminFieldStyle}>Historical encounters<input type="number" min="1" max="1000" value={setupEncounterCount} onChange={(event) => setSetupEncounterCount(Math.min(1000, Math.max(1, Number(event.target.value) || 1)))} style={policyInputStyle} /></label>
                        <label style={adminFieldStyle}>History spread in days<input type="number" min="0" max="3650" value={setupHistoricalDays} onChange={(event) => setSetupHistoricalDays(Math.min(3650, Math.max(0, Number(event.target.value) || 0)))} style={policyInputStyle} /></label>
                        <div style={{ gridColumn: '1 / -1', padding: '10px 12px', borderRadius: 10, background: '#f8fafc', border: '1px solid #e2e8f0', color: '#64748b', fontSize: 12 }}>Your current administrator account will own and manage this hospital. The additional admin credentials create a separate administrator login. Auto-fill uses these quantities; encounter seeding requires at least one nurse.</div>
                        <div style={{ gridColumn: '1 / -1' }}>
                          <div style={{ fontSize: 11, fontWeight: 800, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>Departments</div>
                          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            {HOSPITAL_DEPARTMENTS.map(({ name: department }) => {
                              const active = hospitalBootstrap.departments.includes(department);
                              return (
                                <button
                                  key={department}
                                  type="button"
                                  onClick={() => setHospitalBootstrap((current) => ({
                                    ...current,
                                    departments: active ? current.departments.filter((item) => item !== department) : [...current.departments, department]
                                  }))}
                                  style={{
                                    padding: '7px 10px',
                                    borderRadius: 999,
                                    border: '1px solid #bfdbfe',
                                    background: active ? '#dbeafe' : '#fff',
                                    color: active ? '#1d4ed8' : '#475569',
                                    fontWeight: 700,
                                    fontSize: 12,
                                    cursor: 'pointer'
                                  }}
                                >
                                  {department}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    )}

                    {setupStep === 1 && (
                      <div style={{ display: 'grid', gap: 12 }}>
                        {bootstrapStaffRows.map((staff, index) => (
                          <div key={`setup-staff-${index}`} style={{ padding: 16, border: '1px solid #dbeafe', borderRadius: 14, background: '#fff', boxShadow: '0 4px 12px rgba(15, 23, 42, 0.04)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                              <div>
                                <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Team member {index + 1}</div>
                                <strong style={{ display: 'block', marginTop: 4, color: '#0f172a' }}>{staff.firstName || staff.lastName ? `${staff.firstName} ${staff.lastName}`.trim() : 'New credential'}</strong>
                              </div>
                              <span style={{ padding: '5px 8px', borderRadius: 999, background: '#f0fdfa', color: '#0f766e', fontSize: 11, fontWeight: 800 }}>{HOSPITAL_STAFF_ROLES.find((position) => position.code === staff.positionCode)?.label || staff.role}</span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10 }}>
                              <label style={adminFieldStyle}>First name<input value={staff.firstName} onChange={(event) => setBootstrapStaffRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, firstName: event.target.value } : row))} style={policyInputStyle} /></label>
                              <label style={adminFieldStyle}>Last name<input value={staff.lastName} onChange={(event) => setBootstrapStaffRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, lastName: event.target.value } : row))} style={policyInputStyle} /></label>
                              <label style={adminFieldStyle}>Email<input type="email" value={staff.email} onChange={(event) => setBootstrapStaffRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, email: event.target.value } : row))} style={policyInputStyle} /></label>
                              <PasswordField
                                label="Password"
                                value={staff.password}
                                onChange={(event) => setBootstrapStaffRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, password: event.target.value } : row))}
                                style={adminFieldStyle}
                              />
                              <label style={adminFieldStyle}>Position / job<select value={staff.positionCode || ''} onChange={(event) => updateBootstrapStaffRow(index, 'positionCode', event.target.value)} style={policyInputStyle}>
                                <option value="">Select a position</option>
                                {HOSPITAL_STAFF_ROLES.map((position) => <option key={position.code} value={position.code}>{position.label} — {position.category}</option>)}
                              </select></label>
                              <label style={adminFieldStyle}>Department<select value={staff.department} onChange={(event) => updateBootstrapStaffRow(index, 'department', event.target.value)} style={policyInputStyle}>
                                <option value="">Select a hospital department</option>
                                {hospitalBootstrap.departments.map((department) => <option key={department} value={department}>{department}</option>)}
                              </select></label>
                              <label style={adminFieldStyle}>Specialty (added to this department)<select value={staff.specialty} onChange={(event) => updateBootstrapStaffRow(index, 'specialty', event.target.value)} style={policyInputStyle}>
                                <option value="">Select a specialty</option>
                                {HOSPITAL_SPECIALTIES.map((specialty) => <option key={specialty.name} value={specialty.name}>{specialty.name} — {specialty.category}</option>)}
                              </select></label>
                            </div>
                          </div>
                        ))}
                        <button type="button" onClick={() => setBootstrapStaffRows((rows) => [...rows, { firstName: '', lastName: '', email: '', password: '', role: 'NURSE', positionCode: 'TRIAGE_NURSE', department: 'ED', specialty: '' }])} style={secondaryButtonStyle}><FiPlus size={15} /> Add another team member</button>
                        {setupAutoFilled && <div style={{ fontSize: 12, color: '#64748b' }}>Sample credentials are randomized for this setup session. Replace them with real credentials before creating the hospital if needed.</div>}
                      </div>
                    )}

                    {setupStep === 2 && (
                      <div style={{ display: 'grid', gap: 14 }}>
                        <div style={{ padding: 14, borderRadius: 12, border: '1px solid #dbeafe', background: '#f8fbff' }}>
                          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>Summary</div>
                          <div style={{ marginTop: 8, fontSize: 22, fontWeight: 800, color: '#0f172a' }}>{hospitalBootstrap.hospitalName || 'Unnamed hospital'}</div>
                          <div style={{ marginTop: 8, color: '#475569' }}>{hospitalBootstrap.managerName || 'Manager'} {hospitalBootstrap.managerSurname || ''} • {hospitalBootstrap.managerEmail || 'admin@hospital.org'}</div>
                          <div style={{ marginTop: 8, color: '#0f766e', fontSize: 13, fontWeight: 700 }}>{bootstrapStaffRows.filter((row) => row.firstName || row.lastName || row.email).length} staff users • {setupEncounterCount} historical encounters • {setupHistoricalDays} day history</div>
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                          {hospitalBootstrap.departments.map((department) => (
                            <span key={department} style={{ padding: '7px 10px', borderRadius: 999, background: '#dbeafe', color: '#1d4ed8', fontWeight: 700, fontSize: 12 }}>{department}</span>
                          ))}
                        </div>
                        <div style={{ display: 'grid', gap: 8 }}>
                          {bootstrapStaffRows.filter((row) => row.firstName || row.lastName || row.email).map((staff, index) => (
                            <div key={`review-${index}`} style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff' }}>
                              <strong>{staff.firstName || 'New'} {staff.lastName || 'team member'}</strong>
                              <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{HOSPITAL_STAFF_ROLES.find((position) => position.code === staff.positionCode)?.label || staff.role} • {staff.department} • {staff.specialty || 'No specialty'} • {staff.email || 'No email'}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div style={{ marginTop: 20, display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                      <div>
                        {setupStep > 0 && <button type="button" onClick={() => setSetupStep((current) => Math.max(current - 1, 0))} style={secondaryButtonStyle}>Back</button>}
                      </div>
                      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                        {setupStep < 2 ? (
                          <button type="button" onClick={() => setSetupStep((current) => Math.min(current + 1, 2))} style={policyButtonStyle}>Continue</button>
                        ) : (
                          <button type="button" onClick={async () => { await submitHospitalBootstrap(); }} disabled={managedHospitalLoading} style={policyButtonStyle}>{managedHospitalLoading ? 'Creating...' : 'Create hospital'}</button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </SidebarLayout>
  );
};

const policyInputStyle = {
  width: '100%',
  boxSizing: 'border-box',
  padding: '11px 12px',
  border: '1px solid #d9e4ef',
  borderRadius: 12,
  background: '#f8fbff',
  color: '#0f172a',
  fontSize: 14,
  fontWeight: 500,
  boxShadow: 'inset 0 1px 2px rgba(15, 23, 42, 0.04)',
  transition: 'border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease'
};

const policyButtonStyle = {
  minHeight: 42,
  padding: '10px 16px',
  border: '1px solid transparent',
  borderRadius: 12,
  background: 'linear-gradient(135deg, #0f766e 0%, #0d9488 100%)',
  boxShadow: '0 10px 24px rgba(13, 148, 136, 0.22)',
  color: '#fff',
  cursor: 'pointer',
  fontSize: 13,
  fontWeight: 800,
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 8,
  whiteSpace: 'normal',
  lineHeight: 1.25,
  textAlign: 'center',
  overflowWrap: 'anywhere',
  wordBreak: 'break-word',
  letterSpacing: '0.01em'
};

const secondaryButtonStyle = {
  ...policyButtonStyle,
  background: '#ffffff',
  color: '#0f766e',
  border: '1px solid #9adbcf',
  boxShadow: '0 8px 18px rgba(15, 118, 110, 0.08)'
};

const removeRowButtonStyle = {
  marginTop: 10,
  padding: '7px 10px',
  border: '1px solid #fecac9',
  borderRadius: 10,
  background: 'linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%)',
  color: '#be123c',
  cursor: 'pointer',
  fontSize: 12,
  fontWeight: 800,
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 6,
  whiteSpace: 'normal',
  lineHeight: 1.2,
  textAlign: 'center',
  overflowWrap: 'anywhere',
  wordBreak: 'break-word',
  boxShadow: '0 8px 16px rgba(190, 18, 60, 0.08)'
};

const adminFieldStyle = {
  display: 'grid',
  gap: 6,
  fontSize: 11,
  fontWeight: 800,
  color: '#475569',
  letterSpacing: '0.02em'
};

export default StaffScheduleManager;
