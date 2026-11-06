import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import StaffScheduleDisplay from './StaffScheduleDisplay';
import './ManualTriageForm.css';
import { FiHeart, FiThermometer, FiActivity, FiClock, FiUser, FiMove, FiAlertTriangle } from 'react-icons/fi';
import { buildAssignmentPayload, enrichSelectedBedWithWard } from './assignmentUtils';
import { getSatsAcuityFromTews } from '../utils/triageHelpers';
import { calculateTews, getInitialTewsProfile } from '../utils/tewsCalculator';

const normalizePatientRecord = (input = {}) => {
  const patient = input && typeof input === 'object' ? input : {};
  const idNumber = patient.id_number ?? patient.id ?? patient._id ?? null;
  const ageYears = patient.ageYears ?? (Number.isFinite(Number(patient.age)) ? Number(patient.age) : null);

  return {
    ...patient,
    id_number: idNumber,
    id: idNumber,
    _id: patient._id ?? idNumber,
    name: patient.name ?? null,
    surname: patient.surname ?? null,
    gender: patient.gender ?? null,
    age: patient.age ?? ageYears ?? null,
    ageYears,
    ageMonths: patient.ageMonths ?? null,
    height_cm: patient.height_cm ?? null,
    symptoms: patient.symptoms ?? null,
    duration: patient.duration ?? null,
    painLevel: patient.painLevel ?? null,
    history: patient.history ?? null,
  };
};


/**
 * ManualTriageForm - Complete manual triage workflow (SATS - South African Triage Scale)
 * 
 * Intelligent Hardware Routing:
 * - On mount: Queries /api/capabilities to detect GPU availability
 * - If GPU + RAG: Routes to /api/triage (hybrid RAG + LLM synthesis)
 * - If CPU only: Routes to /api/llm-triage (LLM only, no RAG)
 * 
 * User Journey:
 * 1. Symptoms/vitals already captured (from Vitals component)
 * 2. Nurse manually decides SATS level (Red/Orange/Yellow/Green)
 * 3. Enter clinical reasoning
 * 4. Select available nurse, physician, bed (optional)
 * 5. Submit (routes to appropriate endpoint based on hardware)
 */
function ManualTriageForm({ patientData = {}, vitalsData = {}, ehrSummary = null, selectedMrn = null, onSubmit = () => {}, onPayloadChange = () => {}, onSubmitPayload = () => {}, onAnalysisChange = () => {}, onTewsChange = () => {}, onAssignmentChange = () => {}, onDepartmentChange = () => {}, onAcuityChange = () => {}, onFormReadyChange = () => {}, submitRequest = 0 }) {
  const navigate = useNavigate();
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
  const safePatientData = useMemo(() => normalizePatientRecord(patientData), [patientData]);

  // Manual triage always submits to the dedicated manual triage endpoint
  const [triageEndpoint] = useState('/api/triage');
  
  // Manual triage form state (SATS: Red/Orange/Yellow/Green)
  const [manualSatsOverride, setManualSatsOverride] = useState(false);
  const [satsLevel, setSatsLevel] = useState('green');
  const [urgencyScore, setUrgencyScore] = useState('50');
  const [tewsProfile, setTewsProfile] = useState(() => getInitialTewsProfile(patientData));
  const [tewsHeight, setTewsHeight] = useState(patientData.height_cm || '');
  const [tewsVitals, setTewsVitals] = useState(() => ({
    temperature: vitalsData.temp || '',
    heartRate: vitalsData.hr || '',
    respiratoryRate: vitalsData.resp || '',
    systolicBP: vitalsData.bp_systolic || '',
    avpu: vitalsData.avpu || 'Alert',
    mobility: vitalsData.mobility || 'Walks',
    traumaPresent: Boolean(vitalsData.traumaPresent)
  }));

  useEffect(() => {
    setTewsVitals((current) => ({
      ...current,
      temperature: vitalsData.temp ?? current.temperature,
      heartRate: vitalsData.hr ?? current.heartRate,
      respiratoryRate: vitalsData.resp ?? current.respiratoryRate,
      systolicBP: vitalsData.bp_systolic ?? current.systolicBP,
      avpu: vitalsData.avpu ?? current.avpu,
      mobility: vitalsData.mobility ?? current.mobility,
      traumaPresent: vitalsData.traumaPresent ?? current.traumaPresent
    }));
  }, [vitalsData]);
    const [reasoning, setReasoning] = useState('');
    const reasoningRef = useRef(null);
  const [department, setDepartment] = useState('ED');
  const [interventions, setInterventions] = useState('');

  const inferDepartmentFromPhysicianSpecialty = (specialty, fallback = 'Not assigned') => {
    if (!specialty || typeof specialty !== 'string') return fallback;
    return specialty;
  };

  // Staff selection
  const [selectedNurse, setSelectedNurse] = useState(null);
  const [selectedPhysician, setSelectedPhysician] = useState(null);
  const [selectedSpecialist, setSelectedSpecialist] = useState(null);
  const [selectedBed, setSelectedBed] = useState(null);
  const [lastSubmitRequest, setLastSubmitRequest] = useState(0);

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [showStaffSelection, setShowStaffSelection] = useState(false);
  const [showStaffModal, setShowStaffModal] = useState(false);
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [showTewsModal, setShowTewsModal] = useState(false);

  // Structured reasoning mode to reduce vertical length: 'structured' (compact fields) or 'full' (large editor)
  const [reasoningMode, setReasoningMode] = useState('structured');
  const [keyFindings, setKeyFindings] = useState('');
  const [assessment, setAssessment] = useState('');
  const [plan, setPlan] = useState('');

  // AI-style triage analysis fields
  const [pathophysiology, setPathophysiology] = useState('');
  const [clinicalSummary, setClinicalSummary] = useState('');
  const [clinicalDiscriminators, setClinicalDiscriminators] = useState('');
  const [differentialDiagnoses, setDifferentialDiagnoses] = useState('');
  const [immediateActions, setImmediateActions] = useState('');
  const [diagnosticWorkup, setDiagnosticWorkup] = useState('');
  const [dispositionRationale, setDispositionRationale] = useState('');

  const updateAnalysisField = (field, setter, value) => {
    setter(value);
    onAnalysisChange(field, value);
  };

  // Stepper UI state
  const steps = ['Triage', 'Clinical Analysis', 'Assign Clinicians & Wards'];
  const [currentStep, setCurrentStep] = useState(0);

  // Interventions as chips instead of long textarea
  const [interventionInput, setInterventionInput] = useState('');
  const [interventionTags, setInterventionTags] = useState([]);

  const tewsResult = calculateTews(tewsProfile, tewsVitals);
  const tewsComponentsKey = JSON.stringify(tewsResult.components);
  const tewsAcuityLevel = getSatsAcuityFromTews(tewsResult.total);
  const effectiveSatsLevel = manualSatsOverride ? satsLevel : tewsAcuityLevel;
  const tewsAcuityLabel = {
    red: '7+',
    orange: '5-6',
    yellow: '3-4',
    green: '0-2'
  }[tewsAcuityLevel] || '0-2';

  const hasPatientHistoryLoaded = Boolean(ehrSummary?.lastEncounter);
  const hasPatientContext = Boolean(
    patientData?.id ||
    patientData?._id ||
    patientData?.id_number ||
    patientData?.mrn ||
    selectedMrn ||
    patientData?.name
  );
  const shouldShowNoHistoryMessage = !hasPatientHistoryLoaded && hasPatientContext;

  useEffect(() => {
    const autoProfile = getInitialTewsProfile(patientData);
    setTewsProfile((current) => (current === autoProfile ? current : autoProfile));
  }, [patientData?.ageYears, patientData?.age, patientData?.ageMonths, patientData?.height_cm]);

  useEffect(() => {
    if (!manualSatsOverride) {
      setSatsLevel(tewsAcuityLevel);
    }
  }, [manualSatsOverride, tewsAcuityLevel]);

  const hasCompleteClinicalAnalysis = () => {
    const parsedClinicalDiscriminators = clinicalDiscriminators
      .split('\n')
      .map((item) => item.trim())
      .filter(Boolean);
    const parsedDifferentialDiagnoses = differentialDiagnoses
      .split('\n')
      .map((item) => item.trim())
      .filter(Boolean);
    const parsedImmediateActions = immediateActions
      .split('\n')
      .map((item) => item.trim())
      .filter(Boolean);
    const parsedDiagnosticWorkup = diagnosticWorkup
      .split('\n')
      .map((item) => item.trim())
      .filter(Boolean);

    return (
      pathophysiology.trim().length > 0 &&
      clinicalSummary.trim().length > 0 &&
      parsedClinicalDiscriminators.length > 0 &&
      parsedDifferentialDiagnoses.length > 0 &&
      parsedImmediateActions.length > 0 &&
      parsedDiagnosticWorkup.length > 0 &&
      dispositionRationale.trim().length > 0
    );
  };

  useEffect(() => {
    if (!tewsResult.complete) return;
    setUrgencyScore(String(Math.max(0, Number(tewsResult.total) || 0)));
  }, [tewsResult.complete, tewsResult.total]);

  useEffect(() => {
    const derivedPriority = effectiveSatsLevel || 'green';
    onTewsChange({
      profile: tewsProfile,
      height: tewsHeight,
      total: tewsResult.total,
      priority: derivedPriority,
      manualPriority: satsLevel,
      tewsPriority: tewsAcuityLevel,
      components: tewsResult.components,
      mobility: tewsVitals.mobility,
      traumaPresent: Boolean(tewsVitals.traumaPresent),
      traumaModifier: tewsResult.components.trauma,
      complete: tewsResult.complete
    });
  }, [onTewsChange, effectiveSatsLevel, satsLevel, tewsAcuityLevel, tewsComponentsKey, tewsHeight, tewsProfile, tewsResult.total, tewsResult.complete]);

  const requiredFieldState = useMemo(() => {
    const analysisValid = hasCompleteClinicalAnalysis();
    const heightValid = tewsProfile === 'adult'
      ? true
      : (tewsProfile === 'child-younger' ? Number(tewsHeight) >= 50 && Number(tewsHeight) <= 95 : Number(tewsHeight) >= 96 && Number(tewsHeight) <= 150);

    const triageLevelSelected = !!satsLevel && ['red', 'orange', 'yellow', 'green'].includes(satsLevel);
    const tewsReady = tewsResult.complete && Number.isFinite(Number(tewsResult.total));
    const staffAssigned = !!selectedNurse && !!selectedPhysician;
    const criticalHistoryComplete = ['reported', 'none_established'].includes(safePatientData.criticalHistoryStatus);

    return {
      triageLevelSelected,
      analysisValid,
      tewsReady,
      heightValid,
      staffAssigned,
      criticalHistoryComplete,
      formValid: triageLevelSelected && analysisValid && tewsReady && heightValid && staffAssigned && criticalHistoryComplete
    };
  }, [safePatientData.criticalHistoryStatus, satsLevel, pathophysiology, clinicalSummary, clinicalDiscriminators, differentialDiagnoses, immediateActions, diagnosticWorkup, dispositionRationale, tewsProfile, tewsHeight, tewsResult.complete, tewsResult.total, selectedNurse, selectedPhysician]);

  const formValid = requiredFieldState.formValid;

  useEffect(() => {
    onFormReadyChange(formValid);
  }, [formValid, onFormReadyChange]);

  useEffect(() => {
    onAssignmentChange({ nurse: selectedNurse, physician: selectedPhysician, specialist: selectedSpecialist, bed: selectedBed });
  }, [onAssignmentChange, selectedBed, selectedNurse, selectedPhysician, selectedSpecialist]);

  const currentTriagePayload = useMemo(() => {
    const payload = {
      patientId: safePatientData.id_number || safePatientData.id || safePatientData._id,
      criticalHistoryStatus: safePatientData.criticalHistoryStatus,
      criticalHistory: safePatientData.criticalHistory || {},
      encounterId: safePatientData.encounterId || null,
      symptoms: typeof safePatientData.symptoms === 'string' ? safePatientData.symptoms : (Array.isArray(safePatientData.symptoms) ? safePatientData.symptoms.join(', ') : ''),
      vitalSigns: {
        temperature: parseFloat(vitalsData.temp) || null,
        heartRate: parseInt(vitalsData.hr) || null,
        respiratoryRate: parseInt(vitalsData.resp) || null,
        systolicBP: vitalsData.bp_systolic || null,
        diastolicBP: vitalsData.bp_diastolic || null,
        oxygenSaturation: parseFloat(vitalsData.o2) || null,
        avpu: tewsVitals.avpu || vitalsData.avpu || null,
        mobility: tewsVitals.mobility || vitalsData.mobility || 'Walks',
        traumaPresent: Boolean(tewsVitals.traumaPresent || vitalsData.traumaPresent)
      },
      analysis: {
        pathophysiology,
        clinicalSummary,
        clinicalDiscriminators,
        differentialDiagnoses,
        immediateActions,
        diagnosticWorkup,
        dispositionRationale
      },
      satsLevel: effectiveSatsLevel,
      urgencyScore,
      tews: {
        profile: tewsProfile,
        height: tewsHeight === '' ? null : Number(tewsHeight),
        components: tewsResult.components,
        total: tewsResult.total,
        priority: effectiveSatsLevel,
        mobility: tewsVitals.mobility,
        traumaPresent: Boolean(tewsVitals.traumaPresent),
        traumaModifier: tewsResult.components.trauma,
        complete: tewsResult.complete
      },
      recommendedDepartment: department,
      interventions: interventionTags,
      assignments: {
        nurse: selectedNurse,
        physician: selectedPhysician,
        specialist: selectedSpecialist,
        bed: selectedBed
      }
    };
    console.log('[ManualTriageForm] ⚠️ PAYLOAD BUILD - Checking tews/urgency mapping:', {
      satsLevel: payload.satsLevel,
      urgencyScore: payload.urgencyScore,
      tewsTotal: payload.tews.total,
      tewsPriority: payload.tews.priority,
      mobility: payload.tews.mobility,
      traumaPresent: payload.tews.traumaPresent,
      traumaModifier: payload.tews.traumaModifier,
      warning: 'urgencyScore contains TEWS numeric total. This is the ROOT CAUSE of acuity confusion.'
    });
    return payload;
  }, [
    patientData,
    vitalsData,
    pathophysiology,
    clinicalSummary,
    clinicalDiscriminators,
    differentialDiagnoses,
    immediateActions,
    diagnosticWorkup,
    dispositionRationale,
    satsLevel,
    urgencyScore,
    tewsProfile,
    tewsHeight,
    tewsComponentsKey,
    tewsResult.total,
    tewsResult.priority,
    tewsResult.complete,
    department,
    interventionTags,
    selectedNurse,
    selectedPhysician,
    selectedSpecialist,
    selectedBed
  ]);

  useEffect(() => {
    onPayloadChange(currentTriagePayload);
  }, [currentTriagePayload, onPayloadChange]);

  useEffect(() => {
    if (submitRequest > 0 && submitRequest !== lastSubmitRequest) {
      setLastSubmitRequest(submitRequest);
      if (formValid) {
        handleSubmitTriage();
      }
    }
  }, [submitRequest, lastSubmitRequest, formValid]);

  useEffect(() => {
    if (selectedPhysician) {
      setDepartment(inferDepartmentFromPhysicianSpecialty(selectedPhysician.specialty));
    } else {
      setDepartment('Not assigned');
    }
  }, [selectedPhysician]);

  useEffect(() => {
    onDepartmentChange(department);
  }, [department, onDepartmentChange]);

  useEffect(() => {
    onAcuityChange(urgencyScore);
  }, [onAcuityChange, urgencyScore]);

  // Form validation
  // Bed is OPTIONAL - not every patient gets hospitalized
  const isFormValid = () => requiredFieldState.formValid;

  const handleStaffSelect = {
    onNurseSelect: (nurse) => {
      setSelectedNurse(nurse);
    },
    onPhysicianSelect: (physician) => {
      setSelectedPhysician(physician);
    },
    onSpecialistSelect: (specialist) => {
      setSelectedSpecialist(specialist);
    },
    onBedSelect: (bed) => {
      setSelectedBed(bed);
    }
  };

  // Load saved reasoning draft if present (autosave)
  useEffect(() => {
    try {
      const draft = localStorage.getItem('manual_reasoning_draft');
      if (draft && !reasoning) {
        setReasoning(draft);
      }
    } catch (err) {
      // ignore
    }
  }, []);

  // Autosave reasoning to localStorage (debounced-ish)
  const [autosaveStatus, setAutosaveStatus] = useState('');
  useEffect(() => {
    if (reasoning === '') {
      localStorage.removeItem('manual_reasoning_draft');
      setAutosaveStatus('');
      return;
    }

    setAutosaveStatus('Saving...');
    const t = setTimeout(() => {
      try {
        localStorage.setItem('manual_reasoning_draft', reasoning);
        setAutosaveStatus('Saved');
        setTimeout(() => setAutosaveStatus(''), 1200);
      } catch (err) {
        setAutosaveStatus('Failed to save');
      }
    }, 700);

    return () => clearTimeout(t);
  }, [reasoning]);

  const insertAtCursor = (text) => {
    const el = reasoningRef.current;
    if (!el) return setReasoning((r) => (r ? r + '\n' + text : text));
    const start = el.selectionStart || 0;
    const end = el.selectionEnd || 0;
    const before = reasoning.slice(0, start);
    const after = reasoning.slice(end);
    const newText = before + text + after;
    setReasoning(newText);
    // move cursor after inserted text
    requestAnimationFrame(() => {
      try { el.selectionStart = el.selectionEnd = start + text.length; el.focus(); } catch (e) {}
    });
  };

  const handleInsertTemplate = () => {
    const template = "Key findings:\n- Airway: \n- Breathing: \n- Circulation: \nAssessment:\nPlan:\n";
    insertAtCursor(template);
  };

  const handleFormatBold = () => {
    insertAtCursor('**bold**');
  };

  const handleFormatBullet = () => {
    insertAtCursor('\n- ');
  };
  

  const handleSubmitTriage = async () => {
    const composedReasoning = [
      clinicalSummary && `Clinical Summary:\n${clinicalSummary}`,
      pathophysiology && `Pathophysiology:\n${pathophysiology}`,
      clinicalDiscriminators && `Clinical Discriminators:\n${clinicalDiscriminators}`,
      differentialDiagnoses && `Differential Diagnoses:\n${differentialDiagnoses}`,
      diagnosticWorkup && `Diagnostic Workup:\n${diagnosticWorkup}`,
      immediateActions && `Immediate Actions:\n${immediateActions}`,
      dispositionRationale && `Disposition Rationale:\n${dispositionRationale}`
    ].filter(Boolean).join('\n\n');

    if (!isFormValid()) {
      setSubmitError('Please complete all required fields: SATS level, clinical analysis, and assign a nurse & physician. Bed assignment is optional.');
      return;
    }

    setSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(false);

    try {
      const token = localStorage.getItem('token');
      const assignmentMeta = buildAssignmentPayload(selectedBed, selectedBed?.ward || null);
      const enrichedSelectedBed = enrichSelectedBedWithWard(selectedBed, selectedBed?.ward || null);

      // Build manual triage request (NO computeMode)
      const triagePayload = {
        // NO computeMode - signals MANUAL workflow
        
        // Clinical data from patient intake
        // Convert symptoms to string (never send as array)
        symptoms: typeof patientData.symptoms === 'string' 
          ? patientData.symptoms 
          : (Array.isArray(patientData.symptoms) ? patientData.symptoms.join(", ") : ""),
        vitalSigns: {
          temperature: parseFloat(vitalsData.temp) || null,
          heartRate: parseInt(vitalsData.hr) || null,
          respiratoryRate: parseInt(vitalsData.resp) || null,
          systolicBP: vitalsData.bp_systolic || null,
          diastolicBP: vitalsData.bp_diastolic || null,
          oxygenSaturation: parseFloat(vitalsData.o2) || null,
          avpu: tewsVitals.avpu || vitalsData.avpu || null,
          mobility: tewsVitals.mobility || vitalsData.mobility || 'Walks',
          traumaPresent: Boolean(tewsVitals.traumaPresent || vitalsData.traumaPresent),
          traumaModifier: tewsResult.components.trauma
        },
        medicalHistory: patientData.history ? [patientData.history] : [],
        medications: patientData.medications || [],
        allergies: patientData.allergies || [],
        criticalHistoryStatus: patientData.criticalHistoryStatus || '',
        criticalHistory: patientData.criticalHistory || {},
        chiefComplaint: patientData.symptoms || 'Patient presenting with reported symptoms',
        triageNotes: composedReasoning,

        // Manual SATS from nurse
        manualSatsLevel: effectiveSatsLevel,
        urgencyScore: parseInt(urgencyScore),
        tewsProfile,
        tewsHeight: tewsHeight === '' ? null : Number(tewsHeight),
        mobility: tewsVitals.mobility,
        traumaPresent: Boolean(tewsVitals.traumaPresent),
        SATS_reasoning: {
          chart_version: 'EMSSA SATS 2015',
          profile: tewsProfile,
          height_cm: tewsHeight === '' ? null : Number(tewsHeight),
          components: tewsResult.components,
          total: tewsResult.total,
          priority: effectiveSatsLevel,
          mobility: tewsVitals.mobility,
          traumaPresent: Boolean(tewsVitals.traumaPresent),
          traumaModifier: tewsResult.components.trauma,
          trauma_modifier: tewsResult.components.trauma
        },
        triageReasoningNotes: composedReasoning,
        recommendedDepartment: department,
        immediateInterventions: immediateActions
          .split('\n')
          .filter((i) => i.trim().length > 0),
        pathophysiology,
        clinicalSummary,
        clinicalDiscriminators: clinicalDiscriminators
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean),
        differentialDiagnoses: differentialDiagnoses
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean),
        immediateActions: immediateActions
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean),
        diagnosticWorkup: diagnosticWorkup
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean),
        dispositionRationale,
        analysis: {
          pathophysiology,
          clinicalSummary,
          clinicalDiscriminators: clinicalDiscriminators
            .split('\n')
            .map((item) => item.trim())
            .filter(Boolean),
          differentialDiagnoses: differentialDiagnoses
            .split('\n')
            .map((item) => item.trim())
            .filter(Boolean),
          immediateActions: immediateActions
            .split('\n')
            .map((item) => item.trim())
            .filter(Boolean),
          diagnosticWorkup: diagnosticWorkup
            .split('\n')
            .map((item) => item.trim())
            .filter(Boolean),
          dispositionRationale
        },

        // Selected staff
        assignedNurseId: selectedNurse?.id,
        assignedNurseName: selectedNurse?.name || 'Assigned Nurse',
        assignedPhysicianId: selectedPhysician?.id,
        assignedPhysicianName: selectedPhysician?.name || 'Not Assigned',
        assignedSpecialistId: selectedSpecialist?.id,
        assignedSpecialistName: selectedSpecialist?.name || 'Not Assigned',
        assignedWardId: assignmentMeta.assignedWardId,
        assignedWardName: assignmentMeta.assignedWardName,
        bedAssigned: assignmentMeta.bedAssigned,
        assignedBedNumber: assignmentMeta.assignedBedNumber,
        selectedBed: enrichedSelectedBed,
        ward_display_info: enrichedSelectedBed ? {
          name: assignmentMeta.assignedWardName,
          wardName: assignmentMeta.assignedWardName,
          department: enrichedSelectedBed.department || assignmentMeta.assignedWardName,
          wardId: assignmentMeta.assignedWardId
        } : null,
        bed_display_info: enrichedSelectedBed ? {
          bedId: assignmentMeta.bedAssigned,
          bedNumber: assignmentMeta.assignedBedNumber,
          bed_number: assignmentMeta.assignedBedNumber,
          department: enrichedSelectedBed.department || assignmentMeta.assignedWardName,
          wardName: assignmentMeta.assignedWardName
        } : null,
        
        // Patient data (REQUIRED for display in dashboard)
        patientId: safePatientData.id_number || safePatientData.id || safePatientData._id,
        encounterId: safePatientData.encounterId || null,
        patient: {
          id_number: safePatientData.id_number || safePatientData.id || safePatientData._id || '',
          name: safePatientData.name || '',
          surname: safePatientData.surname || '',
          // Support both new (ageYears/ageMonths) and legacy (age) formats
          age: safePatientData.ageYears ?? safePatientData.age ?? null,
          ageMonths: safePatientData.ageMonths ?? null,
          height_cm: safePatientData.height_cm ?? null,
          gender: safePatientData.gender || '',
          symptoms: typeof safePatientData.symptoms === 'string'
            ? safePatientData.symptoms
            : (Array.isArray(safePatientData.symptoms) ? safePatientData.symptoms.join(", ") : ""),
          history: safePatientData.history || '',
          duration: safePatientData.duration || '',
          painLevel: safePatientData.painLevel || '',
          criticalHistoryStatus: safePatientData.criticalHistoryStatus || '',
          criticalHistory: safePatientData.criticalHistory || {}
        },
        vitals: {
          temp: parseFloat(vitalsData.temp) || null,
          hr: parseInt(vitalsData.hr) || null,
          resp: parseInt(vitalsData.resp) || null,
          bp_systolic: vitalsData.bp_systolic || null,
          bp_diastolic: vitalsData.bp_diastolic || null,
          o2: parseFloat(vitalsData.o2) || null,
          avpu: vitalsData.avpu || null,
          mobility: tewsVitals.mobility || vitalsData.mobility || 'Walks',
          traumaPresent: Boolean(tewsVitals.traumaPresent || vitalsData.traumaPresent)
        }
      };

      const response = await fetch(`${BACKEND_URL}${triageEndpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { Authorization: `Bearer ${token}` })
        },
        body: JSON.stringify(triagePayload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || `Failed to submit triage: ${response.status}`
        );
      }

      const result = await response.json();
      setSubmitSuccess(true);

      // Enhance result with staff details for confirmation page
      // Extract the actual data from the API response structure
      const apiData = result.data || result;
      const enrichedResult = {
        ...apiData,
        triageRequestId: apiData.triageRequestId,
        computeMode: apiData.computeMode,
        manualTriageReport: {
          ...apiData.manualTriageReport,
          assignedNurse: selectedNurse?.name || selectedNurse?.id || 'N/A',
          assignedPhysician: selectedPhysician?.name || selectedPhysician?.id || 'N/A',
          bedAssigned: selectedBed?.id || 'Not Assigned'
        }
      };

      // Call parent callback with enriched triage result
      if (onSubmitPayload) {
        onSubmitPayload(triagePayload);
      }
      if (onSubmit) {
        onSubmit(enrichedResult);
      }
      // Show success for 2 seconds, then clear form
      setTimeout(() => {
        setManualSatsOverride(false);
        setSatsLevel(tewsAcuityLevel);
        setUrgencyScore('50');
        setReasoning('');
        setInterventions('');
        setInterventionTags([]);
        setSelectedNurse(null);
        setSelectedPhysician(null);
        setSelectedBed(null);
        setSubmitSuccess(false);
      }, 2000);
    } catch (err) {
      console.error('Manual triage submission error:', err);
      setSubmitError(err.message || 'Failed to submit manual triage');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="manual-triage-container">
      <div className="manual-triage-form manual-triage-grid">
        <aside className="left-column">
          <div className="patient-card">
            <div className="patient-card-header">
              <div>
                <div className="patient-name">{[safePatientData.name, safePatientData.surname].filter(Boolean).join(' ') || '—'}</div>
                <div className="patient-sub">ID: {safePatientData.id_number || safePatientData.id || safePatientData._id || '—'}</div>
              </div>
              <div className="patient-meta">
                <div className="patient-age">{safePatientData.ageYears ?? safePatientData.age ?? 'N/A'} yrs</div>
                <div className="patient-gender">{safePatientData.gender || '—'}</div>
              </div>
            </div>
            <div className="patient-complaint">{safePatientData.symptoms || 'No chief complaint provided'}</div>
          </div>

          <div className="vitals-card">
            <h4>Vitals</h4>
            <div className="vitals-grid">
              <div className="vital-item"><FiThermometer className="vital-icon"/> <div className="vital-label">Temp</div><div className="vital-value">{vitalsData.temp || '--'} °C</div></div>
              <div className="vital-item"><FiHeart className="vital-icon"/> <div className="vital-label">HR</div><div className="vital-value">{vitalsData.hr || '--'} bpm</div></div>
              <div className="vital-item"><FiActivity className="vital-icon"/> <div className="vital-label">Resp</div><div className="vital-value">{vitalsData.resp || '--'}</div></div>
              <div className="vital-item"><FiClock className="vital-icon"/> <div className="vital-label">O₂ Sat</div><div className="vital-value">{vitalsData.o2 || '--'} %</div></div>
              <div className="vital-item"><FiUser className="vital-icon"/> <div className="vital-label">AVPU</div><div className="vital-value">{vitalsData.avpu || 'Alert'}</div></div>
              <div className="vital-item"><FiMove className="vital-icon" style={{ width: 18, height: 18, color: '#0ea5e9', flexShrink: 0 }} aria-hidden="true"/><div className="vital-label">Mobility</div><div className="vital-value">{tewsVitals.mobility || 'Walks'} <small>(+{tewsResult.components.mobility})</small></div></div>
              <div className="vital-item"><FiAlertTriangle className="vital-icon" style={{ width: 18, height: 18, color: '#0ea5e9', flexShrink: 0 }} aria-hidden="true"/><div className="vital-label">Trauma</div><div className="vital-value">{tewsVitals.traumaPresent ? 'Yes' : 'No'} <small>(+{tewsResult.components.trauma})</small></div></div>

            </div>
          </div>

          <div
            className="vitals-card ehr-context-card"
            role={ehrSummary?.lastEncounter && selectedMrn ? 'button' : undefined}
            tabIndex={ehrSummary?.lastEncounter && selectedMrn ? 0 : undefined}
            style={{ cursor: ehrSummary?.lastEncounter && selectedMrn ? 'pointer' : 'default' }}
            onClick={() => {
              if (ehrSummary?.lastEncounter && selectedMrn) {
                navigate(`/ehr?patient=${encodeURIComponent(selectedMrn)}`);
              }
            }}
            onKeyDown={(event) => {
              if (ehrSummary?.lastEncounter && selectedMrn && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault();
                navigate(`/ehr?patient=${encodeURIComponent(selectedMrn)}`);
              }
            }}
            aria-label={ehrSummary?.lastEncounter && selectedMrn ? 'Open patient EHR history' : undefined}
          >
            <h4>View Patient History</h4>
            {hasPatientHistoryLoaded ? (
              <div className="ehr-context-details">Open the patient history in the EHR library.</div>
            ) : shouldShowNoHistoryMessage ? (
              <div className="ehr-context-empty">No prior hospital encounter loaded from EHR.</div>
            ) : null}
          </div>
        </aside>

        <section className="right-column">
          <div className="form-header">
            <h2>Manual Triage Entry (SATS)</h2>
            <p className="form-subtitle">South African Triage Scale - Nurse-led decision record. Complete required fields and submit for clinical handover.</p>
          </div>

          <div className="form-body">
            <div className="stepper">
              {steps.map((s, idx) => (
                <button
                  key={s}
                  className={`step-pill ${idx === currentStep ? 'active' : ''}`}
                  onClick={() => {
                    if (s === 'Analysis') {
                      setCurrentStep(idx);
                      setShowAnalysisModal(true);
                    } else {
                      setCurrentStep(idx);
                    }
                  }}
                  aria-current={idx===currentStep}
                >
                  {idx+1}. {s}
                </button>
              ))}
            </div>

            <div className="step-content">
              {currentStep === 0 && (
                <div className="step-panel">
                  <div className="form-row">
                    <div className="form-group">
                      <label className="field-label"><strong>SATS Level</strong></label>
                      <div className="esi-grid compact">
                        {['red','orange','yellow','green'].map(l=> (
                          <button
                            key={l}
                            className={`sats-button esi-button ${effectiveSatsLevel===l ? 'selected' : ''}`}
                            style={{backgroundColor:getSATSColor(l)}}
                            onClick={() => {
                              setManualSatsOverride(true);
                              setSatsLevel(l);
                            }}
                          >
                            {l.toUpperCase()}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="tews-panel tews-summary-panel">
                      <div className="tews-panel-header">
                        <div>
                          <strong>TEWS calculator</strong>
                          <span>EMSSA SATS 2015 chart profile</span>
                        </div>
                        <div className="tews-total">{tewsResult.total} <small>points</small></div>
                      </div>
                      <div className="tews-components">{Object.entries(tewsResult.components).map(([key, value]) => <span key={key}>{key}: <strong>{value}</strong></span>)}</div>
                      <button className="toggle-button" onClick={() => setShowTewsModal(true)}>Open TEWS Calculator</button>
                    </div>
                    <div className="form-group narrow-field">
                      <label className="field-label"><strong>Acuity</strong></label>
                      <div
                        className="acuity-display"
                        style={{
                          padding: '10px 12px',
                          background: '#f8fafc',
                          borderRadius: 8,
                          border: '1px solid #e2e8f0',
                          fontWeight: 700,
                          textAlign: 'center',
                          color: tewsAcuityLevel === 'red'
                            ? '#dc2626'
                            : tewsAcuityLevel === 'orange'
                              ? '#ea580c'
                              : tewsAcuityLevel === 'yellow'
                                ? '#a16207'
                                : '#15803d'
                        }}
                      >
                        {tewsAcuityLevel.toUpperCase()} · {tewsAcuityLabel}
                      </div>
                      <div
                        style={{
                          marginTop: 6,
                          fontSize: 11,
                          color: tewsAcuityLevel === 'red'
                            ? '#dc2626'
                            : tewsAcuityLevel === 'orange'
                              ? '#ea580c'
                              : tewsAcuityLevel === 'yellow'
                                ? '#a16207'
                                : '#15803d',
                          textAlign: 'center',
                          fontWeight: 700
                        }}
                      >
                        {tewsProfile === 'adult' ? 'Adult SATS scale' : 'Pediatric SATS scale'}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {currentStep === 1 && (
                <div className="step-panel">
                  <h4>Clinical Analysis</h4>
                  <button className="toggle-button" onClick={() => setShowAnalysisModal(true)}>Open Analysis Modal</button>
                </div>
              )}

              {currentStep === 2 && (
                <div className="step-panel">
                  <label className="field-label">Assign Clinicians & Wards</label>
                  <button className="toggle-button" onClick={()=>setShowStaffModal(true)}>Open Clinician & Ward Assignment</button>
                </div>
              )}

              {showTewsModal && (
                <div className="modal-overlay" role="dialog" aria-modal="true">
                  <div className="modal-content tews-modal-content">
                    <button className="modal-close" onClick={()=>setShowTewsModal(false)}>✕</button>
                    <h3>TEWS Calculator</h3>
                    <div className="tews-panel">
                      <div className="structured-row">
                        <select className="field-input" value={tewsProfile} onChange={(e)=>setTewsProfile(e.target.value)}>
                          <option value="adult">Adult (&gt;12 years)</option>
                          <option value="child-younger">Child 50-95 cm (approximately 1 week to &lt;3 years)</option>
                          <option value="child-older">Child 96-150 cm (approximately 3-12 years)</option>
                        </select>
                        <input className="field-input" type="number" min="30" max="220" value={tewsHeight} onChange={(e)=>setTewsHeight(e.target.value)} placeholder="Height (cm)" />
                      </div>
                      <div className="tews-input-grid">
                        <label>Temp<input className="field-input" type="number" step="0.1" value={tewsVitals.temperature} onChange={(e)=>setTewsVitals({...tewsVitals, temperature:e.target.value})} /></label>
                        <label>Heart rate<input className="field-input" type="number" value={tewsVitals.heartRate} onChange={(e)=>setTewsVitals({...tewsVitals, heartRate:e.target.value})} /></label>
                        <label>Respiratory rate<input className="field-input" type="number" value={tewsVitals.respiratoryRate} onChange={(e)=>setTewsVitals({...tewsVitals, respiratoryRate:e.target.value})} /></label>
                        <label>SBP<input className="field-input" type="number" value={tewsVitals.systolicBP} onChange={(e)=>setTewsVitals({...tewsVitals, systolicBP:e.target.value})} /></label>
                        <label>AVPU<select className="field-input" value={tewsVitals.avpu} onChange={(e)=>setTewsVitals({...tewsVitals, avpu:e.target.value})}><option>Alert</option><option>Voice</option><option>Pain</option><option>Unresponsive</option></select></label>
                        <label>Mobility<select className="field-input" value={tewsVitals.mobility} onChange={(e)=>setTewsVitals({...tewsVitals, mobility:e.target.value})}><option value="Walks">Walks independently (+0)</option><option value="With help">Walks with help (+1)</option><option value="Immobile">Stretcher / immobile (+2)</option></select></label>
                        <label className="tews-trauma-toggle">Trauma present<input className="field-input" type="checkbox" checked={Boolean(tewsVitals.traumaPresent)} onChange={(e)=>setTewsVitals({...tewsVitals, traumaPresent:e.target.checked})} /><span>{tewsVitals.traumaPresent ? 'Yes (+1)' : 'No (+0)'}</span></label>
                      </div>
                      <div className="tews-components">{Object.entries(tewsResult.components).map(([key, value]) => <span key={key}>{key}: <strong>{value}</strong></span>)}</div>
                      <div className="tews-note">TEWS supports, but does not replace, clinical discriminators and nurse judgment. Verify the selected profile, units, and values before submission.</div>
                    </div>
                    <button className="submit-button" onClick={()=>setShowTewsModal(false)}>Save and Close</button>
                  </div>
                </div>
              )}

              {showAnalysisModal && (
                <div className="modal-overlay" role="dialog" aria-modal="true">
                  <div className="modal-content tews-modal-content">
                    <button className="modal-close" onClick={()=>setShowAnalysisModal(false)}>✕</button>
                    <h3>Clinical Analysis</h3>
                    <div className="form-group">
                      <label className="field-label">Pathophysiology</label>
                      <textarea className="field-input" rows="4" value={pathophysiology} onChange={(e)=>updateAnalysisField('pathophysiology', setPathophysiology, e.target.value)} placeholder="Describe underlying pathophysiology" />
                    </div>
                    <div className="form-group">
                      <label className="field-label">Clinical Summary</label>
                      <textarea className="field-input" rows="4" value={clinicalSummary} onChange={(e)=>updateAnalysisField('clinicalSummary', setClinicalSummary, e.target.value)} placeholder="Summarize the clinical picture" />
                    </div>
                    <div className="form-group">
                      <label className="field-label">Clinical Discriminators</label>
                      <textarea className="field-input" rows="3" value={clinicalDiscriminators} onChange={(e)=>updateAnalysisField('clinicalDiscriminators', setClinicalDiscriminators, e.target.value)} placeholder="List discriminators or red flags" />
                    </div>
                    <div className="form-group">
                      <label className="field-label">Differential Diagnoses</label>
                      <textarea className="field-input" rows="3" value={differentialDiagnoses} onChange={(e)=>updateAnalysisField('differentialDiagnoses', setDifferentialDiagnoses, e.target.value)} placeholder="List possible diagnoses" />
                    </div>
                    <div className="form-group">
                      <label className="field-label">Immediate Actions</label>
                      <textarea className="field-input" rows="3" value={immediateActions} onChange={(e)=>updateAnalysisField('immediateActions', setImmediateActions, e.target.value)} placeholder="Document immediate life-saving actions" />
                    </div>
                    <div className="form-group">
                      <label className="field-label">Diagnostic Workup</label>
                      <textarea className="field-input" rows="3" value={diagnosticWorkup} onChange={(e)=>updateAnalysisField('diagnosticWorkup', setDiagnosticWorkup, e.target.value)} placeholder="List tests or imaging needed" />
                    </div>
                    <div className="form-group">
                      <label className="field-label">Disposition Rationale</label>
                      <textarea className="field-input" rows="3" value={dispositionRationale} onChange={(e)=>updateAnalysisField('dispositionRationale', setDispositionRationale, e.target.value)} placeholder="Explain care disposition / next step rationale" />
                    </div>
                    <div className="step-buttons" style={{ marginTop: 12, justifyContent: 'flex-end' }}>
                      <button className="submit-button" onClick={()=>setShowAnalysisModal(false)}>Save and Close</button>
                    </div>
                  </div>
                </div>
              )}

              {showStaffModal && (
                <div className="modal-overlay staff-assignment-overlay" role="dialog" aria-modal="true">
                  <div className="modal-content tews-modal-content staff-assignment-modal">
                    <button className="modal-close" onClick={()=>setShowStaffModal(false)}>✕</button>
                    <div className="staff-assignment-header">
                      <span>Care team & placement</span>
                      <h3>Select staff and bed</h3>
                      <p>Review each clinician’s current workload and choose available downstream coverage.</p>
                    </div>
                    <StaffScheduleDisplay onStaffSelect={handleStaffSelect} />
                    <div className="step-buttons staff-assignment-footer" style={{ marginTop: 16, justifyContent: 'flex-end' }}>
                      <button className="submit-button" type="button" onClick={()=>setShowStaffModal(false)}>
                        Save and Close
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="step-buttons manual-navigation-actions">
              <button className="toggle-button manual-navigation-button" onClick={()=>setCurrentStep(Math.max(0,currentStep-1))} disabled={currentStep===0}>◀ Prev</button>
              {currentStep < steps.length-1 ? (
                <button className="submit-button manual-navigation-button" onClick={()=>setCurrentStep(Math.min(steps.length-1,currentStep+1))}>Next ▶</button>
              ) : (
                <button className="submit-button manual-navigation-button" onClick={handleSubmitTriage} disabled={!isFormValid()||submitting}>{submitting? 'Submitting...':'Submit Manual Triage'}</button>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

// Helper functions for ESI color coding
function getSATSColor(level) {
  const colors = {
    'red': '#ef4444',      // Red - Emergency/Immediate
    'orange': '#f97316',   // Orange - Very Urgent
    'yellow': '#eab308',   // Yellow - Urgent
    'green': '#22c55e'     // Green - Routine/Non-urgent
  };
  return colors[level] || '#cbd5e1';
}

function getSATSName(level) {
  const names = {
    'red': 'Emergency',
    'yellow': 'Urgent',
    'green': 'Routine'
  };
  return names[level];
}

function getSATSDescription(level) {
  const descriptions = {
    'red': 'Life-threatening condition requiring immediate intervention',
    'yellow': 'Urgent condition requiring prompt care',
    'green': 'Non-urgent condition, stable patient'
  };
  return descriptions[level];
}

// Styles
const containerStyle = {
  padding: '24px',
  backgroundColor: '#ffffff',
  borderRadius: '8px',
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
};

const formHeaderStyle = {
  marginBottom: '24px',
  borderBottom: '2px solid #e2e8f0',
  paddingBottom: '16px'
};

const formSubtitleStyle = {
  margin: '8px 0 0 0',
  fontSize: '14px',
  color: '#64748b'
};

const sectionHeadingStyle = {
  marginBottom: '12px',
  fontSize: '14px',
  fontWeight: '600',
  color: '#1e293b',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const patientSummaryStyle = {
  padding: '16px',
  backgroundColor: '#f0f9ff',
  borderRadius: '6px',
  marginBottom: '20px',
  borderLeft: '4px solid #0284c7'
};

const patientInfoGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
  gap: '16px'
};

const vitalsBoxStyle = {
  padding: '16px',
  backgroundColor: '#fef3c7',
  borderRadius: '6px',
  marginBottom: '20px',
  borderLeft: '4px solid #f59e0b'
};

const vitalsGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
  gap: '12px'
};

const vitalItemStyle = {
  padding: '12px',
  backgroundColor: 'white',
  borderRadius: '4px',
  textAlign: 'center'
};

const vitalLabelStyle = {
  fontSize: '12px',
  color: '#64748b',
  fontWeight: '500',
  marginBottom: '4px'
};

const vitalValueStyle = {
  fontSize: '18px',
  fontWeight: '700',
  color: '#1e293b'
};

const valueStyle = {
  fontSize: '14px',
  fontWeight: '600',
  color: '#1e293b',
  marginLeft: '8px'
};

const formGroupStyle = {
  marginBottom: '20px'
};

const labelStyle = {
  display: 'block',
  marginBottom: '8px',
  fontSize: '14px',
  color: '#1e293b',
  fontWeight: '500'
};

const requiredStyle = {
  color: '#dc2626',
  marginLeft: '4px'
};

const esiGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(5, 1fr)',
  gap: '12px',
  marginBottom: '12px'
};

const esiButtonStyle = {
  padding: '12px 8px',
  border: '2px solid #e2e8f0',
  borderRadius: '6px',
  cursor: 'pointer',
  color: 'white',
  fontWeight: '600',
  transition: 'all 0.2s ease',
  minHeight: '80px',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '4px'
};

const esiSelectedStyle = {
  borderColor: '#1e293b',
  borderWidth: '3px',
  boxShadow: '0 0 0 3px rgba(0,0,0,0.1)'
};

const esiLevelNumberStyle = {
  fontSize: '24px',
  fontWeight: '700'
};

const esiLevelNameStyle = {
  fontSize: '11px',
  fontWeight: '500'
};

const esiDescriptionStyle = {
  fontSize: '13px',
  color: '#64748b',
  padding: '8px 12px',
  backgroundColor: '#f1f5f9',
  borderRadius: '4px'
};

const sliderStyle = {
  width: '100%',
  height: '6px',
  marginBottom: '8px',
  cursor: 'pointer'
};

const acuityDisplayStyle = {
  fontSize: '24px',
  fontWeight: '700',
  color: '#2563eb',
  textAlign: 'center'
};

const textareaStyle = {
  width: '100%',
  padding: '14px',
  fontSize: '15px',
  border: '1px solid #e6eefc',
  borderRadius: '12px',
  fontFamily: 'inherit',
  resize: 'vertical',
  boxSizing: 'border-box',
  background: 'linear-gradient(180deg, #fff 0%, #fbfdff 100%)',
  boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.6), 0 6px 18px rgba(16,24,40,0.04)',
  transition: 'box-shadow 150ms ease, border-color 150ms ease'
};

const charCountStyle = {
  fontSize: '12px',
  color: '#64748b',
  marginTop: '6px',
  textAlign: 'right'
};

const smallToolbarButton = {
  padding: '8px 12px',
  borderRadius: 8,
  border: '1px solid rgba(14,165,233,0.12)',
  background: 'linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,252,255,0.9))',
  cursor: 'pointer',
  fontSize: 13,
  fontWeight: 700,
  color: '#0f172a',
  boxShadow: '0 6px 12px rgba(14,165,233,0.06)'
};

const selectStyle = {
  width: '100%',
  padding: '10px 12px',
  fontSize: '14px',
  border: '1px solid #cbd5e1',
  borderRadius: '4px',
  backgroundColor: 'white',
  cursor: 'pointer'
};

const staffSectionStyle = {
  marginBottom: '20px',
  padding: '16px',
  backgroundColor: '#f8fafc',
  borderRadius: '6px',
  border: '2px solid #e2e8f0'
};

const staffHeaderStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: '16px'
};

const toggleButtonStyle = {
  padding: '8px 12px',
  backgroundColor: '#e2e8f0',
  color: '#475569',
  border: '1px solid #cbd5e1',
  borderRadius: '4px',
  cursor: 'pointer',
  fontSize: '13px',
  fontWeight: '500',
  transition: 'all 0.2s ease'
};

const toggleButtonActiveStyle = {
  backgroundColor: '#2563eb',
  color: 'white',
  borderColor: '#2563eb'
};

const selectedStaffStyle = {
  padding: '16px',
  backgroundColor: 'white',
  borderRadius: '6px',
  marginTop: '16px'
};

const assignmentRowStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
  gap: '16px'
};

const assignmentItemStyle = {
  padding: '12px',
  backgroundColor: '#f1f5f9',
  borderRadius: '4px',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center'
};

const assignmentLabelStyle = {
  fontSize: '13px',
  fontWeight: '600',
  color: '#475569'
};

const assignmentValueStyle = {
  fontSize: '14px',
  fontWeight: '600',
  color: '#1e293b'
};

const checkmarkStyle = {
  color: '#10b981',
  fontSize: '18px',
  fontWeight: '700'
};

const submitButtonStyle = {
  width: '100%',
  padding: '14px 20px',
  fontSize: '16px',
  fontWeight: '600',
  color: 'white',
  backgroundColor: '#2563eb',
  border: 'none',
  borderRadius: '6px',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  marginTop: '12px'
};

const disabledButtonStyle = {
  backgroundColor: '#cbd5e1',
  cursor: 'not-allowed',
  opacity: 0.6
};

const errorStyle = {
  padding: '12px 16px',
  backgroundColor: '#fee2e2',
  color: '#991b1b',
  borderRadius: '6px',
  marginBottom: '12px',
  fontSize: '14px',
  border: '1px solid #fecaca'
};

const successStyle = {
  padding: '12px 16px',
  backgroundColor: '#dcfce7',
  color: '#166534',
  borderRadius: '6px',
  marginBottom: '12px',
  fontSize: '14px',
  border: '1px solid #bbf7d0',
  fontWeight: '500'
};

const helpTextStyle = {
  marginTop: '16px',
  padding: '12px',
  backgroundColor: '#f0f9ff',
  borderRadius: '4px',
  fontSize: '13px',
  color: '#0c4a6e',
  borderLeft: '3px solid #0284c7'
};

export default ManualTriageForm;
