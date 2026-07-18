/**
 * Dashboard Controller
 * Handles dashboard configuration and role-based rendering for Full Day Simulation
 * 
 * KEY LOGIC:
 * 1. Decode JWT token to get authenticated staff member's ID
 * 2. Find today's shift for that hospital
 * 3. Check which role they have in shift.teamAssignment (triageNurseId, clinicalNurseId, physicianId)
 * 4. Return dashboard config based on SHIFT-ASSIGNED role, not general staff role
 * 
 * This enforces shift-based access control: You only see dashboards for roles assigned to you TODAY
 */

import { Shift } from '../models/shift.model.js';
import { ClinicalStaff } from '../models/clinicalstaff.model.js';
import mongoose from 'mongoose';

// ============ GET DASHBOARD CONFIGURATION ============
// Returns role-based dashboard configuration based on TODAY's shift assignment
// CRITICAL: Role determined by checking teamAssignment, not staff member's general role
export const getDashboardConfig = async (req, res) => {
  try {
    const userId = req.user?.id;
    const hospitalId = req.hospitalId;

    if (!userId || !hospitalId) {
      return res.status(401).json({
        success: false,
        message: 'User not authenticated or hospital not set'
      });
    }

    console.log('\n[GET DASHBOARD CONFIG]');
    console.log('  User ID:', userId);
    console.log('  Hospital ID:', hospitalId);

    // 1. Get the staff member's details
    const staff = await ClinicalStaff.findById(userId);
    if (!staff) {
      return res.status(404).json({
        success: false,
        message: 'Staff member not found'
      });
    }

    console.log(`  Staff: ${staff.name} ${staff.surname}`);
    console.log(`  Staff Role (general): ${staff.role}`);

    // 2. Get staff's assigned shift for TODAY (find shift where user is in teamAssignment)
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    // Find shift for today where user is assigned to a team role
    const userIdString = userId.toString();
    const shift = await Shift.findOne({
      hospitalId,
      date: { $gte: today, $lt: tomorrow },
      $or: [
        { 'teamAssignment.triageNurseId': userId },
        { 'teamAssignment.clinicalNurseId': userId },
        { 'teamAssignment.physicianId': userId }
      ]
    });

    console.log(`  Today's Shift (where user assigned): ${shift ? 'FOUND' : 'NOT FOUND'}`);

    // 3. If no shift exists for today, show OFF_DUTY
    if (!shift) {
      console.log('  → No shift today, showing OFF_DUTY dashboard');
      return res.status(200).json({
        success: true,
        config: {
          userId,
          hospitalId,
          staffName: `${staff.name} ${staff.surname}`,
          staffGeneralRole: staff.role,
          staffDepartment: staff.department,
          hasShiftToday: false,
          shiftId: null,
          assignedRole: null,
          context: 'NO_SHIFT',
          components: ['OffDutyMessage', 'UpcomingShifts'],
          controls: {
            clockIn: false,
            viewPatients: false,
            displayTriage: false,
            manageTeam: false
          },
          permissions: {
            clockIn: false,
            viewPatients: false,
            displayTriage: false,
            manageTeam: false,
            assignPhysician: false,
            disposition: false
          }
        }
      });
    }

    console.log(`  Shift ID: ${shift._id}`);
    console.log(`  Shift Name: ${shift.shiftName}`);
    console.log(`  Shift Status: ${shift.status}`);

    // 4. Check which ROLE user has in today's shift (CRITICAL LOGIC)
    let assignedRole = null;
    let dashboardContext = null;

    const triageNurseId = shift.teamAssignment?.triageNurseId?.toString();
    const clinicalNurseId = shift.teamAssignment?.clinicalNurseId?.toString();
    const physicianId = shift.teamAssignment?.physicianId?.toString();

    console.log('  Checking team assignment...');
    console.log(`    Triage Nurse ID: ${triageNurseId}`);
    console.log(`    Clinical Nurse ID: ${clinicalNurseId}`);
    console.log(`    Physician ID: ${physicianId}`);
    console.log(`    User ID: ${userIdString}`);

    if (userIdString === triageNurseId) {
      assignedRole = 'TRIAGE_NURSE';
      dashboardContext = 'TRIAGE_NURSE';
      console.log('  ✓ Assigned as: TRIAGE_NURSE');
    } else if (userIdString === clinicalNurseId) {
      assignedRole = 'CLINICAL_NURSE';
      const context = shift.teamAssignment?.clinicalNurseContext || 'CLINICAL_NURSE_ER';
      dashboardContext = context;
      console.log(`  ✓ Assigned as: CLINICAL_NURSE (${context})`);
    } else if (userIdString === physicianId) {
      assignedRole = 'PHYSICIAN';
      const context = shift.teamAssignment?.physicianContext || 'PHYSICIAN_ER';
      dashboardContext = context;
      console.log(`  ✓ Assigned as: PHYSICIAN (${context})`);
    } else {
      console.log('  ✗ User NOT assigned to any role in today\'s shift');
      return res.status(403).json({
        success: false,
        message: 'You are not assigned to any role in today\'s shift. Contact your manager.',
        shift: {
          shiftName: shift.shiftName,
          shiftStart: shift.shiftStart,
          shiftEnd: shift.shiftEnd,
          status: shift.status
        }
      });
    }

    // 5. Build dashboard config based on assigned role
    let dashboardConfig = {
      userId,
      hospitalId,
      staffName: `${staff.name} ${staff.surname}`,
      staffGeneralRole: staff.role, // NURSE or PHYSICIAN
      staffDepartment: staff.department,
      hasShiftToday: true,
      shiftId: shift._id.toString(),
      assignedRole, // TRIAGE_NURSE, CLINICAL_NURSE, PHYSICIAN
      context: dashboardContext,
      components: [],
      controls: {},
      permissions: {},
      shiftStartTime: shift.shiftStart,
      shiftEndTime: shift.shiftEnd,
      shiftName: shift.shiftName,
      department: shift.department,
      shiftStatus: shift.status
    };

    // ============ ROLE-SPECIFIC DASHBOARD CONFIG ============

    if (dashboardContext === 'TRIAGE_NURSE') {
      dashboardConfig.components = [
        'TriageHeader',
        'PatientQueue',
        'TriageForm',
        'VitalsCapture'
      ];
      dashboardConfig.controls = {
        acceptPatient: true,
        triagePatient: true,
        routeToPhysician: true,
        recordVitals: true
      };
      dashboardConfig.permissions = {
        clockIn: true,
        viewPatients: true,
        displayTriage: true,
        manageTeam: false,
        assignPhysician: false,
        disposition: false
      };
    } else if (dashboardContext.startsWith('CLINICAL_NURSE')) {
      dashboardConfig.components = [
        'ClinicalNurseHeader',
        'ObservationPatientList',
        'VitalsMonitoring',
        'MedicationLog',
        'PhysicianNotifications'
      ];
      dashboardConfig.controls = {
        recordVitals: true,
        updateMedications: true,
        markForDisposition: true,
        escalateConcerns: true,
        viewPhysicianNotes: true
      };
      dashboardConfig.permissions = {
        clockIn: true,
        viewPatients: true,
        displayTriage: false,
        manageTeam: false,
        assignPhysician: false,
        disposition: false
      };
    } else if (dashboardContext.startsWith('PHYSICIAN')) {
      dashboardConfig.components = [
        'PhysicianHeader',
        'TriagePatientLane',
        'ActiveCasesLane',
        'ObservationPatientLane',
        'PatientDetails',
        'DispositionPanel'
      ];
      dashboardConfig.controls = {
        viewTriageQueue: true,
        assessPatient: true,
        orderTests: true,
        recordDiagnosis: true,
        setDisposition: true,
        admitToWard: true,
        prescribeMedications: true,
        referToSpecialty: true
      };
      dashboardConfig.permissions = {
        clockIn: true,
        viewPatients: true,
        displayTriage: false,
        manageTeam: false,
        assignPhysician: false,
        disposition: true // ONLY physicians can set disposition
      };
    }

    console.log(`  ✓ Dashboard config generated for: ${dashboardContext}`);

    res.status(200).json({
      success: true,
      message: 'Dashboard configuration retrieved',
      config: dashboardConfig
    });
  } catch (err) {
    console.error('[GET DASHBOARD CONFIG ERROR]', err);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve dashboard configuration',
      error: err.message
    });
  }
};

// ============ GET ROLE-BASED PERMISSIONS ============
// Returns permission matrix for all roles (useful for frontend availability checks)
export const getRolePermissions = async (req, res) => {
  try {
    const permissions = {
      TRIAGE_NURSE: {
        description: 'Emergency Triage Nurse - First point of contact',
        capabilities: [
          'accept_patient',
          'perform_triage',
          'record_vitals',
          'route_to_physician',
          'view_patient_queue'
        ],
        restrictions: [
          'cannot_prescribe',
          'cannot_admit',
          'cannot_set_disposition'
        ],
        dashboardLanes: ['PATIENT_QUEUE']
      },
      CLINICAL_NURSE_ER: {
        description: 'Clinical Nurse - Emergency Department',
        capabilities: [
          'view_assigned_patients',
          'record_vitals_continuous',
          'update_medications',
          'escalate_to_physician',
          'view_physician_notes'
        ],
        restrictions: [
          'cannot_prescribe',
          'cannot_admit',
          'cannot_set_disposition'
        ],
        dashboardLanes: ['OBSERVATION_PATIENTS', 'VITALS_MONITORING']
      },
      CLINICAL_NURSE_ICU: {
        description: 'Clinical Nurse - ICU Ward',
        capabilities: [
          'view_assigned_patients',
          'record_vitals_continuous',
          'update_medications',
          'escalate_to_physician',
          'manage_beds',
          'monitor_alarms'
        ],
        restrictions: [
          'cannot_prescribe',
          'cannot_admit',
          'cannot_set_disposition'
        ],
        dashboardLanes: ['ICU_PATIENTS', 'VITALS_MONITORING', 'ALERTS']
      },
      PHYSICIAN_ER: {
        description: 'Physician - Emergency Department',
        capabilities: [
          'view_triage_queue',
          'assess_patient',
          'order_tests',
          'prescribe_medications',
          'admit_patient',
          'set_disposition'
        ],
        restrictions: [
          'none'
        ],
        dashboardLanes: [
          'TRIAGE_WAITING',
          'ACTIVE_ASSESSMENT',
          'OBSERVATION_PATIENTS'
        ]
      },
      PHYSICIAN_SURGICAL: {
        description: 'Physician - Surgical Ward',
        capabilities: [
          'view_surgical_patients',
          'assess_patient',
          'order_tests',
          'prescribe_medications',
          'admit_patient',
          'set_disposition'
        ],
        restrictions: [
          'cannot_work_in_other_departments'
        ],
        dashboardLanes: ['SURGICAL_QUEUE', 'ACTIVE_ASSESSMENT']
      }
    };

    res.status(200).json({
      success: true,
      message: 'Role-based permissions retrieved',
      permissions
    });
  } catch (err) {
    console.error('[GET ROLE PERMISSIONS ERROR]', err);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve role permissions',
      error: err.message
    });
  }
};

// ============ GET PATIENT QUEUE FOR CONTEXT ============
// Returns patients appropriate for a staff member's assigned role/context
export const getPatientQueueForContext = async (req, res) => {
  try {
    const userId = req.user?.id;
    const hospitalId = req.hospitalId;
    const { context } = req.query;

    if (!context) {
      return res.status(400).json({
        success: false,
        message: 'Context parameter is required'
      });
    }

    console.log('\n[GET PATIENT QUEUE]');
    console.log('  User ID:', userId);
    console.log('  Context:', context);

    // Get today's shift for this user
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const shift = await Shift.findOne({
      hospitalId,
      date: { $gte: today, $lt: tomorrow }
    });

    if (!shift) {
      return res.status(400).json({
        success: false,
        message: 'No shift assigned for today'
      });
    }

    // TODO: Fetch TriageRequest/Patient records based on context
    let patientQueue = {
      context,
      totalPatients: 0,
      patients: [],
      timestamp: new Date()
    };

    res.status(200).json({
      success: true,
      message: 'Patient queue retrieved for context',
      queue: patientQueue
    });
  } catch (err) {
    console.error('[GET PATIENT QUEUE ERROR]', err);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve patient queue',
      error: err.message
    });
  }
};
