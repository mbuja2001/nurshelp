import express from 'express';
import { authenticateToken } from '../middleware/auth.js';
import {
  searchHospitalPatients,
  suggestHospitalPatients,
  getUserSearchHistory,
  recordUserSearchHistory,
  clearUserSearchHistory,
  getPatientEHR,
  getFHIRPatient,
  getPatientTriageReports,
  getPatientHandoverTimeline,
  getPatientNotifications,
  getPatientSummary
} from '../controllers/patient.controller.js';

const router = express.Router();

// Search for patients in hospital (by MRN, name, etc)
router.get('/search', authenticateToken, searchHospitalPatients);
router.get('/suggestions', authenticateToken, suggestHospitalPatients);
router.get('/search-history', authenticateToken, getUserSearchHistory);
router.post('/search-history', authenticateToken, recordUserSearchHistory);
router.delete('/search-history', authenticateToken, clearUserSearchHistory);

// FHIR R4 Patient representation backed by the hospital's Encounter records
router.get('/fhir/Patient/:id', authenticateToken, getFHIRPatient);

// Get complete EHR by patient MRN (hospital-isolated)
router.get('/ehr/:mrn', authenticateToken, getPatientEHR);

// Get all triage reports for a patient
router.get('/triage-reports/:mrn', authenticateToken, getPatientTriageReports);

// Get clinical handover timeline
router.get('/handover-timeline/:mrn', authenticateToken, getPatientHandoverTimeline);

// Get all notifications related to patient's encounters
router.get('/notifications/:mrn', authenticateToken, getPatientNotifications);

// Get summary stats (quick overview)
router.get('/summary/:mrn', authenticateToken, getPatientSummary);

export default router;
