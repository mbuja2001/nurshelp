import express from 'express';
import {
  addAccountHospitalAdmin,
  createAccountHospital,
  listAccountHospitals,
  reconcileAccountHospitalTrialCount,
  resetAccountHospitalAdminPassword,
  updateAccountHospitalEhrAccess,
  updateAccountHospitalTrialMode
} from '../controllers/accountHospital.controller.js';
import { protectSandboxAccount } from '../middleware/accountAuth.js';

const router = express.Router();

router.use(protectSandboxAccount);
router.get('/', listAccountHospitals);
router.post('/', createAccountHospital);
router.patch('/:hospitalId/clinical-trial-mode', updateAccountHospitalTrialMode);
router.post('/:hospitalId/clinical-trial-mode/reconcile', reconcileAccountHospitalTrialCount);
router.patch('/:hospitalId/ehr-integration-access', updateAccountHospitalEhrAccess);
router.post('/:hospitalId/admins', addAccountHospitalAdmin);
router.post('/:hospitalId/admins/:adminId/reset-password', resetAccountHospitalAdminPassword);

export default router;