import { Hospital } from '../models/hospital.model.js';
import { evaluateClinicalTrialModeGate } from '../utils/clinicalTrialModePolicy.js';

export const requireClinicalTrialMode = async (req, res, next) => {
  try {
    const hospital = req.hospitalId
      ? await Hospital.findById(req.hospitalId).select('clinicalTrialMode environment').lean()
      : null;
    const decision = evaluateClinicalTrialModeGate({ hospitalId: req.hospitalId, hospital });
    if (!decision.allowed) {
      return res.status(decision.code === 'HOSPITAL_NOT_FOUND' ? 404 : 403).json({
        success: false,
        code: decision.code,
        message: decision.code === 'HOSPITAL_NOT_FOUND'
          ? 'Hospital could not be resolved from the authenticated session'
          : 'Clinical trial mode is not enabled for this hospital',
      });
    }

    req.clinicalTrialHospital = hospital;
    return next();
  } catch (error) {
    console.error('[CLINICAL TRIAL MODE] Gate check failed:', error);
    return res.status(500).json({
      success: false,
      message: 'Unable to verify clinical trial mode',
    });
  }
};
