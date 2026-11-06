import mongoose from 'mongoose';
import { Hospital } from './hospital.model.js';
import { AuditLog } from './auditLog.model.js';
import {
  activateClinicalTrialWithCounter,
  closeClinicalTrialWithCounter,
  reconcileClinicalTrialCount,
} from '../utils/clinicalTrialModePolicy.js';

const clinicalTrialSchema = new mongoose.Schema(
  {
    hospitalId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Hospital',
      required: true,
      index: true,
    },
    trialId: {
      type: String,
      required: true,
      trim: true,
    },
    protocolNumber: {
      type: String,
      required: true,
      trim: true,
    },
    status: {
      type: String,
      enum: ['DRAFT', 'ACTIVE', 'ON_HOLD', 'CLOSED'],
      default: 'DRAFT',
      index: true,
    },
  },
  { timestamps: true, collection: 'clinicaltrials' }
);

clinicalTrialSchema.index({ hospitalId: 1, trialId: 1 }, { unique: true });

clinicalTrialSchema.statics.activateForHospital = async function ({ trialObjectId, hospitalId }) {
  const trial = await this.findOne({ _id: trialObjectId, hospitalId }).select('status').lean();
  if (!trial) return { success: false, code: 'TRIAL_NOT_FOUND' };
  if (trial.status === 'ACTIVE') return { success: true, changed: false };
  if (!['DRAFT', 'ON_HOLD'].includes(trial.status)) {
    return { success: false, code: 'TRIAL_NOT_ACTIVATABLE' };
  }

  return activateClinicalTrialWithCounter({
    reserveHospital: () => Hospital.findOneAndUpdate(
      { _id: hospitalId, clinicalTrialMode: true },
      { $inc: { activeTrialCount: 1 } },
      { new: true, projection: { _id: 1 } }
    ),
    activateTrial: () => this.findOneAndUpdate(
      { _id: trialObjectId, hospitalId, status: trial.status },
      { $set: { status: 'ACTIVE' } },
      { new: true }
    ),
    decrementHospital: async () => {
      const decrement = await Hospital.updateOne(
        { _id: hospitalId, activeTrialCount: { $gt: 0 } },
        { $inc: { activeTrialCount: -1 } }
      );
      return decrement.modifiedCount === 1;
    },
  });
};

clinicalTrialSchema.statics.closeForHospital = async function ({ trialObjectId, hospitalId }) {
  const result = await closeClinicalTrialWithCounter({
    closeTrial: () => this.findOneAndUpdate(
      { _id: trialObjectId, hospitalId, status: 'ACTIVE' },
      { $set: { status: 'CLOSED' } },
      { new: true }
    ),
    decrementHospital: async () => {
      const decrement = await Hospital.updateOne(
        { _id: hospitalId, activeTrialCount: { $gt: 0 } },
        { $inc: { activeTrialCount: -1 } }
      );
      return decrement.modifiedCount === 1;
    },
  });
  if (!result.changed) {
    const trial = await this.findOne({ _id: trialObjectId, hospitalId }).select('_id').lean();
    return trial
      ? result
      : { success: false, code: 'TRIAL_NOT_FOUND' };
  }
  return result;
};

clinicalTrialSchema.statics.reconcileHospitalActiveTrialCount = async function ({
  hospitalId,
  actorId,
  actorName,
}) {
  const [hospital, activeTrialActualCount] = await Promise.all([
    Hospital.findById(hospitalId).select('activeTrialCount clinicalTrialMode').lean(),
    this.countDocuments({ hospitalId, status: 'ACTIVE' }),
  ]);
  if (!hospital) return { success: false, code: 'HOSPITAL_NOT_FOUND' };

  const activeTrialCount = hospital.activeTrialCount || 0;
  return reconcileClinicalTrialCount({
    activeTrialCount,
    activeTrialActualCount,
    recordMismatch: async (counts) => {
      const at = new Date();
      const reason = 'Reconciliation found a mismatch between the hospital counter and active trials';
      const event = {
        eventType: 'TRIAL_COUNT_MISMATCH',
        reason,
        by: actorId,
        byName: actorName,
        at,
        details: counts,
      };
      const update = await Hospital.updateOne(
        {
          _id: hospitalId,
          $or: [
            { activeTrialCount },
            ...(activeTrialCount === 0 ? [{ activeTrialCount: { $exists: false } }] : []),
          ],
        },
        { $push: { clinicalTrialModeHistory: event } }
      );
      if (update.modifiedCount !== 1) {
        throw new Error('Active-trial count changed during reconciliation; retry the check');
      }

      try {
        await AuditLog.create({
          action: 'CLINICAL_TRIAL_COUNT_MISMATCH',
          actorId,
          actorIdentityType: 'ACCOUNT_OWNER',
          hospitalId,
          resourceType: 'HOSPITAL',
          authorized: true,
          reason,
          changeDetails: counts,
          description: `Active trial count mismatch: stored ${counts.activeTrialCount}, actual ${counts.activeTrialActualCount}`,
        });
      } catch (error) {
        console.error('[AUDIT LOG] Embedded reconciliation history was saved, but the secondary audit copy failed:', error);
      }
    },
  });
};

export const ClinicalTrial = mongoose.model('ClinicalTrial', clinicalTrialSchema);
