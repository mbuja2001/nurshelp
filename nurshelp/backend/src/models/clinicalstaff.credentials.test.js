import assert from 'node:assert/strict';
import mongoose from 'mongoose';
import { describe, it } from 'node:test';
import { ClinicalStaff } from './clinicalstaff.model.js';

const staffRecord = (systemRole, credentials = {}) => new ClinicalStaff({
  hospitalId: new mongoose.Types.ObjectId(),
  firstName: 'QA',
  lastName: 'Staff',
  email: `${systemRole.toLowerCase()}@example.test`,
  password: 'hashed-password',
  systemRole,
  clinicalSpecialty: 'Radiology',
  credentials,
  assignment: { department: 'RADIOLOGY' }
});

describe('ClinicalStaff professional credential requirements', () => {
  it('allows technician records without professional license fields', async () => {
    await assert.doesNotReject(staffRecord('TECHNICIAN').validate());
  });

  it('allows support positions without clinical credentials in specialty departments', async () => {
    const staff = staffRecord('SUPPORT');
    staff.positionTitle = 'Patient transporter';
    staff.assignment.department = 'Cardiology';
    await assert.doesNotReject(staff.validate());
  });

  it('continues requiring title and license for non-technician clinical staff', async () => {
    const staff = staffRecord('RADIOLOGIST');
    await assert.rejects(staff.validate(), (error) => {
      assert.ok(error.errors['credentials.title']);
      assert.ok(error.errors['credentials.licenseNumber']);
      return true;
    });
  });
});
