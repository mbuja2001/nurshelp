import React from 'react';
import PhysicianDashboard from './PhysicianDashboard';
import SidebarLayout from '../../Dashboard/SidebarLayout';

const CLINICAL_NURSE_FEATURES = {
  diagnose: false,
  recordCare: true,
  adjustSeverity: false,
  admit: false,
  orderMedication: false,
  requestLabs: false,
  administerMedication: false,
  manageSurgery: false,
  discharge: false,
  transferCare: false
};

const ClinicalNurseDashboard = ({ config = {}, context = 'ER', embedded = false }) => {
  const dashboard = (
    <PhysicianDashboard
      config={config}
      context={context}
      features={CLINICAL_NURSE_FEATURES}
      dashboardTitle={`Clinical Nurse • ${context}`}
      staffRoleLabel="CLINICAL NURSE"
      acknowledgementActor="clinical nurse"
      handoverActionLabel="✓ Accept Handover"
    />
  );

  return embedded ? dashboard : (
    <SidebarLayout username={config?.staffName || 'Clinical Nurse'}>
      {dashboard}
    </SidebarLayout>
  );
};

export default ClinicalNurseDashboard;
