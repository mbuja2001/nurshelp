import React from 'react';
import MarketingInfoPage from '../components/marketing/MarketingInfoPage';

export default function WhatWeDo() {
  return (
    <MarketingInfoPage
      eyebrow="A connected care workflow"
      title="Bring the moving parts into view."
      description="NurseHelpDesk connects patient intake, clinical context and hospital operations in a shared digital workspace for care teams. We exist to help public facilities in South Africa address fragmented patient records and bring useful, responsible digital tools closer to rural communities."
      sections={[
        { title: 'Address fragmented patient history', description: 'When a patient’s care spans facilities or disconnected systems, prior encounters, allergies, medicines and laboratory context may be difficult to find. Our direction is a more coherent longitudinal view, with the source record kept traceable.' },
        { title: 'Support structured intake', description: 'Capture presenting details and observations in a consistent encounter workflow that helps teams organize the first clinical picture.' },
        { title: 'Keep patient context nearby', description: 'Help authorized staff review relevant encounter and longitudinal information—including clinical and laboratory context—without losing sight of the current task.' },
        { title: 'Coordinate around the record', description: 'Staff chat supports handovers, consultation requests and urgent coordination; EHR referencing helps clinicians return to the patient record for clinical context.' },
        { title: 'Design for connectivity gaps', description: 'Our goal is to make care workflows more resilient in low-connectivity settings and bring responsibly governed AI within reach of rural public facilities. Offline capability and clinical performance must be evaluated in the real deployment context.' }
      ]}
    />
  );
}
