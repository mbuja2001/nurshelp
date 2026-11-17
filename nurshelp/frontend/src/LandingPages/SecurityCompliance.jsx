import React from 'react';
import MarketingInfoPage from '../components/marketing/MarketingInfoPage';

export default function SecurityCompliance() {
  return (
    <MarketingInfoPage
      eyebrow="South Africa | Privacy, safety and accountability"
      title="Trust is built with evidence, not badges."
      description="NurseHelpDesk is being developed with South African privacy and responsible-technology expectations in view. We will publish deployment-specific assurances only when they are verified."
      sections={[
        { title: 'POPIA and purpose limitation', description: 'Our intended practice is to process personal information only for the service purpose requested by the healthcare provider, consistent with POPIA purpose-specification principles. Purposes and retention must be documented before production use.' },
        { title: 'Customer instructions and roles', description: 'Where NurseHelpDesk acts as a service provider, the parties must document the applicable responsible-party/operator roles and processing instructions in their agreements before personal information is handled.' },
        { title: 'Residency and cross-border transfers', description: 'Hosting region and subprocessors depend on the production deployment. Before any cross-border processing, the parties must identify locations and safeguards and assess applicable POPIA section 72 requirements. We do not currently promise EU data residency.' },
        { title: 'Security safeguards', description: 'The application includes role-based access controls and audit logging. Production transport security, storage encryption, backups, monitoring and secrets handling must also be configured and verified for the selected VPS and database.' },
        { title: 'Human oversight for AI', description: 'NurseHelpDesk can support triage and clinical workflows, including AI-assisted suggestions. Outputs require independent review by qualified practitioners; the system does not replace professional judgment or local protocols.' },
        { title: 'HPCSA guidance and governance', description: 'We are assessing our product and operating practices against applicable HPCSA guidance on responsible AI use, with attention to transparency, human oversight and professional accountability. This is not a claim of endorsement or completed assessment.' }
      ]}
      status="We are pursuing the applicable SAHPRA medical-device pathway, with Class A as our current intended classification. No SAHPRA licence, registration or market authorization has been granted. The applicable classification and pathway remain subject to regulatory determination."
      note="Before using real patient information, confirm the production hosting region, security configuration, customer agreements, subprocessors and legal obligations with your privacy, security and regulatory advisers. We do not claim ISO 27001 certification, SOC 2 attestation or other third-party certification."
    />
  );
}
