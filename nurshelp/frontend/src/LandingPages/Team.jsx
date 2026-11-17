import React from 'react';
import MarketingInfoPage from '../components/marketing/MarketingInfoPage';

export default function Team() {
  return (
    <MarketingInfoPage
      eyebrow="People before platforms"
      title="Built around the people doing the work."
      description="NurseHelpDesk is being shaped around the practical realities of emergency care: fast-moving teams, complex context and the need for clear communication."
      sections={[
        { title: 'Listen to frontline work', description: 'Product decisions should start with the people who navigate intake, coordination and handoffs—not assumptions about how care happens.' },
        { title: 'Build across disciplines', description: 'Clinical operations, design and software engineering each bring a necessary perspective to useful healthcare technology.' },
        { title: 'Be clear about what exists', description: 'A responsible product story separates current functionality from future direction and makes no unsupported promises.' },
        { title: 'Improve in partnership', description: 'Meaningful progress comes from testing workflows with the teams who use them and acting on what they learn.' }
      ]}
    />
  );
}
