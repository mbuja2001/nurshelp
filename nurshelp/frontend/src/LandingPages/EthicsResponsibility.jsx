import React from 'react';
import MarketingInfoPage from '../components/marketing/MarketingInfoPage';

export default function EthicsResponsibility() {
  return (
    <MarketingInfoPage
      eyebrow="Our approach"
      title="Technology should earn its place in care."
      description="Healthcare tools must be useful, understandable and accountable to the professionals and communities they serve."
      sections={[
        { title: 'People retain agency', description: 'Digital support can organize information and surface context. It should not replace professional judgment or make care decisions on a clinician’s behalf.' },
        { title: 'Be honest about limits', description: 'Teams should be able to understand what a feature is designed to do, what it cannot establish and when human review is essential.' },
        { title: 'Design for different needs', description: 'Accessible, inclusive workflows should be considered from the beginning and evaluated with a range of users and settings.' },
        { title: 'Learn transparently', description: 'Feedback and evaluation should inform improvement, with clear communication about changes and known limitations.' }
      ]}
      note="NurseHelpDesk is a workflow and information-support platform; it is not a substitute for clinical judgment, local protocols or emergency services."
    />
  );
}
