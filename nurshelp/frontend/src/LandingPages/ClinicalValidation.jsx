import React from 'react';
import MarketingInfoPage from '../components/marketing/MarketingInfoPage';

export default function ClinicalValidation() {
  return (
    <MarketingInfoPage
      eyebrow="Evidence with humility"
      title="Clinical trust is built, not claimed."
      description="Healthcare technology should be evaluated in the context where it will be used. We aim to communicate clearly about product scope, evidence and limitations."
      sections={[
        { title: 'Start with the intended workflow', description: 'Evaluation should ask whether a feature supports its stated task and fits the realities of the people expected to use it.' },
        { title: 'Keep clinicians in the loop', description: 'Clinical review and local governance remain essential when introducing digital tools into care workflows.' },
        { title: 'Measure what matters', description: 'Useful evaluation considers usability, reliability and workflow fit—not just whether a screen or model produces an output.' },
        { title: 'Publish the boundaries', description: 'A responsible product description distinguishes demonstrated functionality from hypotheses, planned work and outcomes that have not been established.' }
      ]}
      note="We do not represent this page as evidence of regulatory clearance, independent clinical validation or improved patient outcomes."
    />
  );
}
