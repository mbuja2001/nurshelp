import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FiArrowUpRight, FiArrowRight, FiActivity, FiUsers, FiLayers, FiShield } from 'react-icons/fi';
import MarketingLayout from './components/marketing/MarketingLayout';

const capabilities = [
  {
    icon: FiActivity,
    number: '01',
    title: 'A more coherent first contact',
    copy: 'Bring intake details, observations and triage context into a structured workflow designed for busy care teams.'
  },
  {
    icon: FiUsers,
    number: '02',
    title: 'The right people, in view',
    copy: 'Keep team assignments, shift coverage and operational handoffs connected across the department.'
  },
  {
    icon: FiLayers,
    number: '03',
    title: 'Context that travels with care',
    copy: 'Make relevant patient history and encounter information easier for authorized clinicians to find.'
  },
  {
    icon: FiShield,
    number: '04',
    title: 'Human-led by design',
    copy: 'Clinical teams remain responsible for reviewing information and making care decisions.'
  }
];

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <MarketingLayout>
      <main className="marketing-home">
        <section className="home-hero">
          <div className="home-hero-grid">
            <div className="home-hero-copy">
              <span className="marketing-kicker"><span className="kicker-pulse" /> Built for South Africa's public healthcare reality</span>
              <h1>Better connected care starts <em>at the front door.</em></h1>
              <p className="home-hero-intro">
                Our goal is to make dependable digital tools and responsible AI more reachable for rural clinics, health centres and public hospitals, including those with unreliable internet access.
              </p>
              <div className="home-hero-actions">
                <button className="marketing-button marketing-button-primary" onClick={() => navigate('/dashboard?register=account')}>
                  Explore the platform <FiArrowUpRight aria-hidden="true" />
                </button>
                <button className="marketing-button marketing-button-quiet" onClick={() => navigate('/what-we-do')}>
                  See how it works <FiArrowRight aria-hidden="true" />
                </button>
              </div>
              <div className="home-hero-note">
                <span className="note-mark"><FiShield aria-hidden="true" /></span>
                <span>Designed to support clinical teams. Decisions stay with clinicians.</span>
              </div>
            </div>
            <div className="home-hero-art" aria-label="Illustration of a connected clinical workflow">
              <div className="hero-art-glow" />
              <img src="/marketing/clinical-workflow.svg" alt="" fetchPriority="high" />
              <div className="hero-float-card hero-float-card-top">
                <span className="float-dot" />
                <span><strong>Care, in context</strong><small>One connected view</small></span>
              </div>
              <div className="hero-float-card hero-float-card-bottom">
                <span className="float-icon"><FiActivity aria-hidden="true" /></span>
                <span><strong>Human-led workflow</strong><small>Built around care teams</small></span>
              </div>
            </div>
          </div>
          <div className="hero-bottom-line">
            <span>Thoughtful tools for complex care environments</span>
            <span className="hero-scroll-cue"><span /> Scroll to explore</span>
          </div>
        </section>

        <section className="home-capabilities marketing-section" id="platform">
          <div className="section-heading">
            <span className="marketing-eyebrow">One connected workflow</span>
            <h2>Less fragmentation.<br /><em>More room for care.</em></h2>
            <p>Emergency care moves quickly. The tools supporting it should make the next step easier to see.</p>
          </div>
          <div className="capability-grid">
            {capabilities.map(({ icon: Icon, number, title, copy }) => (
              <article className="capability-card" key={number}>
                <div className="capability-card-top"><span>{number}</span><Icon aria-hidden="true" /></div>
                <h3>{title}</h3>
                <p>{copy}</p>
                <span className="capability-rule" />
              </article>
            ))}
          </div>
        </section>

        <section className="home-mission">
          <div className="home-mission-inner">
            <div className="home-mission-heading">
              <span className="marketing-eyebrow">Why NurseHelpDesk exists</span>
              <h2>Care should not lose its context when records <em>or connections</em> do.</h2>
              <p>South African care teams work across facilities and information systems that do not always share a complete patient picture. We are building toward a more connected, resilient way to support public healthcare.</p>
            </div>
            <div className="home-mission-grid">
              <article>
                <span>01 / THE PROBLEM</span>
                <h3>Important history can be hard to find.</h3>
                <p>Prior encounters, allergies, medicines, chronic conditions and reasons or results from earlier lab investigations may be missing, buried or inaccessible in another system when a clinician needs them.</p>
              </article>
              <article>
                <span>02 / THE TEAM</span>
                <h3>Communication needs the patient context.</h3>
                <p>Staff chat and EHR referencing can help teams coordinate handovers and consultations while keeping the clinical record—not a message thread—as the source of patient history.</p>
              </article>
              <article>
                <span>03 / THE GOAL</span>
                <h3>Useful technology should reach rural public care.</h3>
                <p>Our goal is to support low-connectivity settings and make carefully governed AI tools more accessible to underserved communities, without asking technology to replace clinical judgment.</p>
              </article>
            </div>
            <button className="text-link" onClick={() => navigate('/blog/why-patient-history-needs-to-travel-with-care')}>
              Read why patient history needs to travel with care <FiArrowUpRight aria-hidden="true" />
            </button>
          </div>
        </section>

        <section className="home-principle">
          <div className="principle-orbit" aria-hidden="true"><span /><span /><span /></div>
          <div className="principle-copy">
            <span className="marketing-eyebrow">Technology in its proper place</span>
            <h2>Support the people<br />who care for people.</h2>
            <p>Useful clinical technology should clarify information, respect professional judgment and fit the work as it happens.</p>
            <button className="marketing-button marketing-button-light" onClick={() => navigate('/ethics')}>
              Our principles <FiArrowUpRight aria-hidden="true" />
            </button>
          </div>
          <div className="principle-quote">
            <span className="quote-mark">“</span>
            <p>Technology should help teams see the whole picture—not make the decision for them.</p>
            <span className="quote-caption">A human-centered approach to clinical workflows</span>
          </div>
        </section>

        <section className="home-journal marketing-section">
          <div className="journal-heading">
            <div><span className="marketing-eyebrow">From the journal</span><h2>Ideas for care in motion.</h2></div>
            <button className="text-link" onClick={() => navigate('/blog')}>Visit the journal <FiArrowUpRight aria-hidden="true" /></button>
          </div>
          <div className="journal-preview-grid">
            <article className="journal-preview journal-preview-feature">
              <div className="journal-art journal-art-one" aria-hidden="true"><span /><span /><span /></div>
              <div className="journal-preview-meta">Perspectives <span>•</span> 6 min read</div>
              <h3>Why patient history needs to travel with care</h3>
              <button className="text-link" onClick={() => navigate('/blog/why-patient-history-needs-to-travel-with-care')}>Read the story <FiArrowUpRight aria-hidden="true" /></button>
            </article>
            <article className="journal-preview">
              <div className="journal-art journal-art-two" aria-hidden="true"><span /><span /><span /><span /></div>
              <div className="journal-preview-meta">Field notes <span>•</span> 4 min read</div>
              <h3>Research note: longitudinal context for local AI</h3>
              <button className="text-link" onClick={() => navigate('/blog/longitudinal-context-for-local-ai')}>Read the research note <FiArrowUpRight aria-hidden="true" /></button>
            </article>
          </div>
        </section>
        <section className="home-cta">
          <div><span className="marketing-eyebrow">A better view starts here</span><h2>Bring your team into<br />a more connected day.</h2></div>
          <button className="marketing-button marketing-button-primary" onClick={() => navigate('/dashboard?register=account')}>Get started <FiArrowUpRight aria-hidden="true" /></button>
        </section>
      </main>
    </MarketingLayout>
  );
};

export default LandingPage;
