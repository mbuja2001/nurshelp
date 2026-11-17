import React from 'react';
import { FiArrowUpRight, FiCheck, FiHeart, FiLayers, FiLock, FiUsers } from 'react-icons/fi';
import MarketingLayout from './MarketingLayout';

const icons = [FiHeart, FiLayers, FiUsers, FiLock, FiCheck];

export default function MarketingInfoPage({ eyebrow, title, description, sections, note, status }) {
  return (
    <MarketingLayout>
      <main className="marketing-info-main">
        <header className="marketing-info-hero">
          <span className="marketing-eyebrow">{eyebrow}</span>
          <h1>{title}</h1>
          <p>{description}</p>
        </header>
        <section className="marketing-info-grid" aria-label={`${title} details`}>
          {sections.map((section, index) => {
            const Icon = icons[index % icons.length];
            return (
              <article className="marketing-info-card" key={section.title}>
                <span className="marketing-info-icon"><Icon aria-hidden="true" /></span>
                <div><h2>{section.title}</h2><p>{section.description}</p></div>
              </article>
            );
          })}
        </section>
        {status && <aside className="marketing-status-card"><span className="marketing-status-label">Regulatory status</span><p>{status}</p></aside>}
        {note && <aside className="marketing-note"><FiArrowUpRight aria-hidden="true" /> {note}</aside>}
      </main>
    </MarketingLayout>
  );
}
