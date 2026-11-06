import React from 'react';

const formatDate = (value) => {
  if (!value) return 'Date unavailable';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleDateString();
};

const EhrContextPanel = ({ context }) => {
  const history = context?.longitudinalHistory || context;
  const metadata = history?.metadata || {};
  const visits = [
    ...(history?.recent_history || []),
    ...(history?.relevant_history || []),
    ...(history?.critical_history || [])
  ]
    .filter((visit, index, all) => all.findIndex((item) => String(item._id || item.id) === String(visit._id || visit.id)) === index)
    .sort((left, right) => new Date(right.date || 0) - new Date(left.date || 0))
    .slice(0, 5);

  return (
    <section style={{
      padding: '16px',
      borderRadius: '12px',
      background: 'rgba(15,23,42,0.6)',
      border: '1px solid rgba(255,255,255,0.08)'
    }} aria-label="Historical EHR context">
      <p style={{ margin: '0 0 8px', fontSize: '12px', color: '#38bdf8', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
        EHR context captured with request
      </p>
      {!context ? (
        <p style={{ margin: 0, fontSize: '12px', color: '#cbd5e1' }}>No historical context snapshot was available for this request.</p>
      ) : (
        <>
          <p style={{ margin: '0 0 10px', fontSize: '11px', color: '#94a3b8' }}>
            Captured {formatDate(context.capturedAt)} · {metadata.source_encounter_count || 0} prior encounter(s) reviewed
          </p>
          {visits.length > 0 ? (
            <div style={{ display: 'grid', gap: 8 }}>
              {visits.map((visit, index) => (
                <div key={visit._id || visit.id || `${visit.date}-${index}`} style={{ padding: 9, borderRadius: 8, background: 'rgba(30,41,59,0.8)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: '11px', fontWeight: 700, color: '#e2e8f0' }}>
                    <span>{formatDate(visit.date)}</span>
                    <span>{String(visit.status || 'HISTORICAL').replaceAll('_', ' ')}</span>
                  </div>
                  {(visit.diagnoses || []).length > 0 && (
                    <p style={{ margin: '5px 0 0', fontSize: '12px', color: '#cbd5e1' }}>Diagnosis: {visit.diagnoses.join(', ')}</p>
                  )}
                  {visit.symptoms && <p style={{ margin: '5px 0 0', fontSize: '12px', color: '#cbd5e1' }}>Presentation: {visit.symptoms}</p>}
                  {visit.patient?.history && <p style={{ margin: '5px 0 0', fontSize: '12px', color: '#cbd5e1' }}>History: {visit.patient.history}</p>}
                  {Object.entries(visit.patient?.criticalHistory || {})
                    .filter(([, values]) => Array.isArray(values) && values.length > 0)
                    .map(([category, values]) => (
                      <p key={category} style={{ margin: '5px 0 0', fontSize: '12px', color: '#fbbf24' }}>
                        {category.replaceAll(/([A-Z])/g, ' $1')}: {values.join(', ')}
                      </p>
                    ))}
                </div>
              ))}
            </div>
          ) : (
            <p style={{ margin: 0, fontSize: '12px', color: '#cbd5e1' }}>No prior encounters were available in the captured history window.</p>
          )}
          {(history?.critical_history || []).some((visit) => visit.patient?.criticalHistory) && (
            <p style={{ margin: '10px 0 0', fontSize: '11px', color: '#fbbf24' }}>Critical history is included in the persisted context snapshot.</p>
          )}
        </>
      )}
    </section>
  );
};

export default EhrContextPanel;
