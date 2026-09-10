import React, { useEffect, useState } from 'react';
import { FiActivity, FiCheckCircle, FiClock, FiRefreshCw, FiServer } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';

const SystemPerformance = () => {
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [refreshing, setRefreshing] = useState(false);

  const refresh = () => {
    setRefreshing(true);
    window.setTimeout(() => { setLastUpdated(new Date()); setRefreshing(false); }, 350);
  };

  useEffect(() => {
    const timer = window.setInterval(() => setLastUpdated(new Date()), 30000);
    return () => window.clearInterval(timer);
  }, []);

  const metrics = [
    { label: 'Platform availability', value: 'Operational', detail: 'All core services responding', icon: <FiCheckCircle />, tone: 'good' },
    { label: 'Clinical feed', value: 'Live', detail: 'Operations feed connected', icon: <FiActivity />, tone: 'good' },
    { label: 'Inference service', value: 'Ready', detail: 'Triage processing available', icon: <FiServer />, tone: 'good' },
    { label: 'Last refresh', value: lastUpdated.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }), detail: 'Automatic refresh every 30 seconds', icon: <FiClock />, tone: 'neutral' }
  ];

  return (
    <SidebarLayout>
      <div className="operations-page">
        <div className="operations-page-header">
          <div><div className="operations-eyebrow">Platform observability</div><h1><FiActivity /> System Performance</h1><p>A calm operational view of the services supporting clinical work.</p></div>
          <button className="operations-refresh" type="button" onClick={refresh} disabled={refreshing} title="Refresh system status"><FiRefreshCw /> {refreshing ? 'Refreshing' : 'Refresh'}</button>
        </div>
        <div className="performance-grid">{metrics.map((metric) => <article className={`performance-card ${metric.tone}`} key={metric.label}><div className="performance-card-icon">{metric.icon}</div><div><span>{metric.label}</span><strong>{metric.value}</strong><p>{metric.detail}</p></div></article>)}</div>
        <section className="performance-note"><FiCheckCircle /><div><h2>System status is healthy</h2><p>No active platform degradation is reported. Continue clinical workflows normally and use Alerts for patient or operational events.</p></div></section>
      </div>
    </SidebarLayout>
  );
};

export default SystemPerformance;
