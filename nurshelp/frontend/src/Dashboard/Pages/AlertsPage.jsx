import React, { useEffect, useState } from 'react';
import { FiAlertCircle, FiBell, FiRefreshCw } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { notificationService } from '../../services/notificationService';

const AlertsPage = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadAlerts = async () => {
    setLoading(true);
    setError('');
    try {
      const feed = await notificationService.getOperationsFeed(50);
      setAlerts(Array.isArray(feed?.feed?.alerts) ? feed.feed.alerts : []);
    } catch (err) {
      setError('Unable to load live alerts. Please refresh and try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadAlerts(); }, []);

  return (
    <SidebarLayout>
      <div className="operations-page">
        <div className="operations-page-header">
          <div>
            <div className="operations-eyebrow">Clinical operations</div>
            <h1><FiAlertCircle /> Alerts</h1>
            <p>Live patient, workflow, and operational alerts requiring attention.</p>
          </div>
          <button className="operations-refresh" type="button" onClick={loadAlerts} disabled={loading} title="Refresh alerts"><FiRefreshCw /> Refresh</button>
        </div>
        {error && <div className="operations-error" role="alert">{error}</div>}
        <div className="operations-summary"><strong>{alerts.length}</strong><span>active alert{alerts.length === 1 ? '' : 's'}</span></div>
        <div className="operations-list">
          {loading ? <div className="operations-empty">Loading live alerts...</div> : alerts.length === 0 ? <div className="operations-empty"><FiBell /><strong>No active alerts</strong><span>The operations feed is clear.</span></div> : alerts.map((alert, index) => (
            <article className={`operations-alert ${alert.severity === 'CRITICAL' ? 'critical' : ''}`} key={alert.id || alert._id || index}>
              <FiAlertCircle />
              <div><h2>{alert.title || 'Clinical alert'}</h2><p>{alert.message || alert.body || 'New clinical update'}</p>{alert.patient?.name && <small>{alert.patient.name}{alert.patient.chiefComplaint ? ` · ${alert.patient.chiefComplaint}` : ''}</small>}</div>
              <span>{alert.severity || 'NOTICE'}</span>
            </article>
          ))}
        </div>
      </div>
    </SidebarLayout>
  );
};

export default AlertsPage;
