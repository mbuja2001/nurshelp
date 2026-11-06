import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

export default function ChartSessionBoundary({ encounterId, children }) {
  const navigate = useNavigate();
  const sessionKeyRef = useRef(globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const sessionIdRef = useRef(null);
  const dirtyRef = useRef(false);
  const heartbeatBusyRef = useRef(false);
  const [sessionStatus, setSessionStatus] = useState('connecting');
  const [sessionError, setSessionError] = useState('');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);

  const updateSession = useCallback(async (payload) => {
    const token = localStorage.getItem('token');
    if (!sessionIdRef.current || !token) throw new Error('Authentication or chart-session tracking is unavailable');
    const response = await fetch(`${BACKEND_URL}/api/chart-sessions/${sessionIdRef.current}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) throw new Error(data.message || `Chart session update failed (${response.status})`);
    return data.session;
  }, []);

  useEffect(() => {
    let cancelled = false;
    const token = localStorage.getItem('token');
    if (!encounterId || !token) {
      setSessionStatus('error');
      setSessionError(!encounterId ? 'No encounter was supplied for chart tracking.' : 'No authenticated session is available for chart tracking.');
      return undefined;
    }

    const openSession = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/chart-sessions/open`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify({ encounterId, sessionKey: sessionKeyRef.current })
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.success || !data.session?._id) {
          throw new Error(data.message || `Unable to establish chart tracking (${response.status})`);
        }
        if (cancelled) return;
        sessionIdRef.current = data.session._id;
        if (dirtyRef.current) {
          await updateSession({ status: 'OPEN', hasUnsavedChanges: true });
        }
        setSessionStatus('open');
        setSessionError('');
      } catch (error) {
        if (cancelled) return;
        setSessionStatus('error');
        setSessionError(error.message);
      }
    };

    openSession();
    return () => { cancelled = true; };
  }, [encounterId, updateSession]);

  useEffect(() => {
    const onEdit = () => {
      if (dirtyRef.current) return;
      dirtyRef.current = true;
      setHasUnsavedChanges(true);
      updateSession({ status: 'OPEN', hasUnsavedChanges: true })
        .catch((error) => {
          setSessionStatus('error');
          setSessionError(error.message);
        });
    };
    document.addEventListener('input', onEdit, true);
    document.addEventListener('change', onEdit, true);
    return () => {
      document.removeEventListener('input', onEdit, true);
      document.removeEventListener('change', onEdit, true);
    };
  }, [updateSession]);

  useEffect(() => {
    const timer = setInterval(async () => {
      if (!sessionIdRef.current || heartbeatBusyRef.current) return;
      heartbeatBusyRef.current = true;
      try {
        await updateSession({ status: 'OPEN', hasUnsavedChanges: dirtyRef.current });
        setSessionError('');
      } catch (error) {
        setSessionStatus('error');
        setSessionError(error.message);
      } finally {
        heartbeatBusyRef.current = false;
      }
    }, 15000);
    return () => clearInterval(timer);
  }, [updateSession]);

  const closeSession = async (status, extra = {}) => {
    setActionBusy(true);
    try {
      await updateSession({ status, ...extra });
      sessionIdRef.current = null;
      setSessionStatus(status.toLowerCase());
      navigate('/shift-dashboard');
    } catch (error) {
      setSessionStatus('error');
      setSessionError(error.message);
    } finally {
      setActionBusy(false);
    }
  };

  const confirmChangesSaved = async () => {
    if (!window.confirm('Confirm that all chart changes have been saved in the patient record.')) return;
    setActionBusy(true);
    try {
      await updateSession({ status: 'OPEN', hasUnsavedChanges: false });
      dirtyRef.current = false;
      setHasUnsavedChanges(false);
      setSessionError('');
    } catch (error) {
      setSessionStatus('error');
      setSessionError(error.message);
    } finally {
      setActionBusy(false);
    }
  };

  const closeAsSaved = () => {
    if (hasUnsavedChanges) return;
    if (!window.confirm('Close this saved chart session?')) return;
    closeSession('CLOSED', { attestedSaved: true });
  };

  const transferSession = () => {
    const transferNote = window.prompt('Enter the receiving clinician or handover details (required):');
    if (!transferNote?.trim()) return;
    closeSession('TRANSFERRED', { transferNote: transferNote.trim() });
  };

  return (
    <>
      <aside role="status" aria-live="polite" style={{ margin: '10px 12px', padding: '10px 12px', border: `1px solid ${sessionStatus === 'error' ? '#fca5a5' : '#bfdbfe'}`, borderRadius: 10, background: sessionStatus === 'error' ? '#fef2f2' : '#eff6ff', color: sessionStatus === 'error' ? '#991b1b' : '#1e3a8a', fontSize: 13 }}>
        <strong>Chart session: {sessionStatus === 'open' ? (hasUnsavedChanges ? 'open — edits detected' : 'open — heartbeat active') : sessionStatus}</strong>
        {sessionError && <div style={{ marginTop: 4 }}>{sessionError} Mid-shift hard stop remains disabled when chart tracking cannot be confirmed.</div>}
        {sessionStatus === 'open' && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8 }}>
            {hasUnsavedChanges && <button type="button" disabled={actionBusy} onClick={confirmChangesSaved}>Confirm chart changes saved</button>}
            <button type="button" disabled={actionBusy || hasUnsavedChanges} onClick={closeAsSaved}>Close saved chart session</button>
            <button type="button" disabled={actionBusy} onClick={transferSession}>Transfer / hand over chart</button>
          </div>
        )}
      </aside>
      {children}
    </>
  );
}
