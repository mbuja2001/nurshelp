import React, { useState, useEffect } from 'react';
import SidebarLayout from '../SidebarLayout';
import { decodeJwt } from '../../utils/jwt';
import { FaEnvelope, FaPills } from 'react-icons/fa';

const StaffToPharmacist = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const token = localStorage.getItem('authToken');
  const user = decodeJwt(token) || {};

  useEffect(() => {
    // lightweight polling to surface outstanding messages to pharmacist
    let mounted = true;
    const load = async () => {
      try {
        const res = await fetch(`/api/pharmacy/messages/pending`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!mounted) return;
        if (res.ok) {
          const data = await res.json();
          setMessages(data || []);
        }
      } catch (err) {
        // ignore - this UI is intentionally resilient
      }
    };
    load();
    const id = setInterval(load, 8000);
    return () => { mounted = false; clearInterval(id); };
  }, [token]);

  const handleSend = async () => {
    if (!message.trim()) return;
    try {
      const res = await fetch('/api/pharmacy/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ text: message })
      });
      if (res.ok) {
        setMessage('');
        const created = await res.json();
        setMessages(prev => [created, ...prev]);
      }
    } catch (err) {
      // ignore
    }
  };

  return (
    <SidebarLayout>
      <div style={{ padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ width: 44, height: 44, borderRadius: 8, background: '#ecfeff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <FaPills size={20} style={{ color: '#059669' }} />
          </div>
          <div>
            <h2 style={{ margin: 0 }}>Message the Pharmacist</h2>
            <div style={{ fontSize: 12, color: '#64748b' }}>Send a message or request to the pharmacy team</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 16 }}>
          <div>
            <div style={{ marginBottom: 12 }}>
              <textarea value={message} onChange={(e) => setMessage(e.target.value)} rows={4} style={{ width: '100%', padding: 12, borderRadius: 8 }} />
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={handleSend} style={{ padding: '10px 14px', background: '#059669', color: '#fff', border: 'none', borderRadius: 8 }}>Send to Pharmacist</button>
              <button onClick={() => setMessage('')} style={{ padding: '10px 14px', background: '#f1f5f9', color: '#111827', border: 'none', borderRadius: 8 }}>Clear</button>
            </div>

            <div style={{ marginTop: 18 }}>
              <h4 style={{ margin: '0 0 8px 0' }}>Recent messages</h4>
              {messages.length === 0 ? (
                <div style={{ color: '#94a3b8' }}>No messages sent yet</div>
              ) : (
                <div style={{ display: 'grid', gap: 8 }}>
                  {messages.map((m) => (
                    <div key={m._id || m.id} style={{ background: '#fff', padding: 12, borderRadius: 8, boxShadow: '0 1px 2px rgba(0,0,0,0.04)' }}>
                      <div style={{ fontSize: 13, fontWeight: 700 }}>{m.text}</div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>From: {m.fromName || user.name || 'Staff'}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div>
            <div style={{ background: '#fff', padding: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Instructions</div>
              <div style={{ fontSize: 13, color: '#64748b' }}>
                Use this to notify the pharmacist about urgent requests or clarifications. Do not use for private staff chat. Messages are forwarded to the on-duty pharmacist queue.
              </div>
            </div>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
};

export default StaffToPharmacist;
