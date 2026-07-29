import React, { useEffect, useState } from 'react';
import SidebarLayout from '../SidebarLayout';
import { FiSend, FiSearch, FiUsers, FiMessageCircle } from 'react-icons/fi';
import { decodeJwt } from '../../utils/jwt';

const StaffTeamChat = () => {
  const token = localStorage.getItem('token');
  const user = decodeJwt(token) || {};
  const [channels, setChannels] = useState([]);
  const [activeChannelId, setActiveChannelId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const loadChannels = async () => {
      try {
        setLoading(true);
        const res = await fetch('/api/staff-chat/channels', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) throw new Error('Failed to load channels');
        const data = await res.json();
        setChannels(data.channels || []);
        if (data.channels?.length) {
          setActiveChannelId(data.channels[0]._id);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadChannels();
  }, [token]);

  useEffect(() => {
    if (!activeChannelId) return;
    const loadMessages = async () => {
      try {
        setLoading(true);
        const res = await fetch(`/api/staff-chat/channels/${activeChannelId}/messages`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) throw new Error('Failed to load messages');
        const data = await res.json();
        setMessages(data.messages || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadMessages();
  }, [activeChannelId, token]);

  const handleSend = async () => {
    if (!draft.trim() || !activeChannelId) return;
    try {
      const res = await fetch(`/api/staff-chat/channels/${activeChannelId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ text: draft })
      });
      if (!res.ok) throw new Error('Failed to send message');
      const created = await res.json();
      setMessages((prev) => [...prev, created]);
      setDraft('');
    } catch (err) {
      setError(err.message);
    }
  };

  const filteredChannels = channels.filter((channel) => {
    if (!search.trim()) return true;
    return channel.name?.toLowerCase().includes(search.toLowerCase()) ||
      channel.participants?.some((user) => user.name?.toLowerCase().includes(search.toLowerCase()));
  });

  return (
    <SidebarLayout>
      <div style={{ display: 'flex', gap: 18, width: '100%', padding: 24, boxSizing: 'border-box' }}>
        <div style={{ minWidth: 300, maxWidth: 360, flex: '0 0 320px', background: '#f8fafc', borderRadius: 20, padding: 18, border: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <div style={{ width: 44, height: 44, borderRadius: 14, background: '#e0f2fe', color: '#0c4a6e', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <FiUsers size={22} />
            </div>
            <div>
              <h2 style={{ margin: 0, fontSize: 22 }}>Staff Chat</h2>
              <p style={{ margin: '4px 0 0', color: '#475569', fontSize: 14 }}>Secure team messaging for clinical staff coordination.</p>
            </div>
          </div>

          <div style={{ marginBottom: 14 }}>
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search channels or people"
              style={{ width: '100%', padding: '10px 14px', borderRadius: 14, border: '1px solid #cbd5e1', background: '#fff' }}
            />
          </div>

          <div style={{ display: 'grid', gap: 12, maxHeight: 'calc(100vh - 240px)', overflowY: 'auto' }}>
            {filteredChannels.length === 0 ? (
              <div style={{ color: '#64748b', fontSize: 13 }}>No channels found.</div>
            ) : (
              filteredChannels.map((channel) => (
                <button
                  key={channel._id}
                  onClick={() => setActiveChannelId(channel._id)}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    padding: '14px 16px',
                    borderRadius: 16,
                    border: activeChannelId === channel._id ? '2px solid #0284c7' : '1px solid #e2e8f0',
                    background: activeChannelId === channel._id ? '#e0f2fe' : '#fff',
                    color: '#0f172a',
                    cursor: 'pointer'
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>{channel.name || 'Unnamed channel'}</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>
                    {channel.participants?.map((p) => p.name).join(', ')}
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 'calc(100vh - 96px)', background: '#fff', borderRadius: 20, padding: 24, border: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 24 }}>Team Messages</h2>
              <p style={{ margin: '6px 0 0', color: '#475569', fontSize: 14 }}>Contextual chat for handovers, consult requests, and urgent staff coordination.</p>
            </div>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', paddingRight: 6, marginBottom: 20 }}>
            {loading && <div style={{ color: '#64748b', marginBottom: 10 }}>Loading messages…</div>}
            {error && <div style={{ color: '#b91c1c', marginBottom: 10 }}>{error}</div>}
            {messages.length === 0 && !loading ? (
              <div style={{ color: '#64748b' }}>Select a channel to view messages.</div>
            ) : (
              <div style={{ display: 'grid', gap: 12 }}>
                {messages.map((message) => (
                  <div key={message._id || message.id} style={{ padding: 16, borderRadius: 18, border: '1px solid #e5e7eb', background: user?.id === message.senderId ? '#dcfce7' : '#f8fafc', maxWidth: '85%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
                      <div style={{ fontSize: 13, fontWeight: 700 }}>{message.senderName || message.sender || 'Unknown'}</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>{new Date(message.createdAt || message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                    </div>
                    <div style={{ color: '#0f172a', lineHeight: 1.6 }}>{message.text}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gap: 12 }}>
            <textarea
              rows={4}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Write a message to the team..."
              style={{ width: '100%', borderRadius: 18, border: '1px solid #cbd5e1', padding: 16, resize: 'vertical' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ color: '#64748b', fontSize: 13 }}>Message is only visible within the selected channel or team thread.</div>
              <button
                onClick={handleSend}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '12px 18px', background: '#0284c7', color: '#fff', border: 'none', borderRadius: 16, cursor: 'pointer' }}
              >
                <FiSend size={16} /> Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
};

export default StaffTeamChat;
