import React, { useState, useEffect } from 'react';
import { FiBell } from 'react-icons/fi';

const NotificationBell = ({ className = '' }) => {
  const [unreadCount, setUnreadCount] = useState(0);
  const [showDropdown, setShowDropdown] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(false);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";
  const token = localStorage.getItem('token');

  const fetchUnreadNotifications = async () => {
    try {
      if (!token) return;
      
      setLoading(true);
      const res = await fetch(`${BACKEND_URL}/api/notifications/unread?limit=10`, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Cache-Control': 'no-cache'
        },
        cache: 'no-store'
      });

      if (!res.ok) throw new Error('Failed to fetch notifications');
      
      const data = await res.json();
      setNotifications(data.notifications || []);
      setUnreadCount(data.count || 0);
    } catch (error) {
      console.error('[NotificationBell] Error fetching notifications:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchUnreadNotifications();

    // Poll every 30 seconds
    const interval = setInterval(fetchUnreadNotifications, 30000);
    
    return () => clearInterval(interval);
  }, [token]);

  const handleNotificationClick = async (notificationId, encounterId) => {
    try {
      // Mark as read
      await fetch(`${BACKEND_URL}/api/notifications/${notificationId}/read`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ read: true })
      });

      // Navigate to handover report
      window.location.href = `/handover/${encounterId}`;
    } catch (error) {
      console.error('[NotificationBell] Error:', error);
    }
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }} className={className}>
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        style={{
          position: 'relative',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          padding: 8,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
        title="Notifications"
      >
        <FiBell size={20} color="#64748b" />
        {unreadCount > 0 && (
          <span
            style={{
              position: 'absolute',
              top: 0,
              right: 0,
              background: '#ef4444',
              color: 'white',
              borderRadius: '50%',
              width: 20,
              height: 20,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 12,
              fontWeight: 700
            }}
          >
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {showDropdown && (
        <div
          style={{
            position: 'absolute',
            top: 40,
            right: 0,
            background: 'white',
            border: '1px solid #e2e8f0',
            borderRadius: 8,
            boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
            zIndex: 1000,
            minWidth: 320,
            maxHeight: 400,
            overflow: 'auto'
          }}
        >
          {/* Header */}
          <div style={{ padding: 12, borderBottom: '1px solid #e2e8f0', background: '#f8fafc' }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#0f172a' }}>
              Notifications {unreadCount > 0 && <span style={{ color: '#ef4444' }}>({unreadCount})</span>}
            </div>
          </div>

          {/* Notifications List */}
          {notifications.length > 0 ? (
            <div>
              {notifications.map((notification) => (
                <div
                  key={notification._id}
                  onClick={() => handleNotificationClick(notification._id, notification.encounter_id)}
                  style={{
                    padding: 12,
                    borderBottom: '1px solid #f1f5f9',
                    cursor: 'pointer',
                    background: notification.status === 'UNREAD' ? '#eff6ff' : '#ffffff',
                    transition: 'background 0.2s'
                  }}
                  onMouseOver={(e) => e.currentTarget.style.background = '#f0f9ff'}
                  onMouseOut={(e) => e.currentTarget.style.background = notification.status === 'UNREAD' ? '#eff6ff' : '#ffffff'}
                >
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#0f172a', marginBottom: 4 }}>
                    {notification.title}
                  </div>
                  {notification.patient_summary && (
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                      <span>ESI {notification.patient_summary.esi_level}</span>
                      {notification.patient_summary.priority_code && (
                        <span> • {notification.patient_summary.priority_code}</span>
                      )}
                    </div>
                  )}
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>
                    {new Date(notification.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: 20, textAlign: 'center', fontSize: 13, color: '#94a3b8' }}>
              {loading ? 'Loading...' : 'No notifications'}
            </div>
          )}

          {/* View All Link */}
          {notifications.length > 0 && (
            <div style={{ padding: 12, borderTop: '1px solid #e2e8f0', textAlign: 'center', background: '#f8fafc' }}>
              <a
                href="/notifications"
                style={{
                  fontSize: 12,
                  color: '#3b82f6',
                  textDecoration: 'none',
                  fontWeight: 600
                }}
                onClick={(e) => {
                  e.preventDefault();
                  setShowDropdown(false);
                  window.location.href = '/notifications';
                }}
              >
                View All Notifications →
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NotificationBell;
