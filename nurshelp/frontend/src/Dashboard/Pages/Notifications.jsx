import React, { useState } from 'react';
import { FiBell, FiX, FiCheck, FiAlertCircle, FiInfo } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';

const Notifications = () => {
  const { userName } = useUserFromToken();
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      type: 'alert',
      title: 'Critical Patient Admission',
      message: 'Patient in Bed 7 requires immediate reassessment. ESI Level 2 indicators detected.',
      timestamp: new Date(Date.now() - 300000),
      read: false,
    },
    {
      id: 2,
      type: 'info',
      title: 'Shift Reminder',
      message: 'Your shift ends in 2 hours. Remember to complete any pending documentation.',
      timestamp: new Date(Date.now() - 1800000),
      read: false,
    },
    {
      id: 3,
      type: 'success',
      title: 'Record Submitted',
      message: 'Patient triage record for John Doe has been successfully submitted.',
      timestamp: new Date(Date.now() - 3600000),
      read: true,
    },
    {
      id: 4,
      type: 'info',
      title: 'System Maintenance',
      message: 'Scheduled system maintenance completed successfully.',
      timestamp: new Date(Date.now() - 7200000),
      read: true,
    },
  ]);

  const markAsRead = (id) => {
    setNotifications(notifications.map(n => n.id === id ? { ...n, read: true } : n));
  };

  const markAllAsRead = () => {
    setNotifications(notifications.map(n => ({ ...n, read: true })));
  };

  const deleteNotification = (id) => {
    setNotifications(notifications.filter(n => n.id !== id));
  };

  const formatTime = (date) => {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'alert':
        return <FiAlertCircle size={20} style={{ color: '#ef4444' }} />;
      case 'success':
        return <FiCheck size={20} style={{ color: '#10b981' }} />;
      case 'info':
        return <FiInfo size={20} style={{ color: '#0369a1' }} />;
      default:
        return <FiBell size={20} style={{ color: '#6b7280' }} />;
    }
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div style={{ maxWidth: 800 }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 12 }}>
            <FiBell size={32} style={{ color: '#0369a1' }} />
            Notifications
            {unreadCount > 0 && (
              <span style={{
                backgroundColor: '#ef4444',
                color: 'white',
                fontSize: 12,
                fontWeight: 600,
                padding: '2px 8px',
                borderRadius: 12,
                marginLeft: 8,
              }}>
                {unreadCount}
              </span>
            )}
          </h1>
          <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>Stay updated with system alerts and important notifications</p>
        </div>

        {/* Action Bar */}
        {unreadCount > 0 && (
          <div style={{ marginBottom: 20, display: 'flex', gap: 12 }}>
            <button
              onClick={markAllAsRead}
              style={{
                padding: '8px 16px',
                backgroundColor: '#f3f4f6',
                border: '1px solid #d1d5db',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 13,
                fontWeight: 500,
                color: '#374151',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#e5e7eb';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#f3f4f6';
              }}
            >
              Mark all as read
            </button>
          </div>
        )}

        {/* Notifications List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {notifications.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: 40,
              backgroundColor: '#f9fafb',
              borderRadius: 12,
              border: '1px solid #e5e7eb',
              color: '#6b7280',
            }}>
              <FiBell size={32} style={{ marginBottom: 12, opacity: 0.5 }} />
              <p style={{ margin: 0, fontSize: 14 }}>No notifications at this time</p>
            </div>
          ) : (
            notifications.map(notification => (
              <div
                key={notification.id}
                onClick={() => !notification.read && markAsRead(notification.id)}
                style={{
                  backgroundColor: notification.read ? 'white' : '#f0f9ff',
                  border: `1px solid ${notification.read ? '#e5e7eb' : '#bfdbfe'}`,
                  borderRadius: 8,
                  padding: 16,
                  display: 'flex',
                  gap: 12,
                  alignItems: 'flex-start',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = notification.read ? '#f9fafb' : '#eff6ff';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = notification.read ? 'white' : '#f0f9ff';
                }}
              >
                {/* Icon */}
                <div style={{ marginTop: 2 }}>
                  {getNotificationIcon(notification.type)}
                </div>

                {/* Content */}
                <div style={{ flex: 1 }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: 12,
                    marginBottom: 4,
                  }}>
                    <h3 style={{
                      margin: 0,
                      fontSize: 14,
                      fontWeight: 600,
                      color: '#0f172a',
                    }}>
                      {notification.title}
                      {!notification.read && (
                        <span style={{
                          display: 'inline-block',
                          width: 8,
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: '#0369a1',
                          marginLeft: 8,
                          verticalAlign: 'middle',
                        }} />
                      )}
                    </h3>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteNotification(notification.id);
                      }}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: '#9ca3af',
                        padding: 4,
                        display: 'flex',
                        alignItems: 'center',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = '#ef4444';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = '#9ca3af';
                      }}
                    >
                      <FiX size={18} />
                    </button>
                  </div>
                  <p style={{
                    margin: '0 0 8px 0',
                    fontSize: 13,
                    color: '#6b7280',
                    lineHeight: 1.5,
                  }}>
                    {notification.message}
                  </p>
                  <p style={{
                    margin: 0,
                    fontSize: 12,
                    color: '#9ca3af',
                  }}>
                    {formatTime(notification.timestamp)}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </SidebarLayout>
  );
};

export default Notifications;
