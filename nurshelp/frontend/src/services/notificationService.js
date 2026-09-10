/**
 * Notification Service
 * Handles all API calls for clinical handover notifications
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  };
};

export const notificationService = {
  getOperationsFeed: async (limit = 50, since = '') => {
    try {
      const query = new URLSearchParams({ limit: String(limit) });
      if (since) query.set('since', since);
      const res = await fetch(`${BACKEND_URL}/api/operations/feed?${query.toString()}`, {
        headers: getAuthHeaders(),
        cache: 'no-store'
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] getOperationsFeed error:', error);
      return { success: false, feed: null, error: error.message };
    }
  },

  /**
   * Get all unread notifications for current user
   */
  getUnreadNotifications: async (limit = 10) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/notifications/unread?limit=${limit}`, {
        headers: getAuthHeaders(),
        cache: 'no-store'
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] getUnreadNotifications error:', error);
      return { notifications: [], count: 0, error: error.message };
    }
  },

  /**
   * Get all notifications with pagination
   */
  getAllNotifications: async (limit = 50, skip = 0) => {
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/notifications?limit=${limit}&skip=${skip}`,
        { headers: getAuthHeaders(), cache: 'no-store' }
      );

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] getAllNotifications error:', error);
      return { notifications: [], total: 0, error: error.message };
    }
  },

  /**
   * Mark a notification as read
   */
  markAsRead: async (notificationId) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/notifications/${notificationId}/read`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify({ read: true })
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] markAsRead error:', error);
      throw error;
    }
  },

  /**
   * Acknowledge a handover notification
   * Updates clinician status in encounter
   */
  acknowledgeHandover: async (notificationId, action = 'ACKNOWLEDGE', notes = '') => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/notifications/${notificationId}/acknowledge`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          acknowledged: true,
          action,  // ACKNOWLEDGE, ACCEPT, BUSY, ESCALATE
          notes
        })
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] acknowledgeHandover error:', error);
      throw error;
    }
  },

  /**
   * Get pending handovers for current clinician
   */
  getPendingHandovers: async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/notifications/pending-handovers`, {
        headers: getAuthHeaders(),
        cache: 'no-store'
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] getPendingHandovers error:', error);
      return { pending_handovers: [], count: 0, error: error.message };
    }
  },

  /**
   * Update clinician status on a patient
   * Called when clinician marks as busy/in-progress/completed
   */
  updateClinicianStatus: async (encounterId, status = 'IN_PROGRESS', notes = '') => {
    try {
      // Note: Backend API expects clinician ID in path, we store in JWT
      const token = localStorage.getItem('token');
      const decoded = JSON.parse(atob(token.split('.')[1]));
      const clinicianId = decoded.id;

      const res = await fetch(
        `${BACKEND_URL}/api/encounters/${encounterId}/clinician/${clinicianId}/status`,
        {
          method: 'PUT',
          headers: getAuthHeaders(),
          body: JSON.stringify({
            status,  // IN_PROGRESS, COMPLETED, ON_HOLD
            notes,
            start_time: new Date()
          })
        }
      );

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] updateClinicianStatus error:', error);
      throw error;
    }
  },

  /**
   * Get handover status for a specific patient
   * Shows nurse how clinicians are responding
   */
  getHandoverStatus: async (encounterId) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover-history`, {
        headers: getAuthHeaders(),
        cache: 'no-store'
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] getHandoverStatus error:', error);
      return { handovers: [], error: error.message };
    }
  },

  /**
   * Create a handover and send notifications
   * Called by nurse confirming handover page
   */
  createHandover: async (encounterId, clinicians, manuallySelected = []) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/encounters/${encounterId}/handover`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          reportType: 'AI_TRIAGE_REPORT',
          clinicians,
          manuallySelected
        })
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      return await res.json();
    } catch (error) {
      console.error('[NotificationService] createHandover error:', error);
      throw error;
    }
  }
};

export default notificationService;
