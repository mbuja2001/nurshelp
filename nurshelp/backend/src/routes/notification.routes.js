import express from 'express';
import { authenticateToken } from '../middleware/auth.js';
import { Notification } from '../models/notification.model.js';

const router = express.Router();

/**
 * GET /api/notifications/unread
 * Get unread notifications for logged-in clinician (for NotificationBell)
 */
router.get('/unread', authenticateToken, async (req, res) => {
  try {
    const { id: clinicianId, hospitalId } = req.user;
    const limit = parseInt(req.query.limit) || 10;

    const notifications = await Notification.find({
      clinician_id: clinicianId,
      hospital_id: hospitalId,
      status: 'UNREAD'
    })
      .populate('encounter_id', 'patient')
      .sort({ created_at: -1 })
      .limit(limit)
      .lean();

    const unreadCount = await Notification.countDocuments({
      clinician_id: clinicianId,
      hospital_id: hospitalId,
      status: 'UNREAD'
    });

    console.log('[UNREAD] Clinician:', clinicianId, '| Count:', unreadCount);

    res.json({
      success: true,
      unreadCount,
      count: notifications.length,
      notifications
    });

  } catch (err) {
    console.error('[UNREAD] Error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/notifications
 * Get all notifications for logged-in clinician (paginated)
 */
router.get('/', authenticateToken, async (req, res) => {
  try {
    const { id: clinicianId, hospitalId } = req.user;
    const limit = parseInt(req.query.limit) || 20;
    const skip = parseInt(req.query.skip) || 0;

    const notifications = await Notification.find({
      clinician_id: clinicianId,
      hospital_id: hospitalId
    })
      .populate('encounter_id', 'patient')
      .sort({ created_at: -1 })
      .skip(skip)
      .limit(limit)
      .lean();

    const total = await Notification.countDocuments({
      clinician_id: clinicianId,
      hospital_id: hospitalId
    });

    console.log('[ALL] Clinician:', clinicianId, '| Total:', total, '| Showing:', notifications.length);

    res.json({
      success: true,
      total,
      count: notifications.length,
      skip,
      limit,
      notifications
    });

  } catch (err) {
    console.error('[ALL] Error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * PUT /api/notifications/:id/read
 * Mark notification as read
 */
router.put('/:id/read', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    const { id: clinicianId } = req.user;

    const notification = await Notification.findByIdAndUpdate(
      id,
      {
        $set: {
          status: 'READ',
          read_at: new Date()
        },
        $push: {
          audit_log: {
            action: 'READ',
            timestamp: new Date()
          }
        }
      },
      { new: true }
    );

    if (!notification) {
      return res.status(404).json({ error: 'Notification not found' });
    }

    console.log('[READ] Notification marked as read by:', clinicianId);

    res.json({ success: true, notification });

  } catch (err) {
    console.error('[READ] Error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * PUT /api/notifications/:id/acknowledge
 * Acknowledge handover - clinician confirms they received and will act on it
 */
router.put('/:id/acknowledge', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    const { notes } = req.body;
    const { id: clinicianId } = req.user;

    const notification = await Notification.findByIdAndUpdate(
      id,
      {
        $set: {
          status: 'ACKNOWLEDGED',
          acknowledged_at: new Date(),
          acknowledged_notes: notes || '',
          engagement_level: 'IMMEDIATE'  // Flag as engaged
        },
        $push: {
          audit_log: {
            action: 'ACKNOWLEDGED',
            timestamp: new Date(),
            notes: notes || 'No notes provided'
          }
        }
      },
      { new: true }
    );

    if (!notification) {
      return res.status(404).json({ error: 'Notification not found' });
    }

    console.log('[ACKNOWLEDGE] Clinician:', clinicianId, '| Notification:', id);

    res.json({ success: true, notification });

  } catch (err) {
    console.error('[ACKNOWLEDGE] Error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * PUT /api/notifications/:id/dismiss
 * Dismiss notification - clinician won't act on it
 */
router.put('/:id/dismiss', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    const { reason } = req.body;
    const { id: clinicianId } = req.user;

    const notification = await Notification.findByIdAndUpdate(
      id,
      {
        $set: {
          status: 'DISMISSED',
          dismissed_at: new Date(),
          dismiss_reason: reason || 'No reason provided'
        },
        $push: {
          audit_log: {
            action: 'DISMISSED',
            timestamp: new Date(),
            notes: reason
          }
        }
      },
      { new: true }
    );

    if (!notification) {
      return res.status(404).json({ error: 'Notification not found' });
    }

    console.log('[DISMISS] Clinician:', clinicianId, '| Reason:', reason);

    res.json({ success: true, notification });

  } catch (err) {
    console.error('[DISMISS] Error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/notifications/encounter/:encounterId
 * Get all notifications for a specific encounter
 */
router.get('/encounter/:encounterId', authenticateToken, async (req, res) => {
  try {
    const { encounterId } = req.params;
    const { hospitalId } = req.user;

    const notifications = await Notification.find({
      encounter_id: encounterId,
      hospital_id: hospitalId
    })
      .populate('clinician_id', 'name role specialty email')
      .populate('triage_nurse_id', 'name email')
      .sort({ created_at: -1 })
      .lean();

    console.log('[ENCOUNTER] Encounter:', encounterId, '| Notifications:', notifications.length);

    res.json({
      success: true,
      encounterId,
      notificationCount: notifications.length,
      notifications
    });

  } catch (err) {
    console.error('[ENCOUNTER] Error:', err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
