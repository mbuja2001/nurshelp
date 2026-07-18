import { NurseClockRecord } from "../models/staffClockRecord.model.js";
import { ClinicalStaff } from "../models/clinicalstaff.model.js";

// ----------------------
// GET STAFF STATUS DASHBOARD
// ----------------------
export const getStaffStatus = async (req, res) => {
  try {
    const managerId = req.user?._id;

    if (!managerId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const { location, status, showPendingApprovals } = req.query;

    // Get today's clock records
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today.getTime() + 24 * 60 * 60 * 1000);

    const query = {
      date: { $gte: today, $lt: tomorrow }
    };

    // Filter by location if provided
    if (location) {
      query.location = location;
    }

    // Filter by clock status if provided
    if (status) {
      query.status = status;
    }

    // Get clock records
    let clockRecords = await NurseClockRecord.find(query)
      .populate("staffId", "firstName lastName email")
      .sort({ clockInTime: -1 });

    // If requesting pending approvals only
    if (showPendingApprovals === "true") {
      clockRecords = clockRecords.filter(record => 
        record.breakApprovalStatus === "PENDING" || 
        record.lateArrivalApprovalStatus === "PENDING"
      );
    }

    // Format response with summary
    const staffStatus = clockRecords.map(record => {
      const now = new Date();
      const elapsedMs = now - record.clockInTime;
      const elapsedMinutes = Math.floor(elapsedMs / 1000 / 60);

      let breakDurationMinutes = 0;
      if (record.breakStartTime && record.breakEndTime) {
        breakDurationMinutes = Math.floor((record.breakEndTime - record.breakStartTime) / 1000 / 60);
      }

      return {
        _id: record._id,
        staff: {
            _id: record.staffId?._id,
            firstName: record.staffId?.firstName,
            lastName: record.staffId?.lastName,
            email: record.staffId?.email
          },
        clockInTime: record.clockInTime,
        clockOutTime: record.clockOutTime,
        status: record.status,
        location: record.location,
        elapsedMinutes,
        breakStartTime: record.breakStartTime,
        breakEndTime: record.breakEndTime,
        breakDurationMinutes,
        // Approval statuses
        lateArrivalMinutes: record.lateArrivalMinutes,
        lateArrivalApprovalStatus: record.lateArrivalApprovalStatus,
        breakApprovalStatus: record.breakApprovalStatus,
        breakReason: record.breakReason,
        // Pending actions
        hasPendingApprovals: 
          record.breakApprovalStatus === "PENDING" || 
          record.lateArrivalApprovalStatus === "PENDING"
      };
    });

    // Summary statistics
    const summary = {
      total: staffStatus.length,
      clockedIn: staffStatus.filter(s => s.status === "CLOCKED_IN").length,
      onBreak: staffStatus.filter(s => s.status === "ON_BREAK").length,
      clockedOut: staffStatus.filter(s => s.status === "CLOCKED_OUT").length,
      pendingApprovals: staffStatus.filter(s => s.hasPendingApprovals).length,
      lateArrivals: staffStatus.filter(s => s.lateArrivalApprovalStatus === "PENDING").length,
      pendingBreaks: staffStatus.filter(s => s.breakApprovalStatus === "PENDING").length
    };

    res.status(200).json({
      success: true,
      summary,
      staff: staffStatus,
      timestamp: new Date()
    });

  } catch (err) {
    console.error("GET STAFF STATUS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// GET PENDING APPROVALS FOR MANAGER
// ----------------------
export const getPendingApprovals = async (req, res) => {
  try {
    const managerId = req.user?._id;

    if (!managerId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    // Get ALL pending approvals (not just today)
    const pendingBreaks = await NurseClockRecord.find({
      breakApprovalStatus: "PENDING"
    }).populate("staffId", "firstName lastName email");

    const pendingLateArrivals = await NurseClockRecord.find({
      lateArrivalApprovalStatus: "PENDING"
    }).populate("staffId", "firstName lastName email");

    // Format pending items
    const pending = [];

    // Add pending breaks
    pendingBreaks.forEach(record => {
      pending.push({
        _id: record._id,
        type: "BREAK_REQUEST",
        staff: {
          _id: record.staffId?._id,
          name: `${record.staffId?.firstName || ''} ${record.staffId?.lastName || ''}`.trim(),
          email: record.staffId?.email
        },
        date: record.date,
        detail: {
          reason: record.breakReason,
          clockInTime: record.clockInTime,
          requestTime: record.updatedAt
        }
      });
    });

    // Add pending late arrivals
    pendingLateArrivals.forEach(record => {
      pending.push({
        _id: record._id,
        type: "LATE_ARRIVAL",
        staff: {
          _id: record.staffId?._id,
          name: `${record.staffId?.firstName || ''} ${record.staffId?.lastName || ''}`.trim(),
          email: record.staffId?.email
        },
        date: record.date,
        detail: {
          minutesLate: record.lateArrivalMinutes,
          reason: record.lateArrivalReason,
          clockInTime: record.clockInTime
        }
      });
    });

    // Sort by most recent first
    pending.sort((a, b) => new Date(b.detail.requestTime || b.detail.clockInTime) - new Date(a.detail.requestTime || a.detail.clockInTime));

    res.status(200).json({
      success: true,
      count: pending.length,
      pending
    });

  } catch (err) {
    console.error("GET PENDING APPROVALS ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};

// ----------------------
// BULK APPROVE/REJECT (Manager)
// ----------------------
export const bulkApproveApprovals = async (req, res) => {
  try {
    const managerId = req.user?._id;
    const { approvals } = req.body; // Array of { recordId, type, approved, reason }

    if (!managerId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    if (!Array.isArray(approvals) || approvals.length === 0) {
      return res.status(400).json({ message: "approvals array required" });
    }

    const results = [];

    for (const approval of approvals) {
      try {
        const { recordId, type, approved, reason } = approval;
        const clockRecord = await NurseClockRecord.findById(recordId);

        if (!clockRecord) {
          results.push({ recordId, success: false, message: "Record not found" });
          continue;
        }

        if (type === "LATE_ARRIVAL") {
          clockRecord.lateArrivalApprovalStatus = approved ? "APPROVED" : "REJECTED";
          clockRecord.lateArrivalApprovedBy = managerId;
          clockRecord.lateArrivalApprovalTime = new Date();
          if (!approved && reason) {
            clockRecord.lateArrivalRejectionReason = reason;
          }
        } else if (type === "BREAK_REQUEST") {
          clockRecord.breakApprovalStatus = approved ? "APPROVED" : "REJECTED";
          clockRecord.breakApprovedBy = managerId;
          clockRecord.breakApprovalTime = new Date();
          if (!approved && reason) {
            clockRecord.breakRejectionReason = reason;
          } else if (approved && !clockRecord.breakStartTime) {
            clockRecord.breakStartTime = new Date();
            clockRecord.status = "ON_BREAK";
          }
        }

        await clockRecord.save();
        results.push({ recordId, success: true, type, approved, message: "Updated" });
      } catch (err) {
        results.push({ recordId: approval.recordId, success: false, message: err.message });
      }
    }

    res.status(200).json({
      success: true,
      message: `Processed ${results.length} approvals`,
      results
    });

  } catch (err) {
    console.error("BULK APPROVE ERROR:", err);
    res.status(500).json({ message: err.message });
  }
};
