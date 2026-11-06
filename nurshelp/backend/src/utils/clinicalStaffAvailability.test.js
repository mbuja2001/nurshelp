import assert from "node:assert/strict";
import test from "node:test";
import { getClinicalStaffAssignmentAvailability } from "./clinicalStaffAvailability.js";

const activeShift = {
  status: "IN_PROGRESS",
  date: "2026-10-16",
  shiftStart: "08:00",
  shiftEnd: "16:30"
};

test("allows a clocked-in clinician during their active shift", () => {
  assert.deepEqual(getClinicalStaffAssignmentAvailability({
    status: "CLOCKED_IN",
    clockOutTime: null,
    shiftId: activeShift
  }, new Date(2026, 9, 16, 12, 0)), {
    available: true,
    availabilityStatus: "ON_SHIFT"
  });
});

test("does not allow a clinician who is on break to receive a new assignment", () => {
  assert.deepEqual(getClinicalStaffAssignmentAvailability({
    status: "ON_BREAK",
    clockOutTime: null,
    shiftId: activeShift
  }, new Date(2026, 9, 16, 12, 0)), {
    available: false,
    availabilityStatus: "ON_BREAK"
  });
});

test("does not allow a clocked-in clinician after their scheduled shift grace period", () => {
  assert.deepEqual(getClinicalStaffAssignmentAvailability({
    status: "CLOCKED_IN",
    clockOutTime: null,
    shiftId: activeShift
  }, new Date(2026, 9, 16, 16, 36)), {
    available: false,
    availabilityStatus: "OFF_SHIFT"
  });
});

test("does not allow a clinician without an active shift", () => {
  assert.deepEqual(getClinicalStaffAssignmentAvailability({
    status: "CLOCKED_IN",
    clockOutTime: null,
    shiftId: { ...activeShift, status: "COMPLETED" }
  }, new Date(2026, 9, 16, 12, 0)), {
    available: false,
    availabilityStatus: "OFF_SHIFT"
  });
});
