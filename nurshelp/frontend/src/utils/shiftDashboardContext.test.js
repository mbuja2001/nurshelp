import { describe, expect, it } from 'vitest';
import { resolveShiftDashboardContext, shouldShowClockInBoard } from './shiftDashboardContext';

describe('resolveShiftDashboardContext', () => {
  it.each([
    ['PHARMACIST', 'PHARMACY'],
    ['RADIOLOGIST', 'RADIOLOGY'],
    ['PATHOLOGIST', 'PATHOLOGY']
  ])('routes a rostered SPECIALIST with role %s to %s', (role, expectedContext) => {
    expect(resolveShiftDashboardContext({
      userId: 'specialist-a',
      role,
      shift: {
        assignedStaff: [{ staffID: 'specialist-a', assignmentRole: 'SPECIALIST' }]
      }
    })).toBe(expectedContext);
  });

  it('does not infer a clinical role for a rostered staff member with an unknown duty', () => {
    expect(resolveShiftDashboardContext({
      userId: 'staff-a',
      role: 'NURSE',
      shift: {
        assignedStaff: [{ staffID: 'staff-a', assignmentRole: 'SPECIALIST' }]
      }
    })).toBe('NO_SHIFT');
  });

  it('routes a rostered technician with an approved specialty to the specialty workspace', () => {
    expect(resolveShiftDashboardContext({
      userId: 'tech-a',
      role: 'TECHNICIAN',
      specialty: 'Radiology',
      shift: {
        assignedStaff: [{ staffID: 'tech-a', assignmentRole: 'SUPPORT' }]
      }
    })).toBe('TECHNICIAN_RADIOLOGY');
  });

  it('routes a Radiology staff member by personal role on an Emergency Medicine shift', () => {
    expect(resolveShiftDashboardContext({
      userId: 'radiologist-a',
      role: 'RADIOLOGIST',
      shift: {
        department: 'ED',
        specialty: 'Emergency Medicine',
        assignedStaff: [{ staffID: 'radiologist-a', assignmentRole: 'SPECIALIST' }]
      }
    })).toBe('RADIOLOGY');
  });

    it('resolves a specialty-specific technician role without relying on a separate specialty claim', () => {
      expect(resolveShiftDashboardContext({
        userId: 'tech-a',
        role: 'RADIOLOGY_TECHNICIAN',
        shift: {
          assignedStaff: [{ staffID: 'tech-a', assignmentRole: 'SUPPORT' }]
        }
      })).toBe('TECHNICIAN_RADIOLOGY');
    });

  it('fails closed for technicians with a missing or unrecognized specialty', () => {
    for (const specialty of ['', 'Laboratory', 'General']) {
      expect(resolveShiftDashboardContext({
        userId: 'tech-a',
        role: 'TECHNICIAN',
        specialty,
        shift: {
          assignedStaff: [{ staffID: 'tech-a', assignmentRole: 'SUPPORT' }]
        }
      })).toBe('TECHNICIAN_UNCONFIGURED');
    }
  });

  it('does not route a technician assigned a specialist duty into the specialist dashboard', () => {
    expect(resolveShiftDashboardContext({
      userId: 'tech-a',
      role: 'TECHNICIAN',
      specialty: 'Pathology',
      shift: {
        assignedStaff: [{ staffID: 'tech-a', assignmentRole: 'SPECIALIST' }]
      }
    })).toBe('TECHNICIAN_UNCONFIGURED');
  });

  it('prefers a specific roster duty over the staff account role', () => {
    expect(resolveShiftDashboardContext({
      userId: 'specialist-a',
      role: 'RADIOLOGIST',
      shift: {
        assignedStaff: [{ staffID: 'specialist-a', assignmentRole: 'PATHOLOGIST' }]
      }
    })).toBe('PATHOLOGY');
  });

  it.each([
    [{ context: 'NO_SHIFT', hasShiftToday: true, isClockedIn: true }, true],
    [{ context: 'RADIOLOGY', hasShiftToday: false, isClockedIn: false }, true],
    [{ context: 'PATHOLOGY', hasShiftToday: true, isClockedIn: false }, true],
    [{ context: 'PHARMACY', hasShiftToday: true, isClockedIn: true }, false]
  ])('shows the shared clock-in board for %o: %s', (accessState, expected) => {
    expect(shouldShowClockInBoard(accessState)).toBe(expected);
  });
});
