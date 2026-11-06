import { afterEach, describe, expect, it, vi } from 'vitest';
import { clearAuthSession } from './authSession';

describe('clearAuthSession', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('removes auth state and announces logout to mounted dashboards', () => {
    const values = new Map([
      ['token', 'test-token'],
      ['currentNurseId', 'nurse-id'],
      ['currentStaffId', 'staff-id'],
      ['userRole', 'TRIAGE_NURSE'],
      ['user', '{"id":"nurse-id"}']
    ]);
    const removeItem = vi.fn((key) => values.delete(key));
    const dispatchEvent = vi.fn();
    vi.stubGlobal('localStorage', { removeItem });
    vi.stubGlobal('window', { dispatchEvent });

    clearAuthSession();

    expect(removeItem.mock.calls.map(([key]) => key)).toEqual([
      'token',
      'currentNurseId',
      'currentStaffId',
      'userRole',
      'user'
    ]);
    expect(values.size).toBe(0);
    expect(dispatchEvent).toHaveBeenCalledTimes(1);
    expect(dispatchEvent.mock.calls[0][0].type).toBe('nursehelpdesk:logout');
  });
});
