import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiFetch } from './apiFetch';

describe('apiFetch authentication handling', () => {
  let values;
  let dispatchSpy;

  beforeEach(() => {
    values = new Map([['token', 'valid-looking-token']]);
    dispatchSpy = vi.fn();
    vi.stubGlobal('localStorage', {
      getItem: (key) => values.get(key) || null,
      removeItem: (key) => values.delete(key)
    });
    vi.stubGlobal('window', { dispatchEvent: dispatchSpy });
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('continues successful authenticated fetches without clearing the session', async () => {
    fetch.mockResolvedValue({
      status: 200,
      ok: true,
      json: async () => ({ encounters: [] })
    });

    await expect(apiFetch('/api/encounters/waiting')).resolves.toEqual({ encounters: [] });

    expect(values.get('token')).toBe('valid-looking-token');
    expect(dispatchSpy).not.toHaveBeenCalled();
  });

  it('clears an invalid protected session in-app without reloading', async () => {
    fetch.mockResolvedValue({
      status: 401,
      ok: false,
      json: async () => ({ message: 'Token expired' })
    });

    await expect(apiFetch('/api/encounters/waiting')).rejects.toThrow('Token expired');

    expect(values.has('token')).toBe(false);
    expect(dispatchSpy).toHaveBeenCalledWith(expect.objectContaining({ type: 'nursehelpdesk:logout' }));
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/encounters/waiting'),
      expect.objectContaining({ headers: expect.objectContaining({ Authorization: 'Bearer valid-looking-token' }) })
    );
  });

  it('keeps the current session intact when login credentials are rejected', async () => {
    fetch.mockResolvedValue({
      status: 401,
      ok: false,
      json: async () => ({ message: 'Invalid email or password' })
    });

    await expect(apiFetch('/api/auth/login', { method: 'POST' })).rejects.toThrow('Invalid email or password');

    expect(values.get('token')).toBe('valid-looking-token');
    expect(dispatchSpy).not.toHaveBeenCalled();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/auth/login'),
      expect.objectContaining({ headers: expect.not.objectContaining({ Authorization: expect.any(String) }) })
    );
  });
});
