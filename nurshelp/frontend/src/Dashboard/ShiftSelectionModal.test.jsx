// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ShiftSelectionModal from './ShiftSelectionModal';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('shift selection clock-in', () => {
  it('reports success and closes after clocking in', async () => {
    const onClose = vi.fn();
    const onResult = vi.fn();
    const onSelectShift = vi.fn();
    const clockRecord = { _id: 'clock-1' };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, shifts: [{ _id: 'shift-1', date: new Date().toISOString(), shiftStart: '08:00', shiftEnd: '16:00' }] })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, record: clockRecord })
      });
    vi.stubGlobal('fetch', fetchMock);

    render(<ShiftSelectionModal isOpen onClose={onClose} onSelectShift={onSelectShift} onResult={onResult} token="token" />);
    fireEvent.click(await screen.findByText('08:00 - 16:00'));
    fireEvent.click(screen.getByRole('button', { name: /clock in/i }));

    await waitFor(() => expect(onResult).toHaveBeenCalledWith({ success: true, message: 'Shift started successfully.' }));
    expect(onSelectShift).toHaveBeenCalledWith(clockRecord);
    expect(onClose).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('reports an API error and closes the modal', async () => {
    const onClose = vi.fn();
    const onResult = vi.fn();
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, shifts: [{ _id: 'shift-1', date: new Date().toISOString(), shiftStart: '08:00', shiftEnd: '16:00' }] })
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ success: false, message: 'Already clocked in today' })
      }));

    render(<ShiftSelectionModal isOpen onClose={onClose} onSelectShift={vi.fn()} onResult={onResult} token="token" />);
    fireEvent.click(await screen.findByText('08:00 - 16:00'));
    fireEvent.click(screen.getByRole('button', { name: /clock in/i }));

    await waitFor(() => expect(onResult).toHaveBeenCalledWith({ success: false, message: 'Already clocked in today' }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
