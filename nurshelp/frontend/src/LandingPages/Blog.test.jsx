// @vitest-environment jsdom

import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import Blog from './Blog';

const renderBlog = (entry = '/blog') => render(
  <MemoryRouter initialEntries={[entry]}>
    <Routes>
      <Route path="/blog" element={<Blog />} />
      <Route path="/blog/:slug" element={<Blog />} />
    </Routes>
  </MemoryRouter>
);

describe('public journal', () => {
  it('filters stories by topic and search text', () => {
    renderBlog();

    fireEvent.click(screen.getByRole('button', { name: 'Operations' }));
    expect(screen.getByRole('heading', { name: /building resilient workflows/i })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: /calmer handoffs/i })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'All stories' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Search stories' }), {
      target: { value: 'human-centered' }
    });
    expect(screen.getByRole('heading', { name: /human-centered decision support/i })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: /calmer handoffs/i })).toBeNull();
  });

  it('renders article content from a direct story URL', () => {
    renderBlog('/blog/designing-calmer-clinical-handoffs');

    expect(screen.getByRole('heading', { name: /designing calmer handoffs/i })).toBeTruthy();
    expect(screen.getByRole('link', { name: /back to the journal/i })).toBeTruthy();
  });

  it('publishes the South African patient-history problem and research limitations', () => {
    renderBlog('/blog/why-patient-history-needs-to-travel-with-care');

    expect(screen.getByRole('heading', { name: /why patient history needs to travel with care/i })).toBeTruthy();
    expect(screen.getByText(/fragmented EHR information across facilities/i)).toBeTruthy();

    renderBlog('/blog/longitudinal-context-for-local-ai');
    expect(screen.getByRole('heading', { name: /research note: longitudinal context for local AI/i })).toBeTruthy();
    expect(screen.getByText(/does not establish clinical validity/i)).toBeTruthy();
  });
});
