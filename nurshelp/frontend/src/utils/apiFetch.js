import { clearAuthSession } from './authSession';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';
const PUBLIC_ENDPOINTS = new Set([
  '/api/auth/login',
  '/api/auth/register-account',
  '/api/hospital/register'
]);

export const apiFetch = async (url, options = {}) => {
  const token = localStorage.getItem('token');
  const isPublicEndpoint = PUBLIC_ENDPOINTS.has(url);
  const response = await fetch(`${BACKEND_URL}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
      ...(token && !isPublicEndpoint ? { Authorization: `Bearer ${token}` } : {})
    }
  });

  const data = await response.json().catch(() => ({}));

  if (response.status === 401) {
    if (token && !isPublicEndpoint) clearAuthSession();
    throw new Error(data.message || (token ? 'Your session has expired. Please sign in again.' : 'Authentication failed.'));
  }

  if (!response.ok) {
    throw new Error(data.message || 'Request failed');
  }

  return data;
};
