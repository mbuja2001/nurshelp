import { useState, useEffect } from 'react';
import { decodeJwt } from '../utils/jwt';

/**
 * Custom hook to extract user info from JWT token
 * @returns {object} User info with name and role
 */
export function useUserFromToken() {
  const [userName, setUserName] = useState('Staff');
  const [userRole, setUserRole] = useState('Nurse');

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      const decoded = decodeJwt(token);
      if (decoded) {
        setUserName(decoded.name || 'Staff');
        setUserRole(decoded.role || 'Nurse');
      }
    }
  }, []);

  return { userName, userRole };
}
