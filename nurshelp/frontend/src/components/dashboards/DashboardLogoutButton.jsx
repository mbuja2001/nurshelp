import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FiLogOut } from 'react-icons/fi';
import { clearAuthSession } from '../../utils/authSession';

const DashboardLogoutButton = ({ onLogout }) => {
  const navigate = useNavigate();

  const logout = () => {
    clearAuthSession();
    if (onLogout) onLogout();
    navigate('/dashboard?login=1', { replace: true });
  };

  return (
    <button type="button" className="routed-dashboard-logout" onClick={logout}>
      <FiLogOut aria-hidden="true" /> Sign out
    </button>
  );
};

export default DashboardLogoutButton;