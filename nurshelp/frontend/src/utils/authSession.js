const AUTH_STORAGE_KEYS = ['token', 'currentNurseId', 'currentStaffId', 'userRole', 'user'];

export const clearAuthSession = () => {
  AUTH_STORAGE_KEYS.forEach((key) => localStorage.removeItem(key));
  window.dispatchEvent(new Event('nursehelpdesk:logout'));
};
