/**
 * Decode JWT token safely
 * @param {string} token - JWT token
 * @returns {object|null} Decoded payload or null if invalid
 */
export function decodeJwt(token) {
  try {
    if (!token) return null;
    const payload = token.split(".")[1];
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(base64);
    return JSON.parse(decodeURIComponent(escape(json)));
  } catch {
    return null;
  }
}

/**
 * Get user info from JWT token
 * @param {string} token - JWT token
 * @returns {object} User info with id, name, email, role, access
 */
export function getUserFromToken(token) {
  const decoded = decodeJwt(token);
  if (!decoded) return null;

  return {
    id: decoded.id,
    name: decoded.name || "Staff",
    email: decoded.email,
    role: decoded.role || "nurse",
    access: decoded.access || [],
    permissions: decoded.permissions || []
  };
}
