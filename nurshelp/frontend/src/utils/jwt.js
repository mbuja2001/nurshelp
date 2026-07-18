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
 * Normalize role labels from JWT payloads so specialist accounts from the
 * schema's technician role still route to the right dashboard.
 */
export function normalizeRoleLabel(role, specialty = "") {
  const normalizedRole = String(role || "").toUpperCase().trim();
  const normalizedSpecialty = String(specialty || "").toUpperCase().trim();

  if (["PHARMACIST", "PHARMACY_TECHNICIAN", "PHARMACY"].includes(normalizedRole) || normalizedSpecialty.includes("PHARM")) {
    return "PHARMACIST";
  }

  if (["RADIOLOGIST", "RADIOLOGY_TECHNICIAN", "RADIOLOGY"].includes(normalizedRole) || normalizedSpecialty.includes("RAD")) {
    return "RADIOLOGIST";
  }

  if (["PATHOLOGIST", "PATHOLOGY_TECHNICIAN", "PATHOLOGY"].includes(normalizedRole) || normalizedSpecialty.includes("PATH")) {
    return "PATHOLOGIST";
  }

  if (["ADMINISTRATIVE", "ADMIN", "MANAGER", "SHIFT_MANAGER", "DEPARTMENT_MANAGER"].includes(normalizedRole)) {
    return "ADMIN";
  }

  if (normalizedRole === "TECHNICIAN") {
    if (normalizedSpecialty.includes("PHARM")) return "PHARMACIST";
    if (normalizedSpecialty.includes("RAD")) return "RADIOLOGIST";
    if (normalizedSpecialty.includes("PATH")) return "PATHOLOGIST";
    return "TECHNICIAN";
  }

  return normalizedRole || "NURSE";
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
    role: normalizeRoleLabel(decoded.role, decoded.specialty || decoded.clinicalSpecialty || decoded.department),
    specialty: decoded.specialty || decoded.clinicalSpecialty || null,
    access: decoded.access || [],
    permissions: decoded.permissions || []
  };
}

export function getDashboardRoute(role, specialty = "") {
  const normalizedRole = normalizeRoleLabel(role, specialty);

  if (normalizedRole === 'PHARMACIST') return 'PHARMACY';
  if (normalizedRole === 'RADIOLOGIST') return 'RADIOLOGY';
  if (normalizedRole === 'PATHOLOGIST') return 'PATHOLOGY';

  return 'PHYSICIAN';
}

export function getRoleDisplayDepartment(role, specialty = "") {
  const normalizedRole = normalizeRoleLabel(role, specialty);
  const normalizedSpecialty = String(specialty || '').trim();

  if (normalizedRole === 'PHARMACIST') {
    return normalizedSpecialty ? `${normalizedSpecialty} Department` : 'Pharmacy Department';
  }

  if (normalizedRole === 'RADIOLOGIST') {
    return normalizedSpecialty ? `${normalizedSpecialty} Department` : 'Radiology Department';
  }

  if (normalizedRole === 'PATHOLOGIST') {
    return normalizedSpecialty ? `${normalizedSpecialty} Department` : 'Pathology Department';
  }

  if (normalizedSpecialty) {
    return `${normalizedSpecialty} Department`;
  }

  return 'Cardiology Department';
}
