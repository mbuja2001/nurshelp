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

const TECHNICIAN_ROLES = new Set([
  "TECHNICIAN",
  "PHARMACY_TECHNICIAN",
  "RADIOLOGY_TECHNICIAN",
  "PATHOLOGY_TECHNICIAN",
  "LAB_TECHNICIAN",
  "LABORATORY_TECHNICIAN"
]);

export function resolveTechnicianSpecialty(role, specialty = "") {
  const normalizedRole = String(role || "").toUpperCase().trim();
  if (!TECHNICIAN_ROLES.has(normalizedRole)) return null;

  if (normalizedRole === "PHARMACY_TECHNICIAN") return "PHARMACY";
  if (normalizedRole === "RADIOLOGY_TECHNICIAN") return "RADIOLOGY";
  if (normalizedRole === "PATHOLOGY_TECHNICIAN") return "PATHOLOGY";

  const normalizedSpecialty = String(specialty || "").toUpperCase().trim();
  if (/\bPHARMACY\b|\bPHARMACIST\b/.test(normalizedSpecialty)) return "PHARMACY";
  if (/\bRADIOLOGY\b|\bRADIOLOGIST\b/.test(normalizedSpecialty)) return "RADIOLOGY";
  if (/\bPATHOLOGY\b|\bPATHOLOGIST\b/.test(normalizedSpecialty)) return "PATHOLOGY";
  return null;
}

/**
 * Normalize identity roles without upgrading technicians to specialist roles.
 */
export function normalizeRoleLabel(role, specialty = "") {
  const normalizedRole = String(role || "").toUpperCase().trim();
  const normalizedSpecialty = String(specialty || "").toUpperCase().trim();

  if (["PHARMACIST", "PHARMACY"].includes(normalizedRole)) {
    return "PHARMACIST";
  }

  if (["RADIOLOGIST", "RADIOLOGY"].includes(normalizedRole)) {
    return "RADIOLOGIST";
  }

  if (["PATHOLOGIST", "PATHOLOGY"].includes(normalizedRole)) {
    return "PATHOLOGIST";
  }

  if (["ADMINISTRATIVE", "ADMIN", "MANAGER", "SHIFT_MANAGER", "DEPARTMENT_MANAGER", "SUPERUSER"].includes(normalizedRole)) {
    return "ADMIN";
  }

  if (TECHNICIAN_ROLES.has(normalizedRole)) return "TECHNICIAN";

  if (normalizedRole === "SPECIALIST") {
    if (/\bPHARMACY\b|\bPHARMACIST\b/.test(normalizedSpecialty)) return "PHARMACIST";
    if (/\bRADIOLOGY\b|\bRADIOLOGIST\b/.test(normalizedSpecialty)) return "RADIOLOGIST";
    if (/\bPATHOLOGY\b|\bPATHOLOGIST\b/.test(normalizedSpecialty)) return "PATHOLOGIST";
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
  if (normalizedRole === "TECHNICIAN") {
    return resolveTechnicianSpecialty(role, specialty)
      ? "TECHNICIAN"
      : "TECHNICIAN_UNCONFIGURED";
  }

  if (normalizedRole === 'PHARMACIST') return 'PHARMACY';
  if (normalizedRole === 'RADIOLOGIST') return 'RADIOLOGY';
  if (normalizedRole === 'PATHOLOGIST') return 'PATHOLOGY';

  return 'PHYSICIAN';
}

export function getRoleDisplayDepartment(role, specialty = "") {
  const normalizedRole = normalizeRoleLabel(role, specialty);
  const normalizedSpecialty = String(specialty || '').trim();
  const technicianSpecialty = resolveTechnicianSpecialty(role, specialty);

  if (normalizedRole === 'TECHNICIAN') {
    return technicianSpecialty
      ? `${technicianSpecialty.charAt(0)}${technicianSpecialty.slice(1).toLowerCase()} Technician Department`
      : 'Technician Department';
  }

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
