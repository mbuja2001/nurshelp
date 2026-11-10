import { HOSPITAL_STAFF_ROLES } from '../config/hospitalStaffRoleCatalog';

export const getConfiguredDepartmentSpecialties = (departments, departmentName) => {
  const department = (Array.isArray(departments) ? departments : []).find((item) => (
    item?.isActive !== false
    && String(item?.name || '').trim().toLowerCase() === String(departmentName || '').trim().toLowerCase()
  ));
  const specialties = (department?.specialties || [])
    .map((item) => String(item?.name || item || '').trim())
    .filter(Boolean);
  return [...new Set(specialties)];
};

export const resolveStaffPositionCode = (staff) => {
  const savedPosition = HOSPITAL_STAFF_ROLES.find((position) => (
    position.code === String(staff?.positionCode || '').trim().toUpperCase()
  ));
  if (savedPosition) return savedPosition.code;

  const positionTitle = String(staff?.positionTitle || '').trim().toLowerCase();
  const titleMatch = HOSPITAL_STAFF_ROLES.find((position) => position.label.toLowerCase() === positionTitle);
  if (titleMatch) return titleMatch.code;

  const legacyPositionByRole = {
    PHYSICIAN: 'PHYSICIAN',
    TECHNICIAN: 'TECHNICIAN',
    PHARMACIST: 'PHARMACIST',
    RADIOLOGIST: 'RADIOLOGIST',
    PATHOLOGIST: 'PATHOLOGIST'
  };
  return legacyPositionByRole[String(staff?.systemRole || staff?.role || '').trim().toUpperCase()] || '';
};

export const resolveConfiguredDepartmentName = (departments, departmentReference) => {
  const value = String(departmentReference || '').trim().toUpperCase();
  const legacyAliases = {
    EMERGENCY: 'ED',
    GENERAL_WARD: 'GENERAL'
  };
  const normalizedValue = legacyAliases[value] || value;
  return (Array.isArray(departments) ? departments : []).find((department) => (
    department?.isActive !== false
    && (
      String(department?.name || '').trim().toUpperCase() === normalizedValue
      || String(department?.departmentId || '').trim().toUpperCase() === value
    )
  ))?.name || '';
};
