export const departmentCodeFromName = (name) => `DEPT-${String(name || '').trim().toUpperCase().replace(/[^A-Z0-9]+/g, '-')}`;

export const resolveConfiguredDepartment = (departments, departmentReference) => {
  const reference = String(departmentReference || '').trim();
  if (!reference || !Array.isArray(departments)) return null;

  const normalizedReference = reference.toUpperCase();
  const department = departments.find((candidate) => (
    candidate?.isActive !== false
    && (
      String(candidate.departmentId || '').trim().toUpperCase() === normalizedReference
      || String(candidate.name || '').trim().toUpperCase() === normalizedReference
    )
  ));
  if (!department) return null;

  return {
    departmentId: String(department.departmentId || departmentCodeFromName(department.name)),
    departmentName: String(department.name).trim(),
  };
};
