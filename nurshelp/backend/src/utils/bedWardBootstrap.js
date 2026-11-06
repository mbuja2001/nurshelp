export const normalizeWardDepartment = (wardUnit = '') => {
  const value = String(wardUnit || '').trim();

  if (!value) return 'GENERAL';

  const normalized = value.toUpperCase();
  if (normalized === 'ED' || normalized.startsWith('ED ') || normalized.includes('EMERGENCY')) {
    return 'ED';
  }

  const mappings = {
    ICU: 'ICU',
    NICU: 'NICU',
    PICU: 'PICU',
    CICU: 'CICU',
    SICU: 'SICU',
    'NEURO ICU': 'NEURO',
    HDU: 'GENERAL',
    PCU: 'GENERAL',
    'MEDICAL WARD': 'GENERAL',
    'SURGICAL WARD': 'GENERAL',
    'ORTHOPAEDIC WARD': 'GENERAL',
    'ONCOLOGY WARD': 'GENERAL',
    'NEUROLOGY WARD': 'GENERAL',
    'GASTROENTEROLOGY WARD': 'GENERAL',
    'PEDIATRIC WARD': 'PEDIATRICS',
    'GERIATRIC WARD': 'GENERAL',
    'MATERNITY WARD': 'MATERNITY',
    'PSYCHIATRIC WARD': 'GENERAL',
    'INFECTIOUS DISEASE WARD': 'GENERAL',
    RADIOLOGY: 'GENERAL',
    PATHOLOGY: 'GENERAL',
    PHARMACY: 'GENERAL',
  };

  return mappings[normalized] || 'GENERAL';
};

export const inferBedType = (bed = {}) => {
  const equipment = Array.isArray(bed.equipment_baseline) ? bed.equipment_baseline : [];
  const ward = String(bed.ward_unit || '').toUpperCase();

  if (ward.includes('ICU') || equipment.some((item) => /ventilator|dialysis|defibrillator|high-acuity monitor/i.test(item))) {
    return 'INTENSIVE';
  }

  if (equipment.some((item) => /cardiac monitor|monitor|portable ultrasound|blood gas analyzer/i.test(item))) {
    return 'MONITORED';
  }

  return 'STANDARD';
};

export const buildWardDefinitionsFromBeds = (beds = []) => {
  const byDepartment = new Map();

  for (const bed of beds) {
    const department = normalizeWardDepartment(bed.ward_unit);

    if (!byDepartment.has(department)) {
      byDepartment.set(department, {
        department,
        wardName: `${department === 'ED' ? 'Emergency Department' : department === 'ICU' ? 'Intensive Care Unit' : department === 'PEDIATRICS' ? 'Pediatrics' : department === 'MATERNITY' ? 'Maternity' : 'General Ward'} - Ward A`,
        wardId: `${department}-Ward-A`,
        floor: department === 'ICU' ? 2 : department === 'ED' ? 1 : 3,
        wing: department === 'PEDIATRICS' || department === 'MATERNITY' ? 'East' : 'Main',
        bedCount: 0,
      });
    }

    const ward = byDepartment.get(department);
    ward.bedCount += 1;
  }

  return [...byDepartment.values()].map((ward) => ({
    ...ward,
    totalBeds: ward.bedCount,
  }));
};
