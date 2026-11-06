export const HOSPITAL_DEPARTMENTS = [
  { name: 'ED', serviceTypes: ['OUTPATIENT', 'INPATIENT'] },
  { name: 'ICU', serviceTypes: ['INPATIENT'] },
  { name: 'OR', serviceTypes: ['INPATIENT'] },
  { name: 'OB', serviceTypes: ['OUTPATIENT', 'INPATIENT'] },
  { name: 'Medical-Surgical', serviceTypes: ['OUTPATIENT', 'INPATIENT'] },
  { name: 'Pediatrics', serviceTypes: ['OUTPATIENT', 'INPATIENT'] },
  { name: 'Psychiatry', serviceTypes: ['OUTPATIENT', 'INPATIENT'] },
  { name: 'General', serviceTypes: ['OUTPATIENT', 'INPATIENT', 'CLINICAL_SUPPORT'] }
];

export const HOSPITAL_SPECIALTIES = [
  ['Cardiology', 'Medical'], ['Emergency Medicine', 'Medical'], ['Gastroenterology', 'Medical'],
  ['Geriatrics', 'Medical'], ['Infectious Disease', 'Medical'], ['Internal Medicine', 'Medical'],
  ['Nephrology', 'Medical'], ['Neurology', 'Medical'], ['Oncology', 'Medical'],
  ['Pulmonology', 'Medical'], ['Psychiatry', 'Medical'], ['General Surgery', 'Surgical'],
  ['Cardiothoracic Surgery', 'Surgical'], ['Neurosurgery', 'Surgical'], ['Orthopedics', 'Surgical'],
  ['OBGYN', 'Maternal & Child'], ['Pediatrics', 'Maternal & Child'], ['Neonatal Medicine', 'Maternal & Child'],
  ['Pediatric ICU', 'Maternal & Child'], ['Clinical Pathology', 'Diagnostic & Support'],
  ['Radiology', 'Diagnostic & Support'], ['Clinical Pharmacist', 'Diagnostic & Support'],
  ['Hospital Pharmacist', 'Diagnostic & Support'], ['CT Technician', 'Diagnostic & Support'],
  ['MRI Technician', 'Diagnostic & Support'], ['Ultrasound Technician', 'Diagnostic & Support'],
  ['X-Ray Technician', 'Diagnostic & Support'], ['Lab Technician', 'Diagnostic & Support'],
  ['Hematology Technician', 'Diagnostic & Support'], ['Emergency Nursing', 'Nursing'],
  ['Critical Care Nursing', 'Nursing'], ['General Nursing', 'Nursing'], ['Geriatric Nursing', 'Nursing'],
  ['Maternity Nursing', 'Nursing'], ['Neonatal Nursing', 'Nursing'], ['Pediatric Nursing', 'Nursing'],
  ['Psychiatric Nursing', 'Nursing'], ['Cardiac Nursing', 'Nursing'], ['ICU Nursing', 'Nursing'],
  ['Infection Control Nursing', 'Nursing'], ['Neurosurgical Nursing', 'Nursing'],
  ['Operating Room Nursing', 'Nursing'], ['Orthopedic Nursing', 'Nursing'], ['Oncology Nursing', 'Nursing']
].map(([name, category]) => ({ name, category }));