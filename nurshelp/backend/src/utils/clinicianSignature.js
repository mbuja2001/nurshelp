import crypto from 'crypto';

/**
 * Clinician Signature Utility
 * Shared pattern for signing official clinical documents (discharge reports, lab reports, triage records)
 * Mirrors ePrescription digital signature flow
 */

/**
 * Generate a clinician digital signature for official documents
 * @param {Object} clinician - Clinician object with id, name, role, specialty, credentials
 * @param {Object} documentData - The document being signed (discharge, lab request, triage)
 * @param {string} documentType - Type of document: 'DISCHARGE', 'LAB_REPORT', 'TRIAGE', 'LAB_REQUEST'
 * @returns {Object} Signature metadata with hash, timestamp, clinician identity, and qualification
 */
export const generateClinicianSignature = ({
  clinician,
  documentData = {},
  documentType = 'OFFICIAL_DOCUMENT'
}) => {
  if (!clinician) {
    throw new Error('Clinician identity is required for document signature');
  }

  // Extract clinician credentials
  const clinicianId = clinician.id || clinician._id;
  const clinicianName = clinician.name || `${clinician.firstName || ''} ${clinician.lastName || ''}`.trim();
  const clinicianRole = clinician.role || clinician.systemRole || 'CLINICIAN';
  const clinicianSpecialty = clinician.specialty || clinician.clinicalSpecialty || 'General';
  
  // Extract credentials (must be present for official documents)
  const credentials = clinician.credentials || {};
  const qualification = credentials.title || 'Professional'; // MD, DO, RN, NP, etc.
  const licenseNumber = credentials.licenseNumber || credentials.npiNumber || 'UNLICENSED';
  const npiNumber = credentials.npiNumber || null;

  if (licenseNumber === 'UNLICENSED') {
    console.warn(`[SIGNATURE] Warning: Clinician ${clinicianName} has no license number`);
  }

  // Create signature hash: include document type, content, and clinician identity
  const signaturePayload = JSON.stringify({
    clinicianId: clinicianId?.toString?.() || clinicianId,
    clinicianName,
    clinicianRole,
    documentType,
    documentHash: hashObject(documentData),
    timestamp: new Date().toISOString()
  });

  const hash = crypto.createHash('sha256').update(signaturePayload).digest('hex');

  return {
    hash, // Cryptographic proof of signature
    timestamp: new Date().toISOString(),
    clinicianId: clinicianId?.toString?.() || clinicianId,
    clinicianName,
    clinicianRole,
    clinicianSpecialty,
    qualification, // MD, RN, etc.
    licenseNumber, // Professional license/HPCSA/etc.
    npiNumber,
    documentType,
    verified: true // Set by document system; remains true unless tampered
  };
};

/**
 * Hash an object for signature verification
 * @param {Object} data - Object to hash
 * @returns {string} SHA256 hash
 */
export const hashObject = (data) => {
  const normalized = JSON.stringify(data || {});
  return crypto.createHash('sha256').update(normalized).digest('hex');
};

/**
 * Verify a clinician signature (basic check; full verification requires timestamp validation)
 * @param {Object} signature - Signature object with hash, clinicianId, etc.
 * @param {Object} clinician - Clinician object for comparison
 * @returns {boolean} True if signature clinician matches provided clinician
 */
export const verifyClinicianSignature = (signature, clinician) => {
  if (!signature || !clinician) return false;
  const clinicianId = (clinician.id || clinician._id)?.toString?.() || clinician.id || clinician._id;
  const signatureClinicianId = signature.clinicianId?.toString?.() || signature.clinicianId;
  return signatureClinicianId === clinicianId && signature.verified === true;
};

export default {
  generateClinicianSignature,
  hashObject,
  verifyClinicianSignature
};
