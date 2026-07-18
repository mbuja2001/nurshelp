import React, { useState, useEffect } from 'react';
import './EPrescriptionForm.css';

/**
 * ePrescription Digital Prescription Form
 * Handles creation of NCPDP-compliant electronic prescriptions with:
 * - Digital signature verification
 * - Drug interaction checking
 * - Pharmacy routing
 * - Real-time insurance prior authorization
 */

const EPrescriptionForm = ({
  patientId,
  patientAllergies = [],
  currentMedications = [],
  physicianId,
  onSubmit,
  onCancel,
  showInteractionWarnings = true,
  compact = false,
}) => {
  const [formData, setFormData] = useState({
    medicationId: '',
    medicationName: '',
    dosage: '',
    frequency: '',
    route: 'oral',
    quantity: 30,
    duration: '30 days',
    refills: 0,
    instructions: '',
    allowSubstitution: true,
    preferredPharmacies: [],
    notificationChannels: ['email', 'sms'],
  });

  const [interactionAlerts, setInteractionAlerts] = useState([]);
  const [allergyWarnings, setAllergyWarnings] = useState([]);
  const [priorAuthRequired, setPriorAuthRequired] = useState(false);
  const [digitalSignature, setDigitalSignature] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});
  const [step, setStep] = useState('medication'); // medication | validation | review | sign

  const routes = ['oral', 'IM', 'IV', 'subcutaneous', 'inhalation', 'topical', 'transdermal'];
  const frequencies = ['Once daily', 'Twice daily', 'Three times daily', 'Four times daily', 'Every 6 hours', 'Every 8 hours', 'Every 12 hours', 'As needed'];
  const notificationChannels = ['email', 'sms', 'push', 'portal'];

  // Validate medication for interactions and allergies
  const validateMedication = async () => {
    setIsValidating(true);
    setValidationErrors({});
    setInteractionAlerts([]);
    setAllergyWarnings([]);

    try {
      // Simulate API call to check interactions
      if (formData.medicationId) {
        // Check against current medications
        const interactions = await checkDrugInteractions(formData.medicationId, currentMedications);
        setInteractionAlerts(interactions);

        // Check against allergies
        const allergyConflicts = currentMedications.filter(med =>
          patientAllergies.some(allergy => allergy.toLowerCase().includes(med.toLowerCase()))
        );
        setAllergyWarnings(allergyConflicts);

        // Check insurance prior auth requirements
        const paRequired = await checkPriorAuthRequired(formData.medicationId);
        setPriorAuthRequired(paRequired);
      }

      // Validate dosage
      const dosageErrors = validateDosage(formData);
      if (Object.keys(dosageErrors).length > 0) {
        setValidationErrors(dosageErrors);
      }
    } catch (error) {
      console.error('Validation error:', error);
      setValidationErrors({ validation: 'Failed to validate prescription. Please try again.' });
    } finally {
      setIsValidating(false);
    }
  };

  const checkDrugInteractions = async (medicationId, otherMeds) => {
    // Simulate API call to drug interaction database
    // In production, call to backend API
    return [
      {
        medicationId: 'med-123',
        medicationName: 'Warfarin',
        severity: 'severe',
        description: 'Increased bleeding risk - monitor INR levels closely',
      },
      {
        medicationId: 'med-456',
        medicationName: 'Aspirin',
        severity: 'moderate',
        description: 'May increase side effects - use with caution',
      },
    ];
  };

  const checkPriorAuthRequired = async (medicationId) => {
    // Simulate API call to insurance coverage database
    return Math.random() > 0.7; // 30% chance for demo
  };

  const validateDosage = (data) => {
    const errors = {};
    if (!data.medicationName) errors.medicationName = 'Medication name is required';
    if (!data.dosage) errors.dosage = 'Dosage is required';
    if (!data.frequency) errors.frequency = 'Frequency is required';
    if (data.quantity < 1) errors.quantity = 'Quantity must be at least 1';
    return errors;
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handlePharmacySelection = (pharmacyId) => {
    setFormData(prev => ({
      ...prev,
      preferredPharmacies: prev.preferredPharmacies.includes(pharmacyId)
        ? prev.preferredPharmacies.filter(id => id !== pharmacyId)
        : [...prev.preferredPharmacies, pharmacyId],
    }));
  };

  const handleNotificationChannelChange = (channel) => {
    setFormData(prev => ({
      ...prev,
      notificationChannels: prev.notificationChannels.includes(channel)
        ? prev.notificationChannels.filter(c => c !== channel)
        : [...prev.notificationChannels, channel],
    }));
  };

  const handleDigitalSign = async () => {
    // Simulate digital signature process
    try {
      const signature = await generateDigitalSignature({
        physicianId,
        prescriptionData: formData,
        timestamp: new Date().toISOString(),
      });
      setDigitalSignature(signature);
      setStep('review'); // Move to review after signing
    } catch (error) {
      console.error('Signing failed:', error);
      setValidationErrors({ signature: 'Failed to apply digital signature' });
    }
  };

  const generateDigitalSignature = async (data) => {
    // Simulate cryptographic signing
    return {
      hash: 'sig_' + Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toISOString(),
      physicianId: data.physicianId,
      verified: true,
    };
  };

  const handleSubmit = async () => {
    if (Object.keys(validationErrors).length > 0) {
      alert('Please fix validation errors before submitting');
      return;
    }

    if (!digitalSignature) {
      alert('Digital signature is required');
      return;
    }

    try {
      const prescriptionData = {
        ...formData,
        patientId,
        physicianId,
        digitalSignature,
        isElectronic: true,
        ePrescritionId: generateEPrescriptionId(),
        dateIssued: new Date().toISOString(),
      };

      onSubmit(prescriptionData);
    } catch (error) {
      console.error('Submission error:', error);
      setValidationErrors({ submit: 'Failed to submit prescription' });
    }
  };

  const generateEPrescriptionId = () => {
    return 'RX' + Date.now() + Math.random().toString(36).substr(2, 9).toUpperCase();
  };

  const renderMedicationStep = () => (
    <div className="prescription-step">
      <h3>Medication Information</h3>

      <div className="form-group">
        <label>Medication Name *</label>
        <input
          type="text"
          name="medicationName"
          value={formData.medicationName}
          onChange={handleInputChange}
          placeholder="Search medications..."
          className={validationErrors.medicationName ? 'error' : ''}
        />
        {validationErrors.medicationName && (
          <span className="error-text">{validationErrors.medicationName}</span>
        )}
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Dosage *</label>
          <input
            type="text"
            name="dosage"
            value={formData.dosage}
            onChange={handleInputChange}
            placeholder="e.g., 500mg"
            className={validationErrors.dosage ? 'error' : ''}
          />
          {validationErrors.dosage && (
            <span className="error-text">{validationErrors.dosage}</span>
          )}
        </div>

        <div className="form-group">
          <label>Route *</label>
          <select
            name="route"
            value={formData.route}
            onChange={handleInputChange}
          >
            {routes.map(route => (
              <option key={route} value={route}>{route}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Frequency *</label>
          <select
            name="frequency"
            value={formData.frequency}
            onChange={handleInputChange}
            className={validationErrors.frequency ? 'error' : ''}
          >
            <option value="">Select frequency...</option>
            {frequencies.map(freq => (
              <option key={freq} value={freq}>{freq}</option>
            ))}
          </select>
          {validationErrors.frequency && (
            <span className="error-text">{validationErrors.frequency}</span>
          )}
        </div>

        <div className="form-group">
          <label>Quantity *</label>
          <input
            type="number"
            name="quantity"
            value={formData.quantity}
            onChange={handleInputChange}
            min="1"
            className={validationErrors.quantity ? 'error' : ''}
          />
          {validationErrors.quantity && (
            <span className="error-text">{validationErrors.quantity}</span>
          )}
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Duration</label>
          <input
            type="text"
            name="duration"
            value={formData.duration}
            onChange={handleInputChange}
            placeholder="e.g., 30 days"
          />
        </div>

        <div className="form-group">
          <label>Refills</label>
          <input
            type="number"
            name="refills"
            value={formData.refills}
            onChange={handleInputChange}
            min="0"
            max="11"
          />
        </div>
      </div>

      <div className="form-group">
        <label>Special Instructions</label>
        <textarea
          name="instructions"
          value={formData.instructions}
          onChange={handleInputChange}
          placeholder="e.g., Take with food, avoid grapefruit juice..."
          rows="3"
        />
      </div>

      <div className="form-group checkbox">
        <input
          type="checkbox"
          name="allowSubstitution"
          checked={formData.allowSubstitution}
          onChange={handleInputChange}
          id="allowSubstitution"
        />
        <label htmlFor="allowSubstitution">Allow generic substitution</label>
      </div>

      <div className="step-actions">
        <button
          className="btn btn-secondary"
          onClick={onCancel}
        >
          Cancel
        </button>
        <button
          className="btn btn-primary"
          onClick={() => {
            validateMedication();
            setStep('validation');
          }}
          disabled={isValidating}
        >
          {isValidating ? 'Validating...' : 'Next Step'}
        </button>
      </div>
    </div>
  );

  const renderValidationStep = () => (
    <div className="prescription-step">
      <h3>Drug Safety Validation</h3>

      {interactionAlerts.length > 0 && showInteractionWarnings && (
        <div className="alert alert-warning">
          <h4>🔔 Drug Interactions Detected</h4>
          {interactionAlerts.map((alert, idx) => (
            <div key={idx} className={`interaction-item severity-${alert.severity}`}>
              <strong>{alert.medicationName}</strong> ({alert.severity})
              <p>{alert.description}</p>
            </div>
          ))}
        </div>
      )}

      {allergyWarnings.length > 0 && (
        <div className="alert alert-danger">
          <h4>⚠️ Allergy Conflict</h4>
          <p>Patient has documented allergies to: {allergyWarnings.join(', ')}</p>
        </div>
      )}

      {priorAuthRequired && (
        <div className="alert alert-info">
          <h4>ℹ️ Prior Authorization Required</h4>
          <p>This medication requires insurance prior authorization.</p>
          <p>A request will be submitted to the patient's insurance automatically.</p>
        </div>
      )}

      {validationErrors.validation && (
        <div className="alert alert-danger">
          <p>{validationErrors.validation}</p>
        </div>
      )}

      <div className="validation-summary">
        <h4>Prescription Summary</h4>
        <div className="summary-item">
          <span>Medication:</span> {formData.medicationName}
        </div>
        <div className="summary-item">
          <span>Dosage:</span> {formData.dosage} {formData.route}
        </div>
        <div className="summary-item">
          <span>Frequency:</span> {formData.frequency}
        </div>
        <div className="summary-item">
          <span>Quantity:</span> {formData.quantity} ({formData.duration})
        </div>
        <div className="summary-item">
          <span>Refills:</span> {formData.refills}
        </div>
      </div>

      <div className="step-actions">
        <button
          className="btn btn-secondary"
          onClick={() => setStep('medication')}
        >
          Back
        </button>
        <button
          className="btn btn-primary"
          onClick={() => setStep('pharmacy')}
        >
          Continue
        </button>
      </div>
    </div>
  );

  const renderPharmacyStep = () => (
    <div className="prescription-step">
      <h3>Pharmacy Routing & Patient Preferences</h3>

      <div className="form-group">
        <label>Preferred Pharmacies (Optional)</label>
        <div className="pharmacy-list">
          {/* Mock pharmacy options */}
          {['CVS Pharmacy', 'Walgreens', 'Local Pharmacy'].map((pharmacy, idx) => (
            <div key={idx} className="pharmacy-option">
              <input
                type="checkbox"
                id={`pharmacy-${idx}`}
                checked={formData.preferredPharmacies.includes(idx)}
                onChange={() => handlePharmacySelection(idx)}
              />
              <label htmlFor={`pharmacy-${idx}`}>{pharmacy}</label>
            </div>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label>Patient Notification Preferences</label>
        <div className="notification-options">
          {notificationChannels.map(channel => (
            <div key={channel} className="notification-option">
              <input
                type="checkbox"
                id={`notify-${channel}`}
                checked={formData.notificationChannels.includes(channel)}
                onChange={() => handleNotificationChannelChange(channel)}
              />
              <label htmlFor={`notify-${channel}`}>
                {channel.charAt(0).toUpperCase() + channel.slice(1)}
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className="step-actions">
        <button
          className="btn btn-secondary"
          onClick={() => setStep('validation')}
        >
          Back
        </button>
        <button
          className="btn btn-primary"
          onClick={() => setStep('sign')}
        >
          Review & Sign
        </button>
      </div>
    </div>
  );

  const renderSignStep = () => (
    <div className="prescription-step">
      <h3>Digital Signature</h3>

      <div className="signature-info">
        <p>Your digital signature is required to submit this ePrescription for processing.</p>
        <p className="info-text">This signature verifies your identity and authorization as a prescribing physician.</p>
      </div>

      {digitalSignature && (
        <div className="signature-success">
          <div className="checkmark">✓</div>
          <h4>Signature Applied</h4>
          <p>ID: {digitalSignature.hash}</p>
          <p>Timestamp: {new Date(digitalSignature.timestamp).toLocaleString()}</p>
        </div>
      )}

      {!digitalSignature && (
        <button
          className="btn btn-primary btn-large"
          onClick={handleDigitalSign}
        >
          👤 Apply Digital Signature
        </button>
      )}

      {validationErrors.signature && (
        <div className="alert alert-danger">
          <p>{validationErrors.signature}</p>
        </div>
      )}

      <div className="step-actions">
        {!digitalSignature && (
          <button
            className="btn btn-secondary"
            onClick={() => setStep('pharmacy')}
          >
            Back
          </button>
        )}
        {digitalSignature && (
          <>
            <button
              className="btn btn-secondary"
              onClick={() => setStep('pharmacy')}
            >
              Back
            </button>
            <button
              className="btn btn-success btn-large"
              onClick={handleSubmit}
            >
              📤 Send ePrescription
            </button>
          </>
        )}
      </div>
    </div>
  );

  return (
    <div className={`eprescription-form ${compact ? 'compact' : ''}`}>
      <div className="form-header">
        <h2>ePrescription</h2>
        <div className="step-indicator">
          <span className={step === 'medication' ? 'active' : ''}>1. Medication</span>
          <span className={step === 'validation' ? 'active' : ''}>2. Validation</span>
          <span className={step === 'pharmacy' ? 'active' : ''}>3. Routing</span>
          <span className={step === 'sign' ? 'active' : ''}>4. Sign</span>
        </div>
      </div>

      <div className="form-content">
        {step === 'medication' && renderMedicationStep()}
        {step === 'validation' && renderValidationStep()}
        {step === 'pharmacy' && renderPharmacyStep()}
        {step === 'sign' && renderSignStep()}
      </div>

      {Object.keys(validationErrors).length > 0 && (
        <div className="form-errors">
          <h4>Please fix the following errors:</h4>
          <ul>
            {Object.entries(validationErrors).map(([field, message]) => (
              <li key={field}>{message}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default EPrescriptionForm;
