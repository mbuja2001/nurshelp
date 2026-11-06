export const FHIR_CANONICAL_FORMAT = 'FHIR R4';

export const protocolAdapterInterface = {
  canonicalFormat: FHIR_CANONICAL_FORMAT,
  description: 'Intended adapter contract; a live HL7 or FHIR connector is not currently provisioned.',
  contract: [
    'parse(rawMessage) -> CanonicalResource',
    'serialize(resource) -> Buffer',
    'validate(rawMessage) -> ValidationResult',
    're-verify hospital scoping on every inbound message'
  ]
};

export const ehrConfigurationBlueprint = {
  canonicalFormat: FHIR_CANONICAL_FORMAT,
  adapters: [
    {
      protocolName: 'HL7v2',
      status: 'Not configured',
      transport: 'MLLP',
      listener: 'Not configured',
      supportedMessages: [],
      lastTest: 'Not run',
      connectionStatus: 'Not configured'
    },
    {
      protocolName: 'FHIR R4',
      status: 'Not configured',
      transport: 'FHIR REST',
      listener: 'Not configured',
      supportedMessages: [],
      lastTest: 'Not run',
      connectionStatus: 'Not configured'
    }
  ],
  terminologyCrosswalk: [
    { localCode: 'RED', standardCode: 'SNOMED CT: 6937004', standardName: 'Severe pain', status: 'Mapped' },
    { localCode: 'YELLOW', standardCode: 'SNOMED CT: 24484000', standardName: 'Pain', status: 'Mapped' },
    { localCode: 'GREEN', standardCode: 'SNOMED CT: 162676003', standardName: 'No acute symptoms', status: 'Mapped' },
    { localCode: 'UNMAPPED-17', standardCode: null, standardName: 'Awaiting review', status: 'Needs review' }
  ],
  retentionPolicy: [
    { category: 'raw transcripts', retention: '90 days', owner: 'Clinical records' },
    { category: 'lab results', retention: '180 days', owner: 'Pathology' },
    { category: 'audit logs', retention: '365 days', owner: 'Security' },
    { category: 'location pings', retention: '30 days', owner: 'Operations' }
  ],
  alertRouting: [
    { alertType: 'QUEUE_CONGESTION', targets: ['Charge nurse', 'ED duty lead'], escalation: 'High' },
    { alertType: 'ACUITY_ESCALATION', targets: ['Physician on call', 'Clinical lead'], escalation: 'Critical' },
    { alertType: 'LAB_RESULT_READY', targets: ['Ordering clinician', 'Ward nurse'], escalation: 'Normal' }
  ],
  accessMatrix: [
    { role: 'Charge nurse', report: 'Live queue', visibility: 'Hospital-scoped' },
    { role: 'Physician', report: 'Encounter summary', visibility: 'Department + own patients' },
    { role: 'Clinical lead', report: 'Audit & escalation', visibility: 'Hospital-scoped' }
  ],
  protocolPolicy: {
    satsVersion: 'SATS v3.1',
    tewsVersion: 'TEWS v2.4',
    reportRequirements: 'System default',
    retentionGovernance: 'POPIA-aligned'
  },
  deadLetterQueue: {
    inboundFailures: 0,
    retryQueue: 0,
    lastFailure: null
  }
};
