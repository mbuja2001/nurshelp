export const EHR_PROTOCOLS = ['hl7v2', 'fhir-r4'];

export const normalizeEhrProtocols = (protocols) => (
  [...new Set((Array.isArray(protocols) ? protocols : []).filter((protocol) => EHR_PROTOCOLS.includes(protocol)))]
);

export const resolveEhrIntegrationAccess = (integrationAccess, environment) => {
  const isSandbox = !environment || environment === 'SANDBOX';
  const enabledProtocols = isSandbox ? normalizeEhrProtocols(integrationAccess?.enabledProtocols) : [];
  return {
    source: integrationAccess?.source || null,
    enabledProtocols,
    canEnableSandboxAccess: isSandbox && !integrationAccess?.source,
    isSandboxConfiguration: isSandbox && integrationAccess?.source === 'HOSPITAL_ADMIN_SANDBOX',
    attestedAt: integrationAccess?.attestedAt || null,
    attestedByName: integrationAccess?.attestedByName || null,
    reason: integrationAccess?.reason || null,
  };
};
