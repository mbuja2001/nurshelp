export const SUPPORT_TICKET_CATEGORIES = Object.freeze([
  'ACCESS',
  'SYSTEM_ERROR',
  'PERFORMANCE',
  'OTHER'
]);

export const SUPPORT_TICKET_SEVERITIES = Object.freeze([
  'LOW',
  'MEDIUM',
  'HIGH',
  'CRITICAL'
]);

export const SUPPORT_TICKET_STATUSES = Object.freeze([
  'NEW',
  'IN_PROGRESS',
  'RESOLVED'
]);

const boundedString = (value, maxLength) => (
  typeof value === 'string' ? value.trim().slice(0, maxLength) : ''
);

export const normalizeSupportTicketInput = (body = {}) => {
  const title = boundedString(body.title, 120);
  const description = boundedString(body.description, 2000);
  const category = String(body.category || '').toUpperCase();
  const severity = String(body.severity || '').toUpperCase();
  const pagePath = boundedString(body.pagePath, 240);
  const failedRequest = boundedString(body.failedRequest, 120);
  const errorCode = boundedString(body.errorCode, 80);
  const errorMessage = boundedString(body.errorMessage, 500);
  const primaryStatus = body.primaryStatus === null || body.primaryStatus === undefined
    ? null
    : Number(body.primaryStatus);
  const fallbackStatus = body.fallbackStatus === null || body.fallbackStatus === undefined
    ? null
    : Number(body.fallbackStatus);
  const occurredAt = body.occurredAt ? new Date(body.occurredAt) : new Date();

  if (title.length < 5) return { error: 'A title of at least 5 characters is required' };
  if (description.length < 10) return { error: 'A description of at least 10 characters is required' };
  if (!SUPPORT_TICKET_CATEGORIES.includes(category)) return { error: 'Select a valid support category' };
  if (!SUPPORT_TICKET_SEVERITIES.includes(severity)) return { error: 'Select a valid severity' };
  if (pagePath && (!pagePath.startsWith('/') || pagePath.includes('?') || pagePath.includes('#'))) {
    return { error: 'Page path must be a pathname without query or fragment data' };
  }
  if (failedRequest && !/^\/api\/[a-z0-9/_-]+$/i.test(failedRequest)) {
    return { error: 'Failed request must be a relative API path' };
  }
  if (![primaryStatus, fallbackStatus].every((status) => status === null || (Number.isInteger(status) && status >= 100 && status <= 599))) {
    return { error: 'HTTP statuses must be valid status codes' };
  }
  if (Number.isNaN(occurredAt.getTime())) return { error: 'Occurrence time must be a valid date' };

  return {
    value: {
      title,
      description,
      category,
      severity,
      pagePath,
      failedRequest,
      errorCode,
      errorMessage,
      primaryStatus,
      fallbackStatus,
      occurredAt
    }
  };
};

export const normalizeSupportTicketUpdate = (body = {}) => {
  const value = {};
  if (body.status !== undefined) {
    const status = String(body.status).toUpperCase();
    if (!SUPPORT_TICKET_STATUSES.includes(status)) return { error: 'Select a valid ticket status' };
    value.status = status;
  }

  if (body.assignToMe !== undefined) {
    if (typeof body.assignToMe !== 'boolean') return { error: 'assignToMe must be a boolean' };
    value.assignToMe = body.assignToMe;
  }

  if (body.resolutionNote !== undefined) {
    if (typeof body.resolutionNote !== 'string' || body.resolutionNote.trim().length > 1000) {
      return { error: 'Resolution note must be 1000 characters or fewer' };
    }
    value.resolutionNote = body.resolutionNote.trim();
  }

  if (!Object.keys(value).length) return { error: 'Provide a status, assignment, or resolution note update' };
  return { value };
};
