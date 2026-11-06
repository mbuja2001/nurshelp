export const ADMIN_WORKSPACE_VIEW_REGISTRY = {
  analytics: 'TriageAnalyticsAdminPage',
  'labs-configurations': 'LabRequestsAdminPage',
  'pharmacy-configurations': 'PharmacyAdminPage',
};

export const getAdminWorkspaceViewRegistration = (workspaceView) => ADMIN_WORKSPACE_VIEW_REGISTRY[workspaceView] || null;
