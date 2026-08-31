const API_URL = import.meta.env.VITE_API_URL;

async function request(path, options) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const getMetrics = () => request("/metrics");
export const getMetricsTrends = (days) => request(`/metrics/trends?days=${days}`);
export const getModelInfo = () => request("/model-info");
export const getCaseFilterOptions = () => request("/cases/filter-options");
export const getCases = (params = {}) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, value);
  }
  const qs = search.toString();
  return request(`/cases${qs ? `?${qs}` : ""}`);
};
export const getCase = (id) => request(`/cases/${id}`);
export const getCaseReport = (id) => request(`/cases/${id}/report`);
export const getCaseReportFindings = (id) => request(`/cases/${id}/report-findings`);
export const getCasePrecedents = (id) => request(`/cases/${id}/precedents`);
export const postDecision = (id, decision) =>
  request(`/cases/${id}/decision`, { method: "POST", body: JSON.stringify(decision) });
export const reopenCase = (id, payload) =>
  request(`/cases/${id}/reopen`, { method: "POST", body: JSON.stringify(payload || {}) });
export const getPendingAiDecision = (id) => request(`/cases/${id}/pending-ai-decision`);
export const confirmAiDecision = (id, payload) =>
  request(`/cases/${id}/confirm-ai-decision`, { method: "POST", body: JSON.stringify(payload || {}) });
export const rejectAiDecision = (id, payload) =>
  request(`/cases/${id}/reject-ai-decision`, { method: "POST", body: JSON.stringify(payload) });
export const getAutomationStatus = () => request("/automation/status");
export const getCaseAutomationGates = (id) => request(`/cases/${id}/automation-gates`);
export const getCaseAuditTrail = (id) => request(`/cases/${id}/audit-trail`);
export const getSimulationTemplates = () => request("/simulation/templates");
export const runSimulation = (payload) =>
  request("/simulation/run", { method: "POST", body: JSON.stringify(payload) });

export const reportAssetUrl = (filename) => `${API_URL}/report-assets/${filename}`;

export const exportCasesCsvUrl = (params = {}) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, value);
  }
  search.set("format", "csv");
  return `${API_URL}/cases/export?${search.toString()}`;
};

export const exportCasesXlsxUrl = (params = {}) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, value);
  }
  search.set("format", "xlsx");
  return `${API_URL}/cases/export?${search.toString()}`;
};
