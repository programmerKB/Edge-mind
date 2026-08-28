/** @file JSON client for research experiments and current trajectory forecasts. */

import { RESEARCH_API_URL } from '../config.js';

/** Extract the most useful human-readable message from a failed API response. */
async function responseError(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    return `研究服務回應錯誤 (${response.status})`;
  }

  const detail = payload?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => item?.msg || item?.message)
      .filter(Boolean)
      .join('、');
  }
  return payload?.message || payload?.error || `研究服務回應錯誤 (${response.status})`;
}

async function request(path, options = {}) {
  const response = await fetch(`${RESEARCH_API_URL}${path}`, {
    ...options,
    headers: {
      Accept: 'application/json',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  });

  if (!response.ok) throw new Error(await responseError(response));
  if (response.status === 204) return null;
  return response.json();
}

/** Load available motors, protocol constraints, defaults, and model catalogue. */
export function getResearchConfig({ signal } = {}) {
  return request('/config', { signal });
}

/** Start one reproducible model-comparison experiment. */
export function createResearchExperiment(payload, { signal } = {}) {
  return request('/experiments', {
    method: 'POST',
    body: JSON.stringify(payload),
    signal,
  });
}

/** Read status and (when complete) all metrics for an experiment. */
export function getResearchExperiment(experimentId, { signal } = {}) {
  return request(`/experiments/${encodeURIComponent(experimentId)}`, { signal });
}

/** Fit the selected deployable model and forecast the newest exact history window. */
export function createResearchForecast(payload, { signal } = {}) {
  return request('/forecasts', {
    method: 'POST',
    body: JSON.stringify(payload),
    signal,
  });
}

/** Reload one immutable forecast artifact by identifier. */
export function getResearchForecast(forecastId, { signal } = {}) {
  return request(`/forecasts/${encodeURIComponent(forecastId)}`, { signal });
}
