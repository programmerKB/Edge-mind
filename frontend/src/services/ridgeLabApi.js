/** @file HTTP client for the compact dual-Ridge experiment workflow. */

import { resolveApiUrl, RIDGE_LAB_API_URL } from '../config.js';

const baseUrl = resolveApiUrl(RIDGE_LAB_API_URL).replace(/\/$/, '');

async function request(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.detail || `請求失敗（HTTP ${response.status}）`);
  }
  return payload;
}

export function loadRidgeLabConfig() {
  return request('/config');
}

export function runRidgeExperiment(trainingMotorId, evaluationMotorId) {
  return request('/experiments', {
    method: 'POST',
    body: JSON.stringify({
      training_motor_id: trainingMotorId,
      evaluation_motor_id: evaluationMotorId || null,
    }),
  });
}

export async function runRidgeForecast({ motorId, trainingMotorId, modelName }) {
  const forecast = await request('/forecasts', {
    method: 'POST',
    body: JSON.stringify({
      motor_id: motorId,
      training_motor_id: trainingMotorId,
      model_name: modelName,
    }),
  });
  return {
    ...forecast,
    attachments: (forecast.attachments || []).map((attachment) => ({
      ...attachment,
      url: new URL(attachment.url, baseUrl).toString(),
    })),
  };
}
