/** @file Canonical forecast-model identifiers shared by diagnostic and research UIs. */

export const FORECAST_MODELS = [
  { id: 'ridge_direct', label: 'Direct Ridge' },
  { id: 'ridge_history_trend', label: 'Ridge + History/Trend' },
  { id: 'dlinear', label: 'DLinear' },
  { id: 'lstm', label: 'LSTM' },
  { id: 'tcn', label: 'TCN' },
  { id: 'patchtst', label: 'PatchTST' },
];

export const DEFAULT_FORECAST_MODEL_ID = 'ridge_history_trend';
export const FORECAST_MODEL_IDS = new Set(
  FORECAST_MODELS.map((model) => model.id),
);

export function forecastModelLabel(modelId) {
  return FORECAST_MODELS.find((model) => model.id === modelId)?.label || modelId;
}
