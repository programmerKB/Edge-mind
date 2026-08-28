/** @file Research config, form state, experiment submission, and polling lifecycle. */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  createResearchExperiment,
  createResearchForecast,
  getResearchConfig,
  getResearchExperiment,
} from '../services/researchApi.js';

const DEFAULT_HORIZONS = [5, 10, 15, 20, 25, 30];
const TERMINAL_STATUSES = new Set(['completed', 'complete', 'succeeded', 'failed', 'error', 'cancelled']);

function motorIdentifier(motor) {
  return typeof motor === 'string'
    ? motor
    : motor?.id || motor?.motor_id || motor?.name || '';
}

function configDefaults(config) {
  const defaults = config?.defaults || config?.default_config || config?.configuration || {};
  const motors = config?.motors || config?.available_motors || config?.devices || config?.datasets || [];
  const firstMotor = motorIdentifier(motors[0]);
  const secondMotor = motorIdentifier(motors[1]) || firstMotor;
  const rawHorizons = defaults.horizons_minutes
    || defaults.forecast_horizons_minutes
    || config?.horizons_minutes
    || DEFAULT_HORIZONS;

  return {
    trainingMotorId: defaults.training_motor_id || firstMotor,
    evaluationMotorId: defaults.evaluation_motor_id || defaults.inference_motor_id || secondMotor,
    forecastModel: defaults.forecast_model || 'ridge_history_trend',
    historyMinutes: defaults.history_minutes ?? 60,
    horizonsMinutes: [...rawHorizons].map(Number).filter(Number.isFinite),
    thresholdC: defaults.threshold_c ?? defaults.temperature_threshold_c ?? 35,
  };
}

function experimentStatus(payload) {
  return String(payload?.status || payload?.experiment?.status || '').toLowerCase();
}

function experimentIdentifier(payload) {
  return payload?.id || payload?.experiment_id || payload?.experiment?.id || null;
}

function hasCompletedResult(payload) {
  const result = payload?.result || payload?.results || payload?.report || payload;
  return Boolean(
    result?.schema_version
    && result?.dataset
    && result?.models,
  );
}

function connectionMessage(error) {
  return error?.message === 'Failed to fetch'
    ? '目前無法連線至研究服務，請確認後端已啟動。'
    : error?.message || '研究服務發生未預期錯誤。';
}

/** Own the complete state machine used by the research workspace. */
export function useResearch() {
  const [config, setConfig] = useState(null);
  const [form, setForm] = useState(configDefaults(null));
  const [experiment, setExperiment] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [forecasting, setForecasting] = useState(false);
  const [configError, setConfigError] = useState('');
  const [experimentError, setExperimentError] = useState('');
  const [forecastError, setForecastError] = useState('');
  const [pollTick, setPollTick] = useState(0);
  const mountedRef = useRef(false);
  const configRequestRef = useRef(null);
  const submitRequestRef = useRef(null);
  const forecastRequestRef = useRef(null);

  const loadConfig = useCallback(async () => {
    configRequestRef.current?.abort();
    const controller = new AbortController();
    configRequestRef.current = controller;
    setConfigLoading(true);
    setConfigError('');

    try {
      const payload = await getResearchConfig({ signal: controller.signal });
      if (!mountedRef.current) return;
      setConfig(payload);
      setForm(configDefaults(payload));
    } catch (error) {
      if (error.name !== 'AbortError' && mountedRef.current) {
        setConfigError(connectionMessage(error));
      }
    } finally {
      if (mountedRef.current && configRequestRef.current === controller) {
        setConfigLoading(false);
        configRequestRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    loadConfig();
    return () => {
      mountedRef.current = false;
      configRequestRef.current?.abort();
      submitRequestRef.current?.abort();
      forecastRequestRef.current?.abort();
    };
  }, [loadConfig]);

  const updateField = useCallback((field, value) => {
    setForm((current) => ({ ...current, [field]: value }));
  }, []);

  const toggleHorizon = useCallback((horizon) => {
    setForm((current) => {
      const selected = current.horizonsMinutes.includes(horizon)
        ? current.horizonsMinutes.filter((value) => value !== horizon)
        : [...current.horizonsMinutes, horizon];
      return { ...current, horizonsMinutes: selected.sort((a, b) => a - b) };
    });
  }, []);

  const runExperiment = useCallback(async () => {
    if (submitting || forecasting || !form.horizonsMinutes.length) return;
    const threshold = Number(form.thresholdC);
    if (!Number.isFinite(threshold) || threshold < 20 || threshold > 120) {
      setExperimentError('警戒溫度必須是 20–120 °C 的有效數值。');
      return;
    }
    submitRequestRef.current?.abort();
    const controller = new AbortController();
    submitRequestRef.current = controller;
    setSubmitting(true);
    setExperimentError('');
    setExperiment(null);

    try {
      const payload = await createResearchExperiment({
        training_motor_id: form.trainingMotorId,
        evaluation_motor_id: form.evaluationMotorId,
        history_minutes: Number(form.historyMinutes),
        horizons_minutes: form.horizonsMinutes,
        threshold_c: threshold,
      }, { signal: controller.signal });
      if (mountedRef.current) setExperiment(payload);
    } catch (error) {
      if (error.name !== 'AbortError' && mountedRef.current) {
        setExperimentError(connectionMessage(error));
      }
    } finally {
      if (mountedRef.current && submitRequestRef.current === controller) {
        setSubmitting(false);
        submitRequestRef.current = null;
      }
    }
  }, [forecasting, form, submitting]);

  const runForecast = useCallback(async () => {
    if (forecasting || submitting || !form.horizonsMinutes.length) return;
    const threshold = Number(form.thresholdC);
    if (!Number.isFinite(threshold) || threshold < 20 || threshold > 120) {
      setForecastError('警戒溫度必須是 20–120 °C 的有效數值。');
      return;
    }
    forecastRequestRef.current?.abort();
    const controller = new AbortController();
    forecastRequestRef.current = controller;
    setForecasting(true);
    setForecastError('');
    setForecast(null);

    try {
      const payload = await createResearchForecast({
        motor_id: form.evaluationMotorId,
        training_motor_id: form.trainingMotorId,
        model_name: form.forecastModel,
        history_minutes: Number(form.historyMinutes),
        horizons_minutes: form.horizonsMinutes,
        threshold_c: threshold,
      }, { signal: controller.signal });
      if (mountedRef.current) setForecast(payload);
    } catch (error) {
      if (error.name !== 'AbortError' && mountedRef.current) {
        setForecastError(connectionMessage(error));
      }
    } finally {
      if (mountedRef.current && forecastRequestRef.current === controller) {
        setForecasting(false);
        forecastRequestRef.current = null;
      }
    }
  }, [forecasting, form, submitting]);

  const experimentId = experimentIdentifier(experiment);
  const status = experimentStatus(experiment);
  const shouldPoll = Boolean(experimentId)
    && !TERMINAL_STATUSES.has(status)
    && !hasCompletedResult(experiment);

  useEffect(() => {
    if (!shouldPoll) return undefined;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const payload = await getResearchExperiment(experimentId, {
          signal: controller.signal,
        });
        if (mountedRef.current) {
          setExperiment(payload);
          setExperimentError('');
        }
      } catch (error) {
        if (error.name !== 'AbortError' && mountedRef.current) {
          setExperimentError(`無法更新實驗進度：${connectionMessage(error)}`);
        }
      } finally {
        if (mountedRef.current) setPollTick((value) => value + 1);
      }
    }, 1500);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [experimentId, shouldPoll, pollTick]);

  return {
    config,
    form,
    experiment,
    forecast,
    configLoading,
    submitting,
    forecasting,
    configError,
    experimentError,
    forecastError,
    loadConfig,
    runExperiment,
    runForecast,
    toggleHorizon,
    updateField,
  };
}
