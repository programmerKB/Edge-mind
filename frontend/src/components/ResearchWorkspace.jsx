/** @file Reproducible multi-horizon forecasting research workspace. */

import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  FlaskConical,
  Gauge,
  Info,
  Layers3,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldCheck,
  Thermometer,
} from 'lucide-react';
import { useMemo } from 'react';
import { useResearch } from '../hooks/useResearch.js';
import {
  FORECAST_MODELS,
  FORECAST_MODEL_IDS,
  forecastModelLabel,
} from '../models/forecastModel.js';
import './ResearchWorkspace.css';

const FALLBACK_HORIZONS = [5, 10, 15, 20, 25, 30];
function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function numberValue(...values) {
  const value = firstDefined(...values);
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function reportOf(experiment) {
  return experiment?.result || experiment?.results || experiment?.report || experiment || {};
}

function motorId(motor) {
  return typeof motor === 'string'
    ? motor
    : motor?.id || motor?.motor_id || motor?.name || '';
}

function motorLabel(motor) {
  if (typeof motor === 'string') return motor;
  const id = motorId(motor);
  return motor?.label || motor?.display_name || motor?.name || id;
}

function formatNumber(value, digits = 2) {
  const number = numberValue(value);
  return number === null ? '—' : number.toFixed(digits);
}

function formatInteger(value) {
  const number = numberValue(value);
  return number === null ? '—' : Math.round(number).toLocaleString('zh-TW');
}

function formatBytes(value) {
  const bytes = numberValue(value);
  if (bytes === null) return '—';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function formatDurationMs(value) {
  const milliseconds = numberValue(value);
  if (milliseconds === null) return '—';
  if (milliseconds < 1) return `${milliseconds.toFixed(3)} ms`;
  if (milliseconds < 1000) return `${milliseconds.toFixed(2)} ms`;
  return `${(milliseconds / 1000).toFixed(2)} s`;
}

function modelsFromConfig(config) {
  const candidates = config?.models || config?.available_models || config?.model_catalogue || config?.model_capabilities;
  if (!candidates) return FORECAST_MODELS.map((model) => model.label);
  if (!Array.isArray(candidates)) {
    return Object.entries(candidates)
      .filter(([name, model]) => FORECAST_MODEL_IDS.has(model?.name || model?.id || name))
      .map(([name, model]) => model?.display_name || model?.label || model?.name || name);
  }
  return candidates
    .map((model) => typeof model === 'string'
      ? FORECAST_MODELS.find((supported) => supported.id === model)
      : {
        id: model.name || model.id || model.model_name,
        label: model.display_name || model.label || model.name || model.id,
      })
    .filter((model) => model?.id && FORECAST_MODEL_IDS.has(model.id))
    .map((model) => model.label);
}

function forecastModelOptions(config) {
  const candidates = config?.models || config?.available_models || [];
  if (!Array.isArray(candidates)) {
    return [{ id: 'ridge_history_trend', label: 'Ridge + History/Trend' }];
  }
  const available = candidates.map((model) => {
    if (typeof model === 'string') return { id: model, label: model };
    return {
      id: model.name || model.id || model.model_name,
      label: model.display_name || model.label || model.name || model.id,
      status: model.status,
    };
  }).filter((model) => (
    model.id && FORECAST_MODEL_IDS.has(model.id) && model.status !== 'unavailable'
  ));
  return available.length
    ? available
    : [{ id: 'ridge_history_trend', label: 'Ridge + History/Trend' }];
}

function horizonOptions(config) {
  const values = config?.allowed_horizons_minutes
    || config?.horizons_minutes
    || config?.defaults?.horizons_minutes
    || config?.default_config?.horizons_minutes
    || config?.configuration?.horizons_minutes
    || config?.constraints?.allowed_horizons_minutes
    || FALLBACK_HORIZONS;
  return [...new Set(values.map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
}

function modelRows(report) {
  const models = report.models || report.model_comparison || report.model_metrics || [];
  const entries = Array.isArray(models)
    ? models.map((model, index) => [model.name || model.model || `Model ${index + 1}`, model])
    : Object.entries(models);

  return entries.map(([name, model]) => {
    const test = model.test || model.evaluation || {};
    const validation = model.validation || {};
    const walkForward = model.walk_forward?.aggregate || {};
    const crossDevice = model.cross_device?.status === 'available' ? model.cross_device : null;
    const overall = test.overall || model.overall || model.metrics || model;
    const validationOverall = validation.overall || {};
    const walkForwardOverall = walkForward.overall || {};
    const crossOverall = crossDevice?.overall || {};
    const latency = model.efficiency?.inference_latency_ms;
    return {
      name: model.display_name || model.label || model.model_name || name,
      status: model.status || 'available',
      reason: model.reason || model.error || '',
      mae: numberValue(overall.mae, overall.mae_c, overall.mean_absolute_error),
      rmse: numberValue(overall.rmse, overall.rmse_c, overall.root_mean_squared_error),
      maxError: numberValue(overall.max_error, overall.max_error_c, overall.maximum_error),
      r2: numberValue(overall.r2, overall.r_squared, overall.r2_score),
      skillVsDirectRidge: numberValue(test.skill_score_vs_ridge_direct),
      targetStdDev: numberValue(test.target_distribution?.stddev_c),
      r2LowVarianceWarning: Boolean(test.target_distribution?.r2_low_variance_warning),
      crossDeviceMae: numberValue(crossOverall.mae, crossOverall.mae_c, crossOverall.mean_absolute_error),
      selectionMae: numberValue(
        walkForwardOverall.mae,
        walkForwardOverall.mae_c,
        validationOverall.mae,
        validationOverall.mae_c,
      ),
      selectionScope: numberValue(walkForwardOverall.mae, walkForwardOverall.mae_c) !== null
        ? '開發期 walk-forward'
        : 'validation',
      latencyMs: numberValue(
        typeof latency === 'object' ? firstDefined(latency.mean_ms, latency.mean) : latency,
        model.inference_latency_ms,
        model.latency_ms,
      ),
      sizeBytes: numberValue(
        model.efficiency?.model_size_bytes,
        model.model_size_bytes,
        numberValue(model.model_size_kb, model.size_kb) === null
          ? null
          : numberValue(model.model_size_kb, model.size_kb) * 1024,
      ),
      trainingMs: numberValue(model.efficiency?.training_time_ms, model.training_time_ms),
      byHorizon: crossDevice?.by_horizon || test.by_horizon || model.by_horizon || model.horizon_metrics,
      horizonScope: crossDevice ? '跨設備' : '保留測試集',
      risk: crossDevice?.risk || test.risk || model.risk || {},
      latestForecast: crossDevice?.latest_forecast || test.latest_forecast || model.latest_forecast || null,
    };
  });
}

function horizonRows(models, report) {
  const direct = report.horizon_metrics || report.by_horizon;
  const rows = [];

  const addMetrics = (modelName, horizon, metrics) => {
    if (!metrics || typeof metrics !== 'object') return;
    rows.push({
      model: modelName,
      horizon: numberValue(metrics.horizon_minutes, metrics.horizon, horizon),
      mae: numberValue(metrics.mae, metrics.mae_c, metrics.mean_absolute_error),
      rmse: numberValue(metrics.rmse, metrics.rmse_c, metrics.root_mean_squared_error),
      r2: numberValue(metrics.r2, metrics.r_squared, metrics.r2_score),
      scope: metrics.scope || '',
    });
  };

  const unfold = (modelName, values) => {
    if (Array.isArray(values)) {
      values.forEach((metrics) => addMetrics(modelName, null, metrics));
    } else if (values && typeof values === 'object') {
      Object.entries(values).forEach(([horizon, metrics]) => addMetrics(modelName, horizon, metrics));
    }
  };

  if (Array.isArray(direct)) {
    direct.forEach((metrics) => addMetrics(metrics.model || metrics.model_name || 'Model', null, metrics));
  } else if (direct && typeof direct === 'object') {
    Object.entries(direct).forEach(([name, values]) => unfold(name, values));
  } else {
    models.forEach((model) => {
      const before = rows.length;
      unfold(model.name, model.byHorizon);
      rows.slice(before).forEach((row) => { row.scope = model.horizonScope; });
    });
  }

  return rows
    .filter((row) => row.horizon !== null)
    .sort((a, b) => a.horizon - b.horizon || a.model.localeCompare(b.model));
}

function ablationRows(report) {
  const ablations = report.ablations || report.feature_ablation || report.ablation_results || [];
  return (Array.isArray(ablations) ? ablations : Object.values(ablations)).map((item, index) => {
    const test = item.test || item.evaluation || {};
    const overall = test.overall || item.overall || item.metrics || item;
    return {
      id: item.id || `ablation-${index}`,
      label: item.label || item.name || item.experiment || `組合 ${index + 1}`,
      model: item.model || item.model_name || '',
      modelLabel: item.model_display_name
        || item.model_label
        || forecastModelLabel(item.model || item.model_name || ''),
      evaluationScope: item.evaluation_scope || 'locked_test',
      features: Array.isArray(item.features) ? item.features.join(' + ') : item.features || '',
      status: item.status || 'available',
      reason: item.reason || item.error || '',
      mae: numberValue(overall.mae, overall.mae_c, overall.mean_absolute_error),
      rmse: numberValue(overall.rmse, overall.rmse_c, overall.root_mean_squared_error),
      r2: numberValue(overall.r2, overall.r_squared, overall.r2_score),
    };
  });
}

function riskSummary(report, models) {
  const direct = report.risk_forecast || report.risk_prediction || report.risk || {};
  const forecastCandidates = models.filter((model) => model.latestForecast);
  const bestModel = [...forecastCandidates]
    .filter((model) => model.selectionMae !== null)
    .sort((a, b) => a.selectionMae - b.selectionMae)[0]
    || forecastCandidates[0];
  const forecast = report.latest_forecast || bestModel?.latestForecast || {};
  const modelRisk = bestModel?.risk || {};
  const risk = {
    ...modelRisk,
    ...(forecast.risk || {}),
    ...direct,
  };
  const crossingMetrics = firstDefined(
    typeof direct.threshold_crossing === 'object' ? direct.threshold_crossing : null,
    typeof modelRisk.threshold_crossing === 'object' ? modelRisk.threshold_crossing : null,
    {},
  );
  const trajectory = direct.trajectory
    || direct.predictions
    || forecast.trajectory
    || report.trajectory
    || report.temperature_trajectory
    || [];
  const crossingValue = firstDefined(
    typeof risk.threshold_crossing === 'boolean' ? risk.threshold_crossing : undefined,
    risk.will_cross_threshold,
    risk.predicted_positive,
  );
  const configuredHorizons = report.configuration?.horizons_minutes || [];
  const trajectoryHorizons = Array.isArray(trajectory)
    ? trajectory.map((point) => numberValue(point.horizon_minutes, point.horizon)).filter((value) => value !== null)
    : [];
  return {
    sourceModel: bestModel?.name || report.model?.display_name || report.model?.name || '',
    deviceId: forecast.device_id || report.motor_id || report.source?.device_id || '',
    anchorTime: forecast.anchor_time || report.source?.anchor_time || '',
    currentTemperature: numberValue(
      forecast.current_temperature_c,
      forecast.current_temperature,
      report.source?.current_temperature_c,
    ),
    forecastId: report.forecast_id || '',
    truthStatus: report.truth_status || forecast.truth_status || '',
    historySteps: numberValue(
      report.source?.history_shape?.[0],
      report.configuration?.history_steps,
      forecast.history_shape?.[0],
    ),
    maxHorizon: numberValue(
      configuredHorizons.length ? Math.max(...configuredHorizons.map(Number)) : null,
      trajectoryHorizons.length ? Math.max(...trajectoryHorizons) : null,
    ),
    selectionMae: bestModel?.selectionMae ?? null,
    selectionScope: bestModel?.selectionScope || '',
    forecastScope: bestModel?.horizonScope || '',
    syntheticDetected: Boolean(report.dataset_provenance?.synthetic_or_demo_detected),
    researchClaimsAllowed: report.dataset_provenance?.research_claims_allowed,
    level: firstDefined(risk.risk_level, risk.level, risk.classification),
    maxTemperature: numberValue(risk.max_temperature_c, risk.max_temp_c, risk.maximum_temperature),
    thresholdCrossing: crossingValue,
    timeToThreshold: numberValue(risk.time_to_threshold_minutes, risk.lead_time_minutes),
    heatingRate: numberValue(risk.heating_rate_c_per_min, risk.heating_rate_c_per_minute, risk.heating_rate),
    maxAdjacentHeatingRate: numberValue(
      risk.max_adjacent_heating_rate_c_per_min,
      risk.max_adjacent_heating_rate_c_per_minute,
      risk.max_adjacent_heating_rate,
    ),
    precision: numberValue(risk.precision, crossingMetrics.precision),
    recall: numberValue(risk.recall, crossingMetrics.recall),
    f1: numberValue(risk.f1, risk.f1_score, crossingMetrics.f1, crossingMetrics.f1_score),
    rocAuc: numberValue(risk.roc_auc, crossingMetrics.roc_auc),
    prAuc: numberValue(risk.pr_auc, risk.average_precision, crossingMetrics.pr_auc, crossingMetrics.average_precision),
    metricSampleCount: numberValue(crossingMetrics.sample_count),
    excludedOngoingCount: numberValue(crossingMetrics.excluded_ongoing_origin_count),
    trajectory: Array.isArray(trajectory) ? trajectory : [],
  };
}

function StatusAlert({ tone = 'info', children, action }) {
  const Icon = tone === 'error' ? AlertTriangle : tone === 'success' ? CheckCircle2 : Info;
  return (
    <div className={`research-alert ${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <Icon size={18} aria-hidden="true" />
      <div>{children}</div>
      {action}
    </div>
  );
}

function ConfigSkeleton() {
  return (
    <section className="research-loading" aria-busy="true" aria-live="polite">
      <LoaderCircle className="spin" size={24} aria-hidden="true" />
      <div><strong>正在載入研究設定</strong><span>檢查設備資料量與可用模型…</span></div>
    </section>
  );
}

function MotorOption({ motor }) {
  const id = motorId(motor);
  const count = typeof motor === 'object'
    ? numberValue(motor.complete_reading_count, motor.reading_count, motor.readings, motor.sample_count)
    : null;
  const defaultInsufficient = typeof motor === 'object' && motor.pipeline_eligible === false;
  return <option value={id}>{motorLabel(motor)}{count === null ? '' : ` · ${formatInteger(count)} 筆`}{defaultInsufficient ? ' · 預設契約資料不足' : ''}</option>;
}

function ReadinessCard({ title, motor, minimumReadings }) {
  const count = typeof motor === 'object'
    ? numberValue(motor.complete_reading_count, motor.reading_count, motor.readings, motor.sample_count)
    : null;
  const eligible = count !== null && minimumReadings !== null
    ? count >= minimumReadings
    : null;
  const studyEligible = typeof motor === 'object' ? motor.study_eligible : null;
  return (
    <div className="readiness-card">
      <div className="readiness-heading">
        <span>{title}</span>
        {eligible !== null && (
          <span className={`qualification ${eligible ? 'qualified' : 'insufficient'}`}>
            {studyEligible ? '探索筆數門檻通過' : eligible ? '筆數預檢通過' : '資料不足'}
          </span>
        )}
      </div>
      <strong>{motor ? motorLabel(motor) : '尚未選擇'}</strong>
      <small>{count === null
        ? '資料量將由後端檢核'
        : `${formatInteger(count)} 筆；目前設定至少需 ${formatInteger(minimumReadings)} 筆`}</small>
    </div>
  );
}

function minimumExperimentReadings({
  historySteps,
  horizonSteps,
  walkForwardFolds,
  trainFraction = 0.6,
  validationFraction = 0.2,
}) {
  // Mirror the backend's chronological_split + development-only
  // walk_forward_splits sizing rules, including integer rounding.
  const firstCandidate = historySteps + 3 * horizonSteps + 2;
  for (let readings = firstCandidate; readings < firstCandidate + 10000; readings += 1) {
    const sequenceCount = readings - historySteps - horizonSteps + 1;
    const usable = sequenceCount - 2 * horizonSteps;
    if (usable < 3) continue;
    let trainCount = Math.max(1, Math.floor(usable * trainFraction));
    let validationCount = Math.max(1, Math.floor(usable * validationFraction));
    if (trainCount + validationCount >= usable) {
      validationCount = 1;
      trainCount = usable - 2;
    }
    const developmentCount = trainCount + horizonSteps + validationCount;
    const foldTestSize = Math.max(
      1,
      Math.floor((developmentCount - horizonSteps) / (walkForwardFolds + 2)),
    );
    const initialFoldTrainingCount = developmentCount
      - horizonSteps
      - walkForwardFolds * foldTestSize;
    if (initialFoldTrainingCount >= 1) return readings;
  }
  return null;
}

function ExperimentForm({
  config, form, submitting, forecasting, onField, onToggleHorizon, onSubmit, onForecast,
}) {
  const motors = config?.motors || config?.available_motors || config?.devices || config?.datasets || [];
  const horizons = horizonOptions(config);
  const models = modelsFromConfig(config);
  const forecastModels = forecastModelOptions(config);
  const historyOptions = [...new Set([
    ...(config?.allowed_history_minutes || [30, 60, 90, 120]),
    Number(form.historyMinutes),
  ].map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  const samplingMinutes = numberValue(config?.sampling_minutes, config?.defaults?.sampling_minutes, config?.configuration?.sampling_minutes, 5);
  const trainingMotor = motors.find((motor) => motorId(motor) === form.trainingMotorId);
  const evaluationMotor = motors.find((motor) => motorId(motor) === form.evaluationMotorId);
  const hasMotors = motors.length > 0;
  const historySteps = Math.ceil(Number(form.historyMinutes) / samplingMinutes);
  const maximumHorizon = form.horizonsMinutes.length
    ? Math.max(...form.horizonsMinutes)
    : samplingMinutes;
  const maximumHorizonSteps = Math.ceil(maximumHorizon / samplingMinutes);
  const walkForwardFolds = numberValue(config?.defaults?.walk_forward_folds, 3);
  const requiredExperimentTrainingReadings = minimumExperimentReadings({
    historySteps,
    horizonSteps: maximumHorizonSteps,
    walkForwardFolds,
    trainFraction: numberValue(config?.defaults?.train_fraction, 0.6),
    validationFraction: numberValue(config?.defaults?.validation_fraction, 0.2),
  });
  const requiredEvaluationReadings = historySteps + maximumHorizonSteps;
  const requiredForecastTrainingReadings = historySteps + maximumHorizonSteps;
  const requiredForecastInferenceReadings = historySteps;
  const readingCount = (motor) => numberValue(
    motor?.complete_reading_count,
    motor?.reading_count,
    motor?.readings,
    motor?.sample_count,
  );
  const trainingCount = readingCount(trainingMotor);
  const evaluationCount = readingCount(evaluationMotor);
  const sameDevice = form.trainingMotorId === form.evaluationMotorId;
  const experimentEligible = trainingCount !== null
    && trainingCount >= requiredExperimentTrainingReadings
    && (sameDevice || (
      evaluationCount !== null && evaluationCount >= requiredEvaluationReadings
    ));
  const forecastEligible = trainingCount !== null
    && trainingCount >= requiredForecastTrainingReadings
    && evaluationCount !== null
    && evaluationCount >= requiredForecastInferenceReadings;

  return (
    <aside className="experiment-panel" aria-labelledby="experiment-config-title">
      <div className="panel-title-row">
        <div className="panel-icon"><FlaskConical size={19} aria-hidden="true" /></div>
        <div><p>實驗設定</p><h2 id="experiment-config-title">建立公平比較</h2></div>
      </div>
      {!hasMotors && (
        <StatusAlert tone="error">目前沒有可用的馬達資料集，請先匯入感測資料。</StatusAlert>
      )}
      <form onSubmit={(event) => { event.preventDefault(); onSubmit(); }}>
        <fieldset disabled={submitting || forecasting || !hasMotors}>
          <legend className="sr-only">研究實驗參數</legend>
          <label className="form-field">
            <span>訓練資料馬達</span>
            <select
              value={form.trainingMotorId}
              onChange={(event) => onField('trainingMotorId', event.target.value)}
            >
              {motors.map((motor) => <MotorOption key={motorId(motor)} motor={motor} />)}
            </select>
            <small>僅使用此設備的過去資料擬合模型</small>
          </label>
          <label className="form-field">
            <span>評估資料馬達</span>
            <select
              value={form.evaluationMotorId}
              onChange={(event) => onField('evaluationMotorId', event.target.value)}
            >
              {motors.map((motor) => <MotorOption key={motorId(motor)} motor={motor} />)}
            </select>
            <small>選不同設備可檢驗跨設備泛化</small>
          </label>

          <div className="two-field-grid">
            <label className="form-field">
              <span>歷史視窗</span>
              <select
                value={form.historyMinutes}
                onChange={(event) => onField('historyMinutes', Number(event.target.value))}
              >
                {historyOptions.map((minutes) => (
                  <option key={minutes} value={minutes}>{minutes} 分鐘</option>
                ))}
              </select>
              <small>{Math.round(form.historyMinutes / samplingMinutes)} 個 time steps</small>
            </label>
            <label className="form-field">
              <span>警戒溫度</span>
              <span className="input-suffix">
                <input
                  type="number"
                  min="20"
                  max="120"
                  step="0.5"
                  value={form.thresholdC}
                  onChange={(event) => onField('thresholdC', event.target.value)}
                />
                <span>°C</span>
              </span>
              <small>用於過熱風險標記</small>
            </label>
          </div>

          <fieldset className="horizon-fieldset">
            <legend>預測 Horizons</legend>
            <div className="horizon-options">
              {horizons.map((horizon) => (
                <label key={horizon} className={form.horizonsMinutes.includes(horizon) ? 'selected' : ''}>
                  <input
                    type="checkbox"
                    checked={form.horizonsMinutes.includes(horizon)}
                    onChange={() => onToggleHorizon(horizon)}
                  />
                  +{horizon}
                </label>
              ))}
            </div>
            <small>所有模型使用相同的多步預測目標</small>
          </fieldset>

          <div className="model-catalogue" aria-label="本次比較模型">
            <span>比較模型</span>
            <div>{models.map((model) => <small key={model}>{model}</small>)}</div>
          </div>

          <label className="form-field forecast-model-field">
            <span>最新軌跡推論模型</span>
            <select
              value={form.forecastModel}
              onChange={(event) => onField('forecastModel', event.target.value)}
            >
              {forecastModels.map((model) => (
                <option key={model.id} value={model.id}>{model.label}</option>
              ))}
            </select>
            <small>只列出後端目前可執行的模型</small>
          </label>

          <div className="experiment-actions">
            <button
              className="run-forecast-button"
              type="button"
              onClick={(event) => {
                if (event.currentTarget.form?.reportValidity()) onForecast();
              }}
              disabled={!form.horizonsMinutes.length || submitting || forecasting || !hasMotors || !forecastEligible}
            >
              {forecasting ? <LoaderCircle className="spin" size={18} /> : <Thermometer size={18} />}
              {forecasting ? '正在預測…' : '預測最新軌跡'}
            </button>
            <button
              className="run-experiment-button"
              type="submit"
              disabled={!form.horizonsMinutes.length || submitting || forecasting || !hasMotors || !experimentEligible}
            >
              {submitting ? <LoaderCircle className="spin" size={18} /> : <Play size={18} fill="currentColor" />}
              {submitting ? '正在建立實驗…' : '啟動完整實驗'}
            </button>
          </div>
        </fieldset>
      </form>

      <div className="readiness-preview" aria-label="資料資格預覽">
        <ReadinessCard title="訓練集" motor={trainingMotor} minimumReadings={requiredExperimentTrainingReadings} />
        <ReadinessCard title="評估集" motor={evaluationMotor} minimumReadings={sameDevice ? requiredExperimentTrainingReadings : requiredEvaluationReadings} />
      </div>
      <p className="protocol-note"><ShieldCheck size={15} />依時間切分、禁止 shuffle，並保留預測 horizon gap 防止洩漏。</p>
      <p className="formal-data-note">依目前參數估算：完整實驗的訓練設備至少需 {formatInteger(requiredExperimentTrainingReadings)} 筆；最新軌跡需訓練來源 {formatInteger(requiredForecastTrainingReadings)} 筆、推論設備 {formatInteger(requiredForecastInferenceReadings)} 筆。實際仍須通過 cadence、重複、finite 與完整序列檢核。</p>
      {config?.constraints?.recommended_formal_readings && (
        <p className="formal-data-note">正式研究建議每設備累積至少 {formatInteger(config.constraints.recommended_formal_readings)} 筆（{formatInteger(config.constraints.recommended_formal_duration_days)} 天）；{formatInteger(config.constraints.minimum_exploratory_readings)} 筆僅達探索門檻。</p>
      )}
      {config?.warnings?.length > 0 && (
        <details className="config-warnings">
          <summary><AlertTriangle size={13} />研究資料警語</summary>
          <ul>{config.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </details>
      )}
    </aside>
  );
}

function ExperimentProgress({ experiment, submitting }) {
  if (!experiment && !submitting) return null;
  const completedPayload = Boolean(reportOf(experiment)?.schema_version && reportOf(experiment)?.models);
  const rawStatus = String(
    experiment?.status
    || experiment?.experiment?.status
    || (completedPayload ? 'completed' : submitting ? 'creating' : ''),
  ).toLowerCase();
  const statusLabel = {
    creating: '正在建立實驗', queued: '等待執行', pending: '等待執行',
    running: '模型訓練與評估中', completed: '實驗已完成', complete: '實驗已完成',
    succeeded: '實驗已完成', failed: '實驗失敗', error: '實驗失敗',
  }[rawStatus] || rawStatus || '處理中';
  const progressRaw = numberValue(experiment?.progress, experiment?.progress_percent);
  const progress = progressRaw === null ? null : Math.min(100, progressRaw <= 1 ? progressRaw * 100 : progressRaw);
  const failed = ['failed', 'error'].includes(rawStatus);
  const completed = ['completed', 'complete', 'succeeded'].includes(rawStatus);
  return (
    <section className={`experiment-progress ${failed ? 'failed' : completed ? 'completed' : ''}`} aria-live="polite" aria-busy={!failed && !completed}>
      <div className="progress-icon">
        {failed ? <AlertTriangle size={20} /> : completed ? <CheckCircle2 size={20} /> : <LoaderCircle className="spin" size={20} />}
      </div>
      <div className="progress-copy">
        <div><strong>{statusLabel}</strong><span>{experiment?.id || experiment?.experiment_id || ''}</span></div>
        <p>{experiment?.message || experiment?.stage || (completed ? '結果已使用保留測試集計算。' : '依序執行資料檢核、模型訓練、測試與消融實驗。')}</p>
        {!failed && !completed && <div className="progress-track"><span style={{ width: `${progress ?? 22}%` }} /></div>}
      </div>
    </section>
  );
}

function MetricCard({ label, value, helper, icon: Icon }) {
  return (
    <div className="metric-card">
      <div className="metric-icon"><Icon size={17} aria-hidden="true" /></div>
      <div><span>{label}</span><strong>{value}</strong>{helper && <small>{helper}</small>}</div>
    </div>
  );
}

function DatasetPanel({ report, form }) {
  const dataset = report.dataset || report.data || report.dataset_summary || {};
  const configuration = report.configuration || report.config || {};
  const training = dataset.training || dataset.train || {};
  const evaluation = dataset.evaluation || dataset.test_device || {};
  const quality = training.quality || dataset.quality || report.data_eligibility || {};
  const split = dataset.split || report.split || report.data_split || {};
  const provenance = report.dataset_provenance || report.provenance || {};
  const eligible = firstDefined(
    quality.eligible,
    quality.qualified,
    dataset.eligible,
    numberValue(training.sequence_count, quality.complete_sequence_count) > 0 ? true : null,
  );
  const sampling = numberValue(configuration.sampling_minutes, dataset.sampling_minutes, 5);
  const historySteps = numberValue(configuration.history_steps, dataset.history_steps, form.historyMinutes / sampling);
  const sequenceCount = numberValue(
    dataset.sequence_count,
    training.sequence_count,
    quality.sequence_count,
    split.total_sequences,
  );
  const trainingIds = Array.isArray(training.device_ids)
    ? training.device_ids.join(', ')
    : training.motor_id || provenance.training_motor_id || form.trainingMotorId;
  const evaluationIds = Array.isArray(evaluation?.device_ids)
    ? evaluation.device_ids.join(', ')
    : evaluation?.motor_id || evaluation?.id || provenance.evaluation_motor_id || trainingIds;
  const partitionCount = (name) => numberValue(
    split[name]?.sample_count,
    split[`${name}_sequences`],
    split[`${name}_samples`],
  );

  return (
    <section className="result-card" aria-labelledby="dataset-title">
      <div className="result-heading">
        <div><span className="section-kicker">Dataset protocol</span><h2 id="dataset-title">資料資格與時間切分</h2></div>
        <div className="result-badges">
          {eligible !== undefined && eligible !== null && <span className={`qualification ${eligible ? 'qualified' : 'insufficient'}`}>{eligible ? 'Pipeline 合格' : '資料不足'}</span>}
          {provenance.research_claims_allowed === false && <span className="qualification insufficient">僅流程驗證</span>}
        </div>
      </div>
      {provenance.evidence_note && (
        <StatusAlert tone={provenance.research_claims_allowed === false ? 'error' : 'info'}>{provenance.evidence_note}</StatusAlert>
      )}
      <div className="metric-grid four">
        <MetricCard icon={Database} label="有效紀錄" value={`${formatInteger(firstDefined(quality.valid_readings, quality.accepted_rows, quality.reading_count, training.reading_count, provenance.training_reading_count))} 筆`} helper={`訓練：${trainingIds}`} />
        <MetricCard icon={Layers3} label="有效序列" value={`${formatInteger(sequenceCount)} 組`} helper={`${formatInteger(historySteps)} × 5 features`} />
        <MetricCard icon={Clock3} label="取樣間隔" value={`${formatInteger(sampling)} 分鐘`} helper="固定時間網格" />
        <MetricCard icon={Activity} label="跨設備評估" value={evaluationIds || '—'} helper={trainingIds === evaluationIds ? '同設備 holdout' : '獨立設備資料'} />
      </div>
      <div className="quality-audit" aria-label="資料品質檢核">
        <span>缺特徵 <strong>{formatInteger(firstDefined(quality.incomplete_feature_rows, quality.missing_feature_rows, 0))}</strong></span>
        <span>重複時間 <strong>{formatInteger(firstDefined(quality.duplicate_timestamp_count, quality.duplicate_rows, 0))}</strong></span>
        <span>不規則間隔 <strong>{formatInteger(firstDefined(quality.irregular_interval_count, quality.irregular_rows, 0))}</strong></span>
        <span>完整序列 <strong>{formatInteger(firstDefined(quality.complete_sequence_count, sequenceCount))}</strong></span>
      </div>
      <div className="split-panel">
        <div className="split-copy">
          <strong>{split.strategy || split.method || 'Walk-forward / chronological holdout'}</strong>
          <span>Train → Gap → Validation → Gap → Test，任何未來樣本都不會進入訓練資料。</span>
        </div>
        <div className="split-bar" aria-label="時間切分示意">
          <span className="train" style={{ flexGrow: numberValue(split.train_fraction, 0.6) }}><b>Train</b><small>{formatInteger(partitionCount('train'))}</small></span>
          <span className="gap"><b>Gap</b><small>{formatInteger(firstDefined(split.gap_steps, configuration.gap_steps, 6))}</small></span>
          <span className="validation" style={{ flexGrow: numberValue(split.validation_fraction, 0.2) }}><b>Validation</b><small>{formatInteger(partitionCount('validation'))}</small></span>
          <span className="gap"><b>Gap</b><small>{formatInteger(firstDefined(split.gap_steps, configuration.gap_steps, 6))}</small></span>
          <span className="test" style={{ flexGrow: numberValue(split.test_fraction, 0.2) }}><b>Test</b><small>{formatInteger(partitionCount('test'))}</small></span>
        </div>
      </div>
      {quality.reasons?.length > 0 && (
        <StatusAlert tone={eligible === false ? 'error' : 'info'}>{quality.reasons.join('；')}</StatusAlert>
      )}
    </section>
  );
}

function ModelComparison({ rows }) {
  const available = rows.filter((row) => row.status === 'available' || row.status === 'completed' || row.mae !== null);
  const selectionValues = available.map((row) => row.selectionMae).filter((value) => value !== null);
  const bestSelectionMae = selectionValues.length ? Math.min(...selectionValues) : null;
  const lowVariance = available.some((row) => row.r2LowVarianceWarning);
  const targetStdDev = available.map((row) => row.targetStdDev).find((value) => value !== null);
  if (!rows.length) return <ResultEmpty title="尚無模型指標" copy="研究服務完成訓練後會在此列出準確度與部署成本。" />;
  return (
    <section className="result-card" aria-labelledby="model-comparison-title">
      <div className="result-heading">
        <div><span className="section-kicker">Holdout test set</span><h2 id="model-comparison-title">模型效能與 Edge 成本</h2></div>
        <span className="metric-direction">MAE / RMSE / 延遲 ↓　R² ↑</span>
      </div>
      {lowVariance && (
        <StatusAlert tone="info">測試真值標準差僅 {formatNumber(targetStdDev, 3)} °C；R² 的分母很小，因此小誤差也可能得到很大的負值。請優先同看 MAE 與相對 Direct Ridge 技能分數，原始 R² 仍完整保留。</StatusAlert>
      )}
      <div className="table-scroll">
        <table className="research-table">
          <caption className="sr-only">各預測模型於保留測試集的準確度與推論成本比較</caption>
          <thead><tr><th scope="col">模型</th><th scope="col">MAE °C</th><th scope="col">RMSE °C</th><th scope="col">Max error</th><th scope="col">R²</th><th scope="col">相對 Direct Ridge</th><th scope="col">跨設備 MAE</th><th scope="col">推論延遲</th><th scope="col">訓練時間</th><th scope="col">模型大小</th></tr></thead>
          <tbody>
            {rows.map((row) => {
              const unavailable = ['unavailable', 'failed', 'error'].includes(row.status) && row.mae === null;
              const selectedDuringDevelopment = row.selectionMae === bestSelectionMae && bestSelectionMae !== null;
              return (
                <tr key={row.name} className={unavailable ? 'unavailable-row' : ''}>
                  <th scope="row"><span className="model-name">{row.name}</span>{selectedDuringDevelopment && <small className="best-value" title={`${row.selectionScope} MAE ${formatNumber(row.selectionMae, 3)}`}>開發期候選</small>}{unavailable && <small title={row.reason}>{row.status === 'unavailable' ? '環境未安裝' : '執行失敗'}</small>}</th>
                  {unavailable ? <td colSpan="9" className="unavailable-reason">{row.reason || '此模型目前不可用，未產生比較數值。'}</td> : <>
                    <td>{formatNumber(row.mae)}</td>
                    <td>{formatNumber(row.rmse)}</td><td>{formatNumber(row.maxError)}</td><td>{formatNumber(row.r2, 3)}</td><td>{row.skillVsDirectRidge === null ? '—' : `${row.skillVsDirectRidge >= 0 ? '+' : ''}${formatNumber(row.skillVsDirectRidge * 100, 1)}%`}</td><td>{formatNumber(row.crossDeviceMae)}</td><td>{formatDurationMs(row.latencyMs)}</td><td>{formatDurationMs(row.trainingMs)}</td><td>{formatBytes(row.sizeBytes)}</td>
                  </>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="table-note">「開發期候選」只依 walk-forward／validation MAE；相對 Direct Ridge 正值代表改善、負值代表退步。本表 test 與跨設備數值只用於凍結後報告，不反向選模。Unavailable 模型保留原因，不補虛構數值。</p>
    </section>
  );
}

function HorizonMetrics({ rows }) {
  const maxMae = Math.max(0, ...rows.map((row) => row.mae || 0));
  if (!rows.length) return <ResultEmpty title="尚無 Horizon 指標" copy="需要至少一個可評估模型才能比較不同預警提前量。" />;
  return (
    <section className="result-card" aria-labelledby="horizon-title">
      <div className="result-heading"><div><span className="section-kicker">Forecast reliability</span><h2 id="horizon-title">各預測 Horizon 誤差</h2></div></div>
      <div className="table-scroll">
        <table className="research-table horizon-table">
          <caption className="sr-only">每個模型於各時間預測點的誤差</caption>
          <thead><tr><th scope="col">Horizon</th><th scope="col">模型</th><th scope="col">評估範圍</th><th scope="col">MAE °C</th><th scope="col">相對誤差</th><th scope="col">RMSE °C</th><th scope="col">R²</th></tr></thead>
          <tbody>{rows.map((row, index) => (
            <tr key={`${row.model}-${row.horizon}-${index}`}><th scope="row">+{formatInteger(row.horizon)} min</th><td>{row.model}</td><td>{row.scope || '保留測試集'}</td><td>{formatNumber(row.mae)}</td><td><span className="error-bar"><i style={{ width: `${maxMae ? Math.max(3, (row.mae / maxMae) * 100) : 0}%` }} /></span></td><td>{formatNumber(row.rmse)}</td><td>{formatNumber(row.r2, 3)}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  );
}

function FeatureAblation({ rows }) {
  const available = rows.filter((row) => row.mae !== null);
  const maxMae = Math.max(0, ...available.map((row) => row.mae));
  const sourceGroups = rows.reduce((groups, row) => {
    const label = row.modelLabel || row.model || '未標示模型';
    groups[label] = [...(groups[label] || []), row.id];
    return groups;
  }, {});
  const sourceSummary = Object.entries(sourceGroups)
    .map(([model, ids]) => `${model}（${ids.join('、')}）`)
    .join('；');
  if (!rows.length) return <ResultEmpty title="尚無 Feature Ablation" copy="完整實驗會比較溫度、濕度、震動與歷史趨勢的增益。" />;
  return (
    <section className="result-card" aria-labelledby="ablation-title">
      <div className="result-heading"><div><span className="section-kicker">Feature contribution</span><h2 id="ablation-title">多感測器特徵消融</h2></div></div>
      <div className="data-source-panel ablation-source" aria-label="特徵消融模型與數據來源">
        <div><span>使用模型</span><strong>{sourceSummary}</strong></div>
        <div><span>數據與指標</span><strong>歷史鎖定測試集（locked test）的 MAE</strong></div>
      </div>
      <div className="ablation-list">
        {rows.map((row) => (
          <div className="ablation-row" key={row.id}>
            <div className="ablation-label"><strong>{row.label}</strong><span>模型：{row.modelLabel || row.model || '未標示'} · 特徵：{row.features || '未標示'}</span></div>
            {row.mae === null ? <span className="ablation-unavailable" title={row.reason}>{row.status === 'unavailable' ? '不可用' : '無數值'}</span> : <>
              <span className="ablation-bar"><i style={{ width: `${maxMae ? Math.max(5, (row.mae / maxMae) * 100) : 0}%` }} /></span>
              <strong className="ablation-value">{formatNumber(row.mae)}<small> MAE</small></strong>
            </>}
          </div>
        ))}
      </div>
    </section>
  );
}

function riskLevelClass(level) {
  const value = String(level || '').toLowerCase();
  if (value.includes('high') || value.includes('高')) return 'high';
  if (value.includes('medium') || value.includes('中')) return 'medium';
  return 'low';
}

function RiskPanel({ risk, threshold, live = false }) {
  const hasForecast = risk.level || risk.maxTemperature !== null || risk.thresholdCrossing !== undefined || risk.trajectory.length;
  const hasMetrics = risk.metricSampleCount !== 0
    && [risk.precision, risk.recall, risk.f1, risk.rocAuc, risk.prAuc].some((value) => value !== null);
  if (!hasForecast && !hasMetrics) return <ResultEmpty title="尚無風險評估" copy="有足夠的正負過熱事件後，系統會顯示預警效能與越界提前量。" />;
  const level = risk.level || (risk.thresholdCrossing ? 'High' : 'Low');
  return (
    <section className={`result-card risk-card${live ? ' live-forecast-card' : ''}`} aria-labelledby={live ? 'live-risk-title' : 'risk-title'}>
      <div className="result-heading"><div><span className="section-kicker">{live ? 'Current pending-truth forecast' : 'Offline risk evaluation'}</span><h2 id={live ? 'live-risk-title' : 'risk-title'}>{live ? '最新溫度軌跡與過熱風險' : '保留樣本風險評估'}</h2></div><span className={`risk-badge ${riskLevelClass(level)}`}>{level}</span></div>
      {live && risk.researchClaimsAllowed === false && <StatusAlert tone="error"><strong>{risk.syntheticDetected ? '合成／DEMO 資料' : '來源尚未驗證'}</strong><p>{risk.syntheticDetected ? '這筆軌跡只能驗證系統流程，不可作為真實效能或設備安全結論。' : '此資料尚未連結經簽核的 frozen manifest，只能作探索與流程驗證。'}</p></StatusAlert>}
      {!live && risk.sourceModel && <div className="data-source-panel risk-source" aria-label="風險評估模型與數據來源">
        <div><span>本區數據模型</span><strong>{risk.sourceModel}</strong></div>
        <div><span>模型選擇依據</span><strong>{risk.selectionScope || '開發期指標'}{risk.selectionMae === null ? '' : ` MAE ${formatNumber(risk.selectionMae, 3)}`}</strong></div>
        <div><span>軌跡與分類指標範圍</span><strong>{risk.forecastScope || '保留測試集'} · 同一模型</strong></div>
      </div>}
      {(risk.sourceModel || risk.deviceId) && <p className="forecast-source">{live
        ? <>以 <strong>{risk.sourceModel}</strong> 對 <strong>{risk.deviceId || '推論設備'}</strong> 最新完整 <strong>{formatInteger(risk.historySteps)}</strong> 筆感測序列預測；未來真值目前為 <strong>{risk.truthStatus || 'pending'}</strong>，未參與推論。{risk.currentTemperature !== null ? <> 起點溫度：<strong>{formatNumber(risk.currentTemperature, 1)} °C</strong>。</> : null}{risk.anchorTime ? <> 預測起點：<strong>{risk.anchorTime}</strong>。</> : null}{risk.forecastId ? <> Forecast ID：<strong>{risk.forecastId}</strong>。</> : null}</>
        : <>依 <strong>{risk.selectionScope || '開發期指標'}</strong>{risk.selectionMae !== null ? <> MAE <strong>{formatNumber(risk.selectionMae, 3)}</strong></> : null}，選取顯示模型 <strong>{risk.sourceModel}</strong>；此卡是 <strong>{risk.forecastScope || '保留測試集'}</strong> 最後一個已具真值樣本的離線例子，不是即時讀值或模型 promotion 證據。</>}</p>}
      {hasForecast && <div className="metric-grid four">
        <MetricCard icon={Thermometer} label="預測最高溫" value={risk.maxTemperature === null ? '—' : `${formatNumber(risk.maxTemperature, 1)} °C`} helper={`門檻 ${formatNumber(threshold, 1)} °C`} />
        <MetricCard icon={AlertTriangle} label="目前／預計越界" value={risk.thresholdCrossing === undefined ? '—' : risk.thresholdCrossing ? '是' : '否'} helper={`${formatInteger(risk.maxHorizon)} 分鐘 horizon；0 代表起點已越界`} />
        <MetricCard icon={Clock3} label="越界提前量" value={risk.timeToThreshold === null ? '—' : `${formatInteger(risk.timeToThreshold)} 分鐘`} helper="Time-to-threshold" />
        <MetricCard icon={Gauge} label="溫升速率" value={risk.heatingRate === null ? '—' : `${formatNumber(risk.heatingRate, 2)} °C/min`} helper={risk.maxAdjacentHeatingRate === null ? '整段線性斜率' : `整段線性；最大相鄰 ${formatNumber(risk.maxAdjacentHeatingRate, 2)} °C/min`} />
      </div>}
      {hasMetrics && <div className="risk-metrics" aria-label="風險分類指標">
        <span><small>Precision</small><strong>{formatNumber(risk.precision, 3)}</strong></span><span><small>Recall</small><strong>{formatNumber(risk.recall, 3)}</strong></span><span><small>F1</small><strong>{formatNumber(risk.f1, 3)}</strong></span><span><small>ROC-AUC</small><strong>{formatNumber(risk.rocAuc, 3)}</strong></span><span><small>PR-AUC</small><strong>{formatNumber(risk.prAuc, 3)}</strong></span>
      </div>}
      {!live && risk.metricSampleCount === 0 && <p className="table-note">此分割沒有 origin 溫度低於門檻的 early-warning window，因此不顯示 Precision／Recall／AUC；已越界起點另列 ongoing cohort。</p>}
      {risk.trajectory.length > 0 && <div className="trajectory-chips" aria-label="未來溫度軌跡">{risk.trajectory.map((point, index) => {
        const horizon = firstDefined(point.horizon_minutes, point.horizon, point.minutes, (index + 1) * 5);
        const temperature = firstDefined(point.temperature_c, point.prediction_c, point.predicted_temperature_c, point.predicted_temperature, point.value);
        return <span key={`${horizon}-${index}`}><small>+{horizon} min</small><strong>{formatNumber(temperature, 1)}°</strong></span>;
      })}</div>}
    </section>
  );
}

function MethodologyPanel({ report, form }) {
  const method = report.methodology || {};
  const configuration = report.configuration || {};
  const historyMinutes = numberValue(
    configuration.history_minutes,
    method.history_minutes,
    form.historyMinutes,
  );
  const samplingMinutes = numberValue(configuration.sampling_minutes, 5);
  const historySteps = numberValue(
    configuration.history_steps,
    method.history_steps,
    historyMinutes / samplingMinutes,
  );
  const horizons = Array.isArray(configuration.horizons_minutes)
    ? configuration.horizons_minutes
    : form.horizonsMinutes;
  const gapSteps = numberValue(
    configuration.gap_steps,
    method.gap_steps,
    Math.max(...horizons) / samplingMinutes,
  );
  return (
    <section className="methodology-card" aria-labelledby="method-title">
      <div className="result-heading"><div><span className="section-kicker">Reproducible protocol</span><h2 id="method-title">研究方法</h2></div><span className="schema-version">{report.schema_version ? `Schema ${report.schema_version}` : 'Multi-horizon v1'}</span></div>
      <div className="method-flow">
        <div><span>01</span><strong>歷史序列</strong><p>最近 {formatInteger(historyMinutes)} 分鐘，{formatInteger(historySteps)} steps × 溫度、濕度與 XYZ。</p></div>
        <div><span>02</span><strong>多步輸出</strong><p>{horizons.map((value) => `+${value}`).join('、')} 分鐘的溫度軌跡。</p></div>
        <div><span>03</span><strong>公平驗證</strong><p>Chronological 60/20/20 split，兩個分割邊界各保留 {formatInteger(gapSteps)} steps gap。</p></div>
        <div><span>04</span><strong>部署決策</strong><p>同時比較誤差、延遲、模型大小、跨設備泛化與過熱風險。</p></div>
      </div>
      <details>
        <summary>查看實驗控制與解讀原則</summary>
        <div className="method-details">
          <p>{typeof method === 'string' ? method : method.description || '模型共用相同合格樣本、預測目標與時間切分；Direct Ridge 僅取當下單點作資訊受限 baseline。'}</p>
          <ul><li>Direct Ridge 是共同 baseline，較複雜模型需證明準確度增益。</li><li>消融實驗回答濕度、震動與歷史趨勢是否真正帶來貢獻。</li><li>若模型套件或資料條件不足，結果明確標記 unavailable，而非補入模擬分數。</li></ul>
        </div>
      </details>
    </section>
  );
}

function ResultEmpty({ title, copy }) {
  return <section className="result-empty"><BarChart3 size={22} aria-hidden="true" /><div><strong>{title}</strong><span>{copy}</span></div></section>;
}

function InitialState({ form }) {
  return (
    <section className="research-initial" aria-labelledby="research-empty-title">
      <div className="initial-visual"><FlaskConical size={29} aria-hidden="true" /></div>
      <span className="section-kicker">Ready to compare</span>
      <h2 id="research-empty-title">建立第一組可重現實驗</h2>
      <p>系統會使用相同時間切分一次比較 baseline、傳統機器學習與序列模型，並保留不可用模型的原因。</p>
      <div className="initial-protocol">
        <span><strong>{form.historyMinutes} min</strong><small>歷史視窗</small></span>
        <i>→</i>
        <span><strong>{form.horizonsMinutes.length} horizons</strong><small>多步溫度</small></span>
        <i>→</i>
        <span><strong>{formatNumber(form.thresholdC, 1)} °C</strong><small>過熱風險</small></span>
      </div>
    </section>
  );
}

/** Render the research page while retaining its state when the chat view is active. */
export default function ResearchWorkspace({ active }) {
  const {
    config, form, experiment, forecast, configLoading, submitting, forecasting, configError,
    experimentError, forecastError, loadConfig, runExperiment, runForecast,
    toggleHorizon, updateField,
  } = useResearch();
  const report = reportOf(experiment);
  const models = useMemo(() => modelRows(report), [report]);
  const horizons = useMemo(() => horizonRows(models, report), [models, report]);
  const ablations = useMemo(() => ablationRows(report), [report]);
  const risk = useMemo(() => riskSummary(report, models), [report, models]);
  const liveRisk = useMemo(() => riskSummary(forecast || {}, []), [forecast]);
  const experimentThreshold = firstDefined(report.configuration?.threshold_c, form.thresholdC);
  const forecastThreshold = firstDefined(forecast?.configuration?.threshold_c, form.thresholdC);
  const hasResults = Boolean(experiment) && (
    models.length > 0 || ablations.length > 0 || Object.keys(report.dataset || {}).length > 0
  );

  return (
    <section className="research-workspace" hidden={!active} aria-labelledby="research-page-title">
      <div className="research-page">
        <header className="research-hero">
          <div>
            <span className="research-eyebrow"><Activity size={14} /> Edge AI forecasting study</span>
            <h1 id="research-page-title">溫度軌跡與過熱風險研究工作台</h1>
            <p>以相同 sample IDs、targets 與時間切分，比較單點 baseline 與歷史序列模型在未來 5–30 分鐘的誤差、運算成本與跨設備表現。</p>
          </div>
          <div className="research-scope"><span>輸入</span><strong>12 × 5</strong><i /><span>輸出</span><strong>6 horizons</strong></div>
        </header>

        {configLoading ? <ConfigSkeleton /> : configError ? (
          <StatusAlert tone="error" action={<button type="button" onClick={loadConfig}><RefreshCw size={15} />重新連線</button>}>
            <strong>無法載入研究設定</strong><p>{configError}</p>
          </StatusAlert>
        ) : (
          <div className="research-layout">
            <ExperimentForm config={config} form={form} submitting={submitting} forecasting={forecasting} onField={updateField} onToggleHorizon={toggleHorizon} onSubmit={runExperiment} onForecast={runForecast} />
            <div className="research-results">
              {forecastError && <StatusAlert tone="error">{forecastError}</StatusAlert>}
              {forecasting && <StatusAlert><strong>正在建立最新多時域預測</strong><p>只用預測起點以前可得的標籤擬合模型，未來六個真值維持 pending。</p></StatusAlert>}
              {forecast && <RiskPanel risk={liveRisk} threshold={forecastThreshold} live />}
              {experimentError && <StatusAlert tone="error">{experimentError}</StatusAlert>}
              <ExperimentProgress experiment={experiment} submitting={submitting} />
              {!experiment && !submitting ? <InitialState form={form} /> : hasResults && <>
                <DatasetPanel report={report} form={form} />
                <ModelComparison rows={models} />
                <HorizonMetrics rows={horizons} />
                <FeatureAblation rows={ablations} />
                <RiskPanel risk={risk} threshold={experimentThreshold} />
              </>}
              <MethodologyPanel report={report} form={form} />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
