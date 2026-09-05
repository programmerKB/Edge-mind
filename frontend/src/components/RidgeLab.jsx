/** @file Focused Direct Ridge versus Ridge + History experiment screen. */

import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Database,
  FlaskConical,
  Play,
  TrendingUp,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import ReportGallery from './ReportGallery.jsx';
import {
  loadRidgeLabConfig,
  runRidgeExperiment,
  runRidgeForecast,
} from '../services/ridgeLabApi.js';

function formatMetric(value, digits = 4) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
}

function formatTime(value) {
  if (!value) return '—';
  return new Intl.DateTimeFormat('zh-TW', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

function ModelCard({ model }) {
  const historical = model.model_name === 'ridge_history';
  return (
    <article className={`ridge-model-card ${historical ? 'history' : ''}`}>
      <div className="ridge-model-icon">
        {historical ? <TrendingUp size={21} /> : <Activity size={21} />}
      </div>
      <div>
        <div className="ridge-model-heading">
          <h3>{model.label}</h3>
          <span>{model.feature_count} 特徵</span>
        </div>
        <p>{model.description}</p>
      </div>
    </article>
  );
}

function DeviceOption({ device }) {
  return (
    <option value={device.motor_id}>
      {device.motor_id} · 完整 {device.complete_row_count.toLocaleString()} 筆
    </option>
  );
}

function ExperimentResult({ result }) {
  const improvement = result.comparison.test_mae_improvement_percent;
  const historyWon = result.comparison.winner === 'ridge_history';
  return (
    <section className="ridge-result-card" aria-live="polite">
      <div className="ridge-result-title">
        <div>
          <span className="ridge-kicker"><CheckCircle2 size={15} />實驗完成</span>
          <h2>鎖定測試集結果</h2>
        </div>
        <div className={`ridge-verdict ${historyWon ? 'positive' : ''}`}>
          {historyWon ? '歷史特徵較佳' : 'Direct Ridge 較佳'}
          <strong>
            {Number.isFinite(improvement)
              ? `${Math.abs(improvement).toFixed(1)}% MAE ${improvement >= 0 ? '改善' : '增加'}`
              : 'MAE 比較完成'}
          </strong>
        </div>
      </div>

      <div className="ridge-table-wrap">
        <table className="ridge-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>特徵</th>
              <th>α</th>
              <th>驗證 MAE</th>
              <th>測試 MAE</th>
              <th>測試 RMSE</th>
              <th>測試 R²</th>
              {result.evaluation_motor_id && <th>外部 MAE</th>}
            </tr>
          </thead>
          <tbody>
            {result.models.map((model) => (
              <tr key={model.model_name} data-winner={result.comparison.winner === model.model_name}>
                <td><strong>{model.label}</strong></td>
                <td>{model.feature_count}</td>
                <td>{model.selected_alpha}</td>
                <td>{formatMetric(model.validation_metrics.mae)}</td>
                <td>{formatMetric(model.test_metrics.mae)}</td>
                <td>{formatMetric(model.test_metrics.rmse)}</td>
                <td>{formatMetric(model.test_metrics.r2_score)}</td>
                {result.evaluation_motor_id && (
                  <td>{formatMetric(model.external_metrics?.mae)}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="ridge-result-meta">
        <span>共同樣本 {result.dataset.usable_sample_count.toLocaleString()} 筆</span>
        <span>測試 {result.dataset.test_sample_count.toLocaleString()} 筆</span>
        <span>耗時 {formatMetric(result.duration_ms, 0)} ms</span>
        <span>實驗 ID {result.experiment_id.slice(0, 8)}</span>
      </div>
      <div className="ridge-artifact-path">
        結果已寫入 <code>backend/outputs/{result.artifacts.run_directory}</code>
      </div>
    </section>
  );
}

export default function RidgeLab() {
  const [config, setConfig] = useState(null);
  const [trainingMotor, setTrainingMotor] = useState('');
  const [evaluationMotor, setEvaluationMotor] = useState('');
  const [forecastMotor, setForecastMotor] = useState('');
  const [forecastModel, setForecastModel] = useState('ridge_history');
  const [experiment, setExperiment] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [forecasting, setForecasting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    loadRidgeLabConfig()
      .then((payload) => {
        if (!active) return;
        setConfig(payload);
        const eligible = payload.devices.filter(
          (device) => device.complete_row_count >= 60,
        );
        const first = eligible[0]?.motor_id || '';
        const second = eligible[1]?.motor_id || '';
        setTrainingMotor(first);
        setEvaluationMotor(second);
        setForecastMotor(second || first);
      })
      .catch((requestError) => active && setError(requestError.message))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, []);

  const eligibleDevices = useMemo(
    () => config?.devices.filter((device) => device.complete_row_count >= 60) || [],
    [config],
  );

  async function handleExperiment() {
    setRunning(true);
    setError('');
    try {
      setExperiment(await runRidgeExperiment(trainingMotor, evaluationMotor));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setRunning(false);
    }
  }

  async function handleForecast() {
    setForecasting(true);
    setForecast(null);
    setError('');
    try {
      setForecast(await runRidgeForecast({
        motorId: forecastMotor,
        trainingMotorId: trainingMotor,
        modelName: forecastModel,
      }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setForecasting(false);
    }
  }

  if (loading) {
    return <div className="ridge-loading">正在讀取訓練資料…</div>;
  }

  return (
    <div className="ridge-lab">
      <header className="ridge-hero">
        <span className="ridge-kicker"><FlaskConical size={15} />Ridge Lab</span>
        <h1>讓模型多看一小時，真的會更準嗎？</h1>
        <p>
          用同一批樣本公平比較當下 5 特徵與 60 分鐘歷史摘要；驗證集只選 α，最終成績只看未碰過的測試集。
        </p>
      </header>

      {config && (
        <>
          <div className="ridge-model-grid">
            {config.models.map((model) => <ModelCard key={model.model_name} model={model} />)}
          </div>

          <section className="ridge-workflow-card">
            <div className="ridge-section-heading">
              <div>
                <span>01 · 比較實驗</span>
                <h2>選擇資料來源</h2>
              </div>
              <div className="ridge-method-strip">
                <span><Clock3 size={15} />歷史 60 分鐘</span>
                <ArrowRight size={14} />
                <span>預測 +30 分鐘</span>
                <span>60 / 20 / 20</span>
              </div>
            </div>
            <div className="ridge-controls">
              <label>
                <span>訓練設備</span>
                <select value={trainingMotor} onChange={(event) => setTrainingMotor(event.target.value)}>
                  {eligibleDevices.map((device) => <DeviceOption key={device.motor_id} device={device} />)}
                </select>
              </label>
              <label>
                <span>外部評估（可選）</span>
                <select value={evaluationMotor} onChange={(event) => setEvaluationMotor(event.target.value)}>
                  <option value="">不做外部評估</option>
                  {eligibleDevices.filter((device) => device.motor_id !== trainingMotor).map(
                    (device) => <DeviceOption key={device.motor_id} device={device} />,
                  )}
                </select>
              </label>
              <button className="ridge-primary-button" onClick={handleExperiment} disabled={!trainingMotor || running}>
                <Play size={16} fill="currentColor" />
                {running ? '正在訓練與測試…' : '執行公平比較'}
              </button>
            </div>
            <div className="ridge-device-list">
              <Database size={15} />目前可訓練：
              {eligibleDevices.map((device) => (
                <span key={device.motor_id}>{device.motor_id} {device.complete_row_count.toLocaleString()} 筆</span>
              ))}
            </div>
          </section>

          {experiment && <ExperimentResult result={experiment} />}

          <section className="ridge-workflow-card ridge-forecast-card">
            <div className="ridge-section-heading">
              <div>
                <span>02 · 最新預測</span>
                <h2>套用選定模型</h2>
              </div>
            </div>
            <div className="ridge-controls ridge-forecast-controls">
              <label>
                <span>預測設備</span>
                <select value={forecastMotor} onChange={(event) => setForecastMotor(event.target.value)}>
                  {eligibleDevices.map((device) => <DeviceOption key={device.motor_id} device={device} />)}
                </select>
              </label>
              <label>
                <span>模型</span>
                <select value={forecastModel} onChange={(event) => setForecastModel(event.target.value)}>
                  {config.models.map((model) => (
                    <option key={model.model_name} value={model.model_name}>{model.label}</option>
                  ))}
                </select>
              </label>
              <button className="ridge-secondary-button" onClick={handleForecast} disabled={!forecastMotor || forecasting}>
                <TrendingUp size={17} />
                {forecasting ? '正在預測並產生報表…' : '預測 30 分鐘後'}
              </button>
            </div>
            {forecast && (
              <div className="ridge-forecast-result">
                <div>
                  <span>{forecast.motor_id} · {forecast.model_label}</span>
                  <strong>{forecast.predicted_temperature.toFixed(2)} °C</strong>
                </div>
                <div>
                  <span>目前溫度</span>
                  <strong>{forecast.current_temperature.toFixed(2)} °C</strong>
                </div>
                <div>
                  <span>預測變化</span>
                  <strong>{forecast.predicted_change >= 0 ? '+' : ''}{forecast.predicted_change.toFixed(2)} °C</strong>
                </div>
                <div>
                  <span>目標時間</span>
                  <strong>{formatTime(forecast.target_time)}</strong>
                </div>
              </div>
            )}
            {forecast?.artifacts && (
              <div className="ridge-forecast-report" aria-live="polite">
                <div className="ridge-result-meta">
                  <span>{forecast.model_label} · {forecast.feature_count} 特徵</span>
                  <span>歷史回測 {forecast.evaluation.completed_samples.toLocaleString()} 筆</span>
                  <span>回測 MAE {formatMetric(forecast.evaluation.mae)} °C</span>
                  <span>回測 RMSE {formatMetric(forecast.evaluation.rmse)} °C</span>
                  <span>鎖定測試 MAE {formatMetric(forecast.test_metrics.mae)} °C</span>
                </div>
                <p className="ridge-report-note">{forecast.evaluation_note}</p>
                <div className="ridge-artifact-path">
                  報表已寫入 <code>{forecast.artifacts.run_directory}</code>
                </div>
                <ReportGallery attachments={forecast.attachments} />
              </div>
            )}
          </section>
        </>
      )}

      {error && <div className="ridge-error" role="alert">{error}</div>}
    </div>
  );
}
