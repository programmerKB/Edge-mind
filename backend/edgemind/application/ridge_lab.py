"""Use cases for comparing and applying the two compact Ridge variants."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from uuid import uuid4

from edgemind.application.ports import RidgeExperimentReportGateway, UnitOfWork
from edgemind.domain.forecasting import ForecastError
from edgemind.domain.ridge_experiments import (
    DIRECT_MODEL,
    FORECAST_HORIZON_MINUTES,
    HISTORY_MODEL,
    MODEL_LABELS,
    experiment_configuration,
    latest_history_features,
    predict_ridge,
    run_ridge_experiment,
    train_for_live_forecast,
)


class RidgeLabService:
    """Coordinate sensor persistence, pure experiments, and result files."""

    def __init__(self, reports: RidgeExperimentReportGateway):
        self._reports = reports

    def configuration(self, uow: UnitOfWork) -> dict:
        """Describe the fixed experiment and currently available devices."""
        return {
            **experiment_configuration(),
            "devices": uow.sensors.list_device_summaries(),
        }

    def run_experiment(
        self,
        uow: UnitOfWork,
        *,
        training_motor_id: str,
        evaluation_motor_id: str | None = None,
    ) -> dict:
        """Compare both models and persist one reproducible report."""
        training_records = uow.sensors.list_readings(training_motor_id)
        if not training_records:
            raise ForecastError(f"資料庫中找不到設備 {training_motor_id} 的感測資料")
        external_motor_id = (
            evaluation_motor_id
            if evaluation_motor_id and evaluation_motor_id != training_motor_id
            else None
        )
        external_records = (
            uow.sensors.list_readings(external_motor_id)
            if external_motor_id
            else None
        )
        if external_motor_id and not external_records:
            raise ForecastError(
                f"資料庫中找不到外部評估設備 {external_motor_id} 的感測資料"
            )
        result = run_ridge_experiment(
            training_records,
            external_records=external_records,
        )
        result.update(
            {
                "experiment_id": uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "training_motor_id": training_motor_id,
                "evaluation_motor_id": external_motor_id,
            }
        )
        return self._reports.save_ridge_experiment(result)

    def get_experiment(self, experiment_id: str) -> dict:
        """Return a saved experiment or a domain-level not-found error."""
        result = self._reports.get_ridge_experiment(experiment_id)
        if result is None:
            raise ForecastError(f"找不到實驗 {experiment_id}")
        return result

    def forecast(
        self,
        uow: UnitOfWork,
        *,
        motor_id: str,
        training_motor_id: str,
        model_name: str,
    ) -> dict:
        """Forecast from the latest window and publish its CSV/SVG report."""
        cpu_started = time.process_time()
        if model_name not in (DIRECT_MODEL, HISTORY_MODEL):
            raise ForecastError(f"不支援的模型：{model_name}")
        training_records = uow.sensors.list_readings(training_motor_id)
        inference_records = uow.sensors.list_readings(motor_id)
        if not training_records:
            raise ForecastError(f"找不到訓練設備 {training_motor_id} 的資料")
        if not inference_records:
            raise ForecastError(f"找不到預測設備 {motor_id} 的資料")
        training_started = time.perf_counter()
        parameters, model_evaluation = train_for_live_forecast(
            training_records,
            model_name,
        )
        training_duration_ms = (time.perf_counter() - training_started) * 1000
        origin_time, direct_features, history_features = latest_history_features(
            inference_records
        )
        features = (
            direct_features if model_name == DIRECT_MODEL else history_features
        )
        inference_started = time.perf_counter()
        predicted_delta = predict_ridge(parameters, features)
        prediction = direct_features[0] + predicted_delta
        inference_duration_ms = (time.perf_counter() - inference_started) * 1000
        process_cpu_time_ms = (time.process_time() - cpu_started) * 1000
        result = {
            "motor_id": motor_id,
            "training_motor_id": training_motor_id,
            "model_name": model_name,
            "model_label": MODEL_LABELS[model_name],
            "selected_alpha": model_evaluation["selected_alpha"],
            "training_sample_count": model_evaluation["sample_count"],
            "feature_count": len(features),
            "history_minutes": 0 if model_name == DIRECT_MODEL else 60,
            "validation_metrics": model_evaluation["validation_metrics"],
            "test_metrics": model_evaluation["test_metrics"],
            "source_recorded_at": origin_time.isoformat(),
            "target_time": (
                origin_time + timedelta(minutes=FORECAST_HORIZON_MINUTES)
            ).isoformat(),
            "current_temperature": round(direct_features[0], 3),
            "predicted_temperature": prediction,
            "predicted_change": round(predicted_delta, 3),
            "unit": "°C",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        report = self._reports.create_ridge_inference_report(
            training_records=training_records,
            inference_records=inference_records,
            parameters=parameters,
            forecast=result,
            training_duration_ms=training_duration_ms,
            model_inference_duration_ms=inference_duration_ms,
            process_cpu_time_ms=process_cpu_time_ms,
        )
        result["predicted_temperature"] = round(prediction, 3)
        result["evaluation"] = {
            "completed_samples": report["completed_evaluation_samples"],
            **{
                name: round(value, 6) if value is not None else None
                for name, value in report["metrics"].items()
                if name != "sample_count"
            },
        }
        result["evaluation_scope"] = report["evaluation_scope"]
        result["evaluation_note"] = report["evaluation_note"]
        result["anomaly_evaluation"] = report["anomaly_classification"][
            result["model_label"]
        ]
        result["artifacts"] = {
            "run_directory": report["run_directory"],
            **report["files"],
        }
        return self._reports.finalize_forecast_result(result)
