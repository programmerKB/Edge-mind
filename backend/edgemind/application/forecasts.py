"""Forecast training and inference use cases with inverted dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Sequence

from edgemind.application.ports import ForecastReportGateway, UnitOfWork
from edgemind.domain.entities import SensorReading
from edgemind.domain.forecasting import (
    FEATURE_NAMES,
    FORECAST_HORIZON_MINUTES,
    ForecastError,
    build_training_examples,
    extract_sensor_features,
    predict_with_payload,
    train_temperature_model,
)


class ForecastService:
    """Coordinate repositories, pure forecasting rules, and report output."""

    def __init__(self, reports: ForecastReportGateway):
        """Inject report output behind its application-owned port."""
        self._reports = reports

    def train(
        self,
        uow: UnitOfWork,
        motor_id: str,
        *,
        records: Sequence[SensorReading] | None = None,
    ) -> dict:
        """Train and persist one model inside the caller's transaction."""
        history = (
            list(records)
            if records is not None
            else uow.sensors.list_readings(motor_id)
        )
        payload = train_temperature_model(build_training_examples(history))
        uow.forecast_models.save_payload(motor_id, payload)
        uow.commit()
        return payload

    def forecast(
        self,
        uow: UnitOfWork,
        motor_id: str,
        *,
        auto_train: bool = True,
        training_motor_id: str | None = None,
    ) -> dict:
        """Forecast a device and finalize its report and public attachments."""
        cpu_started = time.process_time()
        model_motor_id = training_motor_id or motor_id
        inference_records = uow.sensors.list_readings(motor_id)
        if not inference_records:
            raise ForecastError(f"資料庫中找不到設備 {motor_id} 的感測資料")
        training_records = (
            inference_records
            if model_motor_id == motor_id
            else uow.sensors.list_readings(model_motor_id)
        )

        payload = uow.forecast_models.get_payload(model_motor_id)
        if payload is None:
            if not auto_train:
                raise ForecastError(f"設備 {model_motor_id} 尚未訓練預測模型")
            payload = self.train(
                uow,
                model_motor_id,
                records=training_records,
            )

        latest = inference_records[-1]
        features = extract_sensor_features(latest)
        if features is None:
            raise ForecastError(
                f"設備 {motor_id} 最新資料缺少溫度、濕度或三軸 XYZ 特徵"
            )

        inference_started = time.perf_counter()
        prediction = predict_with_payload(payload, features)
        inference_duration_ms = (time.perf_counter() - inference_started) * 1000
        process_cpu_time_ms = (time.process_time() - cpu_started) * 1000
        if latest.recorded_at is None:
            raise ForecastError(f"設備 {motor_id} 最新資料缺少記錄時間")
        target_time = latest.recorded_at + timedelta(
            minutes=FORECAST_HORIZON_MINUTES
        )

        result = {
            "motor_id": motor_id,
            "inference_motor_id": motor_id,
            "training_motor_id": model_motor_id,
            "predicted_temperature": round(prediction, 3),
            "unit": "°C",
            "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
            "source_recorded_at": latest.recorded_at.isoformat(),
            "target_time": target_time.isoformat(),
            "features": dict(zip(FEATURE_NAMES, features)),
            "model": _model_metadata(payload, model_motor_id),
        }
        report = self._reports.create_inference_report(
            training_records=training_records,
            inference_records=inference_records,
            model_payload=payload,
            inference_motor_id=motor_id,
            training_motor_id=model_motor_id,
            latest_prediction=prediction,
            generated_at=datetime.now(timezone.utc),
            model_inference_duration_ms=inference_duration_ms,
            process_cpu_time_ms=process_cpu_time_ms,
        )
        metrics = report["metrics"]
        result["evaluation"] = {
            "completed_samples": report["completed_evaluation_samples"],
            **{
                name: round(value, 6) if value is not None else None
                for name, value in metrics.items()
                if name != "sample_count"
            },
        }
        result["anomaly_evaluation"] = report["anomaly_classification"].get(
            "Five-feature Ridge",
            {},
        )
        result["artifacts"] = {
            "run_directory": report["run_directory"],
            **report["files"],
        }
        return self._reports.finalize_forecast_result(result)


def _optional_metric(payload: dict, name: str, digits: int = 4):
    """Round one optional nested validation metric for API output."""
    value = payload.get("validation_metrics", {}).get(name)
    return round(value, digits) if value is not None else None


def _model_metadata(payload: dict, training_motor_id: str) -> dict:
    """Build the stable model metadata section of a forecast result."""
    duration = payload.get("training_duration_ms")
    return {
        "training_motor_id": training_motor_id,
        "algorithm": payload["algorithm"],
        "sample_count": payload["sample_count"],
        "validation_mae": round(payload["mae"], 4),
        "validation_rmse": round(payload["rmse"], 4),
        "validation_r2": _optional_metric(payload, "r2_score"),
        "validation_mape_percent": _optional_metric(payload, "mape_percent"),
        "training_duration_ms": round(duration, 6) if duration is not None else None,
        "trained_at": payload["trained_at"],
    }
