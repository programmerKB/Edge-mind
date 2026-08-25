"""Database orchestration for model training, inference, and report creation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time
from typing import Any, Sequence

from forecast.engine import (
    FEATURE_NAMES,
    FORECAST_HORIZON_MINUTES,
    ForecastError,
    build_training_examples,
    extract_sensor_features,
    predict_with_payload,
    train_temperature_model,
)
from repositories.forecast_models import get_model, save_model
from repositories.sensors import list_readings


def train_and_save_model(
    db,
    motor_id: str,
    *,
    records: Sequence[Any] | None = None,
) -> dict:
    """Train and persist a device model, reusing preloaded history when given."""
    history = list(records) if records is not None else list_readings(db, motor_id)
    payload = train_temperature_model(build_training_examples(history))
    save_model(db, motor_id, payload)
    return payload


def _optional_metric(payload: dict, name: str, digits: int = 4):
    """Round an optional validation metric without repeating nested lookups."""
    value = payload.get("validation_metrics", {}).get(name)
    return round(value, digits) if value is not None else None


def _model_metadata(payload: dict, training_motor_id: str) -> dict:
    """Build the stable public model section of a forecast response."""
    training_duration = payload.get("training_duration_ms")
    return {
        "training_motor_id": training_motor_id,
        "algorithm": payload["algorithm"],
        "sample_count": payload["sample_count"],
        "validation_mae": round(payload["mae"], 4),
        "validation_rmse": round(payload["rmse"], 4),
        "validation_r2": _optional_metric(payload, "r2_score"),
        "validation_mape_percent": _optional_metric(payload, "mape_percent"),
        "training_duration_ms": (
            round(training_duration, 6) if training_duration is not None else None
        ),
        "trained_at": payload["trained_at"],
    }


def forecast_temperature(
    db,
    motor_id: str,
    auto_train: bool = True,
    training_motor_id: str | None = None,
) -> dict:
    """Forecast one device and create the matching evaluation artifacts.

    Histories are loaded once and reused for latest-reading inference, optional
    auto-training, and report generation. This removes redundant database round
    trips from the original implementation.
    """
    # Local import avoids making the pure numerical package depend on report I/O.
    from reports.inference import create_inference_report

    cpu_started = time.process_time()
    model_motor_id = training_motor_id or motor_id

    inference_records = list_readings(db, motor_id)
    if not inference_records:
        raise ForecastError(f"資料庫中找不到設備 {motor_id} 的感測資料")
    training_records = (
        inference_records
        if model_motor_id == motor_id
        else list_readings(db, model_motor_id)
    )

    stored = get_model(db, model_motor_id)
    if stored is None:
        if not auto_train:
            raise ForecastError(f"設備 {model_motor_id} 尚未訓練預測模型")
        payload = train_and_save_model(
            db,
            model_motor_id,
            records=training_records,
        )
    else:
        payload = json.loads(stored.model_json)

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
    source_time = latest.recorded_at
    target_time = source_time + timedelta(minutes=FORECAST_HORIZON_MINUTES)

    result = {
        "motor_id": motor_id,
        "inference_motor_id": motor_id,
        "training_motor_id": model_motor_id,
        "predicted_temperature": round(prediction, 3),
        "unit": "°C",
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "source_recorded_at": source_time.isoformat(),
        "target_time": target_time.isoformat(),
        "features": dict(zip(FEATURE_NAMES, features)),
        "model": _model_metadata(payload, model_motor_id),
    }
    report = create_inference_report(
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
    evaluation_metrics = report["metrics"]
    result["evaluation"] = {
        "completed_samples": report["completed_evaluation_samples"],
        **{
            name: round(value, 6) if value is not None else None
            for name, value in evaluation_metrics.items()
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
    return result
